from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from contracts import fingerprint
from search.frontier_pricer import FrontierPricer, PricingPolicy
from search.physical_stage_search import PhysicalStageJointSearch


def dag_features(request,report):
    tasks = report['execution_graph']['tasks']
    graph = nx.DiGraph()
    graph.add_nodes_from(tasks)
    graph.add_edges_from((parent,key) for key,row in tasks.items() for parent in row['dependencies'])
    if set(graph) != set(tasks):
        raise ValueError('程序学习需要完整原生执行图')
    ends = {}
    work = dict(compute=0.0,communication=0.0,optimizer=0.0)
    for key in nx.topological_sort(graph):
        row = tasks[key]
        value = row['duration_ms']
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError('程序学习不能使用缺失事件费用')
        ends[key] = value+max((ends[parent] for parent in row['dependencies']),default=0)
        kind = row['kind']
        category = ('compute' if kind in ('forward_compute','backward_compute') else
            'optimizer' if kind=='optimizer_step' else 'communication')
        work[category] += value
    features = {key:float(request.get(key,1)) for key in
        ('tp','pp','vpp','cp','dp','ep','etp','sequence','micro_batch','global_batch')}
    peaks = report['memory'].get('peak_per_rank_bytes', {})
    features.update(event_count=len(tasks),edge_count=graph.number_of_edges(),
        rank_count=len({row['rank'] for row in tasks.values()}),
        critical_dependency_ms=max(ends.values()),
        compute_work_ms=work['compute'],communication_work_ms=work['communication'],
        optimizer_work_ms=work['optimizer'],peak_memory_bytes=max(peaks.values(),default=0),
        memory_evidence_available=int(bool(peaks)),
        fsdp=int(request.get('execution_backend')=='megatron_fsdp'),
        full_recompute=int(request['recompute']=='full'),
        selective_recompute=int(request['recompute'].startswith('selective')),
        resource_count=len({resource for row in tasks.values() for resource in row.get('resources', ())}))
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError('执行图特征需要有限数值')
    return features


def stage_order(oracle,mode):
    if mode=='none':
        return []
    if mode=='reverse':
        return list(reversed(range(oracle.config.pp)))
    if mode!='critical':
        raise ValueError('学习程序引用了未注册的阶段排序方式')
    from search.physical_stage_master import task_stage
    totals = [0.0]*oracle.config.pp
    for row in oracle.tasks.values():
        totals[task_stage(row,oracle.config.world_size//oracle.config.pp)] += row['duration_ms']
    return sorted(range(oracle.config.pp),key=lambda stage:(-totals[stage],stage))


def policy_from_parameters(oracle,values,evidence_refs):
    memory,communication,expansions,mode,rule = values
    return PricingPolicy(memory_priority=float(memory),communication_priority=float(communication),
        max_frontier_expansions=int(expansions),preferred_stages=stage_order(oracle,mode),
        interval_expansion_rule=rule,reason='归档原生执行成本上的贝叶斯程序参数搜索',
        evidence_refs=evidence_refs)


def train_pricing_programs(cases,oracle_factory,budget,n_calls,directory,seed=0):
    if len(cases) < 2 or n_calls < 4:
        raise ValueError('程序学习至少需要两项真实DAG及四次贝叶斯参数搜索')
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    dimensions = [Real(0,4,name='memory'),Real(0,1,name='communication'),
        Integer(64,4096,name='expansions'),Categorical(['critical','reverse','none'],name='stage_order'),
        Categorical(['reduced_cost','seed_distance'],name='rule')]
    training_rows = []
    for index,case in enumerate(cases):
        case_dir = directory/f'case-{index:03d}'
        case_dir.mkdir()
        evaluations,cache = [],{}

        def objective(values):
            identity = fingerprint([float(values[0]),float(values[1]),int(values[2]),values[3],values[4]])
            if identity in cache:
                return cache[identity]
            run_dir = case_dir/f'trial-{len(evaluations):03d}'
            run_dir.mkdir()
            oracle = oracle_factory(case,run_dir)
            policy = policy_from_parameters(oracle,values,[oracle.revision])
            result = PhysicalStageJointSearch(oracle,FrontierPricer(policy),budget).run(run_dir.name)
            best = result['best_prediction'] or result['best_pricing_proxy']
            if best is None:
                raise ValueError('真实搜索没有产生可评价的完整分区')
            full = result['best_prediction'] is not None
            step = best['step_time_ms'] if full else best['pricing_replay_step_time_ms']
            objective_value = math.log(step/result['seed_step_time_ms'])
            objective_value += 0.01*math.log1p(result['usage']['pricing_cost_constructions'])
            record = dict(case=case,parameters=[float(values[0]),float(values[1]),int(values[2]),values[3],values[4]],
                policy=policy.model_dump(mode='json'),objective=objective_value,
                label_scope='complete_dag_prediction' if full else 'archived_native_composite_pricing_model',
                features=dag_features(oracle.request,oracle.seed_report),
                execution_graph_sha256=fingerprint(oracle.seed_report['execution_graph']),
                cost_revision=oracle.revision,profile_id=oracle.profile_id,
                result_path=str(run_dir/'search-result.json'),usage=result['usage'])
            evaluations.append(record)
            cache[identity] = objective_value
            (case_dir/'trials.json').write_text(json.dumps(evaluations,ensure_ascii=False,indent=2)+'\n')
            return objective_value

        optimized = gp_minimize(objective,dimensions,n_calls=n_calls,n_initial_points=3,
            random_state=seed+index,acq_func='EI',acq_optimizer='sampling',n_points=256,n_jobs=1)
        best = min(evaluations,key=lambda row:row['objective'])
        if not math.isclose(best['objective'],float(optimized.fun),abs_tol=1e-10):
            raise ValueError('贝叶斯优化结果与真实搜索记录不一致')
        training_rows.append(best)
    names = sorted(training_rows[0]['features'])
    if len({row['execution_graph_sha256'] for row in training_rows}) != len(training_rows):
        raise ValueError('离线程序学习不能重复计数相同执行图')
    x = np.asarray([[row['features'][name] for name in names] for row in training_rows])
    y = np.asarray([row['parameters'][:3] for row in training_rows],dtype=float)
    labels = [row['parameters'][3]+':'+row['parameters'][4] for row in training_rows]
    regressor = RandomForestRegressor(n_estimators=64,random_state=seed,n_jobs=1)
    classifier = RandomForestClassifier(n_estimators=64,random_state=seed,n_jobs=1)
    regressor.fit(x,y)
    classifier.fit(x,labels)
    model_path = directory/'pricing-program.joblib'
    joblib.dump(dict(regressor=regressor,classifier=classifier,feature_names=names),model_path)
    manifest = dict(schema_version=1,algorithm='Gaussian_process_Bayesian_search_and_random_forest_DAG_policy_mapping',
        feature_names=names,training_rows=training_rows,search_budget=asdict(budget),
        sklearn_version=sklearn.__version__,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        new_gpu_runs=0,training_speedup_claimed=False)
    manifest['evidence_id'] = fingerprint(manifest)
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


class LearnedPricingProgram:
    def __init__(self,directory):
        directory = Path(directory)
        self.manifest = json.loads((directory/'manifest.json').read_text())
        if fingerprint({key:value for key,value in self.manifest.items() if key!='evidence_id'}) != self.manifest['evidence_id']:
            raise ValueError('离线程序训练清单的内容身份不一致')
        if sklearn.__version__ != self.manifest['sklearn_version']:
            raise ValueError('学习程序需要使用训练时登记的scikit-learn版本')
        path = directory/'pricing-program.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['model_sha256']:
            raise ValueError('学习程序与已登记模型校验值不一致')
        self.model = joblib.load(path)
        if self.model['feature_names'] != self.manifest['feature_names']:
            raise ValueError('学习程序特征身份不一致')

    def propose(self,oracle):
        features = dag_features(oracle.request,oracle.seed_report)
        values = [[features[name] for name in self.model['feature_names']]]
        numeric = self.model['regressor'].predict(values)[0]
        mode,rule = self.model['classifier'].predict(values)[0].split(':')
        parameters = [float(np.clip(numeric[0],0,4)),float(np.clip(numeric[1],0,1)),
            int(np.clip(round(numeric[2]),64,4096)),mode,rule]
        policy = policy_from_parameters(oracle,parameters,[self.manifest['evidence_id']])
        return dict(evidence_id=self.manifest['evidence_id'],policy=policy.model_dump(mode='json'),
            features=features,training_graphs=[row['execution_graph_sha256']
                for row in self.manifest['training_rows']],
            prediction_scope='search_program_parameters_without_training_throughput_claim')
