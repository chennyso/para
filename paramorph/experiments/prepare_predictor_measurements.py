import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sqlite3

from predictor.measured_modules import annotate_measurement_semantics
from predictor.profile_database import normalized_execution, plan_signature
from scheduling.schedule_simulator import union_length


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def optimizer_ownership(report, request):
    values = {}
    for cost in report['evidence']['costs']:
        identity = cost.get('identity', {})
        requested = cost.get('profiling_request', {}).get('identity', {})
        if requested.get('operation') == 'local_optimizer_update_and_copy':
            values[requested['physical_rank']] = requested['parameter_count']
        elif cost.get('projection_operation') == 'adam_update':
            shape = identity['features']
            operations, traffic = shape['operation_count'], shape['traffic_bytes']
            if operations % 20 or traffic != operations // 20 * 28:
                raise ValueError('现存 Adam 解析记录与参数所有权公式不一致')
            values[cost['physical_rank']] = operations // 20
    world = request['node_count'] * request['gpus_per_node']
    if set(values) != set(range(world)):
        raise ValueError(f'{request["run_id"]} 的 Optimizer 参数所有权覆盖 {sorted(values)}，需要 {world} 个 Rank')
    return values


def compatibility_key(model, request):
    resolved = normalized_execution(request)
    resolved['precision_aware_optimizer'] = bool(request.get('precision_aware_optimizer', False))
    resolved['main_grads_dtype'] = request.get('main_grads_dtype', 'float32')
    fields = ('execution_backend', 'optimizer', 'optimizer_sharding', 'precision',
              'megatron_fsdp_version', 'precision_aware_optimizer', 'main_grads_dtype')
    return json.dumps(dict(model=model, **{key: resolved[key] for key in fields}), sort_keys=True)


def minimal_endpoint_plans(cases):
    minimum = min(min(case['optimizer_parameter_count_by_rank'].values()) for case in cases)
    maximum = max(max(case['optimizer_parameter_count_by_rank'].values()) for case in cases)
    endpoints = {minimum, maximum}
    selected = None
    for size in range(1, len(cases) + 1):
        for indices in combinations(range(len(cases)), size):
            covered = {value for index in indices for value in cases[index]['optimizer_parameter_count_by_rank'].values()}
            if endpoints <= covered:
                selected = [cases[index] for index in indices]
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError('现存候选没有可采集的参数所有权端点')
    return dict(minimum_parameter_count=minimum, maximum_parameter_count=maximum,
        endpoint_shape_count=len(endpoints), minimum_existing_plan_count=len(selected),
        selected_existing_plans=[dict(candidate_id=case['candidate_id'],
            endpoint_ranks={str(value): [rank for rank, owned in case['optimizer_parameter_count_by_rank'].items()
                if owned == value] for value in sorted(endpoints)},
            request=case['request']) for case in selected],
        covered_candidate_ids=[case['candidate_id'] for case in cases],
        applicability='仅表示当前参数数量插值模型的范围覆盖；真实 Tensor 分组、Kernel、精度及更新与复制阶段仍需采集',
        independent_plan_holdout='候选自身计划的 Profile 不能用于宣称跨计划独立预测；相同计划模块标定必须另行注明')


def timeline(report):
    graph = report.get('execution_graph')
    if graph is None:
        return None
    tasks = graph['tasks']
    grouped = defaultdict(list)
    for task in tasks.values():
        grouped[task['kind']].append(task)
    kinds = {}
    for kind, rows in grouped.items():
        kinds[kind] = dict(count=len(rows), earliest_start_ms=min(row['start_ms'] for row in rows),
            latest_finish_ms=max(row['end_ms'] for row in rows),
            rank_union_busy_ms={str(rank): union_length([(row['start_ms'], row['end_ms']) for row in rows if row['rank'] == rank])
                for rank in sorted({row['rank'] for row in rows})})
    critical = report['diagnosis'].get('critical_tasks', [])
    return dict(raw_step_time_ms=report['performance'].get('raw_resource_aware_step_time_ms'),
        modeled_exposed_communication_ms=report['diagnosis'].get('modeled_exposed_communication_ms'),
        task_kinds=kinds, critical_tasks=[dict(task_id=key, kind=tasks[key]['kind'],
            rank=tasks[key]['rank'], start_ms=tasks[key]['start_ms'], end_ms=tasks[key]['end_ms']) for key in critical],
        interpretation='各类型区间可能重叠，不能将这些区间长度相加解释 Step Time')


def profile_document(database, identifier):
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        row = connection.execute('SELECT document FROM profiles WHERE id=?', (identifier,)).fetchone()
    if row is None:
        raise ValueError(f'成本数据库缺少实际使用的 Profile：{identifier}')
    return annotate_measurement_semantics(json.loads(row[0]))


def runtime_manifests(row):
    source = row.get('measurement_artifacts')
    if source is None:
        return dict(available=False, reason='比较记录没有普通训练的Manifest目录')
    files = sorted(Path(source).glob('rank*.manifest.json'))
    if not files:
        return dict(available=False, reason='普通训练目录没有Rank Manifest')
    fields = ('rank', 'world', 'tp', 'pp', 'ep', 'cp', 'vpp', 'group', 'sequence',
              'global_batch', 'micro_batch', 'parameter_elements', 'gradient_bucket_size',
              'overlap_grad_reduce', 'overlap_param_gather', 'overlap_p2p_comm',
              'params_dtype', 'pipeline_dtype', 'recompute_granularity', 'process_groups')
    return dict(available=True, ranks=[dict(path=str(path), sha256=digest(path),
        values={key: value for key, value in load(path).items() if key in fields}) for path in files])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison = args.comparison.resolve()
    outcomes = load(comparison)
    cases, groups, profiles, diagnosis = [], defaultdict(list), {}, []
    for row in outcomes:
        model, candidate = row['model'], row['candidate_id']
        request = row['frozen_request']
        path = Path(row['after_path'])
        report = load(path)
        if plan_signature(request) != report['plan_signature']:
            raise ValueError('冻结请求与实际预测记录不一致')
        ownership = optimizer_ownership(report, request)
        original = workspace / 'outputs/representative-predictor-validation' / model
        database = original / ('profiles.sqlite' if model == 'falcon' else 'costs.sqlite')
        source_profiles = sorted({identifier for cost in report['evidence']['costs']
            if cost.get('rejected_cost_scope') for identifier in cost['evidence_ids']})
        case = dict(model=model, candidate_id=candidate, request=request,
            plan_signature=report['plan_signature'], predictor_revision=report['evidence']['predictor_revision'],
            prediction_path=str(path), prediction_sha256=digest(path),
            measured_ms=row['measured_ms'], measurement_source=row['measurement_comparison_path'],
            source_database=str(database), source_database_sha256=digest(database),
            source_profiles=source_profiles, optimizer_parameter_count_by_rank=ownership,
            optimizer_parameter_count_origin='现存预测报告中的参数所有权及 Adam 成本定义，属于模型派生数量',
            runtime_manifests=runtime_manifests(row),
            missing_profiles=row['missing_profiles'], profiling_requests=row['profiling_requests'])
        cases.append(case)
        groups[compatibility_key(model, request)].append(case)
        for identifier in source_profiles:
            key = f'{model}:{identifier}'
            if key not in profiles:
                profile = profile_document(database, identifier)
                profiles[key] = dict(id=identifier, request=profile['request'],
                    source_directory=profile.get('source_directory'), evidence_files=profile.get('evidence_files'),
                    measurement_semantics=profile['measurement_semantics'])
        if model == 'qwen14' and request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp':
            before_path = Path(row['before_path'])
            diagnosis.append(dict(candidate_id=candidate, measured_ms=row['measured_ms'],
                before_prediction_path=str(before_path), before_prediction_sha256=digest(before_path),
                before_timeline=timeline(load(before_path)), current_timeline=timeline(report),
                source_profiles=source_profiles, missing_profiles=row['missing_profiles']))
    requirements = [dict(compatibility=json.loads(key), **minimal_endpoint_plans(cases))
                    for key, cases in sorted(groups.items())]
    result = dict(comparison_path=str(comparison), comparison_sha256=digest(comparison),
        cases=cases, optimizer_sampling_groups=requirements, actual_source_profiles=profiles,
        qwen_ddp_diagnosis=diagnosis,
        endpoint_shape_count=sum(row['endpoint_shape_count'] for row in requirements),
        minimum_existing_plan_count=sum(row['minimum_existing_plan_count'] for row in requirements),
        gpu_jobs_started=0)
    (output / 'measurement-manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), sampling_groups=len(requirements),
        endpoint_shapes=result['endpoint_shape_count'], existing_plans=result['minimum_existing_plan_count'],
        output=str(output / 'measurement-manifest.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
