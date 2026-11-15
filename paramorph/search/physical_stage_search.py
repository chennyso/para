import json
from dataclasses import dataclass
import math
from pathlib import Path
import time

from pydantic import Field

from contracts import Contract, RewriteIntent, fingerprint
from cost.runtime_calibration import configuration
from predictor.profile_database import plan_signature, signature
from region_partition.region_partitioner import RegionPartitioner
from search.physical_stage_master import NativeEventMaster, PhysicalStageColumn, task_stage
from search.fragment_cost_cache import FragmentCostCache
from search.impact_neighborhood import cross_stage_interfaces, execution_neighborhood
from search.module_relations import complete_partition_boundaries, prefix_support


class StageProposal(Contract):
    stage: int = Field(ge=0)
    intervals: list[list[int]]
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class PricingProposal(Contract):
    columns: list[StageProposal] = Field(min_length=1, max_length=6)


@dataclass(frozen=True)
class JointSearchBudget:
    column_queries: int = 96
    complete_evaluations: int = 12
    rounds: int = 4
    wall_seconds: float | None = None
    columns_per_round: int = 6

    def __post_init__(self):
        if min(self.column_queries, self.complete_evaluations, self.rounds, self.columns_per_round) < 1:
            raise ValueError('搜索预算必须为正数')
        if self.wall_seconds is not None and (not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0):
            raise ValueError('运行时间预算必须为有限正数')


class PhysicalStageOracle:
    def __init__(self,domain,request,predictor,rewriter,directory,*,seed_report=None,module_profile=None,
                 complete_evaluator=None,projection_observer=None,graph_loader=None,module_relations=()):
        self.domain, self.request = domain, dict(request)
        self.predictor, self.rewriter = predictor, rewriter
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = configuration(request)
        if (self.config.pp, self.config.vpp) != (domain.pp, domain.vpp):
            raise ValueError('候选空间与实际原生配置不同')
        self.rewriter.candidate(request)
        self.revision = predictor.database.cost_database_revision()
        self.snapshots, self.columns = {}, {}
        self.query_count, self.projection_count = 0, 0
        self.snapshot_seconds = 0.0
        self.complete_seconds = 0.0
        self.complete_evaluator = complete_evaluator
        self.projection_observer = projection_observer
        self.graph_loader = graph_loader
        self.module_relations = tuple(module_relations)
        self.deferred_columns = {}
        if seed_report is not None:
            if seed_report.get('plan_signature') != plan_signature(request):
                raise ValueError('定价种子证据的完整配置不一致')
            if seed_report.get('cost_database_revision') != self.revision:
                raise ValueError('定价种子证据的成本版本不一致')
            self.snapshots[tuple(domain.seed)] = seed_report
        self.seed_report = self.snapshot(domain.seed)
        if not (self.seed_report.get('execution_graph') or {}).get('tasks'):
            raise ValueError('物理阶段主问题缺少完整种子执行图')
        self.tasks = self.seed_report['execution_graph']['tasks']
        self.template = {key: (row['rank'], row['kind'], tuple(sorted(set(row['dependencies']))),
            tuple(sorted(row.get('resources', ()))))
            for key, row in self.tasks.items()}
        profile = module_profile if module_profile is not None else self.predictor.projector.reusable_profile
        self.module_costs = ([] if profile is None else [dict(layer=int(key), **row)
            for key, row in sorted(profile['layers'].items(), key=lambda item: int(item[0]))])
        if profile is not None and profile.get('partition_state_bytes') is not None:
            for row in self.module_costs:
                row['state_bytes'] = profile['partition_state_bytes'][str(row['layer'])]
        self.profile_id = None if profile is None else profile['id']
        self.liveness = self.native_liveness()
        self.fragment_cache = None

    def native_liveness(self):
        output = {}
        for stage in range(self.config.pp):
            active, rows = set(), []
            sequence = self.predictor._native_stage_sequence(stage, self.config.pp, self.config.vpp,
                self.config.num_microbatches, self.config.microbatch_group)
            for kind, vp, mb in sequence:
                if kind == 'forward_compute':
                    if (vp, mb) in active:
                        raise ValueError('原生前向重复保存同一微批')
                    active.add((vp, mb))
                rows.append(tuple(sum(v == selected for v, _ in active)
                    for selected in range(self.config.vpp)))
                if kind == 'backward_compute':
                    active.remove((vp, mb))
            if active:
                raise ValueError('原生次序没有释放全部微批')
            output[stage] = sorted(set(rows))
        return output

    def snapshot(self, endpoints):
        self.domain.validate(endpoints)
        if self.predictor.database.cost_database_revision() != self.revision:
            raise ValueError('成本已经更新，必须重新建立候选与主问题')
        endpoints = tuple(endpoints)
        if endpoints not in self.snapshots:
            request = dict(self.request, decoder_chunks=[b - a for a, b in zip(endpoints, endpoints[1:])])
            started = time.perf_counter()
            report = self.predictor.evaluate_projection(request, None, None, persist=False)
            self.snapshot_seconds += time.perf_counter() - started
            self.projection_count += 1
            if self.projection_observer is not None:
                self.projection_observer(request,report)
            if not (report.get('execution_graph') or {}).get('tasks'):
                (self.directory / 'missing-pricing-evidence.json').write_text(json.dumps(dict(
                    status=report['prediction_status'], requests=report['evidence'].get('profiling_requests', [])),
                    ensure_ascii=False, indent=2) + '\n')
            self.snapshots[endpoints] = report
        return self.snapshots[endpoints]

    def witness(self,stage,intervals):
        witness = self.domain.witness(stage,intervals)
        counts = [b-a for a,b in zip(witness,witness[1:])]
        if not prefix_support(counts,self.domain.seed[-1],self.domain.pp,self.domain.vpp,self.module_relations):
            fixed = {index:value for vp,pair in enumerate(intervals)
                for index,value in ((vp*self.domain.pp+stage,pair[0]),(vp*self.domain.pp+stage+1,pair[1]))}
            witness = complete_partition_boundaries(self.domain,fixed,self.module_relations)
        return witness

    def column(self, stage, intervals):
        self.query_count += 1
        intervals = tuple(tuple(pair) for pair in intervals)
        witness = self.witness(stage,intervals)
        if witness is None:
            return None
        identity = fingerprint(dict(stage=stage, intervals=intervals, revision=self.revision,
            family=plan_signature(self.request)))
        if identity in self.columns:
            return self.columns[identity]
        report = self.snapshot(witness)
        if not (report.get('execution_graph') or {}).get('tasks'):
            self.deferred_columns[witness] = dict(request=dict(self.request,
                decoder_chunks=[b-a for a,b in zip(witness,witness[1:])]),prediction=report)
            return None
        tasks = report['execution_graph']['tasks']
        template = {key: (row['rank'], row['kind'], tuple(sorted(set(row['dependencies']))),
            tuple(sorted(row.get('resources', ()))))
            for key, row in tasks.items()}
        if template != self.template:
            self.deferred_columns[witness] = dict(request=dict(self.request,
                decoder_chunks=[b-a for a,b in zip(witness,witness[1:])]),prediction=report,
                reason='REQUIRES_NEW_EVENT_MASTER')
            return None
        lanes = self.config.world_size // self.config.pp
        local = {key: row for key, row in tasks.items() if task_stage(row,lanes) == stage}
        request = dict(self.request, decoder_chunks=[b - a for a, b in zip(witness, witness[1:])])
        projector = self.predictor.projector
        regions = RegionPartitioner(projector.model, self.config).from_counts(request['decoder_chunks'])
        physical_ranks = {rank:self.request.get('rank_mapping',list(range(self.config.world_size)))[rank]
            for rank in range(stage*lanes,(stage+1)*lanes)}
        observed_peaks = {int(rank):value for rank,value in report['memory'].get('peak_per_rank_bytes', {}).items()}
        complete_memory = (report['memory'].get('admission')=='WITHIN_ESTIMATED_CAPACITY'
            and set(physical_ranks.values()) <= observed_peaks.keys())
        counts = projector._parameter_counts(self.config, regions)
        if counts is None and not complete_memory:
            self.deferred_columns[witness] = dict(request=request,prediction=report,
                reason='PARAMETER_OWNERSHIP_EVIDENCE_MISSING')
            return None
        states = projector._state_memory(counts,self.config,request) if counts is not None else None
        memory = projector.parameters.get('memory', {})
        required = ('workspace_bytes', 'reserve_bytes', 'communication_workspace_bytes')
        known_workspace = all(name in memory and memory.get(name.removesuffix('_bytes')+'_source') for name in required)
        if known_workspace and any(type(memory[name]) is not int or memory[name] < 0 for name in required):
            raise ValueError('局部准入的显存常数非法')
        peaks = {}
        for rank in range(stage * lanes, (stage + 1) * lanes):
            physical = physical_ranks[rank]
            if complete_memory:
                peaks[physical] = observed_peaks[physical]
            elif known_workspace:
                activations = [projector._activation_bytes(list(range(a,b)),rank,self.config,request)
                    for a,b in intervals]
                peak = max(sum(n*activation for n,activation in zip(row,activations))
                    for row in self.liveness[stage])
                peaks[physical] = states[rank]['persistent_bytes']+peak+sum(memory[name] for name in required)
            else:
                peaks[physical] = states[rank]['persistent_bytes']
        capacity = int(projector.topology.memory_per_gpu_gb * 2**30)
        column = PhysicalStageColumn(identity, stage, intervals,
            {key: row['duration_ms'] for key, row in local.items()}, peaks,
            max(peaks.values()) <= capacity,
            dict(cost_revision=self.revision, witness=witness, prediction_id=report['prediction_id'],
                profile_id=self.profile_id, capacity_bytes=capacity,
                memory_scope='complete_witness_memory_evidence' if complete_memory else
                    'native_liveness_and_parameter_state_model' if known_workspace else
                    'parameter_state_ownership_lower_bound_requires_full_memory_evaluation',
                local_memory_complete=complete_memory,
                full_memory_admission=report['memory']['admission'],
                cross_stage_interfaces={key:value for key,value in cross_stage_interfaces(
                    report['execution_graph'],lanes).items() if stage in value['stages']},
                time_scope='existing_projection_costs_with_recorded_measurement_limits'))
        self.columns[identity] = column
        return column

    def complete(self, endpoints, evidence_refs):
        self.domain.validate(endpoints)
        counts = [b - a for a, b in zip(endpoints, endpoints[1:])]
        intent = RewriteIntent(operation='RepartitionRegion', values=dict(decoder_chunks=counts),
            adjustable=[], preserve=[], hypothesis='原生事件主问题选择物理阶段联合区间',
            evidence_refs=evidence_refs,parent_plan_id=self.request['run_id'],module_relations=list(self.module_relations))
        if not prefix_support(counts,self.domain.seed[-1],self.domain.pp,self.domain.vpp,self.module_relations):
            raise ValueError('整数主问题的完整分区违反原生模块归属条件')
        if counts == self.request['decoder_chunks']:
            from agents.program_agent import ProgramAgent
            candidate = self.rewriter.candidate(self.request)
            ProgramAgent().compile(candidate)
            rewrite = dict(request=self.request,
                execution_graph=self.rewriter.compiler.observation(candidate, [], [])['training_graph'])
        else:
            rewrite = self.rewriter.complete_and_rewrite(self.request, intent, limit=1)[0]
        if rewrite['request']['decoder_chunks'] != counts:
            raise ValueError('原生补全改变了主问题选择的分区')
        if signature(rewrite['request']) != signature(self.request):
            raise ValueError('原生补全改变了固定执行条件')
        started = time.perf_counter()
        report = (self.predictor.evaluate_projection(rewrite['request'],rewrite['execution_graph'],None,persist=False)
            if self.complete_evaluator is None else self.complete_evaluator(rewrite['request'],rewrite['execution_graph']))
        self.complete_seconds += time.perf_counter() - started
        if report is None:
            return None
        if (self.graph_loader is not None and report.get('execution_graph') is not None
                and 'tasks' not in report['execution_graph']):
            report = dict(report,execution_graph=self.graph_loader(report))
        neighborhood = execution_neighborhood(self.seed_report,report,self.request,
            rewrite['request'],self.revision).report() if (report.get('execution_graph') or {}).get('tasks') else None
        identity = fingerprint(dict(endpoints=endpoints, revision=self.revision))
        artifact = self.directory / f'{identity}.prediction.json'
        artifact.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
        return dict(endpoints=list(endpoints), decoder_chunks=counts,
            request=rewrite['request'], prediction_id=report['prediction_id'],
            step_time_ms=report['performance']['step_time_ms'],
            prediction_status=report['prediction_status'],
            memory_admission=report['memory']['admission'],
            memory_missing=report['memory'].get('missing_profiles', []),
            maximum_predicted_memory_bytes=max(report['memory'].get('peak_per_rank_bytes', {}).values(), default=None),
            missing_cost_request_count=len(report['evidence'].get('profiling_requests', [])),
            execution_neighborhood=neighborhood,
            artifact=str(artifact), scope='full_projection_not_new_training_measurement')

    def estimated_column(self, master, stage, intervals):
        self.domain.witness(stage, intervals)
        base = next(column for column in master.columns.values() if column.stage == stage
            and column.intervals == self.domain.intervals(self.domain.seed, stage))
        if not self.module_costs:
            raise ValueError('确定性定价的排序需要同一成本工具提供的模块成本')
        cache = self.cost_fragments()
        durations = dict(base.durations)
        old_intervals = self.domain.intervals(self.domain.seed, stage)
        old_state = self.fragment_state_bytes(old_intervals)
        new_state = self.fragment_state_bytes(intervals)
        for key in durations:
            task = self.tasks[key]
            if task['kind'] in ('forward_compute', 'backward_compute'):
                vp = task['region'] // self.config.pp
                field = 'forward_ms' if task['kind'] == 'forward_compute' else 'backward_ms'
                a, b = old_intervals[vp]
                c, d = intervals[vp]
                durations[key] = max(0.0, durations[key] + cache.interval(c,d)[field]
                    - cache.interval(a,b)[field])
            elif task['kind'] == 'optimizer_step' or task['metadata'].get('state_family') in ('dense', 'expert'):
                if old_state <= 0:
                    raise ValueError('Optimizer定价需要正值的原始参数状态字节数')
                durations[key] *= new_state / old_state
        return PhysicalStageColumn('estimated', stage, intervals, durations, {}, True,
            dict(cost_revision=self.revision,
                cross_stage_interfaces=base.evidence.get('cross_stage_interfaces', {}),
                scope='pricing_order_only_not_measured_cost'))

    def cost_fragments(self):
        if self.predictor.database.cost_database_revision() != self.revision:
            raise ValueError('成本版本改变后需要重新建立局部成本目录')
        if self.fragment_cache is None:
            self.fragment_cache = FragmentCostCache(dict(request=self.request,
                cost_revision=self.revision, profile_id=self.profile_id,
                execution_graph=self.seed_report['execution_graph']), self.module_costs)
        return self.fragment_cache

    def fragment_state_bytes(self, intervals):
        cache = self.cost_fragments()
        return math.fsum(cache.interval(a,b)['state_bytes'] for a,b in intervals)


class NativeCompositePricingOracle(PhysicalStageOracle):
    def __init__(self, domain, request, predictor, rewriter, directory, profile):
        if signature(request) != signature(profile['request']):
            raise ValueError('复合事件定价要求完整运行条件与归档采集一致')
        self.profile = predictor._special_module_breakdown(profile)
        self.pricing_plans = {}
        super().__init__(domain, request, predictor, rewriter, directory)
        self.profile_id = profile['id']
        self.module_costs = [dict(layer=int(key), **row)
            for key, row in sorted(profile['layers'].items(), key=lambda item: int(item[0]))]
        for rank, row in self.profile['sequences'].items():
            expected = predictor._native_stage_sequence(row['pp_rank'], self.config.pp, self.config.vpp,
                self.config.num_microbatches, self.config.microbatch_group)
            if [tuple(item) for item in row['events']] != expected:
                raise ValueError(f'Rank {rank} 的归档次序与当前原生次序不同')

    def snapshot(self, endpoints):
        self.domain.validate(endpoints)
        if self.predictor.database.cost_database_revision() != self.revision:
            raise ValueError('成本已经更新，必须重新建立候选与主问题')
        endpoints = tuple(endpoints)
        if endpoints not in self.snapshots:
            request = dict(self.request, decoder_chunks=[b - a for a, b in zip(endpoints, endpoints[1:])])
            started = time.perf_counter()
            prediction, plan = self.predictor._simulate(request, self.profile)
            self.snapshot_seconds += time.perf_counter() - started
            self.projection_count += 1
            dependencies = {key: set(task.dependencies) for key, task in plan.tasks.items()}
            for sequence in plan.rank_schedules.values():
                resources = {}
                for key in sequence:
                    for resource in plan.tasks[key].resource_requirements:
                        if resource in resources:
                            dependencies[key].add(resources[resource])
                        resources[resource] = key
            tasks = {key: dict(rank=task.rank, kind=task.task_type.value, region=task.region_id,
                microbatch=task.microbatch_id, dependencies=sorted(dependencies[key]),
                duration_ms=task.duration, resources=list(task.resource_requirements),
                metadata=task.metadata) for key, task in plan.tasks.items()}
            memory = self.predictor.memory_admission(request, self.profile)
            self.snapshots[endpoints] = dict(prediction_id=fingerprint(dict(endpoints=endpoints,
                profile=self.profile['id'], revision=self.revision, kind='native_composite_pricing_proxy')),
                performance=dict(step_time_ms=prediction.predicted_iteration_time_ms), memory=memory,
                execution_graph=dict(tasks=tasks), evidence=dict(costs=[]),
                prediction_status='ARCHIVED_NATIVE_COMPOSITE_PRICING_PROXY')
            self.pricing_plans[endpoints] = plan
        return self.snapshots[endpoints]

    def column(self, stage, intervals):
        self.query_count += 1
        intervals = tuple(tuple(pair) for pair in intervals)
        witness = self.domain.witness(stage, intervals)
        identity = fingerprint(dict(stage=stage, intervals=intervals, revision=self.revision,
            family=plan_signature(self.request), basis='native_composite_pricing_proxy'))
        if identity in self.columns:
            return self.columns[identity]
        report = self.snapshot(witness)
        tasks = report['execution_graph']['tasks']
        template = {key: (row['rank'], row['kind'], tuple(sorted(set(row['dependencies']))),
            tuple(sorted(row.get('resources', ()))))
            for key, row in tasks.items()}
        if template != self.template:
            raise ValueError('复合事件分区改变了固定定价模板')
        lanes = self.config.world_size // self.config.pp
        ranks = range(stage * lanes, (stage + 1) * lanes)
        peaks = {rank: report['memory']['peak_per_rank_bytes'][rank] for rank in ranks}
        capacities = {rank: self.profile['memory'][str(rank)]['capacity'] for rank in ranks}
        column = PhysicalStageColumn(identity, stage, intervals,
            {key: row['duration_ms'] for key, row in tasks.items() if task_stage(row,lanes) == stage}, peaks,
            all(peaks[rank] <= capacities[rank] for rank in ranks),
            dict(cost_revision=self.revision, witness=witness, prediction_id=report['prediction_id'],
                profile_id=self.profile_id, capacity_bytes=min(capacities.values()),
                cross_stage_interfaces={key:value for key,value in cross_stage_interfaces(
                    report['execution_graph'],lanes).items() if stage in value['stages']},
                memory_scope='existing_archived_composite_activation_and_workspace_projection',
                time_scope='inclusive_layer_region_and_optimizer_proxy_not_isolated_service_times'))
        self.columns[identity] = column
        return column


class DeterministicPricer:
    def __init__(self):
        self.proposed = set()

    def propose(self, master, oracle, lp, count, observation):
        scored = []
        for stage in range(master.domain.pp):
            for intervals in master.domain.columns(stage):
                key = (stage, intervals)
                if key in self.proposed:
                    continue
                estimated = oracle.estimated_column(master, stage, intervals)
                scored.append((master.reduced_cost(estimated, lp['prices']), stage, intervals))
        chosen = sorted(scored)[:count]
        self.proposed.update((stage, intervals) for _, stage, intervals in chosen)
        return [dict(stage=stage, intervals=intervals, reason='模块成本及实际LP价格的确定性枚举排序',
            evidence_refs=[oracle.profile_id, oracle.revision], estimated_reduced_cost=value)
            for value, stage, intervals in chosen], dict(provider_calls=0,
                enumerated_columns=len(scored), scoring_scope='linear_cost_pricing_order_then_full_oracle_check')


class AgentPricer:
    def __init__(self, generate):
        self.generate = generate

    def propose(self, master, oracle, lp, count, observation):
        response, provenance = self.generate('PhysicalStagePricingAgent',
            '为原生混合并行提出具体物理阶段候选。每项必须包含该阶段全部VPP区间，端点位于给定范围。'
            '利用实际事件价格、首尾模块成本、共同保存系数及历史检查结果选择端点。'
            '优先提出能够降低事件目标且容量允许的候选；工具负责数值核查及生成必要连接候选。'
            '证据引用只能来自所给cost_revision、profile_id或历史prediction_id。'
            f'最多返回{count}项。禁止声称已测量或已加速。', observation, PricingProposal)
        if len(response.columns) > count:
            raise ValueError('Agent候选数量超过当前预算')
        known = {oracle.revision, oracle.profile_id} | {row['prediction_id'] for row in observation['evaluations']}
        rows = [row.model_dump(mode='json') for row in response.columns]
        if any(not set(row['evidence_refs']) <= known for row in rows):
            raise ValueError('Agent引用了没有提供的证据')
        return rows, dict(provider_calls=1, provenance=provenance)


class PhysicalStageJointSearch:
    def __init__(self, oracle, pricer, budget):
        self.oracle, self.pricer, self.budget = oracle, pricer, budget
        self.master = NativeEventMaster(oracle.domain, oracle.tasks,
            oracle.config.world_size,oracle.revision,module_relations=oracle.module_relations)

    def run(self, run_id):
        started = time.perf_counter()
        oracle, master, budget = self.oracle, self.master, self.budget
        seed_columns = [oracle.column(stage, oracle.domain.intervals(oracle.domain.seed, stage))
            for stage in range(oracle.config.pp)]
        if not all(column.local_admitted for column in seed_columns):
            raise ValueError('完整种子没有通过局部容量准入')
        for column in seed_columns:
            master.add(column)
        visited, evaluations, history = [], [], []
        stop = 'ROUND_BUDGET'
        for round_index in range(budget.rounds):
            if budget.wall_seconds is not None and time.perf_counter() - started >= budget.wall_seconds:
                stop = 'WALL_BUDGET'; break
            proposal_limit = min(budget.columns_per_round,
                (budget.column_queries - oracle.query_count) // master.domain.pp)
            if proposal_limit < 1:
                stop = 'COLUMN_BUDGET'; break
            lp = master.solve_lp()
            complete_prices,visible_prices = master.compact_price_observation(lp)
            pricing_evidence_id = fingerprint(complete_prices)
            pricing_artifact = oracle.directory/f'pricing-lp-{round_index:03d}-{pricing_evidence_id}.json'
            pricing_artifact.write_text(json.dumps(complete_prices,ensure_ascii=False,allow_nan=False)+'\n')
            observation = dict(run_id=run_id, round=round_index,
                domain=dict(pp=master.domain.pp, vpp=master.domain.vpp,
                    boundaries=master.domain.boundaries, seed=master.domain.seed,
                    complete_layout_count=master.domain.layout_count()),
                family={key: oracle.request.get(key) for key in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp',
                    'vpp', 'group', 'global_batch', 'sequence', 'micro_batch', 'recompute', 'overlap_p2p')},
                profile_id=oracle.profile_id, cost_revision=oracle.revision,
                module_costs=[{key:row[key] for key in ('layer','forward_ms','backward_ms','state_bytes')}
                    for row in oracle.module_costs],native_liveness=oracle.liveness,
                model_structure=[dict(layer=index, type=kind.value,
                    module_kind=(oracle.rewriter.model.module_kinds[index]
                        if oracle.rewriter.model.module_kinds is not None else 'transformer_block'))
                    for index, kind in enumerate(oracle.rewriter.model.layer_types)],
                seed_region_costs=[dict(rank=row['rank'], region=row['region'], kind=row['kind'],
                    duration_ms=row['duration_ms']) for row in oracle.tasks.values()
                    if row['kind'] in ('forward_compute', 'backward_compute') and row['microbatch'] == 0],
                seed_persistent_bytes=oracle.seed_report['memory'].get('persistent_per_rank_bytes'),
                pricing_estimate=dict(compute='seed_duration_plus_sum_of_module_cost_changes',
                    optimizer='seed_duration_times_candidate_persistent_bytes_over_seed_persistent_bytes',
                    persistent='seed_persistent_plus_sum_of_migrated_module_state_bytes',
                    other_events='unchanged_for_pricing_order_then_verified_by_common_cost_tool'),
                capacity_bytes=seed_columns[0].evidence['capacity_bytes'],
                seed_local_peak_bytes={column.stage: column.local_peaks for column in seed_columns},
                prices=visible_prices,pricing_evidence_id=pricing_evidence_id,pricing_artifact=str(pricing_artifact),
                evaluations=[{key:row.get(key) for key in ('prediction_id','decoder_chunks','step_time_ms',
                    'prediction_status','memory_admission','maximum_predicted_memory_bytes',
                    'missing_cost_request_count','scope')} for row in evaluations],
                history=[dict(round=row['round'],lp_before_ms=row['lp_before_ms'],lp_after_ms=row['lp_after_ms'],
                    proposals=[dict(stage=item['stage'],intervals=item['intervals']) for item in row['proposals']],
                    integers=row['integers'],rejected=[{key:checked.get(key) for key in
                        ('stage','intervals','local_peaks','status')} for checked in row['checked']
                        if not checked['local_admitted']][:8]) for row in history],
                remaining_column_queries=budget.column_queries - oracle.query_count,
                remaining_wall_seconds=None if budget.wall_seconds is None else
                    budget.wall_seconds - (time.perf_counter() - started))
            proposals, provider = self.pricer.propose(master, oracle, lp,
                proposal_limit, observation)
            checked = []
            for proposal in proposals:
                witness = oracle.witness(proposal['stage'],proposal['intervals'])
                if witness is None:
                    checked.append(dict(proposed_stage=proposal['stage'],stage=proposal['stage'],
                        added=False,local_admitted=False,status='REJECTED_MODULE_RELATION',reason=proposal['reason']))
                    continue
                for stage in range(master.domain.pp):
                    if oracle.query_count >= budget.column_queries:
                        stop = 'COLUMN_BUDGET'; break
                    column = oracle.column(stage, master.domain.intervals(witness, stage))
                    if column is None:
                        checked.append(dict(proposed_stage=proposal['stage'],stage=stage,
                            added=False,local_admitted=False,status='NEEDS_PROFILE',reason=proposal['reason']))
                        continue
                    reduced = master.reduced_cost(column, lp['prices'])
                    added = master.add(column)
                    checked.append(dict(proposed_stage=proposal['stage'], stage=stage,
                        intervals=column.intervals, column_id=column.identifier, reduced_cost=reduced,
                        added=added, local_admitted=column.local_admitted,
                        local_peaks=column.local_peaks, reason=proposal['reason']))
                if oracle.query_count >= budget.column_queries:
                    break
            new_lp = master.solve_lp()
            integer_results = []
            for _ in range(max(1, math.ceil(budget.complete_evaluations / budget.rounds))):
                if len(evaluations) >= budget.complete_evaluations or (budget.wall_seconds is not None
                        and time.perf_counter() - started >= budget.wall_seconds):
                    break
                integer = master.solve_integer(visited)
                if integer is None:
                    break
                visited.append(integer['column_ids'])
                evaluation = oracle.complete(integer['endpoints'], [oracle.revision])
                if evaluation is None:
                    stop = 'COMPLETE_EVALUATION_DEFERRED'
                    break
                evaluation['proxy_step_time_ms'] = integer['objective_ms']
                evaluation['pricing_replay_step_time_ms'] = oracle.snapshot(integer['endpoints'])['performance']['step_time_ms']
                evaluations.append(evaluation)
                integer_results.append(dict(endpoints=integer['endpoints'], objective_ms=integer['objective_ms'],
                    integer_optimal=integer['optimal'], prediction_id=evaluation['prediction_id'],
                    full_step_ms=evaluation['step_time_ms'], memory=evaluation['memory_admission']))
            history.append(dict(round=round_index, lp_before_ms=lp['objective_ms'],
                lp_after_ms=new_lp['objective_ms'], provider=provider, proposals=proposals,
                checked=checked, integers=integer_results,
                column_queries=oracle.query_count, pricing_cost_constructions=oracle.projection_count,
                elapsed_seconds=time.perf_counter() - started))
            if oracle.query_count >= budget.column_queries:
                break
            if stop == 'COMPLETE_EVALUATION_DEFERRED':
                break
            if not any(row['added'] for row in checked) and oracle.deferred_columns:
                stop = 'AWAITING_NATIVE_COST_OR_NEW_STRUCTURE'
                break
            if not proposals and provider.get('pending_frontier_states', 0):
                continue
            if not any(row['added'] for row in checked) and not integer_results:
                stop = 'NO_PROGRESS'; break
        finite = [row for row in evaluations if row['step_time_ms'] is not None
            and row['memory_admission'] != 'REJECT_OOM']
        result = dict(run_id=run_id, cost_revision=oracle.revision,
            domain=dict(boundaries=oracle.domain.boundaries, complete_layout_count=oracle.domain.layout_count()),
            budget=budget.__dict__, usage=dict(column_queries=oracle.query_count,
                complete_evaluations=len(evaluations), pricing_cost_constructions=oracle.projection_count,
                pricing_cost_seconds=oracle.snapshot_seconds, complete_evaluation_seconds=oracle.complete_seconds,
                elapsed_seconds=time.perf_counter() - started,
                provider_calls=sum(row['provider']['provider_calls'] for row in history)),
            fragment_cost_cache=oracle.cost_fragments().report() if oracle.module_costs else None,
            frontier_search=dict(frontier_states=sum(row['provider'].get('frontier_states',0)
                    for row in history),
                infeasible_pruned=sum(row['provider'].get('infeasible_pruned',0) for row in history),
                dominance_pruned=sum(row['provider'].get('dominance_pruned',0) for row in history),
                pricing_certified=False),
            best_prediction=min(finite, key=lambda row: row['step_time_ms']) if finite else None,
            best_pricing_proxy=min(evaluations, key=lambda row: row['pricing_replay_step_time_ms'])
                if evaluations else None,
            seed_step_time_ms=oracle.seed_report['performance']['step_time_ms'],
            evaluations=evaluations, history=history, stop=stop,
            deferred_cost_requests=[dict(request=row['request'],
                prediction_id=row['prediction']['prediction_id'],
                reason=row.get('reason','MISSING_NATIVE_EXECUTION_COST'),
                profiling_requests=row['prediction'].get('evidence', {}).get('profiling_requests', []))
                for row in oracle.deferred_columns.values()],
            evidence_level='archived_pricing_proxy_and_full_prediction_statuses', new_gpu_runs=0,
            training_speedup=None, optimality_claimed=False)
        (oracle.directory / 'search-result.json').write_text(json.dumps(result, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n')
        return result
