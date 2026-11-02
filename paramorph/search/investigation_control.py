from copy import deepcopy
from itertools import accumulate, product, groupby
import json
import time
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from contracts import Contract, ModuleRelation, RewriteIntent, fingerprint
from cost.runtime_calibration import configuration
from predictor.profile_database import normalized_execution, plan_signature, signature
from predictor.dag_projection import PROJECTION_REVISION
from region_partition.region_partitioner import RegionPartitioner
from search.conditional_partition import ConditionalPartitionSolver
from search.complete_strategy import complete_values
from search.continuation import ContinuationGuide
from search.decision_continuation import DecisionContinuationGuide
from search.execution_comparison import compare_execution_reports
from search.module_relations import owner_patterns, prefix_support
from search.complete_partition import CompletePartitionSearch
from search.execution_path_model import ExecutionPathModel
from search.communication_placement import placement_families
from search.planning_portfolio import PortfolioControl
from validation.parallel_validator import validate_candidate
from scheduling.native_microbatches import valid_microbatch_group


class ParallelInvestigationRanges(Contract):
    tp: list[StrictInt] | None = None
    pp: list[StrictInt] | None = None
    cp: list[StrictInt] | None = None
    ep: list[StrictInt] | None = None
    etp: list[StrictInt] | None = None
    dp: list[StrictInt] | None = None
    expert_dp: list[StrictInt] | None = None
    vpp: list[StrictInt] | None = None

    @model_validator(mode='after')
    def check_ranges(self):
        for values in self.model_dump(exclude_none=True).values():
            if not values or len(set(values)) != len(values) or min(values) < 1:
                raise ValueError('并行范围必须包含互不重复的正整数')
        return self


class InvestigationBudget(Contract):
    expansions: StrictInt = Field(ge=1)
    states_per_boundary: StrictInt = Field(ge=1)
    full_evaluations: StrictInt = Field(ge=1)
    candidates: StrictInt = Field(ge=1, le=64)
    seconds: float = Field(gt=0, allow_inf_nan=False)


class RuntimeInvestigationRanges(Contract):
    overlap_p2p: list[StrictBool] | None = Field(default=None,
        description='显式调查的P2P重叠取值；null保留指定参照的设置；VPP1由后端解析为false')
    micro_batch: list[StrictInt] | None = None
    group: list[StrictInt] | None = Field(default=None,
        description='显式调查的原生微批分组；每项取值与PP、DP及MBS联合检查，允许合法的末尾不足整组')
    recompute: list[Literal['none', 'full', 'selective-mlp']] | None = None
    optimizer_sharding: list[Literal['distributed', 'replicated']] | None = None
    rank_mapping: list[list[StrictInt]] | None = None

    @model_validator(mode='after')
    def check_values(self):
        if self.overlap_p2p is not None and (not self.overlap_p2p or len(set(self.overlap_p2p)) != len(self.overlap_p2p)):
            raise ValueError('P2P重叠范围必须包含互不重复的布尔值')
        for name in ('micro_batch', 'group', 'recompute', 'optimizer_sharding', 'rank_mapping'):
            values = getattr(self, name)
            if values is not None and (not values or len({fingerprint(value) for value in values}) != len(values)):
                raise ValueError(f'{name}范围必须非空且没有重复取值')
        if self.micro_batch is not None and min(self.micro_batch) < 1:
            raise ValueError('微批范围必须包含正整数')
        if self.group is not None and min(self.group) < 1:
            raise ValueError('微批分组范围必须包含正整数')
        return self


class FullPartitionSeed(Contract):
    pp: StrictInt = Field(ge=1)
    vpp: StrictInt = Field(ge=1)
    decoder_chunks: list[StrictInt] = Field(min_length=1)

class PlanningInvestigation(Contract):
    parent_trial_id: str
    module_scope: Literal['all', 'selected'] = Field(description='all开放全部模块边界，movable_module_ids必须为[]；selected只开放列表中的模块')
    movable_module_ids: list[StrictInt] = Field(description='module_scope为all时必须为[]，selected时填写互不重复的真实模块编号且不能为空')
    parallel_ranges: ParallelInvestigationRanges
    runtime_ranges: RuntimeInvestigationRanges = Field(default_factory=RuntimeInvestigationRanges)
    partition_seeds: list[FullPartitionSeed] = Field(default_factory=list, max_length=2,
        description='可以直接生成至多两项完整非均匀布局；没有具体布局时使用[]；种子参与评价并可继续数值重构')
    budget: InvestigationBudget
    hypothesis: str = Field(min_length=1)
    expected_execution_change: str = Field(min_length=1)
    possible_side_effects: list[str]
    evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode='after')
    def check_scope(self):
        if len(set(self.movable_module_ids)) != len(self.movable_module_ids):
            raise ValueError('可变模块集合包含重复编号')
        if self.module_scope == 'all' and self.movable_module_ids:
            raise ValueError('完整分区调查使用空的可变模块列表')
        if self.module_scope == 'selected' and not self.movable_module_ids:
            raise ValueError('局部分区调查必须指定可变模块')
        return self


class PlanningQuery(Contract):
    tool: Literal['inspect_execution_graph', 'inspect_critical_path', 'inspect_region_cost',
        'inspect_task_dependencies', 'inspect_memory', 'inspect_communication_flow',
        'inspect_rank_topology', 'get_budget_status', 'get_trial_history', 'compare_candidate_graph_delta']
    candidate_ids: list[str] = Field(default_factory=list, max_length=2)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4, ge=1, le=8)
    region_id: int | None = Field(default=None, ge=0)
    task_ids: list[str] = Field(default_factory=list, max_length=8)
    dependency_depth: int = Field(default=1, ge=1, le=3)
    dependency_direction: Literal['predecessors', 'successors', 'both'] = 'both'


class PlanningInvestigationBatch(Contract):
    actions: list[PlanningInvestigation] = Field(default_factory=list, max_length=4)
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str
    tool_requests: list[PlanningQuery] = Field(default_factory=list, max_length=4)

    @property
    def intents(self):
        return []


class InvestigationControl(PortfolioControl):
    @staticmethod
    def planning_cost_readiness(prediction):
        if prediction is None:
            return dict(status='UNAVAILABLE', source_prediction_id=None, required_costs=[])
        requests = prediction.get('evidence', {}).get('profiling_requests', [])
        required = [row for row in requests if row.get('profile_required_occurrence_count', 0) > 0]
        ordered = sorted(required, key=lambda row: (-row.get('profile_required_occurrence_count', 0),
            row.get('kind', ''), row.get('cost_key', '')))
        return dict(status=prediction.get('prediction_status'),
            source_prediction_id=prediction.get('prediction_id'),
            rankable_time_available=(prediction.get('prediction_status') != 'NEEDS_PROFILE'
                and prediction.get('performance', {}).get('step_time_ms') is not None),
            required_cost_record_count=len(required),
            required_costs=[{key: row[key] for key in ('kind', 'cost_key', 'identity', 'reason',
                'profile_required_occurrence_count', 'cost_statuses') if key in row} for row in ordered[:4]],
            capacity_status=prediction.get('memory', {}).get('admission'),
            scope='this_complete_candidate_and_recorded_execution_identity')

    def planning_rankable_trials(self):
        return [row for row in self.store.trials() if row.get('prediction') and not row.get('failure')
            and row['prediction'].get('cost_database_revision') == self.cost_revision
            and row['prediction'].get('evidence', {}).get('predictor_revision') == PROJECTION_REVISION
            and row['prediction'].get('prediction_status') != 'NEEDS_PROFILE'
            and row['prediction']['performance'].get('step_time_ms') is not None
            and row['prediction']['memory'].get('admission') != 'REJECT_OOM'
            and row['prediction'].get('planning_evaluation', {}).get('within_global_budget', True)]

    def restore_planning_parent(self, initial):
        candidates = self.planning_rankable_trials()
        parent = min(candidates, key=lambda row: row['prediction']['performance']['step_time_ms']) if candidates else initial
        feedback = self.store.search_states('PLANNING_FEEDBACK')
        if feedback and feedback[-1]['document']['selected'] == parent['trial_id']:
            self._planning_last_feedback = feedback[-1]['document']['execution_comparison']
        return parent

    def investigation_shapes(self, action, parent):
        extra = action.runtime_ranges.model_dump(exclude_none=True, exclude={'overlap_p2p'})
        if not extra:
            return self._parallel_investigation_shapes(action, parent)
        if not self.config.planning_full_strategy_enabled:
            raise ValueError('当前实验没有开放完整运行设置')
        fields = list(extra)
        shapes, rejected = [], []
        for values in product(*(extra[name] for name in fields)):
            changed = deepcopy(parent)
            request = changed['plan']['backend_config']
            request.update(dict(zip(fields, values)))
            if 'rank_mapping' in extra and sorted(request['rank_mapping']) != list(range(self.topology.num_gpus)):
                raise ValueError('Rank映射必须完整覆盖物理设备且不得重复')
            allowed_batch = self.domains.get('micro_batch', self.domains.get('micro_batch_size', []))
            if 'micro_batch' in extra and request['micro_batch'] not in allowed_batch:
                raise ValueError('微批取值超出原生后端域')
            if 'group' in extra and request['group'] not in self.domains.get('group', [parent['plan']['backend_config']['group']]):
                raise ValueError('微批分组取值超出原生后端域')
            if request.get('execution_backend', 'megatron_ddp') != 'megatron_ddp' and 'optimizer_sharding' in extra:
                raise ValueError('当前状态分片调查要求保持DDP执行后端')
            if request.get('memory_policy') is not None:
                request['memory_policy'] = dict(request['memory_policy'],
                    recompute_granularity='full' if request['recompute'] == 'full' else
                        'selective' if request['recompute'] == 'selective-mlp' else None,
                    recompute_modules=['mlp'] if request['recompute'] == 'selective-mlp' else [])
            rows, failures = self._parallel_investigation_shapes(action, changed)
            if action.runtime_ranges.rank_mapping is not None:
                for row in rows:
                    allowed = [item['request']['rank_mapping'] for item in placement_families(row)]
                    if row['rank_mapping'] not in allowed and row['rank_mapping'] != parent['plan']['backend_config']['rank_mapping']:
                        raise ValueError('当前共同映射域限定为原生并行轴排列')
            shapes.extend(rows)
            rejected.extend(failures)
        return list({plan_signature(shape): shape for shape in shapes}.values()), rejected

    def _parallel_investigation_shapes(self, action, parent):
        request = parent['plan']['backend_config']
        ranges = action.parallel_ranges.model_dump(exclude_none=True)
        if any(module < 0 or module >= self.model.num_layers for module in action.movable_module_ids):
            raise ValueError('可变模块超出真实模型范围')
        if action.module_scope == 'selected' and any(
                value != request[name] for name in ('pp', 'vpp') for value in ranges.get(name, [])):
            raise ValueError('改变PP或VPP需要完整分区调查')
        for name, values in ranges.items():
            if name in self.domains and not set(values) <= set(self.domains[name]):
                raise ValueError(f'并行范围{name}超出当前后端域')
        names = ('tp', 'pp', 'cp', 'ep', 'etp', 'vpp')
        shapes, rejected = [], []
        runtime_source = self.config.request if self.config.planning_runtime_reference == 'initial' else request
        overlaps = action.runtime_ranges.overlap_p2p
        if overlaps is not None and not self.config.planning_execution_context_enabled:
            raise ValueError('当前调查入口没有启用运行设置范围')
        overlap_values = [runtime_source.get('overlap_p2p', False)] if overlaps is None else overlaps
        for values in product(*(ranges.get(name, [request[name]]) for name in names), overlap_values):
            candidate = dict(request, **dict(zip(names, values[:-1])), overlap_p2p=values[-1])
            if self.model.num_experts is None and 'etp' not in ranges:
                candidate['etp'] = candidate['tp']
            world = self.topology.num_gpus
            dense = candidate['tp'] * candidate['pp'] * candidate['cp']
            expert = candidate['etp'] * candidate['ep'] * candidate['pp']
            if world % dense or world % expert:
                rejected.append(dict(values=dict(zip(names, values)), reason='world_size_divisibility'))
                continue
            candidate.update(dp=world // dense, expert_dp=world // expert)
            if any(candidate[name] not in ranges[name] for name in ('dp', 'expert_dp') if name in ranges):
                continue
            if candidate['global_batch'] % (candidate['dp'] * candidate['micro_batch']):
                rejected.append(dict(values=dict(zip(names, values)), reason='global_batch_divisibility'))
                continue
            slots = candidate['pp'] * candidate['vpp']
            if slots > self.model.num_layers:
                continue
            if candidate.get('execution_backend', 'megatron_ddp') == 'megatron_ddp':
                candidate.update(dp_inner_size=candidate['dp'] * candidate['cp'], dp_outer_size=1,
                    expert_dp_inner_size=candidate['expert_dp'], expert_dp_outer_size=1)
            else:
                dense_outer, expert_outer = candidate.get('dp_outer_size', 1), candidate.get('expert_dp_outer_size', 1)
                if candidate['dp'] * candidate['cp'] % dense_outer or candidate['expert_dp'] % expert_outer:
                    rejected.append(dict(values=dict(zip(names, values)), reason='device_mesh_divisibility'))
                    continue
                candidate.update(dp_inner_size=candidate['dp'] * candidate['cp'] // dense_outer,
                    expert_dp_inner_size=candidate['expert_dp'] // expert_outer)
            microbatches = candidate['global_batch'] // (candidate['dp'] * candidate['micro_batch'])
            requested_groups = ([request['group']] if action.runtime_ranges.group is not None
                else self.domains.get('group', [request['group']]))
            groups = [value for value in requested_groups
                if valid_microbatch_group(microbatches, candidate['pp'], candidate['vpp'], value)]
            if not groups:
                rejected.append(dict(values=dict(zip(names, values)), reason='no_supported_microbatch_group'))
                continue
            candidate['group'] = min(groups, key=lambda value: (value != request['group'], abs(value - request['group']), value))
            if candidate['pp'] != request['pp'] or candidate['vpp'] != request['vpp']:
                quotient, remainder = divmod(self.model.num_layers, slots)
                candidate['decoder_chunks'] = [quotient + (index < remainder) for index in range(slots)]
                candidate['pipeline_schedule'] = '1f1b' if candidate['vpp'] == 1 else 'interleaved_1f1b'
            if candidate['vpp'] == 1:
                candidate['overlap_p2p'] = False
            if (self.config.planning_runtime_scope == 'fixed' and candidate['vpp'] > 1
                    and candidate['overlap_p2p'] != self.config.request.get('overlap_p2p', False)):
                rejected.append(dict(values=dict(zip(names, values)), reason='fixed_runtime_context'))
                continue
            regions = RegionPartitioner(self.model, configuration(candidate)).from_counts(candidate['decoder_chunks'])
            valid, errors = validate_candidate(configuration(candidate), self.model, self.topology, regions)
            if not valid:
                rejected.append(dict(values=dict(zip(names, values)), reason=errors))
                continue
            self.require_fixed_training_semantics(request, candidate)
            shapes.append(candidate)
        return list({plan_signature(shape): shape for shape in shapes}.values()), rejected

    def investigation_execution_context(self, action, parent, shape):
        requested = parent['plan']['backend_config']
        proposal = getattr(self, '_planning_portfolio_origins', {}).get(
            fingerprint(action.model_dump(mode='json')), action.model_dump(mode='json'))
        before, after = normalized_execution(requested), normalized_execution(shape)
        explicit = set(action.parallel_ranges.model_dump(exclude_none=True))
        explicit.update(action.runtime_ranges.model_dump(exclude_none=True))
        fields = {}
        derived = {'dp', 'expert_dp', 'dp_inner_size', 'dp_outer_size', 'expert_dp_inner_size',
            'expert_dp_outer_size', 'group', 'pipeline_schedule'}
        if self.model.num_experts is None:
            derived.add('etp')
        for name, value in after.items():
            origin = 'PROPOSED' if name in explicit else 'INHERITED'
            rule = None
            if name in derived and name not in explicit and value != before.get(name):
                origin, rule = 'BACKEND_RESOLVED', 'native_world_mesh_or_pipeline_schedule'
            if name == 'overlap_p2p' and shape['vpp'] == 1:
                origin, rule = 'BACKEND_RESOLVED', 'noninterleaved_pipeline_disables_overlap'
            elif name == 'overlap_p2p' and name not in explicit and self.config.planning_runtime_reference == 'initial':
                origin, rule = 'REFERENCE_FIXED', 'shared_initial_runtime_condition'
            fields[name] = dict(value=value, parent_value=before.get(name), origin=origin, rule=rule,
                proposal_values=(proposal['parallel_ranges'].get(name) if name in proposal['parallel_ranges']
                    else proposal.get('runtime_ranges', {}).get(name)))
        return dict(parent_trial_id=parent['trial_id'], execution_signature=signature(shape),
            candidate_signature=plan_signature(shape), runtime_reference=self.config.planning_runtime_reference,
            fields=fields, cost_revision=self.cost_revision)

    def validate_investigation(self, action):
        action = PlanningInvestigation.model_validate(action)
        trials = {row['trial_id']: row for row in self.store.trials()}
        if action.parent_trial_id not in trials:
            raise ValueError('规划动作引用了不存在的父候选')
        if self.config.planning_fixed_parent_id is not None and action.parent_trial_id != self.config.planning_fixed_parent_id:
            raise ValueError('当前控制实验需要使用共同指定的父策略')
        known = {row[0] for row in self.store.connection.execute('SELECT id FROM evidence')}
        if not set(action.evidence_refs) <= known:
            raise ValueError('规划动作引用了未取得的证据')
        parent = trials[action.parent_trial_id]
        self.require_fixed_training_semantics(self.config.request, parent['plan']['backend_config'])
        shapes, rejected = self.investigation_shapes(action, parent)
        if not shapes:
            raise ValueError(dict(reason='调查没有满足当前执行条件的原生配置', rejected=rejected))
        if action.partition_seeds and not self.config.planning_full_strategy_enabled:
            raise ValueError('完整布局种子需要开放全维度策略入口')
        for seed in action.partition_seeds:
            if len(seed.decoder_chunks) != seed.pp * seed.vpp or min(seed.decoder_chunks) < 1:
                raise ValueError('完整布局种子需要PP乘VPP个正长度区间')
            if sum(seed.decoder_chunks) != self.model.num_layers:
                raise ValueError('完整布局种子没有覆盖模型全部模块')
            matched = [shape for shape in shapes if shape['pp'] == seed.pp and shape['vpp'] == seed.vpp]
            if not matched:
                raise ValueError('完整布局种子不属于本项调查的合法并行条件')
            for shape in matched:
                if not prefix_support(seed.decoder_chunks, self.model.num_layers, shape['pp'], shape['vpp'],
                        self.investigation_relations(action, shape)):
                    raise ValueError('完整布局种子改变了冻结模块归属')
                self.rewriter.candidate(dict(shape, decoder_chunks=seed.decoder_chunks))
        return action, parent, shapes, rejected

    def planning_validation_feedback(self, batch):
        issues = []
        if not batch.actions:
            issues.append(dict(reason='完成查询后至少提交一项规划动作'))
        for index, action in enumerate(batch.actions):
            try:
                self.validate_investigation(action)
            except ValueError as error:
                issues.append(dict(action_index=index, reason=str(error)))
        if not issues:
            return None
        result = dict(status='REVISION_REQUIRED', issues=issues,
            instruction='根据真实父候选、模块编号、后端域和已登记证据重新提交动作；改变PP/VPP使用all。')
        tool = dict(tool='validate_planning_investigation')
        identifier = fingerprint(dict(tool=tool, result=result, batch=batch.model_dump(mode='json')))
        receipt = dict(tool=tool, result=result, evidence_id=identifier)
        self.store.add_evidence(identifier, 'FAILURE', receipt)
        return receipt

    def investigation_relations(self, action, request):
        if action.module_scope == 'all':
            return ()
        boundaries = list(accumulate(request['decoder_chunks'], initial=0))
        movable = set(action.movable_module_ids)
        return tuple(ModuleRelation(relation='allowed_virtual', module_ids=modules,
            virtual_slots=[slot], evidence_refs=action.evidence_refs,
            rationale='本次调查保留开放集合以外模块的父配置归属')
            for slot, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
            if (modules := [module for module in range(start, end) if module not in movable]))

    def execute_investigation(self, action):
        self.check_walltime()
        action, parent, shapes, rejected = self.validate_investigation(action)
        started = time.perf_counter()
        usage = self.store.usage()
        requested = action.budget
        remaining_seconds = max(0.0, self.config.budget.max_walltime_seconds - self.store.search_seconds())
        grant = dict(expansions=min(requested.expansions, self.config.conditional_partition_max_expansions),
            states_per_boundary=min(requested.states_per_boundary, self.config.conditional_partition_max_states),
            full_evaluations=min(requested.full_evaluations, max(0,
                self.config.budget.max_cpu_predictions - int(usage.get('cpu_predictions', 0)))),
            candidates=min(requested.candidates, self.config.budget.max_candidates_per_round),
            seconds=min(requested.seconds, self.config.investigation_max_seconds, remaining_seconds))
        deadline = started + grant['seconds']
        identity = fingerprint(dict(action=action.model_dump(mode='json'), cost_revision=self.cost_revision,
            prior_usage=usage, implementation='planning_investigation_v1'))
        profiles = [json.loads(row[0]) for row in self.predictor.database.connection.execute('SELECT document FROM profiles')]
        solver = ConditionalPartitionSolver(self.rewriter, profiles, backend_source_hashes=self.backend_identity)
        reference = getattr(self, '_planning_projection_reference', parent['plan']['backend_config'])
        evaluated, retained, expanded, scope_records = {}, {}, 0, []
        completed_evaluations = []
        missing_cost_signature, repeated_missing_costs = None, 0
        budget_admission_closed = False
        current_origin = 'native_reference'
        shape_prediction_start, shape_prediction_limit = 0, grant['full_evaluations']

        def evaluate(candidate):
            nonlocal missing_cost_signature, repeated_missing_costs, budget_admission_closed
            count = self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0)
            if budget_admission_closed or time.perf_counter() >= deadline or count >= grant['full_evaluations']:
                return None
            if count - shape_prediction_start >= shape_prediction_limit:
                return None
            result = self.evaluate_planning_request(candidate, reference)
            if result is not None:
                result = dict(result, planning_prediction_new=(
                    self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0) > count))
                candidate_key = plan_signature(candidate)
                existing = self._planning_registered.get(candidate_key)
                if existing is None:
                    intent = RewriteIntent(operation='JointTransform', values=complete_values(candidate),
                        adjustable=[], preserve=[], parent_plan_id=parent['plan']['backend_config']['run_id'],
                        hypothesis=action.hypothesis, evidence_refs=action.evidence_refs)
                    rewrite = self.rewriter.complete_and_rewrite(parent['plan']['backend_config'], intent,
                        limit=1, allow_unchanged=True)[0]
                    rewrite['proposal_kind'] = 'planning_investigation'
                    existing = self.register_constraint_candidate(rewrite, parent, intent)
                self.store.record_session_time(self.session_id, time.perf_counter() - self.started)
                completed_seconds = self.store.search_seconds()
                result = dict(result, planning_evaluation=dict(
                    completed_search_seconds=completed_seconds,
                    within_global_budget=completed_seconds <= self.config.budget.max_walltime_seconds,
                    action_admission_deadline_seconds=grant['seconds']))
                evaluated[plan_signature(candidate)] = (deepcopy(candidate), result)
                if existing.get('prediction') is None:
                    existing = self.store.update_trial(existing['trial_id'], prediction=result,
                        validation=result['validation'], state='PROJECTED')
                    self.family_ledger.record_trial(existing, self.cost_revision)
                self._planning_registered[candidate_key] = existing
                retained[existing['trial_id']] = existing
                completed_evaluations.append(dict(plan_signature=plan_signature(candidate),
                    decoder_chunks=candidate['decoder_chunks'],
                    values={name: candidate[name] for name in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'group', 'overlap_p2p',
                        'micro_batch', 'recompute', 'optimizer_sharding', 'rank_mapping')},
                    status=result['prediction_status'], step_time_ms=result['performance'].get('step_time_ms'),
                    completed_search_seconds=completed_seconds,
                    within_global_budget=result['planning_evaluation']['within_global_budget']))
                completed_evaluations[-1]['candidate_origin'] = current_origin
                completed_evaluations[-1]['new_prediction'] = result['planning_prediction_new']
                self.store.add_search_state('PLANNING_EVALUATION', dict(action_id=identity,
                    action=action.model_dump(mode='json'), candidate_id=existing['trial_id'],
                    evaluation=completed_evaluations[-1]))
                if self.config.planning_portfolio_enabled and result['prediction_status'] == 'NEEDS_PROFILE':
                    signature = tuple(sorted(row['cost_key'] for row in result.get('evidence', {}).get('profiling_requests', [])
                        if row.get('profile_required_occurrence_count', 0) > 0 and row.get('cost_key')))
                    repeated_missing_costs = repeated_missing_costs + 1 if signature == missing_cost_signature else 1
                    missing_cost_signature = signature
                    budget_admission_closed = bool(signature and repeated_missing_costs >= 2)
                if not result['planning_evaluation']['within_global_budget']:
                    return None
            return result

        for shape_index, shape in enumerate(shapes):
            if budget_admission_closed or time.perf_counter() >= deadline or expanded >= grant['expansions']:
                break
            if grant['expansions'] - expanded < shape['pp'] * shape['vpp']:
                scope_records.append(dict(status='UNKNOWN', reason='expansion_budget'))
                break
            shape_prediction_start = int(self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0))
            available = grant['full_evaluations'] - shape_prediction_start
            if available <= 0:
                break
            # 保持提议顺序，同时为后续执行条件保留完整评价机会。
            shape_prediction_limit = (available + len(shapes) - shape_index - 1) // (len(shapes) - shape_index)
            relations = self.investigation_relations(action, shape)
            execution_context = self.investigation_execution_context(action, parent, shape)
            self.store.add_search_state('PLANNING_EXECUTION_CONTEXT', dict(action_id=identity, **execution_context))
            patterns, complete = owner_patterns(relations, self.model.num_layers,
                shape['pp'], shape['vpp'], grant['states_per_boundary'],
                time_limit_s=max(0.001, min(5.0, deadline - time.perf_counter())))
            if not patterns:
                scope_records.append(dict(status='UNKNOWN', reason='owner_pattern_budget'))
                continue
            donors = [shape]
            donors.extend(row['plan']['backend_config'] for row in self.store.trials()
                if not row.get('failure') and (row.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM')
            donors = [row for row in donors if signature(row) == signature(shape)]
            projection_context = self.projection_parent_context(reference)
            cache = getattr(self, '_continuation_predictions', {})
            donor_reports = {}
            for donor in donors:
                candidate_id = plan_signature(donor)
                cache_key = fingerprint(dict(cost_revision=self.cost_revision,
                    candidate=candidate_id, parent=projection_context))
                cached = cache.get(cache_key)
                if cached is not None:
                    donor_reports[candidate_id] = cached
            donor_scores = {key: prediction['performance']['step_time_ms'] for key, prediction in donor_reports.items()
                if prediction['prediction_status'] != 'NEEDS_PROFILE'
                and prediction['performance'].get('step_time_ms') is not None
                and prediction['memory'].get('admission') != 'REJECT_OOM'}
            current_origin = 'native_reference'
            reference_report = (evaluate(shape)
                if self.config.planning_budget_allocation == 'completion_reserve'
                    or self.config.planning_partition_engine == 'path' else None)
            if (reference_report is not None and reference_report['prediction_status'] != 'NEEDS_PROFILE'
                    and reference_report['performance'].get('step_time_ms') is not None
                    and reference_report['memory'].get('admission') != 'REJECT_OOM'):
                donor_scores[plan_signature(shape)] = reference_report['performance']['step_time_ms']
            if reference_report is not None:
                donor_reports[plan_signature(shape)] = reference_report
            current_origin = 'agent_complete_layout' if self.config.planning_selector == 'agent' else 'deterministic_complete_layout'
            for seed in action.partition_seeds:
                if seed.pp != shape['pp'] or seed.vpp != shape['vpp']:
                    continue
                seeded = dict(shape, decoder_chunks=seed.decoder_chunks)
                seeded_report = evaluate(seeded)
                if seeded_report is None:
                    break
                donors.append(seeded)
                donor_reports[plan_signature(seeded)] = seeded_report
                if (seeded_report['prediction_status'] != 'NEEDS_PROFILE'
                        and seeded_report['performance'].get('step_time_ms') is not None
                        and seeded_report['memory'].get('admission') != 'REJECT_OOM'):
                    donor_scores[plan_signature(seeded)] = seeded_report['performance']['step_time_ms']
            current_origin = self.config.planning_partition_engine + '_reconstruction'
            if self.config.planning_partition_engine in ('complete', 'path'):
                profile = solver._profile(shape)
                full = CompletePartitionSearch(self.model.num_layers, shape, profile, relations,
                    evaluate, deadline, donors, donor_scores, reports=donor_reports,
                    graph_loader=self.load_prediction_graph, engine=self.config.planning_partition_engine)
                remaining_queries = max(0, grant['full_evaluations'] - int(
                    self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0)))
                remaining_queries = min(remaining_queries, max(0, shape_prediction_limit - int(
                    self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0) - shape_prediction_start)))
                statistics = full.run(remaining_queries, grant['expansions'] - expanded)
                expanded += statistics['expansions']
                scope_records.append(dict(global_values={name: shape[name] for name in (
                    'tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'group', 'overlap_p2p',
                    'micro_batch', 'recompute', 'optimizer_sharding')},
                    fixed_module_count=sum(len(row.module_ids) for row in relations),
                    prediction_quota=shape_prediction_limit,
                    statistics=statistics, continuation=dict(engine=self.config.planning_partition_engine,
                        evaluations=statistics['evaluations'])))
                continue
            remaining_queries = max(0, grant['full_evaluations'] - int(
                self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0)))
            remaining_queries = min(remaining_queries, max(0, shape_prediction_limit - int(
                self.store.usage().get('cpu_predictions', 0) - usage.get('cpu_predictions', 0) - shape_prediction_start)))
            guide_type = ContinuationGuide if self.config.continuation_evaluation_timing == 'late' else DecisionContinuationGuide
            timing_options = {} if guide_type is ContinuationGuide else dict(
                timing=self.config.continuation_evaluation_timing, early_fraction=self.config.continuation_early_fraction)
            guide = guide_type(donors, evaluate, remaining_queries,
                admission_evaluator=self.predictor.evaluate_memory_bound,
                deadline=deadline,
                selection_policy=self.config.continuation_donor_selection,
                donor_limit=self.config.continuation_donor_limit,
                donor_scores=donor_scores, **timing_options,
                completion_filter=(lambda candidate: prefix_support(candidate['decoder_chunks'],
                    self.model.num_layers, shape['pp'], shape['vpp'], relations)) if relations else None)
            if reference_report is not None:
                guide.prime_reference(shape, reference_report)
            profile = solver._profile(shape)
            allowance = max(shape['pp'] * shape['vpp'],
                (grant['expansions'] - expanded) // max(1, len(shapes) - shape_index))
            states, statistics = solver._partition(shape, profile, grant['states_per_boundary'], allowance,
                module_relations=relations, continuation=guide, deadline=deadline)
            expanded += statistics['expansions']
            for state in sorted(states, key=lambda item: guide.score(item, shape, solver._score)):
                candidate = dict(shape, decoder_chunks=list(state.counts))
                if plan_signature(candidate) not in evaluated:
                    evaluate(candidate)
            scope_records.append(dict(global_values={name: shape[name] for name in (
                'tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'group', 'overlap_p2p')},
                fixed_module_count=sum(len(row.module_ids) for row in relations),
                prediction_quota=shape_prediction_limit,
                cp_sat_owner_pattern_count=len(patterns), cp_sat_owner_patterns_complete=complete,
                statistics=statistics, continuation=guide.report()))
        after = self.store.usage()
        elapsed = time.perf_counter() - started
        selected = self.retain_candidate_portfolio(list(retained.values()), grant['candidates'])
        receipt = dict(action_id=identity, action=action.model_dump(mode='json'), granted_budget=grant,
            budget_allocation=self.config.planning_budget_allocation,
            completed_shapes=scope_records, rejected_shapes=rejected,
            candidate_ids=[row['trial_id'] for row in selected],
            evaluated_candidate_ids=list(retained), expansions=expanded,
            completed_evaluations=completed_evaluations,
            actual_predictions=after.get('cpu_predictions', 0) - usage.get('cpu_predictions', 0),
            elapsed_seconds=elapsed, prediction_seconds=after.get('cpu_seconds', 0) - usage.get('cpu_seconds', 0),
            deadline_exceeded=elapsed > grant['seconds'], scope='declared_investigation_only',
            repeated_missing_costs=repeated_missing_costs,
            admission_closed_for_repeated_costs=budget_admission_closed,
            cost_revision=self.cost_revision, projection_reference=plan_signature(reference),
            time_budget_semantics='deadline_for_admitting_work_nonpreemptive_evaluations_fully_charged')
        self.store.add_evidence(identity, 'PREDICTED', receipt)
        self.store.add_search_state('PLANNING_INVESTIGATION', receipt)
        return receipt

    def planning_observation(self, parent):
        request = parent['plan']['backend_config']
        solver = ConditionalPartitionSolver(self.rewriter, [json.loads(row[0]) for row in
            self.predictor.database.connection.execute('SELECT document FROM profiles')],
            backend_source_hashes=self.backend_identity)
        profile = solver._profile(request)
        source = None
        if profile is not None:
            source = dict(profile_id=profile['id'],
                modules=[[index, profile['layers'][str(index)]['forward_ms'],
                    profile['layers'][str(index)]['backward_ms']] for index in range(self.model.num_layers)],
                module_columns=['module_id', 'forward_ms', 'backward_ms'], regions=profile['regions'])
            self.store.add_evidence(profile['id'], 'MEASURED', dict(kind='source_module_costs',
                profile_id=profile['id'], source_request=profile['request']))
        prediction = parent.get('prediction')
        complete = dict(prediction, execution_graph=self.load_prediction_graph(prediction)) if prediction else None
        path_evidence = None
        if (self.config.planning_partition_engine == 'path' and profile is not None and complete is not None
                and prediction['prediction_status'] != 'NEEDS_PROFILE'):
            cache = getattr(self, '_planning_path_guides', {})
            key = prediction['prediction_id']
            if key not in cache:
                cache[key] = ExecutionPathModel(request, profile, complete['execution_graph'])
                self._planning_path_guides = cache
            path_evidence = dict(kind='execution_path_boundary_effects', **cache[key].influence(request),
                source_prediction_id=key, cost_revision=self.cost_revision)
            identifier = fingerprint(path_evidence)
            self.store.add_evidence(identifier, 'PREDICTED', path_evidence)
            path_evidence = dict(path_evidence, evidence_id=identifier)
        comparison = getattr(self, '_planning_last_feedback', None) or compare_execution_reports(
            request, complete, request, complete, limit=self.topology.num_gpus)
        visible_comparison = {key: value for key, value in comparison.items() if key not in ('rank_changes', 'before', 'after')}
        for side in ('before', 'after'):
            visible_comparison[side] = {key: value for key, value in comparison.get(side, {}).items()
                if key in ('step_time_ms', 'modeled_exposed_communication_ms', 'exposed_communication_scope',
                    'memory_admission', 'module_cost_source')}
        visible_comparison['rank_columns'] = ['rank', 'before_compute_ms', 'after_compute_ms',
            'before_critical_tasks', 'after_critical_tasks', 'after_peak_bytes']
        visible_comparison['ranks'] = [[row['rank'], row['before']['compute_occupied_ms'],
            row['after']['compute_occupied_ms'], row['before']['critical_task_count'],
            row['after']['critical_task_count'], row['after']['peak_bytes']]
            for row in comparison.get('rank_changes', []) if row['rank'] % request['tp'] == 0]
        critical_tasks = []
        if complete is not None and complete.get('execution_graph'):
            tasks = complete['execution_graph'].get('tasks', {})
            critical_ids = prediction.get('diagnosis', {}).get('critical_tasks', [])
            ordered = sorted((key for key in critical_ids if key in tasks),
                key=lambda key: tasks[key]['end_ms'] - tasks[key]['start_ms'], reverse=True)
            critical_tasks = [[key, tasks[key]['kind'], tasks[key]['rank'],
                tasks[key]['start_ms'], tasks[key]['end_ms'], tasks[key].get('metadata', {}).get('layer_ids', [])]
                for key in ordered[:12]]
        trial_rows = []
        history = self.store.trials()
        visible_ids = {row['trial_id'] for row in history[-2:]} | {parent['trial_id']}
        visible_ids.update(row['trial_id'] for row in sorted(self.planning_rankable_trials(),
            key=lambda row: row['prediction']['performance']['step_time_ms'])[:2])
        cost_evidence = {}
        def compact_readiness(prediction):
            readiness = self.planning_cost_readiness(prediction)
            required_refs = []
            for record in readiness.pop('required_costs')[:2]:
                key = record.get('cost_key') or fingerprint(record)
                cost_evidence[key] = {field: record[field] for field in ('kind', 'identity', 'cost_statuses') if field in record}
                if self.store.connection.execute('SELECT id FROM evidence WHERE id=?', (key,)).fetchone() is None:
                    self.store.add_evidence(key, 'PREDICTED', dict(kind='required_cost_identity',
                        cost_key=key, source_kind=record.get('kind'), identity=record.get('identity')))
                required_refs.append(key)
            readiness['required_cost_refs'] = required_refs
            return readiness

        parent_readiness = compact_readiness(prediction)
        for row in history:
            if row['trial_id'] not in visible_ids:
                continue
            pred = row.get('prediction') or {}
            readiness = compact_readiness(pred)
            trial_rows.append(dict(trial_id=row['trial_id'], values={name: row['plan']['backend_config'][name]
                for name in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'decoder_chunks')},
                prediction_status=pred.get('prediction_status'), step_time_ms=pred.get('performance', {}).get('step_time_ms'),
                memory_admission=pred.get('memory', {}).get('admission'),
                cost_readiness=readiness,
                evidence_id=pred.get('prediction_id'), runtime=dict(
                    overlap_p2p=row['plan']['backend_config'].get('overlap_p2p', False),
                    recompute=row['plan']['backend_config']['recompute'],
                    micro_batch=row['plan']['backend_config']['micro_batch'],
                    optimizer_sharding=row['plan']['backend_config']['optimizer_sharding'],
                    rank_mapping=row['plan']['backend_config']['rank_mapping'],
                    pipeline_schedule=normalized_execution(row['plan']['backend_config'])['pipeline_schedule'])))
        states = [row['document'] for row in self.store.search_states('PLANNING_INVESTIGATION')]
        coverage_feedback = [{key: record[key] for key in ('depth', 'policy', 'selected_layouts',
            'prefix_count', 'unique_new_completions', 'uncovered_prefix_count', 'uncovered_prefixes',
            'structurally_distinct_donor_sets', 'redundant_donor_count', 'equivalent_donor_groups')}
            for state in states[-3:] for shape in state['completed_shapes']
            for record in shape.get('continuation', {}).get('donor_selection_rounds', [])[-1:]]
        units = []
        for key, rows in groupby(self.model_graph['nodes'],
                key=lambda node: (node['kind'], node['module_kind'], node['backend_callable'])):
            units.append(dict(module_ids=[row['layer_id'] for row in rows], kind=key[0],
                module_kind=key[1], backend_callable=key[2]))
        return dict(parent_trial_id=parent['trial_id'], parent_request={name: request[name] for name in (
            'tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'decoder_chunks', 'micro_batch',
            'sequence', 'global_batch', 'recompute', 'pipeline_schedule', 'overlap_p2p', 'rank_mapping')},
            model_module_types=[kind.value for kind in self.model.layer_types], source_costs=source,
            model_structure=dict(partition_units=units,
                forward_dependencies=self.model_graph['forward_dependencies'],
                special_modules=self.model_graph['special_modules'],
                supported_model_operations=self.model_graph['supported_model_operations']),
            parent_cost_readiness=parent_readiness,
            required_cost_evidence=cost_evidence,
            execution_feedback=visible_comparison,
            execution_path_evidence=path_evidence,
            critical_task_columns=['task_id', 'operation', 'rank', 'start_ms', 'end_ms', 'module_ids'],
            predicted_critical_tasks=critical_tasks, candidate_history=trial_rows,
            previous_investigations=[dict(action_id=record['action_id'],
                parent_trial_id=record['action']['parent_trial_id'],
                module_scope=record['action']['module_scope'], modules=record['action']['movable_module_ids'],
                parallel_ranges=record['action']['parallel_ranges'],
                runtime_ranges=record['action'].get('runtime_ranges', {}), granted_budget=record['granted_budget'],
                actual_predictions=record['actual_predictions'], elapsed_seconds=record['elapsed_seconds'])
                for record in states[-3:]], backend_domains=self.domains,
            continuation_coverage_feedback=coverage_feedback,
            execution_context_rules=dict(runtime_reference=self.config.planning_runtime_reference,
                runtime_scope=self.config.planning_runtime_scope,
                fixed_parent_trial_id=self.config.planning_fixed_parent_id,
                initial_overlap_p2p=self.config.request.get('overlap_p2p', False),
                overlap_range_enabled=self.config.planning_execution_context_enabled,
                noninterleaved_overlap=False,
                continuation_timing=self.config.continuation_evaluation_timing),
            decision_feedback=[dict(action_id=record['action_id'], summaries=[dict(
                global_values=shape['global_values'],
                timing=shape['continuation'].get('evaluation_policy'),
                first_depth=shape['continuation'].get('first_evaluation_depth'),
                comparisons=shape['continuation'].get('comparable_pairs', 0),
                rank_reversals=shape['continuation'].get('rank_reversals', 0),
                changed_retention=shape['continuation'].get('changed_pair_decisions', 0))
                for shape in record['completed_shapes']]) for record in states[-2:]],
            evidence_ids=list(dict.fromkeys(([profile['id']] if profile is not None else [])
                + ([prediction['prediction_id']] if prediction else [])
                + ([path_evidence['evidence_id']] if path_evidence is not None else [])
                + [row['evidence_id'] for row in trial_rows if row.get('evidence_id')]
                + [record['action_id'] for record in states[-3:]])),
            budget=dict(usage=self.store.usage(), limits=self.config.budget.model_dump(),
                partition_caps=dict(expansions=self.config.conditional_partition_max_expansions,
                    states_per_boundary=self.config.conditional_partition_max_states,
                    seconds=self.config.investigation_max_seconds)),
            module_scope_semantics='selected固定集合以外模块；改变PP或VPP使用all；未指定DP与Expert-DP由设备数量推导')

    def deterministic_investigation(self, parent, round_index):
        request = parent['plan']['backend_config']
        boundaries = list(accumulate(request['decoder_chunks'], initial=0))
        critical = (parent.get('prediction') or {}).get('diagnosis', {}).get('critical_regions', [])
        if self.config.planning_selector == 'deterministic_restart' or round_index % 3 == 2:
            modules, mode = [], 'all'
        elif round_index % 3 == 0:
            modules, mode = sorted({boundaries[1] - 1, boundaries[-2]}), 'selected'
        else:
            region = critical[0] if critical and isinstance(critical[0], int) else len(boundaries) - 2
            region = min(region, len(boundaries) - 2)
            modules, mode = sorted({max(0, boundaries[region] - 1), min(self.model.num_layers - 1, boundaries[region])}), 'selected'
        width = (1, 2, 4)[round_index % 3] if self.config.planning_selector == 'deterministic_restart' else 4
        parameter_ranges = ParallelInvestigationRanges()
        if mode == 'all' and round_index >= 2:
            parameter_ranges = ParallelInvestigationRanges(tp=list(self.domains['tp']),
                pp=list(self.domains['pp']), etp=list(self.domains['etp']), vpp=list(self.domains['vpp']))
        prediction = parent.get('prediction') or {}
        return PlanningInvestigation(parent_trial_id=parent['trial_id'], module_scope=mode,
            movable_module_ids=modules, parallel_ranges=parameter_ranges,
            runtime_ranges=RuntimeInvestigationRanges(overlap_p2p=(
                ([self.config.request.get('overlap_p2p', False)] if self.config.planning_runtime_scope == 'fixed'
                    else [False, True]) if self.config.planning_execution_context_enabled else None)),
            budget=InvestigationBudget(expansions=self.config.conditional_partition_max_expansions,
                states_per_boundary=width, full_evaluations=8, candidates=8, seconds=30),
            hypothesis='使用首尾任务、关键区域及完整分区重启调查同一合法域',
            expected_execution_change='通过同一完整执行评价器比较调查结果', possible_side_effects=[],
            evidence_refs=[prediction['prediction_id']])

    def run_investigation_loop(self):
        if self.config.gpu_execution_enabled:
            raise ValueError('规划调查入口需要关闭GPU执行，使用已有成本完成搜索')
        parent = self.initialize()
        if parent is None:
            return self.finish()
        self._planning_projection_reference = self.projection_parent_context(parent['plan']['backend_config'])
        self._planning_projection_reference['run_id'] = 'planning-fixed-reference-' + parent['plan_hash'][:16]
        self._planning_registered = {plan_signature(row['plan']['backend_config']): row for row in self.store.trials()}
        reference_key = fingerprint(dict(cost_revision=self.cost_revision,
            candidate=plan_signature(parent['plan']['backend_config']), parent=self._planning_projection_reference))
        if parent.get('prediction') is not None:
            self._continuation_predictions = {reference_key: parent['prediction']}
        reusable = {identifier for row in self.store.search_states('PLANNING_INVESTIGATION')
            if row['document']['cost_revision'] == self.cost_revision
            and row['document']['projection_reference'] == plan_signature(self._planning_projection_reference)
            for identifier in row['document']['evaluated_candidate_ids']}
        for row in self.store.trials():
            if (row['trial_id'] in reusable and row.get('prediction') is not None
                    and row['prediction'].get('evidence', {}).get('predictor_revision') == PROJECTION_REVISION):
                key = fingerprint(dict(cost_revision=self.cost_revision,
                    candidate=plan_signature(row['plan']['backend_config']), parent=self._planning_projection_reference))
                self._continuation_predictions[key] = row['prediction']
        parent = self.restore_planning_parent(parent)
        for round_index in range(int(self.store.usage().get('agent_rounds', 0)), self.config.budget.max_agent_rounds):
            self.check_walltime()
            if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                break
            observation = self.planning_observation(parent)
            if self.config.planning_portfolio_enabled:
                actions = self.portfolio_actions(parent, round_index, observation, PlanningInvestigationBatch)
                if not actions:
                    break
            elif self.config.agent_execution_enabled and self.config.planning_selector == 'agent':
                batch, _ = self.invoke_with_tools('JointStrategyAgent',
                    '根据真实模块成本、DAG任务占用、通信与显存、历史调查及剩余费用，自主提交1至4项规划动作。'
                    '动作必须明确父候选、联合可变模块或完整分区、并行参数范围和调查预算。'
                    '保持固定训练负载。选择相隔区域需要说明共同执行依据；也可以调查全局分区。'
                    'DP和Expert-DP可留空以按设备数量推导。改变PP/VPP时使用all。'
                    '不要重复没有新证据的调查。查询通过tool_requests返回；完成时将其置空。'
                    '没有GPU预算时仅进行CPU评价，预测与容量未知状态分别解释。',
                    observation, PlanningInvestigationBatch)
                if not batch.actions:
                    raise ValueError('完成查询后必须提交至少一项规划动作')
                actions = batch.actions
            else:
                actions = [self.deterministic_investigation(parent, round_index)]
            receipts = [self.execute_investigation(action) for action in actions]
            previous_parent = parent
            candidates = self.planning_rankable_trials()
            if candidates:
                parent = min(candidates, key=lambda row: row['prediction']['performance']['step_time_ms'])
            investigated_ids = {identifier for row in receipts for identifier in row['evaluated_candidate_ids']}
            investigated = [row for row in candidates if row['trial_id'] in investigated_ids
                and row['trial_id'] != previous_parent['trial_id']]
            if investigated:
                compared = min(investigated, key=lambda row: row['prediction']['performance']['step_time_ms'])
                before = dict(previous_parent['prediction'], execution_graph=self.load_prediction_graph(previous_parent['prediction']))
                after = dict(compared['prediction'], execution_graph=self.load_prediction_graph(compared['prediction']))
                self._planning_last_feedback = compare_execution_reports(previous_parent['plan']['backend_config'], before,
                    compared['plan']['backend_config'], after, limit=self.topology.num_gpus)
                self.store.add_search_state('PLANNING_FEEDBACK', dict(
                    previous_parent=previous_parent['trial_id'], compared=compared['trial_id'],
                    selected=parent['trial_id'], execution_comparison=self._planning_last_feedback))
            self.store.save_round(round_index, stage='COMPLETE', parent_trial_id=parent['trial_id'],
                planning_action_ids=[row['action_id'] for row in receipts],
                best_predicted_ms=(parent.get('prediction') or {}).get('performance', {}).get('step_time_ms'))
            self.store.charge(f'round:{round_index}', 'agent_rounds', 1)
        return self.finish()
