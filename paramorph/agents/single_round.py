from copy import deepcopy
from itertools import groupby
import json
from typing import Literal

from pydantic import Field

from contracts import Contract


PARALLEL_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'micro_batch')
STATE_FIELDS = ('execution_backend', 'optimizer_sharding', 'precision_aware_optimizer',
    'fsdp_sharding_strategy', 'dense_sharding_strategy', 'expert_sharding_strategy',
    'megatron_fsdp_version', 'fsdp_unit', 'fsdp_prefetch', 'fsdp_prefetch_distance',
    'fsdp_reshard_after_forward', 'fsdp_double_buffer', 'fsdp_bucket_elements',
    'dp_inner_size', 'dp_outer_size', 'expert_dp_inner_size', 'expert_dp_outer_size',
    'outer_sharding_strategy', 'expert_outer_sharding_strategy', 'memory_policy')


class DecisionRow(Contract):
    parallel: list[int] = Field(min_length=9, max_length=9)
    layers: list[int] = Field(min_length=1)
    microbatch_group: int = Field(ge=1)
    recompute: Literal['none', 'full', 'selective']
    modules: list[str]
    optimizer_offload: float = Field(ge=0, le=1)
    transfer_overlap: bool
    p2p_overlap: bool
    ranks: list[int]
    bucket: int = Field(ge=1)
    schedule: str
    state: str
    reason: str = Field(min_length=1, max_length=100)
    revision_refs: list[str] = Field(default_factory=list, max_length=2)
    test_target: str = Field(default='', max_length=120)


class ScopeRow(Contract):
    proposal_kind: Literal['conditional_intent']
    parallel: list[int] = Field(min_length=7, max_length=7)
    ranks: list[int]
    adjustable: list[str] = Field(min_length=1, max_length=16)
    allowed_ranges: str
    fixed_values: str
    reason: str = Field(min_length=1, max_length=100)
    revision_refs: list[str] = Field(default_factory=list, max_length=2)
    test_target: str = Field(default='', max_length=120)


class RoundDecision(Contract):
    candidates: list[ScopeRow | DecisionRow] = Field(min_length=1, max_length=3)
    measurement_preference: Literal['training', 'missing_execution_costs'] = 'training'


class ScopeRoundDecision(Contract):
    candidates: list[ScopeRow] = Field(min_length=1, max_length=2)
    measurement_preference: Literal['training', 'missing_execution_costs'] = 'training'


INSTRUCTIONS = (
    '依据本次证据和收益机制，按优先级提出一至两项conditional_intent联合调查。'
    'parallel从输出约束选择已验证的后端:TP/PP/CP/EP/ETP；DP与Expert-DP由两条进程组等式推导。ranks表示物理映射。'
    'adjustable仅列允许调整的局部字段；allowed_ranges的键必须在adjustable中。'
    '全局并行度与后端填写在parallel，Rank映射填写在ranks；这三项不进入局部字段。'
    '未开放变量继承父配置。改变PP或VPP时开放decoder_chunks，检查group、pipeline_schedule与overlap_p2p。'
    '开放recompute时同时开放memory_policy；如需指定选择性模块，将recompute_modules写入memory_policy对象，禁止在范围根部写该键。'
    'none保存激活并省去检查点重算，full重算完整模块，selective重算指定模块。'
    '相对full，selective可能减少重算并增加激活保存；容量需要验证。冻结存储设置需说明本组证据。'
    'fixed_values是扁平JSON对象，如{"vpp":1,"recompute":"full"}；joint_scope_fields和state_fields是字段名清单，不能作为键。'
    'optimizer_transfer_overlap需要正数卸载比例。'
    'FSDP通过选择以megatron_fsdp开头的parallel启用，optimizer_sharding仍为distributed；省略未改变的字段。'
    '稠密模型EP固定为1；DDP目录使用ETP=TP，当前Megatron FSDP适配器使用ETP=1。'
    'VPP=1使用1f1b并关闭overlap_p2p；group在VPP=1时不改变原生调度，禁止仅改变它。'
    '在VPP=1的父配置上调查P2P重叠时，联合提出合法的VPP>1、层分区、group及interleaved_1f1b；'
    '确认固定GBS下的微批数不少于PP，并按当前原生约束选择group。'
    '依据结构、通信、显存、调度及本组历史选择联合参数；改变并行组合后重新判断MBS、重计算和布局。'
    'physical_parallel_alternatives是不同TP/PP组合的通信放置与层归属假设。局部尝试未改善时，'
    '用这些证据判断是否值得改变全局并行；分组数量、层数和部分成本不能直接用于吞吐排序。'
    'schedule_controls基于本组通过训练的父配置和原生事件顺序；分别考虑仅改P2P重叠或仅改分组的调查。'
    '提交单变量调查时，parallel和ranks沿用父配置，只开放该变量并在allowed_ranges提交目标值。'
    '比较实测显存、硬件容量、节点间通信与调度证据，自主判断应验证调度、分片、重计算或全局并行。'
    '切换存储后端或重计算需要说明容量证据或具体计算收益；已失败的配置避免重复提交。'
    '若runtime_failure指向DDP梯度缓冲区初始化，先比较减少常驻参数和梯度的合法变换。'
    'MoE的普通TP与专家ETP/EP分别分组；保持PP、EP和ETP不变时，只增加普通TP不会分摊专家权重或专家梯度。'
    'MoE可改变EP/ETP并验证两类Rank组等式；比较TP/PP及后端支持的参数、梯度和优化器状态分片。'
    '激活重计算影响训练激活峰值，不能缩小初始化的梯度缓冲区；启动容量恢复后再比较重计算代价。'
    'execution_revisions提供父子执行差异，proposal_archive保留本组未测候选，policy_comparisons记录存储策略评价。'
    'prior_episode是本组其他节点的策略假设，需重新测量；引用其evidence_id。'
    'revision_refs从evidence_refs选择，test_target说明验证对象，reason说明收益和新增代价。'
    '已测点可作参照，新训练配置避免重复。未知性能保持假设。'
    'measurement_preference选择training或missing_execution_costs；补测费用计入本轮。'
    '本轮只调用模型一次，程序执行检查、评价及训练。'
)


def parent_row(request):
    policy = request.get('memory_policy') or {}
    state_fields = STATE_FIELDS
    if request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp':
        state_fields = ('execution_backend', 'optimizer_sharding', 'precision_aware_optimizer')
    state = {key: request[key] for key in state_fields if request.get(key) is not None}
    active_policy = {key: policy[key] for key in ('activation_offload_modules', 'activation_offload_fraction',
        'max_inflight_microbatches', 'cpu_memory_budget_bytes', 'prefetch_distance') if policy.get(key)}
    if policy.get('reuse_activation_slots') is False:
        active_policy['reuse_activation_slots'] = False
    if active_policy:
        state['memory_policy'] = active_policy
    return dict(parallel=[request.get(key, 1) for key in PARALLEL_FIELDS],
        layers=request['decoder_chunks'], microbatch_group=request.get('group', 1),
        recompute=request['recompute'], modules=policy.get('recompute_modules', []),
        optimizer_offload=request.get('optimizer_offload_fraction', 0),
        transfer_overlap=request.get('optimizer_transfer_overlap', False),
        p2p_overlap=request.get('overlap_p2p', False), ranks=request['rank_mapping'],
        bucket=request.get('bucket_elements', 40000000), schedule=request.get('pipeline_schedule', '1f1b'),
        state=state)


def reference_view(context):
    identifiers = []
    fields = {'parent_id', 'evidence_id', 'parent', 'candidate', 'candidate_id',
        'source_trial_id', 'layout_id'}

    def register(value):
        if isinstance(value, str) and value not in identifiers:
            identifiers.append(value)

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in fields:
                    register(item)
                if key == 'evidence_refs':
                    for identifier in item:
                        register(identifier)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(context)
    for row in context.get('recent', {}).get('rows', []):
        register(row[0])
    aliases = {f'R{index}': identifier for index, identifier in enumerate(identifiers)}
    names = {identifier: alias for alias, identifier in aliases.items()}

    def replace(value):
        if isinstance(value, str):
            return names.get(value, value)
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    compact = replace(context)
    compact['reference_format'] = 'R编号对应本轮真实证据或策略，由后端恢复原始ID。'
    return compact, aliases


def compact_revision(document):
    comparison = document['comparison']
    operations = sorted(comparison.get('operation_changes', []), key=lambda row: (
        -row['after']['critical_tasks'], -abs(row['delta']['gpu_work_ms']), row['operation']))[:4]
    outcome = document['outcome']
    measured = document.get('measured_after') or {}
    before = document.get('measured_before') or {}
    return dict(evidence_id=document['evidence_id'], parent=document['parent_trial_id'],
        candidate=document['candidate_id'], changed=list(outcome['changed_settings']),
        comparison_status=comparison['status'],
        comparison_reason=comparison.get('reason'), cost_coverage=document.get('cost_coverage'),
        predicted_delta_ms=comparison.get('predicted_step_delta_ms'),
        operation_columns=['operation', 'gpu_work_delta_ms', 'ready_wait_delta_ms',
            'critical_tasks_before', 'critical_tasks_after'],
        operation_rows=[[row['operation'], round(row['delta']['gpu_work_ms'], 3),
            round(row['delta']['ready_wait_sum_ms'], 3), row['before']['critical_tasks'],
            row['after']['critical_tasks']] for row in operations],
        operation_scope='modeled_task_sums_not_additive_iteration_time',
        predicted_memory=[comparison.get(side, {}).get('memory_admission') for side in ('before', 'after')],
        measured_status=measured.get('status'), measured_tps=outcome['measured_tokens_per_second'],
        measured_gain=outcome['measured_gain_vs_parent'],
        measured_peak_gib=outcome['measured_peak_gib'],
        measured_peak_delta_gib=(measured['peak_allocated_gib'] - before['peak_allocated_gib']
            if measured.get('peak_allocated_gib') is not None
            and before.get('peak_allocated_gib') is not None else None),
        attribution='complete_joint_strategy')


def observation(context, request, count):
    history = context.get('measured_strategy_memory') or {}
    columns = history.get('columns', [])
    rows = []
    for values in history.get('rows', [])[:6]:
        record = dict(zip(columns, values, strict=True))
        runtime = record.get('runtime') or {}
        policy = runtime.get('memory_policy') or {}
        rows.append([record['trial_id'], record.get('status'),
            None if record.get('tokens_per_second') is None else round(record['tokens_per_second'], 2),
            None if record.get('peak_allocated_gib') is None else round(record['peak_allocated_gib'], 2),
            [record['parallelism'][name] for name in
                ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'micro_batch')],
            record.get('decoder_chunks'), runtime.get('recompute'),
            policy.get('recompute_modules', []), runtime.get('group'),
            runtime.get('overlap_p2p'), runtime.get('optimizer_offload_fraction')])
    model = context.get('model_structure') or {}
    result = dict(count=min(count, 2), parent_id=context['current_training_plan']['plan_id'],
        parent=parent_row(request), model=model.get('configuration', {}),
        module_runs=[[kind, sum(1 for _ in group)]
            for kind, group in groupby(model.get('module_types', []))],
        domains=context.get('backend_domains', {}), state_fields=(['execution_backend', 'optimizer_sharding',
            'precision_aware_optimizer', 'memory_policy'] if request.get('allowed_execution_backends') == ['megatron_ddp']
            else list(STATE_FIELDS)),
        world_size=len(request['rank_mapping']), native_rules=context.get('parallel_legality', {}),
        supported_backends=(context.get('backend_state_access') or {}).get('allowed_execution_backends', []),
        recomputation=context.get('native_recomputation_options', {}).get('supported_modules', []),
        workload={key: request[key] for key in ('sequence', 'global_batch', 'precision')},
        recent=dict(columns=['id', 'status', 'tokens_per_second', 'memory_gib',
            'TP_PP_CP_DP_EP_ETP_VPP_MBS', 'layers', 'recompute', 'recompute_modules',
            'microbatch_group', 'p2p_overlap', 'optimizer_offload'], rows=rows),
        evidence_refs=context.get('diagnosis', {}).get('evidence_refs', []))
    result['joint_scope_fields'] = ['vpp', 'micro_batch', 'group', 'decoder_chunks', 'recompute',
        'memory_policy', 'pipeline_schedule', 'overlap_p2p', 'execution_backend', 'optimizer_sharding',
        'optimizer_offload_fraction', 'optimizer_transfer_overlap', 'optimizer_pin_cpu_params']
    if 'megatron_fsdp' in result['supported_backends']:
        result['native_fsdp_constraints'] = dict(expert_tensor_parallel_size=1,
            rank_mapping='identity', max_virtual_chunks=None if request.get('fsdp_multi_chunk_supported') else 1,
            wrapping_units=request.get('fsdp_supported_units', ['transformer_layer']),
            dense_shard_group='dp*cp', expert_shard_group='expert_dp', outer_dp_size=1,
            dense_strategies=request.get('fsdp_supported_sharding_strategies', []),
            expert_strategies=request.get('fsdp_supported_expert_sharding_strategies', []))
    if request.get('allowed_optimizer_offload_fractions'):
        result['native_optimizer_offload'] = dict(
            fractions=request['allowed_optimizer_offload_fractions'], backend='megatron_ddp',
            optimizer_sharding='distributed', precision_aware_optimizer=True,
            effect='moves_optimizer_states_and_master_shards_to_cpu_not_the_ddp_gradient_buffer',
            pin_cpu_params_options=[True, False],
            pin_cpu_params_false='skips_the_second_pin_of_the_FP32_host_copy_and_keeps_the_initial_copy',
            overlap_initialization='this_backend_builds_one_cpu_adamw_instance_per_offloaded_parameter_with_overlap',
            no_overlap_initialization='this_backend_builds_one_cpu_adamw_instance_for_all_cpu_parameter_groups')
    rejected = []
    outcomes = []
    for previous in context.get('recent_search_rounds', []):
        feedback = previous.get('feedback') or {}
        if previous.get('stop_reason') and feedback.get('native_validation') is None:
            outcomes.append(dict(round=previous['round_index'], outcome=previous['stop_reason']))
        if feedback.get('native_validation') is not None:
            validation = feedback['native_validation']
            reasons = list(validation.get('conflicts', []))
            partition_capacity_rejections = []
            if isinstance(validation.get('reason'), str):
                reasons.append(validation['reason'][:240])
            for item in validation.get('legal_counts', {}).values():
                conflict = item.get('conflict') or {}
                if isinstance(conflict, str):
                    reasons.append(conflict)
                    continue
                partition_capacity_rejections.extend(conflict.get('capacity_rejections', []))
                reasons.extend(conflict.get('conflicts', []))
                for rejection in conflict.get('layout_rejections', []):
                    reasons.extend(rejection.get('reasons', []))
                for rejection in conflict.get('policy_rejections', []):
                    reason = rejection['reason']
                    if isinstance(reason, str):
                        reasons.append(reason)
                    else:
                        reasons.extend(reason.get('conflicts', []))
                reason = conflict.get('reason')
                if isinstance(reason, str):
                    reasons.append(reason)
                elif isinstance(reason, list):
                    reasons.extend(reason)
            capacities = [dict(admission=item['admission'],
                capacity_bytes=item['capacity_per_gpu_bytes'],
                predicted_peak_bytes=max(item['peak_per_rank_bytes'].values()))
                for items in validation.get('capacity_checks', {}).values() for item in items]
            rejected.append(dict(round=previous['round_index'], status=validation['status'],
                reasons=sorted(set(reasons)), capacities=capacities,
                partition_capacity_rejections=partition_capacity_rejections[:2],
                evidence_refs=feedback.get('evidence_refs', [])))
    if rejected:
        result['recent_native_rejections'] = rejected
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'],
            *(identifier for row in rejected for identifier in row['evidence_refs'])]))
    failure = context.get('runtime_failure')
    if failure is not None:
        result['runtime_failure'] = {key: failure[key] for key in
            ('status', 'observed', 'execution_deadline_reached', 'evidence_refs')
            if key in failure}
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'], *failure['evidence_refs']]))
    if context.get('recent_candidate_runtime_failures'):
        result['recent_candidate_runtime_failures'] = context['recent_candidate_runtime_failures']
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'],
            *(identifier for row in result['recent_candidate_runtime_failures']
                for identifier in row['evidence_refs'])]))
    if outcomes:
        result['recent_search_outcomes'] = outcomes
    if context.get('recent_profile_failures'):
        result['recent_profile_failures'] = context['recent_profile_failures']
    if context.get('recent_runtime_timeouts'):
        result['recent_runtime_timeouts'] = context['recent_runtime_timeouts']
    if context.get('optimizer_init_witness'):
        result['optimizer_init_witness'] = context['optimizer_init_witness']
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'],
            result['optimizer_init_witness']['evidence_id']]))
    if context.get('prior_episode'):
        prior = context['prior_episode']
        result['prior_episode'] = {key: prior[key] for key in
            ('source_node', 'target_node', 'variant', 'scope', 'evidence_id')}
        result['prior_episode']['strategies'] = [dict(strategy=row['strategy'], state=row['state'],
            source_trial_id=row['source_trial_id'])
            for row in prior['strategies']]
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'], prior['evidence_id']]))
    if context.get('proposal_archive'):
        result['proposal_archive'] = [{key: row[key] for key in
            ('candidate_id', 'strategy', 'predicted_step_ms', 'measured')}
            for row in context['proposal_archive']]
    if context.get('policy_comparisons'):
        result['policy_comparisons'] = context['policy_comparisons']
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'],
            *(row['evidence_id'] for row in result['policy_comparisons'])]))
    if context.get('execution_revisions'):
        result['execution_revisions'] = [compact_revision(row) for row in context['execution_revisions'][:2]]
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'],
            *(row['evidence_id'] for row in result['execution_revisions'])]))
    if context.get('physical_parallel_alternatives'):
        result['physical_parallel_alternatives'] = context['physical_parallel_alternatives']
    if context.get('schedule_controls'):
        result['schedule_controls'] = context['schedule_controls']
        result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'],
            result['schedule_controls']['evidence_id']]))
    if context.get('cross_shape_investigations'):
        result['cross_shape_investigations'] = [dict(values=row['values'],
            evidence_id=row['graph_evidence_id'], target_region_placements=row['target_region_placements'],
            layout_witness=row['layout_witness'], interpretation=row['interpretation'])
            for row in context['cross_shape_investigations'][:2]]
    result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'], *(row['evidence_id']
        for name in ('physical_parallel_alternatives', 'cross_shape_investigations')
        for row in result.get(name, []))]))
    if context.get('execution_analysis') is not None:
        analysis = context['execution_analysis']
        if analysis.get('evidence_id'):
            result['evidence_refs'] = list(dict.fromkeys([*result['evidence_refs'], analysis['evidence_id']]))
        result['execution'] = {key: analysis[key] for key in
            ('evidence_id', 'status', 'measured_step_ms', 'predicted_step_ms',
             'known_cost_tasks', 'unresolved_task_count', 'critical_tasks', 'bottlenecks') if key in analysis}
        if analysis.get('unresolved_task_count'):
            result['execution']['missing_groups'] = analysis.get('unresolved_groups', [])[:4]
            result['execution']['missing_measurements'] = analysis.get('missing_measurements', [])[:2]
            result['execution']['decision_status'] = analysis['decision_status']
        result['execution']['critical_tasks'] = [dict(task_id=row['task_id'], rank=row['rank'],
            duration_ms=round(row['duration_ms'], 3), region_id=row.get('region_id'),
            resources=[resource for resource in row.get('resources', []) if not resource.startswith('pcie:')])
            for row in result['execution'].get('critical_tasks', [])[:3]]
        for key in ('critical_tasks', 'bottlenecks'):
            if isinstance(result['execution'].get(key), list) and len(result['execution'][key]) > 3:
                result['execution'][key + '_total_count'] = len(result['execution'][key])
                result['execution'][key] = result['execution'][key][:3]
                result['execution']['detail_scope'] = 'first_three_records_from_registered_execution_analysis'
        result['execution']['bottlenecks'] = [dict(
            region_id=row['region_id'], operation=row['operation'],
            target_task_count=row['target_task_count'],
            controlling_parameters=[item['parameter'] for item in row['controlling_parameters']],
            epsilon=row['epsilon'], local_cost_reduction_ms=round(row['local_cost_reduction_ms'], 3),
            full_iteration_impact_ms=round(row['full_iteration_impact_ms'], 3),
            evidence_type=row['evidence_type'],
            wait_causes=[{key: round(value[key], 3) for key in
                ('dependency_ready_ms', 'resource_wait_ms', 'resource_blocker_count')}
                for value in row['target_wait_causes'].values()])
            for row in result['execution'].get('bottlenecks', [])]
        parameter_sets = []
        for row in result['execution']['bottlenecks']:
            parameters = row.pop('controlling_parameters')
            if parameters not in parameter_sets:
                parameter_sets.append(parameters)
            row['parameter_set'] = parameter_sets.index(parameters)
        if parameter_sets:
            result['execution']['parameter_sets'] = parameter_sets
        native = context.get('parent_native_execution_effects') or {}
        result['schedule'] = {key: native[key] for key in ('num_microbatches', 'physical_stage_layers') if key in native}
    return result


def complete_values(row, parent):
    state = json.loads(row.state)
    if not isinstance(state, dict) or set(state) - set(STATE_FIELDS):
        raise ValueError('状态设置必须是仅包含已声明字段的JSON对象')
    values = {key: deepcopy(parent[key]) for key in STATE_FIELDS if parent.get(key) is not None}
    values.update(state)
    values.update(dict(zip(PARALLEL_FIELDS, row.parallel, strict=True)))
    if values.get('execution_backend', 'megatron_ddp') == 'megatron_ddp':
        values.update(dp_inner_size=values['dp'] * values['cp'], dp_outer_size=1,
            expert_dp_inner_size=values['expert_dp'], expert_dp_outer_size=1)
    policy = deepcopy(state.get('memory_policy', parent.get('memory_policy')) or {})
    policy.update(optimizer_offload_fraction=row.optimizer_offload,
        recompute_granularity=None if row.recompute == 'none' else row.recompute,
        recompute_modules=row.modules)
    values.update(decoder_chunks=row.layers, group=row.microbatch_group, recompute=row.recompute,
        optimizer_offload_fraction=row.optimizer_offload, optimizer_transfer_overlap=row.transfer_overlap,
        overlap_p2p=row.p2p_overlap, rank_mapping=row.ranks, bucket_elements=row.bucket,
        pipeline_schedule=row.schedule, memory_policy=policy)
    return values
