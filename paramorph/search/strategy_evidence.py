from collections import Counter
from copy import deepcopy

from graph.process_groups import process_groups


STRATEGY_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'micro_batch',
    'group', 'decoder_chunks', 'rank_mapping', 'recompute', 'overlap_p2p',
    'execution_backend', 'dense_sharding_strategy', 'expert_sharding_strategy',
    'fsdp_prefetch', 'fsdp_bucket_elements', 'optimizer_offload_fraction', 'memory_policy')
WORKLOAD_FIELDS = ('name', 'model', 'data', 'sequence', 'global_batch', 'precision',
    'optimizer', 'optimizer_hyperparameters', 'main_grads_dtype', 'main_params_dtype',
    'exp_avg_dtype', 'exp_avg_sq_dtype', 'node_count', 'gpus_per_node')


def execution_structure(request, devices, model=None):
    groups = process_groups(request)
    nodes = {row['physical_rank']: row['node_index'] for row in devices}
    world = request['node_count'] * request['gpus_per_node']
    if set(nodes) != set(range(world)):
        raise ValueError('策略证据需要全部设备的节点身份')
    chunks = request['decoder_chunks']
    if len(chunks) != request['pp'] * request['vpp'] or any(value <= 0 for value in chunks):
        raise ValueError('策略证据需要完整且为正数的模块分区')
    denominator = request['dp'] * request['micro_batch']
    if request['global_batch'] % denominator:
        raise ValueError('微批数量必须满足固定Global Batch')
    intervals, start = [], 0
    for index, size in enumerate(chunks):
        intervals.append([start, start + size, index % request['pp'], index // request['pp']])
        start += size
    structure = dict(parallelism={name: request[name] for name in
            ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'micro_batch')},
        expert_dp=groups['expert_dp_size'],
        num_microbatches=request['global_batch'] // denominator,
        physical_stage_layers=[sum(chunks[stage::request['pp']])
            for stage in range(request['pp'])],
        decoder_chunks=chunks,
        module_interval_columns=['start_inclusive', 'end_exclusive', 'pp_rank', 'vp_rank'],
        module_intervals=intervals,
        process_groups={name: dict(size=len(rows[0]), count=len(rows),
            cross_node_count=sum(len({nodes[rank] for rank in row}) > 1 for row in rows))
            for name, rows in groups['physical'].items()
            if name in ('tp', 'cp', 'dp_cp', 'pp', 'ep', 'etp', 'expert_dp')},
        runtime={name: request.get(name) for name in STRATEGY_FIELDS
            if name not in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp',
                'micro_batch', 'decoder_chunks', 'rank_mapping')},
        scope='native_group_membership_and_complete_layout_without_latency_estimation')
    if model is not None:
        if start != model.num_layers:
            raise ValueError('策略模块分区没有覆盖当前模型')
        if model.num_experts and model.num_experts % request['ep']:
            raise ValueError('专家数量必须能够被EP整除')
        kinds = [Counter() for stage in range(request['pp'])]
        for start, end, stage, vp in intervals:
            kinds[stage].update(model.layer_types[index].value for index in range(start, end))
        structure['module_ownership'] = dict(stage_module_types=[dict(row) for row in kinds],
            experts_per_moe_layer_per_ep_rank=(model.num_experts // request['ep']
                if model.num_experts else 0), expert_tensor_shards=request['etp'],
            expert_modules_per_stage_rank=[row['moe'] * (model.num_experts or 0) // request['ep']
                for row in kinds],
            scope='module_ownership_before_state_sharding_excluding_embedding_and_output')
        sharding = request.get('expert_sharding_strategy', request.get('fsdp_sharding_strategy'))
        shard_factor = (request.get('expert_dp_inner_size', groups['expert_dp_size'])
            if request.get('execution_backend') == 'megatron_fsdp'
                and sharding == 'optim_grads_params' else 1)
        structure['module_ownership']['expert_parameter_partition'] = dict(
            expert_parallel=request['ep'], expert_tensor_parallel=request['etp'],
            parameter_state_shards=shard_factor,
            units_after_tp_ep=[value / request['etp'] for value in
                structure['module_ownership']['expert_modules_per_stage_rank']],
            resident_parameter_units=[value / (request['etp'] * shard_factor) for value in
                structure['module_ownership']['expert_modules_per_stage_rank']],
            unit='one_expert_parameter_set_before_tensor_and_state_sharding',
            scope='logical_parameter_ownership_not_full_peak_memory_or_communication_latency')
    return structure


def collective_summary(graph, world):
    counts, payloads, units = {}, {}, {}
    missing_payloads = Counter()
    for task in graph['tasks'].values():
        metadata = task.get('metadata', {})
        operation = metadata.get('operation', task['operation'])
        if not (operation.startswith('fsdp_') or operation in
                ('pp_p2p_send', 'pp_p2p_recv', 'tp_reduce', 'dp_reduce',
                 'moe_dispatch', 'moe_combine', 'cp_exchange')):
            continue
        key = operation + ':' + metadata.get('state_family', 'model')
        rank = metadata['physical_rank']
        counts.setdefault(key, Counter())[rank] += 1
        full_payloads = metadata.get('communication_full_payloads_bytes')
        payload = sum(full_payloads) if full_payloads is not None else metadata.get('payload_bytes')
        units.setdefault(key, Counter())[rank] += len(full_payloads) if full_payloads is not None else 1
        if payload is None:
            missing_payloads[key] += 1
        else:
            payloads.setdefault(key, Counter())[rank] += payload
    return dict(columns=['operation_and_state', 'min_rank_tasks', 'max_rank_tasks', 'max_rank_units',
            'min_rank_payload_bytes', 'max_rank_payload_bytes', 'tasks_without_payload'],
        rows=[[key, min(counter[rank] for rank in range(world)),
            max(counter.values()), max(units[key].values()),
            None if missing_payloads[key] else min(payloads.get(key, {}).get(rank, 0) for rank in range(world)),
            None if missing_payloads[key] else max(payloads.get(key, {}).values(), default=0), missing_payloads[key]]
            for key, counter in sorted(counts.items())],
        scope='structural_per_rank_task_counts_and_declared_payloads',
        byte_semantics='sum_unit_full_payloads_when_declared_otherwise_task_payload_not_measured_wire_traffic',
        coverage='declared_dag_tasks_only_internal_compute_bundled_communication_requires_its_cost_record')


def compare_strategy_structure(parent, candidate, devices, model=None):
    before, after = execution_structure(parent, devices, model), execution_structure(candidate, devices, model)
    return dict(changed_settings={name: dict(before=parent.get(name), after=candidate.get(name))
            for name in STRATEGY_FIELDS if parent.get(name) != candidate.get(name)},
        parent=before, candidate=after,
        decision_questions=[
            '哪些已测有效的通信、存储或调度条件得到保留？',
            '哪些新增执行代价可能抵消预期收益？',
            '当前完整策略中哪些关联修改能够补偿新增代价？',
            '现有证据支持继续测量、修改策略，还是调查其他候选？'],
        outcome_scope='structural_changes_require_matched_costs_or_training_for_performance_ranking')


def proposed_strategy_structure(parent, values, devices, model=None):
    proposed = deepcopy(parent)
    proposed.update(values)
    # Expert-DP由并行形状推导；显式提交值仍接受原生一致性检查。
    if 'expert_dp' not in values:
        proposed.pop('expert_dp', None)
    return compare_strategy_structure(parent, proposed, devices, model)


def measured_strategy_memory(trials, parent, devices, limit=8, model=None):
    comparable, excluded = [], 0
    for trial in trials:
        request = trial['plan']['backend_config']
        measurement = trial.get('measurement')
        if measurement is None:
            continue
        if any(request.get(name) != parent.get(name) for name in WORKLOAD_FIELDS):
            excluded += 1
            continue
        comparable.append(trial)
    successes = [row for row in comparable if row['measurement']['status'] == 'PASS'
        and row['measurement'].get('performance_eligible', False)]
    chosen = []

    def retain(row):
        if row['trial_id'] not in {item['trial_id'] for item in chosen} and len(chosen) < limit:
            chosen.append(row)

    if successes:
        retain(max(successes, key=lambda row: row['measurement']['tokens_per_second']))
    if comparable:
        retain(comparable[-1])
    if successes:
        retain(min(successes, key=lambda row: row['measurement']['tokens_per_second']))
    failures = [row for row in comparable if row['measurement']['status'] != 'PASS']
    failure_statuses = set()
    for row in reversed(failures):
        status = row['measurement']['status']
        if status not in failure_statuses:
            retain(row)
            failure_statuses.add(status)
    families = set()
    for row in sorted(successes, key=lambda row: row['measurement']['tokens_per_second'], reverse=True):
        request = row['plan']['backend_config']
        family = tuple(request.get(name) for name in
            ('tp', 'pp', 'cp', 'ep', 'etp', 'vpp', 'execution_backend'))
        if family not in families:
            retain(row)
            families.add(family)
    for row in reversed(comparable):
        if row['measurement']['status'] != 'PASS':
            retain(row)
    for row in reversed(comparable):
        retain(row)
    columns = ['trial_id', 'run_id', 'status', 'tokens_per_second', 'step_seconds',
        'peak_allocated_gib', 'parallelism', 'expert_dp', 'num_microbatches',
        'decoder_chunks', 'runtime', 'process_groups', 'module_ownership']
    rows = []
    for trial in chosen:
        result = trial['measurement']
        structure = execution_structure(trial['plan']['backend_config'], devices, model)
        rows.append([trial['trial_id'], result['run_id'], result['status'],
            result.get('tokens_per_second'), result.get('step_seconds'),
            result.get('peak_allocated_gib'), structure['parallelism'], structure['expert_dp'],
            structure['num_microbatches'], structure['decoder_chunks'],
            structure['runtime'], structure['process_groups'], structure.get('module_ownership')])
    return dict(columns=columns, rows=rows, comparable_measured_count=len(comparable),
        different_workload_count=excluded, retained_count=len(rows),
        source='current_search_database_real_training_records',
        selection='best_observed_recent_outcome_distinct_measured_families_and_failures',
        interpretation='joint_configuration_observations_without_single_parameter_causal_attribution',
        invalidation='backend_source_repairs_and_profile_versions_require_rechecking_old_failure_conditions')
