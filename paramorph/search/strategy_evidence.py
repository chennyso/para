from collections import Counter
from copy import deepcopy
from dataclasses import asdict

from core_data_models import MemoryPolicy
from graph.process_groups import process_groups
from predictor.profile_database import normalized_execution


STRATEGY_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'micro_batch',
    'group', 'decoder_chunks', 'rank_mapping', 'recompute', 'overlap_p2p',
    'execution_backend', 'dense_sharding_strategy', 'expert_sharding_strategy',
    'fsdp_prefetch', 'fsdp_bucket_elements', 'optimizer_offload_fraction', 'memory_policy')
WORKLOAD_FIELDS = ('name', 'model', 'data', 'sequence', 'global_batch', 'precision',
    'optimizer', 'optimizer_hyperparameters', 'main_grads_dtype', 'main_params_dtype',
    'exp_avg_dtype', 'exp_avg_sq_dtype', 'node_count', 'gpus_per_node')


def strategy_decision_values(request):
    effective = normalized_execution(request)
    policy = asdict(MemoryPolicy(**(request.get('memory_policy') or {})))
    return dict(
        parallelism={key: effective[key] for key in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp',
            'expert_dp', 'sequence_parallel')},
        micro_batch=request['micro_batch'],
        recomputation=dict(mode=request['recompute'],
            granularity=policy['recompute_granularity'], modules=sorted(policy['recompute_modules'])),
        module_partition=request['decoder_chunks'],
        schedule={key: effective[key] for key in ('vpp', 'group', 'pipeline_schedule', 'overlap_p2p')},
        communication=dict(bucket_elements=effective['bucket_elements'], rank_mapping=effective['rank_mapping']),
        state_sharding={key: effective[key] for key in ('execution_backend', 'optimizer_sharding',
            'dense_sharding_strategy', 'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size',
            'expert_dp_inner_size', 'expert_dp_outer_size', 'fsdp_prefetch', 'fsdp_bucket_elements')},
        storage_policy={key: value for key, value in policy.items()
            if key not in ('recompute_granularity', 'recompute_modules')}
            | {key: effective[key] for key in ('optimizer_offload_fraction', 'optimizer_transfer_overlap')})


def strategy_decision_evidence(parent, trials, domains, *, limit=8):
    request = parent['plan']['backend_config']
    current = strategy_decision_values(request)
    identity = normalized_execution(request)
    variable_fields = {'micro_batch', 'recompute', 'memory_policy'}
    for group in ('parallelism', 'schedule', 'communication', 'state_sharding', 'storage_policy'):
        variable_fields.update(current[group])
    records = []
    for trial in trials:
        measured = trial.get('measurement') or {}
        other = trial['plan']['backend_config']
        if measured.get('status') != 'PASS' or measured.get('performance_eligible') is not True:
            continue
        if trial.get('measurement_reuse'):
            continue
        if any(other.get(key) != request.get(key) for key in WORKLOAD_FIELDS):
            continue
        effective = normalized_execution(other)
        if any(effective.get(key) != value for key, value in identity.items() if key not in variable_fields):
            continue
        records.append(dict(trial_id=trial['trial_id'], run_id=measured['run_id'],
            decisions=strategy_decision_values(other),
            tokens_per_second=measured['tokens_per_second'], step_seconds=measured['step_seconds'],
            peak_allocated_gib=measured.get('peak_allocated_gib')))
    coverage = []
    for name, value in current.items():
        matched = [row for row in records if all(row['decisions'][other] == current[other]
            for other in current if other != name)]
        observations = [dict(trial_id=row['trial_id'], run_id=row['run_id'], value=row['decisions'][name],
            tokens_per_second=row['tokens_per_second'], peak_allocated_gib=row['peak_allocated_gib'])
            for row in matched]
        coverage.append(dict(decision=name, incumbent_value=value,
            measured_at_other_incumbent_settings=observations[-limit:],
            distinct_measured_values=len({str(row['value']) for row in observations}),
            status='ALTERNATIVES_MEASURED' if any(row['value'] != value for row in observations)
                else 'INCUMBENT_ONLY' if observations else 'NO_MATCHED_MEASUREMENT'))
    contrasts = []
    for left, right in zip(records, records[1:]):
        changed = [name for name in current if left['decisions'][name] != right['decisions'][name]]
        contrasts.append(dict(before_trial_id=left['trial_id'], after_trial_id=right['trial_id'],
            before_run_id=left['run_id'], after_run_id=right['run_id'], changed_decisions=changed,
            changes={name: dict(before=left['decisions'][name], after=right['decisions'][name]) for name in changed},
            observed_throughput_change=right['tokens_per_second'] / left['tokens_per_second'] - 1,
            attribution='single_decision_observation' if len(changed) == 1 else 'joint_configuration_observation',
            replication='single_record_pair_without_statistical_significance_test'))
    batches = [value for value in domains['micro_batch']
        if request['global_batch'] % (request['dp'] * value) == 0]
    return dict(source='current_search_successful_training_records_only',
        incumbent_trial_id=parent['trial_id'], effective_decisions=current,
        coverage=coverage, recent_measured_contrasts=contrasts[-limit:],
        micro_batch_schedule=[dict(micro_batch=value,
            num_microbatches=request['global_batch'] // (request['dp'] * value)) for value in batches],
        legal_scope='joint_candidates_require_native_validation_of_all_changed_settings',
        evidence_scope='successful_configuration_does_not_establish_optimality_of_inherited_settings',
        selection_owner='Agent_selects_joint_changes_using_measured_tradeoffs_and_unresolved_alternatives',
        prediction_scope='parent_task_sensitivity_does_not_rank_unmeasured_joint_strategies')


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
        vocabulary = model.finalize_grad_semantics.get('vocab_size')
        structure['boundary_modules'] = dict(
            embedding=dict(pp_rank=0, vocabulary_rows=vocabulary, hidden_size=model.hidden_size,
                tensor_partition_axis='vocabulary', tensor_shards=request['tp']),
            output=dict(pp_rank=request['pp'] - 1, vocabulary_rows=vocabulary,
                hidden_size=model.hidden_size, tensor_partition_axis='vocabulary',
                tensor_shards=request['tp']),
            loss=dict(pp_rank=request['pp'] - 1),
            tied_weights=model.finalize_grad_semantics.get('share_embeddings_and_output_weights', False),
            counted_in_decoder_chunks=False,
            decision_dependencies=dict(
                tp='改变词汇矩阵的张量分片及对应计算通信',
                pp='改变首末物理阶段及其共驻Transformer模块',
                decoder_chunks='改变共驻Transformer模块，输出层继续位于末端阶段'),
            source='actual_model_configuration_and_native_megatron_boundary_module_placement',
            scope='structural_ownership_without_measured_boundary_latency_or_padded_parameter_bytes')
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


def complete_strategy_comparisons(parent, candidates, devices, model=None):
    before_request = parent['plan']['backend_config']
    before = execution_structure(before_request, devices, model)
    parent_prediction = parent.get('prediction') or {}
    parent_time = parent_prediction.get('performance', {}).get('step_time_ms')
    if parent_prediction.get('prediction_status') == 'NEEDS_PROFILE':
        parent_time = None
    parent_execution = normalized_execution(before_request)
    parent_keys = {item.get('cost_key') for item in
        parent_prediction.get('evidence', {}).get('profiling_requests', []) if item.get('cost_key')}
    rows = []
    for trial in candidates:
        request = trial['plan']['backend_config']
        if any(request.get(name) != before_request.get(name) for name in WORKLOAD_FIELDS):
            raise ValueError('完整策略比较需要相同训练负载和数值设置')
        comparison = compare_strategy_structure(before_request, request, devices, model)
        after = comparison['candidate']
        prediction = trial.get('prediction') or {}
        evidence = prediction.get('evidence', {})
        missing = evidence.get('missing_profiles', [])
        keys = {item.get('cost_key') for item in evidence.get('profiling_requests', [])
            if item.get('cost_key')}
        complete_time = prediction.get('performance', {}).get('step_time_ms')
        if prediction.get('prediction_status') == 'NEEDS_PROFILE':
            complete_time = None
        measurement = trial.get('measurement') or {}
        measured = measurement.get('status') == 'PASS' and measurement.get('performance_eligible') is True
        candidate_execution = normalized_execution(request)
        changes = {name: dict(before=parent_execution.get(name), after=value)
            for name, value in candidate_execution.items() if parent_execution.get(name) != value}
        if before_request['decoder_chunks'] != request['decoder_chunks']:
            changes['decoder_chunks'] = dict(before=before_request['decoder_chunks'], after=request['decoder_chunks'])
        rows.append(dict(trial_id=trial['trial_id'],
            changed_settings=changes,
            measured_tokens_per_second=measurement.get('tokens_per_second') if measured else None,
            predicted_step_ms=complete_time,
            predicted_parent_step_ms=parent_time,
            prediction_status=prediction.get('prediction_status', 'UNKNOWN'),
            memory_admission=prediction.get('memory', {}).get('admission', 'UNKNOWN'),
            missing_profile_count=len(missing), missing_profiles=missing[:3],
            execution_effects=dict(
                parent_num_microbatches=before['num_microbatches'],
                candidate_num_microbatches=after['num_microbatches'],
                parent_logical_tokens_per_cp_rank=(before_request['micro_batch']
                    * before_request['sequence'] / before_request['cp']),
                candidate_logical_tokens_per_cp_rank=(request['micro_batch']
                    * request['sequence'] / request['cp']),
                logical_tokens_scope='microbatch_times_sequence_divided_by_CP_before_sequence_parallelism',
                parent_physical_stage_layers=before['physical_stage_layers'],
                candidate_physical_stage_layers=after['physical_stage_layers'],
                changed_process_groups={key: dict(before=before['process_groups'].get(key), after=value)
                    for key, value in after['process_groups'].items()
                    if value != before['process_groups'].get(key)},
                shared_missing_cost_count=len(keys & parent_keys),
                shared_missing_cost_keys=sorted(keys & parent_keys)[:2],
                candidate_only_missing_cost_count=len(keys - parent_keys),
                candidate_only_missing_cost_keys=sorted(keys - parent_keys)[:2],
                complete_cost_access=dict(tool='evaluate_candidate', candidate_ids=[trial['trial_id']]),
                cost_key_scope='requested_missing_components_only',
                scope='structural_effects_with_complete_prediction_when_available')))
    return dict(objective='maximize_measured_throughput_under_fixed_training_contract_and_budget',
        decision_rule='compare_complete_candidate_gains_added_costs_and_unknowns',
        parent_bottleneck_scope='conditional_observation_of_parent_execution',
        candidates=rows)


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
