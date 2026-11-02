from collections import Counter
import hashlib
import json

from graph.process_groups import process_groups


VIEW_REVISION = 'native_execution_view_v2'


def execution_view(request, graph, devices, native_sequence, mode, region_ids=()):
    if mode not in ('execution_view', 'flat_records'):
        raise ValueError('执行信息格式没有登记')
    pp, vpp = request['pp'], request['vpp']
    lanes = request['tp'] * request['cp'] * request['dp']
    microbatches = request['global_batch'] // (request['dp'] * request['micro_batch'])
    regions = {row['region_id']: row for row in graph['regions']}
    selected = set(region_ids) or set(regions)
    if not selected <= set(regions):
        raise ValueError('执行视图引用未知模块区间')
    # 同一物理阶段的全部虚拟区间共享原生执行顺序。
    stages = {regions[index]['pp_rank'] for index in selected}
    selected |= {index for index, row in regions.items() if row['pp_rank'] in stages}
    nodes = {row['physical_rank']: row['node_index'] for row in devices}
    mapping = request['rank_mapping']
    tasks = graph['tasks']
    sequences, intervals, ownership, boundaries = [], [], [], []
    relations = []
    for stage in sorted(stages):
        sequence = native_sequence(stage, pp, vpp, microbatches, request['group'])
        expected = {(kind, virtual, mb) for kind in ('forward_compute', 'backward_compute')
            for virtual in range(vpp) for mb in range(microbatches)}
        if len(sequence) != len(expected) or set(sequence) != expected:
            raise ValueError('原生执行序列缺少事件或包含重复事件')
        rows, open_intervals = [], {}
        outstanding, largest = set(), []
        for order, (kind, virtual, mb) in enumerate(sequence):
            region = virtual * pp + stage
            prefix = 'F' if kind == 'forward_compute' else 'B'
            task_id = f'{prefix}:0:{mb}:{region}'
            if task_id not in tasks:
                raise ValueError('原生执行序列与后端任务身份不同')
            rows.append([prefix, virtual, mb])
            pair = (region, mb)
            if prefix == 'F':
                open_intervals[pair] = order
                outstanding.add(pair)
            else:
                start = open_intervals.pop(pair)
                if mb in (0, microbatches - 1):
                    intervals.append([stage, region, mb, start, order])
                outstanding.remove(pair)
            if len(outstanding) > len(largest):
                largest = sorted(outstanding)
        if outstanding or open_intervals:
            raise ValueError('原生执行序列没有完成全部反向事件')
        stage_regions = sorted(index for index in selected if regions[index]['pp_rank'] == stage)
        physical = [mapping[stage * lanes + lane] for lane in range(lanes)]
        sequences.append([stage, physical, rows])
        for index in stage_regions:
            row = regions[index]
            ownership.append([index, stage, row['vp_rank'], row['layer_start'], row['layer_end']])
        relations.append(dict(pp_rank=stage, physical_ranks=physical,
            co_resident_regions=stage_regions,
            maximum_pending_backward_chunks=len(largest),
            witness_pending_backward_pairs=[list(pair) for pair in largest],
            forward_event_count=len(sequence) // 2,
            backward_event_count=len(sequence) // 2))
    # 保留局部范围与外部任务相接的真实依赖，避免切断上游等待。
    for key, task in tasks.items():
        for predecessor in task['dependencies']:
            source = tasks[predecessor]
            if task['region_id'] in selected or source['region_id'] in selected:
                if task['microbatch'] not in (None, 0) or source['microbatch'] not in (None, 0):
                    continue
                if task['rank'] % lanes or source['rank'] % lanes:
                    continue
                boundaries.append([predecessor, key])
    groups = process_groups(request)['physical']
    group_rows = [[name, members, len({nodes[rank] for rank in members}) > 1]
        for name, rows in sorted(groups.items()) for members in rows
        if name in ('tp', 'cp', 'pp', 'ep', 'etp', 'dp_cp', 'expert_dp')]
    counts = Counter(task['operation'] for task in tasks.values())
    costs = Counter(task['cost_status'] for task in tasks.values())
    facts = dict(
        workload=dict(global_batch=request['global_batch'], micro_batch=request['micro_batch'],
            num_microbatches=microbatches, sequence=request['sequence'], precision=request['precision']),
        module_columns=['region_id', 'pp_rank', 'vp_rank', 'layer_start', 'layer_end'],
        module_ownership=ownership,
        schedule_columns=['pp_rank', 'physical_ranks', 'ordered_phase_vp_microbatch'],
        native_compute_sequences=sequences,
        pending_backward_columns=['pp_rank', 'region_id', 'microbatch', 'forward_order', 'backward_order'],
        pending_backward_intervals=intervals,
        interval_scope='first_and_last_microbatch_examples_full_intervals_recoverable_from_native_sequences',
        dependency_templates=sorted(boundaries),
        dependency_template_scope='lane0_microbatch0_direct_dependencies_with_external_endpoints',
        group_columns=['kind', 'physical_members', 'cross_node'], physical_groups=group_rows,
        physical_devices=[[rank, nodes[rank]] for rank in sorted(nodes)],
        native_configuration_constraints=[
            'world=TP*CP*PP*DP', 'world=ETP*EP*ExpertDP*PP',
            'GBS=DP*MBS*microbatches', 'region_count=PP*VPP',
            'overlap_p2p=true_requires_VPP>1', 'VPP>1_requires_PP>1',
            'VPP>1_requires_PP<=group<=microbatches',
            'current_training_entry_requires_microbatches%group=0',
            'TP_CP_EP_ETP_are_global_groups_in_this_backend'],
        operation_counts=dict(sorted(counts.items())), cost_status_counts=dict(sorted(costs.items())),
        opaque_operations=graph['opaque_operations'],
        runtime={key: request.get(key) for key in ('group', 'overlap_p2p', 'pipeline_schedule',
            'recompute', 'memory_policy', 'execution_backend', 'optimizer_offload_fraction')})
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output = dict(view_derivation_version=VIEW_REVISION, presentation=mode,
        raw_fact_inventory_hash=digest, facts=facts,
        scope='stage_microbatch_rank_execution_with_opaque_tp_cp_ep_internals',
        activation_scope='completed_forward_to_backward_order_intervals_without_tensor_byte_prediction',
        task_identity='F_or_B:lane:microbatch:region_id',
        source_graph_sha256=graph['graph_sha256'],
        full_evaluation='unchanged_native_full_iteration_evaluator')
    if mode == 'execution_view':
        output['relations'] = dict(physical_stage_schedules=relations,
            decoder_layer_microbatch_invocations_per_pipeline_replica=sum(request['decoder_chunks']) * microbatches,
            conserved_decoder_layer_samples=sum(request['decoder_chunks']) * request['global_batch'],
            comparisons=['compare_microbatch_granularity_and_native_order_with_communication_multiplicity',
                'check_co_resident_regions_before_expanding_module_boundaries',
                'preserve_external_dependencies_and_unknown_costs_in_local_investigation'])
    return output
