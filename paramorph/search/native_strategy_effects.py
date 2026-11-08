from collections import Counter

from search.strategy_evidence import execution_structure


def native_strategy_effects(rewriter, request, native_sequence):
    graph = rewriter.compiler.graph(rewriter.candidate(request))
    structure = execution_structure(request, rewriter.topology.devices, rewriter.model)
    counts = Counter(task.task_type.value for task in graph.tasks.values())
    pending = []
    pp, vpp = request['pp'], request['vpp']
    microbatches = structure['num_microbatches']
    for stage in range(pp):
        sequence = native_sequence(stage, pp, vpp, microbatches, request['group'])
        expected = {(kind, virtual, mb) for kind in ('forward_compute', 'backward_compute')
            for virtual in range(vpp) for mb in range(microbatches)}
        if len(sequence) != len(expected) or set(sequence) != expected:
            raise ValueError('原生候选执行顺序没有完整覆盖微批与虚拟分块')
        outstanding = set()
        maximum_chunks = maximum_modules = 0
        for kind, virtual, mb in sequence:
            pair = (virtual * pp + stage, mb)
            if kind == 'forward_compute':
                outstanding.add(pair)
            else:
                outstanding.remove(pair)
            maximum_chunks = max(maximum_chunks, len(outstanding))
            maximum_modules = max(maximum_modules,
                sum(request['decoder_chunks'][region] for region, _ in outstanding))
        if outstanding:
            raise ValueError('原生候选执行顺序存在未完成的反向任务')
        pending.append(dict(pp_rank=stage, maximum_pending_backward_chunks=maximum_chunks,
            maximum_pending_module_microbatch_pairs=maximum_modules))
    return dict(num_microbatches=microbatches,
        physical_stage_layers=structure['physical_stage_layers'],
        process_groups=structure['process_groups'],
        native_task_counts={kind: counts[kind] for kind in
            ('forward_compute', 'backward_compute', 'pp_p2p_send', 'pp_p2p_recv',
             'dp_reduce', 'optimizer_step')},
        pending_backward=pending,
        interpretation='task_counts_describe_granularity_and_message_events_not_compute_work_or_latency',
        storage_scope='logical_pending_backward_objects_not_peak_bytes',
        performance_scope='requires_matched_costs_and_full_iteration_evaluation')
