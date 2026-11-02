from dataclasses import dataclass

from megatron.core.frontier.pipeline.readiness import ExecutionNode, ReadinessDAG, ResourceSignature


@dataclass(frozen=True)
class PipelinePrediction:
    step_us: float
    peak_activation_bytes: tuple[int, ...]
    critical_path: tuple[str, ...]
    intervals: dict
    dag: ReadinessDAG


def simulate_native_program(program, microbatches, schedule_table, profiles, pp_us):
    pp, vp = program.pp_size, program.vp_size
    if microbatches < 1 or pp_us < 0 or len(schedule_table) != microbatches * vp:
        raise ValueError('schedule and measured communication cost must match the pipeline program')
    if set(map(tuple, schedule_table)) != {(mb, chunk) for mb in range(microbatches) for chunk in range(vp)}:
        raise ValueError('native schedule table does not cover every microbatch exactly once')
    chunks = sorted(program.chunks, key=lambda chunk: (chunk.vp_stage, chunk.pp_rank))
    nodes = []
    starts, finishes = {}, {}
    allocations = {}
    for stage in range(pp):
        forward = [(mb, chunk) for mb, chunk in schedule_table]
        backward = [(mb, vp - chunk - 1) for mb, chunk in schedule_table]
        if vp == 1:
            warmup = min(pp - stage - 1, microbatches)
        else:
            first_group = next((index for index, (_, chunk) in enumerate(schedule_table) if chunk != 0), len(schedule_table))
            warmup = min((pp - stage - 1) * 2 + (vp - 1) * first_group, len(forward))
        sequence = [('f', *item) for item in forward[:warmup]]
        for index in range(len(forward) - warmup):
            sequence.extend((('f', *forward[index + warmup]), ('b', *backward[index])))
        sequence.extend(('b', *item) for item in backward[len(forward) - warmup:])
        previous = None
        for direction, mb, chunk in sequence:
            index = chunk * pp + stage
            regions = chunks[index].region_ids
            if direction == 'b':
                regions = tuple(reversed(regions))
            first = f'{direction}:m{mb}:c{index}:{regions[0]}'
            for position, region in enumerate(regions):
                identity = f'{direction}:m{mb}:c{index}:{region}'
                cost = profiles[region]
                duration = cost['forward_us' if direction == 'f' else 'backward_us']
                dependencies = [] if previous is None else [previous]
                if position == 0:
                    if direction == 'f' and index > 0:
                        dependencies.append(f'pf:m{mb}:c{index - 1}')
                    elif direction == 'b':
                        dependencies.append(f'f:m{mb}:c{index}:{chunks[index].region_ids[-1]}')
                        if index + 1 < len(chunks):
                            dependencies.append(f'pb:m{mb}:c{index + 1}')
                resource = ResourceSignature(('GPU_COMPUTE',), f'pp:{stage}')
                produces = (f'activation:m{mb}:{region}',) if direction == 'f' else ()
                consumes = (f'activation:m{mb}:{region}',) if direction == 'b' else ()
                allocation = cost['activation_bytes'] if direction == 'f' else 0
                nodes.append(ExecutionNode(identity, 'REGION_FWD' if direction == 'f' else 'REGION_BWD',
                                           tuple(dict.fromkeys(dependencies)), duration, resource,
                                           allocation, produces, consumes))
                if direction == 'f':
                    allocations[produces[0]] = stage
                previous = identity
            starts[(direction, mb, index)] = first
            finishes[(direction, mb, index)] = previous
    for index in range(len(chunks)):
        for mb in range(microbatches):
            for direction, neighbor in (('f', index + 1), ('b', index - 1)):
                if 0 <= neighbor < len(chunks):
                    stage, destination = index % pp, neighbor % pp
                    link = f'pp-link:{min(stage, destination)}:{max(stage, destination)}'
                    nodes.append(ExecutionNode(f'p{direction}:m{mb}:c{index}', 'PP_SEND',
                                               (finishes[(direction, mb, index)],), pp_us,
                                               ResourceSignature(('DMA', 'IB'), link)))
    dag = ReadinessDAG(tuple(nodes))
    intervals, availability, parents = {}, {}, {}
    for identity in dag.order:
        node = dag.nodes[identity]
        blockers = [(intervals[parent][1], parent) for parent in node.dependencies]
        for resource in node.resource.resources:
            key = (node.resource.device, resource)
            if key in availability:
                blockers.append(availability[key])
        start, parent = max(blockers, default=(0.0, None))
        intervals[identity] = (start, start + node.duration_us)
        parents[identity] = parent
        for resource in node.resource.resources:
            availability[(node.resource.device, resource)] = (intervals[identity][1], identity)
    terminal = max(intervals, key=lambda identity: intervals[identity][1])
    critical = []
    cursor = terminal
    while cursor is not None:
        critical.append(cursor)
        cursor = parents[cursor]
    timelines = [[] for _ in range(pp)]
    for node in nodes:
        if node.alloc_bytes:
            value = node.produces[0]
            consumers = [item.node_id for item in nodes if value in item.consumes]
            end = max(intervals[identity][1] for identity in consumers)
            stage = allocations[value]
            timelines[stage].extend(((intervals[node.node_id][0], node.alloc_bytes), (end, -node.alloc_bytes)))
    peaks = []
    for timeline in timelines:
        current = peak = 0
        for _, delta in sorted(timeline, key=lambda item: (item[0], -item[1])):
            current += delta
            peak = max(peak, current)
        assert current == 0
        peaks.append(peak)
    return PipelinePrediction(intervals[terminal][1], tuple(peaks), tuple(reversed(critical)), intervals, dag)
