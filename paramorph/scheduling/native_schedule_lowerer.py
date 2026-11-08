from collections import defaultdict

from core_data_models import Task, TaskType
from scheduling.native_microbatches import native_schedule_table


def native_stage_sequence(stage, pp, vpp, microbatches, group):
    if not 0 <= stage < pp:
        raise ValueError('流水阶段编号超出范围')
    table = native_schedule_table(microbatches, pp, vpp, group)
    backward = [(mb, vpp - vp - 1) for mb, vp in table]
    if vpp == 1:
        warmup = min(pp - stage - 1, microbatches)
    else:
        first_group = next((i for i, (_, vp) in enumerate(table) if vp != 0), len(table))
        warmup = min((pp - stage - 1) * 2 + (vpp - 1) * first_group, len(table))
    forward = [('forward_compute', vp, mb) for mb, vp in table]
    backward = [('backward_compute', vp, mb) for mb, vp in backward]
    steady = [event for pair in zip(forward[warmup:], backward[:len(table) - warmup], strict=True)
        for event in pair]
    return forward[:warmup] + steady + backward[len(table) - warmup:]


def lower_native_schedule(tasks, config, overlap, add_dependency):
    lanes = config.world_size // config.pp
    compute = {}
    sends = defaultdict(list)
    for key, task in tuple(tasks.items()):
        if task.task_type in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE):
            identity = (task.rank, task.task_type.value, task.region_id // config.pp, task.microbatch_id)
            if identity in compute:
                raise ValueError('原生执行图出现重复计算事件')
            compute[identity] = key
        elif task.task_type == TaskType.PP_P2P_SEND:
            producer = task.metadata['producer_task']
            launch = 'launch:' + key
            if launch not in tasks:
                tasks[launch] = Task(launch, task.rank, TaskType.CUDA_SYNC,
                    microbatch_id=task.microbatch_id, region_id=task.region_id,
                    metadata={name: task.metadata[name] for name in ('physical_rank', 'vp_stage')
                        if name in task.metadata} | dict(operation='p2p_launch', transfer_event=key,
                        cost_scope='launch_latency_included_in_transfer_cost'))
                add_dependency(producer, launch)
                add_dependency(launch, key)
            task.metadata['launch_event'] = launch
            sends[producer].append(key)
            task.metadata['native_launch_policy'] = 'nonblocking' if overlap else 'blocking'
        elif task.task_type == TaskType.PP_P2P_RECV:
            task.metadata['native_wait_policy'] = 'consumer_waits_for_receive_completion'
    if len(compute) != 2 * config.world_size * config.vpp * config.num_microbatches:
        raise ValueError('原生执行图未覆盖全部逐 Rank 微批事件')
    for rank in range(config.world_size):
        sequence = native_stage_sequence(rank // lanes, config.pp, config.vpp,
            config.num_microbatches, config.microbatch_group)
        ordered = [compute[rank, kind, vp, mb] for kind, vp, mb in sequence]
        for position, key in enumerate(ordered):
            tasks[key].metadata['native_schedule_position'] = position
        for previous, following in zip(ordered, ordered[1:]):
            for parent in (previous, *(() if overlap else sends[previous])):
                if parent not in tasks[following].dependencies:
                    add_dependency(parent, following)
