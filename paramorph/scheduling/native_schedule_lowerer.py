from collections import Counter, defaultdict

from core_data_models import Task, TaskType
from scheduling.native_microbatches import native_schedule_table


def native_stage_events(stage, pp, vpp, microbatches, group):
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
    return ([(*event, 'warmup') for event in forward[:warmup]]
        + [(*event, 'steady') for event in steady]
        + [(*event, 'cooldown') for event in backward[len(table) - warmup:]])


def native_stage_sequence(stage, pp, vpp, microbatches, group):
    return [event[:3] for event in native_stage_events(stage, pp, vpp, microbatches, group)]


def verify_native_execution_events(request, ranks):
    world = request['node_count'] * request['gpus_per_node']
    group = request.get('group', request['pp']) if request['vpp'] > 1 else 1
    microbatches = request['global_batch'] // (request['dp'] * request['micro_batch'])
    sends, receives = Counter(), Counter()
    compute_count, exchanges, inline_waits, asynchronous = 0, 0, 0, 0
    observed_ranks = set()
    for row in ranks:
        if row['rank'] in observed_ranks:
            raise ValueError('原生执行事件重复登记同一设备')
        observed_ranks.add(row['rank'])
        expected = native_stage_sequence(row['pp_rank'], request['pp'], request['vpp'], microbatches, group)
        by_step = defaultdict(list)
        for event in row['events']:
            if event['kind'] in ('forward_compute', 'backward_compute', 'pp_p2p_exchange'):
                by_step[event['step']].append(event)
        if not by_step or set(by_step) != {window['step'] for window in row['windows']}:
            raise ValueError(f'原生执行事件缺少迭代：Rank {row["rank"]}')
        for step, events in by_step.items():
            observed = []
            for event in sorted(events, key=lambda value: value['host_start_ns']):
                identity = event['identity']
                if event['kind'] != 'pp_p2p_exchange':
                    observed.append((event['kind'], identity['vp_stage'], identity['microbatch']))
                    continue
                exchanges += 1
                inline_waits += int(identity['completion_waited'])
                asynchronous += int(not identity['completion_waited'])
                for message in identity['messages']:
                    peer_key = (step, message['source'], message['destination'], message['direction'],
                        message['message_id'], identity['dtype'])
                    if message['role'] == 'send' and message['source'] == row['rank']:
                        sends[peer_key] += 1
                    elif message['role'] == 'recv' and message['destination'] == row['rank']:
                        receives[peer_key] += 1
                    else:
                        raise ValueError('原生 P2P 事件与实际通信端点不一致')
            if observed != expected:
                raise ValueError(f'原生前向与反向顺序和执行图不同：Rank {row["rank"]}，迭代 {step}')
            compute_count += len(observed)
    if observed_ranks != set(range(world)):
        raise ValueError('原生执行事件未覆盖全部设备')
    if sends != receives or any(count != 1 for count in sends.values()):
        raise ValueError('原生 P2P 发送和接收事件无法逐条配对')
    return dict(ranks=world, ordered_compute_events=compute_count, p2p_exchanges=exchanges,
        matched_messages=sum(sends.values()), p2p_inline_waits=inline_waits, p2p_async_launches=asynchronous,
        async_wait_order_verified=False, source='actual_native_schedule_runtime_events')


def lower_native_schedule(tasks, config, settings, add_dependency):
    overlap = settings['overlap_p2p']
    batch = settings.get('batch_p2p_comm')
    batch = not overlap if batch is None else batch
    flush = settings.get('overlap_p2p_comm_warmup_flush') or False
    if (overlap and (batch or config.vpp == 1)) or (flush and not overlap):
        raise ValueError('P2P 通信开关不满足 Megatron 流水线约束')
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
        elif task.task_type == TaskType.PP_P2P_RECV:
            task.metadata['native_wait_policy'] = 'consumer_waits_for_receive_completion'
    if len(compute) != 2 * config.world_size * config.vpp * config.num_microbatches:
        raise ValueError('原生执行图未覆盖全部逐 Rank 微批事件')
    for rank in range(config.world_size):
        sequence = native_stage_events(rank // lanes, config.pp, config.vpp,
            config.num_microbatches, config.microbatch_group)
        ordered = [compute[rank, kind, vp, mb] for kind, vp, mb, _ in sequence]
        blocking = {}
        for position, (key, (_, _, _, phase)) in enumerate(zip(ordered, sequence, strict=True)):
            tasks[key].metadata.update(native_schedule_position=position, native_schedule_phase=phase)
            asynchronous = overlap and (phase == 'steady' or flush)
            blocking[key] = [] if asynchronous else sends[key]
            for transfer in sends[key]:
                task = tasks[transfer]
                launch = tasks[task.metadata['launch_event']]
                launch_settings = dict(native_schedule_phase=phase,
                    native_launch_policy='nonblocking' if asynchronous else 'blocking',
                    concurrency_mode='batched_p2p' if batch else 'unbatched_p2p')
                task.metadata.update(launch_settings)
                launch.metadata.update(launch_settings)
        for previous, following in zip(ordered, ordered[1:]):
            for parent in (previous, *blocking[previous]):
                if parent not in tasks[following].dependencies:
                    add_dependency(parent, following)
