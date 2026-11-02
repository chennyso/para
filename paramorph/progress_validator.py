from collections import defaultdict, deque

from schedule_validator import ScheduleValidator


class DistributedProgressValidator:
    def validate(self, plan, schedule_key='rank_schedules'):
        ScheduleValidator().validate(plan, schedule_key)
        reports = [self.simulate(plan, schedule_key, reverse)
                   for reverse in (False, True)]
        return dict(valid=all(report['valid'] for report in reports),
                    source='bounded_distributed_event_simulation',
                    runtime_protocol='all_control_receives_preposted_before_iteration_tasks',
                    verification_scope='two_deterministic_host_interleavings_with_channel_fifo_tp_and_collective_rendezvous',
                    exhaustive_asynchronous_interleavings_verified=False,
                    interleavings=reports)

    def simulate(self, plan, schedule_key, reverse=False):
        tasks = plan['tasks']
        sequences = {int(rank): sequence for rank, sequence in plan[schedule_key].items()}
        ranks = sorted(sequences, reverse=reverse)
        cursors = {rank: 0 for rank in ranks}
        main = {rank: deque() for rank in ranks}
        channels = defaultdict(deque)
        main_tail = {rank: None for rank in ranks}
        gpu_done, cpu_done, recv_posted = set(), set(), set()
        notification, recv_intents, consumed, released = set(), set(), set(), set()
        transfer_done = set()
        occupied = defaultdict(int)
        sent, received = {}, {}
        receive_order = defaultdict(list)
        receive_cursors = defaultdict(int)
        compute_units = defaultdict(list)
        for rank in ranks:
            for key in sequences[rank]:
                task = tasks[key]
                kind, metadata = task['kind'], task['metadata']
                if kind.startswith('pp_p2p_'):
                    identity = metadata['channel'], task['microbatch']
                    if kind == 'pp_p2p_send':
                        sent[identity] = key
                    else:
                        received[identity] = key
                        receive_order[metadata['channel']].append(key)
                elif kind in ('forward_compute', 'backward_compute'):
                    compute_units[(rank // plan['config']['tp'], kind,
                                   task['microbatch'], task['region_id'])].append(key)
        unit_for = {key: keys for keys in compute_units.values() for key in keys}
        window = plan.get('communication_receive_window_per_channel', len(tasks))
        witness = []

        def emit(event, key, **state):
            witness.append(dict(index=len(witness), event=event,
                                task_id=key, rank=tasks[key]['rank'], **state))

        def identity(key):
            task = tasks[key]
            return task['metadata']['channel'], task['microbatch']

        def ready_head(rank, key):
            if not main[rank] or main[rank][0] != key:
                return False
            received_key = tasks[key]['metadata'].get('input_task')
            return received_key is None or identity(received_key) in transfer_done

        def completed(key):
            kind = tasks[key]['kind']
            if kind.startswith('pp_p2p_'):
                return identity(key) in transfer_done
            if kind in ('forward_compute', 'backward_compute', 'communication_consume'):
                return key in gpu_done
            return key in cpu_done

        while True:
            changed = False
            for rank in ranks:
                if cursors[rank] == len(sequences[rank]):
                    continue
                key = sequences[rank][cursors[rank]]
                task, metadata = tasks[key], tasks[key]['metadata']
                kind = task['kind']
                if kind == 'pp_p2p_recv':
                    channel = metadata['channel']
                    if occupied[channel] >= window:
                        continue
                    recv_intents.add(key)
                    occupied[channel] += 1
                    channels[(rank, channel)].append(('wait', main_tail[rank]))
                    emit('RECEIVE_INTENT', key, occupied=occupied[channel], capacity=window)
                elif kind == 'pp_p2p_send':
                    producer = metadata.get('producer_task')
                    if producer is not None and producer not in cpu_done:
                        continue
                    notification.add(identity(key))
                    channels[(rank, metadata['channel'])].append(('send', key, main_tail[rank]))
                    emit('NOTIFY_POSTED', key)
                    emit('NOTIFY_DELIVERED', received[identity(key)])
                    emit('SEND_POSTED', key)
                elif kind in ('forward_compute', 'backward_compute', 'communication_consume'):
                    received_key = metadata.get('input_task')
                    if received_key is not None:
                        if received_key not in recv_posted:
                            continue
                        consumed.add(received_key)
                        emit('WAIT_RECV_ENQUEUED', received_key, consumer=key)
                    main[rank].append(key)
                    main_tail[rank] = key
                    emit('COMPUTE_ENQUEUED', key)
                elif kind == 'dp_reduce':
                    own_receives = {key for key, item in tasks.items()
                                    if item['rank'] == rank and item['kind'] == 'pp_p2p_recv'}
                    own_messages = {identity(key) for key, item in tasks.items()
                                    if item['rank'] == rank and item['kind'].startswith('pp_p2p_')}
                    if main[rank] or not own_messages <= transfer_done or not own_receives <= released:
                        continue
                    emit('ITERATION_DRAINED', key)
                elif kind in ('optimizer_step', 'cuda_sync', 'optimizer_prepare_grads',
                        'optimizer_local_grad_norm', 'optimizer_local_grad_norm_finalize',
                        'optimizer_local_grad_clip'):
                    if any(not completed(dependency) for dependency in task['dependencies']):
                        continue
                    emit('TASK_COMPLETED', key)
                    gpu_done.add(key)
                elif kind == 'nccl_generic':
                    participants = metadata.get('group_ranks')
                    if (not participants or len(set(participants)) != len(participants)
                            or any(participant not in sequences for participant in participants)):
                        raise ValueError('Collective进展验证需要完整合法的参与Rank')
                    if (any(not completed(dependency) for dependency in task['dependencies'])
                            or any(main[participant] for participant in participants)):
                        continue
                    emit('COLLECTIVE_COMPLETED', key, participants=list(participants),
                        collective=metadata['collective'])
                    gpu_done.add(key)
                else:
                    raise ValueError(f'进展状态模型不支持任务类型：{kind}')
                cpu_done.add(key)
                cursors[rank] += 1
                changed = True

            for channel, order in receive_order.items():
                cursor = receive_cursors[channel]
                if cursor == len(order):
                    continue
                key = order[cursor]
                if key in recv_intents and identity(key) in notification:
                    recv_posted.add(key)
                    channels[(tasks[key]['rank'], channel)].append(('recv', key))
                    receive_cursors[channel] += 1
                    emit('RECV_POSTED', key)
                    changed = True

            for queue in channels.values():
                if queue and queue[0][0] == 'wait' and (
                    queue[0][1] is None or queue[0][1] in gpu_done):
                    queue.popleft()
                    changed = True
            for message, send_key in sent.items():
                if message in transfer_done:
                    continue
                receive_key = received[message]
                outgoing_queue = channels[(tasks[send_key]['rank'], message[0])]
                incoming_queue = channels[(tasks[receive_key]['rank'], message[0])]
                if not outgoing_queue or not incoming_queue:
                    continue
                outgoing, incoming = outgoing_queue[0], incoming_queue[0]
                if outgoing[:2] != ('send', send_key) or incoming != ('recv', receive_key):
                    continue
                if outgoing[2] is not None and outgoing[2] not in gpu_done:
                    continue
                outgoing_queue.popleft()
                incoming_queue.popleft()
                transfer_done.add(message)
                occupied[message[0]] -= 1
                emit('TRANSFER_COMPLETED', incoming[1])
                changed = True

            visited = set()
            for rank in ranks:
                if not main[rank]:
                    continue
                key = main[rank][0]
                unit = unit_for.get(key, [key])
                if key in visited or not all(ready_head(tasks[member]['rank'], member) for member in unit):
                    continue
                visited.update(unit)
                for member in unit:
                    task = tasks[member]
                    main[task['rank']].popleft()
                    gpu_done.add(member)
                    emit('COMPUTE_COMPLETED', member)
                    if task['kind'] in ('backward_compute', 'communication_consume'):
                        for receive_key in consumed - released:
                            receive = tasks[receive_key]
                            if (receive['rank'] == task['rank'] and receive['microbatch'] == task['microbatch']
                                and receive['region_id'] == task['region_id']):
                                released.add(receive_key)
                                emit('BUFFER_RELEASED', receive_key)
                changed = True

            if (all(cursors[rank] == len(sequences[rank]) and not main[rank] for rank in ranks)
                and len(transfer_done) == len(sent) and len(released) == len(received)):
                return dict(valid=True, host_order=ranks, events=witness,
                            completed_messages=len(transfer_done), released_buffers=len(released))
            if not changed:
                return dict(valid=False, host_order=ranks, events=witness,
                    blocked=[dict(rank=rank, task_id=sequences[rank][cursors[rank]],
                                  main_pending=list(main[rank])) for rank in ranks
                             if cursors[rank] < len(sequences[rank])],
                    occupied_windows=dict(occupied),
                    pending_channels={str(key): list(value) for key, value in channels.items() if value})
