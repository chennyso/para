from collections import defaultdict
from graphlib import TopologicalSorter


class ScheduleValidator:
    def validate(self, plan, schedule_key='rank_schedules', blocking_send=False):
        tasks, sequences = plan['tasks'], plan[schedule_key]
        assigned = [key for sequence in sequences.values() for key in sequence]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(tasks):
            raise ValueError('执行计划必须完整覆盖全部任务一次')
        dependencies = {key: set(task['dependencies']) for key, task in tasks.items()}
        sends, receives, consumers = {}, {}, defaultdict(list)
        orders = defaultdict(lambda: dict(send=[], recv=[]))
        compute_orders = {}
        for rank, sequence in sequences.items():
            computation = []
            for key in sequence:
                task, metadata = tasks[key], tasks[key]['metadata']
                if task['rank'] != int(rank) or not dependencies[key] <= tasks.keys():
                    raise ValueError('任务设备或者依赖身份不合法')
                kind = task['kind']
                if kind in ('pp_p2p_send', 'pp_p2p_recv'):
                    source, destination = metadata['group_ranks']
                    role = 'send' if kind == 'pp_p2p_send' else 'recv'
                    if task['rank'] != (source if role == 'send' else destination) or source == destination:
                        raise ValueError('通信任务与实际Peer不一致')
                    identity = (metadata['channel'], task['microbatch'])
                    table = sends if role == 'send' else receives
                    if identity in table:
                        raise ValueError('通信身份重复发布')
                    table[identity] = key
                    orders[metadata['channel']][role].append(task['microbatch'])
                if kind in ('forward_compute', 'backward_compute', 'communication_consume'):
                    if kind != 'communication_consume':
                        computation.append((kind, task['microbatch'], task['region_id']))
                    received = metadata.get('input_task')
                    expected = metadata['direction'] if kind == 'communication_consume' else (
                        'forward' if kind == 'forward_compute' else 'backward')
                    if received is not None:
                        receiver = tasks[received]
                        if (receiver['kind'] != 'pp_p2p_recv' or receiver['rank'] != task['rank']
                            or receiver['microbatch'] != task['microbatch']
                            or receiver['region_id'] != task['region_id']
                            or receiver['metadata']['direction'] != expected):
                            raise ValueError('计算任务使用了不匹配的接收Buffer')
                        consumers[received].append(key)
                    elif (kind == 'forward_compute' and task['region_id'] != 0) or (
                        kind == 'backward_compute' and task['region_id'] != plan['num_regions'] - 1):
                        raise ValueError('计算任务缺少必须接收的数据')
            compute_orders[int(rank)] = computation
            for left, right in zip(sequence, sequence[1:]):
                dependencies[right].add(left)
        if sends.keys() != receives.keys():
            raise ValueError('Send与Recv没有完整匹配')
        for identity, send_key in sends.items():
            receive_key = receives[identity]
            left, right = tasks[send_key]['metadata'], tasks[receive_key]['metadata']
            if any(left[field] != right[field] for field in ('group_ranks', 'shape', 'dtype', 'direction')):
                raise ValueError('匹配的Send与Recv具有不同的通信载荷')
            if send_key not in tasks[receive_key]['dependencies'] or len(consumers[receive_key]) != 1:
                raise ValueError('接收完成与Tensor消费依赖不完整')
            dependencies[receive_key].discard(send_key)
            consumer = consumers[receive_key][0]
            dependencies[consumer].add(send_key)
        for order in orders.values():
            if order['send'] != order['recv'] or len(set(order['send'])) != len(order['send']):
                raise ValueError('NCCL通道中的Send与Recv顺序不一致')
        tp = plan['config']['tp']
        for rank, computation in compute_orders.items():
            leader = rank - rank % tp
            if computation != compute_orders[leader]:
                raise ValueError('同一TP组的Collective执行顺序不一致')
        tuple(TopologicalSorter(dependencies).static_order())
        progress_dependencies = {key: set(value) for key, value in dependencies.items()}
        previous_compute = {}
        for sequence in sequences.values():
            compute = None
            pending = defaultdict(int)
            for key in sequence:
                previous_compute[key] = compute
                task = tasks[key]
                if task['kind'] == 'pp_p2p_recv':
                    channel = task['metadata']['channel']
                    pending[channel] += 1
                    if pending[channel] > plan.get('communication_receive_window_per_channel', len(tasks)):
                        raise ValueError('提前发布的接收数量超过计划显存窗口')
                elif task['kind'] in ('forward_compute', 'backward_compute', 'communication_consume'):
                    received = task['metadata'].get('input_task')
                    if received is not None:
                        pending[tasks[received]['metadata']['channel']] -= 1
                        if pending[tasks[received]['metadata']['channel']] < 0:
                            raise ValueError('Tensor消费发生在接收发布之前')
                    compute = key
        for channel, order in orders.items():
            previous_completion = None
            for microbatch in order['send']:
                identity = channel, microbatch
                send, receive = sends[identity], receives[identity]
                completion = send + ':matched_completion'
                progress_dependencies[completion] = {send, receive}
                if plan.get('communication_protocol') in ('producer_notification_gloo_and_nccl_v1',
                                                        'producer_notification_gloo_preposted_v2'):
                    notification = send + ':producer_notification'
                    progress_dependencies[notification] = set(dependencies[send])
                    publication = receive + ':nccl_publication'
                    progress_dependencies[publication] = {receive, notification}
                    progress_dependencies[completion].add(publication)
                for key in (send, receive):
                    if previous_compute[key] is not None:
                        progress_dependencies[completion].add(previous_compute[key])
                if previous_completion is not None:
                    progress_dependencies[completion].add(previous_completion)
                progress_dependencies[consumers[receive][0]].add(completion)
                previous_completion = completion
        if tp > 1:
            compute_units = defaultdict(list)
            for key, task in tasks.items():
                if task['kind'] in ('forward_compute', 'backward_compute'):
                    compute_units[(task['rank'] // tp, task['kind'], task['microbatch'],
                                   task['region_id'])].append(key)
            collective_completion = {}
            for index, keys in enumerate(compute_units.values()):
                if len(keys) != tp:
                    raise ValueError('TP Collective任务没有覆盖全部Tensor Rank')
                completion = f'tp_collective_completion:{index}'
                progress_dependencies[completion] = set(keys)
                collective_completion.update({key: completion for key in keys})
            for key, values in progress_dependencies.items():
                if key.startswith('tp_collective_completion:'):
                    continue
                progress_dependencies[key] = {collective_completion.get(value, value)
                                              for value in values}
        tuple(TopologicalSorter(progress_dependencies).static_order())
        if blocking_send:
            aliases = {key: key for key in tasks}
            for identity, receiver in receives.items():
                aliases[receiver] = sends[identity]
            contracted = defaultdict(set)
            for key, predecessors in dependencies.items():
                for predecessor in predecessors:
                    if aliases[key] != aliases[predecessor]:
                        contracted[aliases[key]].add(aliases[predecessor])
                contracted[aliases[key]]
            tuple(TopologicalSorter(contracted).static_order())
        return dict(valid=True, tasks=len(tasks), matched_messages=len(sends),
                    tp_collective_order_verified=True, cuda_communication_progress_verified=True,
                    blocking_send_progress_verified=blocking_send)

    def validate_progress(self, payloads):
        messages = defaultdict(dict)
        publications = defaultdict(lambda: dict(send=[], recv=[]))
        for payload in payloads:
            for row in payload['messages']:
                identity = (payload['step'], row['channel'], row['microbatch'])
                role = row['role']
                if role in messages[identity]:
                    raise ValueError('通信进展记录存在重复身份')
                messages[identity][role] = row
                publications[(payload['step'], row['channel'])][role].append((
                    row.get('nccl_publication_started_host_ns', row['published_host_ns']),
                    row['microbatch']))
        for order in publications.values():
            sends = [microbatch for _, microbatch in sorted(order['send'])]
            receives = [microbatch for _, microbatch in sorted(order['recv'])]
            if sends != receives:
                raise ValueError('实际NCCL通道的Send与Recv发布顺序不同')
        for identity, pair in messages.items():
            if pair.keys() != {'send', 'recv'}:
                raise ValueError(f'实际通信{identity}缺少匹配Peer记录')
            for row in pair.values():
                if (not row['completed'] or row['completion_observed_host_ns'] < row['published_host_ns']
                    or not row['buffer_released']):
                    raise ValueError('通信或者Buffer生命周期没有完整结束')
            if pair['recv']['consumer_task'] is None:
                raise ValueError('实际接收Tensor没有被训练任务消费')
            sender, receiver = pair['send'], pair['recv']
            if 'producer_notification_dependencies' in sender:
                if sender['producer_notification_dependencies'] != ['producer_task_host_submission']:
                    raise ValueError('生产者通知包含通信完成等待依赖')
                if not (sender['producer_notification_started_host_ns']
                        <= sender['producer_notification_published_host_ns']
                        <= sender['nccl_publication_started_host_ns']
                        <= sender['publication_completed_host_ns']
                        <= sender['work_wait_enqueued_host_ns']):
                    raise ValueError('生产者通知没有在NCCL发送及等待之前发布')
                if not (receiver['producer_notification_received_host_ns']
                        <= receiver['nccl_publication_started_host_ns']
                        <= receiver['publication_completed_host_ns']):
                    raise ValueError('接收通知与NCCL发布顺序不合法')
            if any(pair['send'][field] != pair['recv'][field] for field in
                   ('source', 'destination', 'microbatch', 'shape', 'dtype', 'direction', 'message_id')):
                raise ValueError('实际通信进展记录的载荷身份不匹配')
            for row in pair.values():
                if 'buffer_release_cuda_ms_from_iteration_origin' in row and (
                    row['buffer_release_cuda_ms_from_iteration_origin'] + 0.001
                    < row['completion_cuda_ms_from_iteration_origin']):
                    raise ValueError('实际GPU Buffer在通信完成前释放')
        return dict(valid=True, matched_messages=len(messages), source='actual_nccl_progress_records')
