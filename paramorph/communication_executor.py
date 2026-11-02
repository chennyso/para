from datetime import timedelta
import json
import hashlib
import math
import time
import threading
import os
import traceback

import torch
import torch.distributed as dist

from schedule_validator import ScheduleValidator


class CommunicationExecutor:
    def __init__(self, evidence, plan, dtype):
        self.evidence, self.plan, self.dtype = evidence, plan, dtype
        self.validator = ScheduleValidator()
        self.validator.validate(plan)
        self.groups, self.streams, self.states = {}, {}, {}
        self.receive_buffers = {}
        self.control_works = []
        self.control_rows = []
        self.receive_publication_tails = {}
        self.notification_receives = {}
        self.message_tags = {key: index for index, key in enumerate(sorted(
            key for key, task in plan['tasks'].items() if task['kind'] == 'pp_p2p_recv'))}
        self.receive_keys = {(task['metadata']['channel'], task['microbatch']): key
                             for key, task in plan['tasks'].items() if task['kind'] == 'pp_p2p_recv'}
        self.iterations = evidence.communication_iterations
        self.plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        self.last_snapshot = 0.0

    def initialize(self):
        if self.groups:
            return
        for key, task in self.plan['tasks'].items():
            if task['kind'] == 'pp_p2p_recv' and task['rank'] == dist.get_rank():
                metadata = task['metadata']
                self.receive_buffers[key] = torch.empty(metadata['shape'], dtype=self.dtype,
                    device='cuda', requires_grad=metadata['direction'] == 'forward')
        channels = {task['metadata']['channel']: tuple(task['metadata']['group_ranks'])
                    for task in self.plan['tasks'].values() if task['kind'] == 'pp_p2p_send'}
        for channel, ranks in sorted(channels.items()):
            group = dist.new_group(list(ranks), timeout=timedelta(seconds=120), backend='nccl',
                                  device_id=torch.device('cuda', torch.cuda.current_device()))
            self.groups[channel] = group
            if dist.get_rank() in ranks:
                self.streams[channel] = torch.cuda.Stream()
        self.control_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=120))
        source_tensor = self.evidence.communication_initialization_payload
        maximum = max(math.prod(task['metadata']['shape'])
                      for task in self.plan['tasks'].values() if task['kind'] == 'pp_p2p_send')
        if source_tensor.numel() < maximum:
            raise ValueError('通信初始化缺少足够的真实载荷')
        payload = source_tensor.detach().flatten()[:maximum].to(self.dtype).contiguous()
        receive = torch.empty_like(payload)
        initialization = []
        for channel, ranks in sorted(channels.items()):
            if dist.get_rank() in ranks:
                published = time.perf_counter_ns()
                source, destination = ranks
                sending = dist.get_rank() == source
                operation = dist.P2POp(dist.isend if sending else dist.irecv,
                    payload if sending else receive,
                    destination if sending else source, group=self.groups[channel])
                for work in dist.batch_isend_irecv([operation]):
                    work.wait()
                torch.cuda.synchronize()
                checksum = (payload if sending else receive).double().sum()
                checksums = [torch.empty_like(checksum) for _ in ranks]
                dist.all_gather(checksums, checksum, group=self.groups[channel])
                if not torch.equal(checksums[0], checksums[1]):
                    raise ValueError('NCCL通道初始化的真实载荷校验失败')
                initialization.append(dict(rank=dist.get_rank(), channel=channel,
                    source=source, destination=destination, bytes=payload.numel() * payload.element_size(),
                    publication_host_ns=published, completion_observed_host_ns=time.perf_counter_ns(),
                    payload_checksum_verified=True))
            dist.barrier()
        self.initialization_source = 'actual_loaded_parameter_or_c4_payload_channel_priming'
        self.evidence.communication_initialization_records.extend(initialization)
        (self.evidence.output / f'rank{dist.get_rank()}.paramorph-communication-initialization.json').write_text(
            json.dumps(dict(source=self.initialization_source,
                            channels=self.evidence.communication_initialization_records), indent=2) + '\n')

    def begin_iteration(self, serial, schedule_key):
        if any(not state['row']['completed'] for state in self.states.values()):
            raise ValueError('前一次迭代仍存在未完成通信')
        self.states = {}
        self.control_works = []
        self.control_rows = []
        self.receive_publication_tails = {}
        self.notification_receives = {}
        self.serial = serial
        for key, task in self.plan['tasks'].items():
            if not serial and task['kind'] == 'pp_p2p_recv' and task['rank'] == dist.get_rank():
                self.prepare_notification_receive(key)
        for tensor in self.receive_buffers.values():
            tensor.grad = None
        self.validator.validate(self.plan, schedule_key, blocking_send=serial)

    def prepare_notification_receive(self, key):
        if key in self.notification_receives:
            raise ValueError('控制通知接收重复登记')
        tensor = torch.empty(1, dtype=torch.int64)
        posted = time.perf_counter_ns()
        work = dist.irecv(tensor, self.plan['tasks'][key]['metadata']['group_ranks'][0],
                          group=self.control_group, tag=self.message_tags[key])
        self.notification_receives[key] = dict(tensor=tensor, work=work, posted_host_ns=posted)

    def identity(self, key, role):
        task, rank = self.plan['tasks'][key], dist.get_rank()
        metadata = task['metadata']
        source, destination = metadata['group_ranks']
        destination_region = task['region_id']
        source_region = destination_region + (-1 if metadata['direction'] == 'forward' else 1)
        region = source_region if role == 'send' else destination_region
        return dict(task_id=key, step=self.evidence.step, rank=rank,
            microbatch=task['microbatch'], vp_stage=region // self.plan['config']['pp'],
            region_id=region, source_region=source_region, destination_region=destination_region,
            peer=destination if role == 'send' else source,
            source=source, destination=destination, role=role, channel=metadata['channel'],
            direction=metadata['direction'],
            message_id=f'{self.evidence.step}:{metadata["channel"]}:{task["microbatch"]}',
            published_host_ns=time.perf_counter_ns(), completed=False,
            completion_observed_host_ns=None, consumer_task=None, buffer_released=False,
            shape=metadata['shape'], dtype=metadata['dtype'], wait_dependencies=[])

    def publish_receive(self, key):
        if key in self.states:
            raise ValueError('接收任务重复发布')
        task = self.plan['tasks'][key]
        metadata = task['metadata']
        deadline = time.monotonic() + 120
        while True:
            self.reap_completed()
            pending = sum(state['row']['role'] == 'recv'
                          and state['row']['channel'] == metadata['channel']
                          and not state['row']['completed'] for state in self.states.values())
            if pending < self.plan.get('communication_receive_window_per_channel', len(self.plan['tasks'])):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'接收窗口没有释放：{metadata["channel"]}')
            self.snapshot()
            time.sleep(0.001)
        tensor = self.receive_buffers[key]
        stream = self.streams[metadata['channel']]
        stream.wait_stream(torch.cuda.current_stream())
        row = self.identity(key, 'recv')
        if self.serial:
            self.prepare_notification_receive(key)
        notification_state = self.notification_receives[key]
        row['notification_receive_posted_host_ns'] = notification_state['posted_host_ns']
        row['notification_receive_dependencies'] = ['serial_receive_intent' if self.serial else 'iteration_start']
        row['communication_cuda_stream'] = stream.cuda_stream
        end = torch.cuda.Event(enable_timing=self.evidence.request.get('paramorph_calibration', False))
        row['publication_status'] = 'awaiting_producer_notification'
        ready = threading.Event()
        predecessor = self.receive_publication_tails.get(row['channel'])
        self.receive_publication_tails[row['channel']] = ready
        self.states[key] = dict(row=row, tensor=tensor, work=None, end=None, ready=ready)
        self.snapshot(force=len(self.states) == 1)
        device = torch.cuda.current_device()
        state = self.states[key]
        def receive_when_producer_ready():
            torch.cuda.set_device(device)
            notification_state['work'].wait()
            notification = notification_state['tensor']
            if notification.item() != row['step']:
                raise ValueError('通信生产者通知与训练迭代不同')
            row['producer_notification_received_host_ns'] = time.perf_counter_ns()
            if predecessor is not None and not predecessor.wait(120):
                raise TimeoutError('同一通信通道的前序接收发布超时')
            with torch.cuda.stream(stream):
                row['nccl_publication_started_host_ns'] = time.perf_counter_ns()
                work, = dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, row['source'],
                                                        group=self.groups[row['channel']])])
                row['publication_status'] = 'published'
                row['publication_completed_host_ns'] = time.perf_counter_ns()
                state['work'] = work
                work.wait()
                end.record()
            state['end'] = end
            ready.set()
        def run_receiver():
            try:
                receive_when_producer_ready()
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        worker = threading.Thread(target=run_receiver, name=f'paramorph-receive-{key}', daemon=True)
        state['worker'] = worker
        worker.start()
        self.snapshot()

    def consume_receive(self, key, consumer):
        state = self.states[key]
        if not state['ready'].wait(120):
            raise TimeoutError('通信生产者通知或者NCCL接收发布超时')
        if state['row']['consumer_task'] is not None:
            raise ValueError('接收Tensor被重复消费')
        state['row']['wait_dependencies'].append(dict(operation='tensor_consumption', task_id=consumer))
        state['row']['wait_enqueued_host_ns'] = time.perf_counter_ns()
        state['row']['consumer_cuda_stream'] = torch.cuda.current_stream().cuda_stream
        previous_task = self.evidence.active_plan_task
        self.evidence.active_plan_task = key
        self.evidence.record('pp_p2p_recv', 'receive_completion_before_consume',
                             torch.cuda.current_stream().wait_event,
                             (state['end'],), {}, dict(task_id=key, **self.plan['tasks'][key]['metadata']))
        self.evidence.active_plan_task = previous_task
        state['row']['consumer_task'] = consumer
        state['tensor'].record_stream(torch.cuda.current_stream())
        self.snapshot()
        return state['tensor']

    def release_receive(self, key):
        if key is None:
            return
        state = self.states[key]
        if state['row']['consumer_task'] is None or state['row']['buffer_released']:
            raise ValueError('接收Buffer释放顺序不合法')
        state['tensor'] = None
        state['row']['buffer_released'] = True
        state['row']['buffer_released_host_ns'] = time.perf_counter_ns()
        state['release_event'] = torch.cuda.Event(enable_timing=self.evidence.request.get('paramorph_calibration', False))
        state['release_event'].record()

    def publish_send(self, key, tensor):
        if key in self.states:
            raise ValueError('发送任务重复发布')
        self.reap_completed()
        task = self.plan['tasks'][key]
        metadata = task['metadata']
        if list(tensor.shape) != metadata['shape'] or tensor.dtype != self.dtype:
            raise ValueError('实际发送Tensor与计划载荷不一致')
        row = self.identity(key, 'send')
        stream = self.streams[row['channel']]
        producer = torch.cuda.Event()
        producer.record()
        stream.wait_event(producer)
        row['communication_cuda_stream'] = stream.cuda_stream
        row['wait_dependencies'].append(dict(operation='producer_cuda_event',
            task_id=metadata.get('producer_task'), cuda_stream=torch.cuda.current_stream().cuda_stream))
        timed = self.evidence.request.get('paramorph_calibration', False)
        start, end = torch.cuda.Event(enable_timing=timed), torch.cuda.Event(enable_timing=timed)
        tensor = tensor.detach()
        tensor.record_stream(stream)
        state = dict(row=row, tensor=tensor, work=None, start=start, end=None)
        self.states[key] = state
        receive_key = self.receive_keys[(row['channel'], row['microbatch'])]
        notification = torch.tensor([self.evidence.step], dtype=torch.int64)
        row['producer_notification_started_host_ns'] = time.perf_counter_ns()
        control_work = dist.isend(notification, row['destination'], group=self.control_group,
                                  tag=self.message_tags[receive_key])
        self.control_works.append((control_work, notification))
        self.control_rows.append(row)
        row['producer_notification_published_host_ns'] = time.perf_counter_ns()
        row['producer_notification_dependencies'] = ['producer_task_host_submission']
        with torch.cuda.stream(stream):
            start.record()
            row['nccl_publication_started_host_ns'] = time.perf_counter_ns()
            row['publication_status'] = 'publishing'
            self.snapshot()
            work, = dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, row['destination'],
                                                    group=self.groups[row['channel']])])
            row['publication_completed_host_ns'] = time.perf_counter_ns()
            row['publication_status'] = 'published'
            row['work_wait_enqueued_host_ns'] = time.perf_counter_ns()
            work.wait()
            end.record()
        state.update(work=work, end=end)
        if timed:
            self.evidence.events.append(dict(step=self.evidence.step, kind='pp_p2p_send',
                operation='send_completion', identity=dict(task_id=key, **metadata),
                host_start_ns=row['published_host_ns'], host_end_ns=time.perf_counter_ns(),
                allocated_before=tensor.numel() * tensor.element_size(),
                allocated_after=tensor.numel() * tensor.element_size(), cuda_start=start, cuda_end=end))
        if self.serial:
            self.wait_send(key, 'serial_calibration_boundary', block_host=True)
        self.snapshot()

    def wait_send(self, key, boundary, block_host=False):
        state = self.states[key]
        state['row']['wait_dependencies'].append(dict(operation=boundary, task_id=key))
        torch.cuda.current_stream().wait_event(state['end'])
        if block_host:
            state['end'].synchronize()
            self.complete(state)

    def complete(self, state):
        if not state['row']['completed']:
            state['row']['completed'] = True
            state['row']['completion_observed_host_ns'] = time.perf_counter_ns()
        if state['row']['role'] == 'send' and not state['row']['buffer_released']:
            state['tensor'] = None
            state['row']['buffer_released'] = True
            state['row']['buffer_released_host_ns'] = time.perf_counter_ns()

    def reap_completed(self):
        for state in self.states.values():
            if not state['row']['completed'] and state['end'] is not None and state['end'].query():
                self.complete(state)

    def finish_iteration(self):
        notifications_ready = threading.Event()
        def finish_notifications():
            for (work, notification), row in zip(self.control_works, self.control_rows, strict=True):
                work.wait(timedelta(seconds=120))
                row['producer_notification_completed_host_ns'] = time.perf_counter_ns()
            notifications_ready.set()
        def run_notification_completion():
            try:
                finish_notifications()
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        notification_worker = threading.Thread(target=run_notification_completion,
            name='paramorph-notification-completion', daemon=True)
        notification_worker.start()
        deadline = time.monotonic() + 120
        while True:
            self.reap_completed()
            notifications_done = notifications_ready.is_set()
            messages_done = all(state['row']['completed'] for state in self.states.values())
            releases_done = all(state.get('release_event', state['end']) is not None
                                and state.get('release_event', state['end']).query()
                                for state in self.states.values())
            if notifications_done and messages_done and releases_done:
                break
            if time.monotonic() >= deadline:
                self.snapshot(force=True)
                raise TimeoutError('迭代结束前仍有未完成的通知、NCCL通信或者Buffer使用')
            self.snapshot()
            time.sleep(0.001)
        notification_worker.join()
        for state in self.states.values():
            if state['row']['role'] == 'recv' and not state['ready'].wait(120):
                raise TimeoutError('迭代结束前接收发布仍然没有完成')
            if state['end'] is None:
                raise ValueError('接收通信没有对应的Tensor消费任务')
            state['end'].synchronize()
            self.complete(state)
            if not state['row']['buffer_released']:
                raise ValueError('迭代结束后仍然保留接收Buffer')
            release_event = state.get('release_event', state['end'])
            release_event.synchronize()
            if self.evidence.request.get('paramorph_calibration', False):
                state['row']['completion_cuda_ms_from_iteration_origin'] = (
                    self.evidence.cuda_origins[self.evidence.step].elapsed_time(state['end']))
                state['row']['buffer_release_cuda_ms_from_iteration_origin'] = (
                    self.evidence.cuda_origins[self.evidence.step].elapsed_time(release_event))
        self.iterations.append(dict(rank=dist.get_rank(), step=self.evidence.step,
            plan_sha256=self.plan_digest,
            serial_calibration=self.serial,
            notification_receive_policy=('serial_control_receives_follow_matched_global_plan' if self.serial
                else 'all_control_receives_preposted_before_iteration_tasks'),
            notification_receive_bytes=sum(value['tensor'].numel() * value['tensor'].element_size()
                                           for value in self.notification_receives.values()),
            receive_slot_policy='persistent_slots_reused_after_backward_and_iteration_completion',
            reserved_receive_slot_bytes=sum(tensor.numel() * tensor.element_size()
                                            for tensor in self.receive_buffers.values()),
            messages=[state['row'] for state in self.states.values()]))
        if self.evidence.request.get('paramorph_calibration', False):
            self.save()

    def snapshot(self, force=False):
        now = time.monotonic()
        if (self.evidence.request.get('paramorph_calibration', False)
            and self.evidence.request.get('paramorph_progress_snapshots', True)
            and (force or now - self.last_snapshot >= 1)):
            self.reap_completed()
            payload = dict(rank=dist.get_rank(), step=self.evidence.step,
                           pending=[state['row'] for state in self.states.values() if not state['row']['completed']])
            path = self.evidence.output / f'rank{dist.get_rank()}.paramorph-communication-progress.json'
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
            self.last_snapshot = now

    def save(self):
        path = self.evidence.output / f'rank{dist.get_rank()}.paramorph-communication.json'
        path.write_text(json.dumps(dict(iterations=self.iterations), indent=2, allow_nan=False) + '\n')
