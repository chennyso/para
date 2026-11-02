from collections import defaultdict
from graphlib import TopologicalSorter

from schedule_validator import ScheduleValidator
from progress_validator import DistributedProgressValidator


class CalibrationPlanCompiler:
    def compile_async_publication(self, plan, receive_window=2, require_progress=True):
        if receive_window < 1:
            raise ValueError('提前接收窗口必须大于零')
        tasks, schedules = plan['tasks'], {}
        for rank, original in plan['rank_schedules'].items():
            outgoing = {tasks[key]['metadata']['producer_task']: key for key in original
                        if tasks[key]['kind'] == 'pp_p2p_send'}
            ordered = []
            for key in original:
                if tasks[key]['kind'] == 'pp_p2p_send':
                    continue
                ordered.append(key)
                if key in outgoing:
                    ordered.append(outgoing[key])
            original = ordered
            receives = defaultdict(list)
            for key in original:
                if tasks[key]['kind'] == 'pp_p2p_recv':
                    receives[tasks[key]['metadata']['channel']].append(key)
            posted = set()
            pending = defaultdict(int)
            result = []
            def post_ready():
                for channel, sequence in receives.items():
                    for key in sequence:
                        if key in posted:
                            continue
                        if pending[channel] >= receive_window:
                            break
                        result.append(key)
                        posted.add(key)
                        pending[channel] += 1
            post_ready()
            for key in original:
                task = tasks[key]
                if task['kind'] == 'pp_p2p_recv':
                    continue
                received = task['metadata'].get('input_task')
                if task['kind'] in ('forward_compute', 'backward_compute') and received is not None and received not in posted:
                    raise ValueError(f'消费者{key}的接收任务{received}超出合法发布窗口')
                result.append(key)
                if task['kind'] in ('forward_compute', 'backward_compute') and received is not None:
                    pending[tasks[received]['metadata']['channel']] -= 1
                post_ready()
            schedules[rank] = result
        result = dict(plan, rank_schedules=schedules,
                      communication_receive_window_per_channel=receive_window,
                      communication_protocol='producer_notification_gloo_preposted_v2')
        result['asynchronous_communication_legality'] = ScheduleValidator().validate(result)
        result['distributed_progress_validation'] = DistributedProgressValidator().validate(result)
        if require_progress and not result['distributed_progress_validation']['valid']:
            raise ValueError('全局计算通信计划无法完成资源约束下的事件推进')
        return result

    def compile_communication(self, plan):
        tasks = plan['tasks']
        sequences = {str(rank): [] for rank in {task['rank'] for task in tasks.values()}}
        def append(key):
            sequences[str(tasks[key]['rank'])].append(key)
        for mb in range(plan['num_microbatches']):
            for direction, prefix, regions in (
                ('forward', 'F', range(1, plan['num_regions'])),
                ('backward', 'B', reversed(range(plan['num_regions'] - 1)))):
                for region in regions:
                    for role in ('S', 'R', 'C'):
                        append(f'{role}{prefix}:0:{mb}:{region}')
        result = dict(plan, serial_calibration_rank_schedules=sequences)
        result['serial_calibration_legality'] = ScheduleValidator().validate(
            result, 'serial_calibration_rank_schedules', blocking_send=True)
        return result

    def compile(self, plan):
        tasks = plan['tasks']
        sequences = {str(rank): [] for rank in {task['rank'] for task in tasks.values()}}
        lanes = plan['config']['tp'] * plan['config']['cp'] * plan['config']['dp']
        def append(key):
            sequences[str(tasks[key]['rank'])].append(key)
        for mb in range(plan['num_microbatches']):
            for region in range(plan['num_regions']):
                for lane in range(lanes):
                    if region:
                        append(f'RF:{lane}:{mb}:{region}')
                    append(f'F:{lane}:{mb}:{region}')
                    if region + 1 < plan['num_regions']:
                        append(f'SF:{lane}:{mb}:{region + 1}')
            for region in reversed(range(plan['num_regions'])):
                for lane in range(lanes):
                    if region + 1 < plan['num_regions']:
                        append(f'RB:{lane}:{mb}:{region}')
                    append(f'B:{lane}:{mb}:{region}')
                    if region:
                        append(f'SB:{lane}:{mb}:{region - 1}')
        assigned = {key for sequence in sequences.values() for key in sequence}
        dependencies = {key: set(task['dependencies']) for key, task in tasks.items()}
        for key in TopologicalSorter(dependencies).static_order():
            if key not in assigned:
                append(key)
        result = dict(plan, serial_calibration_rank_schedules=sequences)
        result['serial_calibration_legality'] = ScheduleValidator().validate(
            result, 'serial_calibration_rank_schedules', blocking_send=True)
        return result
