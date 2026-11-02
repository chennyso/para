from collections import defaultdict
import json
from pathlib import Path
import statistics

from core_data_models import Task, TaskType
from cost.cost_model import MissingCalibration
from cost.runtime_calibration import configuration
from schedule_validator import ScheduleValidator


class PlanCalibration:
    def __init__(self, directory, plan):
        self.directory = Path(directory)
        self.plan = plan
        self.payloads = [json.loads(path.read_text()) for path in sorted(
            self.directory.glob('rank*.paramorph-cost.json'))]
        if not self.payloads:
            raise MissingCalibration('ExecutionPlan校准缺少真实Rank任务记录')
        self.config = configuration(self.payloads[0]['request'])
        if sorted(payload['rank'] for payload in self.payloads) != list(range(self.config.world_size)):
            raise MissingCalibration('执行计划校准需要全部训练Rank的真实记录')
        progress = [json.loads((self.directory / f'rank{rank}.paramorph-communication.json').read_text())
                    for rank in range(self.config.world_size)]
        self.communication_validation = ScheduleValidator().validate_progress([
            iteration for payload in progress for iteration in payload['iterations']])

    def ingest(self, costs, dag):
        measurements = defaultdict(list)
        retained = defaultdict(list)
        workspace = defaultdict(list)
        persistent = {}
        evidence = defaultdict(list)
        actual_pairs = []
        receive_completion = defaultdict(list)
        for path in self.directory.glob('rank*.paramorph-links.json'):
            links = json.loads(path.read_text())
            for row in links['rows']:
                if row['role'] == 'recv':
                    if 'completed_wait_milliseconds' not in row:
                        raise MissingCalibration('接收完成任务缺少已完成通信的同步成本')
                    receive_completion[(row['source'], row['destination'], row['direction'])].append(
                        (row['completed_wait_milliseconds'], row['evidence_id']))
        for payload in self.payloads:
            rank, request = payload['rank'], payload['request']
            if not payload.get('executed_plans') or any(not row['matches_plan'] for row in payload['executed_plans']):
                raise ValueError('真实训练未完整执行指定ExecutionPlan')
            persistent[rank] = max(row['persistent_bytes'] for row in payload['windows'])
            serial_steps = request['paramorph_serial_calibration_steps']
            if serial_steps < request['warmup']:
                raise MissingCalibration('缺少独立计算与通信的实际校准迭代')
            event_index = {}
            for event in payload['events']:
                step = event['step']
                if step < request['warmup'] or (request.get('paramorph_trace') and
                    step == request['warmup'] + request['measured'] - 1):
                    continue
                identifier = event['identity'].get('task_id')
                if identifier is None or identifier not in dag.tasks:
                    continue
                if event['kind'] not in ('forward_compute', 'backward_compute',
                                        'pp_p2p_send', 'pp_p2p_recv'):
                    continue
                if event['cuda_ms'] is None:
                    raise MissingCalibration('执行任务缺少真实GPU时间')
                event_index[(step, identifier)] = event
                if step <= serial_steps:
                    measurements[identifier].append(event['cuda_ms'])
                    evidence[identifier].append(f'{request["run_id"]}:{rank}:{step}:{event["host_start_ns"]}')
                if event['kind'] in ('forward_compute', 'backward_compute'):
                    if 'workspace_upper_bound_bytes' not in event:
                        raise MissingCalibration('计算任务缺少实际临时显存测量')
                    workspace[identifier].append(event['workspace_upper_bound_bytes'])
                if event['kind'] == 'forward_compute' and step <= serial_steps:
                    if 'saved_activation_bytes' not in event:
                        raise MissingCalibration('Forward缺少实际保存的激活显存记录')
                    retained[identifier].append(event['saved_activation_bytes'])
            sequence = self.plan['rank_schedules'][str(rank)]
            execution = [key for key in sequence if dag.tasks[key].task_type != TaskType.PP_P2P_RECV]
            def precedes(left, right):
                pending, visited = list(dag.tasks[right].dependencies), set()
                while pending:
                    key = pending.pop()
                    if key == left:
                        return True
                    if key not in visited:
                        visited.add(key)
                        pending.extend(dag.tasks[key].dependencies)
                return False
            for comm_id, compute_id in zip(execution, execution[1:]):
                comm, compute = dag.tasks[comm_id], dag.tasks[compute_id]
                if comm.task_type != TaskType.PP_P2P_SEND or not compute.is_compute:
                    continue
                if precedes(comm_id, compute_id) or precedes(compute_id, comm_id):
                    continue
                if not measurements[comm_id] or not measurements[compute_id]:
                    raise MissingCalibration('候选配对缺少对应的独立成本')
                for (step, key), communication in event_index.items():
                    if key != comm_id or step <= serial_steps or (step, compute_id) not in event_index:
                        continue
                    computation = event_index[(step, compute_id)]
                    start = min(communication['cuda_start_ms'], computation['cuda_start_ms'])
                    end = max(communication['cuda_end_ms'], computation['cuda_end_ms'])
                    costs.update_pair(compute, comm, self.config,
                        statistics.median(measurements[compute_id]), statistics.median(measurements[comm_id]),
                        end - start, f'{request["run_id"]}:{rank}:{step}:{comm_id}+{compute_id}')
                    actual_pairs.append((compute_id, comm_id))
            for event in payload['events']:
                if event['step'] < request['warmup']:
                    continue
                kind = event['kind']
                if kind == 'optimizer_step':
                    key = f'OPT:{rank}'
                elif kind == 'dp_reduce':
                    key = f'DP:{rank}'
                elif kind == 'cuda_sync':
                    key = f'END:{rank}'
                else:
                    continue
                measurements[key].append(event['cuda_ms'])
                evidence[key].append(f'{request["run_id"]}:{rank}:{event["step"]}:{event["host_start_ns"]}')
                if kind in ('optimizer_step', 'dp_reduce'):
                    if 'workspace_upper_bound_bytes' not in event:
                        raise MissingCalibration('更新任务缺少实际临时显存测量')
                    workspace[key].append(event['workspace_upper_bound_bytes'])
        for key, task in dag.tasks.items():
            if task.task_type == TaskType.PP_P2P_RECV:
                metadata = task.metadata
                source, destination = metadata['group_ranks']
                rows = receive_completion[(source, destination, metadata['direction'])]
                if not rows:
                    raise MissingCalibration('接收任务缺少独立的通信完成处理成本')
                measurements[key] = [row[0] for row in rows]
                evidence[key] = [row[1] for row in rows]
            if not measurements[key]:
                raise MissingCalibration(f'执行任务{key}缺少独立GPU成本')
            for value, identifier in zip(measurements[key], evidence[key]):
                costs.update(dict(task=task, config=self.config, milliseconds=value,
                    source='actual_execution_plan_serial_gpu_calibration', evidence_id=identifier))
            task.metadata['memory_evidence'] = evidence[key]
            if task.is_compute or task.task_type in (TaskType.OPTIMIZER_STEP, TaskType.DP_REDUCE):
                if not workspace[key]:
                    raise MissingCalibration(f'任务{key}缺少实际临时显存上界')
                task.workspace_bytes = max(workspace[key])
                task.metadata['workspace_scope'] = 'actual_allocator_peak_increase_upper_bound'
            if task.task_type == TaskType.FORWARD_COMPUTE:
                task.activation_delta_bytes = max(retained[key])
            elif task.task_type == TaskType.BACKWARD_COMPUTE:
                forward = key.replace('B:', 'F:', 1)
                if not retained[forward]:
                    raise MissingCalibration('Backward没有对应的实际保存激活记录')
                task.activation_delta_bytes = -max(retained[forward])
        return dict(persistent_bytes=persistent, calibrated_pair_candidates=list(dict.fromkeys(actual_pairs)),
            calibrated_tasks=len(measurements), missing_tasks=[key for key in dag.tasks if not measurements[key]],
            source='real_serial_and_concurrent_training_task_cuda_events',
            calibrated_pair_keys=len(costs.pairs), communication_validation=self.communication_validation)
