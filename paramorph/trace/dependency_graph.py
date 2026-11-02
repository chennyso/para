from collections import defaultdict
from dataclasses import dataclass

from core_data_models import Task, TaskType, ExecutionDAG
from paramorph.trace.trace_parser import TraceEventType
from scheduling.schedule_simulator import topological_order


@dataclass
class CommunicationEdge:
    send_task_id: str
    recv_task_id: str
    send_rank: int
    recv_rank: int
    message_size_bytes: int
    comm_type: str


class ExecutionGraphBuilder:
    def build(self, bundle):
        dag = ExecutionDAG({}, [], dict(coverage='gpu_kernel_only', data_gaps=[]))
        streams = defaultdict(list)
        messages = defaultdict(dict)
        for event in bundle.events:
            if event.event_type not in (TraceEventType.CUDA_KERNEL, TraceEventType.NCCL_CALL):
                continue
            identity = event.kernel_args.get('paramorph_identity')
            kind = TaskType.NCCL_GENERIC if event.event_type == TraceEventType.NCCL_CALL else TaskType.CUDA_KERNEL
            if identity is not None:
                operation = identity['operation']
                if event.event_type == TraceEventType.CUDA_KERNEL:
                    kind = {'region_forward': TaskType.FORWARD_COMPUTE,
                        'region_backward': TaskType.BACKWARD_COMPUTE,
                        'optimizer_step': TaskType.OPTIMIZER_STEP}.get(operation, TaskType.CUDA_KERNEL)
                if 'communication_type' in identity:
                    kind = TaskType(identity['communication_type'])
                elif operation == 'pp_p2p_exchange' and event.event_type == TraceEventType.NCCL_CALL:
                    kind = TaskType.PP_P2P_EXCHANGE
            task = Task(event.event_id, event.rank, kind,
                microbatch_id=identity.get('microbatch') if identity else None,
                region_id=identity.get('vp_stage') if identity else None,
                start_time=event.start_ns / 1e6, end_time=event.end_ns / 1e6,
                duration=event.duration_ms, stream_id=event.stream_id,
                metadata=dict(duration_evidence=event.event_id, kernel_name=event.name,
                              identity=identity, correlation_id=event.correlation_id))
            dag.add_task(task)
            streams[event.rank, event.stream_id].append(task)
            if identity and 'message_id' in identity:
                role = identity['role']
                if role in messages[identity['message_id']]:
                    raise ValueError('消息身份重复，通信依赖无法确定')
                messages[identity['message_id']][role] = task
            if event.event_type == TraceEventType.NCCL_CALL and kind == TaskType.NCCL_GENERIC:
                dag.metadata['data_gaps'].append(f'{event.event_id}:缺少实际进程组和通信身份')
        for tasks in streams.values():
            ordered = sorted(tasks, key=lambda task: task.start_time)
            for previous, following in zip(ordered, ordered[1:]):
                dag.add_dependency(previous.task_id, following.task_id)
        for message_id, tasks in messages.items():
            if set(tasks) != {'send', 'recv'}:
                raise ValueError(f'消息{message_id}缺少实际发送或接收身份')
            dag.add_dependency(tasks['send'].task_id, tasks['recv'].task_id)
        for source, target in bundle.metadata.get('cuda_event_dependencies', []):
            dag.add_dependency(source, target)
        dag.metadata['data_gaps'].extend(['完整迭代CPU等待与同步任务', 'CUDA event record/wait依赖'])
        topological_order(dag.tasks)
        return dag

    def topological_sort(self, dag):
        return topological_order(dag.tasks)[0]
