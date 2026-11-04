import hashlib
import json
import math
import statistics
from dataclasses import dataclass

from core_data_models import TaskType


# 成本复用按计算形状、执行组织和物理通信路径分别判断。
COMPUTE_SHAPE_FIELDS = ('tp', 'cp', 'ep', 'etp', 'micro_batch', 'sequence', 'sequence_length',
                        'recompute', 'use_activation_recomputation', 'precision')

EXECUTION_ORGANIZATION_FIELDS = ('pp', 'vpp', 'decoder_chunks', 'group', 'microbatch_group',
                                 'bucket_elements', 'overlap_p2p', 'dp', 'device_mapping')
TOPOLOGY_FIELDS = ('rank_mapping', 'node_count', 'gpus_per_node', 'hardware_fingerprint', 'backend_fingerprint')
PHYSICAL_PATH_FIELDS = frozenset(('pp', 'dp', 'device_mapping'))

INERT_FIELDS = ('run_id', 'port', 'warmup', 'measured', 'master', 'variant',
                'prediction_gate', 'name', 'profile')

_COMPUTE_TASK_TYPES = frozenset((
    TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE, TaskType.RECOMPUTE,
    TaskType.CUDA_KERNEL, TaskType.OPTIMIZER_STEP,
))
_COMMUNICATION_TASK_TYPES = frozenset((
    TaskType.TP_COMM, TaskType.PP_P2P_SEND, TaskType.PP_P2P_RECV, TaskType.PP_P2P_EXCHANGE,
    TaskType.CP_COMM, TaskType.EP_ALL_TO_ALL, TaskType.DP_REDUCE, TaskType.NCCL_GENERIC,
))


class CostDecompositionError(ValueError):
    pass


@dataclass(frozen=True)
class CostReuseAssessment:
    reuse_layer_costs: bool
    reuse_link_costs: bool
    rebuild_schedule: bool
    requires_new_profiling: bool
    changed_compute_shape_fields: tuple
    changed_organization_fields: tuple
    reason: str

    def as_dict(self):
        return dict(reuse_layer_costs=self.reuse_layer_costs,
                    reuse_link_costs=self.reuse_link_costs,
                    rebuild_schedule=self.rebuild_schedule,
                    requires_new_profiling=self.requires_new_profiling,
                    changed_compute_shape_fields=list(self.changed_compute_shape_fields),
                    changed_organization_fields=list(self.changed_organization_fields),
                    reason=self.reason)


def _normalized(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def assess_reuse(parent_request, candidate_request):
    if parent_request is None or candidate_request is None:
        raise CostDecompositionError('成本复用判断需要两个执行配置')
    keys = set(parent_request) | set(candidate_request)
    changed = {key for key in keys
               if _normalized(parent_request.get(key)) != _normalized(candidate_request.get(key))}
    shape = tuple(sorted(changed & set(COMPUTE_SHAPE_FIELDS)))
    organization = tuple(sorted(changed & set(EXECUTION_ORGANIZATION_FIELDS)))
    unknown = tuple(sorted(changed - set(COMPUTE_SHAPE_FIELDS)
                           - set(EXECUTION_ORGANIZATION_FIELDS) - set(INERT_FIELDS) - set(TOPOLOGY_FIELDS)))
    topology = tuple(sorted(changed & set(TOPOLOGY_FIELDS)))
    physical_path = tuple(sorted(changed & PHYSICAL_PATH_FIELDS))
    if topology or unknown:
        return CostReuseAssessment(not bool(shape or unknown), False, True, True, shape, organization,
            '通信路径、后端或者未分类执行条件发生变化，需要查询适用的实际成本：' + str(list(topology + unknown)))
    if shape:
        return CostReuseAssessment(False, False, True, True, shape, organization,
            '计算形状变化会改变原子成本，需要解析估计或新的代表性Profiling')
    if physical_path:
        # PP、DP和设备映射可能改变进程组及跨节点连接，逐层计算形状仍可保持相同。
        return CostReuseAssessment(True, False, True, True, (), organization,
            '物理通信参与者可能变化，必须重新检查链路成本身份：' + str(list(physical_path)))
    if organization:
        # VPP与模块分区改变调度；原子链路费用仅代表相同物理通信条件下的独立操作。
        return CostReuseAssessment(True, True, True, False, (), organization,
            '执行调度变化，允许复用相容的独立原子成本，必须重新计算重叠和资源等待')
    return CostReuseAssessment(True, True, False, False, (), (),
        '执行配置等价，可直接复用已有模板与预测')


def atomic_layer_key(layer_index, layer_signature=None, operation='region', dtype='bfloat16'):
    return hashlib.sha256(json.dumps(dict(
        layer_index=layer_index,
        layer_signature=layer_signature,
        operation=operation,
        dtype=dtype,
    ), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_link_key(source_stage, destination_stage, direction, shape=None, dtype='bfloat16',
                    *, source_physical_rank=None, destination_physical_rank=None,
                    topology_signature=None, communication_backend=None,
                    concurrent_execution_signature=None):
    if (source_physical_rank is None) != (destination_physical_rank is None):
        raise CostDecompositionError('物理通信成本需要同时记录发送与接收Rank')
    if source_physical_rank is not None and (topology_signature is None or communication_backend is None):
        raise CostDecompositionError('物理通信成本需要设备拓扑与通信后端身份')
    return hashlib.sha256(json.dumps(dict(
        source_stage=source_stage,
        destination_stage=destination_stage,
        direction=direction,
        shape=list(shape) if shape is not None else None,
        dtype=dtype,
        source_physical_rank=source_physical_rank,
        destination_physical_rank=destination_physical_rank,
        topology_signature=topology_signature,
        communication_backend=communication_backend,
        concurrent_execution_signature=concurrent_execution_signature,
    ), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def message_bytes(shape, dtype='bfloat16'):
    if shape is None:
        raise CostDecompositionError('消息字节需要算子形状')
    dtype_sizes = {'bfloat16': 2, 'bf16': 2, 'float16': 2, 'fp16': 2,
                   'float32': 4, 'fp32': 4, 'int8': 1, 'uint8': 1}
    if dtype not in dtype_sizes:
        raise CostDecompositionError(f'未知dtype：{dtype}')
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise CostDecompositionError('算子形状维度必须为非负整数')
        elements *= dimension
    return elements * dtype_sizes[dtype]


def aggregate_samples(values):
    if not values:
        raise CostDecompositionError('原子成本聚合需要至少一个样本')
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise CostDecompositionError('原子成本样本必须为有限非负数值')
    median = statistics.median(values)
    dispersion = statistics.pstdev(values)
    confidence = 1 / (1 + dispersion / median) if median else float(dispersion == 0)
    return median, dispersion, confidence, len(values)
