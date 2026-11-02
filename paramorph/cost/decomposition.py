"""参数化成本分解：把"原子成本"与"执行组织"分离。

参考 Primus Projection 的建模方式。核心思想：

- **原子成本（atomic）** 只依赖实际影响该成本项的条件，与它被如何调度无关：
  - 层计算成本 key = (层索引/层类型, 算子, 形状, dtype, 精度)
  - P2P 通信成本 key = (消息字节桶, 链路, 方向)
  这些在 PP/VPP/Region/DP 变化时保持不变，因此可以复用实测数据。

- **执行组织（execution organization）** 决定原子成本如何被拼接成一次迭代：
  - pp / vpp / decoder_chunks / group / bucket_elements / overlap_p2p
  这些变化需要重建 schedule 与依赖，但不需要重新采集逐层成本。

- **计算形状（compute shape）** 变化会改变原子成本本身，需要解析模型或新 profiling：
  - tp / cp / ep / micro_batch / recompute / sequence_length

这个模块只做分类与成本查询，不做调度；调度由 TaskGraphBuilder + ScheduleCostEvaluator 负责。
"""
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field

from core_data_models import TaskType


# 改变这些字段会改变"原子计算形状"，逐层成本不可直接复用。
COMPUTE_SHAPE_FIELDS = ('tp', 'cp', 'ep', 'etp', 'micro_batch', 'sequence', 'sequence_length',
                        'recompute', 'use_activation_recomputation', 'precision')

# 改变这些字段只影响"执行组织"，逐层原子成本可复用，但需要重建 schedule。
EXECUTION_ORGANIZATION_FIELDS = ('pp', 'vpp', 'decoder_chunks', 'group', 'microbatch_group',
                                 'bucket_elements', 'overlap_p2p', 'dp', 'device_mapping')
TOPOLOGY_FIELDS = ('rank_mapping', 'node_count', 'gpus_per_node', 'hardware_fingerprint', 'backend_fingerprint')

# 这些字段既不改变原子成本也不改变执行组织（对成本无直接影响）。
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
    """两个执行配置之间，各类成本的可复用性判断。"""
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
    """判断从 parent 到 candidate 的原子成本复用性。

    - 只改 execution organization：复用逐层与链路成本，重建 schedule。
    - 改了 compute shape：原子成本变化，需要解析模型或新 profiling。
    """
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
    if topology or unknown:
        return CostReuseAssessment(not bool(shape or unknown), False, True, True, shape, organization,
            '通信路径、后端或者未分类执行条件发生变化，需要查询适用的实际成本：' + str(list(topology + unknown)))
    if shape:
        return CostReuseAssessment(False, False, True, True, shape, organization,
            '计算形状变化会改变原子成本，需要解析估计或新的代表性 Profiling'
            + (f'；另有未分类字段 {list(unknown)}' if unknown else ''))
    if organization:
        return CostReuseAssessment(True, True, True, False, (), organization,
            '仅执行组织变化：逐层与链路原子成本可复用，需要重建 schedule 与依赖'
            + (f'；另有未分类字段 {list(unknown)}' if unknown else ''))
    return CostReuseAssessment(True, True, False, False, (), (),
        '执行配置等价，可直接复用已有模板与预测')


def atomic_layer_key(layer_index, layer_signature=None, operation='region', dtype='bfloat16'):
    """逐层计算成本的复用键：只看层身份与算子形状，与调度无关。"""
    return hashlib.sha256(json.dumps(dict(
        layer_index=layer_index,
        layer_signature=layer_signature,
        operation=operation,
        dtype=dtype,
    ), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_link_key(source_stage, destination_stage, direction, shape=None, dtype='bfloat16'):
    """P2P 通信成本的复用键：只看消息形状与连接方向，与实际 pp/vpp 取值无关。"""
    return hashlib.sha256(json.dumps(dict(
        source_stage=source_stage,
        destination_stage=destination_stage,
        direction=direction,
        shape=list(shape) if shape is not None else None,
        dtype=dtype,
    ), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def message_bytes(shape, dtype='bfloat16'):
    """按 shape 与 dtype 计算消息字节数。"""
    if shape is None:
        raise CostDecompositionError('消息字节需要算子形状')
    dtype_sizes = {'bfloat16': 2, 'bf16': 2, 'float16': 2, 'fp16': 2,
                   'float32': 4, 'fp32': 4, 'int8': 1, 'uint8': 1}
    if dtype not in dtype_sizes:
        raise CostDecompositionError(f'未知 dtype：{dtype}')
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise CostDecompositionError('算子形状维度必须为非负整数')
        elements *= dimension
    return elements * dtype_sizes[dtype]


def aggregate_samples(values):
    """把同一原子键下的多次测量聚合为 (中位数, 离散度, 置信度, 样本数)。"""
    if not values:
        raise CostDecompositionError('原子成本聚合需要至少一个样本')
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise CostDecompositionError('原子成本样本必须为有限非负数值')
    median = statistics.median(values)
    dispersion = statistics.pstdev(values)
    confidence = 1 / (1 + dispersion / median) if median else float(dispersion == 0)
    return median, dispersion, confidence, len(values)
