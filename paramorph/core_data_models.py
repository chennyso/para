from dataclasses import dataclass, field, replace, asdict
from typing import List, Dict, Literal, Optional, Set, Tuple
from enum import Enum
import json
from datetime import datetime


class LayerType(Enum):
    ATTENTION = "attention"
    HYBRID = "attention_mamba_composite"
    MAMBA = "mamba"
    MLP = "mlp"
    MOE = "moe"
    EMBEDDING = "embedding"
    OUTPUT = "output"
    NORM = "norm"


class TaskType(Enum):
    PIPELINE_UPDATE = "pipeline_update"
    ITERATION_OVERHEAD = "iteration_overhead"
    CUDA_KERNEL = "cuda_kernel"
    NCCL_GENERIC = "nccl_generic"
    FORWARD_COMPUTE = "forward_compute"
    BACKWARD_COMPUTE = "backward_compute"
    TP_COMM = "tp_comm"
    PP_P2P_SEND = "pp_p2p_send"
    PP_P2P_RECV = "pp_p2p_recv"
    PP_P2P_EXCHANGE = "pp_p2p_exchange"
    CP_COMM = "cp_comm"
    EP_ALL_TO_ALL = "ep_all_to_all"
    DP_REDUCE = "dp_reduce"
    OPTIMIZER_STEP = "optimizer_step"
    OPTIMIZER_PREPARE = "optimizer_prepare_grads"
    GRADIENT_NORM = "optimizer_local_grad_norm"
    GRADIENT_NORM_FINALIZE = "optimizer_local_grad_norm_finalize"
    GRADIENT_CLIP = "optimizer_local_grad_clip"
    CUDA_SYNC = "cuda_sync"
    OFFLOAD_D2H = "offload_d2h"
    PREFETCH_H2D = "prefetch_h2d"
    RECOMPUTE = "recompute"
    ACTIVATION_RELEASE = "activation_release"
    OFFLOAD_WAIT = "offload_wait"
    FSDP_UNSHARD = "fsdp_unshard"
    FSDP_PREFETCH = "fsdp_prefetch"
    FSDP_RESHARD = "fsdp_reshard"
    FSDP_GRAD_REDUCE_SCATTER = "fsdp_grad_reduce_scatter"
    FSDP_OUTER_GRAD_REDUCE = "fsdp_outer_grad_reduce"
    FSDP_OUTER_PARAM_SYNC = "fsdp_outer_param_sync"


@dataclass
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    layer_types: List[LayerType]
    layer_order: List[int]
    num_experts: Optional[int] = None
    expert_interval: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    head_dimension: Optional[int] = None
    num_query_groups: Optional[int] = None
    moe_top_k: Optional[int] = None
    shared_expert_size: Optional[int] = None
    module_kinds: Optional[List[str]] = None
    finalize_grad_semantics: Optional[Dict] = None

    def __post_init__(self):
        if min(self.num_layers, self.hidden_size, self.num_attention_heads) < 1:
            raise ValueError('模型层数、Hidden维度与Attention heads必须为正数')
        if len(self.layer_types) != self.num_layers or sorted(self.layer_order) != list(range(self.num_layers)):
            raise ValueError('模型描述必须按实际顺序完整覆盖全部层')
        if any(not isinstance(kind, LayerType) for kind in self.layer_types):
            raise ValueError('模型层类型必须来自LayerType')
        if self.module_kinds is not None and len(self.module_kinds) != self.num_layers:
            raise ValueError('模块类型必须按实际模型顺序覆盖全部层')
        if self.finalize_grad_semantics is not None and not isinstance(
                self.finalize_grad_semantics, dict):
            raise ValueError('FinalizeModelGrads语义必须使用明确字典')

    def get_layer_type(self, layer_idx: int) -> LayerType:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range")
        return self.layer_types[layer_idx]


@dataclass
class HardwareTopology:
    num_gpus: int
    num_nodes: int
    gpus_per_node: int
    memory_per_gpu_gb: float
    nvlink_bandwidth_gbps: Optional[float] = None
    pcie_bandwidth_gbps: Optional[float] = None
    ib_bandwidth_gbps: Optional[float] = None
    devices: List[Dict] = field(default_factory=list)
    network_paths: List[Dict] = field(default_factory=list)
    software: Dict = field(default_factory=dict)
    environment_per_node: Dict = field(default_factory=dict)

    def validate(self) -> bool:
        return self.num_gpus == self.num_nodes * self.gpus_per_node


@dataclass
class MemoryPolicy:
    optimizer_offload_fraction: float = 0.0
    activation_offload_modules: Tuple[str, ...] = ()
    activation_offload_fraction: float = 0.0
    recompute_granularity: Optional[Literal['full', 'selective']] = None
    recompute_modules: Tuple[str, ...] = ()
    max_inflight_microbatches: Optional[int] = None
    cpu_memory_budget_bytes: Optional[int] = None
    prefetch_distance: int = 0
    reuse_activation_slots: bool = True

    def __post_init__(self):
        if not 0 <= self.optimizer_offload_fraction <= 1 or not 0 <= self.activation_offload_fraction <= 1:
            raise ValueError('Offload比例必须位于零至一之间')
        if self.activation_offload_fraction and not self.activation_offload_modules:
            raise ValueError('Activation Offload必须指定实际模块')
        if self.recompute_granularity not in (None, 'full', 'selective'):
            raise ValueError('Recomputation策略不受当前执行接口支持')
        if self.recompute_granularity == 'selective' and not self.recompute_modules:
            raise ValueError('选择性Recomputation必须指定实际模块')
        if self.prefetch_distance < 0 or (self.max_inflight_microbatches is not None
                and self.max_inflight_microbatches < 1):
            raise ValueError('Prefetch距离或者在途Microbatch限制不合法')
        if self.cpu_memory_budget_bytes is not None and self.cpu_memory_budget_bytes <= 0:
            raise ValueError('CPU显存管理需要实际可用内存容量')


@dataclass
class ParallelConfig:
    tp: int
    pp: int
    dp: int
    cp: int = 1
    ep: int = 1
    vpp: int = 1
    global_batch_size: int = 16
    micro_batch_size: int = 1
    sequence_length: int = 4096
    use_sequence_parallel: bool = False
    use_activation_recomputation: bool = False
    etp: Optional[int] = None
    optimizer_offload: bool = False
    microbatch_group: int = 1
    device_mapping: Tuple[int, ...] = ()
    execution_implementation: Dict = field(default_factory=dict)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)

    def __post_init__(self):
        for name in ('tp', 'pp', 'dp', 'cp', 'ep', 'vpp', 'global_batch_size',
                     'micro_batch_size', 'sequence_length', 'microbatch_group'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name}必须为正整数')
        if self.global_batch_size % (self.dp * self.micro_batch_size):
            raise ValueError('global_batch_size必须被dp×micro_batch_size整除')
        if self.etp is None:
            self.etp = self.tp
        if self.etp <= 0 or (self.tp * self.cp * self.dp) % (self.etp * self.ep):
            raise ValueError('Expert进程组不能整除当前训练设备集合')
        if self.device_mapping and sorted(self.device_mapping) != list(range(self.world_size)):
            raise ValueError('device_mapping必须覆盖全部设备一次')

    def normalized_key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

    def with_updates(self, **updates):
        return replace(self, **updates)

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp * self.cp

    @property
    def num_microbatches(self) -> int:
        return self.global_batch_size // (self.dp * self.micro_batch_size)

    @property
    def total_vp_stages(self) -> int:
        return self.pp * self.vpp

    def to_dict(self) -> Dict:
        return asdict(self) | {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "cp": self.cp,
            "ep": self.ep,
            "vpp": self.vpp,
            "gbs": self.global_batch_size,
            "mbs": self.micro_batch_size,
            "seq_len": self.sequence_length,
            "world": self.world_size,
            "num_microbatches": self.num_microbatches,
        }


@dataclass
class Region:
    region_id: int
    pp_rank: int
    vp_rank: int
    layer_start: int
    layer_end: int
    estimated_forward_time_ms: Optional[float] = None
    estimated_backward_time_ms: Optional[float] = None
    estimated_memory_mb: Optional[float] = None

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def contains_layer(self, layer_idx: int) -> bool:
        return self.layer_start <= layer_idx < self.layer_end


@dataclass
class Task:
    task_id: str
    rank: int
    task_type: TaskType
    microbatch_id: Optional[int] = None
    region_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    stream_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    activation_delta_bytes: int = 0
    workspace_bytes: int = 0
    resource_requirements: Tuple[str, ...] = ()

    @property
    def is_compute(self) -> bool:
        return self.task_type in [TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE,
            TaskType.CUDA_KERNEL, TaskType.OPTIMIZER_PREPARE,
            TaskType.GRADIENT_NORM, TaskType.GRADIENT_NORM_FINALIZE, TaskType.GRADIENT_CLIP]

    @property
    def is_communication(self) -> bool:
        return self.task_type in [
            TaskType.TP_COMM,
            TaskType.PP_P2P_SEND,
            TaskType.PP_P2P_RECV,
            TaskType.PP_P2P_EXCHANGE,
            TaskType.CP_COMM,
            TaskType.EP_ALL_TO_ALL,
            TaskType.DP_REDUCE,
            TaskType.NCCL_GENERIC,
            TaskType.FSDP_UNSHARD,
            TaskType.FSDP_PREFETCH,
            TaskType.FSDP_GRAD_REDUCE_SCATTER,
            TaskType.FSDP_OUTER_GRAD_REDUCE,
            TaskType.FSDP_OUTER_PARAM_SYNC,
            TaskType.OFFLOAD_D2H,
            TaskType.PREFETCH_H2D,
        ]


@dataclass
class ExecutionDAG:
    tasks: Dict[str, Task]
    edges: List[Tuple[str, str]]
    metadata: Dict = field(default_factory=dict)

    def add_task(self, task: Task):
        if task.task_id in self.tasks:
            raise ValueError(f'重复Task身份：{task.task_id}')
        self.tasks[task.task_id] = task

    def add_dependency(self, from_task_id: str, to_task_id: str):
        if from_task_id not in self.tasks or to_task_id not in self.tasks:
            raise ValueError("Both tasks must exist in DAG")
        if (from_task_id, to_task_id) not in self.edges:
            self.edges.append((from_task_id, to_task_id))
        if from_task_id not in self.tasks[to_task_id].dependencies:
            self.tasks[to_task_id].dependencies.append(from_task_id)

    def get_predecessors(self, task_id: str) -> List[str]:
        return [src for src, dst in self.edges if dst == task_id]

    def get_successors(self, task_id: str) -> List[str]:
        return [dst for src, dst in self.edges if src == task_id]

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def visit(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for successor in self.get_successors(node):
                if visit(successor):
                    return True

            rec_stack.remove(node)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if visit(task_id):
                    return True

        return False


@dataclass
class ExecutionPlan:
    parallel_config: ParallelConfig
    regions: List[Region]
    rank_schedules: Dict[int, List[str]]
    tasks: Dict[str, Task] = field(default_factory=dict)
    task_times: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    concurrent_pairs: List[Tuple[str, str]] = field(default_factory=list)
    persistent_bytes: Dict[int, int] = field(default_factory=dict)
    cost_model_version: str = ''
    candidate_id: str = ''
    runtime_table: List[Tuple[int, int]] = field(default_factory=list)

    def get_rank_tasks(self, rank: int) -> List[str]:
        return self.rank_schedules.get(rank, [])


@dataclass
class Measurement:
    name: str
    value: float
    unit: str
    source: str
    is_predicted: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "is_predicted": self.is_predicted,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Bottleneck:
    bottleneck_type: str
    severity: float
    affected_ranks: List[int]
    description: str
    evidence_ids: List[str]
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "type": self.bottleneck_type,
            "severity": self.severity,
            "affected_ranks": self.affected_ranks,
            "description": self.description,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
        }


@dataclass
class CriticReport:
    baseline_config: ParallelConfig
    critical_path_tasks: List[str]
    critical_path_length_ms: float
    bottlenecks: List[Bottleneck]
    measurements: List[Measurement]
    data_gaps: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    layer_forward_costs_ms: Dict[int, float] = field(default_factory=dict)
    agent_decision: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "baseline_config": self.baseline_config.to_dict(),
            "critical_path_tasks": self.critical_path_tasks,
            "critical_path_length_ms": self.critical_path_length_ms,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "measurements": [m.to_dict() for m in self.measurements],
            "data_gaps": self.data_gaps,
            "timestamp": self.timestamp.isoformat(),
            "layer_forward_costs_ms": self.layer_forward_costs_ms,
            "agent_decision": self.agent_decision,
        }


@dataclass
class Candidate:
    candidate_id: str
    parent_id: Optional[str]
    parallel_config: ParallelConfig
    region_layout: Optional[List[Region]]
    optimization_actions: List[str]
    predicted_speedup: Optional[float] = None
    predicted_critical_path_ms: Optional[float] = None
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    actual_throughput: Optional[float] = None
    actual_iteration_time: Optional[float] = None
    execution_status: str = "pending"
    execution_plan: Optional[ExecutionPlan] = None
    execution_plan_sha256: Optional[str] = None
    transformation_record: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "config": self.parallel_config.to_dict(),
            "region_layout": [
                {
                    "region_id": r.region_id,
                    "pp_rank": r.pp_rank,
                    "vp_rank": r.vp_rank,
                    "layers": f"[{r.layer_start}:{r.layer_end})",
                    "num_layers": r.num_layers,
                }
                for r in (self.region_layout or [])
            ],
            "optimization_actions": self.optimization_actions,
            "predicted_speedup": self.predicted_speedup,
            "predicted_critical_path_ms": self.predicted_critical_path_ms,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "actual_throughput": self.actual_throughput,
            "actual_iteration_time": self.actual_iteration_time,
            "execution_status": self.execution_status,
            "execution_plan_sha256": self.execution_plan_sha256,
            "transformation_record": self.transformation_record,
            "runtime_executor": 'paramorph_task_dag' if self.execution_plan is not None else 'megatron_native',
        }


@dataclass
class SearchHistory:
    candidates: List[Candidate] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    best_throughput: Optional[float] = None
    total_gpu_hours: float = 0.0

    def add_candidate(self, candidate: Candidate):
        self.candidates.append(candidate)
        if candidate.actual_throughput and (
            self.best_throughput is None or candidate.actual_throughput > self.best_throughput
        ):
            self.best_throughput = candidate.actual_throughput
            self.best_candidate_id = candidate.candidate_id

    def to_dict(self) -> Dict:
        return {
            "num_candidates": len(self.candidates),
            "best_candidate_id": self.best_candidate_id,
            "best_throughput": self.best_throughput,
            "total_gpu_hours": self.total_gpu_hours,
            "candidates": [c.to_dict() for c in self.candidates],
        }
