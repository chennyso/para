from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LinkClass(Enum):
    LOCAL = "LOCAL"
    PIX = "PIX"
    PXB = "PXB"
    PHB = "PHB"
    SYS = "SYS"
    NET = "NET"


class MappingTemplate(Enum):
    DEFAULT_CONTIGUOUS = "DEFAULT_CONTIGUOUS"
    TP_LOCAL = "TP_LOCAL"
    EP_LOCAL = "EP_LOCAL"
    PP_INTER_NODE = "PP_INTER_NODE"
    NIC_AWARE_PP = "NIC_AWARE_PP"
    DP_CROSS_NODE = "DP_CROSS_NODE"


@dataclass(frozen=True)
class GPUDevice:
    node: str
    gpu_id: int
    memory_bytes: int
    numa_id: int | None = None
    pci_bus: str | None = None
    nic_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.gpu_id < 0 or self.memory_bytes <= 0:
            raise ValueError("GPU identity and memory must be positive")

    @property
    def key(self) -> tuple[str, int]:
        return self.node, self.gpu_id


@dataclass(frozen=True)
class NICDevice:
    node: str
    name: str
    numa_id: int | None = None
    link_speed_gbps: float | None = None


@dataclass(frozen=True)
class LinkMeasurement:
    source: tuple[str, int]
    destination: tuple[str, int]
    collective: str
    payload_bytes: int
    path_class: LinkClass
    median_us: float
    p10_us: float
    p90_us: float
    effective_gbps: float

    def __post_init__(self) -> None:
        if self.payload_bytes <= 0 or self.median_us <= 0:
            raise ValueError("link measurement payload and median must be positive")
        if self.p10_us <= 0 or self.p90_us < self.p10_us or self.effective_gbps < 0:
            raise ValueError("invalid link measurement quantiles")


@dataclass(frozen=True)
class NodeTopology:
    name: str
    gpus: tuple[GPUDevice, ...]
    nics: tuple[NICDevice, ...] = ()

    def __post_init__(self) -> None:
        if not self.gpus:
            raise ValueError("a node must contain at least one GPU")
        if any(gpu.node != self.name for gpu in self.gpus):
            raise ValueError("GPU node names must match the containing node")


@dataclass(frozen=True)
class HardwareTopology:
    nodes: tuple[NodeTopology, ...]
    gpu_links: tuple[LinkMeasurement, ...] = ()
    nic_links: tuple[LinkMeasurement, ...] = ()

    def __post_init__(self) -> None:
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("node names must be unique")
        gpu_keys = [gpu.key for node in self.nodes for gpu in node.gpus]
        if len(gpu_keys) != len(set(gpu_keys)):
            raise ValueError("GPU identities must be unique")

    @property
    def gpus(self) -> tuple[GPUDevice, ...]:
        return tuple(gpu for node in self.nodes for gpu in node.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def get_gpu(self, node: str, gpu_id: int) -> GPUDevice:
        for gpu in self.gpus:
            if gpu.node == node and gpu.gpu_id == gpu_id:
                return gpu
        raise KeyError((node, gpu_id))

    def measure(self, source: tuple[str, int], destination: tuple[str, int], collective: str, payload_bytes: int) -> LinkMeasurement:
        matches = [
            item for item in self.gpu_links + self.nic_links
            if item.source == source and item.destination == destination
            and item.collective == collective and item.payload_bytes == payload_bytes
        ]
        if not matches:
            raise KeyError((source, destination, collective, payload_bytes))
        return matches[0]

    def groups(self, template: MappingTemplate, group_size: int) -> tuple[tuple[tuple[str, int], ...], ...]:
        if group_size < 1:
            raise ValueError("group size must be positive")
        ordered = list(self.gpus)
        if template is MappingTemplate.TP_LOCAL:
            ordered.sort(key=lambda gpu: (gpu.node, gpu.numa_id if gpu.numa_id is not None else 10**9, gpu.gpu_id))
        elif template is MappingTemplate.EP_LOCAL:
            ordered.sort(key=lambda gpu: (gpu.node, gpu.gpu_id))
        elif template in {MappingTemplate.PP_INTER_NODE, MappingTemplate.NIC_AWARE_PP, MappingTemplate.DP_CROSS_NODE}:
            ordered.sort(key=lambda gpu: (gpu.gpu_id, gpu.node))
        else:
            ordered.sort(key=lambda gpu: (gpu.node, gpu.gpu_id))
        return tuple(tuple(gpu.key for gpu in ordered[index:index + group_size]) for index in range(0, len(ordered), group_size))

    def validate_mapping(self, mapping: tuple[tuple[str, int], ...]) -> None:
        if len(mapping) != self.gpu_count:
            raise ValueError("mapping must contain every GPU exactly once")
        if len(set(mapping)) != len(mapping):
            raise ValueError("mapping contains duplicate GPUs")
        known = {gpu.key for gpu in self.gpus}
        if set(mapping) != known:
            raise ValueError("mapping contains an unknown or missing GPU")
