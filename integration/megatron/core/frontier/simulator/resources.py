"""Resource pool for the pipeline simulator."""
from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum, auto


class ResourceKind(Enum):
    GPU_COMPUTE = auto()
    NVLINK = auto()
    NIC = auto()
    PP_STREAM = auto()
    COLLECTIVE_STREAM = auto()
    PCIE = auto()
    HOST_MEMORY = auto()
    DMA = auto()
    BOUNDARY_COMM = auto()
    EP_COMM = auto()


@dataclass
class Resource:
    name: str
    kind: ResourceKind
    # Bandwidth in bytes/us (for communication resources).
    bandwidth_bytes_per_us: float = 0.0
    # Whether concurrent events on this resource cause contention.
    supports_concurrency: bool = False

    # Current occupancy end time (us) — used during schedule simulation.
    available_at_us: float = 0.0
    busy_time_us: float = 0.0


class ResourcePool:
    """Tracks resource availability during simulation."""

    STANDARD_RESOURCES = [
        Resource("gpu_compute", ResourceKind.GPU_COMPUTE),
        Resource("nvlink", ResourceKind.NVLINK, bandwidth_bytes_per_us=600.0, supports_concurrency=True),
        Resource("nic", ResourceKind.NIC, bandwidth_bytes_per_us=25.0, supports_concurrency=True),
        Resource("pp_stream", ResourceKind.PP_STREAM, supports_concurrency=True),
        Resource("collective_stream", ResourceKind.COLLECTIVE_STREAM, supports_concurrency=True),
        Resource("pcie", ResourceKind.PCIE, bandwidth_bytes_per_us=32.0, supports_concurrency=False),
        Resource("host_memory", ResourceKind.HOST_MEMORY, bandwidth_bytes_per_us=80.0, supports_concurrency=False),
        Resource("dma", ResourceKind.DMA, bandwidth_bytes_per_us=32.0, supports_concurrency=False),
        Resource("boundary_comm", ResourceKind.BOUNDARY_COMM, bandwidth_bytes_per_us=25.0, supports_concurrency=False),
        Resource("ep_comm", ResourceKind.EP_COMM, bandwidth_bytes_per_us=25.0, supports_concurrency=False),
        Resource("pp_comm", ResourceKind.PP_STREAM, bandwidth_bytes_per_us=25.0, supports_concurrency=False),
    ]

    def __init__(self):
        self._resources: dict[str, Resource] = {
            r.name: replace(r) for r in self.STANDARD_RESOURCES
        }

    def reset(self):
        for r in self._resources.values():
            r.available_at_us = 0.0
            r.busy_time_us = 0.0

    def earliest_start(self, resource_id: str, ready_at_us: float) -> float:
        r = self._resources[resource_id]
        if r.supports_concurrency:
            return ready_at_us
        return max(r.available_at_us, ready_at_us)

    def reserve(self, resource_id: str, start_us: float, duration_us: float) -> float:
        r = self._resources[resource_id]
        end = start_us + duration_us
        if not r.supports_concurrency:
            r.available_at_us = end
        r.busy_time_us += duration_us
        return end

    def contention_factor(self, resource_id: str) -> float:
        """Ratio of available bandwidth being used (0.0–1.0 range)."""
        r = self._resources[resource_id]
        if r.bandwidth_bytes_per_us == 0:
            return 0.0
        return min(1.0, r.busy_time_us / max(1.0, r.available_at_us))

    def resource_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._resources))
