from __future__ import annotations

from dataclasses import dataclass


STATE_SHARDING_VALUES = frozenset({"replicated", "distributed_optimizer", "fsdp"})
RECOMPUTE_VALUES = frozenset({"none", "selective", "prefix", "full"})
ACTIVATION_OFFLOAD_VALUES = frozenset({"none", "cpu_sync", "cpu_overlap", "cpu_prefetch"})
OPTIMIZER_OFFLOAD_VALUES = frozenset({"none", "cpu"})
PRECISION_AWARE_VALUES = frozenset({"enabled", "disabled"})


@dataclass(frozen=True)
class MemoryExecutionPolicy:
    """训练期间实际使用的状态、激活和优化器内存策略。"""

    state_sharding: str = "replicated"
    recompute: str = "none"
    activation_offload: str = "none"
    optimizer_offload: str = "none"
    precision_aware_optimizer: str = "disabled"
    activation_bytes: int = 0
    host_memory_bandwidth_bytes_per_us: float = 0.0
    pcie_bandwidth_bytes_per_us: float = 0.0
    dma_engines: int = 1
    memory_pressure_threshold_bytes: int | None = None

    def __post_init__(self) -> None:
        fields = (
            ("state_sharding", self.state_sharding, STATE_SHARDING_VALUES),
            ("recompute", self.recompute, RECOMPUTE_VALUES),
            ("activation_offload", self.activation_offload, ACTIVATION_OFFLOAD_VALUES),
            ("optimizer_offload", self.optimizer_offload, OPTIMIZER_OFFLOAD_VALUES),
            ("precision_aware_optimizer", self.precision_aware_optimizer, PRECISION_AWARE_VALUES),
        )
        for name, value, allowed in fields:
            if value not in allowed:
                raise ValueError(f"invalid {name}: {value!r}; expected one of {sorted(allowed)}")
        if self.activation_bytes < 0:
            raise ValueError("activation_bytes must be non-negative")
        if self.host_memory_bandwidth_bytes_per_us < 0 or self.pcie_bandwidth_bytes_per_us < 0:
            raise ValueError("memory bandwidth must be non-negative")
        if self.dma_engines < 1:
            raise ValueError("dma_engines must be positive")
        if self.memory_pressure_threshold_bytes is not None and self.memory_pressure_threshold_bytes < 0:
            raise ValueError("memory_pressure_threshold_bytes must be non-negative")

    @property
    def optimizer_offload_allowed(self) -> bool:
        return self.optimizer_offload == "cpu"

    def as_dict(self) -> dict[str, object]:
        return {
            "state_sharding": self.state_sharding,
            "recompute": self.recompute,
            "activation_offload": self.activation_offload,
            "optimizer_offload": self.optimizer_offload,
            "precision_aware_optimizer": self.precision_aware_optimizer,
            "activation_bytes": self.activation_bytes,
            "host_memory_bandwidth_bytes_per_us": self.host_memory_bandwidth_bytes_per_us,
            "pcie_bandwidth_bytes_per_us": self.pcie_bandwidth_bytes_per_us,
            "dma_engines": self.dma_engines,
            "memory_pressure_threshold_bytes": self.memory_pressure_threshold_bytes,
        }
