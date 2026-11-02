"""Simulation events for the pipeline schedule."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class EventKind(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    TRANSITION_FWD = auto()
    TRANSITION_BWD = auto()
    PP_SEND = auto()
    PP_RECV = auto()
    OPTIMIZER = auto()
    COLLECTIVE = auto()
    RECOMPUTE = auto()
    ACTIVATION_D2H = auto()
    ACTIVATION_H2D = auto()


@dataclass
class SimEvent:
    event_id: int
    kind: EventKind
    stage: int
    vp_chunk: int
    microbatch: int
    duration_us: float
    start_us: float = 0.0
    dependencies: tuple[int, ...] = field(default_factory=tuple)
    resource_id: str = "gpu_compute"
    memory_delta: int = 0  # bytes; positive = alloc, negative = free

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us
