"""Event-driven memory tracker for the simulator."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from megatron.core.frontier.simulator.event import SimEvent


@dataclass
class MemorySnapshot:
    time_us: float
    allocated_bytes: int


class MemoryTracker:
    """
    Tracks GPU memory usage over time using the event timeline.

    Accounts for:
    - Parameter memory (constant)
    - Optimizer state (constant)
    - Activation memory (produced at fwd, freed after bwd)
    - Transition buffer memory (produced at fwd, freed after resolve)
    - Temporary memory (allocated and freed within one event)
    """

    def __init__(
        self,
        param_bytes: int = 0,
        optimizer_bytes: int = 0,
    ):
        self.param_bytes = param_bytes
        self.optimizer_bytes = optimizer_bytes
        self._snapshots: list[MemorySnapshot] = []

    def simulate(self, events: Sequence[SimEvent]) -> int:
        """
        Replay events in timeline order and return peak allocated bytes.
        """
        sorted_events = sorted(events, key=lambda e: e.start_us)
        current = self.param_bytes + self.optimizer_bytes
        peak = current

        for ev in sorted_events:
            current += ev.memory_delta
            if current > peak:
                peak = current
            self._snapshots.append(MemorySnapshot(
                time_us=ev.start_us,
                allocated_bytes=current,
            ))

        return peak

    def peak_bytes(self) -> int:
        if not self._snapshots:
            return self.param_bytes + self.optimizer_bytes
        return max(s.allocated_bytes for s in self._snapshots)

    def snapshots(self) -> list[MemorySnapshot]:
        return list(self._snapshots)
