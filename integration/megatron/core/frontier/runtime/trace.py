"""
FrontierTrace — lightweight per-step execution trace for the agent evidence
collector and offline Nsight-based profiling.

The trace is opt-in; it adds per-boundary timestamp pairs to a ring buffer.
The agent evidence module reads the ring buffer between steps to build
TransitionEvidence summaries (exposed_us, bytes, etc.).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class TraceEvent:
    boundary_id: int
    direction: str           # "fwd" or "bwd"
    microbatch_slot: int
    timestamp_ns: int = field(default_factory=lambda: time.perf_counter_ns())


class FrontierTrace:
    """
    Ring buffer of trace events.  Thread-unsafe — intended for use from the
    main training thread only.
    """

    def __init__(self, maxlen: int = 4096):
        self._events: Deque[TraceEvent] = deque(maxlen=maxlen)

    def record_forward(self, boundary_id: int, microbatch_slot: int) -> None:
        self._events.append(TraceEvent(
            boundary_id=boundary_id,
            direction="fwd",
            microbatch_slot=microbatch_slot,
        ))

    def record_backward(self, boundary_id: int, microbatch_slot: int) -> None:
        self._events.append(TraceEvent(
            boundary_id=boundary_id,
            direction="bwd",
            microbatch_slot=microbatch_slot,
        ))

    def drain(self) -> list[TraceEvent]:
        """Return all buffered events and clear the buffer."""
        events = list(self._events)
        self._events.clear()
        return events

    def exposed_us_for_boundary(self, boundary_id: int) -> float:
        """
        Compute the mean time (µs) between the forward resolve and the
        subsequent backward resolve for a given boundary.  This approximates
        how long the resolved buffer must remain live — the "exposed" time.
        """
        fwd_times: list[int] = []
        bwd_times: list[int] = []
        for e in self._events:
            if e.boundary_id != boundary_id:
                continue
            if e.direction == "fwd":
                fwd_times.append(e.timestamp_ns)
            else:
                bwd_times.append(e.timestamp_ns)

        if not fwd_times or not bwd_times:
            return 0.0

        pairs = min(len(fwd_times), len(bwd_times))
        total_ns = sum(
            bwd_times[i] - fwd_times[i] for i in range(pairs)
        )
        return (total_ns / pairs) / 1_000.0  # ns → µs
