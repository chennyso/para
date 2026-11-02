"""
SegmentCandidate — one partial plan covering layers [start_region, end_region].

Λ (pending_transitions) is the set of TransitionObligations that have been
created but not yet resolved.  These survive PP/VPP boundaries and must be
resolved before the end of the pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from megatron.core.frontier.state import DistributedState
from megatron.core.frontier.transition import PendingTransition, TransitionDecision
from megatron.core.frontier.ir.memory import MemoryExecutionPolicy


@dataclass
class SegmentCandidate:
    """
    Partial plan for layers [start_region, end_region] inclusive.

    Carries enough information to:
    - compare with other candidates at the same semantic position (dominance)
    - extend to the next region (beam expansion)
    - evaluate cost via the simulator

    pending_transitions = Λ from the paper: obligations not yet resolved.
    """
    start_region: int
    end_region: int

    # Contract name chosen for each layer in [start_region, end_region].
    contract_ids: tuple[str, ...]

    # TransitionDecisions for transitions whose resolve_boundary falls inside
    # this segment.
    internal_transitions: tuple[TransitionDecision, ...]

    # Obligations that started in this segment but will be resolved later.
    pending_transitions: tuple[PendingTransition, ...]

    # Distributed state of the tensor leaving this segment's last layer.
    output_state: DistributedState

    # Performance estimates (from profile database or simulator).
    forward_us: float = 0.0
    backward_us: float = 0.0
    peak_memory_bytes: int = 0

    # NIC / NVLink pressure scores (0.0–1.0, higher = more contention).
    nic_pressure: float = 0.0
    nvlink_pressure: float = 0.0

    # Maximum ready time across all stages covered (for pipeline idle analysis).
    max_ready_time_us: float = 0.0
    stage_ready_times: tuple[float, ...] = ()

    # Max load of any single stage inside this segment.
    max_stage_load_us: float = 0.0

    # Planner bookkeeping — the PP stage index this segment belongs to.
    pp_stage: int = 0
    vp_chunk: int = 0

    # Identifies which resource allocation signature was used (for dominance
    # grouping across candidates that share the same resource mesh).
    resource_signature: tuple = field(default_factory=tuple)
    memory_policy: MemoryExecutionPolicy = field(default_factory=MemoryExecutionPolicy)

    def total_us(self) -> float:
        return self.forward_us + self.backward_us

    def pending_bytes(self) -> int:
        """Sum of unresolved buffer bytes across all pending transitions."""
        return sum(t.spec.source.pending_bytes for t in self.pending_transitions)
