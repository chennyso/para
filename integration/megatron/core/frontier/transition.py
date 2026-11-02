"""
Transition specification, decision, and analysis objects.

A TransitionSpec describes *what* needs to happen between two adjacent regions
(source state → target state, which layer indices are involved).

A TransitionDecision says *where* the transition will be resolved and *how*
(which lowering to use).  The planner produces decisions; the compiler turns
them into ResolveScheduleNodes.

A TransitionAnalysis bundles spec + mobility window + legal lowerings; it is
the output of the mobility analysis pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megatron.core.frontier.state import DistributedState


@dataclass(frozen=True)
class TransitionSpec:
    """
    Describes a single required layout conversion between two adjacent regions.

    transition_id is unique within a model-instance planning run.
    producer_region / consumer_region are layer indices in the full model.
    source / target describe the DistributedState before and after.
    """
    transition_id: int
    producer_region: int
    consumer_region: int
    source: DistributedState
    target: DistributedState

    @property
    def is_trivial(self) -> bool:
        """True when source and target layouts are identical — no-op."""
        return self.source == self.target

    @property
    def communication_bytes(self) -> int:
        """Rough estimate of bytes transferred; used for quick scoring."""
        return self.source.pending_bytes


@dataclass(frozen=True)
class TransitionDecision:
    """
    Planner output for one TransitionSpec.

    resolve_boundary: the layer index *after* which the resolve node fires.
    lowering_id: which TransitionLowering implementation to use.
    """
    transition_id: int
    resolve_boundary: int
    lowering_id: str

    # Estimated cost of executing this transition at the chosen boundary,
    # as computed by the simulator.  None until the simulator fills it in.
    estimated_us: float | None = None


@dataclass
class TransitionAnalysis:
    """
    Full analysis result for one TransitionSpec.

    mobility_window: tuple of layer indices where this transition may legally
        be resolved (computed by mobility.compute_mobility_window).
    legal_lowerings: which lowering implementations are applicable.
    """
    spec: TransitionSpec
    mobility_window: tuple[int, ...]
    legal_lowerings: tuple[str, ...]

    # Filled in by the planner.
    decision: TransitionDecision | None = None

    @property
    def is_movable(self) -> bool:
        """True when more than one resolve position exists."""
        return len(self.mobility_window) > 1

    @property
    def mobility_span(self) -> int:
        """Number of layers the transition can be moved across."""
        return len(self.mobility_window)


@dataclass
class PendingTransition:
    """
    A transition that has been created but not yet resolved during planning.
    Carried in SegmentCandidate.pending_transitions — this is Λ from the paper.
    """
    spec: TransitionSpec
    remaining_window: tuple[int, ...]  # shrinks as we advance through regions
    chosen_lowering: str | None = None
