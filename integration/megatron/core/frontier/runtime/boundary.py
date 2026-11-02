"""
BoundarySchema — static metadata for one transition boundary.

The schema is computed once at plan-compile time and distributed to all ranks
before training starts.  During training, the hot path reads only pre-computed
arrays from the schema; no Python metadata is created per microbatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from megatron.core.frontier.state import DistributedState
from megatron.core.frontier.lowering.reduce_handoff import SliceInterval


@dataclass(frozen=True)
class BoundarySchema:
    """
    Immutable descriptor for one transition boundary in the compiled plan.

    Everything needed by a TransitionLowering to execute the data-plane
    operation is stored here, so the hot path never queries the planner.
    """
    boundary_id: int

    source_state: DistributedState
    target_state: DistributedState

    lowering_id: str

    tensor_shape: tuple[int, ...]

    # Ranks involved on the sending side and the receiving side.
    source_ranks: tuple[int, ...]
    destination_ranks: tuple[int, ...]

    # Pre-computed per-destination slice intervals (used by ReduceOnHandoff).
    destination_intervals: tuple[SliceInterval, ...] = field(default_factory=tuple)

    # Which side of the PP boundary this rank is on.
    is_source_side: bool = True

    # Pre-allocated buffer slots.  These are integer indices into a buffer pool
    # managed by runtime/resolve.py — not raw tensors, so the schema is
    # picklable and safely shared before CUDA device init.
    buffer_slot: int = 0

    # Process group identifier (string key into ProcessGroupCollection).
    pg_key: str | None = None

    # Own slice interval (source side only; used by ReduceOnHandoff).
    own_interval: SliceInterval | None = None

    def for_microbatch(self, slot: int) -> "BoundarySchema":
        """Return a schema view for a specific microbatch buffer slot."""
        # Slots are pre-allocated; this just rebinds the slot index.
        return BoundarySchema(
            boundary_id=self.boundary_id,
            source_state=self.source_state,
            target_state=self.target_state,
            lowering_id=self.lowering_id,
            tensor_shape=self.tensor_shape,
            source_ranks=self.source_ranks,
            destination_ranks=self.destination_ranks,
            destination_intervals=self.destination_intervals,
            is_source_side=self.is_source_side,
            buffer_slot=slot,
            pg_key=self.pg_key,
            own_interval=self.own_interval,
        )
