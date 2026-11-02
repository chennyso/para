"""
Distributed-state representation used throughout Frontier.

We define our own minimal placement algebra rather than depending on DTensor
so that the hot path carries no torch.distributed.tensor overhead.  The
concepts (Shard/Replicate/Partial) are borrowed from the DTensor vocabulary
but the objects here are plain frozen dataclasses — no actual tensor wrapping.

First-version placement kinds: Shard(dim), Replicate, Partial(sum).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PlacementKind(Enum):
    SHARD = auto()
    REPLICATE = auto()
    PARTIAL = auto()  # accumulation pending — sum is the only supported op now


class ObligationKind(Enum):
    REDUCE = auto()          # all-reduce or reduce-scatter still needed
    REPARTITION = auto()     # shard dimension or mesh changes
    LAYOUT_CONVERT = auto()  # e.g. contiguous ↔ zigzag CP layout


@dataclass(frozen=True)
class Placement:
    kind: PlacementKind
    dim: int | None = None          # meaningful for SHARD only
    reduce_op: str | None = None    # meaningful for PARTIAL only; "sum" for now

    def __post_init__(self):
        if self.kind is PlacementKind.SHARD and self.dim is None:
            raise ValueError("Shard placement requires a dim")
        if self.kind is PlacementKind.PARTIAL and self.reduce_op is None:
            # default to sum
            object.__setattr__(self, "reduce_op", "sum")

    def is_resolved(self) -> bool:
        """True when no further collective is needed to produce a local shard."""
        return self.kind in (PlacementKind.SHARD, PlacementKind.REPLICATE)


@dataclass(frozen=True)
class TransitionObligation:
    """Describes a pending layout conversion that has not yet been executed."""
    kind: ObligationKind
    target: Placement
    # The process-group name (mesh axis) responsible for the operation.
    contributor_group: str | None = None


@dataclass(frozen=True)
class DistributedState:
    """
    Describes the distributed layout of a tensor at a particular point in the
    computation graph.

    `obligation` is non-None when this tensor is *unresolved* — its current
    local data is a partial result that still needs a collective to become the
    intended target placement.  A resolved tensor has obligation=None.
    """
    placement: Placement
    mesh_id: str
    logical_shape: tuple[int, ...]
    obligation: TransitionObligation | None = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.dtype not in {"bfloat16", "float16", "float32", "float64", "int32", "int64", "bool"}:
            raise ValueError("unsupported distributed-state dtype")
        if not self.logical_shape or any(dimension < 1 for dimension in self.logical_shape):
            raise ValueError("distributed state requires positive logical dimensions")

    @property
    def is_resolved(self) -> bool:
        return self.obligation is None

    @property
    def pending_bytes(self) -> int:
        """Bytes held in the unresolved buffer (element count × 2 for bf16)."""
        if not self.logical_shape:
            return 0
        count = 1
        for d in self.logical_shape:
            count *= d
        item_size = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8,
                     "int32": 4, "int64": 8, "bool": 1}[self.dtype]
        return count * item_size

    def with_obligation(self, ob: TransitionObligation) -> "DistributedState":
        return DistributedState(
            placement=self.placement,
            mesh_id=self.mesh_id,
            logical_shape=self.logical_shape,
            obligation=ob,
            dtype=self.dtype,
        )

    def resolved(self, target_placement: Placement) -> "DistributedState":
        """Return a copy where the obligation has been discharged."""
        return DistributedState(
            placement=target_placement,
            mesh_id=self.mesh_id,
            logical_shape=self.logical_shape,
            obligation=None,
            dtype=self.dtype,
        )

    def signature(self) -> tuple:
        """Hashable summary for use in SemanticKey."""
        ob_sig = (
            None if self.obligation is None
            else (self.obligation.kind, self.obligation.target.kind,
                  self.obligation.target.dim, self.obligation.contributor_group)
        )
        return (
            self.placement.kind,
            self.placement.dim,
            self.placement.reduce_op,
            self.mesh_id,
            self.logical_shape,
            self.dtype,
            ob_sig,
        )
