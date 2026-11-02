"""
Base protocol and registry for Transition Lowerings.

A TransitionLowering knows how to physically execute a particular kind of
distributed-state conversion.  The planner selects a lowering_id; the compiler
emits a ResolveScheduleNode that wraps the corresponding implementation.

Key design points:
- Lowerings are stateless descriptors; state lives in pre-allocated buffers
  that the BoundarySchema tracks.
- forward/backward are called on the training hot path.  They must never
  allocate new process groups, create new Python objects beyond buffer slices,
  or call the planner.
- profile_key returns a tuple that uniquely identifies the cost entry in the
  profile database for this lowering + this transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.frontier.transition import TransitionSpec
    from megatron.core.frontier.runtime.boundary import BoundarySchema


@dataclass
class LoweringContext:
    """
    Per-execution context handed to a lowering at forward/backward time.
    Everything here is compiled from the FrontierPlan before training starts.
    """
    schema: "BoundarySchema"
    microbatch_slot: int      # 0..num_microbatches-1
    vp_chunk: int             # 0..vpp-1
    stream: object | None = None  # torch.cuda.Stream or None for default
    process_groups: dict[str, object] | None = None


@runtime_checkable
class TransitionLowering(Protocol):
    lowering_id: str

    def supports(
        self,
        transition: "TransitionSpec",
        context: LoweringContext,
    ) -> bool: ...

    def forward(self, tensor: object, context: LoweringContext) -> object: ...

    def backward(self, grad: object, context: LoweringContext) -> object: ...

    def profile_key(
        self,
        transition: "TransitionSpec",
        context: LoweringContext,
    ) -> tuple: ...


class LoweringRegistry:
    def __init__(self):
        self._registry: dict[str, TransitionLowering] = {}

    def register(self, lowering: TransitionLowering) -> None:
        self._registry[lowering.lowering_id] = lowering

    def get(self, lowering_id: str) -> TransitionLowering:
        if lowering_id not in self._registry:
            raise KeyError(f"No lowering registered: {lowering_id!r}")
        return self._registry[lowering_id]

    def all_ids(self) -> list[str]:
        return list(self._registry.keys())


LOWERING_REGISTRY = LoweringRegistry()
