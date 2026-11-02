"""TransitionProfiler: measures transition lowering cost in isolation."""
from __future__ import annotations

from typing import TYPE_CHECKING

from megatron.core.frontier.profiling.region_profiler import profile_callable

if TYPE_CHECKING:
    from megatron.core.frontier.profiling.database import ProfileDatabase
    from megatron.core.frontier.transition import TransitionSpec
    from megatron.core.frontier.lowering.base import LoweringContext


class TransitionProfiler:
    def __init__(self, db: "ProfileDatabase"):
        self.db = db

    def profile_lowering(
        self,
        lowering_id: str,
        transition: "TransitionSpec",
        fwd_fn,
        bwd_fn=None,
        warmup: int = 10,
        measured: int = 50,
    ):
        result = profile_callable(fwd_fn, bwd_fn, warmup=warmup, measured=measured)
        ob = transition.source.obligation
        self.db.insert_transition(
            lowering_id=lowering_id,
            obligation_kind=ob.kind.name if ob else "none",
            shape_sig=transition.source.logical_shape,
            mesh_id=transition.source.mesh_id,
            forward_us=result.forward_us_p50,
            backward_us=result.backward_us_p50,
            bytes_count=transition.source.pending_bytes,
            temp_memory_bytes=0,
        )
        return result
