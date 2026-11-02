"""
ResolveScheduleNode — executed on the training hot path.

One node is inserted per transition boundary in the compiled
TransformerLayerSchedulePlan.  It wraps a pre-looked-up TransitionLowering and
a pre-compiled BoundarySchema.  The forward/backward calls are designed to be
as thin as possible — no Python dict lookups, no planner queries.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from megatron.core.frontier.lowering.base import LoweringContext, LOWERING_REGISTRY
from megatron.core.frontier.runtime.boundary import BoundarySchema
from megatron.core.frontier.runtime.trace import FrontierTrace


class ResolveScheduleNode:
    """
    Forward/backward node that executes a compiled TransitionLowering.

    Instances are created once during plan compilation.  The training loop
    calls .forward() / .backward() per microbatch.
    """

    __slots__ = ("_lowering", "_schema", "_trace", "_nvtx_name")

    def __init__(
        self,
        schema: BoundarySchema,
        trace: FrontierTrace | None = None,
        nvtx_enabled: bool = False,
    ):
        self._lowering = LOWERING_REGISTRY.get(schema.lowering_id)
        self._schema = schema
        self._trace = trace
        self._nvtx_name = (
            f"FRONTIER/RESOLVE/{schema.boundary_id}" if nvtx_enabled else None
        )

    def forward(
        self,
        tensor: torch.Tensor,
        microbatch_slot: int = 0,
        vp_chunk: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        ctx = LoweringContext(
            schema=self._schema.for_microbatch(microbatch_slot),
            microbatch_slot=microbatch_slot,
            vp_chunk=vp_chunk,
            stream=stream,
        )
        with self._nvtx_range():
            result = self._lowering.forward(tensor, ctx)
        if self._trace is not None:
            self._trace.record_forward(
                boundary_id=self._schema.boundary_id,
                microbatch_slot=microbatch_slot,
            )
        return result

    def backward(
        self,
        grad: torch.Tensor,
        microbatch_slot: int = 0,
        vp_chunk: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        ctx = LoweringContext(
            schema=self._schema.for_microbatch(microbatch_slot),
            microbatch_slot=microbatch_slot,
            vp_chunk=vp_chunk,
            stream=stream,
        )
        with self._nvtx_range():
            result = self._lowering.backward(grad, ctx)
        if self._trace is not None:
            self._trace.record_backward(
                boundary_id=self._schema.boundary_id,
                microbatch_slot=microbatch_slot,
            )
        return result

    @contextlib.contextmanager
    def _nvtx_range(self):
        if self._nvtx_name is None:
            yield
        else:
            try:
                torch.cuda.nvtx.range_push(self._nvtx_name)
                yield
            finally:
                torch.cuda.nvtx.range_pop()


class NoopScheduleNode:
    """Placeholder inserted when a transition boundary has no lowering to execute."""

    __slots__ = ()

    def forward(self, tensor: torch.Tensor, **_) -> torch.Tensor:
        return tensor

    def backward(self, grad: torch.Tensor, **_) -> torch.Tensor:
        return grad
