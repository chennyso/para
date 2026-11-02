"""
ReduceOnHandoffLowering: fused reduce + PP handoff.

Instead of:
    source_stage: all-reduce (Partial → Replicate)
    PP send:      Replicate → dest
    dest_stage:   slice to own Shard

We do:
    source_stage: slice to dest intervals, P2P-send slices to dest ranks
    dest_stage:   receive partial slices, local accumulate → Shard

This saves one full all-reduce and one full tensor copy across the PP boundary.
The savings are largest when the reduce group and the PP group overlap in ranks
(common when TP=1 and PP uses the same nodes).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.core.frontier.lowering.base import (
    LoweringContext,
    LOWERING_REGISTRY,
)
from megatron.core.frontier.state import ObligationKind


@dataclass
class SliceInterval:
    start: int
    end: int


def build_source_slices(
    partial_tensor: torch.Tensor,
    destination_intervals: list[SliceInterval],
) -> list[torch.Tensor]:
    """Extract destination-rank slices from the partial tensor."""
    return [
        partial_tensor[iv.start:iv.end].contiguous()
        for iv in destination_intervals
    ]


def issue_reduce_handoff_sends(
    slices: list[torch.Tensor],
    destination_ranks: list[int],
    process_group,
) -> list:
    """Non-blocking P2P sends of per-destination slices."""
    ops = [
        dist.P2POp(dist.isend, tensor, rank, process_group)
        for tensor, rank in zip(slices, destination_ranks, strict=True)
    ]
    return dist.batch_isend_irecv(ops)


def receive_and_accumulate(
    buffers: list[torch.Tensor],
    source_ranks: list[int],
    process_group,
) -> torch.Tensor:
    """Receive partial contributions and reduce-sum them locally."""
    ops = [
        dist.P2POp(dist.irecv, buf, rank, process_group)
        for buf, rank in zip(buffers, source_ranks, strict=True)
    ]
    requests = dist.batch_isend_irecv(ops)
    for req in requests:
        req.wait()

    output = torch.zeros_like(buffers[0])
    for buf in buffers:
        output.add_(buf)
    return output


class ReduceOnHandoffLowering:
    lowering_id = "reduce_on_handoff"

    def supports(self, transition, context: LoweringContext) -> bool:
        ob = transition.source.obligation
        if ob is None:
            return False
        return ob.kind == ObligationKind.REDUCE

    def forward(self, tensor: torch.Tensor, context: LoweringContext) -> torch.Tensor:
        """
        Source side: slice and send partial contributions to PP neighbors.
        Destination side: receive and accumulate.

        In a compiled plan, source and destination sides execute different
        code paths (determined at plan-compile time by the BoundarySchema).
        """
        schema = context.schema
        is_source = getattr(schema, "is_source_side", True)

        if is_source:
            intervals = getattr(schema, "destination_intervals", [])
            dest_ranks = getattr(schema, "destination_ranks", [])
            pg = getattr(schema, "process_group", None)
            if not intervals:
                # Fallback: standard all-reduce
                dist.all_reduce(tensor)
                return tensor
            slices = build_source_slices(tensor, intervals)
            works = issue_reduce_handoff_sends(slices, dest_ranks, pg)
            for w in works:
                w.wait()
            # Source side returns its own local slice
            own_iv = getattr(schema, "own_interval", SliceInterval(0, tensor.shape[0]))
            return tensor[own_iv.start:own_iv.end].contiguous()
        else:
            buffers = getattr(schema, "receive_buffers", [])
            src_ranks = getattr(schema, "source_ranks", [])
            pg = getattr(schema, "process_group", None)
            return receive_and_accumulate(buffers, src_ranks, pg)

    def backward(self, grad: torch.Tensor, context: LoweringContext) -> torch.Tensor:
        # Backward of reduce_on_handoff is an all-gather on the gradient.
        schema = context.schema
        pg = getattr(schema, "process_group", None)
        world = dist.get_world_size(pg)
        output = torch.empty(
            grad.shape[0] * world,
            *grad.shape[1:],
            dtype=grad.dtype,
            device=grad.device,
        )
        dist.all_gather_into_tensor(output, grad.contiguous(), group=pg)
        return output

    def profile_key(self, transition, context: LoweringContext) -> tuple:
        s = transition.source
        return (
            self.lowering_id,
            s.obligation.kind if s.obligation else None,
            s.logical_shape,
            s.mesh_id,
            getattr(context.schema, "is_source_side", None),
        )


LOWERING_REGISTRY.register(ReduceOnHandoffLowering())
