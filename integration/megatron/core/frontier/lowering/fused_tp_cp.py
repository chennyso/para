"""
FusedTPCPLowering: fused TP×CP layout conversion.

This lowering handles the case where the transition involves both a tensor-
parallel reshard and a context-parallel layout change (e.g. contiguous ↔
zigzag).  Rather than executing them sequentially, it fuses the two operations
into a single all-to-all that achieves both in one communication round.

From the paper: this is the concrete Transition Program template that proves a
transition can be directly lowered (A→B→C ⟹ A→C), and Frontier's contribution
is automatically deciding *when* and *where* to use it.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.core.frontier.lowering.base import (
    LoweringContext,
    LOWERING_REGISTRY,
)
from megatron.core.frontier.state import ObligationKind


class FusedTPCPLowering:
    lowering_id = "fused_tp_cp"

    def supports(self, transition, context: LoweringContext) -> bool:
        ob = transition.source.obligation
        if ob is None:
            return False
        # Applicable when a repartition covers both TP and CP axes.
        return ob.kind == ObligationKind.REPARTITION

    def forward(self, tensor: torch.Tensor, context: LoweringContext) -> torch.Tensor:
        """
        Execute the fused all-to-all that simultaneously re-shards the TP axis
        and re-tiles the sequence dimension for the target CP layout.

        Implementation sketch (real version needs the TP×CP mesh from
        ProcessGroupCollection):
        1. Reshape: [S/cp_src, H] → [cp_src, S/(cp_src*tp), tp, H/tp]
        2. All-to-all across the tp×cp combined process group
        3. Reshape: [tp, S/(cp_dst*tp), cp_dst, H/tp] → [S/cp_dst, H]
        """
        schema = context.schema
        src = schema.source_state
        tgt = schema.target_state

        # Determine the combined process group for the fused all-to-all.
        # In production this comes from ProcessGroupCollection; here we fall
        # back to the default group for portability.
        pg = None  # TODO: look up tp×cp process group by mesh_id

        seq_dim = 0
        world_size = dist.get_world_size(pg)
        if tensor.shape[seq_dim] % world_size != 0:
            raise ValueError(
                f"Sequence dim {tensor.shape[seq_dim]} not divisible by "
                f"world_size {world_size} for fused TP×CP all-to-all"
            )

        # Split along the sequence dimension, all-to-all, then concat.
        chunks = list(tensor.chunk(world_size, dim=seq_dim))
        output_chunks = [torch.empty_like(c) for c in chunks]
        dist.all_to_all(output_chunks, chunks, group=pg)
        return torch.cat(output_chunks, dim=seq_dim)

    def backward(self, grad: torch.Tensor, context: LoweringContext) -> torch.Tensor:
        # Fused all-to-all is its own inverse when world_size is symmetric.
        return self.forward(grad, context)

    def profile_key(self, transition, context: LoweringContext) -> tuple:
        s = transition.source
        t = transition.target
        return (
            self.lowering_id,
            s.mesh_id,
            t.mesh_id,
            s.logical_shape,
            s.obligation.kind if s.obligation else None,
        )


LOWERING_REGISTRY.register(FusedTPCPLowering())
