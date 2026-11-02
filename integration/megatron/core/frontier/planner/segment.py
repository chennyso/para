"""
Segment candidate generation.

For a range of layers [start, end], enumerate the SegmentCandidates that can
be formed by choosing a contract for each layer, deciding where to resolve
pending transitions, and recording the resulting state and cost.

This is the innermost expansion loop — called by the beam planner once per
region step.
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from megatron.core.frontier.contracts import RegionContract, REGISTRY
from megatron.core.frontier.mobility import compute_mobility_window
from megatron.core.frontier.planner.candidate import SegmentCandidate
from megatron.core.frontier.state import DistributedState
from megatron.core.frontier.transition import (
    PendingTransition,
    TransitionDecision,
    TransitionSpec,
)

if TYPE_CHECKING:
    from megatron.core.frontier.profiling.database import ProfileDatabase


def _resolve_cost_us(transition: PendingTransition, boundary: int, db: "ProfileDatabase | None") -> float:
    """Look up or estimate the cost of resolving a transition at a given boundary."""
    if db is None:
        # No database — use a rough estimate proportional to bytes.
        return transition.spec.source.pending_bytes / 1e9 * 1e6  # ~1 GB/s
    key = (
        transition.chosen_lowering or "standard_nccl",
        transition.spec.source.obligation.kind.name if transition.spec.source.obligation else "none",
        transition.spec.source.logical_shape,
        transition.spec.source.mesh_id,
    )
    return db.lookup_transition_us(key) or 100.0  # fallback 100 µs


def build_segment_candidates(
    layer_index: int,
    input_state: DistributedState,
    pending: tuple[PendingTransition, ...],
    pp_stage: int,
    vp_chunk: int,
    contracts: Sequence[RegionContract],
    all_contracts: Sequence[RegionContract],
    db: "ProfileDatabase | None" = None,
    legal_lowerings: tuple[str, ...] = ("standard_nccl", "fused_tp_cp", "reduce_on_handoff"),
) -> list[SegmentCandidate]:
    """
    Generate SegmentCandidates for a single layer step.

    For each legal contract at layer_index:
      1. Check it accepts input_state.
      2. Decide for each pending transition whether to resolve now or carry forward.
      3. Build the output state.
      4. Score with profile database if available.
    """
    candidates: list[SegmentCandidate] = []

    legal = [c for c in contracts if c.accepts_input(input_state)]
    if not legal:
        return []

    for contract in legal:
        # Propagate each pending transition through this contract.
        still_pending: list[PendingTransition] = []
        resolved_decisions: list[TransitionDecision] = []
        extra_cost_us = 0.0

        for pt in pending:
            # Can this transition be deferred through this contract?
            can_defer = layer_index in pt.remaining_window and len(pt.remaining_window) > 1

            if can_defer:
                from megatron.core.frontier.mobility import propagate_state
                result = propagate_state(state=pt.spec.source, contract=contract)
                if result is not None:
                    # Update the remaining window to exclude passed layers.
                    new_window = tuple(
                        w for w in pt.remaining_window if w > layer_index
                    )
                    still_pending.append(PendingTransition(
                        spec=pt.spec,
                        remaining_window=new_window,
                        chosen_lowering=pt.chosen_lowering,
                    ))
                    continue

            # Must resolve here — pick the first legal lowering.
            lowering = pt.chosen_lowering or legal_lowerings[0]
            resolve_cost = _resolve_cost_us(pt, layer_index, db)
            extra_cost_us += resolve_cost
            resolved_decisions.append(TransitionDecision(
                transition_id=pt.spec.transition_id,
                resolve_boundary=layer_index,
                lowering_id=lowering,
                estimated_us=resolve_cost,
            ))

        # Build output state from contract.
        from megatron.core.frontier.state import Placement
        out_placement = Placement(
            kind=contract.output_pattern.placement_kind,
            dim=contract.output_pattern.dim,
        )
        output_state = DistributedState(
            placement=out_placement,
            mesh_id=input_state.mesh_id,
            logical_shape=input_state.logical_shape,
            obligation=input_state.obligation if contract.output_pattern.allow_pending else None,
        )

        # Profile-based cost lookup.
        fwd_us = 0.0
        bwd_us = 0.0
        peak_mem = 0
        if db is not None:
            profile = db.lookup_region(contract.profile_id)
            if profile is not None:
                fwd_us = profile.forward_us
                bwd_us = profile.backward_us
                peak_mem = profile.peak_memory_bytes

        fwd_us += extra_cost_us

        candidates.append(SegmentCandidate(
            start_region=layer_index,
            end_region=layer_index,
            contract_ids=(contract.name,),
            internal_transitions=tuple(resolved_decisions),
            pending_transitions=tuple(still_pending),
            output_state=output_state,
            forward_us=fwd_us,
            backward_us=bwd_us,
            peak_memory_bytes=peak_mem,
            pp_stage=pp_stage,
            vp_chunk=vp_chunk,
        ))

    return candidates
