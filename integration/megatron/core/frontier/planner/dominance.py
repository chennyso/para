"""
Frontier-Aware Dominance pruning for the beam search.

Standard beam pruning would drop candidate B whenever candidate A has strictly
better throughput.  But B may have better Λ (fewer pending obligations, or
pending obligations with wider future mobility windows) that will let it find
a better global plan.

We therefore key dominance on SemanticKey: only candidates that occupy the
SAME structural position — same region index, same pp_stage, same vp_chunk,
same output state, same pending obligation signature — compete for pruning.

Within a SemanticKey group, A dominates B on all performance dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from megatron.core.frontier.planner.candidate import SegmentCandidate


@dataclass(frozen=True)
class SemanticKey:
    """
    The position fingerprint used to group candidates before comparing them.

    Two candidates with different SemanticKeys are not comparable — they
    represent structurally different partial plans and are both kept.
    """
    region_index: int
    pp_stage: int
    vp_chunk: int
    output_state_signature: tuple
    # Pending obligations signature: sorted tuple of (obligation_kind, target_kind)
    pending_signature: tuple
    resource_signature: tuple

    @classmethod
    def from_candidate(cls, c: SegmentCandidate) -> "SemanticKey":
        pending_sig = tuple(sorted((
            (transition.spec.transition_id, transition.spec.producer_region, transition.spec.consumer_region,
             transition.spec.source.signature(), transition.spec.target.signature(),
             transition.remaining_window, transition.chosen_lowering)
            for transition in c.pending_transitions
        ), key=repr))
        return cls(
            region_index=c.end_region,
            pp_stage=c.pp_stage,
            vp_chunk=c.vp_chunk,
            output_state_signature=c.output_state.signature(),
            pending_signature=pending_sig,
            resource_signature=c.resource_signature,
        )


def dominates(a: SegmentCandidate, b: SegmentCandidate) -> bool:
    """
    Return True when a strictly dominates b on all performance dimensions.
    Candidates must share the same SemanticKey (callers enforce this).
    """
    if SemanticKey.from_candidate(a) != SemanticKey.from_candidate(b):
        return False
    if len(a.stage_ready_times) != len(b.stage_ready_times):
        return False
    if a.stage_ready_times and any(left > right for left, right in zip(a.stage_ready_times, b.stage_ready_times, strict=True)):
        return False
    return (
        a.max_stage_load_us <= b.max_stage_load_us
        and a.peak_memory_bytes <= b.peak_memory_bytes
        and a.max_ready_time_us <= b.max_ready_time_us
        and a.nic_pressure <= b.nic_pressure
        and a.nvlink_pressure <= b.nvlink_pressure
        # at least one dimension must be strictly better
        and (
            a.max_stage_load_us < b.max_stage_load_us
            or a.peak_memory_bytes < b.peak_memory_bytes
            or a.max_ready_time_us < b.max_ready_time_us
            or a.nic_pressure < b.nic_pressure
            or a.nvlink_pressure < b.nvlink_pressure
            or any(left < right for left, right in zip(a.stage_ready_times, b.stage_ready_times, strict=True))
        )
    )


def filter_dominated(candidates: Sequence[SegmentCandidate]) -> list[SegmentCandidate]:
    """
    Remove Pareto-dominated candidates.

    Groups by SemanticKey first.  Within each group, keeps only the Pareto
    frontier.  Returns at most len(candidates) results but typically far fewer.
    """
    from collections import defaultdict
    groups: dict[SemanticKey, list[SegmentCandidate]] = defaultdict(list)
    for c in candidates:
        groups[SemanticKey.from_candidate(c)].append(c)

    kept: list[SegmentCandidate] = []
    for group in groups.values():
        pareto: list[SegmentCandidate] = []
        for candidate in group:
            if not any(dominates(other, candidate) for other in group):
                pareto.append(candidate)
        kept.extend(pareto)
    return kept


def select_top_k(
    candidates: Sequence[SegmentCandidate], k: int
) -> list[SegmentCandidate]:
    """
    First filter dominated candidates, then keep the k cheapest remaining by
    max_stage_load_us.  This is the beam selection step.
    """
    filtered = filter_dominated(candidates)
    filtered.sort(key=lambda c: (c.max_stage_load_us, c.peak_memory_bytes))
    if k < 1:
        raise ValueError("beam selection requires a positive capacity")
    representatives = {}
    for candidate in filtered:
        representatives.setdefault(SemanticKey.from_candidate(candidate), candidate)
    selected = list(representatives.values())
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in filtered:
        if len(selected) >= max(k, len(representatives)):
            break
        if id(candidate) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(id(candidate))
    return selected
