"""
SearchNeighborhood — the agent's proposal for the next planning sub-problem.

The agent (Diagnosis → Strategy → Critic) produces a SearchNeighborhood that
narrows the search to a sub-range of the model with specific action types.
The planner then solves that sub-problem exactly rather than re-planning the
whole model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from megatron.core.frontier.agent.actions import SearchAction
from megatron.core.frontier.agent.evidence import StepEvidence

if TYPE_CHECKING:
    from megatron.core.frontier.planner.compiler import FrontierPlan


@dataclass
class SearchNeighborhood:
    """
    Defines the next local search sub-problem.

    start_region / end_region: layer range to re-plan.
    *_changes: lists of actions to try within this neighborhood.
    vpp_choices: which VPP values to try for the affected stages.
    """
    start_region: int
    end_region: int
    contract_changes: tuple[SearchAction, ...] = field(default_factory=tuple)
    transition_moves: tuple[SearchAction, ...] = field(default_factory=tuple)
    lowering_changes: tuple[SearchAction, ...] = field(default_factory=tuple)
    pp_cut_changes: tuple[SearchAction, ...] = field(default_factory=tuple)
    vpp_choices: tuple[int, ...] = field(default_factory=lambda: (1, 2, 3))

    def all_actions(self) -> tuple[SearchAction, ...]:
        return (
            self.contract_changes
            + self.transition_moves
            + self.lowering_changes
            + self.pp_cut_changes
        )


def build_neighborhood(
    evidence: StepEvidence,
    plan: "FrontierPlan",
    strategy: str = "bottleneck_first",
) -> SearchNeighborhood:
    """
    Heuristic neighborhood builder used when no LLM agent is available.

    strategy="bottleneck_first": focus on the stage with highest load and the
    transitions with the most exposed time.
    """
    from megatron.core.frontier.agent.actions import MoveTransition, MovePPCut

    # Find the highest-exposed transition.
    if not evidence.transitions:
        # No transitions — widen a PP cut near the bottleneck stage.
        bottleneck = evidence.bottleneck_stage
        layers = plan.layers_for_stage(bottleneck)
        mid = layers[len(layers) // 2] if layers else 0
        return SearchNeighborhood(
            start_region=max(0, mid - 2),
            end_region=min(len(plan.contract_ids) - 1, mid + 2),
            pp_cut_changes=(MovePPCut(from_layer=mid, to_layer=mid + 1),),
        )

    worst_t = max(evidence.transitions, key=lambda t: t.exposed_us)
    boundary = worst_t.current_boundary
    window = worst_t.legal_window

    moves = []
    if len(window) > 1:
        # Try moving the transition one step later.
        next_pos = next((w for w in window if w > boundary), None)
        if next_pos is not None:
            moves.append(MoveTransition(
                transition_id=worst_t.transition_id,
                from_boundary=boundary,
                to_boundary=next_pos,
            ))

    return SearchNeighborhood(
        start_region=max(0, boundary - 1),
        end_region=min(len(plan.contract_ids) - 1, boundary + 2),
        transition_moves=tuple(moves),
        vpp_choices=(1, 2, 3),
    )
