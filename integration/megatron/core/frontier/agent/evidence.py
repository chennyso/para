"""
Evidence collector: turns raw FrontierTrace events into structured summaries
that the agent can reason about without seeing raw tensors or topology details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.frontier.planner.compiler import FrontierPlan
    from megatron.core.frontier.runtime.trace import FrontierTrace


@dataclass(frozen=True)
class TransitionEvidence:
    """Per-transition evidence for one training step."""
    transition_id: int
    current_boundary: int
    legal_window: tuple[int, ...]
    current_lowering: str
    exposed_us: float         # measured time between fwd resolve and bwd resolve
    bytes: int
    source_stage: int
    destination_stage: int

    @property
    def is_movable(self) -> bool:
        return len(self.legal_window) > 1

    @property
    def is_bottleneck(self) -> bool:
        """Heuristic: transition contributes more than 5% of exposed time."""
        return self.exposed_us > 0 and self.bytes > 0


@dataclass(frozen=True)
class StepEvidence:
    """Full evidence for one training step."""
    step: int
    step_time_us: float
    pp_bubble_fraction: float
    transitions: tuple[TransitionEvidence, ...]

    # Per-stage load imbalance (max/mean ratio).
    stage_load_imbalance: float = 1.0

    # Stage with the highest load.
    bottleneck_stage: int = 0


def collect_evidence(
    plan: "FrontierPlan",
    trace: "FrontierTrace",
    step_time_us: float,
    pp_bubble_fraction: float = 0.0,
) -> StepEvidence:
    """
    Build a StepEvidence from a FrontierPlan and a FrontierTrace.
    Called once per step (or every N steps) between the training loop and
    the agent.
    """
    transition_evidences = []
    for tid, decision in plan.transition_decisions.items():
        exposed = trace.exposed_us_for_boundary(tid)
        schema = plan.boundary_schemas.get(tid)
        src_stage = 0
        dst_stage = 0
        if schema:
            # Approximate stage from rank position in source/destination ranks.
            src_stage = min(schema.source_ranks) // 8 if schema.source_ranks else 0
            dst_stage = min(schema.destination_ranks) // 8 if schema.destination_ranks else 0
        transition_evidences.append(TransitionEvidence(
            transition_id=tid,
            current_boundary=decision.resolve_boundary,
            legal_window=(),  # filled in by the agent from mobility analysis
            current_lowering=decision.lowering_id,
            exposed_us=exposed,
            bytes=0,  # filled in from schema.tensor_shape if available
            source_stage=src_stage,
            destination_stage=dst_stage,
        ))

    return StepEvidence(
        step=0,
        step_time_us=step_time_us,
        pp_bubble_fraction=pp_bubble_fraction,
        transitions=tuple(transition_evidences),
    )
