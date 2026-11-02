from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class EditKind(Enum):
    SPLIT_DOMAIN = auto()
    MERGE_DOMAIN = auto()
    CHANGE_POLICY = auto()
    MOVE_PP_CUT = auto()
    CHANGE_VPP = auto()
    CHANGE_BOUNDARY_PROGRAM = auto()
    MOVE_BOUNDARY_OP = auto()
    CHANGE_MICROBATCH = auto()
    CHANGE_RECOMPUTE = auto()


@dataclass(frozen=True)
class ProgramEdit:
    kind: EditKind
    target: str
    value: object


@dataclass(frozen=True)
class AgentProposal:
    agent_id: str
    edits: tuple[ProgramEdit, ...]
    mechanism: str
    predicted_step_delta_us: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("proposal confidence must be within [0, 1]")


def edits_conflict(left: ProgramEdit, right: ProgramEdit) -> bool:
    if left.target != right.target:
        return False
    if left.kind is right.kind and left.value == right.value:
        return False
    return left.kind in {EditKind.CHANGE_POLICY, EditKind.MOVE_PP_CUT, EditKind.CHANGE_VPP, EditKind.CHANGE_BOUNDARY_PROGRAM, EditKind.MOVE_BOUNDARY_OP, EditKind.CHANGE_MICROBATCH, EditKind.CHANGE_RECOMPUTE} or right.kind in {EditKind.CHANGE_POLICY, EditKind.MOVE_PP_CUT, EditKind.CHANGE_VPP, EditKind.CHANGE_BOUNDARY_PROGRAM, EditKind.MOVE_BOUNDARY_OP, EditKind.CHANGE_MICROBATCH, EditKind.CHANGE_RECOMPUTE}


def proposals_conflict(left: AgentProposal, right: AgentProposal) -> bool:
    return any(edits_conflict(a, b) for a in left.edits for b in right.edits)
