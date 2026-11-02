from __future__ import annotations
from megatron.core.frontier.agent.evidence import TransitionEvidence, collect_evidence
from megatron.core.frontier.agent.actions import (
    SearchAction,
    ChangeRegionImpl,
    MoveTransition,
    MovePPCut,
    ChangeVPP,
    ChangeLowering,
)
from megatron.core.frontier.agent.neighborhood import SearchNeighborhood, build_neighborhood
