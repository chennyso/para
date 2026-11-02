from __future__ import annotations

from megatron.core.frontier.agents.actions import AgentProposal, EditKind, ProgramEdit
from megatron.core.frontier.agents.base import FrontierAgent


class BoundaryAgent(FrontierAgent):
    agent_id = "boundary"

    def propose(self, program, evidence, ledger):
        if not program.boundaries:
            return ()
        boundary = program.boundaries[0]
        edit = ProgramEdit(EditKind.CHANGE_BOUNDARY_PROGRAM, f"boundary:{boundary.boundary_id}", boundary.signature())
        return (AgentProposal(self.agent_id, (edit,), "boundary_exposure", 0.0, 0.5),)
