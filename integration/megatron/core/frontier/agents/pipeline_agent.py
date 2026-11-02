from __future__ import annotations

from megatron.core.frontier.agents.actions import AgentProposal, EditKind, ProgramEdit
from megatron.core.frontier.agents.base import FrontierAgent


class PipelineAgent(FrontierAgent):
    agent_id = "pipeline"

    def propose(self, program, evidence, ledger):
        cuts = program.pipeline.pp_cuts
        proposals = []
        if cuts:
            edit = ProgramEdit(EditKind.MOVE_PP_CUT, f"pp-cut:{cuts[0]}", cuts[0])
            proposals.append(AgentProposal(self.agent_id, (edit,), "pipeline_bubble", 0.0, 0.5))
        current_vpp = program.pipeline.vp_assignment[0] if program.pipeline.vp_assignment else 1
        for vpp in (1, 2, 4):
            if vpp == current_vpp:
                continue
            edit = ProgramEdit(EditKind.CHANGE_VPP, "pipeline", vpp)
            proposals.append(AgentProposal(self.agent_id, (edit,), "pipeline_vpp", 0.0, 0.4))
        return tuple(proposals)
