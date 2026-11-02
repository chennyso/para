from __future__ import annotations

from megatron.core.frontier.agents.actions import AgentProposal, EditKind, ProgramEdit
from megatron.core.frontier.agents.base import FrontierAgent


class StructureAgent(FrontierAgent):
    agent_id = "structure"

    def __init__(self, tensor_parallel_candidates: tuple[int, ...] = (2, 4, 8)) -> None:
        self.tensor_parallel_candidates = tensor_parallel_candidates

    def propose(self, program, evidence, ledger):
        domains = tuple(program.domains)
        if not domains:
            return ()
        domain = domains[0]
        target = f"domain:{domain.start_region}:{domain.end_region}"
        proposals = []
        for tensor_parallel in self.tensor_parallel_candidates:
            if tensor_parallel == domain.policy.tp:
                continue
            value = {
                "tp": tensor_parallel,
                "cp": domain.policy.cp,
                "ep": domain.policy.ep,
                "etp": domain.policy.etp,
                "dp": domain.policy.dp,
                "backend": domain.policy.backend,
            }
            edit = ProgramEdit(EditKind.CHANGE_POLICY, target, value)
            proposals.append(AgentProposal(self.agent_id, (edit,), "structure_profile", 0.0, 0.5))
        return tuple(proposals)
