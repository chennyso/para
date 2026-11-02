from __future__ import annotations

from abc import ABC, abstractmethod

from megatron.core.frontier.agents.actions import AgentProposal


class FrontierAgent(ABC):
    agent_id: str

    @abstractmethod
    def propose(self, program, evidence, ledger) -> tuple[AgentProposal, ...]:
        raise NotImplementedError
