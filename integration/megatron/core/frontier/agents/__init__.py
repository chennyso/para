from megatron.core.frontier.agents.actions import AgentProposal, EditKind, ProgramEdit, proposals_conflict
from megatron.core.frontier.agents.ledger import LedgerEntry, ParallelProgramLedger, ProgramMeasurement
from megatron.core.frontier.agents.base import FrontierAgent
from megatron.core.frontier.agents.structure_agent import StructureAgent
from megatron.core.frontier.agents.pipeline_agent import PipelineAgent
from megatron.core.frontier.agents.boundary_agent import BoundaryAgent
from megatron.core.frontier.agents.critic_agent import CriticAgent, Critique
from megatron.core.frontier.agents.parameter_ownership import ParameterPlacement, ParameterOwnership

__all__ = [
    "AgentProposal", "EditKind", "ProgramEdit", "proposals_conflict",
    "LedgerEntry", "ParallelProgramLedger", "ProgramMeasurement",
    "FrontierCoordinator",
    "FrontierAgent", "StructureAgent", "PipelineAgent", "BoundaryAgent",
    "CriticAgent", "Critique", "ParameterPlacement", "ParameterOwnership",
]


def __getattr__(name):
    # FrontierCoordinator 需要 torch（仅训练期使用）。延迟到真正访问时再导入，
    # 使搜索流程（只需 LLM provider）可以在无 torch 的环境里导入本包。
    if name == 'FrontierCoordinator':
        from megatron.core.frontier.agents.coordinator import FrontierCoordinator
        globals()['FrontierCoordinator'] = FrontierCoordinator
        return FrontierCoordinator
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
