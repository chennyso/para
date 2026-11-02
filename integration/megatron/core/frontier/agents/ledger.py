from __future__ import annotations

from dataclasses import dataclass

from megatron.core.frontier.agents.actions import AgentProposal


@dataclass(frozen=True)
class ProgramMeasurement:
    step_time_us: float
    peak_memory_bytes: int
    bubble_us: float = 0.0
    exposed_boundary_us: float = 0.0
    exposed_tp_us: float = 0.0
    exposed_cp_us: float = 0.0
    exposed_ep_us: float = 0.0
    exposed_pp_us: float = 0.0


@dataclass(frozen=True)
class LedgerEntry:
    program_id: str
    parent_program_id: str | None
    proposals: tuple[AgentProposal, ...]
    prediction: ProgramMeasurement | None
    measurement: ProgramMeasurement
    mechanism_confirmed: bool | None


class ParallelProgramLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> None:
        if any(existing.program_id == entry.program_id for existing in self._entries):
            raise KeyError(f"program already recorded: {entry.program_id}")
        self._entries.append(entry)

    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def best(self) -> LedgerEntry | None:
        return min(self._entries, key=lambda item: item.measurement.step_time_us, default=None)
