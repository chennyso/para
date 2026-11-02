from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterPlacement:
    logical_key: str
    owner_pp_stage: int
    replica_group_key: str | None = None


class ParameterOwnership:
    def __init__(self, placements: tuple[ParameterPlacement, ...] = ()) -> None:
        self._placements = {item.logical_key: item for item in placements}

    def register(self, placement: ParameterPlacement) -> None:
        if placement.logical_key in self._placements:
            raise KeyError(f"parameter placement already registered: {placement.logical_key}")
        self._placements[placement.logical_key] = placement

    def owner(self, logical_key: str) -> ParameterPlacement:
        return self._placements[logical_key]

    def all(self) -> tuple[ParameterPlacement, ...]:
        return tuple(self._placements.values())
