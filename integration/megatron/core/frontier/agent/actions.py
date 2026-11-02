"""
Agent action space: the four compound edit types the agent proposes.

Actions are purely descriptive — the planner validates and executes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ChangeRegionImpl:
    """Switch layer `layer_index` to a different RegionContract."""
    layer_index: int
    new_contract_name: str


@dataclass(frozen=True)
class MoveTransition:
    """Move a transition's resolve boundary from one layer to another."""
    transition_id: int
    from_boundary: int
    to_boundary: int


@dataclass(frozen=True)
class MovePPCut:
    """Move a PP stage boundary from layer `from_layer` to `to_layer`."""
    from_layer: int
    to_layer: int


@dataclass(frozen=True)
class ChangeVPP:
    """Change the VPP factor for a PP stage."""
    pp_stage: int
    new_vpp: int


@dataclass(frozen=True)
class ChangeLowering:
    """Switch a transition to a different lowering implementation."""
    transition_id: int
    new_lowering_id: str


SearchAction = Union[
    ChangeRegionImpl,
    MoveTransition,
    MovePPCut,
    ChangeVPP,
    ChangeLowering,
]
