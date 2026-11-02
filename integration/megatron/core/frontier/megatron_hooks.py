"""
Megatron-LM integration hooks for the Frontier planning subsystem.

These functions are called from:
  megatron/core/pipeline_parallel/schedules.py  — to inject ResolveScheduleNodes
  megatron/core/models/gpt/fine_grained_callables.py — to add pre/post_transition nodes
  megatron/core/models/common/model_chunk_schedule_plan.py — to set layer layouts

The training hot path calls only get_resolve_node() and get_noop_node(); all
planning is done before training starts by calling install_frontier_plan().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.frontier.planner.compiler import FrontierPlan
    from megatron.core.frontier.runtime.boundary import BoundarySchema


# Module-level plan installed once before training.
_ACTIVE_PLAN: "FrontierPlan | None" = None
_RESOLVE_NODES: dict[int, object] = {}   # boundary_id → ResolveScheduleNode
_NOOP_NODE: object | None = None


def install_frontier_plan(
    plan: "FrontierPlan",
    nvtx_enabled: bool = False,
    trace=None,
) -> None:
    """
    Install a compiled FrontierPlan for the current training run.

    Call this once, after the plan is compiled and distributed to all ranks,
    and before the first training step.  It pre-creates all ResolveScheduleNodes
    so the hot path never creates Python objects.
    """
    global _ACTIVE_PLAN, _RESOLVE_NODES, _NOOP_NODE
    from megatron.core.frontier.runtime.resolve import ResolveScheduleNode, NoopScheduleNode

    _ACTIVE_PLAN = plan
    _RESOLVE_NODES = {}
    _NOOP_NODE = NoopScheduleNode()

    for tid, schema in plan.boundary_schemas.items():
        _RESOLVE_NODES[schema.boundary_id] = ResolveScheduleNode(
            schema=schema,
            trace=trace,
            nvtx_enabled=nvtx_enabled,
        )


def get_resolve_node(boundary_id: int) -> object:
    """
    Return the pre-compiled ResolveScheduleNode for this boundary.

    Called from fine_grained_callables.py during each microbatch forward/backward.
    If no plan is installed or this boundary has no resolve node, returns a NoopNode.
    """
    if _RESOLVE_NODES:
        node = _RESOLVE_NODES.get(boundary_id)
        if node is not None:
            return node
    return _NOOP_NODE or _make_noop()


def get_active_plan() -> "FrontierPlan | None":
    return _ACTIVE_PLAN


def frontier_layer_contract(layer_index: int) -> str | None:
    """
    Return the implementation_id for layer_index from the active plan.

    Called by the model provider to select which implementation to use.
    Returns None when no plan is active (fall back to Megatron defaults).
    """
    if _ACTIVE_PLAN is None:
        return None
    if layer_index < len(_ACTIVE_PLAN.contract_ids):
        contract_name = _ACTIVE_PLAN.contract_ids[layer_index]
        try:
            from megatron.core.frontier.contracts import REGISTRY
            return REGISTRY.get(contract_name).implementation_id
        except (KeyError, Exception):
            return None
    return None


def frontier_pp_layout() -> tuple[int, ...] | None:
    """
    Return the PP cut positions from the active plan.
    Called by the Megatron pipeline builder to set stage boundaries.
    """
    if _ACTIVE_PLAN is None:
        return None
    return _ACTIVE_PLAN.pp_cuts


def _make_noop():
    from megatron.core.frontier.runtime.resolve import NoopScheduleNode
    global _NOOP_NODE
    _NOOP_NODE = NoopScheduleNode()
    return _NOOP_NODE
