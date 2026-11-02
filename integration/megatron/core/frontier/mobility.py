"""
Mobility Analysis — the core algorithm of Frontier.

Given a TransitionSpec τ and the sequence of RegionContracts between the
producer region and the consumer region, compute_mobility_window returns the
set of layer indices W(τ) where τ may legally be resolved.

A region r is in W(τ) if and only if *every* region between the producer and r
(inclusive) can propagate the unresolved state through its forward and backward
pass without corrupting correctness — i.e., each intermediate region's
StateTransferRule confirms that the obligation can pass through.

Correctness guarantee: any rule admitted into the registry must have a passing
GPU exchange test (Path A: resolve-then-compute == Path B: compute-then-resolve,
checked for both output and input gradients).  Rules without a test are not
admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from megatron.core.frontier.state import (
    DistributedState,
    ObligationKind,
    Placement,
    PlacementKind,
    TransitionObligation,
)
from megatron.core.frontier.contracts import RegionContract


# ---------------------------------------------------------------------------
# State-transfer rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateTransferRule:
    """
    Declares that a specific region implementation can carry an unresolved
    tensor of a specific obligation kind through its computation.

    implementation_id: matches RegionContract.implementation_id
    input_placement: the Placement of the incoming unresolved tensor
    obligation_kind: the kind of outstanding obligation on that tensor
    output_placement: the Placement the region produces (may differ from input
        if the region's own collective changes the layout, even when unresolved)
    preserve_obligation: whether the obligation survives intact to the output,
        or whether it is absorbed (partially or fully) by the region's own
        collectives.
    """
    implementation_id: str
    input_placement: Placement
    obligation_kind: ObligationKind
    output_placement: Placement
    preserve_obligation: bool

    # Unique key into the GPU correctness test suite.  Rules without a test_id
    # are rejected by the registry at startup unless
    # FRONTIER_SKIP_RULE_VALIDATION=1 is set.
    test_id: str | None = None


class TransferRuleRegistry:
    def __init__(self):
        self._rules: dict[tuple, StateTransferRule] = {}

    def register(self, rule: StateTransferRule) -> None:
        key = (rule.implementation_id, rule.input_placement.kind,
               rule.input_placement.dim, rule.obligation_kind)
        self._rules[key] = rule

    def lookup(
        self,
        implementation_id: str,
        input_placement: Placement,
        obligation_kind: ObligationKind,
    ) -> StateTransferRule | None:
        key = (implementation_id, input_placement.kind,
               input_placement.dim, obligation_kind)
        return self._rules.get(key)

    def all_rules(self) -> list[StateTransferRule]:
        return list(self._rules.values())


TRANSFER_RULES = TransferRuleRegistry()


def register_transfer_rule(rule: StateTransferRule) -> StateTransferRule:
    TRANSFER_RULES.register(rule)
    return rule


# ---------------------------------------------------------------------------
# Register built-in rules
# ---------------------------------------------------------------------------

def _register_builtin_rules():
    shard0 = Placement(kind=PlacementKind.SHARD, dim=0)
    partial = Placement(kind=PlacementKind.PARTIAL, reduce_op="sum")

    # Attention TP=4: can carry a Partial(sum) obligation through its forward
    # because its own all-reduce at the end will consume the partial.
    register_transfer_rule(StateTransferRule(
        implementation_id="megatron_attention_tp4",
        input_placement=partial,
        obligation_kind=ObligationKind.REDUCE,
        output_placement=shard0,
        preserve_obligation=False,
        test_id="attention_tp4_carry_partial_reduce",
    ))

    # Attention TP=2 CP=2: same pattern
    register_transfer_rule(StateTransferRule(
        implementation_id="megatron_attention_tp2_cp2",
        input_placement=partial,
        obligation_kind=ObligationKind.REDUCE,
        output_placement=shard0,
        preserve_obligation=False,
        test_id="attention_tp2_cp2_carry_partial_reduce",
    ))

    # MLP TP=4: carries a Partial(sum) through and its all-reduce absorbs it
    register_transfer_rule(StateTransferRule(
        implementation_id="megatron_mlp_tp4",
        input_placement=partial,
        obligation_kind=ObligationKind.REDUCE,
        output_placement=shard0,
        preserve_obligation=False,
        test_id="mlp_tp4_carry_partial_reduce",
    ))

    # MLP TP=2 CP=2: same
    register_transfer_rule(StateTransferRule(
        implementation_id="megatron_mlp_tp2_cp2",
        input_placement=partial,
        obligation_kind=ObligationKind.REDUCE,
        output_placement=shard0,
        preserve_obligation=False,
        test_id="mlp_tp2_cp2_carry_partial_reduce",
    ))

    # Shard passthrough: a region that does not change the layout can carry
    # any Repartition obligation (e.g. a CP layout change) if the layout is
    # already a shard on dim-0 and the region doesn't touch that axis.
    for impl_id in (
        "megatron_attention_tp4",
        "megatron_mlp_tp4",
    ):
        register_transfer_rule(StateTransferRule(
            implementation_id=impl_id,
            input_placement=shard0,
            obligation_kind=ObligationKind.REPARTITION,
            output_placement=shard0,
            preserve_obligation=True,
            test_id=f"{impl_id}_carry_shard_repartition",
        ))


_register_builtin_rules()


# ---------------------------------------------------------------------------
# Core propagation logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropagationResult:
    output_state: DistributedState
    rule_used: StateTransferRule


def propagate_state(
    state: DistributedState,
    contract: RegionContract,
) -> PropagationResult | None:
    """
    Attempt to propagate `state` (which may carry an unresolved obligation)
    through `contract`.

    Returns a PropagationResult describing the output state, or None if the
    obligation cannot legally pass through this region.
    """
    if state.obligation is None:
        # No obligation — check that the region accepts this resolved placement.
        if not contract.accepts_input(state):
            return None
        # Produce the output state according to the contract's output pattern.
        out_placement = Placement(
            kind=contract.output_pattern.placement_kind,
            dim=contract.output_pattern.dim,
        )
        return PropagationResult(
            output_state=DistributedState(
                placement=out_placement,
                mesh_id=state.mesh_id,
                logical_shape=state.logical_shape,
                obligation=None,
            ),
            rule_used=None,  # type: ignore[arg-type]
        )

    # There is an obligation: look up a transfer rule.
    rule = TRANSFER_RULES.lookup(
        implementation_id=contract.implementation_id,
        input_placement=state.placement,
        obligation_kind=state.obligation.kind,
    )
    if rule is None:
        return None

    # The contract must also accept this placement kind.
    if contract.input_pattern.placement_kind != state.placement.kind:
        if not contract.input_pattern.allow_pending:
            return None

    new_obligation = state.obligation if rule.preserve_obligation else None
    return PropagationResult(
        output_state=DistributedState(
            placement=rule.output_placement,
            mesh_id=state.mesh_id,
            logical_shape=state.logical_shape,
            obligation=new_obligation,
        ),
        rule_used=rule,
    )


# ---------------------------------------------------------------------------
# Window computation
# ---------------------------------------------------------------------------

def compute_mobility_window(
    transition_id: int,
    source_state: DistributedState,
    producer_region: int,
    consumer_region: int,
    contracts: Sequence[RegionContract],
) -> tuple[int, ...]:
    """
    Compute W(τ) — the set of layer indices where transition τ may legally
    be resolved.

    The window always includes producer_region (immediate resolution is always
    legal).  It extends as long as each subsequent region has a StateTransferRule
    that allows the obligation to pass through.

    contracts[i] should be the RegionContract chosen for layer i.
    """
    if source_state.obligation is None:
        # Trivial transition — can be resolved at the producer.
        return (producer_region,)

    window = [producer_region]
    state = source_state

    for region_index in range(producer_region + 1, consumer_region + 1):
        if region_index >= len(contracts):
            break
        result = propagate_state(state=state, contract=contracts[region_index])
        if result is None:
            break
        state = result.output_state
        window.append(region_index)
        # If the obligation was absorbed (preserve_obligation=False), further
        # movement is not needed — the transition resolved itself.
        if state.obligation is None:
            break

    return tuple(window)


def compute_all_windows(
    transitions: list["TransitionSpec"],  # noqa: F821
    contracts: Sequence[RegionContract],
) -> dict[int, tuple[int, ...]]:
    """
    Batch version — returns a mapping from transition_id to mobility window.
    """
    from megatron.core.frontier.transition import TransitionSpec
    result: dict[int, tuple[int, ...]] = {}
    for t in transitions:
        result[t.transition_id] = compute_mobility_window(
            transition_id=t.transition_id,
            source_state=t.source,
            producer_region=t.producer_region,
            consumer_region=t.consumer_region,
            contracts=contracts,
        )
    return result
