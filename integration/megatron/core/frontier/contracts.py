"""
Region Contract registry.

A RegionContract describes a single *implementation* of a model region
(Attention, MLP, MoE, …) including its parallel policy, the distributed-state
patterns it accepts on input and produces on output, and references to the
profile and Megatron implementation that backs it.

Contracts are registered once at startup (before planning) and are immutable
during training.  The planner only sees the contracts it can legally use for
each layer; Megatron model code only sees the implementation_id it is told to
run by the compiled FrontierPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from megatron.core.frontier.state import PlacementKind
from megatron.core.frontier.ir.contract import DistributedTensorContract, ParameterContract, ProcessGroupContract


class RegionKind(Enum):
    ATTENTION = auto()
    MLP = auto()
    MOE = auto()
    GDN = auto()   # grouped dense norm / feed-forward variant
    LAYER = auto() # full transformer layer (coarse granularity)


@dataclass(frozen=True)
class ParallelPolicy:
    """The parallel degrees in effect inside this region."""
    tp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1
    backend: str | None = None  # e.g. "deepep", "hybridep", "nccl_ep"


@dataclass(frozen=True)
class StatePattern:
    """
    What placement kind a region implementation expects on its input (or
    produces on its output).

    allow_pending=True means the implementation can legally receive a tensor
    whose obligation has not yet been discharged — the mobility analysis uses
    this to decide whether a pending transition can be deferred past this region.
    """
    placement_kind: PlacementKind
    dim: int | None = None
    allow_pending: bool = False


@dataclass(frozen=True)
class RegionContract:
    """
    Immutable descriptor for one (region, implementation) pair.

    Planner sees a list of RegionContracts for each layer.  It picks exactly
    one per layer.  The chosen contract's implementation_id is what gets
    compiled into the FrontierPlan and eventually handed to Megatron.
    """
    name: str
    region_kind: RegionKind

    input_pattern: StatePattern
    output_pattern: StatePattern

    policy: ParallelPolicy

    # Identifier understood by the Megatron model provider / fine-grained
    # schedule builder.
    implementation_id: str

    # Key into the profile database.
    profile_id: str

    # Optional: explicitly restrict which layer indices this contract applies
    # to.  None means "any layer".
    applicable_layer_indices: frozenset[int] | None = None
    forward_inputs: tuple[DistributedTensorContract, ...] = ()
    forward_outputs: tuple[DistributedTensorContract, ...] = ()
    backward_gradient_inputs: tuple[DistributedTensorContract, ...] = ()
    backward_gradient_outputs: tuple[DistributedTensorContract, ...] = ()
    parameter_contracts: tuple[ParameterContract, ...] = ()
    required_process_groups: tuple[ProcessGroupContract, ...] = ()

    def __post_init__(self):
        if not any((self.forward_inputs, self.forward_outputs, self.backward_gradient_inputs,
                    self.backward_gradient_outputs, self.parameter_contracts, self.required_process_groups)):
            return
        if not self.forward_inputs or not self.forward_outputs or not self.required_process_groups:
            raise ValueError('Distributed region contracts require complete forward interfaces and groups')
        if (len(self.backward_gradient_inputs) != sum(value.value.gradient_eligible for value in self.forward_outputs)
                or len(self.backward_gradient_outputs) != sum(value.value.gradient_eligible for value in self.forward_inputs)):
            raise ValueError('Distributed region contracts must close both backward gradient interfaces')
        groups = {group.key: group for group in self.required_process_groups}
        if len(groups) != len(self.required_process_groups):
            raise ValueError('Region process group identities must be distinct')
        parameters = [parameter.parameter_key for parameter in self.parameter_contracts]
        if len(set(parameters)) != len(parameters):
            raise ValueError('A region contains duplicate parameter ownership')
        for parameter in self.parameter_contracts:
            for group in (*parameter.gradient_reduce_groups, *parameter.replica_groups):
                if groups.get(group.key) != group:
                    raise ValueError('Parameter ownership references a process group absent from its region')

    def accepts_input(self, state: "DistributedState") -> bool:  # noqa: F821
        from megatron.core.frontier.state import DistributedState
        if self.input_pattern.placement_kind != state.placement.kind:
            return False
        if self.input_pattern.dim is not None and self.input_pattern.dim != state.placement.dim:
            return False
        if state.obligation is not None and not self.input_pattern.allow_pending:
            return False
        return True


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

class ContractRegistry:
    """Thread-unsafe but process-local registry built at import / startup time."""

    def __init__(self):
        self._contracts: dict[str, RegionContract] = {}
        self._by_kind: dict[RegionKind, list[RegionContract]] = {}

    def register(self, contract: RegionContract) -> None:
        if contract.name in self._contracts:
            raise KeyError(f"Contract already registered: {contract.name!r}")
        self._contracts[contract.name] = contract
        self._by_kind.setdefault(contract.region_kind, []).append(contract)

    def get(self, name: str) -> RegionContract:
        return self._contracts[name]

    def by_kind(self, kind: RegionKind) -> list[RegionContract]:
        return list(self._by_kind.get(kind, []))

    def all_contracts(self) -> list[RegionContract]:
        return list(self._contracts.values())

    def contracts_for_layer(
        self, layer_index: int, kind: RegionKind | None = None
    ) -> list[RegionContract]:
        source = self.by_kind(kind) if kind else self.all_contracts()
        return [
            c for c in source
            if c.applicable_layer_indices is None
            or layer_index in c.applicable_layer_indices
        ]


REGISTRY = ContractRegistry()


def register_contract(contract: RegionContract) -> RegionContract:
    """Decorator / call helper — register and return the contract."""
    REGISTRY.register(contract)
    return contract


def lookup_contracts(
    layer_index: int, kind: RegionKind | None = None
) -> list[RegionContract]:
    return REGISTRY.contracts_for_layer(layer_index, kind)


# ---------------------------------------------------------------------------
# Built-in contracts for Dense attention
# ---------------------------------------------------------------------------

def _register_builtin_contracts():
    from megatron.core.frontier.state import Placement, PlacementKind

    # Attention TP=4 CP=1 — standard Megatron tensor-parallel attention
    register_contract(RegionContract(
        name="attention.tp4_cp1",
        region_kind=RegionKind.ATTENTION,
        input_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        output_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        policy=ParallelPolicy(tp=4, cp=1),
        implementation_id="megatron_attention_tp4",
        profile_id="attention_tp4_cp1",
    ))

    # Attention TP=2 CP=2 — context-parallel attention with halved TP
    register_contract(RegionContract(
        name="attention.tp2_cp2",
        region_kind=RegionKind.ATTENTION,
        input_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        output_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        policy=ParallelPolicy(tp=2, cp=2),
        implementation_id="megatron_attention_tp2_cp2",
        profile_id="attention_tp2_cp2",
    ))

    # MLP TP=4
    register_contract(RegionContract(
        name="mlp.tp4",
        region_kind=RegionKind.MLP,
        input_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        output_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        policy=ParallelPolicy(tp=4, cp=1),
        implementation_id="megatron_mlp_tp4",
        profile_id="mlp_tp4",
    ))

    # MLP TP=2 CP=2
    register_contract(RegionContract(
        name="mlp.tp2_cp2",
        region_kind=RegionKind.MLP,
        input_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        output_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        policy=ParallelPolicy(tp=2, cp=2),
        implementation_id="megatron_mlp_tp2_cp2",
        profile_id="mlp_tp2_cp2",
    ))

    # MoE EP=4 ETP=1 (NCCL backend)
    register_contract(RegionContract(
        name="moe.ep4_etp1_nccl",
        region_kind=RegionKind.MOE,
        input_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        output_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        policy=ParallelPolicy(tp=1, ep=4, etp=1, backend="nccl_ep"),
        implementation_id="megatron_moe_ep4_nccl",
        profile_id="moe_ep4_etp1_nccl",
    ))

    # MoE EP=8 ETP=1 (DeepEP backend)
    register_contract(RegionContract(
        name="moe.ep8_etp1_deepep",
        region_kind=RegionKind.MOE,
        input_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        output_pattern=StatePattern(
            placement_kind=PlacementKind.SHARD, dim=0, allow_pending=True
        ),
        policy=ParallelPolicy(tp=1, ep=8, etp=1, backend="deepep"),
        implementation_id="megatron_moe_ep8_deepep",
        profile_id="moe_ep8_etp1_deepep",
    ))


_register_builtin_contracts()
