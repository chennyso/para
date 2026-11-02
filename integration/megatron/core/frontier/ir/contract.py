from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from math import prod


class PlacementKind(Enum):
    SHARD = auto()
    REPLICATE = auto()
    PARTIAL = auto()


class ObligationKind(Enum):
    REDUCE = auto()
    REPARTITION = auto()
    ROUTE = auto()


@dataclass(frozen=True)
class Placement:
    kind: PlacementKind
    dim: int | None = None
    reduce_op: str | None = None

    def __post_init__(self) -> None:
        if self.kind is PlacementKind.SHARD and self.dim is None:
            raise ValueError("sharded placement requires a dimension")
        if self.kind is PlacementKind.PARTIAL and self.reduce_op is None:
            object.__setattr__(self, "reduce_op", "sum")
        if self.kind is not PlacementKind.PARTIAL and self.reduce_op is not None:
            raise ValueError("only partial placement may define reduce_op")


@dataclass(frozen=True)
class PendingObligation:
    kind: ObligationKind
    target: Placement
    contributor_group: str | None = None


@dataclass(frozen=True)
class ValueContract:
    logical_value: str
    logical_shape: tuple[int | str, ...]
    placement: Placement
    mesh_key: str
    obligation: PendingObligation | None = None
    adjoint_key: str = "identity"
    dtype: str = "bfloat16"
    gradient_eligible: bool = True
    presence_policy: str = "required"
    process_group_key: str | None = None
    logical_slice: tuple[int, int, int] | None = None

    def __post_init__(self):
        valid_shape = all((isinstance(dimension, int) and dimension > 0)
                          or (isinstance(dimension, str) and dimension.isidentifier())
                          for dimension in self.logical_shape)
        if not self.logical_value or not valid_shape:
            raise ValueError("value contracts require a name and positive logical dimensions")
        if self.dtype not in {"float32", "bfloat16", "float16", "int64", "int32", "bool"}:
            raise ValueError("unsupported payload dtype")
        if self.presence_policy not in {"required", "optional"}:
            raise ValueError("invalid payload presence policy")
        if self.gradient_eligible and self.dtype in {"int64", "int32", "bool"}:
            raise ValueError("integer metadata cannot have a tensor gradient")
        if self.logical_slice is not None:
            dimension, start, end = self.logical_slice
            if not 0 <= dimension < len(self.logical_shape) or start < 0 or end <= start:
                raise ValueError("projection slice must have a valid dimension and interval")
            bound = self.logical_shape[dimension]
            if not isinstance(bound, int) or end > bound:
                raise ValueError("projection slice requires a concrete bounded logical dimension")

    def with_placement(
        self,
        placement: Placement,
        obligation: PendingObligation | None = None,
    ) -> "ValueContract":
        return replace(self, placement=placement, obligation=obligation)

    @property
    def is_resolved(self) -> bool:
        return self.obligation is None

    def signature(self) -> tuple:
        obligation = self.obligation
        return (
            self.logical_value,
            self.logical_shape,
            self.placement,
            self.mesh_key,
            None if obligation is None else (
                obligation.kind,
                obligation.target,
                obligation.contributor_group,
            ),
            self.adjoint_key,
            self.dtype,
            self.gradient_eligible,
            self.presence_policy,
            self.process_group_key,
            self.logical_slice,
        )


@dataclass(frozen=True)
class DistributedTensorContract:
    value: ValueContract
    logical_mesh: tuple[tuple[str, int], ...]
    physical_ranks: tuple[int, ...]
    sharding: tuple[tuple[str, int], ...]
    replication: tuple[str, ...]
    layout: str
    storage_owners: tuple[int, ...]
    storage_id: str
    partial: tuple[tuple[str, str], ...] = ()
    storage_intervals: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        axes = dict(self.logical_mesh)
        if (not axes or len(axes) != len(self.logical_mesh)
                or any(not name or size < 1 for name, size in self.logical_mesh)):
            raise ValueError('A tensor mesh requires distinct named positive axes')
        if (len(self.physical_ranks) != prod(axes.values())
                or len(set(self.physical_ranks)) != len(self.physical_ranks)
                or min(self.physical_ranks) < 0):
            raise ValueError('Physical tensor ownership must cover every mesh coordinate once')
        shard_axes = tuple(axis for axis, _ in self.sharding)
        partial_axes = tuple(axis for axis, _ in self.partial)
        if (len(set(shard_axes)) != len(shard_axes)
                or len(set(self.replication)) != len(self.replication)
                or len(set(partial_axes)) != len(partial_axes)
                or set(shard_axes) & set(self.replication)
                or set(partial_axes) & (set(shard_axes) | set(self.replication))
                or set(shard_axes) | set(self.replication) | set(partial_axes) != set(axes)
                or any(operation != 'sum' for _, operation in self.partial)):
            raise ValueError('Every mesh axis requires exactly one sharding or replication specification')
        if self.partial and self.value.placement.kind is not PlacementKind.PARTIAL:
            raise ValueError('Partial tensor axes require an explicit partial value contract')
        if not self.partial and self.sharding and (self.value.placement.kind is not PlacementKind.SHARD
                or self.value.placement.dim not in {dimension for _, dimension in self.sharding}):
            raise ValueError('Sharded tensor axes require a compatible sharded value contract')
        if not self.partial and not self.sharding and self.value.placement.kind is not PlacementKind.REPLICATE:
            raise ValueError('Replicated tensor axes require an explicit replicated value contract')
        shard_factors = {}
        for axis, dimension in self.sharding:
            if not 0 <= dimension < len(self.value.logical_shape):
                raise ValueError('Tensor sharding references an invalid logical dimension')
            shard_factors[dimension] = shard_factors.get(dimension, 1) * axes[axis]
        for dimension, factor in shard_factors.items():
            size = self.value.logical_shape[dimension]
            if isinstance(size, int) and size % factor:
                raise ValueError('Logical tensor dimension is indivisible by its complete sharding factor')
        if (not self.layout or not self.storage_id or not self.storage_owners
                or len(set(self.storage_owners)) != len(self.storage_owners)
                or not set(self.storage_owners) <= set(self.physical_ranks)):
            raise ValueError('Tensor storage requires a layout, identity and actual rank owners')
        if self.storage_intervals:
            if self.sharding or self.partial or any(not isinstance(size, int) for size in self.value.logical_shape):
                raise ValueError('Flattened storage intervals require a concrete unsharded logical tensor')
            position = 0
            owners = set()
            for rank, start, end in sorted(self.storage_intervals, key=lambda interval: interval[1]):
                if rank not in self.storage_owners or rank in owners or start != position or end <= start:
                    raise ValueError('Flattened storage intervals require distinct actual owners and complete contiguous coverage')
                owners.add(rank)
                position = end
            if owners != set(self.storage_owners) or position != prod(self.value.logical_shape):
                raise ValueError('Flattened storage intervals omit actual tensor elements or owners')

    def compatible_with(self, other):
        return (self.value.logical_shape, self.value.dtype, self.logical_mesh, self.physical_ranks,
                self.sharding, self.replication, self.partial, self.layout, self.storage_owners,
                self.value.logical_slice, self.value.presence_policy, self.value.gradient_eligible,
                self.storage_intervals) == (
                other.value.logical_shape, other.value.dtype, other.logical_mesh, other.physical_ranks,
                other.sharding, other.replication, other.partial, other.layout, other.storage_owners,
                other.value.logical_slice, other.value.presence_policy, other.value.gradient_eligible,
                other.storage_intervals)


@dataclass(frozen=True)
class ProcessGroupContract:
    key: str
    ranks: tuple[int, ...]
    backend: str
    collective_semantics: tuple[str, ...]

    def __post_init__(self):
        allowed = {'AllReduce', 'AllGather', 'ReduceScatter', 'AllToAll', 'P2P'}
        if (not self.key or not self.backend or not self.ranks or min(self.ranks) < 0
                or len(set(self.ranks)) != len(self.ranks)
                or not self.collective_semantics or not set(self.collective_semantics) <= allowed):
            raise ValueError('Process group contracts require actual members and registered collective semantics')


@dataclass(frozen=True)
class ParameterContract:
    parameter_key: str
    parameter: DistributedTensorContract
    gradient: DistributedTensorContract
    optimizer_states: tuple[DistributedTensorContract, ...]
    gradient_reduce_groups: tuple[ProcessGroupContract, ...]
    replica_groups: tuple[ProcessGroupContract, ...]

    def __post_init__(self):
        if (not self.parameter_key or not self.optimizer_states or not self.gradient_reduce_groups or not self.replica_groups
                or self.parameter.value.logical_shape != self.gradient.value.logical_shape
                or not self.gradient.value.gradient_eligible):
            raise ValueError('Parameter contracts require complete gradient and optimizer ownership')
        if len({state.value.logical_value for state in self.optimizer_states}) != len(self.optimizer_states):
            raise ValueError('Optimizer state identities must be distinct within each parameter')
        gradient_owners = set(self.gradient.storage_owners)
        gradient_members = {rank for group in self.gradient_reduce_groups for rank in group.ranks}
        replica_members = {rank for group in self.replica_groups for rank in group.ranks}
        if (not gradient_owners <= gradient_members
                or not set(self.parameter.storage_owners) <= replica_members
                or any(not set(state.storage_owners) <= gradient_members for state in self.optimizer_states)):
            raise ValueError('Parameter and gradient owners must belong to their declared reduction groups')
