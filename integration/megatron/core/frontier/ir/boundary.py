from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, auto
import hashlib
import json

from megatron.core.frontier.ir.contract import ValueContract


class BoundaryOpKind(Enum):
    RESOLVE = auto()
    REPARTITION = auto()
    ROUTE = auto()
    TRANSFER = auto()
    DISPATCH = auto()
    COMBINE = auto()
    PROJECT = auto()
    FUSED = auto()


@dataclass(frozen=True)
class BoundaryOp:
    op_id: str
    kind: BoundaryOpKind
    input_contract: ValueContract
    output_contract: ValueContract
    implementation_key: str
    movable_before_region: int | None = None
    movable_after_region: int | None = None


@dataclass(frozen=True)
class BufferSpecification:
    owner_rank: int
    field: str
    shape: tuple[int, ...]
    dtype: str
    slot_id: int

    def __post_init__(self):
        if min(self.owner_rank, self.slot_id) < 0 or not self.shape or min(self.shape) < 1:
            raise ValueError("buffer specification requires valid ownership, shape and lifetime slot")


@dataclass(frozen=True)
class DynamicRoutingMap:
    source_ranks: tuple[int, ...]
    destination_ranks: tuple[int, ...]
    control_field: str
    maximum_source_rows: int
    maximum_destination_rows: int
    forward_rule: str
    adjoint_rule: str

    def __post_init__(self):
        if (not self.source_ranks or len(self.source_ranks) != len(self.destination_ranks)
                or set(self.source_ranks) & set(self.destination_ranks)
                or len(set(self.source_ranks + self.destination_ranks)) != 2 * len(self.source_ranks)
                or min(self.maximum_source_rows, self.maximum_destination_rows) < 1
                or not self.control_field or not self.forward_rule or not self.adjoint_rule):
            raise ValueError('dynamic routing map requires actual disjoint domains, bounded rows and an inverse rule')


@dataclass(frozen=True)
class ExecutableWitness:
    process_groups: tuple[tuple[str, tuple[int, ...]], ...]
    parameter_ownership: tuple[tuple[str, tuple[int, ...]], ...]
    tensor_maps: tuple[tuple[int, int, int, int, int], ...]
    forward_nodes: tuple[str, ...]
    backward_nodes: tuple[str, ...]
    buffer_specs: tuple[BufferSpecification, ...]
    schedule_requirements: tuple[str, ...]
    implementation_key: str
    dynamic_tensor_maps: tuple[DynamicRoutingMap, ...] = ()

    def __post_init__(self):
        if not self.process_groups or not self.forward_nodes or not self.backward_nodes or not self.buffer_specs:
            raise ValueError("executable witness requires groups, forward, backward and allocated buffers")
        group_names = tuple(key for key, _ in self.process_groups)
        if len(set(group_names)) != len(group_names):
            raise ValueError("executable witness contains duplicate process groups")
        members = {rank for _, ranks in self.process_groups for rank in ranks}
        if any(len(set(ranks)) != len(ranks) or not ranks for _, ranks in self.process_groups):
            raise ValueError("witness process groups require distinct nonempty members")
        if any(spec.owner_rank not in members for spec in self.buffer_specs):
            raise ValueError("buffer owner is absent from the process group witness")
        for routing in self.dynamic_tensor_maps:
            if (any(rank not in members for rank in routing.source_ranks + routing.destination_ranks)
                    or any(not any(spec.owner_rank == rank and spec.field == routing.control_field
                                   and spec.dtype == 'int64' for spec in self.buffer_specs)
                           for rank in routing.destination_ranks)):
                raise ValueError('dynamic routing requires an actual non-gradient control buffer on every receiver')
        for source, destination, dimension, start, end in self.tensor_maps:
            if source not in members or destination not in members or dimension < 0 or start < 0 or end <= start:
                raise ValueError("invalid executable tensor map")

    def identifier(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class BoundaryProgram:
    boundary_id: int
    source_region: int
    destination_region: int
    source_contract: ValueContract
    destination_contract: ValueContract
    ops: tuple[BoundaryOp, ...]
    latency_lower_bound_us: float = 0.0
    scratch_bytes: int = 0
    transferred_bytes: int = 0
    executable_witness_key: str | None = None
    executable_witness: ExecutableWitness | None = None

    def signature(self) -> tuple:
        return (
            self.boundary_id,
            self.source_region,
            self.destination_region,
            self.source_contract.signature(),
            self.destination_contract.signature(),
            tuple((op.kind, op.implementation_key, op.input_contract.signature(), op.output_contract.signature(),
                   op.movable_before_region, op.movable_after_region) for op in self.ops),
            self.executable_witness_key,
        )


@dataclass(frozen=True)
class PendingOpportunity:
    boundary_id: int
    current_contract: ValueContract
    completion_sites: tuple[int, ...]
    alternatives: tuple[BoundaryProgram, ...]
    compose_keys: tuple[str, ...]
    live_bytes: int

    def __post_init__(self):
        if self.boundary_id < 0 or self.live_bytes < 0 or not self.completion_sites or not self.alternatives:
            raise ValueError('pending opportunity requires completion sites, realizations and live storage')
        if any(site < 0 for site in self.completion_sites):
            raise ValueError('opportunity completion sites must identify actual regions')
        if any(program.boundary_id != self.boundary_id or program.executable_witness is None
               or program.executable_witness_key != program.executable_witness.identifier()
               for program in self.alternatives):
            raise ValueError('opportunity realization requires its registered executable witness')
        if any(not key for key in self.compose_keys):
            raise ValueError('composition capabilities require nonempty identities')

    def signature(self):
        return (self.boundary_id, self.current_contract.signature(), tuple(sorted(set(self.completion_sites))),
                tuple(sorted({program.signature() for program in self.alternatives}, key=repr)),
                tuple(sorted(set(self.compose_keys))), self.live_bytes)
