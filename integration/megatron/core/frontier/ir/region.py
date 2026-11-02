from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from megatron.core.frontier.ir.contract import ValueContract

if TYPE_CHECKING:
    from megatron.core.frontier.contracts import RegionContract


class RegionKind(Enum):
    ATTENTION = auto()
    DENSE_MLP = auto()
    MLP = DENSE_MLP
    MOE = auto()
    HYBRID = auto()
    PRE_DISPATCH = auto()
    DISPATCH = auto()
    EXPERT = auto()
    COMBINE = auto()
    ROUTER = auto()
    LINEAR_ATTENTION = auto()
    SSM = auto()
    GDN = auto()
    RECURRENT_MIXER = auto()
    MTP = auto()
    OTHER_REGISTERED = auto()


@dataclass(frozen=True)
class ParallelPolicy:
    tp: int
    cp: int
    ep: int = 1
    etp: int = 1
    dp: int = 1
    backend: str | None = None

    def __post_init__(self) -> None:
        if min(self.tp, self.cp, self.ep, self.etp, self.dp) < 1:
            raise ValueError("parallel degrees must be positive")


@dataclass(frozen=True)
class RegionMode:
    mode_id: str
    region_kind: RegionKind
    input_contract: ValueContract
    output_contract: ValueContract
    policy: ParallelPolicy
    implementation_key: str
    profile_key: str


@dataclass(frozen=True)
class Region:
    region_id: int
    logical_layer_id: int
    kind: RegionKind
    legal_modes: tuple[str, ...]


@dataclass(frozen=True)
class ParallelDomain:
    start_region: int
    end_region: int
    policy: ParallelPolicy
    mode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_region < 0 or self.end_region < self.start_region:
            raise ValueError("invalid parallel domain range")


@dataclass(frozen=True)
class PipelineRegion:
    region_id: str
    logical_layer_id: int
    region_kind: RegionKind
    input_contracts: tuple[ValueContract, ...]
    output_contracts: tuple[ValueContract, ...]
    parameter_keys: tuple[str, ...]
    forward_impl: str
    backward_impl: str
    forward_profile_key: str
    backward_profile_key: str
    transportable_output: bool = False
    model_family: str = ''
    supported_parallel_modes: tuple[str, ...] = ()
    forward_impl_sequence: tuple[str, ...] = ()
    training_contract: RegionContract | None = None
    parameter_elements: int | None = None
    parameter_storage_bytes: int | None = None

    def __post_init__(self):
        if self.logical_layer_id < 0 or not self.region_id or not self.input_contracts or not self.output_contracts:
            raise ValueError("pipeline regions require complete typed input and output contracts")
        if not all((self.forward_impl, self.backward_impl, self.forward_profile_key, self.backward_profile_key)):
            raise ValueError("pipeline region callable and profile identities are required")
        if len(set(self.parameter_keys)) != len(self.parameter_keys):
            raise ValueError("a pipeline region contains duplicate parameter keys")
        if len(set(self.supported_parallel_modes)) != len(self.supported_parallel_modes):
            raise ValueError('region parallel modes must have distinct implementation identities')
        if ((self.parameter_elements is None) != (self.parameter_storage_bytes is None)
                or (self.parameter_elements is not None
                    and min(self.parameter_elements, self.parameter_storage_bytes) < 0)):
            raise ValueError('Region参数存储需要完整的实际元素数量和字节数量')


@dataclass(frozen=True)
class PipelineRegionGraph:
    regions: tuple[PipelineRegion, ...]
    edges: tuple[tuple[str, str], ...]
    tensor_connections: tuple[TensorConnection, ...] = ()
    expected_parameter_keys: tuple[str, ...] | None = None
    external_inputs: tuple[tuple[str, int], ...] = ()
    scope: str = 'UNSPECIFIED'

    def __post_init__(self):
        identifiers = tuple(region.region_id for region in self.regions)
        if len(set(identifiers)) != len(identifiers) or not identifiers:
            raise ValueError("region graph requires distinct nonempty region identities")
        positions = {identity: position for position, identity in enumerate(identifiers)}
        for source, destination in self.edges:
            if source not in positions or destination not in positions or positions[source] >= positions[destination]:
                raise ValueError("region graph dependencies must follow registered execution order")
        parameters = tuple(key for region in self.regions for key in region.parameter_keys)
        if len(set(parameters)) != len(parameters):
            raise ValueError("each parameter must have one region owner")
        if self.scope not in ('UNSPECIFIED', 'OPERATOR', 'MODEL'):
            raise ValueError('Region graph verification scope must be explicit')
        regions = {region.region_id: region for region in self.regions}
        for connection in self.tensor_connections:
            if ((connection.producer, connection.consumer) not in self.edges
                    or not 0 <= connection.producer_output < len(regions[connection.producer].output_contracts)
                    or not 0 <= connection.consumer_input < len(regions[connection.consumer].input_contracts)):
                raise ValueError('Tensor connections must reference registered region edges and interfaces')
        if len(set(self.external_inputs)) != len(self.external_inputs):
            raise ValueError('External input bindings must be distinct')
        for identity, index in self.external_inputs:
            if identity not in regions or not 0 <= index < len(regions[identity].input_contracts):
                raise ValueError('External input binding references an unknown region interface')
        if self.scope == 'OPERATOR':
            bound_inputs = [(connection.consumer, connection.consumer_input) for connection in self.connections()]
            bound_inputs.extend(self.external_inputs)
            required_inputs = {(region.region_id, index) for region in self.regions
                               for index in range(len(region.input_contracts))}
            if len(bound_inputs) != len(set(bound_inputs)) or set(bound_inputs) != required_inputs:
                raise ValueError('每个算子Region输入必须具有唯一Tensor生产者或者外部输入')

    def connections(self):
        if self.tensor_connections:
            return self.tensor_connections
        regions = {region.region_id: region for region in self.regions}
        result = []
        for source, destination in self.edges:
            outputs = regions[source].output_contracts
            names = [contract.logical_value for contract in outputs]
            if len(set(names)) != len(names):
                raise ValueError('Repeated output identities require explicit tensor connections')
            for index, contract in enumerate(regions[destination].input_contracts):
                if contract.logical_value in names:
                    result.append(TensorConnection(source, names.index(contract.logical_value), destination, index))
        return tuple(result)


@dataclass(frozen=True)
class TensorConnection:
    producer: str
    producer_output: int
    consumer: str
    consumer_input: int
