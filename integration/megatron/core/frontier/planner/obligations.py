import hashlib
import json
import math
from dataclasses import asdict, dataclass

from megatron.core.frontier.ir.contract import DistributedTensorContract
from megatron.core.frontier.ir.program import RegionParallelMapping


class Obligation:
    def identifier(self):
        encoded = json.dumps({'kind': type(self).__name__, 'properties': asdict(self)},
            sort_keys=True, separators=(',', ':'), default=lambda value: value.name, allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class MemoryDeficit(Obligation):
    gpu: str
    interval: tuple[float, float]
    bytes: int

    def __post_init__(self):
        if (not self.gpu or self.bytes <= 0 or self.interval[0] < 0
                or self.interval[1] <= self.interval[0] or not all(map(math.isfinite, self.interval))):
            raise ValueError('MemoryDeficit requires an actual positive deficit interval')


@dataclass(frozen=True)
class LayoutMismatch(Obligation):
    producer: str
    consumer: str
    producer_layout: DistributedTensorContract
    consumer_layout: DistributedTensorContract
    direction: str

    def __post_init__(self):
        if self.direction not in ('forward', 'backward'):
            raise ValueError('Layout mismatch direction must be explicit')
        if self.producer_layout.compatible_with(self.consumer_layout):
            raise ValueError('Compatible contracts cannot produce LayoutMismatch')


@dataclass(frozen=True)
class ProcessGroupUnsatisfied(Obligation):
    region: str
    requested_mapping: RegionParallelMapping
    available_ranks: tuple[int, ...]


@dataclass(frozen=True)
class GradientOwnershipConflict(Obligation):
    parameter: str
    expected_group: tuple[int, ...]
    actual_group: tuple[int, ...]
    ownership_kind: str

    def __post_init__(self):
        if self.ownership_kind not in ('parameter', 'gradient', 'optimizer'):
            raise ValueError('Ownership conflict must identify its training state category')


@dataclass(frozen=True)
class NativeTrainingStateConflict(Obligation):
    category: str
    parameter: str
    ranks: tuple[int, ...]
    expected: tuple
    actual: tuple
    detail: str

    def __post_init__(self):
        if self.category not in ('Layout', 'Ownership', 'ProcessGroup', 'Coverage') or not self.detail:
            raise ValueError('Native training conflicts require a registered category and concrete evidence')


@dataclass(frozen=True)
class CollectiveOrderConflict(Obligation):
    process_group: str
    op_a: str
    op_b: str


@dataclass(frozen=True)
class DeadlineMiss(Obligation):
    task: str
    deadline: float
    actual_finish: float

    def __post_init__(self):
        if not all(map(math.isfinite, (self.deadline, self.actual_finish))) or self.actual_finish <= self.deadline:
            raise ValueError('DeadlineMiss requires a finite actual missed deadline')


@dataclass(frozen=True)
class UnsupportedImplementation(Obligation):
    region: str
    requested_parallel_mapping: RegionParallelMapping
    reason: str


@dataclass(frozen=True)
class CriticalPathRegression(Obligation):
    old_path: tuple[str, ...]
    new_path: tuple[str, ...]
    delta: float

    def __post_init__(self):
        if not self.old_path or not self.new_path or not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError('Critical path regression requires two actual paths and a positive duration difference')


@dataclass(frozen=True)
class ProgramCoverageConflict(Obligation):
    component: str
    expected: tuple[str, ...]
    actual: tuple[str, ...]


@dataclass(frozen=True)
class InterfaceBindingConflict(Obligation):
    region: str
    input_index: int
    providers: tuple[str, ...]


@dataclass(frozen=True)
class DependencyViolation(Obligation):
    predecessor: str
    consumer: str
    predecessor_finish: float
    consumer_start: float


@dataclass(frozen=True)
class EventOrderViolation(Obligation):
    predecessor: str
    consumer: str
    predecessor_event: str
    consumer_event: str
    predecessor_time: float
    consumer_time: float


@dataclass(frozen=True)
class ResourceConflict(Obligation):
    device: str
    resource: str
    task_a: str
    task_b: str


@dataclass(frozen=True)
class UpdateIncomplete(Obligation):
    parameter: str
    missing_event: str


@dataclass(frozen=True)
class TensorLifetimeConflict(Obligation):
    storage: str
    release_task: str
    consumer: str
