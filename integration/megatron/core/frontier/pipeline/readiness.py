import heapq
import math
from dataclasses import dataclass


RESOURCE_CLASSES = frozenset(('GPU_COMPUTE', 'SM_COMM', 'PCIE', 'IB', 'DMA', 'HOST_MEMORY', 'GPU_MEMORY'))
EVENT_CLASSES = frozenset(('REGION_FWD', 'REGION_BWD', 'TP_COMM', 'CP_COMM', 'EP_DISPATCH',
                           'EP_COMBINE', 'PP_SEND', 'PP_RECV', 'BOUNDARY', 'RECOMPUTE',
                           'ACTIVATION_D2H', 'ACTIVATION_H2D', 'OPTIMIZER', 'COMPUTE', 'FORWARD',
                           'DGRAD', 'WGRAD', 'SEND', 'RECV', 'COLLECTIVE', 'REDISTRIBUTE',
                           'GRADIENT_REDUCE', 'GRADIENT_READY', 'OPTIMIZER_STEP', 'BUFFER_ALLOCATE', 'BUFFER_RELEASE'))


@dataclass(frozen=True)
class ResourceSignature:
    resources: tuple[str, ...]
    device: str

    def __post_init__(self):
        if not self.device or not self.resources or not set(self.resources) <= RESOURCE_CLASSES:
            raise ValueError('resource signature must identify supported physical resources')
        if len(set(self.resources)) != len(self.resources):
            raise ValueError('resource classes must be distinct')


@dataclass(frozen=True)
class ExecutionNode:
    node_id: str
    node_type: str
    dependencies: tuple[str, ...]
    duration_us: float
    resource: ResourceSignature
    alloc_bytes: int = 0
    produces: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    parameter_keys: tuple[str, ...] = ()
    process_group_key: str | None = None
    collective_index: int | None = None
    rank: int | None = None
    communication_id: str | None = None
    deadline_us: float | None = None
    workspace_bytes: int = 0
    region_id: str | None = None
    microbatch_id: int | None = None
    finish_dependencies: tuple[str, ...] = ()
    launch_dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.node_id or self.node_type not in EVENT_CLASSES:
            raise ValueError('execution node must have an identity and supported event class')
        if not math.isfinite(self.duration_us) or self.duration_us < 0:
            raise ValueError('execution duration must be finite and nonnegative')
        if self.alloc_bytes < 0 or len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError('invalid allocation size or duplicate dependency')
        if self.workspace_bytes < 0 or len(set(self.parameter_keys)) != len(self.parameter_keys):
            raise ValueError('Invalid task workspace or duplicate parameter identities')
        if self.deadline_us is not None and (not math.isfinite(self.deadline_us) or self.deadline_us < 0):
            raise ValueError('Task deadlines must be finite and nonnegative')
        if self.microbatch_id is not None and self.microbatch_id < 0:
            raise ValueError('Actual microbatch identities must be nonnegative')
        if any(len(set(values)) != len(values) for values in (self.finish_dependencies, self.launch_dependencies)):
            raise ValueError('Event dependencies must contain distinct task identities')


class ReadinessDAG:
    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError('execution node identities must be unique')
        self.successors = {identity: [] for identity in self.nodes}
        degrees = {identity: len(set((*node.dependencies, *node.finish_dependencies))) for identity, node in self.nodes.items()}
        for identity, node in self.nodes.items():
            for dependency in (*node.dependencies, *node.finish_dependencies, *node.launch_dependencies):
                if dependency not in self.nodes:
                    raise ValueError(f'unknown dependency {dependency}')
            for dependency in set((*node.dependencies, *node.finish_dependencies)):
                self.successors[dependency].append(identity)
        ready = [identity for identity, degree in degrees.items() if degree == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            identity = heapq.heappop(ready)
            order.append(identity)
            for child in self.successors[identity]:
                degrees[child] -= 1
                if degrees[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(self.nodes):
            raise ValueError('execution dependencies contain a cycle')
        self.order = tuple(order)
        self.event_successors = {(identity, phase): set() for identity in self.nodes for phase in ('START', 'FINISH')}
        for identity, node in self.nodes.items():
            self.event_successors[(identity, 'START')].add((identity, 'FINISH'))
            for dependency in node.dependencies:
                self.event_successors[(dependency, 'FINISH')].add((identity, 'START'))
            for dependency in node.finish_dependencies:
                self.event_successors[(dependency, 'FINISH')].add((identity, 'FINISH'))
            for dependency in node.launch_dependencies:
                self.event_successors[(dependency, 'START')].add((identity, 'FINISH'))
        event_degrees = dict.fromkeys(self.event_successors, 0)
        for children in self.event_successors.values():
            for child in children:
                event_degrees[child] += 1
        event_ready = [event for event, degree in event_degrees.items() if degree == 0]
        event_count = 0
        while event_ready:
            event = event_ready.pop()
            event_count += 1
            for child in self.event_successors[event]:
                event_degrees[child] -= 1
                if event_degrees[child] == 0:
                    event_ready.append(child)
        if event_count != len(self.event_successors):
            raise ValueError('Execution start and completion dependencies contain a cycle')
        self.descendants = {}
        for identity in reversed(order):
            children = set(self.successors[identity])
            for child in self.successors[identity]:
                children.update(self.descendants[child])
            self.descendants[identity] = frozenset(children)
        producers = {}
        for node in self.nodes.values():
            for value in node.produces:
                if value in producers:
                    raise ValueError(f'multiple producers for {value}')
                producers[value] = node.node_id
        for node in self.nodes.values():
            for value in node.consumes:
                if value not in producers or not self.reachable(producers[value], node.node_id):
                    raise ValueError(f'consumer {node.node_id} lacks readiness dependency for {value}')

    def reachable(self, left, right):
        if left not in self.nodes or right not in self.nodes:
            raise KeyError((left, right))
        return left == right or right in self.descendants[left]

    def may_overlap(self, left, right):
        return left != right and not self.event_reachable((left, 'FINISH'), (right, 'START')) and not self.event_reachable(
            (right, 'FINISH'), (left, 'START'))

    def event_reachable(self, source, destination):
        if source not in self.event_successors or destination not in self.event_successors:
            raise KeyError((source, destination))
        pending, visited = [source], set()
        while pending:
            current = pending.pop()
            if current == destination:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(self.event_successors[current])
        return False

    def lifetimes(self, intervals):
        if set(intervals) != set(self.nodes):
            raise ValueError('buffer lifetime requires the complete finalized schedule')
        for identity, (start, end) in intervals.items():
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError('invalid finalized schedule interval')
            if any(intervals[parent][1] > start for parent in self.nodes[identity].dependencies):
                raise ValueError('finalized schedule violates readiness dependencies')
            if any(intervals[parent][1] > end for parent in self.nodes[identity].finish_dependencies):
                raise ValueError('Finalized communication completion precedes its required input completion')
            if any(intervals[parent][0] > end for parent in self.nodes[identity].launch_dependencies):
                raise ValueError('Finalized communication completion precedes its required peer launch')
        from megatron.core.frontier.planner.memory import Lifetime
        lifetimes = []
        for node in self.nodes.values():
            if node.alloc_bytes:
                if len(node.produces) != 1:
                    raise ValueError('allocated buffers require one explicit value identity')
                value = node.produces[0]
                consumers = [intervals[item.node_id][1] for item in self.nodes.values() if value in item.consumes]
                lifetimes.append(Lifetime(value, intervals[node.node_id][0],
                                          max(consumers, default=intervals[node.node_id][1]), node.alloc_bytes))
        return tuple(lifetimes)
