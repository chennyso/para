from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from time import perf_counter

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class UpdateSchedule:
    status: str
    intervals_us: dict[str, tuple[float, float]]
    predicted_step_us: float | None
    lower_bound_us: float | None
    planning_seconds: float


def schedule_update_dag(dag, memory_limits=None, persistent_bytes=None, timeout_seconds=10.0):
    if not dag.nodes or timeout_seconds <= 0:
        raise ValueError('Update scheduling requires a nonempty DAG and positive solve budget')
    if (memory_limits is None) != (persistent_bytes is None):
        raise ValueError('Memory scheduling requires both actual capacities and persistent allocations')
    scale = 1000
    durations = {identity: ceil(node.duration_us * scale) for identity, node in dag.nodes.items()}
    horizon = max(1, sum(durations.values()), max((ceil(node.deadline_us * scale)
        for node in dag.nodes.values() if node.deadline_us is not None), default=0))
    model = cp_model.CpModel()
    starts, ends, intervals = {}, {}, {}
    for identity in dag.order:
        starts[identity] = model.new_int_var(0, horizon, identity + ':start')
        ends[identity] = model.new_int_var(0, horizon, identity + ':finish')
        intervals[identity] = model.new_interval_var(starts[identity], durations[identity], ends[identity], identity)
    resources = defaultdict(list)
    for identity, node in dag.nodes.items():
        for predecessor in node.dependencies:
            model.add(starts[identity] >= ends[predecessor])
        for predecessor in node.finish_dependencies:
            model.add(ends[identity] >= ends[predecessor])
        for predecessor in node.launch_dependencies:
            model.add(ends[identity] >= starts[predecessor])
        if node.deadline_us is not None:
            model.add(ends[identity] <= int(node.deadline_us * scale))
        for resource in node.resource.resources:
            if resource != 'GPU_MEMORY':
                resources[(node.resource.device, resource)].append(intervals[identity])
    for members in resources.values():
        model.add_no_overlap(members)
    if memory_limits is not None:
        devices = {node.resource.device for node in dag.nodes.values()}
        if not devices <= set(memory_limits) or not devices <= set(persistent_bytes):
            raise ValueError('Every scheduled device requires an actual memory capacity and persistent allocation')
        memory_intervals, demands = defaultdict(list), defaultdict(list)
        for identity, node in dag.nodes.items():
            device = node.resource.device
            if node.workspace_bytes:
                memory_intervals[device].append(intervals[identity])
                demands[device].append(node.workspace_bytes)
            if node.alloc_bytes:
                if len(node.produces) != 1:
                    raise ValueError('Scheduled allocation requires one explicit storage identity')
                storage = node.produces[0]
                release, = (item for item in dag.nodes.values() if item.node_type == 'BUFFER_RELEASE' and storage in item.consumes)
                for consumer in dag.nodes.values():
                    if storage in consumer.consumes and consumer.node_type != 'BUFFER_RELEASE':
                        model.add(starts[release.node_id] >= ends[consumer.node_id])
                duration = model.new_int_var(0, horizon, storage + ':lifetime')
                lifetime = model.new_interval_var(starts[identity], duration, ends[release.node_id], storage)
                memory_intervals[device].append(lifetime)
                demands[device].append(node.alloc_bytes)
        for device in devices:
            if persistent_bytes[device] < 0 or memory_limits[device] <= 0:
                raise ValueError('Actual device memory declarations require valid byte counts')
            capacity = memory_limits[device] - persistent_bytes[device]
            if capacity < 0:
                model.add(False)
            elif memory_intervals[device]:
                model.add_cumulative(memory_intervals[device], demands[device], capacity)
    finish = model.new_int_var(0, horizon, 'complete_update_finish')
    model.add_max_equality(finish, list(ends.values()))
    model.minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    solver.parameters.num_search_workers = 1
    started = perf_counter()
    status = solver.solve(model)
    elapsed = perf_counter() - started
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return UpdateSchedule(solver.status_name(status), {}, None, None, elapsed)
    result = {identity: (solver.value(starts[identity]) / scale, solver.value(ends[identity]) / scale) for identity in dag.order}
    dag.lifetimes(result)
    return UpdateSchedule(solver.status_name(status), result, solver.value(finish) / scale,
        solver.best_objective_bound / scale, elapsed)
