from collections import defaultdict
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
import heapq
import math

from core_data_models import ExecutionPlan, TaskType
from cost.cost_model import MissingCalibration


@dataclass
class SchedulePrediction:
    predicted_iteration_time_ms: float
    critical_path_tasks: list
    critical_path_length_ms: float
    predicted_peak_memory_mb: float
    exposed_communication_ms: float
    idle_time_ms: float
    confidence: float
    missing_measurements: list
    rank_peak_bytes: dict = field(default_factory=dict)
    task_times: dict = field(default_factory=dict)
    measured_pair_savings_ms: float = 0.0


def topological_order(tasks, extra_edges=()):
    predecessors = {key: set(task.dependencies) for key, task in tasks.items()}
    for source, target in extra_edges:
        if source not in tasks or target not in tasks or source == target:
            raise ValueError('执行计划依赖指向不存在或相同的任务')
        predecessors[target].add(source)
    successors = defaultdict(list)
    for target, values in predecessors.items():
        for source in values:
            if source not in tasks:
                raise ValueError(f'缺少前驱任务：{source}')
            successors[source].append(target)
    indegree = {key: len(values) for key, values in predecessors.items()}
    TopologicalSorter(predecessors).prepare()
    ready = sorted(key for key, count in indegree.items() if count == 0)
    heapq.heapify(ready)
    ordered = []
    while ready:
        key = heapq.heappop(ready)
        ordered.append(key)
        for successor in successors[key]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if len(ordered) != len(tasks):
        raise ValueError('执行计划存在循环依赖或通信顺序冲突')
    return ordered, predecessors, successors


def union_length(intervals):
    total = 0.0
    left = right = None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0.0)


class ScheduleCostEvaluator:
    def __init__(self, cost_model):
        self.cost_model = cost_model

    def simulate_events(self, tasks, predecessors, durations, pairs, config):
        representatives = {key: min(key, pairs[key]) if key in pairs else key for key in tasks}
        units = defaultdict(list)
        for key, representative in representatives.items():
            units[representative].append(key)
        dependencies = {}
        successors = defaultdict(set)
        resources = {}
        unit_durations = {}
        pair_costs = {}
        for unit, members in units.items():
            parents = {representatives[parent] for member in members for parent in predecessors[member]}
            if unit in parents:
                raise ValueError('并发配对的内部任务存在依赖关系')
            dependencies[unit] = parents
            for parent in parents:
                successors[parent].add(unit)
            resources[unit] = {resource for member in members for resource in
                (tasks[member].resource_requirements or (f'rank:{tasks[member].rank}:device',))}
            if len(members) == 2:
                first, second = (tasks[key] for key in members)
                compute, communication = (first, second) if first.is_compute else (second, first)
                if not compute.is_compute or not communication.is_communication:
                    raise ValueError('并发配对必须包含实际计算和通信任务')
                cost = self.cost_model.estimate_pair(compute, communication, config)
                unit_durations[unit] = cost.overlapped_ms
                pair_costs[frozenset(members)] = cost
            else:
                unit_durations[unit] = durations[members[0]]
        remaining = {unit: len(parents) for unit, parents in dependencies.items()}
        ready = {unit for unit, count in remaining.items() if count == 0}
        running = []
        resource_owners = {}
        times = {}
        critical_parent = {}
        clock = 0.0
        completed = 0
        while completed < len(units):
            for unit in sorted(ready):
                if any(resource_owners.get(resource, (0.0, None))[0] > clock for resource in resources[unit]):
                    continue
                members = units[unit]
                causes = [(times[units[parent][0]][1], units[parent][0]) for parent in dependencies[unit]]
                causes.extend(resource_owners[resource] for resource in resources[unit] if resource in resource_owners)
                cause = max(causes, key=lambda item: item[0])[1] if causes else None
                end = clock + unit_durations[unit]
                for member in members:
                    times[member] = (clock, end)
                    critical_parent[member] = cause
                for resource in resources[unit]:
                    resource_owners[resource] = (end, members[0])
                ready.remove(unit)
                heapq.heappush(running, (end, unit))
            if not running:
                raise ValueError('执行事件队列没有可完成的任务，依赖或者资源约束无法继续')
            clock = running[0][0]
            while running and running[0][0] == clock:
                _, unit = heapq.heappop(running)
                completed += 1
                for successor in successors[unit]:
                    remaining[successor] -= 1
                    if remaining[successor] == 0:
                        ready.add(successor)
        return times, critical_parent, pair_costs

    @staticmethod
    def _overlap_gamma(compute, communications, resource_model, missing):
        if not communications:
            return 1.0, []
        records = resource_model.get('compute_communication_slowdowns', [])
        matched = []
        for communication in communications:
            collective = communication.metadata.get('collective',
                communication.metadata.get('operation', communication.task_type.value))
            candidates = [row for row in records
                if row['compute_kind'] in (compute.task_type.value, compute.metadata.get('operation'))
                and row['collective'] in (collective, communication.task_type.value)
                and (row.get('physical_group_ranks') is None
                    or row['physical_group_ranks'] == communication.metadata.get('physical_group_ranks',
                        communication.metadata.get('group_ranks', [])))
                and (row.get('payload_bytes') is None
                    or row['payload_bytes'] == communication.metadata.get('payload_bytes'))
                and compute.rank in communication.metadata.get('participant_ranks',
                    communication.metadata.get('group_ranks', [communication.rank]))]
            if candidates:
                row = max(candidates, key=lambda item: item['gamma'])
                gamma = row['gamma']
                if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 1:
                    raise ValueError('计算通信重叠减速系数必须为不小于一的有限数值')
                matched.append((communication.task_id, float(gamma), row.get('evidence_ids', [])))
            else:
                identity = (compute.task_type.value, collective,
                    tuple(communication.metadata.get('physical_group_ranks',
                        communication.metadata.get('group_ranks', []))),
                    compute.task_id, communication.task_id,
                    communication.metadata.get('payload_bytes'))
                missing.add(identity)
                matched.append((communication.task_id,
                    float(resource_model.get('unmeasured_overlap_gamma', 1.0)), []))
        gamma = max([1.0] + [row[1] for row in matched])
        return gamma, matched

    def simulate_resource_aware_events(self, tasks, predecessors, durations, config,
                                       resource_model=None):
        resource_model = resource_model or {}
        successors = defaultdict(set)
        for target, parents in predecessors.items():
            for parent in parents:
                successors[parent].add(target)
        remaining_dependencies = {key: len(values) for key, values in predecessors.items()}
        ready = {key for key, count in remaining_dependencies.items() if count == 0}
        active = {}
        exclusive_owners = {}
        last_resource_owner = {}
        times = {}
        critical_parent = {}
        completed = set()
        missing_overlap = set()
        bandwidth_intervals = []
        overlap_intervals = []
        task_wait_causes = {}
        pending_resource_waits = {}
        waits_by_owner = defaultdict(set)
        resource_waits = defaultdict(list)
        clock = 0.0

        def resources(key):
            task = tasks[key]
            values = task.resource_requirements or (f'rank:{task.rank}:device',)
            if task.is_communication:
                exclusive = {value for value in values
                    if value.endswith(':compute') or value.startswith('rank:')}
                shareable = set(values) - exclusive
            else:
                exclusive = set(values)
                shareable = set()
            return exclusive, shareable

        def finish(key):
            row = active.pop(key)
            times[key] = (row['start'], clock)
            completed.add(key)
            for waiting in waits_by_owner.pop(key, set()):
                start = pending_resource_waits.pop(waiting)
                if clock > start:
                    target, resource, owner = waiting
                    resource_waits[target].append(dict(task_id=owner, resource=resource,
                        start_ms=start, end_ms=clock))
            for resource in row['exclusive']:
                if exclusive_owners.get(resource) != key:
                    raise ValueError('资源感知Replay的排他资源所有权不一致')
                del exclusive_owners[resource]
                last_resource_owner[resource] = key
            for successor in successors[key]:
                remaining_dependencies[successor] -= 1
                if remaining_dependencies[successor] == 0:
                    ready.add(successor)

        while len(completed) < len(tasks):
            started = True
            while started:
                started = False
                for key in sorted(list(ready)):
                    exclusive, shareable = resources(key)
                    blocked = exclusive & set(exclusive_owners)
                    if blocked:
                        for resource in blocked:
                            owner = exclusive_owners[resource]
                            waiting = (key, resource, owner)
                            pending_resource_waits.setdefault(waiting, clock)
                            waits_by_owner[owner].add(waiting)
                        continue
                    dependency_ready = max((times[parent][1] for parent in predecessors[key]), default=0.0)
                    task_wait_causes[key] = dict(dependency_ready_ms=dependency_ready,
                        start_ms=clock, resource_wait_ms=max(0.0, clock - dependency_ready),
                        dependency_task_ids=sorted(parent for parent in predecessors[key]
                            if dependency_ready > 0 and math.isclose(times[parent][1], dependency_ready,
                                rel_tol=0, abs_tol=1e-9)),
                        resource_blockers=resource_waits[key])
                    causes = [(times[parent][1], parent) for parent in predecessors[key]]
                    causes.extend((times[owner][1], owner) for resource in exclusive
                        if (owner := last_resource_owner.get(resource)) in times)
                    critical_parent[key] = max(causes, default=(0.0, None))[1]
                    ready.remove(key)
                    active[key] = dict(start=clock, remaining=durations[key],
                        exclusive=exclusive, shareable=shareable)
                    for resource in exclusive:
                        exclusive_owners[resource] = key
                    started = True
                zero = [key for key, row in active.items() if row['remaining'] == 0]
                if zero:
                    for key in sorted(zero):
                        finish(key)
                    started = True
            if not active:
                if len(completed) == len(tasks):
                    break
                unresolved = {key: sorted(parent for parent in predecessors[key]
                    if parent not in completed) for key in tasks if key not in completed}
                raise ValueError(dict(reason='资源感知Replay没有可推进的任务',
                    completed_task_count=len(completed), total_task_count=len(tasks),
                    ready_task_ids=sorted(ready)[:16], exclusive_owners=dict(exclusive_owners),
                    unresolved_dependencies={key: value for key, value in list(unresolved.items())[:16]}))
            communication_keys = [key for key in active if tasks[key].is_communication]
            compute_keys = [key for key in active if tasks[key].is_compute]
            share_counts = defaultdict(int)
            for key in communication_keys:
                for resource in active[key]['shareable']:
                    share_counts[resource] += 1
            rates = {}
            overlap_rows = {}
            for key in active:
                task = tasks[key]
                if task.is_communication:
                    contenders = [share_counts[resource] for resource in active[key]['shareable']
                        if share_counts[resource] > 0]
                    rates[key] = 1 / max(contenders, default=1)
                elif task.is_compute:
                    overlapping = [tasks[communication] for communication in communication_keys
                        if task.rank in tasks[communication].metadata.get('participant_ranks',
                            tasks[communication].metadata.get('group_ranks', [tasks[communication].rank]))]
                    gamma, evidence = self._overlap_gamma(task, overlapping, resource_model, missing_overlap)
                    rates[key] = 1 / gamma
                    if overlapping:
                        overlap_rows[key] = dict(gamma=gamma,
                            communication_task_ids=[row.task_id for row in overlapping], evidence=evidence)
                else:
                    rates[key] = 1.0
            if any(rate <= 0 or not math.isfinite(rate) for rate in rates.values()):
                raise ValueError('资源感知Replay任务推进率必须为有限正数')
            delta = min(active[key]['remaining'] / rates[key] for key in active)
            if not math.isfinite(delta) or delta < 0:
                raise ValueError('资源感知Replay产生非法事件间隔')
            interval_end = clock + delta
            for resource, count in share_counts.items():
                if count > 1:
                    bandwidth_intervals.append(dict(resource=resource, start_ms=clock,
                        end_ms=interval_end, concurrent_communications=count,
                        fair_share_fraction=1 / count, task_ids=sorted(key for key in communication_keys
                            if resource in active[key]['shareable'])))
            for key, row in overlap_rows.items():
                overlap_intervals.append(dict(compute_task_id=key, start_ms=clock,
                    end_ms=interval_end, **row))
            for key in active:
                active[key]['remaining'] = max(0.0,
                    active[key]['remaining'] - rates[key] * delta)
            clock = interval_end
            finished = [key for key, row in active.items() if row['remaining'] <= 1e-9]
            if not finished:
                raise ValueError('资源感知Replay事件推进后没有任务完成')
            for key in sorted(finished):
                finish(key)
        diagnostics = dict(
            task_wait_causes=task_wait_causes,
            bandwidth_sharing_intervals=bandwidth_intervals,
            compute_communication_overlap_intervals=overlap_intervals,
            missing_overlap_profiles=[dict(compute_kind=row[0], collective=row[1],
                physical_group_ranks=list(row[2]), compute_task_id=row[3],
                communication_task_id=row[4], payload_bytes=row[5])
                for row in sorted(missing_overlap)],
            bandwidth_policy='dynamic_equal_share_per_contended_physical_resource',
            compute_progress_policy='integral_of_inverse_calibrated_overlap_gamma')
        return times, critical_parent, diagnostics

    def evaluate(self, plan, topology, dag_tasks=None, enforce_memory_limit=True):
        tasks = plan.tasks if dag_tasks is None else dag_tasks
        if not tasks or topology.num_gpus != plan.parallel_config.world_size or not topology.validate():
            raise ValueError('执行计划或硬件设备数量不一致')
        scheduled = [key for sequence in plan.rank_schedules.values() for key in sequence]
        if sorted(scheduled) != sorted(tasks) or len(set(scheduled)) != len(tasks):
            raise ValueError('执行计划必须覆盖全部任务一次')
        pairs = {}
        for left, right in plan.concurrent_pairs:
            if left == right or left in pairs or right in pairs:
                raise ValueError('并发配对重复覆盖任务')
            if left not in tasks or right not in tasks or tasks[left].rank != tasks[right].rank:
                raise ValueError('配对任务必须位于同一设备')
            pairs[left] = right
            pairs[right] = left
        edges = []
        for rank, sequence in plan.rank_schedules.items():
            if any(tasks[key].rank != rank for key in sequence):
                raise ValueError('执行计划包含其他rank的任务')
            by_resource = defaultdict(list)
            for key in sequence:
                # 未实测并发关系的任务共享设备资源。
                resources = tasks[key].resource_requirements or (f'rank:{rank}:device',)
                for resource in resources:
                    by_resource[resource].append(key)
            for sequence in by_resource.values():
                groups = []
                index = 0
                while index < len(sequence):
                    key = sequence[index]
                    group = [key]
                    if index + 1 < len(sequence) and pairs.get(key) == sequence[index + 1]:
                        group.append(sequence[index + 1])
                        index += 1
                    groups.append(group)
                    index += 1
                for previous, following in zip(groups, groups[1:]):
                    edges.extend((left, right) for left in previous for right in following)
        order, predecessors, successors = topological_order(tasks, edges)
        durations = {}
        for key, task in tasks.items():
            if task.duration is not None:
                if not task.metadata.get('duration_evidence'):
                    raise MissingCalibration(f'{key}缺少耗时来源')
                value = task.duration
            else:
                value = self.cost_model.estimate(task, plan.parallel_config).estimated_ms
            if not math.isfinite(value) or value < 0:
                raise ValueError('任务耗时必须为有限非负数值')
            durations[key] = value
        times, critical_parent, pair_costs = self.simulate_events(tasks, predecessors,
            durations, pairs, plan.parallel_config)
        end_key = max(times, key=lambda key: times[key][1])
        makespan = times[end_key][1]
        critical = []
        while end_key is not None:
            critical.append(end_key)
            end_key = critical_parent[end_key]
        critical.reverse()
        plan.tasks, plan.task_times = tasks, times
        peaks = self.cost_model.estimate_memory(plan)
        if enforce_memory_limit and max(peaks.values()) > topology.memory_per_gpu_gb * 2**30:
            raise ValueError('执行计划的峰值显存超过实际设备容量')
        busy = defaultdict(list)
        communication = []
        for key, task in tasks.items():
            if task.is_compute:
                busy[task.rank].append(times[key])
            if task.is_communication and key in critical and key not in pairs:
                communication.append(times[key])
        idle = sum(makespan - union_length(busy[rank]) for rank in plan.rank_schedules)
        pair_exposure = sum(max(0.0, cost.overlapped_ms - cost.compute_alone_ms)
            for keys, cost in pair_costs.items() if keys.intersection(critical))
        composite = any(task.task_type == TaskType.PIPELINE_UPDATE for task in tasks.values())
        exposure = None if composite else union_length(communication) + pair_exposure
        missing = ['阶段成本尚未分解为Kernel通信暴露与空闲时间'] if composite else []
        return SchedulePrediction(makespan, critical, makespan, max(peaks.values()) / 2**20,
            exposure, None if composite else idle, 1.0, missing, peaks, times,
            sum(cost.benefit_ms for cost in pair_costs.values()))
