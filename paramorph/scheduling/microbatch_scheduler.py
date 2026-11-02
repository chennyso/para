from collections import defaultdict
from dataclasses import replace
import heapq

from core_data_models import ExecutionPlan
from cost.cost_model import MissingCalibration
from scheduling.schedule_simulator import topological_order, ScheduleCostEvaluator


class MicrobatchScheduler:
    def __init__(self, cost_model):
        self.cost_model = cost_model

    def schedule(self, dag, config, memory_budget_mb=None, persistent_bytes=None, regions=(),
                 pair_candidates=(), topology=None):
        if memory_budget_mb is None or persistent_bytes is None:
            raise ValueError('调度需要实际显存容量和常驻显存测量')
        if any(not task.metadata.get('memory_evidence') for task in dag.tasks.values()):
            raise MissingCalibration('调度任务缺少实际激活与Workspace显存成本')
        ordered, predecessors, successors = topological_order(dag.tasks)
        durations = {key: self.cost_model.estimate(task, config).estimated_ms
                     for key, task in dag.tasks.items()}
        upward = {}
        for key in reversed(ordered):
            upward[key] = durations[key] + max((upward[successor]
                for successor in successors[key]), default=0.0)
        indegree = {key: len(values) for key, values in predecessors.items()}
        collective_units = defaultdict(list)
        for key, task in dag.tasks.items():
            collective_units[(task.rank // config.tp, task.task_type,
                              task.microbatch_id, task.region_id,
                              task.metadata.get('direction'))].append(key)
        bundles = {}
        for keys in collective_units.values():
            if config.tp > 1 and len(keys) != config.tp:
                raise ValueError('计算通信调度单元没有完整覆盖TP组')
            for key in keys:
                bundles[key] = keys
        ready = [(-upward[key], key) for key, count in indegree.items() if count == 0]
        heapq.heapify(ready)
        schedules = defaultdict(list)
        allocated = dict(persistent_bytes)
        count = 0
        scheduled = set()
        while ready:
            deferred = []
            selected = None
            while ready:
                priority, key = heapq.heappop(ready)
                if key in scheduled:
                    continue
                keys = bundles.get(key, [key])
                if any(indegree[member] for member in keys):
                    deferred.append((priority, key))
                    continue
                if all(allocated[dag.tasks[member].rank] + dag.tasks[member].activation_delta_bytes
                       + dag.tasks[member].workspace_bytes <= memory_budget_mb * 2**20 for member in keys):
                    selected = keys
                    break
                deferred.append((priority, key))
            for value in deferred:
                heapq.heappush(ready, value)
            if selected is None:
                if count == len(dag.tasks):
                    break
                raise ValueError('Ready任务均超过显存限制，调度无法继续')
            for key in selected:
                task = dag.tasks[key]
                schedules[task.rank].append(key)
                allocated[task.rank] += task.activation_delta_bytes
                if allocated[task.rank] < persistent_bytes[task.rank]:
                    raise ValueError('激活释放没有对应申请')
                scheduled.add(key)
                count += 1
            for key in selected:
                for successor in successors[key]:
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        heapq.heappush(ready, (-upward[successor], successor))
        if count != len(dag.tasks):
            raise ValueError('调度没有完整覆盖任务图')
        for rank, sequence in schedules.items():
            outgoing = {dag.tasks[key].metadata['producer_task']: key for key in sequence
                        if dag.tasks[key].task_type.value == 'pp_p2p_send'
                        and dag.tasks[key].metadata.get('producer_task') is not None}
            anchored_sends = set(outgoing.values())
            compiled = []
            for key in sequence:
                if key in anchored_sends:
                    continue
                compiled.append(key)
                if key in outgoing:
                    compiled.append(outgoing[key])
            schedules[rank] = compiled
        plan = ExecutionPlan(config, list(regions), dict(schedules), tasks=dag.tasks,
            persistent_bytes=persistent_bytes, cost_model_version=self.cost_model.version)
        if not pair_candidates:
            return plan
        if topology is None:
            raise ValueError('并发调度比较需要实际设备拓扑')
        evaluator = ScheduleCostEvaluator(self.cost_model)
        best_time = evaluator.evaluate(plan, topology).predicted_iteration_time_ms
        paired = set()
        for compute_id, communication_id in pair_candidates:
            compute, communication = dag.tasks[compute_id], dag.tasks[communication_id]
            if not compute.is_compute or not communication.is_communication or compute.rank != communication.rank:
                raise ValueError('并发候选必须包含同一设备的计算与通信')
            cost = self.cost_model.estimate_pair(compute, communication, config)
            if cost.benefit_ms <= 0 or compute_id in paired or communication_id in paired:
                continue
            sequence = plan.rank_schedules[compute.rank]
            if abs(sequence.index(compute_id) - sequence.index(communication_id)) != 1:
                continue
            if compute_id in communication.dependencies or communication_id in compute.dependencies:
                continue
            candidate = replace(plan, concurrent_pairs=plan.concurrent_pairs + [(compute_id, communication_id)],
                                task_times={})
            predicted = evaluator.evaluate(candidate, topology).predicted_iteration_time_ms
            if predicted < best_time:
                plan, best_time = candidate, predicted
                paired.update((compute_id, communication_id))
        return plan
