from copy import deepcopy
from dataclasses import replace, asdict
from graphlib import CycleError
import hashlib
import json

from calibration_plan_compiler import CalibrationPlanCompiler
from scheduling.schedule_simulator import ScheduleCostEvaluator, topological_order


class GPCSOptimizer:
    def __init__(self, costs):
        self.costs = costs
        self.history = []

    def digest(self, plan, window):
        return hashlib.sha256(json.dumps(dict(schedule=plan.rank_schedules, window=window,
            pairs=plan.concurrent_pairs), sort_keys=True).encode()).hexdigest()

    def executable(self, plan, window):
        config = plan.parallel_config
        payload = dict(executor='paramorph_task_dag', config=asdict(config), vpp=config.vpp,
            num_regions=config.total_vp_stages, num_microbatches=config.num_microbatches,
            tasks={key: dict(rank=task.rank, kind=task.task_type.value,
                microbatch=task.microbatch_id, region_id=task.region_id,
                dependencies=list(task.dependencies), metadata=deepcopy(task.metadata))
                for key, task in plan.tasks.items()},
            rank_schedules={str(rank): list(sequence) for rank, sequence in plan.rank_schedules.items()})
        return CalibrationPlanCompiler().compile_async_publication(payload, window, require_progress=False)

    def rebuild(self, plan, computation):
        schedules = {}
        for rank, sequence in computation.items():
            outgoing = {task.metadata['producer_task']: key for key, task in plan.tasks.items()
                        if task.rank == rank and task.task_type.value == 'pp_p2p_send'}
            result = []
            for key in sequence:
                received = plan.tasks[key].metadata.get('input_task')
                if received is not None:
                    result.append(received)
                result.append(key)
                if key in outgoing:
                    result.append(outgoing[key])
            result.extend(key for key in plan.rank_schedules[rank]
                          if plan.tasks[key].task_type.value in ('dp_reduce', 'optimizer_step', 'cuda_sync'))
            schedules[rank] = result
        return replace(plan, rank_schedules=schedules, concurrent_pairs=[], task_times={})

    def coupled_moves(self, plan, critical, limit):
        computation = {rank: [key for key in sequence if plan.tasks[key].is_compute]
                       for rank, sequence in plan.rank_schedules.items()}
        _, predecessors, _ = topological_order(plan.tasks)
        ancestors = {}
        def depends(left, right):
            if right not in ancestors:
                pending, found = list(predecessors[right]), set()
                while pending:
                    key = pending.pop()
                    if key not in found:
                        found.add(key)
                        pending.extend(predecessors[key])
                ancestors[right] = found
            return left in ancestors[right]
        emitted = set()
        for key in critical:
            task = plan.tasks[key]
            if not task.is_compute:
                key = task.metadata.get('producer_task')
                if key is None:
                    continue
                task = plan.tasks[key]
            leader = task.rank - task.rank % plan.parallel_config.tp
            if task.rank != leader:
                key = next(candidate for candidate, item in plan.tasks.items()
                           if item.rank == leader and item.task_type == task.task_type
                           and item.microbatch_id == task.microbatch_id and item.region_id == task.region_id)
                task = plan.tasks[key]
            sequence = computation[leader]
            position = sequence.index(key)
            for other_position in (position - 1, position + 1):
                if other_position < 0 or other_position >= len(sequence):
                    continue
                other = sequence[other_position]
                identity = tuple(sorted((key, other)))
                if identity in emitted or depends(key, other) or depends(other, key):
                    continue
                emitted.add(identity)
                changed = deepcopy(computation)
                affected = []
                for rank in range(leader, leader + plan.parallel_config.tp):
                    left, right = changed[rank][position], changed[rank][other_position]
                    if depends(left, right) or depends(right, left):
                        raise ValueError('TP伙伴上的计算关联变换依赖不同')
                    changed[rank][position], changed[rank][other_position] = right, left
                    affected.extend((left, right))
                yield self.rebuild(plan, changed), dict(type='COUPLED_COMPUTE_COMMUNICATION_REORDER',
                    affected_compute=affected, affected_ranks=list(range(leader, leader + plan.parallel_config.tp)),
                    communication_policy='matched_receives_and_producer_adjacent_sends_recompiled')
                if len(emitted) >= limit:
                    return

    def search(self, reference, topology, beam_width=4, iterations=4, candidate_budget=32,
               critical_task_budget=8, receive_windows=(1, 2, 4)):
        if min(beam_width, iterations, candidate_budget, critical_task_budget) < 1:
            raise ValueError('GPCS搜索预算必须大于零')
        evaluator = ScheduleCostEvaluator(self.costs)
        initial = self.rebuild(deepcopy(reference), {
            rank: [key for key in sequence if reference.tasks[key].is_compute]
            for rank, sequence in reference.rank_schedules.items()})
        initial_prediction = evaluator.evaluate(initial, topology)
        executable = self.executable(initial, 2)
        if not executable['distributed_progress_validation']['valid']:
            raise ValueError('GPCS参考计划没有通过进展验证')
        beam = [(initial_prediction.predicted_iteration_time_ms,
                 initial_prediction.predicted_peak_memory_mb, initial, 2, executable)]
        visited = {self.digest(initial, 2)}
        evaluated = 0
        self.history = [dict(candidate_sha256=self.digest(initial, 2), action='REFERENCE',
                             valid=True, prediction=asdict(initial_prediction))]
        for iteration in range(iterations):
            candidates = list(beam)
            for _, _, plan, window, _ in beam:
                prediction = evaluator.evaluate(plan, topology)
                variants = [(plan, proposed, dict(type='CHANGE_RECEIVE_WINDOW',
                            from_window=window, to_window=proposed))
                            for proposed in receive_windows if proposed != window]
                variants.extend((candidate, window, action) for candidate, action in
                    self.coupled_moves(plan, prediction.critical_path_tasks, critical_task_budget))
                for candidate, proposed_window, action in variants:
                    digest = self.digest(candidate, proposed_window)
                    if digest in visited:
                        continue
                    if evaluated >= candidate_budget:
                        break
                    visited.add(digest)
                    evaluated += 1
                    # 循环依赖属于不可行候选，其他实现错误直接传播。
                    try:
                        executable = self.executable(candidate, proposed_window)
                    except CycleError as error:
                        self.history.append(dict(candidate_sha256=digest, iteration=iteration,
                            action=action, valid=False, rejection='dependency_cycle',
                            blocking_cycle=error.args[1]))
                        continue
                    progress = executable['distributed_progress_validation']
                    if not progress['valid']:
                        self.history.append(dict(candidate_sha256=digest, iteration=iteration,
                            action=action, valid=False, rejection='resource_progress', progress=progress))
                        continue
                    outcome = evaluator.evaluate(candidate, topology)
                    self.history.append(dict(candidate_sha256=digest, iteration=iteration,
                        action=action, valid=True, prediction=asdict(outcome), progress=progress))
                    candidates.append((outcome.predicted_iteration_time_ms,
                        outcome.predicted_peak_memory_mb, candidate, proposed_window, executable))
                if evaluated >= candidate_budget:
                    break
            candidates.sort(key=lambda row: (row[0], row[1], row[3]))
            nondominated = []
            for candidate in candidates:
                if not any(other[0] <= candidate[0] and other[1] <= candidate[1]
                           and other[3] <= candidate[3]
                           and (other[0], other[1], other[3]) != (candidate[0], candidate[1], candidate[3])
                           for other in candidates):
                    nondominated.append(candidate)
            beam = nondominated[:beam_width]
            if evaluated >= candidate_budget:
                break
        best = min(beam, key=lambda row: (row[0], row[1], row[3]))
        return dict(plan=best[2], executable_plan=best[4], receive_window=best[3],
                    prediction=evaluator.evaluate(best[2], topology), history=self.history,
                    evaluated_candidates=evaluated, source='measured_cost_global_beam_search',
                    global_optimum_certified=False)
