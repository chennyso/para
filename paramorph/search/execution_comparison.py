from collections import Counter
import math


def _occupied_time(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise ValueError('执行比较需要有限且有序的任务时间')
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def _rank_records(prediction):
    tasks = prediction['execution_graph']['tasks']
    critical = set(prediction.get('diagnosis', {}).get('critical_tasks', []))
    memory = prediction.get('memory', {})
    ranks = {str(task['rank']) for task in tasks.values()}
    records = {}
    owners = {}
    for rank in sorted(ranks, key=int):
        selected = {key: task for key, task in tasks.items() if str(task['rank']) == rank}
        compute = [task for task in selected.values()
            if task['kind'] in ('forward_compute', 'backward_compute')
            or task['kind'].startswith('optimizer_')]
        for task in selected.values():
            for layer in task.get('metadata', {}).get('layer_ids', []):
                owners.setdefault(layer, set()).add(int(rank))
        breakdown = memory.get('breakdown_per_rank', {}).get(rank, {})
        records[rank] = dict(
            task_count=len(selected),
            critical_task_count=sum(key in critical for key in selected),
            compute_occupied_ms=_occupied_time([(task['start_ms'], task['end_ms']) for task in compute]),
            persistent_bytes=breakdown.get('persistent_bytes'),
            dynamic_peak_bytes=breakdown.get('dynamic_peak_bytes'),
            peak_bytes=memory.get('peak_per_rank_bytes', {}).get(rank))
    return records, {key: sorted(value) for key, value in owners.items()}


def operation_totals(prediction):
    tasks = prediction['execution_graph']['tasks']
    critical = set(prediction.get('diagnosis', {}).get('critical_tasks', []))
    groups = {}
    for key, task in tasks.items():
        start, end = task['start_ms'], task['end_ms']
        if not all(math.isfinite(value) for value in (start, end)) or end < start:
            raise ValueError('执行差异需要有限且有序的任务时间')
        ready = max((tasks[parent]['end_ms'] for parent in task.get('dependencies', [])), default=0)
        row = groups.setdefault(task['kind'], dict(tasks=0, gpu_work_ms=0.0,
            ready_wait_sum_ms=0.0, critical_tasks=0))
        row['tasks'] += 1
        row['gpu_work_ms'] += end - start
        row['ready_wait_sum_ms'] += max(0, start - ready)
        row['critical_tasks'] += int(key in critical)
    return groups


def compare_execution_reports(left_request, left_prediction, right_request, right_prediction,
                              *, offset=0, limit=8):
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
        raise ValueError('执行比较的分页参数无效')
    required = ('model', 'sequence', 'global_batch', 'precision', 'node_count', 'gpus_per_node')
    missing = [key for key in required if key not in left_request or key not in right_request]
    if missing:
        return dict(status='UNAVAILABLE', reason='missing_workload_identity', missing_fields=missing)
    fixed = (*required, 'checkpoint', 'data', 'optimizer', 'random_seed', 'loss_normalization',
        'main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype',
        'optimizer_hyperparameters')
    changed = [key for key in fixed if left_request.get(key) != right_request.get(key)]
    if changed:
        return dict(status='INCOMPARABLE', reason='different_training_workload', changed_fields=changed)
    if left_prediction is None or right_prediction is None:
        return dict(status='UNAVAILABLE', reason='missing_prediction', additional_predictions=0)
    predictions = (left_prediction, right_prediction)
    if any(not (row.get('execution_graph') or {}).get('tasks') for row in predictions):
        return dict(status='UNAVAILABLE', reason='missing_execution_graph', additional_predictions=0)
    if any(task.get(field) is None for row in predictions
            for task in row['execution_graph']['tasks'].values() for field in ('start_ms', 'end_ms')):
        return dict(status='UNAVAILABLE', reason='incomplete_task_timing', additional_predictions=0)
    revisions = [(row.get('cost_database_revision'), row.get('evidence', {}).get('predictor_revision'))
        for row in predictions]
    comparable = all(all(value is not None for value in row) for row in revisions) and revisions[0] == revisions[1]
    anchors = [row.get('evidence', {}).get('compatible_anchor_id') for row in predictions]
    same_calibration = anchors[0] == anchors[1]
    comparable = comparable and same_calibration
    comparable = comparable and all(row['performance'].get('ranking_eligible', True) for row in predictions)
    left_ranks, left_owners = _rank_records(left_prediction)
    right_ranks, right_owners = _rank_records(right_prediction)
    all_ranks = sorted(left_ranks.keys() | right_ranks.keys(), key=int)
    page = all_ranks[offset:offset + limit]
    metrics = ('task_count', 'critical_task_count', 'compute_occupied_ms',
        'persistent_bytes', 'dynamic_peak_bytes', 'peak_bytes')
    rank_changes = []
    for rank in page:
        before, after = left_ranks.get(rank, {}), right_ranks.get(rank, {})
        delta = {key: after[key] - before[key] for key in metrics
            if comparable and before.get(key) is not None and after.get(key) is not None}
        rank_changes.append(dict(rank=int(rank), before=before, after=after, delta=delta))
    def summary(prediction):
        evidence = prediction.get('evidence', {})
        diagnosis = prediction.get('diagnosis', {})
        peaks = prediction.get('memory', {}).get('peak_per_rank_bytes') or {}
        p2p = [task for task in prediction['execution_graph']['tasks'].values()
            if task['kind'] in ('pp_p2p_send', 'pp_p2p_recv')]
        return dict(prediction_id=prediction['prediction_id'],
            prediction_status=prediction['prediction_status'],
            step_time_ms=prediction['performance']['step_time_ms'],
            task_kinds=dict(Counter(task['kind'] for task in prediction['execution_graph']['tasks'].values())),
            modeled_exposed_communication_ms=diagnosis.get('modeled_exposed_communication_ms'),
            estimated_peak_bytes=max(peaks.values(), default=None),
            critical_path_task_count=len(diagnosis.get('critical_tasks', [])),
            p2p_timeline_samples=[{key: task.get(key) for key in
                ('kind', 'rank', 'start_ms', 'end_ms', 'resources')} for task in p2p[:3]],
            exposed_communication_scope=diagnosis.get('exposed_communication_scope'),
            memory_admission=prediction.get('memory', {}).get('admission'),
            module_cost_source=evidence.get('module_profile_selection'),
            whole_step_anchor=evidence.get('exact_profile_anchor'),
            blind=evidence.get('blind'))
    before, after = summary(left_prediction), summary(right_prediction)
    comparable_times = comparable and all(row['step_time_ms'] is not None
        and row['prediction_status'] != 'NEEDS_PROFILE' for row in (before, after))
    modules = sorted(left_owners.keys() | right_owners.keys())
    movements = [dict(module_id=module, before_ranks=left_owners.get(module, []),
        after_ranks=right_owners.get(module, [])) for module in modules
        if left_owners.get(module, []) != right_owners.get(module, [])]
    operations = []
    if comparable:
        left_operations, right_operations = map(operation_totals, predictions)
        for kind in sorted(left_operations.keys() | right_operations.keys()):
            empty = dict(tasks=0, gpu_work_ms=0.0, ready_wait_sum_ms=0.0, critical_tasks=0)
            old, new = left_operations.get(kind, empty), right_operations.get(kind, empty)
            operations.append(dict(operation=kind, before=old, after=new,
                delta={name: new[name] - old[name] for name in empty}))
    def difference(key):
        return (after[key] - before[key] if comparable and before[key] is not None and after[key] is not None else None)

    return dict(status='AVAILABLE' if comparable else 'INCOMPARABLE_CALIBRATION' if not same_calibration
            else 'NEEDS_PROFILE' if any(not row['performance'].get('ranking_eligible', True) for row in predictions)
            else 'INCOMPARABLE_COST_REVISIONS',
        calibration_anchors=anchors,
        numeric_comparison_eligible=comparable, before=before, after=after,
        predicted_step_delta_ms=after['step_time_ms'] - before['step_time_ms'] if comparable_times else None,
        exposed_communication_delta_ms=difference('modeled_exposed_communication_ms'),
        peak_memory_delta_bytes=difference('estimated_peak_bytes'),
        prediction_uncertainty=[row['performance'].get('prediction_uncertainty') for row in predictions],
        missing_cost_evidence=[row.get('evidence', {}).get('missing_profiles', []) for row in predictions],
        module_ownership_changes=movements[offset:offset + limit], module_change_count=len(movements),
        rank_changes=rank_changes, rank_count=len(all_ranks), offset=offset,
        operation_changes=operations,
        operation_scope='sum_of_modeled_task_intervals_and_ready_delays_not_additive_iteration_time',
        next_offset=offset + limit if offset + limit < max(len(all_ranks), len(movements)) else None,
        additional_predictions=0,
        scope='existing_DAG_predictions_and_modeled_memory_not_paired_training_or_causal_attribution')
