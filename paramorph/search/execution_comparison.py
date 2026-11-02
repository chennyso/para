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


def compare_execution_reports(left_request, left_prediction, right_request, right_prediction,
                              *, offset=0, limit=8):
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
        raise ValueError('执行比较的分页参数无效')
    required = ('model', 'sequence', 'global_batch', 'precision', 'node_count', 'gpus_per_node')
    missing = [key for key in required if key not in left_request or key not in right_request]
    if missing:
        return dict(status='UNAVAILABLE', reason='missing_workload_identity', missing_fields=missing)
    fixed = (*required, 'checkpoint', 'data', 'optimizer', 'random_seed', 'loss_normalization')
    changed = [key for key in fixed if left_request.get(key) != right_request.get(key)]
    if changed:
        return dict(status='INCOMPARABLE', reason='different_training_workload', changed_fields=changed)
    if left_prediction is None or right_prediction is None:
        return dict(status='UNAVAILABLE', reason='missing_prediction', additional_predictions=0)
    predictions = (left_prediction, right_prediction)
    if any(not row.get('execution_graph', {}).get('tasks') for row in predictions):
        return dict(status='UNAVAILABLE', reason='missing_execution_graph', additional_predictions=0)
    if any(task.get(field) is None for row in predictions
            for task in row['execution_graph']['tasks'].values() for field in ('start_ms', 'end_ms')):
        return dict(status='UNAVAILABLE', reason='incomplete_task_timing', additional_predictions=0)
    revisions = [(row.get('cost_database_revision'), row.get('evidence', {}).get('predictor_revision'))
        for row in predictions]
    comparable = all(all(value is not None for value in row) for row in revisions) and revisions[0] == revisions[1]
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
        return dict(prediction_id=prediction['prediction_id'],
            prediction_status=prediction['prediction_status'],
            step_time_ms=prediction['performance']['step_time_ms'],
            task_kinds=dict(Counter(task['kind'] for task in prediction['execution_graph']['tasks'].values())),
            modeled_exposed_communication_ms=diagnosis.get('modeled_exposed_communication_ms'),
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
    return dict(status='AVAILABLE' if comparable else 'INCOMPARABLE_COST_REVISIONS',
        numeric_comparison_eligible=comparable, before=before, after=after,
        predicted_step_delta_ms=after['step_time_ms'] - before['step_time_ms'] if comparable_times else None,
        module_ownership_changes=movements[offset:offset + limit], module_change_count=len(movements),
        rank_changes=rank_changes, rank_count=len(all_ranks), offset=offset,
        next_offset=offset + limit if offset + limit < max(len(all_ranks), len(movements)) else None,
        additional_predictions=0,
        scope='existing_DAG_predictions_and_modeled_memory_not_paired_training_or_causal_attribution')
