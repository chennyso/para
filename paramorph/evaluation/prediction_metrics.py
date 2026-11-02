from collections import Counter
from datetime import datetime
import itertools
import math
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def comparable_identity(trial):
    plan = trial['plan']
    return tuple(plan[key] for key in
                 ('model_fingerprint', 'hardware_fingerprint', 'workload_fingerprint', 'backend_fingerprint'))


def measured_trials(trials):
    return [trial for trial in trials if trial.get('measurement')
            and trial['measurement'].get('status') == 'PASS'
            and trial['measurement'].get('performance_eligible') is True
            and _finite(trial['measurement'].get('step_seconds'), positive=True)
            and _finite(trial['measurement'].get('tokens_per_second'), positive=True)]


def prediction_samples(trials):
    return [trial for trial in measured_trials(trials) if trial.get('prediction')
            and _finite(trial['prediction'].get('performance', {}).get('step_time_ms'), positive=True)]


def _source_ids(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('evidence_ids', 'evidence_refs', 'calibration_evidence_ids', 'profile_ids') and isinstance(child, list):
                yield from (item for item in child if isinstance(item, str))
            elif key in ('compatible_anchor_id', 'profile_id', 'measured_run_id') and isinstance(child, str):
                yield child
            else:
                yield from _source_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_ids(child)


def prediction_provenance(trial, evidence_order=None):
    prediction, measurement = trial.get('prediction'), trial.get('measurement')
    if prediction is None or measurement is None:
        return dict(status='UNKNOWN', independent=False, reason='缺少预测或者真实测量')
    evidence = prediction.get('evidence', {})
    run_id = measurement.get('run_id')
    references = sorted(set(_source_ids(evidence)))
    candidate_sources = [identifier for identifier in references if run_id and
                         (identifier == run_id or identifier.startswith(run_id + ':'))]
    if candidate_sources:
        return dict(status='SELF_CALIBRATED', independent=False,
                    candidate_measurement_sources=candidate_sources,
                    reason='预测成本或者Anchor引用当前候选的测量记录')
    recorded_before_result = False
    if evidence_order is not None:
        prediction_order = evidence_order.get(prediction.get('prediction_id'))
        measurement_order = evidence_order.get(run_id)
        recorded_before_result = (prediction_order is not None and measurement_order is not None
                                  and prediction_order < measurement_order)
    provenance = prediction.get('provenance', {})
    generated_at = provenance.get('generated_at', prediction.get('generated_at'))
    measurement_started_at = measurement.get('started_at', measurement.get('gpu_started_at'))
    recorded_before_start = False
    if generated_at and measurement_started_at:
        generated = datetime.fromisoformat(generated_at)
        started = datetime.fromisoformat(measurement_started_at)
        if generated.tzinfo is None or started.tzinfo is None:
            raise ValueError('预测与GPU开始时间需要明确时区')
        recorded_before_start = generated < started
    excludes_candidate = evidence.get('calibration_excludes_candidate_measurement') is True
    theoretical_only = (evidence.get('cost_mode') == 'analytical_only'
                        and bool(evidence.get('costs')) and all(
                            row.get('status') == 'ANALYTICAL'
                            and (row.get('calibration') or {}).get('kind') == 'theoretical_peak'
                            and not row.get('evidence_ids') for row in evidence['costs']))
    excludes_candidate = excludes_candidate or theoretical_only
    blind = (evidence.get('blind') is True
             and evidence.get('candidate_plan_excluded_from_calibration') is True)
    if excludes_candidate and (recorded_before_start or recorded_before_result):
        return dict(status='BLIND' if blind else 'PRE_RESULT_CANDIDATE_EXCLUDED', independent=True,
                    candidate_measurement_excluded=True, recorded_before_start=recorded_before_start,
                    recorded_before_result=recorded_before_result, blind=blind,
                    scope='held_out_candidate_plan' if blind else 'candidate_benchmark_result_excluded')
    return dict(status='UNKNOWN', independent=False, recorded_before_start=recorded_before_start,
                recorded_before_result=recorded_before_result,
                candidate_measurement_excluded=excludes_candidate,
                reason='缺少候选测量排除记录或者预测前置记录')


def _memory_comparisons(trial):
    memory = trial['prediction'].get('memory', {})
    measurement = trial['measurement']
    scopes = (
        ('torch_cuda_allocated', ('allocated_peak_per_rank_bytes', 'peak_allocated_per_rank_bytes'), 'peak_allocated_gib'),
        ('torch_cuda_reserved', ('reserved_peak_per_rank_bytes', 'peak_reserved_per_rank_bytes'), 'peak_reserved_gib'),
        ('device_used', ('device_used_peak_per_rank_bytes', 'peak_device_used_per_rank_bytes'), 'peak_device_used_gib'),
    )
    declared_scope = memory.get('peak_memory_scope', memory.get('comparison_scope'))
    result = []
    for scope, fields, measured_field in scopes:
        ranks = next((memory[field] for field in fields if memory.get(field)), None)
        if ranks is None and declared_scope == scope:
            ranks = memory.get('peak_per_rank_bytes')
        if not isinstance(ranks, dict) or not ranks or any(not _finite(value) for value in ranks.values()):
            continue
        request = trial['plan']['backend_config']
        world = request.get('node_count', 0) * request.get('gpus_per_node', 0)
        if world and len(ranks) != world:
            continue
        actual = measurement.get(measured_field)
        if not _finite(actual):
            continue
        result.append(dict(scope=scope, predicted_gib=max(ranks.values()) / 2**30,
                           measured_gib=actual, error_gib=max(ranks.values()) / 2**30 - actual,
                           measurement_field=measured_field))
    return result


def prediction_metrics(trials):
    tested = prediction_samples(trials)
    groups = {}
    for trial in tested:
        groups.setdefault(comparable_identity(trial), []).append(trial)
    all_pairs = [pair for group in groups.values() for pair in itertools.combinations(group, 2)]
    pairs = [(left, right) for left, right in all_pairs
             if left['prediction']['performance']['step_time_ms'] != right['prediction']['performance']['step_time_ms']
             and left['measurement']['step_seconds'] != right['measurement']['step_seconds']]
    correct = sum((left['prediction']['performance']['step_time_ms'] - right['prediction']['performance']['step_time_ms'])
                  * (left['measurement']['step_seconds'] - right['measurement']['step_seconds']) > 0
                  for left, right in pairs)
    top_k = {}
    for k in (1, 2, 3, 5):
        recalls = []
        for group in groups.values():
            if len(group) < 2:
                continue
            count = min(k, len(group))
            predicted = {trial['plan_hash'] for trial in sorted(group,
                         key=lambda row: (row['prediction']['performance']['step_time_ms'], row['plan_hash']))[:count]}
            actual = {trial['plan_hash'] for trial in sorted(group,
                      key=lambda row: (row['measurement']['step_seconds'], row['plan_hash']))[:count]}
            recalls.append(len(predicted & actual) / count)
        top_k[str(k)] = statistics.mean(recalls) if recalls else None
    intervals = []
    for trial in tested:
        interval = trial['prediction']['performance'].get('prediction_interval_ms')
        if (isinstance(interval, (list, tuple)) and len(interval) == 2
                and all(_finite(value) for value in interval) and interval[0] <= interval[1]):
            intervals.append(trial)
    covered = sum(trial['prediction']['performance']['prediction_interval_ms'][0]
                  <= trial['measurement']['step_seconds'] * 1000
                  <= trial['prediction']['performance']['prediction_interval_ms'][1] for trial in intervals)
    memory_rows = [(trial['trial_id'], row) for trial in tested for row in _memory_comparisons(trial)]
    memory_by_scope = {}
    for scope in sorted({row['scope'] for _, row in memory_rows}):
        errors = [abs(row['error_gib']) for _, row in memory_rows if row['scope'] == scope]
        memory_by_scope[scope] = dict(sample_count=len(errors), mae_gib=statistics.mean(errors),
                                     max_error_gib=max(errors))
    allocated = memory_by_scope.get('torch_cuda_allocated', {})
    latencies = [trial['prediction']['evaluation_seconds'] for trial in trials
                 if trial.get('prediction') is not None
                 and _finite(trial['prediction'].get('evaluation_seconds'))]
    provenance = [trial['prediction_provenance_assessment'] if 'prediction_provenance_assessment' in trial
                  else prediction_provenance(trial) for trial in tested]
    return dict(sample_count=len(tested), comparable_groups=len(groups),
        comparison_scope='recorded_measured_candidates_with_equal_model_hardware_workload_backend',
        independent_prediction_count=sum(row['independent'] for row in provenance),
        prediction_provenance_counts=dict(Counter(row['status'] for row in provenance)),
        step_time_mape=statistics.mean(abs(trial['prediction']['performance']['step_time_ms']
            / (trial['measurement']['step_seconds'] * 1000) - 1) for trial in tested) if tested else None,
        pairwise_count=len(pairs), pairwise_tie_count=len(all_pairs) - len(pairs),
        pairwise_ranking_accuracy=correct / len(pairs) if pairs else None,
        top_k_recall=top_k, top_k_scope='tested_candidates_only',
        peak_memory_comparison_count=len(memory_rows), peak_memory_metric_scope='torch_cuda_allocated',
        peak_memory_mae_gib=allocated.get('mae_gib'), peak_memory_max_error_gib=allocated.get('max_error_gib'),
        peak_memory_errors_by_scope=memory_by_scope,
        peak_memory_comparisons=[dict(trial_id=identifier, **row) for identifier, row in memory_rows],
        peak_memory_uncompared_count=len(tested) - len({identifier for identifier, _ in memory_rows}),
        peak_memory_comparison_rule='explicit_equal_memory_scope; physical_reserve_excluded_from_allocator_comparison',
        prediction_interval_count=len(intervals),
        prediction_interval_coverage=covered / len(intervals) if intervals else None,
        prediction_interval_scopes=dict(Counter(trial['prediction']['performance'].get('interval_scope', 'UNSPECIFIED')
                                               for trial in intervals)),
        prediction_p50_seconds=percentile(latencies, .5) if latencies else None,
        prediction_p95_seconds=percentile(latencies, .95) if latencies else None,
        prediction_latency_scope='latest_persisted_prediction_per_trial; total_cpu_cost_in_usage_ledger',
        evaluations_per_second=len(latencies) / sum(latencies) if latencies and sum(latencies) > 0 else None,
        oom_validation_count=0, oom_accuracy=None,
        oom_validation_scope='requires_explicit_runtime_oom_records')
