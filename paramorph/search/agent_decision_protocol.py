import math
from statistics import median
from typing import Literal

from pydantic import Field

from contracts import Contract
from predictor.profile_database import plan_signature, same_training_plan


class CandidateVerification(Contract):
    trial_id: str = Field(min_length=1)
    prediction_evidence_id: str = Field(min_length=1)
    prediction_status: str = Field(min_length=1)
    memory_admission: str = Field(min_length=1)
    recommended_action: Literal['PROFILE', 'LOAD_VALIDATE', 'BENCHMARK', 'DEFER']
    reason: str = Field(min_length=1)


class CandidateReview(Contract):
    candidate_verifications: list[CandidateVerification]
    evidence_refs: list[str] = Field(min_length=1)


def review_conflicts(review, catalog):
    expected = {row['trial_id']: row for row in catalog}
    submitted = {row.trial_id: row for row in review.candidate_verifications}
    problems = []
    if len(submitted) != len(review.candidate_verifications) or submitted.keys() != expected.keys():
        problems.append(dict(expected=sorted(expected), submitted=sorted(submitted)))
    for identifier in sorted(submitted.keys() & expected.keys()):
        observed = submitted[identifier].model_dump()
        row = expected[identifier]
        fields = ('prediction_evidence_id', 'prediction_status', 'memory_admission')
        if (any(row[key] != observed[key] for key in fields)
                or observed['recommended_action'] not in row['allowed_actions']):
            problems.append(dict(trial_id=identifier, expected=row, submitted=observed))
    return problems


def experiment_conflicts(decision, observation):
    scope = set(observation['allowed_benchmark_ids'])
    profiles = {row['request_id'] for row in observation['profiling_requests']}
    load = set(observation['allowed_load_validation_ids'])
    choices = (('accepted_candidate_ids', scope), ('accepted_profiling_request_ids', profiles),
               ('load_validation_candidate_ids', load))
    conflicts = []
    for name, allowed in choices:
        values = getattr(decision, name)
        invalid = set(values) - allowed
        if invalid or len(values) != len(set(values)):
            conflicts.append(dict(field=name, invalid_ids=sorted(invalid), unique_ids_required=True))
    retained = decision.retained_candidate_ids
    unknown_retained = set(retained) - set(observation['active_candidate_ids'])
    if unknown_retained or len(retained) != len(set(retained)):
        conflicts.append(dict(field='retained_candidate_ids', invalid_ids=sorted(unknown_retained),
            unique_ids_required=True))
    selected = any(getattr(decision, name) for name, _ in choices)
    if decision.accepted_candidate_ids and (decision.accepted_profiling_request_ids
            or decision.load_validation_candidate_ids):
        conflicts.append(dict(reason='成本或加载验证完成后，重新评价候选再选择正式训练'))
    if decision.exploration_candidate_ids:
        conflicts.append(dict(field='exploration_candidate_ids', reason='严格图引导搜索不允许无预测的正式训练'))
    if selected:
        if decision.stop_reason != 'CONTINUE':
            conflicts.append(dict(field='stop_reason', expected='CONTINUE'))
    else:
        allowed_stops = observation['allowed_stop_reasons']
        if decision.stop_reason not in allowed_stops:
            conflicts.append(dict(field='stop_reason', allowed=allowed_stops,
                runnable_profile_ids=sorted(profiles), runnable_benchmark_ids=sorted(scope)))
    return conflicts


def exact_native_observation(request, profile_id, profile):
    if not same_training_plan(profile['request'], request):
        return None
    count = request['node_count'] * request['gpus_per_node']
    ranks = {str(rank) for rank in range(count)}
    windows, memory = profile.get('measured_windows', {}), profile.get('memory', {})
    if set(windows) != ranks or set(memory) != ranks:
        return None
    measured_request = profile['request']
    warmup, measured = measured_request['warmup'], measured_request['measured']
    trace_steps = measured_request.get('paramorph_trace_steps', 1) if measured_request.get('paramorph_trace') else 0
    stable_steps = set(range(warmup, warmup + measured - trace_steps))
    mapping = request.get('rank_mapping', list(range(count)))
    if sorted(mapping) != list(range(count)):
        raise ValueError('原生校准的 Rank 映射不完整')
    peaks, capacities, durations = {}, {}, {}
    for logical, physical in enumerate(mapping):
        row = memory[str(logical)]
        peak, capacity = row['peak'], row['capacity']
        if type(peak) is not int or type(capacity) is not int or not 0 < peak <= capacity:
            raise ValueError('原生校准的逐卡峰值或容量无效')
        peaks[physical], capacities[physical] = peak, capacity
        durations[physical] = {}
        for sample in windows[str(logical)]:
            step = sample['step']
            if step not in stable_steps:
                continue
            elapsed = (sample['host_end_ns'] - sample['host_start_ns']) / 1e6
            if not math.isfinite(elapsed) or elapsed <= 0 or step in durations[physical]:
                raise ValueError('原生校准时间窗口无效或重复')
            durations[physical][step] = elapsed
    clean = len(stable_steps) >= 2 and all(row.keys() == stable_steps for row in durations.values())
    step_ms = median([max(row[step] for row in durations.values()) for step in stable_steps]) if clean else None
    return dict(profile_id=profile_id, source_run_id=measured_request['run_id'],
        plan_signature=plan_signature(request), step_time_ms=step_ms,
        tokens_per_second=None if step_ms is None else request['global_batch'] * request['sequence'] * 1000 / step_ms,
        stable_step_count=len(stable_steps) if clean else 0, peak_per_rank_bytes=peaks,
        capacity_per_rank_bytes=capacities, scope='SAME_CONFIGURATION_CALIBRATION',
        independent_prediction_error_eligible=False)
