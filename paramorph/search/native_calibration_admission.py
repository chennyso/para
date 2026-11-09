from copy import deepcopy
import math
import statistics

from contracts import fingerprint
from predictor.profile_database import same_training_plan


def admit_exact_native_calibration(report, request, profile_id, profile):
    if not same_training_plan(profile['request'], request):
        raise ValueError('完整成本校准必须来自完全相同的训练配置')
    if report['prediction_status'] != 'NEEDS_PROFILE':
        return report
    graph = report.get('execution_graph') or {}
    raw_time = report.get('performance', {}).get('raw_resource_aware_step_time_ms')
    missing = report.get('diagnosis', {}).get(
        'resource_aware_replay', {}).get('missing_overlap_profiles', [])
    if not graph.get('tasks') or not missing or raw_time is None:
        return report

    count = request['node_count'] * request['gpus_per_node']
    windows = profile['measured_windows']
    memory_rows = profile['memory']
    if (set(windows) != {str(rank) for rank in range(count)}
            or set(memory_rows) != set(windows)):
        raise ValueError('原生Profile缺少完整Rank的时间窗口或显存记录')

    trace_start = None
    if profile['request'].get('paramorph_trace', False):
        warmup = profile['request']['warmup']
        measured = profile['request']['measured']
        trace_steps = profile['request'].get('paramorph_trace_steps', 1)
        trace_start = warmup + measured - trace_steps

    by_rank = {}
    for rank in range(count):
        indexed = {}
        for row in windows[str(rank)]:
            step = row['step']
            if trace_start is not None and step >= trace_start:
                continue
            elapsed = (row['host_end_ns'] - row['host_start_ns']) / 1_000_000
            if (step in indexed or not math.isfinite(elapsed) or elapsed <= 0):
                raise ValueError('原生Profile训练时间窗口重复或耗时非法')
            indexed[step] = elapsed
        by_rank[rank] = indexed
    steps = set.intersection(*(set(rows) for rows in by_rank.values()))
    if len(steps) < 2:
        return report
    if any(set(rows) != steps for rows in by_rank.values()):
        raise ValueError('原生Profile各Rank的有效稳定迭代集合不同')

    time_samples = [
        max(by_rank[rank][step] for rank in range(count))
        for step in sorted(steps)
    ]
    step_ms = statistics.median(time_samples)
    if not math.isfinite(step_ms) or step_ms <= 0:
        raise ValueError('原生Profile未生成有效的完整迭代时间')

    rank_mapping = request.get('rank_mapping', list(range(count)))
    if sorted(rank_mapping) != list(range(count)):
        raise ValueError('原生Profile关联的物理Rank映射无效')
    measured_memory, headroom, capacities = {}, {}, {}
    for logical in range(count):
        row = memory_rows[str(logical)]
        physical = rank_mapping[logical]
        peak = row['peak']
        capacity = row['capacity']
        if (type(peak) is not int or type(capacity) is not int
                or peak < 0 or capacity <= 0 or peak > capacity):
            raise ValueError('完整原生校准的逐卡显存峰值或容量非法')
        measured_memory[physical] = peak
        capacities[physical] = capacity
        headroom[physical] = capacity - peak

    updated = deepcopy(report)
    updated['prediction_status'] = 'PREDICTED_CALIBRATED'
    updated['validation'].update(
        status='PROJECTABLE',
        reason='完全相同配置的原生训练步骤已经完成成本和显存采集',
    )
    updated['performance'].update(
        step_time_ms=step_ms,
        tokens_per_second=request['global_batch'] * request['sequence'] * 1000 / step_ms,
        ranking_eligible=True,
        prediction_source='SAME_PLAN_NATIVE_CALIBRATION_WINDOWS',
        prediction_interval_ms=None,
    )
    memory = updated['memory']
    memory.update(
        admission='WITHIN_ESTIMATED_CAPACITY',
        status='MEASURED_EXACT_PLAN_TRAINING_PEAK',
        physical_memory_status='MEASURED_SAME_PLAN_TRAINING',
        peak_per_rank_bytes=measured_memory,
        memory_headroom_per_rank_bytes=headroom,
        memory_budget_per_rank_bytes=capacities,
        measured_peak_per_rank_bytes=measured_memory,
        measurement_scope='same_exact_plan_warm_training_steps',
        source_profile_id=profile_id,
    )
    updated['evidence']['native_calibration_admission'] = dict(
        profile_id=profile_id,
        source_run_id=profile['request']['run_id'],
        source_plan_signature=report['plan_signature'],
        measured_step_time_ms=step_ms,
        stable_step_ids=sorted(steps),
        maximum_rank_step_samples_ms=time_samples,
        complete_rank_memory=True,
        remaining_kernel_level_overlap_cost_unknown=True,
        independent_prediction_error_eligible=False,
        scope='same_complete_plan_calibration_only',
    )
    updated['diagnosis']['raw_dag_replay_is_diagnostic_only'] = True
    updated['prediction_id'] = fingerprint(dict(
        original_prediction_id=report['prediction_id'],
        profile_id=profile_id,
        calibration_windows=by_rank,
        measured_memory=measured_memory,
    ))
    return updated
