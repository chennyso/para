import math
from statistics import mean


WORKLOAD_FIELDS = ('model', 'checkpoint', 'data', 'training_data_contract', 'sequence', 'global_batch',
    'precision', 'node_count', 'gpus_per_node', 'hardware_fingerprint', 'backend_fingerprint',
    'optimizer', 'optimizer_hyperparameters', 'optimizer_schedule', 'main_params_dtype',
    'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype', 'warmup', 'measured')
STRATUM_FIELDS = ('execution_backend', 'tp', 'pp', 'cp', 'ep', 'vpp', 'pipeline_schedule')


def budgeted_selection_values(options, *, gpu_execution_enabled, full_training_trials):
    ids = [row['candidate_id'] for row in options]
    first_probe = (bool(ids) and gpu_execution_enabled and full_training_trials == 0
        and all(row['expected_improvement_per_gpu_hour'] is None for row in options))
    return (ids if first_probe else ['DEFER', *ids]), first_probe


def same_condition(first, second, fields):
    return all(first.get(key) == second.get(key) for key in fields)


def _quantile(values, fraction):
    position = fraction * (len(values) - 1)
    first, last = math.floor(position), math.ceil(position)
    return values[first] + (position - first) * (values[last] - values[first])


def experiment_value(request, prediction, history, best, estimated_seconds, *, action='BENCHMARK'):
    """只用同组已完成的真实测量估计残差、通过率和用量。"""
    prediction = prediction or {}
    step = prediction.get('performance', {}).get('step_time_ms')
    step = step if step is not None and math.isfinite(step) and step > 0 else None
    residuals, costs, runs, validity, seen = [], [], [], [], set()
    world = request['node_count'] * request['gpus_per_node']
    for trial in history:
        other = trial['plan']['backend_config']
        if not same_condition(request, other, WORKLOAD_FIELDS + STRATUM_FIELDS):
            continue
        measured = trial.get('measurement') or {}
        if action == 'LOAD_VALIDATION':
            measured = (trial.get('validation') or {}).get('load_validation') or {}
        run = measured.get('run_id')
        if not run or run in seen:
            continue
        seen.add(run)
        seconds = measured.get('gpu_allocation_seconds')
        if seconds is not None and math.isfinite(seconds) and seconds > 0:
            costs.append(seconds * world / 3600)
        success = measured.get('status') == 'PASS' and (action == 'LOAD_VALIDATION'
            or measured.get('performance_eligible') is True)
        if success or measured.get('status') in ('CUDA_OOM', 'HOST_MEMORY_BUDGET_EXCEEDED'):
            validity.append(success)
        snapshot = measured.get('prediction_at_execution') or {}
        if action == 'BENCHMARK' and success and snapshot.get('predictor_revision') == prediction.get(
                'evidence', {}).get('predictor_revision') and snapshot.get('step_time_ms') is not None:
            residuals.append(measured['step_seconds'] * 1000 - snapshot['step_time_ms'])
            runs.append(run)
    p_valid = (sum(validity) + 1) / (len(validity) + 2) if len(validity) >= 3 else None
    cost = mean(costs) if costs else world * estimated_seconds / 3600
    result = dict(predicted_step_ms=step, throughput_interval=None, predicted_gain=None,
        expected_gain=None, expected_improvement_per_gpu_hour=None, p_valid=p_valid,
        validity_observations=len(validity), expected_gpu_hours=cost,
        cost_basis='same_workload_actual_gpu_receipts' if costs else 'declared_budget_estimate',
        cost_observations=len(costs), calibration_runs=runs, residual_observations=len(residuals),
        uncertainty='UNKNOWN', measurement_information='训练结果可更新吞吐与容量记录')
    if action != 'BENCHMARK':
        result['measurement_information'] = '核实检查点、有限梯度和优化器更新；单步用时不代表正式吞吐'
    elif step is not None and len(residuals) >= 3 and all(step + value > 0 for value in residuals):
        throughput = sorted(request['sequence'] * request['global_batch'] * 1000 / (step + value)
            for value in residuals)
        result.update(throughput_interval=[_quantile(throughput, 0.05), _quantile(throughput, 0.95)],
            uncertainty='EMPIRICAL_5_TO_95_PERCENTILES_WITHOUT_COVERAGE_GUARANTEE',
            predicted_gain=None if best is None else mean(throughput) - best,
            expected_gain=None if best is None else mean(max(value - best, 0.0) for value in throughput))
        if p_valid is not None and result['expected_gain'] is not None and cost > 0:
            result['expected_improvement_per_gpu_hour'] = p_valid * result['expected_gain'] / cost
    return result
