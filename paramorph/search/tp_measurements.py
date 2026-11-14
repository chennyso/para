from math import isfinite

from contracts import fingerprint
from predictor.profile_database import effective_execution_config, plan_signature


def comparison_scope(trial):
    plan = trial['plan']
    request = plan['backend_config']
    execution = effective_execution_config(request)
    identity = {key: plan.get(key) for key in (
        'model_fingerprint', 'workload_fingerprint', 'hardware_fingerprint', 'backend_fingerprint')}
    if not all(identity.values()):
        return None
    controls = {key: execution.get(key) for key in (
        'model', 'checkpoint', 'data', 'sequence', 'global_batch', 'node_count', 'gpus_per_node',
        'precision', 'execution_backend', 'precision_aware_optimizer', 'optimizer', 'optimizer_config',
        'optimizer_hyperparameters', 'optimizer_schedule', 'training_data_contract', 'random_seed',
        'data_random_seed', 'loss_normalization', 'clip_grad', 'optimizer_sharding',
        'micro_batch', 'recompute', 'recompute_num_layers', 'mlp_recompute_every',
        'optimizer_offload_fraction', 'offload')}
    return dict(identity=identity, controls=controls,
        steps={key: request.get(key) for key in ('warmup', 'measured')})


def valid_throughput(trial):
    measurement = trial.get('measurement') or {}
    rate = measurement.get('tokens_per_second')
    return rate if (measurement.get('status') == 'PASS' and measurement.get('performance_eligible') is True
        and isinstance(rate, (int, float)) and not isinstance(rate, bool) and isfinite(rate) and rate > 0) else None


def compare_tp_increase(first, second):
    scope = comparison_scope(first)
    if scope is None or scope != comparison_scope(second):
        return None
    left, right = first['plan']['backend_config'], second['plan']['backend_config']
    if left['tp'] >= right['tp']:
        return None
    before, after = valid_throughput(first), valid_throughput(second)
    if before is None or after is None or after >= before:
        return None
    baseline, candidate = (effective_execution_config(row) | dict(decoder_chunks=row['decoder_chunks'])
        for row in (left, right))
    changes = {key: [baseline.get(key), candidate.get(key)] for key in sorted(baseline.keys() | candidate.keys())
        if baseline.get(key) != candidate.get(key)}
    return dict(kind='measured_tp_regression', scope=scope, scope_id=fingerprint(scope),
        baseline_run=first['measurement']['run_id'], candidate_run=second['measurement']['run_id'],
        baseline_config=baseline, candidate_config=candidate, changed_fields=changes,
        baseline_plan_signature=plan_signature(left), candidate_plan_signature=plan_signature(right),
        baseline_tokens_per_s=before, candidate_tokens_per_s=after,
        throughput_change_fraction=after / before - 1,
        interpretation='TP_ONLY' if set(changes) == {'tp'} else 'JOINT_STRATEGY',
        decision='AVOID_TP_INCREASE_IN_MATCHED_CONDITIONS')


def tp_selection_exclusion(parent, candidate, regressions):
    scope = comparison_scope(candidate)
    if scope is None or scope != comparison_scope(parent) or valid_throughput(parent) is None:
        return None
    tp = parent['plan']['backend_config']['tp']
    if candidate['plan']['backend_config']['tp'] <= tp:
        return None
    run_id = parent['measurement']['run_id']
    matching = [row['state_id'] for row in regressions if row['document']['baseline_run'] == run_id
        and row['document']['scope_id'] == fingerprint(scope)]
    if not matching:
        return None
    return dict(candidate_id=candidate['trial_id'], baseline_run=run_id,
        maximum_tp=tp, regression_evidence_ids=matching, reason='MEASURED_TP_INCREASE_REGRESSION')
