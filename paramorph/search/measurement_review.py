from contracts import fingerprint
from predictor.profile_database import plan_signature


def structural_validation_key(parent, trial, backend_identity, capability_domains, gpu_count, constraint_implementation):
    return fingerprint(dict(protocol='native_configuration_validation_v1',
        parent_plan_signature=plan_signature(parent['plan']['backend_config']),
        candidate_plan_hash=trial['plan_hash'],
        candidate_plan_signature=plan_signature(trial['plan']['backend_config']),
        backend_identity=backend_identity, capability_domains=capability_domains, gpu_count=gpu_count,
        constraint_implementation=constraint_implementation))


def native_review_binding(catalog):
    if len({row['trial_id'] for row in catalog}) != len(catalog):
        raise ValueError('原生审核包含重复候选')
    for row in catalog:
        if row['native_configuration_validation']['status'] != 'PASS':
            raise ValueError('测量候选缺少原生配置核验')
        actions = row['allowed_actions']
        if len(set(actions)) != len(actions) or not set(actions) <= {
                'DEFER', 'PROFILE', 'LOAD_VALIDATE', 'BENCHMARK'}:
            raise ValueError('测量候选的允许操作无效')
        if ('BENCHMARK' in actions and (row['prediction_status'] == 'NEEDS_PROFILE'
                or row['memory_admission'] != 'WITHIN_ESTIMATED_CAPACITY')):
            raise ValueError('正式测量候选缺少完整预测或容量准入')
    return dict(validation_source='deterministic_current_native_evidence',
        decision_owner='ExperimentInvestigationAgent', candidate_verifications=catalog,
        scope='configuration_semantics_current_prediction_and_per_rank_admission')
