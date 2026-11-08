from predictor.profile_database import resolved_clip_grad


def require_fixed_training_semantics(parent_request, candidate_request):
    fields = ('model', 'checkpoint', 'data', 'global_batch', 'sequence', 'precision', 'optimizer',
        'node_count', 'gpus_per_node', 'random_seed', 'loss_normalization', 'optimizer_config',
        'optimizer_hyperparameters', 'optimizer_schedule', 'native_uniform_pipeline_layout', 'training_data_contract')
    changed = [name for name in fields if parent_request.get(name) != candidate_request.get(name)]
    if bool(parent_request.get('precision_aware_optimizer', False)) != bool(
            candidate_request.get('precision_aware_optimizer', False)):
        allowed = parent_request.get('allowed_precision_aware_optimizer',
            [bool(parent_request.get('precision_aware_optimizer', False))])
        if bool(candidate_request.get('precision_aware_optimizer', False)) not in allowed:
            changed.append('precision_aware_optimizer')
        for name in ('main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
            if parent_request.get(name) != candidate_request.get(name):
                changed.append(name)
    if resolved_clip_grad(parent_request) != resolved_clip_grad(candidate_request):
        changed.append('clip_grad')
    if changed:
        raise ValueError(dict(status='CONSTRAINT_CONFLICT',
            conflicts=['固定训练语义发生变化：' + ', '.join(changed)]))
