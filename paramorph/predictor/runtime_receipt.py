from contracts import fingerprint
from predictor.profile_database import effective_execution_config


def effective_runtime_receipt(request, source_hashes, manifests):
    expected = effective_execution_config(request) | dict(decoder_chunks=request['decoder_chunks'])
    fields = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'expert_dp', 'params_dtype',
        'optimizer_state_dtypes', 'precision_aware_optimizer', 'recompute_granularity',
        'recompute_modules', 'recompute_num_layers', 'overlap_grad_reduce',
        'overlap_param_gather', 'gradient_bucket_size', 'native_pipeline_schedule')
    observed = {}
    for name in fields:
        values = [row[name] for row in manifests if name in row]
        if not values:
            continue
        if len(values) != len(manifests) or any(value != values[0] for value in values[1:]):
            raise ValueError(f'逐 Rank 生效配置不一致：{name}')
        observed[name] = values[0]
    schedule = observed.get('native_pipeline_schedule') or {}
    updates = {name: observed[name] for name in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp',
        'vpp', 'expert_dp', 'precision_aware_optimizer', 'overlap_grad_reduce',
        'overlap_param_gather') if name in observed}
    updates.update({name: schedule[name] for name in ('group', 'overlap_p2p', 'batch_p2p_comm',
        'overlap_p2p_comm_warmup_flush') if name in schedule})
    if 'params_dtype' in observed:
        updates['precision'] = observed['params_dtype'].removeprefix('torch.')
    for name, value in observed.get('optimizer_state_dtypes', {}).items():
        if name in ('main_grads', 'main_params', 'exp_avg', 'exp_avg_sq'):
            updates[name + '_dtype'] = value.removeprefix('torch.')
    if observed.get('recompute_granularity') == 'full':
        updates['recompute_num_layers'] = observed.get('recompute_num_layers')
    effective = expected | updates
    return dict(effective_config=effective, effective_plan_hash=fingerprint(
        dict(config=effective, backend_source_hashes=source_hashes)),
        runtime_observed_fields=sorted(observed), runtime_rank_count=len(manifests),
        schedule_identity=fingerprint(schedule) if schedule else None)
