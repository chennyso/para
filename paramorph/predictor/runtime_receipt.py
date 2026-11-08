from contracts import fingerprint
from predictor.profile_database import effective_execution_config
from scheduling.native_microbatches import verify_native_schedule


class RuntimeReceiptError(ValueError):
    pass


def effective_runtime_receipt(request, source_hashes, manifests):
    expected = effective_execution_config(request) | dict(decoder_chunks=request['decoder_chunks'])
    world = request['node_count'] * request['gpus_per_node']
    if len(manifests) != world:
        raise RuntimeReceiptError('逐 Rank 回执未覆盖全部设备')
    if any(row.get('world') != world for row in manifests):
        raise RuntimeReceiptError('逐 Rank 回执的设备总数不同')
    fields = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'expert_dp', 'params_dtype',
        'sequence', 'micro_batch', 'global_batch', 'optimizer_sharding', 'optimizer_pin_cpu_params',
        'optimizer_cpu_copy_mode',
        'optimizer_state_dtypes', 'precision_aware_optimizer', 'recompute_granularity',
        'recompute_modules', 'recompute_num_layers', 'overlap_grad_reduce',
        'overlap_param_gather', 'gradient_bucket_size', 'native_pipeline_schedule')
    observed = {}
    for name in fields:
        values = [row[name] for row in manifests if name in row]
        if not values:
            continue
        if len(values) != len(manifests) or any(value != values[0] for value in values[1:]):
            raise RuntimeReceiptError(f'逐 Rank 生效配置不一致：{name}')
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
    if request.get('record_native_schedule') or schedule:
        try:
            verify_native_schedule(request, manifests)
        except ValueError as error:
            raise RuntimeReceiptError(str(error)) from error
    for name in ('precision', 'main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype',
                 'tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp'):
        if expected.get(name) is not None and updates.get(name) != expected[name]:
            raise RuntimeReceiptError(f'原生训练设置缺失或与请求不同：{name}')
    fixed = {name: request[name] for name in ('sequence', 'micro_batch', 'global_batch')}
    fixed.update(expert_dp=world // (request.get('etp', request['tp']) * request['ep'] * request['pp']),
        optimizer_sharding=request.get('optimizer_sharding', 'distributed'))
    fixed.update({name: request[name] for name in ('optimizer_pin_cpu_params', 'optimizer_cpu_copy_mode')
        if name in request})
    for name, value in fixed.items():
        if observed.get(name) != value:
            raise RuntimeReceiptError(f'原生训练设置缺失或与请求不同：{name}')
    granularity = {'none': None, 'full': 'full', 'selective': 'selective', 'selective-mlp': 'selective'}[request['recompute']]
    if 'recompute_granularity' not in observed or observed['recompute_granularity'] != granularity:
        raise RuntimeReceiptError('原生重计算粒度与请求不同')
    if granularity == 'full' and expected['recompute_num_layers'] is not None and (
            updates.get('recompute_num_layers') != expected['recompute_num_layers']):
        raise RuntimeReceiptError('原生重计算层数与请求不同')
    if granularity == 'selective':
        modules = (request.get('memory_policy') or {}).get('recompute_modules',
            ['mlp'] if request['recompute'] == 'selective-mlp' else [])
        if sorted(observed.get('recompute_modules') or []) != sorted(modules):
            raise RuntimeReceiptError('原生选择性重计算模块与请求不同')
    for name in ('overlap_grad_reduce', 'overlap_param_gather', 'precision_aware_optimizer'):
        if name in request and updates.get(name) != request[name]:
            raise RuntimeReceiptError(f'原生优化器设置缺失或与请求不同：{name}')
    effective = expected | updates
    return dict(effective_config=effective, effective_plan_hash=fingerprint(
        dict(config=effective, backend_source_hashes=source_hashes)),
        runtime_observed_fields=sorted(observed), runtime_rank_count=len(manifests),
        native_pipeline_schedule=schedule or None, schedule_identity=fingerprint(schedule) if schedule else None)


def validate_training_result(record, request, source_hashes, manifests):
    try:
        receipt = effective_runtime_receipt(request, source_hashes, manifests)
    except RuntimeReceiptError as error:
        record.update(status='VALIDATION_FAILED', performance_eligible=False, runtime_validation_error=str(error))
    else:
        record.update(receipt)
