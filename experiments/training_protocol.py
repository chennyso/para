def optimizer_schedule(request):
    if 'optimizer_schedule' in request:
        return dict(request['optimizer_schedule'])
    if 'optimizer_hyperparameters' not in request:
        return None
    return dict(lr_decay_style='cosine', lr_decay_iters=request['warmup'] + request['measured'],
        lr_warmup_iters=0)


def training_data_contract(request):
    if 'training_data_contract' in request:
        return dict(request['training_data_contract'])
    return dict(training_index_samples=request['global_batch'] * (request['warmup'] + request['measured']),
        workers=0, persistent_workers=False)


def bind_profile_protocol(execution, reference):
    schedule = optimizer_schedule(reference)
    data = training_data_contract(reference)
    if schedule is None or (execution['warmup'] + execution['measured']) * execution['global_batch'] > data['training_index_samples']:
        execution.update(warmup=reference['warmup'], measured=reference['measured'])
    if schedule is not None:
        execution['optimizer_schedule'] = schedule
    execution['training_data_contract'] = data
    for field in ('random_seed', 'data_random_seed'):
        if field in reference:
            execution[field] = reference[field]
    return execution
