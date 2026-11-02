import math


def program_features(program, simulation):
    policy = program['global_policy']
    dense = policy['dense']
    expert = policy['expert']
    pipeline = program['pipeline_program']
    memory = pipeline['memory_policy']
    chunks = sorted(pipeline['chunks'], key=lambda chunk: (chunk['vp_stage'], chunk['pp_rank']))
    if not 1 <= len(chunks) <= 48:
        raise ValueError('program features support one to forty-eight executable virtual chunks')
    sizes = [len(chunk['region_ids']) for chunk in chunks]
    average = sum(sizes) / len(sizes)
    imbalance = max(sizes) / average
    variance = sum((size - average)**2 for size in sizes) / len(sizes)
    values = [dense['tp'], dense['cp'], dense['pp'], dense['dp'],
              1 if expert is None else expert['ep'], 1 if expert is None else expert['etp'],
              pipeline['vp_size'], imbalance, variance / average**2,
              ('replicated', 'distributed_optimizer', 'fsdp').index(policy['state_sharding']),
              ('none', 'selective', 'prefix', 'full').index(memory['recompute']),
              ('none', 'cpu_sync', 'cpu_overlap', 'cpu_prefetch').index(memory['activation_offload']),
              int(memory['optimizer_offload'] == 'cpu')]
    alternatives = pipeline['boundary_alternatives']
    values.extend((sizes[index] / sum(sizes) if index < len(sizes) else 0.0) for index in range(48))
    for kind in ('canonical', 'MOVE', 'PROJECT', 'COMPOSE'):
        values.append(sum(alternative['rewrite_family'] == kind for alternative in alternatives))
    for key in ('exposed_tp_us', 'exposed_cp_us', 'exposed_ep_us', 'exposed_pp_us',
                'exposed_boundary_us', 'peak_memory_bytes'):
        values.append(simulation[key])
    if not all(math.isfinite(value) for value in values):
        raise ValueError('program features require finite measured or simulated evidence')
    return tuple(float(value) for value in values)
