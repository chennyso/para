from copy import deepcopy


EXPERT_ORIGINS = frozenset(('expert', 'manual', 'expert_current', 'expert_reference'))


def filter_target_expert_history(history, workload_key):
    required = ('model', 'sequence_length')
    if any(key not in workload_key for key in required):
        raise ValueError('target workload requires model and sequence length')
    safe = []
    for record in history:
        if any(key not in record for key in (*required, 'origin')):
            continue
        target = all(record[key] == workload_key[key] for key in required)
        if target and record['origin'] in EXPERT_ORIGINS:
            continue
        if record['origin'] not in EXPERT_ORIGINS | {'agent_trial', 'profile', 'failure', 'seed'}:
            continue
        safe.append(deepcopy(record))
    return tuple(safe)
