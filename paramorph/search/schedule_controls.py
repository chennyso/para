from dataclasses import asdict

from cost.runtime_calibration import configuration
from predictor.profile_database import normalized_execution
from scheduling.native_microbatches import native_schedule_table, valid_microbatch_group


def measured_schedule_parent(trial):
    measured = trial.get('measurement') or {}
    request = trial['plan']['backend_config']
    return (measured.get('status') == 'PASS' and measured.get('performance_eligible') is True
        and request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp' and request['vpp'] > 1)


def schedule_change(parent, candidate):
    def effective(request):
        result = normalized_execution(request)
        policy = asdict(configuration(request).memory_policy)
        for field in ('activation_offload_modules', 'recompute_modules'):
            policy[field] = tuple(policy[field])
        result['memory_policy'] = policy
        for field in ('backend_fingerprint', 'hardware_fingerprint'):
            result.pop(field, None)
        result['decoder_chunks'] = request['decoder_chunks']
        return result

    left, right = effective(parent), effective(candidate)
    changed = {field for field in left.keys() | right.keys() if left.get(field) != right.get(field)}
    if changed in ({'group'}, {'overlap_p2p'}):
        name = next(iter(changed))
        return name, right[name]
    return None


def schedule_control_options(parent, history, groups):
    if not measured_schedule_parent(parent):
        return []
    request = parent['plan']['backend_config']
    total = request['global_batch'] // (request['dp'] * request['micro_batch'])
    pp, vpp, current = request['pp'], request['vpp'], request['group']
    previous = {schedule_change(request, row['plan']['backend_config']) for row in history
        if (row.get('measurement') or {}).get('status') in ('PASS', 'CUDA_OOM', 'FAILED',
            'HOST_MEMORY_BUDGET_EXCEEDED')}
    original = native_schedule_table(total, pp, vpp, current)
    options = []
    overlap = not request.get('overlap_p2p', False)
    if ('overlap_p2p', overlap) not in previous:
        options.append(dict(field='overlap_p2p', from_value=not overlap, to_value=overlap,
            effect='keeps_compute_order_changes_p2p_overlap_permission'))
    legal = [value for value in groups if value != current and total % value == 0
        and valid_microbatch_group(total, pp, vpp, value) and ('group', value) not in previous]
    for value in sorted(legal, key=lambda item: (abs(item - current), item))[:2]:
        revised = native_schedule_table(total, pp, vpp, value)
        if revised != original:
            options.append(dict(field='group', from_value=current, to_value=value,
                effect='changes_native_virtual_stage_order',
                changed_positions=sum(first != second for first, second in zip(original, revised, strict=True)),
                schedule_entries=len(original)))
    return options
