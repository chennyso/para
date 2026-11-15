from copy import deepcopy

from predictor.profile_database import resolved_cuda_connections


def batch_nccl_request(request, profile_requests, trials):
    if request.get('nccl_cases') or request.get('nccl_concurrent_group_batches'):
        return request
    descriptions = {}
    for trial in trials:
        for item in (trial.get('prediction') or {}).get('evidence', {}).get('profiling_requests', []):
            if (item.get('kind') == 'nccl' and item.get('cost_key')
                    and item.get('profile_required_occurrence_count', 1) > 0):
                descriptions[item['cost_key']] = item
    dtype = request.get('nccl_dtype', 'bfloat16')
    groups, cases, members, cost_keys = set(), set(), [], set()
    for profile in profile_requests:
        execution = profile['execution_request']
        if (profile['measurement_type'] != 'nccl'
                or profile['state'] not in ('PENDING', 'APPROVED', 'RUNNING', 'DEFERRED')
                or execution.get('nccl_dtype', 'bfloat16') != dtype
                or execution.get('nccl_concurrent_group_batches')
                or resolved_cuda_connections(execution) != resolved_cuda_connections(request)
                or any(execution.get(key) != request.get(key) for key in
                       ('node_count', 'gpus_per_node', 'gpus_per_worker', 'warmup', 'measured'))):
            continue
        included = False
        for key in profile['cost_keys']:
            item = descriptions.get(key)
            if item is None:
                continue
            identity = item['identity']
            if identity.get('concurrent_operations'):
                continue
            operation = identity['collective']
            ranks = tuple(identity.get('physical_ranks', identity.get('ranks', [])))
            if operation not in ('all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all', 'p2p_send_recv'):
                raise ValueError('NCCL成本查询包含未登记的通信操作')
            if len(ranks) < 2:
                continue
            group = tuple(sorted(ranks))
            groups.add(group)
            direction = ranks if operation == 'p2p_send_recv' else group
            for payload in execution['nccl_payloads_bytes']:
                cases.add((operation, direction, payload))
            cost_keys.add(key)
            included = True
        if included:
            members.append(profile['request_id'])
    if len(members) < 2:
        return request
    batched = deepcopy(request)
    batched.update(nccl_rank_groups=[list(group) for group in sorted(groups)],
        nccl_group_sizes=sorted({len(group) for group in groups}),
        nccl_payloads_bytes=sorted({payload for _, _, payload in cases}),
        nccl_cases=[dict(collective=operation, ranks=list(ranks), payload_bytes=payload)
                    for operation, ranks, payload in sorted(cases)],
        nccl_batch_profile_request_ids=sorted(members),
        nccl_batch_cost_keys=sorted(cost_keys),
        nccl_batch_scope='current_campaign_pending_requests_same_launch_conditions_dtype_and_measurement_protocol')
    return batched
