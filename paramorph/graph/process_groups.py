from itertools import product


def process_groups(request):
    world = request['node_count'] * request['gpus_per_node']
    tp, cp, pp, dp = (request[name] for name in ('tp', 'cp', 'pp', 'dp'))
    ep, etp = request['ep'], request.get('etp', tp)
    if world != tp * cp * pp * dp or world % (etp * ep * pp):
        raise ValueError('Dense或者Expert并行组不能完整覆盖实际设备')
    expert_dp = world // (etp * ep * pp)
    if request.get('expert_dp', expert_dp) != expert_dp:
        raise ValueError('Expert-DP与真实Parallel Folding关系不同')
    mapping = request.get('rank_mapping', list(range(world)))
    if sorted(mapping) != list(range(world)):
        raise ValueError('物理Rank映射必须完整覆盖实际设备一次')

    def groups(dimensions, varying):
        names = list(dimensions)
        strides = {}
        stride = 1
        for name, size in dimensions.items():
            strides[name] = stride
            stride *= size
        fixed = [name for name in names if name not in varying]
        rows = []
        for constants in product(*(range(dimensions[name]) for name in fixed)):
            ranks = []
            for values in product(*(range(dimensions[name]) for name in varying)):
                coordinates = dict(zip(fixed, constants)) | dict(zip(varying, values))
                ranks.append(sum(coordinates[name] * strides[name] for name in names))
            rows.append(sorted(ranks))
        return rows

    dense = dict(tp=tp, cp=cp, dp=dp, pp=pp)
    expert = dict(tp=etp, ep=ep, dp=expert_dp, pp=pp)
    logical = dict(tp=groups(dense, ['tp']), cp=groups(dense, ['cp']),
        dp=groups(dense, ['dp']), dp_cp=groups(dense, ['cp', 'dp']),
        pp=groups(dense, ['pp']), etp=groups(expert, ['tp']),
        ep=groups(expert, ['ep']), expert_dp=groups(expert, ['dp']))
    logical['embedding'] = [[ranks[0]] if len(ranks) == 1 else [ranks[0], ranks[-1]]
                            for ranks in logical['pp']]
    position_policy = request.get('position_embedding_group_policy') or 'first_only'
    if position_policy == 'first_only':
        logical['position_embedding'] = [[ranks[0]] for ranks in logical['pp']]
    elif position_policy == 'first_last':
        logical['position_embedding'] = [[ranks[0]] if len(ranks) == 1 else [ranks[0], ranks[-1]]
                                         for ranks in logical['pp']]
    else:
        raise ValueError('Position Embedding进程组策略未注册')
    return dict(logical=logical,
        physical={name: [[mapping[rank] for rank in group] for group in rows]
                  for name, rows in logical.items()},
        expert_dp_size=expert_dp, rank_mapping=mapping,
        rank_order='tp-cp-ep-dp-pp',
        source='megatron.core.parallel_state.RankGenerator_pp_last')
