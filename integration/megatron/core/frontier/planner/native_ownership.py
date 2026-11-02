import hashlib
import json
from collections import defaultdict
from math import prod

from megatron.core.frontier.planner.obligations import NativeTrainingStateConflict


def parameter_identity(parameter):
    fragments = tuple(sorted((fragment['key'], tuple(fragment['global_shape']),
        tuple(fragment['global_offset']), tuple(fragment['local_shape']), fragment['prepend_axis_num'])
        for fragment in parameter['checkpoint_fragments']))
    return parameter['parameter_key'], tuple(parameter['reduction_group_ranks']), fragments


def check_native_training_ownership(records, world_size):
    if world_size < 1:
        raise ValueError('Native ownership verification requires a positive actual allocation')
    conflicts, declarations, owners = [], defaultdict(dict), defaultdict(list)

    def conflict(category, parameter, ranks, expected, actual, detail):
        conflicts.append(NativeTrainingStateConflict(category, parameter, tuple(ranks),
            tuple(expected), tuple(actual), detail))

    observed = tuple(record['rank'] for record in records)
    if tuple(sorted(observed)) != tuple(range(world_size)):
        conflict('Coverage', '', observed, tuple(range(world_size)), observed,
            'Native records must cover every actual rank exactly once')
    for record in records:
        rank = record['rank']
        if record['world_size'] != world_size:
            conflict('ProcessGroup', '', (rank,), (world_size,), (record['world_size'],),
                'Native records disagree on the actual process allocation')
        for parameter in record['parameters']:
            key = parameter_identity(parameter)
            if rank in declarations[key]:
                conflict('Ownership', key[0], key[1], (1,), (2,),
                    'A native parameter is registered multiple times on the same rank')
            declarations[key][rank] = parameter
            if (rank not in key[1] or len(set(key[1])) != len(key[1])
                    or any(member not in range(world_size) for member in key[1])
                    or parameter['reduction_group_backend'] != 'nccl'):
                conflict('ProcessGroup', key[0], key[1], ('nccl', rank),
                    (parameter['reduction_group_backend'], *key[1]),
                    'Native parameter ownership requires its actual NCCL reduction group')
            local_numel = prod(parameter['parameter_shape'])
            fragment_numel = sum(prod(fragment['local_shape']) for fragment in parameter['checkpoint_fragments'])
            if not parameter['checkpoint_fragments'] or fragment_numel != local_numel:
                conflict('Layout', key[0], (rank,), (local_numel,), (fragment_numel,),
                    'Native checkpoint fragments do not cover their actual parameter tensor')
        for owner in record['owners']:
            key = parameter_identity(owner)
            if rank not in declarations[key]:
                conflict('Ownership', key[0], (rank,), ('registered_parameter',), (),
                    'Native optimizer state belongs to an undeclared actual parameter')
            begin, end = owner['optimizer_flattened_range']
            if not 0 <= begin < end <= prod(owner['parameter_shape']):
                conflict('Ownership', key[0], (rank,), (0, prod(owner['parameter_shape'])), (begin, end),
                    'Native optimizer interval exceeds its actual parameter storage')
            if owner['gradient_shape'] != owner['parameter_shape']:
                conflict('Layout', key[0], (rank,), owner['parameter_shape'], owner['gradient_shape'],
                    'Native gradient storage differs from its actual parameter shape')
            if set(owner['states']) != {'param', 'exp_avg', 'exp_avg_sq'}:
                conflict('Coverage', key[0], (rank,), ('param', 'exp_avg', 'exp_avg_sq'),
                    tuple(sorted(owner['states'])), 'Native Adam state inventory is incomplete')
            if owner['reduction_group_backend'] != 'nccl':
                conflict('ProcessGroup', key[0], (rank,), ('nccl',), (owner['reduction_group_backend'],),
                    'Native optimizer state uses an unsupported reduction backend')
            for name, state in owner['states'].items():
                if (state['numel'] != end - begin or prod(state['shape']) != end - begin
                        or state['dtype'] != 'torch.float32'):
                    conflict('Layout', key[0] + ':' + name, (rank,), (end - begin, 'torch.float32'),
                        (state['numel'], state['dtype']), 'Native Adam tensor differs from its declared ownership interval')
            owners[key].append((begin, end, rank))
    for key in declarations.keys() | owners.keys():
        rows = declarations[key]
        if set(rows) != set(key[1]):
            conflict('ProcessGroup', key[0], key[1], key[1], tuple(sorted(rows)),
                'Native parameter replicas omit actual reduction group members')
        if not rows:
            continue
        shape = next(iter(rows.values()))['parameter_shape']
        for rank, row in rows.items():
            if row['parameter_shape'] != shape:
                conflict('Layout', key[0], (rank,), shape, row['parameter_shape'],
                    'Native parameter replicas disagree on actual tensor shape')
        position, ranks = 0, set()
        for begin, end, rank in sorted(owners[key]):
            if begin != position or rank in ranks:
                conflict('Ownership', key[0], (rank,), (position,), (begin,),
                    'Native optimizer intervals contain a gap, overlap or duplicate owner')
            position = end
            ranks.add(rank)
        if position != prod(shape):
            conflict('Ownership', key[0], key[1], (prod(shape),), (position,),
                'Native optimizer intervals do not cover the complete parameter')
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return digest, tuple(conflicts)
