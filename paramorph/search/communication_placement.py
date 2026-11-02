from copy import deepcopy
from itertools import permutations
import math

from graph.process_groups import process_groups
from predictor.profile_database import plan_signature


AXES = ('tp', 'cp', 'dp', 'pp')


def axis_placement(request, order):
    if tuple(sorted(order)) != tuple(sorted(AXES)):
        raise ValueError('物理Rank构造需要完整且不重复的稠密并行维度')
    sizes = {axis: request[axis] for axis in AXES}
    world = request['node_count'] * request['gpus_per_node']
    if math.prod(sizes.values()) != world:
        raise ValueError('并行形状没有覆盖实际GPU')
    mapping = []
    for logical in range(world):
        remaining = logical
        coordinates = {}
        for axis in AXES:
            coordinates[axis] = remaining % sizes[axis]
            remaining //= sizes[axis]
        physical, stride = 0, 1
        for axis in order:
            physical += coordinates[axis] * stride
            stride *= sizes[axis]
        mapping.append(physical)
    if sorted(mapping) != list(range(world)):
        raise ValueError('维度次序生成了无效的物理Rank映射')
    return mapping


def placement_families(request):
    candidates = {}
    for order in permutations(AXES):
        candidate = deepcopy(request)
        candidate['rank_mapping'] = axis_placement(request, order)
        key = plan_signature(candidate)
        if key not in candidates:
            candidates[key] = dict(request=candidate, axis_order=list(order))
    return list(candidates.values())


def shared_block_rank_mappings(request, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError('共同分组映射预算必须为正整数')
    world = request['node_count'] * request['gpus_per_node']
    dense = request['tp'] * request['cp']
    expert = request.get('etp', request['tp']) * request['ep']
    block = math.lcm(dense, expert)
    if world % block:
        raise ValueError('稠密与专家分解的共同分组没有覆盖完整设备')
    blocks = world // block
    rows = []
    for order in permutations(range(blocks)):
        if len(rows) == limit:
            return rows, False
        mapping = [target * block + offset for target in order for offset in range(block)]
        rows.append(dict(rank_mapping=mapping, shared_block_size=block,
            block_order=list(order)))
    return rows, True


def shared_block_placements(request, limit):
    mappings, complete = shared_block_rank_mappings(request, limit)
    rows = []
    for row in mappings:
        candidate = deepcopy(request)
        candidate['rank_mapping'] = row['rank_mapping']
        rows.append(dict(request=candidate, shared_block_size=row['shared_block_size'],
            block_order=row['block_order']))
    return rows, complete


def placement_evidence(request, devices):
    nodes = {row['physical_rank']: row['node_index'] for row in devices}
    groups = process_groups(request)
    world = request['node_count'] * request['gpus_per_node']
    if set(nodes) != set(range(world)):
        raise ValueError('通信放置核查需要全部物理GPU的实际节点身份')
    result = {}
    for name, physical_groups in groups['physical'].items():
        rows = [dict(ranks=row, nodes=sorted({nodes[rank] for rank in row}))
                for row in physical_groups]
        result[name] = dict(groups=rows,
            cross_node_groups=sum(len(row['nodes']) > 1 for row in rows),
            total_groups=len(rows))
    return dict(rank_mapping=groups['rank_mapping'], group_placement=result,
        performance_status='REQUIRES_MATCHED_PHYSICAL_COMMUNICATION_COSTS',
        memory_scope='logical_ownership_preserved_physical_owners_remapped',
        native_lowering='hybrid_training_entry.py:RANK=rank_mapping.index(physical_rank)')


def compact_placement_evidence(evidence):
    result = {key: deepcopy(value) for key, value in evidence.items() if key != 'group_placement'}
    unique, references, summaries = {}, {}, {}
    for name, placement in evidence['group_placement'].items():
        identity = tuple((tuple(row['ranks']), tuple(row['nodes'])) for row in placement['groups'])
        if identity not in references:
            identifier = str(len(unique))
            references[identity] = identifier
            unique[identifier] = [[list(ranks), list(nodes)] for ranks, nodes in identity]
        summaries[name] = [references[identity], placement['cross_node_groups'], placement['total_groups']]
    result.update(group_columns=['group_set_id', 'cross_node_groups', 'total_groups'],
        group_placement=summaries, group_set_columns=['ranks', 'nodes'], group_sets=unique)
    return result
