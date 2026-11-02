from collections import defaultdict
from copy import deepcopy
import math
import re

from graph.process_groups import process_groups


def checkpoint_keys(parameter, request, structure, rank):
    name = parameter['param_key']
    if parameter['layer_id'] is not None:
        match = re.fullmatch(r'decoder\.layers\.\d+\.(.+)', name)
        if match is None:
            raise ValueError(f'原生Decoder参数名称没有模块映射：{name}')
        prefix, suffix = f'backbone.layers.{parameter["layer_id"]}.', match[1]
    elif name.startswith('mtp.'):
        inner = re.fullmatch(r'mtp\.layers\.(\d+)\.mtp_model_layer\.layers\.(\d+)\.(.+)', name)
        outer = re.fullmatch(r'mtp\.layers\.(\d+)\.(.+)', name)
        width = len(structure['mtp_layers_block_type'])
        if inner:
            prefix = f'mtp.layers.{int(inner[1]) * width + int(inner[2])}.'
            suffix = inner[3]
        elif outer:
            index = int(outer[1]) * width + (width - 1 if outer[2] == 'final_layernorm.weight' else 0)
            return [f'mtp.layers.{index}.{outer[2]}']
        else:
            raise ValueError(f'原生MTP参数名称没有模块映射：{name}')
    else:
        special = {'embedding.word_embeddings.weight': 'backbone.embeddings.weight',
            'decoder.final_norm.weight': 'backbone.norm_f.weight', 'output_layer.weight': 'lm_head.weight'}
        if name not in special:
            raise ValueError(f'原生特殊模块参数名称没有Checkpoint映射：{name}')
        return [special[name]]
    aliases = {
        'mixer.conv1d_weight': 'mixer.conv1d.weight',
        'mixer.conv1d_bias': 'mixer.conv1d.bias',
        'mixer.in_proj.layer_norm_weight': 'norm.weight',
        'self_attention.linear_proj.weight': 'mixer.o_proj.weight',
        'self_attention.linear_qkv.layer_norm_weight': 'norm.weight',
        'pre_mlp_layernorm.weight': 'norm.weight', 'mlp.router.weight': 'mixer.gate.weight',
        'mlp.shared_experts.linear_fc1.weight': 'mixer.shared_experts.up_proj.weight',
        'mlp.shared_experts.linear_fc2.weight': 'mixer.shared_experts.down_proj.weight'}
    if suffix == 'self_attention.linear_qkv.weight':
        return [prefix + f'mixer.{role}_proj.weight' for role in ('q', 'k', 'v')]
    expert = re.fullmatch(r'mlp\.experts\.linear_fc([12])\.weight(\d+)', suffix)
    if expert:
        groups = process_groups(request)['logical']['ep']
        group = next(group for group in groups if rank in group)
        index = group.index(rank) * (structure['n_routed_experts'] // request['ep']) + int(expert[2])
        role = 'up_proj' if expert[1] == '1' else 'down_proj'
        return [prefix + f'mixer.experts.{index}.{role}.weight']
    return [prefix + aliases.get(suffix, suffix)]


def bind_runtime_order(metadata, documents, structure, owners):
    modules = {}
    report = []
    section_order = ('embedding', 'decoder', 'final_norm', 'mtp', 'output')
    for document in documents:
        rank, request = document['rank'], document['request']
        expected = {(row['vp_stage'], row['param_key']): row for row in owners[rank]
            if row['family'] != 'buffer'}
        used, grouped, section_positions = set(), defaultdict(list), defaultdict(list)
        for parameter in document['parameters']:
            keys = checkpoint_keys(parameter, request, structure, rank)
            rows = [expected[parameter['vp_stage'], key] for key in keys]
            shape = list(rows[0]['local_shape'])
            if len(rows) > 1:
                if any(row['local_shape'][1:] != shape[1:] for row in rows):
                    raise ValueError('原生QKV合并的Checkpoint Shape不相容')
                shape[0] = sum(row['local_shape'][0] for row in rows)
            if shape != parameter['local_shape'] or math.prod(shape) != parameter['local_elements']:
                raise ValueError(f'原生参数与Checkpoint切片Shape不同：rank={rank}, {parameter["param_key"]}')
            if any(row['requires_grad'] != parameter['requires_grad'] for row in rows):
                raise ValueError(f'原生参数训练属性与Ledger不同：{parameter["param_key"]}')
            used.update((parameter['vp_stage'], key) for key in keys)
            component = rows[0]['component']
            section_positions[parameter['vp_stage']].append(section_order.index(component))
            module = f'layer:{parameter["layer_id"]}' if component == 'decoder' else component
            name = parameter['param_key']
            if component == 'decoder':
                name = re.sub(r'^decoder\.layers\.\d+\.', '', name)
            template = dict(native_key=name, checkpoint_keys=keys,
                requires_grad=parameter['requires_grad'], shared_embedding=parameter['shared_embedding'],
                tensor_model_parallel=parameter['tensor_model_parallel'],
                partition_axis=parameter['partition_axis'], expert_parallel=parameter['expert_parallel'])
            expert = re.fullmatch(r'(.+\.linear_fc[12]\.weight)\d+', name)
            if expert:
                template['native_key'] = expert[1] + '{local_expert}'
                template['checkpoint_keys'] = [re.sub(r'\.experts\.\d+\.', '.experts.{expert}.', key) for key in keys]
                template['expert_block'] = True
            else:
                template['expert_block'] = False
            if not grouped[module] or template != grouped[module][-1]:
                grouped[module].append(template)
        if used != set(expected):
            raise ValueError(f'真实模型参数没有覆盖全部Checkpoint参数切片：rank={rank}')
        for key, templates in grouped.items():
            if key in modules and modules[key] != templates:
                raise ValueError(f'不同Rank的原生参数顺序或属性不一致：{key}')
            modules[key] = templates
        for vp in range(request['vpp']):
            positions = section_positions[vp]
            if positions != sorted(positions):
                raise ValueError('真实模型的参数模块顺序与已登记的后端顺序不同')
        report.append(dict(rank=rank, native_parameters=len(document['parameters']),
            matched_checkpoint_slices=len(used), shape_mismatches=0, trainability_mismatches=0))
    expected_modules = {f'layer:{row["layer_id"]}' if row['component'] == 'decoder'
        else row['component'] for row in metadata['tensors'] if row['family'] != 'buffer'}
    if set(modules) != expected_modules:
        raise ValueError('原生参数顺序记录没有覆盖模型全部模块')
    return dict(modules=modules, section_order=list(section_order), validation=report,
        source='actual_deployed_named_parameters_order_and_checkpoint_shape_verification')


def candidate_native_parameters(metadata, config, regions, request, rank):
    order = metadata['native_parameter_order']
    tensors = {row['param_key']: row for row in metadata['tensors']}
    groups = process_groups(request)['logical']
    ep_group = next(group for group in groups['ep'] if rank in group)
    expert_count = max((row['expert_id'] for row in tensors.values()
        if row['expert_id'] is not None), default=-1) + 1
    local_experts = expert_count // config.ep
    expert_start = ep_group.index(rank) * local_experts
    stage = rank // (config.world_size // config.pp)
    output = {}
    for vp in range(config.vpp):
        region = next(region for region in regions if region.pp_rank == stage and region.vp_rank == vp)
        modules = []
        embedding = stage == 0 and vp == 0 or metadata['mtp_depths'] and stage == config.pp - 1 and vp == config.vpp - 1
        if embedding:
            modules.append(('embedding', None))
        modules.extend((f'layer:{layer}', layer - region.layer_start)
            for layer in range(region.layer_start, region.layer_end))
        if stage == config.pp - 1 and vp == config.vpp - 1:
            modules.append(('final_norm', None))
            if metadata['mtp_depths']:
                modules.append(('mtp', None))
            if not metadata['tie_word_embeddings'] or config.pp > 1:
                modules.append(('output', None))
        parameters = []
        for module, local_layer in modules:
            for template in order['modules'][module]:
                experts = range(local_experts) if template['expert_block'] else [None]
                for local_expert in experts:
                    keys = [key.format(expert=expert_start + local_expert)
                        if local_expert is not None else key for key in template['checkpoint_keys']]
                    rows = [tensors[key] for key in keys]
                    divisor = config.etp if rows[0]['family'] == 'expert' else config.tp
                    shape = list(rows[0]['global_shape'])
                    if len(rows) > 1:
                        shape[0] = sum(row['global_shape'][0] for row in rows)
                    axis = rows[0]['partition_axis']
                    if axis is not None:
                        if shape[axis] % divisor:
                            raise ValueError('原生参数Shape无法整除当前分片组')
                        shape[axis] //= divisor
                    native_key = template['native_key']
                    if local_expert is not None:
                        native_key = native_key.format(local_expert=local_expert)
                    if local_layer is not None:
                        native_key = f'decoder.layers.{local_layer}.' + native_key
                    shared_embedding = template['shared_embedding'] and config.pp > 1
                    parameters.append(dict(template, native_key=native_key, shared_embedding=shared_embedding,
                        local_shape=shape, local_elements=math.prod(shape)))
        output[vp] = parameters
    return output


def distributed_buffer_layout(parameters, bucket_size, shards, shard_rank, high_bandwidth):
    divisor = math.lcm(shards, 128, 65536) if high_bandwidth else math.lcm(shards, 128)
    buckets, pending = [], []
    position, start = 0, 0

    def finalize(end):
        nonlocal position, start, pending
        padded_end = math.ceil(end / divisor) * divisor
        width = (padded_end - start) // shards
        shard_start = start + shard_rank * width
        shard_end = shard_start + width
        for row in pending:
            row['optimizer_owned_elements'] = max(0,
                min(row['end'], shard_end) - max(row['start'], shard_start))
        buckets.append(dict(bucket_id=len(buckets), start=start, end=padded_end,
            numel_unpadded=end - start, optimizer_shard_start=shard_start,
            optimizer_shard_end=shard_end,
            optimizer_owned_elements=sum(row['optimizer_owned_elements'] for row in pending),
            parameters=pending))
        position, start, pending = padded_end, padded_end, []

    # 参数顺序、64元素起始位置和Bucket结束位置遵循已部署DistributedOptimizer规则。
    for parameter in reversed(parameters):
        position = math.ceil(position / 64) * 64
        if parameter['shared_embedding'] and pending:
            finalize(position)
        end = position + parameter['local_elements']
        pending.append(dict(param_key=parameter['native_key'], start=position, end=end))
        if parameter['shared_embedding'] or bucket_size is not None and end - start >= bucket_size:
            finalize(end)
        else:
            position = end
    if pending:
        finalize(position)
    return buckets


def candidate_buffer_layouts(metadata, config, regions, request):
    if request.get('execution_backend', 'megatron_ddp') != 'megatron_ddp' or request.get('optimizer_sharding', 'distributed') != 'distributed':
        return None
    if request.get('num_distributed_optimizer_instances', 1) != 1:
        raise ValueError('原生参数布局尚未登记多个Optimizer实例的参数组规则')
    groups = process_groups(request)
    result = {}
    cache = {}
    for rank in range(config.world_size):
        stage = rank // (config.world_size // config.pp)
        if stage not in cache:
            cache[stage] = candidate_native_parameters(metadata, config, regions, request, rank)
        inventory = cache[stage]
        layouts = []
        for vp, parameters in inventory.items():
            for family in ('dense', 'expert'):
                selected = [row for row in parameters if row['requires_grad']
                    and row['expert_parallel'] == (family == 'expert')]
                if not selected:
                    continue
                group = next(group for group in groups['logical']['expert_dp' if family == 'expert' else 'dp_cp'] if rank in group)
                bucket_size = request.get('bucket_elements', max(40000000, 1000000 * config.dp))
                if stage > 0 or vp > 0 or not request.get('overlap_grad_reduce', True):
                    bucket_size = None
                high_bandwidth = request.get('pad_buckets_for_high_nccl_busbw', False)
                key = (stage, vp, family, len(group), group.index(rank), bucket_size, high_bandwidth)
                if key not in cache:
                    cache[key] = distributed_buffer_layout(selected, bucket_size, len(group), group.index(rank), high_bandwidth)
                layouts.append(dict(vp_stage=vp, family=family,
                    param_dtype='torch.' + request['precision'].removeprefix('torch.'),
                    grad_dtype='torch.' + request['main_grads_dtype'].removeprefix('torch.'),
                    physical_ranks=[groups['rank_mapping'][value] for value in group],
                    bucket_size=bucket_size, pad_buckets_for_high_nccl_busbw=high_bandwidth,
                    num_optimizer_shards=len(group), buckets=deepcopy(cache[key])))
        result[rank] = layouts
    return result
