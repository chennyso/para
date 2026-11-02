from collections import defaultdict
from copy import deepcopy
import math
import re

from contracts import fingerprint
from graph.process_groups import process_groups


NEMOTRON_PARTITION_RULES = 'megatron_bridge_nemotron_h_parameter_mapping_v1'


def bind_checkpoint_parameters(snapshot, structure):
    if snapshot.get('version') != 1 or snapshot.get('source') != 'safetensors.safe_open.get_slice':
        raise ValueError('参数元数据需要实际Safetensors Shape记录')
    if fingerprint(snapshot['config']) != fingerprint(structure):
        raise ValueError('参数元数据与当前模型配置不同')
    if structure.get('model_type') != 'nemotron_h':
        raise ValueError('当前模型尚未登记Checkpoint参数分片规则')
    rows = []
    layers = set()
    mtp_indices = set()
    for name, value in snapshot['tensors'].items():
        shape = value['shape']
        if not shape or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError(f'Tensor Shape非法：{name}')
        layer_id, expert_id = None, None
        match = re.fullmatch(r'backbone\.layers\.(\d+)\.(.+)', name)
        mtp = re.fullmatch(r'mtp\.layers\.(\d+)\.(.+)', name)
        if match:
            layer_id, suffix = int(match[1]), match[2]
            layers.add(layer_id)
            component, placement = 'decoder', 'region'
        elif mtp:
            mtp_indices.add(int(mtp[1]))
            suffix = mtp[2]
            component, placement = 'mtp', 'last'
        elif name == 'backbone.embeddings.weight':
            suffix, component, placement = 'embedding.weight', 'embedding', 'first_and_mtp'
        elif name == 'lm_head.weight':
            suffix, component, placement = 'output.weight', 'output', 'last'
        elif name == 'backbone.norm_f.weight':
            suffix, component, placement = 'final_layernorm.weight', 'final_norm', 'last'
        else:
            raise ValueError(f'Checkpoint参数没有实际后端映射：{name}')
        family, partition, axis = 'dense', 'replicated', None
        expert = re.fullmatch(r'mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight', suffix)
        if expert:
            family, partition = 'expert', 'etp_ep'
            expert_id = int(expert[1])
            axis = 0 if expert[2] == 'up_proj' else 1
        elif suffix == 'mixer.gate.e_score_correction_bias':
            family, partition = 'buffer', 'replicated'
        elif suffix in ('norm.weight', 'enorm.weight', 'hnorm.weight',
                         'final_layernorm.weight', 'mixer.gate.weight'):
            pass
        elif suffix in ('mixer.out_proj.weight', 'mixer.o_proj.weight',
                         'mixer.down_proj.weight', 'mixer.shared_experts.down_proj.weight'):
            partition, axis = 'tp', 1
        elif suffix in ('embedding.weight', 'output.weight', 'eh_proj.weight',
                         'mixer.in_proj.weight', 'mixer.q_proj.weight',
                         'mixer.k_proj.weight', 'mixer.v_proj.weight',
                         'mixer.up_proj.weight', 'mixer.shared_experts.up_proj.weight',
                         'mixer.conv1d.weight', 'mixer.conv1d.bias',
                         'mixer.A_log', 'mixer.D', 'mixer.dt_bias', 'mixer.norm.weight'):
            partition, axis = 'tp', 0
        else:
            raise ValueError(f'参数缺少明确的分片语义：{name}')
        partition_segments = None
        if suffix in ('mixer.in_proj.weight', 'mixer.conv1d.weight', 'mixer.conv1d.bias'):
            inner = structure['mamba_num_heads'] * structure['mamba_head_dim']
            state = structure['n_groups'] * structure['ssm_state_size']
            partition_segments = ([inner, inner, state, state, structure['mamba_num_heads']]
                if suffix == 'mixer.in_proj.weight' else [inner, state, state])
            if sum(partition_segments) != shape[0]:
                raise ValueError(f'Mamba打包参数Shape与模型定义不同：{name}')
        module_type = (structure['layers_block_type'][layer_id] if layer_id is not None
            else 'mtp' if component == 'mtp' else component)
        inner_module_type = (mtp_pattern[int(mtp[1]) % len(mtp_pattern)]
            if mtp and (mtp_pattern := structure.get('mtp_layers_block_type')) else module_type)
        rows.append(dict(name=name, param_key=name, shape=shape, global_shape=shape,
            dtype=value['dtype'], checkpoint_dtype=value['dtype'], module_type=module_type,
            inner_module_type=inner_module_type, parameter_role=suffix,
            layer_id=layer_id, expert_id=expert_id, component=component,
            placement=placement, pipeline_owner_rule=placement,
            family=family, partition=partition, partition_axis=axis,
            partition_segments=partition_segments,
            tp_rule='shard' if partition == 'tp' else 'replicated',
            ep_rule='expert_id' if expert_id is not None else 'replicated',
            etp_rule='shard' if partition == 'etp_ep' else 'replicated',
            shared_weight_id=('word_embedding' if component == 'embedding'
                or component == 'output' and structure.get('tie_word_embeddings', False)
                else name), optimizer_group=family,
            requires_grad=family != 'buffer',
            source=dict(checkpoint_shard=value['shard'], partition_rules=NEMOTRON_PARTITION_RULES,
                trainability='native_NemotronH_default_unfrozen_Parameters_and_registered_buffers'),
            recovery_status='BACKEND_RULE_BOUND'))
    if layers != set(range(structure['num_hidden_layers'])):
        raise ValueError('参数元数据没有覆盖全部Decoder模块')
    mtp_depths = structure.get('num_nextn_predict_layers', 0) or 0
    mtp_pattern = structure.get('mtp_layers_block_type')
    if mtp_depths and not mtp_pattern:
        raise ValueError('MTP参数所有权需要实际内部模块列表')
    repeated = structure.get('mtp_use_repeated_layer', True)
    stored_depths = 1 if repeated and mtp_depths else mtp_depths
    if mtp_indices != set(range(stored_depths * len(mtp_pattern or []))):
        raise ValueError('MTP Checkpoint参数与实际深度和重复层配置不同')
    experts = structure.get('n_routed_experts', 0)
    expert_modules = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row['family'] == 'expert':
            key = (row['component'], row['layer_id'])
            expert_modules[key][row['expert_id']] += math.prod(row['shape'])
    for records in expert_modules.values():
        if set(records) != set(range(experts)) or len(set(records.values())) != 1:
            raise ValueError('Expert参数需要完整且等尺寸的真实专家记录')
    required = {'embedding', 'output', 'final_norm'}
    if not required <= {row['component'] for row in rows}:
        raise ValueError('参数元数据缺少Embedding、Output或Final Norm')
    ledger = dict(version=1, kind='ModuleParameterLedger', source=NEMOTRON_PARTITION_RULES,
        model_fingerprint=fingerprint(structure), checkpoint_source=deepcopy({
            key: snapshot[key] for key in ('source', 'checkpoint_directory',
                'config_sha256', 'index_sha256')}),
        tensors=rows, mtp_depths=mtp_depths, mtp_repeated_layer=repeated,
        tie_word_embeddings=structure.get('tie_word_embeddings', False))
    if snapshot.get('native_parameter_order') is not None:
        ledger['native_parameter_order'] = deepcopy(snapshot['native_parameter_order'])
    return ledger


def candidate_rank_ownership(metadata, config, regions, request):
    groups = process_groups(request)
    owners = {rank: [] for rank in range(config.world_size)}
    layer_stage = {}
    layer_vp = {}
    for region in regions:
        for layer in range(region.layer_start, region.layer_end):
            if layer in layer_stage:
                raise ValueError('模型层在Region分区中重复出现')
            layer_stage[layer] = region.pp_rank
            layer_vp[layer] = region.vp_rank
    lanes = config.world_size // config.pp
    experts = max((row['expert_id'] for row in metadata['tensors']
        if row['expert_id'] is not None), default=-1) + 1
    if experts and experts % config.ep:
        raise ValueError('Expert数量无法整除当前EP')
    for row in metadata['tensors']:
        placement = row['placement']
        copies = []
        if placement == 'region':
            copies.append((layer_stage[row['layer_id']], layer_vp[row['layer_id']], 'decoder'))
        elif placement == 'first_and_mtp':
            copies.append((0, 0, 'input_embedding'))
            if metadata['mtp_depths'] and (config.pp > 1 or config.vpp > 1):
                copies.append((config.pp - 1, config.vpp - 1, 'mtp_embedding'))
        elif placement == 'last':
            if row['component'] == 'output' and metadata['tie_word_embeddings'] and config.pp == 1:
                continue
            copies.append((config.pp - 1, config.vpp - 1, row['component']))
        else:
            raise ValueError('参数元数据的Pipeline归属非法')
        axis = row['partition_axis']
        divisor = config.etp if row['partition'] == 'etp_ep' else config.tp if row['partition'] == 'tp' else 1
        if axis is not None and row['shape'][axis] % divisor:
            raise ValueError(f'参数分片维度无法整除：{row["name"]}, divisor={divisor}')
        for stage, vp, copy_id in copies:
            for rank in range(stage * lanes, (stage + 1) * lanes):
                if row['expert_id'] is not None:
                    ep_group = next(group for group in groups['logical']['ep'] if rank in group)
                    ep_rank = ep_group.index(rank)
                    if row['expert_id'] // (experts // config.ep) != ep_rank:
                        continue
                local_shape = list(row['shape'])
                offset = [0] * len(local_shape)
                slices = []
                if axis is not None:
                    group_kind = 'etp' if row['partition'] == 'etp_ep' else 'tp'
                    group = next(group for group in groups['logical'][group_kind] if rank in group)
                    local_shape[axis] //= divisor
                    offset[axis] = group.index(rank) * local_shape[axis]
                    segments = row['partition_segments'] or [row['shape'][axis]]
                    base, local_base = 0, 0
                    for length in segments:
                        if length % divisor:
                            raise ValueError(f'打包参数分段无法整除：{row["name"]}')
                        width = length // divisor
                        start = base + group.index(rank) * width
                        slices.append(dict(axis=axis, global_start=start, global_end=start + width,
                            local_start=local_base, local_end=local_base + width))
                        base += length
                        local_base += width
                owners[rank].append(dict(row, logical_rank=rank,
                    physical_rank=groups['rank_mapping'][rank], pipeline_stage=stage,
                    vp_stage=vp,
                    local_shape=local_shape, global_offset=offset if len(slices) <= 1 else None,
                    global_slices=slices,
                    local_elements=math.prod(local_shape), physical_copy_id=copy_id,
                    train_param_dtype=request['precision'], grad_dtype=request['main_grads_dtype']))
    return owners


def checkpoint_parameter_counts(metadata, config, regions, request):
    dense = {rank: 0 for rank in range(config.world_size)}
    expert = dict(dense)
    buffers = dict(dense)
    by_component = {rank: defaultdict(int) for rank in dense}
    layer_stage = {layer: region.pp_rank for region in regions
        for layer in range(region.layer_start, region.layer_end)}
    if len(layer_stage) != sum(region.layer_end - region.layer_start for region in regions):
        raise ValueError('模型层在Region分区中重复出现')
    module_counts = defaultdict(int)
    widths = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8}
    for row in metadata['tensors']:
        axis = row['partition_axis']
        divisor = config.etp if row['partition'] == 'etp_ep' else config.tp if row['partition'] == 'tp' else 1
        if axis is not None and any(length % divisor for length in
                (row['partition_segments'] or [row['shape'][axis]])):
            raise ValueError(f'参数分片维度无法整除：{row["name"]}')
        elements = math.prod(row['shape']) // divisor
        placement = row['placement']
        stages = []
        if placement == 'region':
            stages.append(layer_stage[row['layer_id']])
        elif placement == 'first_and_mtp':
            stages.append(0)
            if metadata['mtp_depths'] and (config.pp > 1 or config.vpp > 1):
                stages.append(config.pp - 1)
        elif placement == 'last':
            if row['component'] == 'output' and metadata['tie_word_embeddings'] and config.pp == 1:
                continue
            stages.append(config.pp - 1)
        else:
            raise ValueError('参数元数据的Pipeline归属非法')
        for stage in stages:
            module_counts[stage, row['component'], row['family'], row['dtype']] += elements
    lanes = config.world_size // config.pp
    for (stage, component, family, dtype), elements in module_counts.items():
        if family == 'expert':
            if elements % config.ep:
                raise ValueError('完整Expert参数组无法整除当前EP')
            elements //= config.ep
        for rank in range(stage * lanes, (stage + 1) * lanes):
            if family == 'buffer':
                buffers[rank] += elements * widths[dtype]
            else:
                target = expert if family == 'expert' else dense
                target[rank] += elements
                by_component[rank][component] += elements
    result = dict(dense=dense, expert=expert, model_buffer_bytes=buffers,
        parameter_components={rank: dict(values) for rank, values in by_component.items()},
        source=metadata['source'])
    if metadata.get('native_parameter_order') is not None:
        from predictor.native_parameter_layout import candidate_buffer_layouts
        layouts = candidate_buffer_layouts(metadata, config, regions, request)
        if layouts is not None:
            result['native_ddp_buffer_counts'] = {}
            for rank, values in layouts.items():
                families = {}
                for family in ('dense', 'expert'):
                    buckets = [bucket for layout in values if layout['family'] == family for bucket in layout['buckets']]
                    families[family] = dict(buffer_elements=sum(bucket['end'] - bucket['start'] for bucket in buckets),
                        optimizer_owned_elements=sum(bucket['optimizer_owned_elements'] for bucket in buckets))
                result['native_ddp_buffer_counts'][rank] = families
    return result
