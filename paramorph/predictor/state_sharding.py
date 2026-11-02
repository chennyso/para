import math

from graph.process_groups import process_groups


FSDP_STRATEGIES = ('no_shard', 'optim', 'optim_grads', 'optim_grads_params')
OUTER_FSDP_STRATEGIES = ('no_shard', 'optim')


def _value(request, name, default):
    value = request.get(name)
    return default if value is None else value


def _expert_dp_size(config):
    return config.world_size // (config.etp * config.ep * config.pp)


def state_sharding_errors(request, config):
    errors = []
    backend = _value(request, 'execution_backend', 'megatron_ddp')
    optimizer_sharding = _value(request, 'optimizer_sharding', 'distributed')
    strategy = _value(request, 'fsdp_sharding_strategy', 'no_shard')
    dense = _value(request, 'dense_sharding_strategy', strategy)
    expert = _value(request, 'expert_sharding_strategy', strategy)
    dense_dp = config.dp * config.cp
    inner = _value(request, 'dp_inner_size', dense_dp)
    outer = _value(request, 'dp_outer_size', 1)
    expert_dp = _expert_dp_size(config)
    expert_inner = _value(request, 'expert_dp_inner_size', expert_dp)
    expert_outer = _value(request, 'expert_dp_outer_size', 1)
    outer_strategy = _value(request, 'outer_sharding_strategy', 'no_shard')
    expert_outer_strategy = _value(request, 'expert_outer_sharding_strategy', 'no_shard')
    memory_policy = request.get('memory_policy') or {}
    optimizer_offload_fraction = _value(request, 'optimizer_offload_fraction',
        memory_policy.get('optimizer_offload_fraction', 0.0))
    optimizer_transfer_overlap = bool(_value(request, 'optimizer_transfer_overlap', False))
    if (isinstance(optimizer_offload_fraction, bool)
            or not isinstance(optimizer_offload_fraction, (int, float))
            or not 0 <= optimizer_offload_fraction <= 1):
        errors.append('CPU Optimizer Offload比例必须位于零至一之间')
    if bool(request.get('offload')) != (optimizer_offload_fraction > 0):
        errors.append('Offload开关与CPU Optimizer Offload比例不一致')
    if backend not in ('megatron_ddp', 'megatron_fsdp'):
        errors.append('执行后端需要使用megatron_ddp或者megatron_fsdp')
    if any(value not in FSDP_STRATEGIES for value in (strategy, dense, expert)):
        errors.append('FSDP分片策略未注册')
    if outer_strategy not in OUTER_FSDP_STRATEGIES or expert_outer_strategy not in OUTER_FSDP_STRATEGIES:
        errors.append('Outer FSDP仅支持no_shard或者optim')
    if (type(inner) is not int or type(outer) is not int or inner < 1 or outer < 1
            or inner * outer != dense_dp):
        errors.append('Dense DP-Inner与DP-Outer必须完整覆盖Dense DP-CP分片轴')
    if (type(expert_inner) is not int or type(expert_outer) is not int
            or expert_inner < 1 or expert_outer < 1 or expert_inner * expert_outer != expert_dp):
        errors.append('Expert DP-Inner与DP-Outer必须完整覆盖Expert DP')
    if (outer_strategy != 'no_shard' or expert_outer_strategy != 'no_shard') and not _value(
            request, 'fsdp_hybrid_outer_supported', False):
        errors.append('当前后端没有声明Hybrid FSDP Outer Sharding能力')
    if outer_strategy == 'optim' and dense != 'optim_grads_params':
        errors.append('Dense HFSDP要求Inner策略使用optim_grads_params')
    if expert_outer_strategy == 'optim' and expert != 'optim_grads_params':
        errors.append('Expert HFSDP要求Inner策略使用optim_grads_params')
    if backend == 'megatron_ddp':
        if optimizer_sharding not in ('distributed', 'replicated'):
            errors.append('Megatron DDP需要选择distributed或者replicated Optimizer状态')
        if (_value(request, 'precision_aware_optimizer', False)
                and optimizer_sharding != 'distributed'):
            errors.append('Precision-Aware Optimizer要求Megatron Distributed Optimizer')
        if any(value != 'no_shard' for value in
                (strategy, dense, expert, outer_strategy, expert_outer_strategy)):
            errors.append('Megatron DDP执行后端不能启用FSDP分片')
        if (inner, outer, expert_inner, expert_outer) != (dense_dp, 1, expert_dp, 1):
            errors.append('Megatron DDP没有FSDP DeviceMesh分解')
    if backend == 'megatron_fsdp':
        if config.etp != 1:
            errors.append('当前Megatron-FSDP适配器要求ETP为一，缺少Stage内TP-DP-CP组合进程组')
        if request.get('rank_mapping', list(range(config.world_size))) != list(range(config.world_size)):
            errors.append('当前Megatron-FSDP适配器要求连续物理Rank映射，非连续映射会产生张量元数据不一致')
        fsdp_version = _value(request, 'megatron_fsdp_version', 1)
        if fsdp_version != 1:
            errors.append('ParaMorph固定使用Megatron-FSDP v1')
        supported_dense = tuple(_value(request, 'fsdp_supported_sharding_strategies', FSDP_STRATEGIES))
        supported_expert = tuple(_value(request, 'fsdp_supported_expert_sharding_strategies', supported_dense))
        if dense not in supported_dense:
            errors.append('当前部署的Megatron-FSDP不支持所选Dense分片策略')
        if expert not in supported_expert:
            errors.append('当前部署的Megatron-FSDP不支持所选Expert分片策略')
        if optimizer_sharding != 'distributed':
            errors.append('Megatron-FSDP v1需要使用DistributedOptimizer')
        if (_value(request, 'precision_aware_optimizer', False)
                and not _value(request, 'fsdp_precision_aware_optimizer_supported', False)):
            errors.append('当前Megatron-FSDP版本没有声明Precision-Aware Optimizer能力')
        if config.vpp > 1 and not _value(request, 'fsdp_multi_chunk_supported', False):
            errors.append('当前部署的Megatron-FSDP版本没有声明多Model Chunk能力')
        if _value(request, 'optimizer_checkpoint_mode', 'disabled') != 'disabled' and not _value(
                request, 'fsdp_optimizer_checkpoint_supported', False):
            errors.append('当前Megatron-FSDP版本没有声明Optimizer Checkpoint保存或恢复能力')
        if (outer != expert_outer and not _value(
                request, 'fsdp_separate_expert_outer_mesh_supported', False)):
            errors.append('当前部署的Megatron-FSDP要求Dense与Expert共享Outer轴大小')
    if optimizer_offload_fraction > 0:
        if backend != 'megatron_ddp' or optimizer_sharding != 'distributed':
            errors.append('CPU Optimizer Offload要求Megatron DDP与Distributed Optimizer')
        if not _value(request, 'precision_aware_optimizer', False):
            errors.append('CPU Optimizer Offload要求Precision-Aware Optimizer')
    if optimizer_transfer_overlap and optimizer_offload_fraction == 0:
        errors.append('Optimizer传输重叠要求启用CPU Optimizer Offload')
    unit = _value(request, 'fsdp_unit', 'transformer_layer')
    if unit not in ('transformer_layer', 'model_chunk'):
        errors.append('FSDP Unit需要使用transformer_layer或者model_chunk')
    if unit not in tuple(_value(request, 'fsdp_supported_units', ('transformer_layer', 'model_chunk'))):
        errors.append('当前部署的Megatron-FSDP不支持所选FSDP Unit')
    if type(_value(request, 'fsdp_prefetch_distance', 1)) is not int or _value(
            request, 'fsdp_prefetch_distance', 1) < 1:
        errors.append('FSDP Prefetch距离必须为正整数')
    checkpoint_format = _value(request, 'checkpoint_format',
        'fsdp_dtensor' if backend == 'megatron_fsdp' else 'torch_dist')
    if checkpoint_format not in ('torch_dist', 'fsdp_dtensor'):
        errors.append('检查点格式需要使用torch_dist或者fsdp_dtensor')
    return errors


def state_sharding_config(request, config):
    errors = state_sharding_errors(request, config)
    if errors:
        raise ValueError('；'.join(errors))
    backend = _value(request, 'execution_backend', 'megatron_ddp')
    strategy = _value(request, 'fsdp_sharding_strategy', 'no_shard')
    expert_dp = _expert_dp_size(config)
    return dict(execution_backend=backend,
        optimizer_sharding=_value(request, 'optimizer_sharding',
            'distributed'),
        fsdp_sharding_strategy=strategy,
        dense_sharding_strategy=_value(request, 'dense_sharding_strategy', strategy),
        expert_sharding_strategy=_value(request, 'expert_sharding_strategy', strategy),
        dp_inner_size=_value(request, 'dp_inner_size', config.dp * config.cp),
        dp_outer_size=_value(request, 'dp_outer_size', 1),
        expert_dp_inner_size=_value(request, 'expert_dp_inner_size', expert_dp),
        expert_dp_outer_size=_value(request, 'expert_dp_outer_size', 1),
        outer_sharding_strategy=_value(request, 'outer_sharding_strategy', 'no_shard'),
        expert_outer_sharding_strategy=_value(request, 'expert_outer_sharding_strategy', 'no_shard'),
        fsdp_unit=_value(request, 'fsdp_unit', 'transformer_layer'),
        fsdp_prefetch=bool(_value(request, 'fsdp_prefetch', False)),
        fsdp_prefetch_distance=_value(request, 'fsdp_prefetch_distance', 1),
        fsdp_reshard_after_forward=bool(_value(request, 'fsdp_reshard_after_forward', True)),
        fsdp_double_buffer=bool(_value(request, 'fsdp_double_buffer', False)),
        fsdp_bucket_elements=_value(request, 'fsdp_bucket_elements',
            _value(request, 'bucket_elements', 40000000)),
        checkpoint_format=_value(request, 'checkpoint_format',
            'fsdp_dtensor' if backend == 'megatron_fsdp' else 'torch_dist'),
        checkpoint_reshard=bool(_value(request, 'checkpoint_reshard', True)),
        optimizer_checkpoint_mode=_value(request, 'optimizer_checkpoint_mode', 'disabled'),
        optimizer_offload_fraction=float(_value(request, 'optimizer_offload_fraction',
            (request.get('memory_policy') or {}).get('optimizer_offload_fraction', 0.0))),
        optimizer_transfer_overlap=bool(_value(request, 'optimizer_transfer_overlap', False)),
        megatron_fsdp_version=_value(request, 'megatron_fsdp_version', 1),
        megatron_revision=request.get('megatron_revision'),
        supported_dense_sharding_strategies=list(_value(
            request, 'fsdp_supported_sharding_strategies', FSDP_STRATEGIES)),
        supported_expert_sharding_strategies=list(_value(
            request, 'fsdp_supported_expert_sharding_strategies',
            _value(request, 'fsdp_supported_sharding_strategies', FSDP_STRATEGIES))),
        supported_fsdp_units=list(_value(request, 'fsdp_supported_units',
            ('transformer_layer', 'model_chunk'))),
        fsdp_multi_chunk_supported=bool(_value(request, 'fsdp_multi_chunk_supported', False)),
        fsdp_hybrid_outer_supported=bool(_value(request, 'fsdp_hybrid_outer_supported', False)),
        fsdp_optimizer_checkpoint_supported=bool(_value(
            request, 'fsdp_optimizer_checkpoint_supported', False)),
        fsdp_separate_expert_outer_mesh_supported=bool(_value(
            request, 'fsdp_separate_expert_outer_mesh_supported', False)))


def _widths(request):
    widths = {'bfloat16': 2, 'float16': 2, 'float32': 4}
    fsdp = request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp'
    fields = dict(parameter='precision', gradient='main_grads_dtype',
        master='megatron_fsdp_main_params_dtype' if fsdp else 'main_params_dtype',
        first_moment='exp_avg_dtype', second_moment='exp_avg_sq_dtype')
    if fsdp and request.get('megatron_fsdp_main_grads_dtype') is not None:
        fields['gradient'] = 'megatron_fsdp_main_grads_dtype'
    result = {}
    for role, field in fields.items():
        dtype = request.get(field)
        if not isinstance(dtype, str):
            raise ValueError(f'状态所有权需要实际后端的明确{field}')
        dtype = dtype.removeprefix('torch.')
        if dtype not in widths:
            raise ValueError(f'状态所有权不支持{field}={dtype}')
        result[role] = widths[dtype]
    return result


def _factors(strategy, group_size, outer_strategy, outer_size):
    shards = max(1, group_size)
    if strategy == 'no_shard':
        factors = dict(parameter=1, gradient=1, optimizer=1, unshard=0)
    elif strategy == 'optim':
        factors = dict(parameter=1, gradient=1, optimizer=shards, unshard=0)
    elif strategy == 'optim_grads':
        factors = dict(parameter=1, gradient=shards, optimizer=shards, unshard=0)
    else:
        factors = dict(parameter=shards, gradient=shards, optimizer=shards,
            unshard=1 if shards > 1 else 0)
    if outer_strategy == 'optim':
        factors['optimizer'] *= max(1, outer_size)
    return factors


def _mesh_groups(group, inner_size):
    if len(group) % inner_size:
        raise ValueError('FSDP DeviceMesh不能完整覆盖状态分片进程组')
    inner = [group[offset:offset + inner_size] for offset in range(0, len(group), inner_size)]
    outer = [[row[position] for row in inner] for position in range(inner_size)]
    return inner, outer


def _rank_groups(group, rank, inner_size):
    inner_groups, outer_groups = _mesh_groups(group, inner_size)
    return (next(row for row in inner_groups if rank in row),
        next(row for row in outer_groups if rank in row))


def parameter_state_ownership(counts, config, request):
    policy = state_sharding_config(request, config)
    groups = process_groups(request)['logical']
    widths = _widths(request)
    backend = policy['execution_backend']
    offload_fraction = policy['optimizer_offload_fraction']
    bucket_elements = policy['fsdp_bucket_elements']
    if type(bucket_elements) is not int or bucket_elements < 1:
        raise ValueError('FSDP Bucket元素数量必须为正整数')
    output = {}
    for rank in range(config.world_size):
        dense_group = next(group for group in groups['dp_cp'] if rank in group)
        expert_group = next(group for group in groups['expert_dp'] if rank in group)
        components = {}
        specifications = (
            ('dense', counts['dense'][rank], dense_group,
                policy['dense_sharding_strategy'], policy['dp_inner_size'],
                policy['dp_outer_size'], policy['outer_sharding_strategy']),
            ('expert', counts['expert'][rank], expert_group,
                policy['expert_sharding_strategy'], policy['expert_dp_inner_size'],
                policy['expert_dp_outer_size'], policy['expert_outer_sharding_strategy']))
        for family, elements, group, strategy, inner_size, outer_size, outer_strategy in specifications:
            inner_group, outer_group = _rank_groups(group, rank, inner_size)
            active_strategy = strategy if backend == 'megatron_fsdp' else 'no_shard'
            active_outer = outer_strategy if backend == 'megatron_fsdp' else 'no_shard'
            factors = _factors(active_strategy, len(inner_group), active_outer, len(outer_group))
            optimizer_shards = (len(group) if backend == 'megatron_ddp'
                and policy['optimizer_sharding'] == 'distributed' else factors['optimizer'])
            parameter_elements = math.ceil(elements / factors['parameter'])
            gradient_elements = math.ceil(elements / factors['gradient'])
            optimizer_elements = math.ceil(elements / optimizer_shards)
            native = counts.get('native_ddp_buffer_counts', {}).get(rank, {}).get(family)
            if native is not None:
                if backend != 'megatron_ddp' or policy['optimizer_sharding'] != 'distributed':
                    raise ValueError('原生DistributedOptimizer参数布局与当前后端不同')
                parameter_elements = gradient_elements = native['buffer_elements']
                optimizer_elements = native['optimizer_owned_elements']
            parameter = parameter_elements * widths['parameter']
            gradient = gradient_elements * widths['gradient']
            optimizer_state_width = widths['master'] + widths['first_moment'] + widths['second_moment']
            offloaded_elements = math.ceil(optimizer_elements * offload_fraction)
            gpu_optimizer_elements = optimizer_elements - offloaded_elements
            optimizer_gpu = gpu_optimizer_elements * optimizer_state_width
            optimizer_host = offloaded_elements * optimizer_state_width
            pinned_gradient = offloaded_elements * widths['gradient']
            parameter_padding = parameter_elements * factors['parameter'] - elements
            gradient_padding = gradient_elements * factors['gradient'] - elements
            components[family] = dict(local_unsharded_elements=elements,
                data_parallel_group=list(group), inner_sharding_group=list(inner_group),
                outer_sharding_group=list(outer_group), inner_sharding_group_size=len(inner_group),
                outer_sharding_group_size=len(outer_group), strategy=active_strategy,
                outer_strategy=active_outer, persistent_parameter_bytes=parameter,
                persistent_gradient_bytes=gradient, gpu_optimizer_bytes=optimizer_gpu,
                host_optimizer_bytes=optimizer_host, host_pinned_gradient_bytes=pinned_gradient,
                unshard_parameter_bytes=elements * widths['parameter'] if factors['unshard'] else 0,
                parameter_padding_bytes=parameter_padding * widths['parameter'],
                gradient_padding_bytes=gradient_padding * widths['gradient'],
                state_layout_source=('native_distributed_optimizer_parameter_intervals'
                    if native is not None else 'aggregate_parameter_ownership'),
                sharding_effective=(factors['parameter'] > 1 or factors['gradient'] > 1
                    or optimizer_shards > 1))
        gpu_persistent = sum(row['persistent_parameter_bytes'] + row['persistent_gradient_bytes']
            + row['gpu_optimizer_bytes'] for row in components.values())
        model_buffers = counts.get('model_buffer_bytes', {}).get(rank, 0)
        gpu_persistent += model_buffers
        output[rank] = dict(components=components, gpu_persistent_bytes=gpu_persistent,
            model_buffer_bytes=model_buffers,
            host_optimizer_bytes=sum(row['host_optimizer_bytes'] for row in components.values()),
            host_pinned_buffer_bytes=sum(row['host_pinned_gradient_bytes'] for row in components.values()),
            fsdp_unshard_bytes=max(row['unshard_parameter_bytes'] for row in components.values()),
            fsdp_padding_bytes=sum(row['parameter_padding_bytes'] + row['gradient_padding_bytes']
                for row in components.values()),
            fsdp_communication_unit_bytes=min(bucket_elements * widths['parameter'],
                max((row['unshard_parameter_bytes'] for row in components.values()), default=0)),
            source='model_local_parameter_ownership_and_actual_dense_expert_process_groups',
            policy=policy, widths=widths)
    return output


def phased_memory_bounds(counts, config, request, activation_bytes, workspace_bytes,
                         reserve_bytes, checkpoint_buffer_bytes=0):
    ownership = parameter_state_ownership(counts, config, request)
    results = {}
    capacity = request.get('gpu_capacity_bytes')
    host_capacity = request.get('host_capacity_bytes')
    for rank, row in ownership.items():
        policy = row['policy']
        unshard = row['fsdp_unshard_bytes']
        prefetch = unshard if policy['fsdp_prefetch'] or policy['fsdp_double_buffer'] else 0
        communication_unit = row['fsdp_communication_unit_bytes']
        gpu_phases = dict(
            initialization=row['gpu_persistent_bytes'] + workspace_bytes + reserve_bytes,
            checkpoint_load=row['gpu_persistent_bytes'] + checkpoint_buffer_bytes
                + (unshard if policy['checkpoint_reshard'] else 0) + workspace_bytes + reserve_bytes,
            forward_backward=row['gpu_persistent_bytes'] + activation_bytes.get(rank, 0)
                + unshard + prefetch + communication_unit + workspace_bytes + reserve_bytes,
            gradient_communication=row['gpu_persistent_bytes'] + communication_unit
                + workspace_bytes + reserve_bytes,
            optimizer_step=row['gpu_persistent_bytes'] + workspace_bytes + reserve_bytes)
        host_phases = dict(checkpoint_load=checkpoint_buffer_bytes,
            forward_backward=row['host_optimizer_bytes'],
            optimizer_step=row['host_optimizer_bytes'] + row['host_pinned_buffer_bytes'])
        danger_phase = max(gpu_phases, key=gpu_phases.get)
        results[rank] = dict(gpu_phases=gpu_phases, host_phases=host_phases,
            gpu_training_peak_bytes=gpu_phases['forward_backward'],
            gpu_loading_peak_bytes=gpu_phases['checkpoint_load'],
            gpu_optimizer_peak_bytes=gpu_phases['optimizer_step'],
            host_cpu_memory_peak_bytes=max(host_phases.values()),
            most_dangerous_phase=danger_phase, most_dangerous_gpu_bytes=gpu_phases[danger_phase],
            capacity_bytes=capacity, host_capacity_bytes=host_capacity, ownership=row)
    return results
