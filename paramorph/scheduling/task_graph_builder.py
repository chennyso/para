from core_data_models import Task, TaskType, ExecutionDAG
from graph.finalize_model_grads import FinalizeModelGradsCompiler
from graph.process_groups import process_groups
from predictor.state_sharding import state_sharding_config


class TaskGraphBuilder:
    @staticmethod
    def fsdp_collective_metadata(unit_elements, group, dtype, collective):
        widths = {'bfloat16': 2, 'float16': 2, 'float32': 4}
        if dtype not in widths:
            raise ValueError('FSDP通信需要明确支持的Tensor精度')
        if (not group or len(set(group)) != len(group)
                or any(type(rank) is not int or rank < 0 for rank in group)):
            raise ValueError('FSDP通信需要不同的合法Rank')
        if not unit_elements or any(type(value) is not int or value <= 0 for value in unit_elements):
            raise ValueError('FSDP通信Unit需要正整数参数数量')
        if collective not in ('all_gather', 'reduce_scatter', 'all_reduce'):
            raise ValueError('FSDP通信类型没有定义消息字节语义')
        width, participants = widths[dtype], len(group)
        full = [value * width for value in unit_elements]
        if collective == 'all_reduce':
            padded, payloads = full, full
            definition = 'local_full_tensor_bytes'
        else:
            shard_elements = [(value + participants - 1) // participants for value in unit_elements]
            payloads = [value * width for value in shard_elements]
            padded = [value * participants for value in payloads]
            definition = ('local_input_shard_bytes' if collective == 'all_gather'
                else 'local_output_shard_bytes')
        return dict(collective=collective, payload_bytes=max(padded),
            communication_full_payloads_bytes=padded,
            communication_unpadded_payloads_bytes=full,
            communication_payloads=payloads,
            communication_padding_bytes=[left - right for left, right in zip(padded, full)],
            communication_dtype=dtype, communication_payload_definition=definition,
            communication_padding_rule='equal_shards_element_count_ceiling',
            communication_padding_source='analytical_equal_partition',
            actual_padding_measured=False)

    def build(self, model, config, regions, num_microbatches):
        if num_microbatches != config.num_microbatches:
            raise ValueError('任务图microbatch数量与有效global batch不一致')
        regions = sorted(regions, key=lambda region: region.region_id)
        if len(regions) != config.total_vp_stages:
            raise ValueError('Region数量与PP×VPP不一致')
        offset = 0
        for index, region in enumerate(regions):
            if (region.layer_start != offset or region.layer_end <= offset or
                region.pp_rank != index % config.pp or region.vp_rank != index // config.pp):
                raise ValueError('Region顺序或物理设备映射不合法')
            offset = region.layer_end
        if offset != model.num_layers:
            raise ValueError('Region没有完整覆盖模型层')
        dag = ExecutionDAG({}, [])
        lanes = config.tp * config.cp * config.dp
        compute_dtype = config.execution_implementation.get('precision', 'bfloat16')
        if compute_dtype not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('流水线通信需要明确的实际计算精度')

        def add(key, rank, kind, mb=None, region=None, metadata=None):
            task = Task(key, rank, kind, mb, region, metadata=dict(metadata or {}))
            task.metadata['logical_rank'] = rank
            task.metadata['physical_rank'] = config.device_mapping[rank] if config.device_mapping else rank
            if 'group_ranks' in task.metadata:
                task.metadata['physical_group_ranks'] = [config.device_mapping[participant]
                    if config.device_mapping else participant for participant in task.metadata['group_ranks']]
            task.metadata['cuda_stream'] = 'backend_assigned'
            task.metadata['backend_callable'] = {
                TaskType.FORWARD_COMPUTE: 'megatron.core.pipeline_parallel.schedules.forward_step',
                TaskType.BACKWARD_COMPUTE: 'megatron.core.pipeline_parallel.schedules.backward_step',
                TaskType.PP_P2P_SEND: 'megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator',
                TaskType.PP_P2P_RECV: 'megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator',
                TaskType.DP_REDUCE: 'megatron.core.distributed.finalize_model_grads.finalize_model_grads',
                TaskType.OPTIMIZER_STEP: 'megatron.core.optimizer.optimizer.MegatronOptimizer.step',
                TaskType.CUDA_SYNC: 'torch.cuda.synchronize',
                TaskType.FSDP_UNSHARD: 'megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module.FsdpModule.unshard',
                TaskType.FSDP_PREFETCH: 'megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module.FsdpModule._prefetch_parameter_groups',
                TaskType.FSDP_RESHARD: 'megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module.FsdpModule.reshard',
                TaskType.FSDP_GRAD_REDUCE_SCATTER: 'megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer.DBuffer.reduce_scatter',
                TaskType.FSDP_OUTER_GRAD_REDUCE: 'megatron.core.distributed.fsdp.mcore_fsdp_adapter.outer_fsdp_gradient_reduce',
                TaskType.FSDP_OUTER_PARAM_SYNC: 'megatron.core.distributed.fsdp.mcore_fsdp_adapter.outer_fsdp_parameter_sync'}[kind]
            dag.add_task(task)
            return key

        for lane in range(lanes):
            for mb in range(num_microbatches):
                for index, region in enumerate(regions):
                    rank = region.pp_rank * lanes + lane
                    metadata = dict(layer_ids=list(range(region.layer_start, region.layer_end)),
                        rank_class=rank, includes=['tp', 'cp', 'ep'], operation='region',
                        vp_stage=region.vp_rank)
                    forward = add(f'F:{lane}:{mb}:{index}', rank, TaskType.FORWARD_COMPUTE,
                                  mb, index, metadata)
                    backward = add(f'B:{lane}:{mb}:{index}', rank, TaskType.BACKWARD_COMPUTE,
                                   mb, index, metadata)
                    dag.add_dependency(forward, backward)
                    if mb:
                        dag.add_dependency(f'F:{lane}:{mb - 1}:{index}', forward)
                        dag.add_dependency(f'B:{lane}:{mb - 1}:{index}', backward)
                    if index:
                        previous_rank = regions[index - 1].pp_rank * lanes + lane
                        previous_forward = f'F:{lane}:{mb}:{index - 1}'
                        if previous_rank == rank:
                            dag.add_dependency(previous_forward, forward)
                        else:
                            shape = [config.sequence_length // config.cp //
                                (config.tp if config.use_sequence_parallel else 1),
                                config.micro_batch_size, model.hidden_size]
                            channel = f'forward:{lane}:{index}'
                            send = add(f'SF:{lane}:{mb}:{index}', previous_rank, TaskType.PP_P2P_SEND,
                                mb, index, dict(operation='p2p', direction='forward',
                                    group_ranks=[previous_rank, rank], shape=shape, dtype=compute_dtype,
                                    channel=channel, producer_task=previous_forward))
                            recv = add(f'RF:{lane}:{mb}:{index}', rank, TaskType.PP_P2P_RECV,
                                mb, index, dict(operation='receive_completion', direction='forward',
                                    group_ranks=[previous_rank, rank], shape=shape, dtype=compute_dtype, channel=channel))
                            dag.tasks[forward].metadata['input_task'] = recv
                            if mb:
                                dag.add_dependency(f'SF:{lane}:{mb - 1}:{index}', send)
                            dag.add_dependency(previous_forward, send)
                            dag.add_dependency(send, recv)
                            dag.add_dependency(recv, forward)
                for index in reversed(range(len(regions) - 1)):
                    rank = regions[index].pp_rank * lanes + lane
                    next_rank = regions[index + 1].pp_rank * lanes + lane
                    if rank == next_rank:
                        dag.add_dependency(f'B:{lane}:{mb}:{index + 1}', f'B:{lane}:{mb}:{index}')
                    else:
                        shape = [config.sequence_length // config.cp //
                            (config.tp if config.use_sequence_parallel else 1),
                            config.micro_batch_size, model.hidden_size]
                        producer = f'B:{lane}:{mb}:{index + 1}'
                        channel = f'backward:{lane}:{index}'
                        send = add(f'SB:{lane}:{mb}:{index}', next_rank, TaskType.PP_P2P_SEND,
                            mb, index, dict(operation='p2p', direction='backward',
                                group_ranks=[next_rank, rank], shape=shape, dtype=compute_dtype,
                                channel=channel, producer_task=producer))
                        recv = add(f'RB:{lane}:{mb}:{index}', rank, TaskType.PP_P2P_RECV,
                            mb, index, dict(operation='receive_completion', direction='backward',
                                group_ranks=[next_rank, rank], shape=shape, dtype=compute_dtype, channel=channel))
                        dag.tasks[f'B:{lane}:{mb}:{index}'].metadata['input_task'] = recv
                        if mb:
                            dag.add_dependency(f'SB:{lane}:{mb - 1}:{index}', send)
                        dag.add_dependency(f'B:{lane}:{mb}:{index + 1}', send)
                        dag.add_dependency(send, recv)
                        dag.add_dependency(recv, f'B:{lane}:{mb}:{index}')
            for pp in range(config.pp):
                rank = pp * lanes + lane
                sync = add(f'DP:{rank}', rank, TaskType.DP_REDUCE,
                    metadata=dict(operation='gradient_finalize', rank_class=rank))
                for mb in range(num_microbatches):
                    for region in regions:
                        if region.pp_rank == pp:
                            dag.add_dependency(f'B:{lane}:{mb}:{region.region_id}', sync)
                optimizer = add(f'OPT:{rank}', rank, TaskType.OPTIMIZER_STEP,
                    metadata=dict(operation='optimizer_step', rank_class=rank))
                dag.add_dependency(sync, optimizer)
                tail = add(f'END:{rank}', rank, TaskType.CUDA_SYNC,
                    metadata=dict(operation='post_optimizer_sync', rank_class=rank))
                dag.add_dependency(optimizer, tail)
        for pp in range(config.pp):
            for tensor_lane in range(config.tp * config.cp):
                ranks = [pp * lanes + dp * config.tp * config.cp + tensor_lane for dp in range(config.dp)]
                for rank in ranks:
                    for other in ranks:
                        dag.add_dependency(f'DP:{other}', f'OPT:{rank}')
        implementation = config.execution_implementation
        if implementation.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
            fsdp_request = dict(implementation, tp=config.tp, pp=config.pp, cp=config.cp,
                dp=config.dp, ep=config.ep, etp=config.etp, offload=False,
                node_count=1, gpus_per_node=config.world_size)
            groups = process_groups(fsdp_request)['logical']
            policy = state_sharding_config(fsdp_request, config)
            prefetch = policy['fsdp_prefetch']
            reshard = policy['fsdp_reshard_after_forward']
            parameter_dtype = implementation.get('precision')
            if parameter_dtype not in ('bfloat16', 'float16', 'float32'):
                raise ValueError('FSDP参数通信需要明确的训练精度')
            gradient_dtype = (implementation.get('grad_comm_dtype')
                or implementation.get('main_grads_dtype'))
            if gradient_dtype is None:
                if implementation.get('precision_aware_optimizer', False):
                    raise ValueError('Precision-aware FSDP需要明确的grad_comm_dtype或者main_grads_dtype')
                gradient_dtype = 'float32'
            gradient_dtype = gradient_dtype.removeprefix('torch.')
            if gradient_dtype not in ('bfloat16', 'float16', 'float32'):
                raise ValueError('FSDP梯度通信需要明确支持的梯度精度')

            def mesh_group(kind, rank, inner_size):
                full = next(row for row in groups[kind] if rank in row)
                rows = [full[offset:offset + inner_size]
                    for offset in range(0, len(full), inner_size)]
                return next(row for row in rows if rank in row)

            def replace_predecessors(target, predecessors):
                original = list(dag.tasks[target].dependencies)
                dag.edges = [(source, destination) for source, destination in dag.edges
                    if not (destination == target and source in original)]
                dag.tasks[target].dependencies = []
                for predecessor in predecessors:
                    dag.add_dependency(predecessor, target)

            def replace_successors(source, replacement, excluded):
                targets = [target for predecessor, target in dag.edges
                    if predecessor == source and target not in excluded]
                dag.edges = [(predecessor, target) for predecessor, target in dag.edges
                    if not (predecessor == source and target in targets)]
                for target in targets:
                    dag.tasks[target].dependencies.remove(source)
                    dag.add_dependency(replacement, target)

            for key, task in list(dag.tasks.items()):
                if task.task_type not in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE):
                    continue
                layers = task.metadata['layer_ids']
                original = list(task.dependencies)
                components = self.region_parameter_components(model, config, layers)
                specifications = (
                    ('dense', components['dense'], policy['dense_sharding_strategy'], 'dp_cp',
                        policy['dp_inner_size']),
                    ('expert', components['expert'], policy['expert_sharding_strategy'], 'expert_dp',
                        policy['expert_dp_inner_size']))
                unshards = []
                for family, elements, strategy, group_kind, inner_size in specifications:
                    group = mesh_group(group_kind, task.rank, inner_size)
                    if not elements or strategy != 'optim_grads_params' or len(group) == 1:
                        continue
                    unit_elements = ([self.region_parameter_components(model, config, [layer])[family]
                        for layer in layers] if policy['fsdp_unit'] == 'transformer_layer' else [elements])
                    communication = self.fsdp_collective_metadata(
                        [value for value in unit_elements if value], group, parameter_dtype, 'all_gather')
                    if prefetch:
                        prefetch_task = add(f'FSP:{family}:{key}', task.rank, TaskType.FSDP_PREFETCH,
                            task.microbatch_id, task.region_id,
                            dict(operation='fsdp_parameter_prefetch', state_family=family,
                                group_ranks=group, unit_layer_ids=layers, **communication,
                                prefetch_distance=policy['fsdp_prefetch_distance'],
                                compute_task=key, cost_role='prefetched_all_gather'))
                        prefetch_dependencies = list(original)
                        for _ in range(policy['fsdp_prefetch_distance']):
                            expanded = []
                            for dependency in prefetch_dependencies:
                                predecessor = dag.tasks[dependency]
                                if predecessor.rank == task.rank and predecessor.is_compute:
                                    expanded.extend(predecessor.dependencies)
                                else:
                                    expanded.append(dependency)
                            prefetch_dependencies = list(dict.fromkeys(expanded))
                        for dependency in prefetch_dependencies:
                            dag.add_dependency(dependency, prefetch_task)
                    unshard = add(f'FSU:{family}:{key}', task.rank, TaskType.FSDP_UNSHARD,
                        task.microbatch_id, task.region_id, dict(operation='fsdp_parameter_unshard',
                            state_family=family, group_ranks=group, unit_layer_ids=layers,
                            **communication,
                            unit=policy['fsdp_unit'], compute_task=key,
                            cost_role='prefetch_wait' if prefetch else 'synchronous_all_gather',
                            resource_kind='shared_nic_or_pcie_path'))
                    if prefetch:
                        dag.add_dependency(prefetch_task, unshard)
                        for dependency in original:
                            dag.add_dependency(dependency, unshard)
                    else:
                        for dependency in original:
                            dag.add_dependency(dependency, unshard)
                    unshards.append(unshard)
                if unshards:
                    replace_predecessors(key, unshards)
                release_tasks = []
                if task.task_type == TaskType.FORWARD_COMPUTE and reshard:
                    for family, elements, strategy, group_kind, inner_size in specifications:
                        group = mesh_group(group_kind, task.rank, inner_size)
                        if not elements or strategy != 'optim_grads_params' or len(group) == 1:
                            continue
                        release = add(f'FSR:{family}:{key}', task.rank, TaskType.FSDP_RESHARD,
                            task.microbatch_id, task.region_id,
                            dict(operation='fsdp_post_forward_reshard', state_family=family,
                                group_ranks=group, payload_bytes=0, unit_layer_ids=layers,
                                releases=f'FSU:{family}:{key}'))
                        dag.add_dependency(key, release)
                        release_tasks.append(release)
                if release_tasks:
                    replace_successors(key, release_tasks[-1], set(release_tasks))
                    for left, right in zip(release_tasks, release_tasks[1:]):
                        dag.add_dependency(left, right)
                reduce_tasks = []
                if task.task_type == TaskType.BACKWARD_COMPUTE:
                    for family, elements, strategy, group_kind, inner_size in specifications:
                        group = mesh_group(group_kind, task.rank, inner_size)
                        if not elements or strategy not in ('optim_grads', 'optim_grads_params') or len(group) == 1:
                            continue
                        unit_elements = ([self.region_parameter_components(model, config, [layer])[family]
                            for layer in layers] if policy['fsdp_unit'] == 'transformer_layer' else [elements])
                        communication = self.fsdp_collective_metadata(
                            [value for value in unit_elements if value], group, gradient_dtype,
                            'reduce_scatter')
                        reduce_task = add(f'FSG:{family}:{key}', task.rank,
                            TaskType.FSDP_GRAD_REDUCE_SCATTER, task.microbatch_id, task.region_id,
                            dict(operation='fsdp_gradient_reduce_scatter', state_family=family,
                                group_ranks=group, unit_layer_ids=layers, **communication,
                                compute_task=key,
                                resource_kind='shared_nic_or_pcie_path'))
                        dag.add_dependency(key, reduce_task)
                        reduce_tasks.append(reduce_task)
                    if reduce_tasks:
                        replace_successors(key, reduce_tasks[-1], set(reduce_tasks))
                        for left, right in zip(reduce_tasks, reduce_tasks[1:]):
                            dag.add_dependency(left, right)
            active_strategies = [policy['dense_sharding_strategy']]
            if model.num_experts is not None:
                active_strategies.append(policy['expert_sharding_strategy'])
            if all(strategy in ('optim_grads', 'optim_grads_params') for strategy in active_strategies):
                for rank in range(config.world_size):
                    task = dag.tasks[f'DP:{rank}']
                    task.task_type = TaskType.CUDA_SYNC
                    task.metadata.update(operation='fsdp_gradient_finalize_barrier',
                        state_families=['dense'] + (['expert'] if model.num_experts is not None else []))
            else:
                for rank in range(config.world_size):
                    dag.tasks[f'DP:{rank}'].metadata['state_families'] = [
                        family for family, strategy in (('dense', policy['dense_sharding_strategy']),
                            ('expert', policy['expert_sharding_strategy']))
                        if (family == 'dense' or model.num_experts is not None)
                        and strategy not in ('optim_grads', 'optim_grads_params')]
            tail_predecessors = {rank: f'OPT:{rank}' for rank in range(config.world_size)}
            for family, group_kind, inner_size, outer_strategy in (
                    ('dense', 'dp_cp', policy['dp_inner_size'], policy['outer_sharding_strategy']),
                    ('expert', 'expert_dp', policy['expert_dp_inner_size'],
                        policy['expert_outer_sharding_strategy'])):
                if family == 'expert' and model.num_experts is None:
                    continue
                for rank in range(config.world_size):
                    full = next(row for row in groups[group_kind] if rank in row)
                    inner_rows = [full[offset:offset + inner_size]
                        for offset in range(0, len(full), inner_size)]
                    position = next(index for index, row in enumerate(inner_rows) if rank in row)
                    local = inner_rows[position].index(rank)
                    outer_group = [row[local] for row in inner_rows]
                    if len(outer_group) == 1:
                        continue
                    sync = f'DP:{rank}'
                    outer_gradient = add(f'FSOG:{family}:{rank}', rank,
                        TaskType.FSDP_OUTER_GRAD_REDUCE,
                        metadata=dict(operation='fsdp_outer_gradient_reduce', state_family=family,
                            group_ranks=outer_group,
                            collective='reduce_scatter' if outer_strategy == 'optim' else 'all_reduce',
                            communication_dtype=gradient_dtype,
                            communication_payload_definition=('local_output_shard_bytes'
                                if outer_strategy == 'optim' else 'local_full_tensor_bytes'),
                            payload_source='local_state_family_gradient', resource_kind='shared_nic_or_pcie_path'))
                    for dependency in list(dag.tasks[sync].dependencies):
                        dag.add_dependency(dependency, outer_gradient)
                    dag.add_dependency(outer_gradient, sync)
                    if outer_strategy == 'optim':
                        previous, tail = tail_predecessors[rank], f'END:{rank}'
                        dag.tasks[tail].dependencies.remove(previous)
                        dag.edges.remove((previous, tail))
                        parameter_sync = add(f'FSOP:{family}:{rank}', rank,
                            TaskType.FSDP_OUTER_PARAM_SYNC,
                            metadata=dict(operation='fsdp_outer_parameter_sync', state_family=family,
                                group_ranks=outer_group, collective='all_gather',
                                communication_dtype=parameter_dtype,
                                communication_payload_definition='local_input_shard_bytes',
                                payload_source='local_state_family_parameter',
                                resource_kind='shared_nic_or_pcie_path'))
                        dag.add_dependency(previous, parameter_sync)
                        dag.add_dependency(parameter_sync, tail)
                        tail_predecessors[rank] = parameter_sync
        groups = process_groups(dict(config.execution_implementation,
            tp=config.tp, pp=config.pp, cp=config.cp, dp=config.dp, ep=config.ep,
            etp=config.etp, expert_dp=config.world_size // (config.etp * config.ep * config.pp),
            node_count=1, gpus_per_node=config.world_size,
            rank_mapping=list(config.device_mapping) if config.device_mapping else list(range(config.world_size))))
        return FinalizeModelGradsCompiler(model, config, groups).compile(dag)

    @staticmethod
    def region_parameter_elements(model, config, layers):
        components = TaskGraphBuilder.region_parameter_components(model, config, layers)
        return components['dense'] + components['expert']

    @staticmethod
    def region_parameter_components(model, config, layers):
        hidden = model.hidden_size
        dimension = model.head_dimension or hidden // model.num_attention_heads
        heads = model.num_attention_heads // config.tp
        query_groups = max(1, (model.num_query_groups or model.num_attention_heads) // config.tp)
        attention = hidden * ((heads + 2 * query_groups) * dimension + heads * dimension) + 2 * hidden
        if (model.finalize_grad_semantics or {}).get('qk_layernorm', False):
            attention += 2 * dimension
        dense = 0
        expert = 0
        for layer in layers:
            dense += attention
            if model.layer_types[layer].value == 'moe':
                dense += hidden * model.num_experts
                expert += 3 * hidden * model.ffn_hidden_size * model.num_experts // (config.etp * config.ep)
            else:
                dense += 3 * hidden * model.ffn_hidden_size // config.tp
        return dict(dense=dense, expert=expert)
