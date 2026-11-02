import math

from core_data_models import Task, TaskType


GRADIENT_BYTES = {'float32': 4, 'bfloat16': 2, 'float16': 2}


class FinalizeModelGradsCompiler:
    def __init__(self, model, config, groups):
        self.model = model
        self.config = config
        self.groups = groups
        self.implementation = config.execution_implementation
        self.semantics = model.finalize_grad_semantics or {}

    def _gradient_dtype(self):
        value = self.implementation.get('main_grads_dtype')
        if value is None:
            value = 'bfloat16' if self.implementation.get('precision_aware_optimizer', False) else 'float32'
        value = str(value).removeprefix('torch.')
        if value not in GRADIENT_BYTES:
            raise ValueError('FinalizeModelGrads需要明确的梯度精度')
        return value

    def _embedding_payload(self):
        vocab = self.semantics.get('vocab_size')
        if type(vocab) is not int or vocab <= 0:
            raise ValueError('词嵌入梯度同步需要实际Vocabulary大小')
        padded = math.ceil(vocab / self.config.tp) * self.config.tp
        elements = padded * self.model.hidden_size // self.config.tp
        dtype = self._gradient_dtype()
        backend = self.implementation.get('execution_backend', 'megatron_ddp')
        version = self.implementation.get('megatron_fsdp_version', 1)
        layout = 'full_gradient'
        if backend == 'megatron_fsdp' and version == 1:
            strategy = self.implementation.get('dense_sharding_strategy',
                self.implementation.get('fsdp_sharding_strategy', 'no_shard'))
            if strategy in ('optim_grads', 'optim_grads_params'):
                shard_size = self.implementation.get('dp_inner_size', self.config.dp * self.config.cp)
                elements = math.ceil(elements / shard_size)
                layout = 'megatron_fsdp_v1_local_gradient_shard'
        elif backend == 'megatron_fsdp' and version == 2:
            layout = 'megatron_fsdp_v2_full_logical_gradient_after_uneven_dtensor_unshard'
        return elements * GRADIENT_BYTES[dtype], dtype, layout

    def optimizer_statistics_groups(self):
        config = self.config
        sharding = self.implementation.get('optimizer_sharding') or 'distributed'
        backend = self.implementation.get('execution_backend') or 'megatron_ddp'
        fsdp_strategy = (self.implementation.get('dense_sharding_strategy')
            or self.implementation.get('fsdp_sharding_strategy') or 'no_shard')
        model_parallel = sharding == 'replicated' or (
            backend == 'megatron_fsdp' and fsdp_strategy == 'no_shard')
        if model_parallel:
            groups = [[((pp * config.dp + dp) * config.cp + cp) * config.tp + tp
                for pp in range(config.pp) for tp in range(config.tp)]
                for dp in range(config.dp) for cp in range(config.cp)]
            return groups, 'model_parallel_tp_pp'
        instances = self.implementation.get('num_distributed_optimizer_instances')
        instances = 1 if instances is None else instances
        if type(instances) is not int or instances != 1:
            raise ValueError('Optimizer梯度统计需要已支持的单个DistributedOptimizer实例进程组')
        if sharding != 'distributed':
            raise ValueError('Optimizer梯度统计没有对应的后端进程组语义')
        return [list(range(config.world_size))], 'intra_distributed_optimizer_single_instance'

    def _compile_optimizer_preparation(self, dag):
        clip_grad = self.implementation.get('clip_grad')
        clip_grad = 1.0 if clip_grad is None else clip_grad
        if (isinstance(clip_grad, bool) or not isinstance(clip_grad, (int, float))
                or not math.isfinite(clip_grad) or clip_grad < 0):
            raise ValueError('Optimizer clip_grad需要有限非负数值')
        dtype = self._gradient_dtype()
        groups, group_policy = self.optimizer_statistics_groups()
        group_by_rank = {rank: group for group in groups for rank in group}
        optimizers = {task.rank: task for task in dag.tasks.values()
            if task.task_type == TaskType.OPTIMIZER_STEP}
        if sorted(optimizers) != list(range(self.config.world_size)):
            raise ValueError('Optimizer任务需要完整覆盖每个Rank一次')

        def local(key, rank, kind, dependencies, operation, cost_scope, callable_name):
            physical = self.groups['rank_mapping'][rank]
            task = Task(key, rank, kind, dependencies=list(dependencies),
                resource_requirements=(f'gpu:{physical}:compute',),
                metadata=dict(operation=operation, cost_scope=cost_scope, dtype=dtype,
                    logical_rank=rank, physical_rank=physical,
                    grad_stats_group_ranks=group_by_rank[rank],
                    grad_stats_group_policy=group_policy, clip_grad=clip_grad,
                    cuda_stream='backend_assigned', backend_callable=callable_name,
                    cost_requires_actual_optimizer_component=True,
                    gradient_selection='optimizer.get_grads_for_grad_norm_excludes_shared_and_tp_duplicates'
                        if kind == TaskType.GRADIENT_NORM else 'optimizer.get_parameters'))
            dag.add_task(task)
            dag.edges.extend((dependency, key) for dependency in dependencies)
            return key

        norm_tasks = {}
        for rank, optimizer in optimizers.items():
            old_dependencies = list(optimizer.dependencies)
            prepare_callable = ('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads'
                if (self.implementation.get('optimizer_sharding') or 'distributed') == 'distributed'
                else 'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._copy_model_grads_to_main_grads'
                if self.implementation.get('precision') in ('bfloat16', 'float16')
                else 'megatron.core.optimizer.optimizer.FP32Optimizer.prepare_grads')
            prepared = local(f'OPREP:{rank}', rank, TaskType.OPTIMIZER_PREPARE,
                old_dependencies, 'optimizer_prepare_grads', 'local_optimizer_prepare_grads',
                prepare_callable)
            dag.edges = [(source, destination) for source, destination in dag.edges
                if destination != optimizer.task_id or source not in old_dependencies]
            optimizer.dependencies = []
            norm_tasks[rank] = local(f'ONORM:{rank}', rank, TaskType.GRADIENT_NORM,
                [prepared], 'optimizer_local_grad_norm', 'local_optimizer_gradient_norm',
                'megatron.core.optimizer.clip_grads.get_grad_norm_fp32')
            dag.tasks[norm_tasks[rank]].metadata['component_phase'] = 'pre_collective'

        for index, ranks in enumerate(groups):
            if len(ranks) > 1:
                physical = [self.groups['rank_mapping'][rank] for rank in ranks]
                key = f'ONORM:AR:{index}'
                dependencies = [norm_tasks[rank] for rank in ranks]
                task = Task(key, ranks[0], TaskType.NCCL_GENERIC, dependencies=dependencies,
                    resource_requirements=tuple(f'gpu:{rank}:communication' for rank in physical),
                    metadata=dict(operation='optimizer_gradient_norm_all_reduce',
                        collective='all_reduce', reduce_op='sum', payload_bytes=4,
                        communication_buffer_bytes=4, dtype='float32',
                        group_ranks=ranks, participant_ranks=ranks, physical_group_ranks=physical,
                        grad_stats_group_policy=group_policy, logical_rank=ranks[0],
                        physical_rank=physical[0], cuda_stream='backend_assigned',
                        backend_callable='megatron.core.optimizer.clip_grads.get_grad_norm_fp32',
                        finalize_phase='optimizer_after_local_gradient_norm'))
                dag.add_task(task)
                dag.edges.extend((dependency, key) for dependency in dependencies)
            for rank in ranks:
                predecessor = key if len(ranks) > 1 else norm_tasks[rank]
                normalized = local(f'ONORM_POST:{rank}', rank, TaskType.GRADIENT_NORM_FINALIZE,
                    [predecessor], 'optimizer_local_grad_norm_finalize',
                    'local_optimizer_gradient_norm_finalize',
                    'megatron.core.optimizer.clip_grads.get_grad_norm_fp32')
                dag.tasks[normalized].metadata['component_phase'] = 'post_collective'
                if clip_grad == 0:
                    dag.add_dependency(normalized, optimizers[rank].task_id)
                    continue
                clipped = local(f'OCLIP:{rank}', rank, TaskType.GRADIENT_CLIP,
                    [normalized], 'optimizer_local_grad_clip', 'local_optimizer_gradient_clip',
                    'megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32')
                dag.add_dependency(clipped, optimizers[rank].task_id)
        return dag

    def compile(self, dag):
        finalizers = {}
        for rank in range(self.config.world_size):
            key = f'FINAL:{rank}'
            task = Task(key, rank, TaskType.CUDA_SYNC,
                dependencies=[f'DP:{rank}'], metadata=dict(
                    operation='finalize_model_grads_complete', logical_rank=rank,
                    physical_rank=self.groups['rank_mapping'][rank],
                    cuda_stream='backend_assigned',
                    backend_callable='megatron.core.distributed.finalize_model_grads.finalize_model_grads'))
            dag.add_task(task)
            dag.edges.append((f'DP:{rank}', key))
            finalizers[rank] = key

        def collective(key, ranks, payload, dtype, operation, callable_name, metadata=None):
            if len(ranks) <= 1:
                return None
            physical = [self.groups['rank_mapping'][rank] for rank in ranks]
            dependencies = [f'DP:{rank}' for rank in ranks]
            task = Task(key, ranks[0], TaskType.NCCL_GENERIC,
                dependencies=dependencies, resource_requirements=tuple(
                    f'gpu:{participant}:communication' for participant in physical),
                metadata=dict(operation=operation, collective='all_reduce',
                    group_ranks=list(ranks), physical_group_ranks=physical,
                    participant_ranks=list(ranks), payload_bytes=payload,
                    communication_buffer_bytes=payload, dtype=dtype,
                    logical_rank=ranks[0], physical_rank=physical[0],
                    cuda_stream='backend_assigned', backend_callable=callable_name,
                    finalize_phase='after_finish_grad_sync_before_optimizer',
                    **(metadata or {})))
            dag.add_task(task)
            dag.edges.extend((dependency, key) for dependency in dependencies)
            for rank in ranks:
                dag.add_dependency(key, finalizers[rank])
            return key

        dtype = self._gradient_dtype()
        if self.config.use_sequence_parallel or self.semantics.get('qk_layernorm', False):
            per_layer = 0
            if self.config.use_sequence_parallel:
                per_layer = self.semantics.get('non_tensor_parallel_grad_elements_per_layer')
                if per_layer is None and self.model.module_kinds is None:
                    per_layer = 2 * self.model.hidden_size
                if type(per_layer) is not int or per_layer <= 0:
                    raise ValueError('Sequence Parallel梯度同步需要逐层元素数')
            if self.semantics.get('qk_layernorm', False):
                head_dimension = self.model.head_dimension or self.model.hidden_size // self.model.num_attention_heads
                qk_elements = self.semantics.get('qk_layernorm_grad_elements_per_layer', 2 * head_dimension)
                if type(qk_elements) is not int or qk_elements <= 0:
                    raise ValueError('QK LayerNorm梯度同步需要逐层元素数')
                per_layer += qk_elements
            if type(per_layer) is not int or per_layer <= 0:
                raise ValueError('TP非分片梯度同步需要逐层元素数')
            lanes = self.config.world_size // self.config.pp
            for index, ranks in enumerate(self.groups['logical']['tp']):
                rank = ranks[0]
                layers = {layer for task in dag.tasks.values()
                    if task.rank == rank and task.task_type == TaskType.BACKWARD_COMPUTE
                    for layer in task.metadata['layer_ids']}
                elements = len(layers) * per_layer
                if self.config.use_sequence_parallel and rank // lanes == self.config.pp - 1:
                    elements += self.semantics.get(
                        'final_layernorm_grad_elements', self.model.hidden_size)
                collective(f'FG:TP:{index}', ranks, elements * GRADIENT_BYTES[dtype], dtype,
                    'non_tensor_parallel_gradient_all_reduce',
                    'megatron.core.distributed.finalize_model_grads._allreduce_non_tensor_model_parallel_grads',
                    dict(reduce_op='sum', semantic_branch='sequence_parallel_or_qk_layernorm'))

        share = self.semantics.get('share_embeddings_and_output_weights', False)
        mtp_layers = self.semantics.get('mtp_num_layers', 0) or 0
        if (share or mtp_layers > 0) and self.semantics.get('embedding_requires_grad', True):
            payload, embedding_dtype, layout = self._embedding_payload()
            groups = [ranks for ranks in self.groups['logical']['embedding'] if len(ranks) > 1]
            concurrency = f'word_embedding_gradient_all_reduce:{len(groups)}'
            for index, ranks in enumerate(groups):
                collective(f'FG:WORD:{index}', ranks, payload, embedding_dtype,
                    'word_embedding_gradient_all_reduce',
                    'megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                    dict(process_group='embedding', mtp_num_layers=mtp_layers,
                        shared_embeddings_and_output_weights=bool(share),
                        selected_model_chunk='first_on_pp_first_last_on_pp_last_or_mtp',
                        gradient_layout=layout, concurrent_collective_family=concurrency))

        position_elements = self.semantics.get('position_embedding_gradient_elements', 0)
        if position_elements:
            payload = position_elements // self.config.tp * GRADIENT_BYTES[dtype]
            for index, ranks in enumerate(self.groups['logical']['position_embedding']):
                collective(f'FG:POS:{index}', ranks, payload, dtype,
                    'position_embedding_gradient_all_reduce',
                    'megatron.core.distributed.finalize_model_grads._allreduce_position_embedding_grads',
                    dict(process_group='position_embedding'))

        for task in list(dag.tasks.values()):
            if task.task_type != TaskType.OPTIMIZER_STEP:
                continue
            original = list(task.dependencies)
            replacements = [finalizers[int(dependency.removeprefix('DP:'))]
                if dependency.startswith('DP:') else dependency for dependency in original]
            dag.edges = [(source, destination) for source, destination in dag.edges
                         if destination != task.task_id or source not in original]
            task.dependencies = []
            for dependency in replacements:
                dag.add_dependency(dependency, task.task_id)
        return self._compile_optimizer_preparation(dag)
