import importlib.util
import faulthandler
from functools import wraps
import hashlib
import json
import math
import sys
import os
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
sys.path.append('/workspace/paper-repros-20261003')
REQUEST = json.loads((ROOT := Path(__file__).resolve().parent.parent).joinpath('request.json').read_text())
if 'fp16_bounded_adam_sha256' in REQUEST:
    from transformer_engine.pytorch import optimizers

    adam_source = Path(__file__).with_name('bounded_fused_adam.py')
    if hashlib.sha256(adam_source.read_bytes()).hexdigest() != REQUEST['fp16_bounded_adam_sha256']:
        raise ValueError('FusedAdam源文件与执行记录的SHA-256不一致')
    adam_name = 'transformer_engine.pytorch.optimizers.fused_adam'
    adam_spec = importlib.util.spec_from_file_location(adam_name, adam_source)
    adam_module = importlib.util.module_from_spec(adam_spec)
    sys.modules[adam_name] = adam_module
    adam_spec.loader.exec_module(adam_module)
    if not hasattr(adam_module.FusedAdam, '_workspace_groups'):
        raise ValueError('FusedAdam源文件缺少分批更新实现')
    optimizers.fused_adam = adam_module
    optimizers.FusedAdam = adam_module.FusedAdam
source = Path(__file__).with_name('native_hybrid_train.py')
spec = importlib.util.spec_from_file_location('native_hybrid_training', source)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
original_pretrain = native.pretrain
original_evidence_start = native.Evidence.on_train_start
original_step_start = native.Evidence.on_train_step_start
original_step_end = native.Evidence.on_train_step_end
original_evidence_end = native.Evidence.on_train_end
from checkpoint_io_guard import install_checkpoint_io_guard

install_checkpoint_io_guard(ROOT, REQUEST)


def optimizer_parameter_groups_with_cpu_copy(optimizer, offload_fraction):
    # 保留Megatron的参数分组和FP32主参数映射。
    gpu_total = sum(parameter.numel() for group in optimizer.param_groups
        for parameter in group['params'] if parameter.is_cuda)
    threshold = gpu_total * offload_fraction
    batched_copies = None
    mode = REQUEST['optimizer_cpu_copy_mode']
    if mode == 'batched':
        mode = 'batched_pinned' if optimizer.pin_cpu_params else 'batched_pageable'
    if mode in ('batched_pageable', 'batched_pinned'):
        batched_copies = {}
        batch = []
        batch_bytes = 0
        selected = 0
        transfers = 0

        def transfer():
            if not batch:
                return
            flattened = [parameter.detach().reshape(-1) for parameter in batch]
            flat = flattened[0] if len(flattened) == 1 else native.torch.cat(flattened)
            target_dtype = native.torch.float32 if optimizer.param_update_in_fp32 else flat.dtype
            if mode == 'batched_pinned':
                cpu_flat = native.torch.empty(flat.shape, device='cpu', dtype=target_dtype, pin_memory=True)
                cpu_flat.copy_(flat)
            else:
                cpu_flat = flat.to(device='cpu', copy=True)
                if cpu_flat.dtype != target_dtype:
                    cpu_flat = cpu_flat.to(dtype=target_dtype)
            offset = 0
            for parameter in batch:
                length = parameter.numel()
                batched_copies[parameter] = cpu_flat.narrow(0, offset, length).view(parameter.shape)
                offset += length

        for group in optimizer.param_groups:
            for parameter in group['params']:
                if not parameter.is_cuda or selected >= threshold:
                    continue
                size = parameter.numel() * parameter.element_size()
                if batch and (parameter.dtype != batch[0].dtype or batch_bytes + size > 64 * 2**20):
                    transfer()
                    transfers += 1
                    batch.clear()
                    batch_bytes = 0
                batch.append(parameter)
                batch_bytes += size
                selected += parameter.numel()
        if batch:
            transfer()
            transfers += 1
        print(f'OPTIMIZER_OFFLOAD_BATCHED_COPY tensors={len(batched_copies)} transfers={transfers}', flush=True)
    offloaded = 0
    cpu_groups, gpu_groups = [], []
    gpu_to_cpu, cpu_to_gpu, to_fp32 = {}, {}, {}
    for group in optimizer.param_groups:
        cpu, gpu = [], []
        for original in group['params']:
            parameter = original
            copy_to_cpu = parameter.is_cuda and offloaded < threshold
            convert_to_fp32 = optimizer.param_update_in_fp32 and parameter.dtype != native.torch.float32
            if copy_to_cpu:
                if batched_copies is not None:
                    parameter = batched_copies[original]
                else:
                    dtype = native.torch.float32 if convert_to_fp32 else parameter.dtype
                    parameter = parameter.detach().to(device='cpu', dtype=dtype, copy=True)
                offloaded += parameter.numel()
                gpu_to_cpu[original], cpu_to_gpu[parameter] = parameter, original
            elif convert_to_fp32:
                parameter = parameter.detach().clone().float()
            if convert_to_fp32:
                to_fp32[original] = parameter
            (gpu if parameter.is_cuda else cpu).append(parameter)
        if cpu:
            cpu_groups.append(dict(group, params=cpu))
        if gpu:
            gpu_groups.append(dict(group, params=gpu))
    return cpu_groups, gpu_groups, gpu_to_cpu, cpu_to_gpu, to_fp32


if REQUEST.get('paramorph_numerical_reference', False):
    native.torch.use_deterministic_algorithms(True)
if 'rank_mapping' in REQUEST:
    world = int(os.environ['WORLD_SIZE'])
    mapping = REQUEST['rank_mapping']
    if sorted(mapping) != list(range(world)):
        raise ValueError('实际训练Rank Mapping没有完整覆盖设备')
    physical_rank = int(os.environ['RANK'])
    os.environ['RANK'] = str(mapping.index(physical_rank))


def profile_step_start(self, context):
    original_step_start(self, context)
    if REQUEST.get('profile', False) and len(self.rows) == self.args.warmup:
        native.torch.cuda.nvtx.range_push('FrontierMeasured')
        native.torch.cuda.cudart().cudaProfilerStart()


def profile_step_end(self, context):
    original_step_end(self, context)
    if REQUEST.get('profile', False) and len(self.rows) == self.args.warmup + self.args.measured:
        native.torch.cuda.synchronize()
        native.torch.cuda.cudart().cudaProfilerStop()
        native.torch.cuda.nvtx.range_pop()


def record_ownership(self, context):
    faulthandler.cancel_dump_traceback_later()
    original_evidence_start(self, context)
    manifest_path = self.args.output / f'rank{self.rank}.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    actual_cp = context.state.cfg.model.context_parallel_size
    if 'cp' in manifest and manifest['cp'] != actual_cp:
        raise ValueError('运行时CP与训练配置记录不一致')
    manifest['cp'] = actual_cp
    manifest['overlap_p2p_comm'] = context.state.cfg.model.overlap_p2p_comm
    if REQUEST.get('record_native_schedule'):
        from megatron.core.pipeline_parallel.schedules import get_schedule_table
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        settings = context.state.cfg.model
        vpp = settings.virtual_pipeline_model_parallel_size or 1
        group = settings.microbatch_group_size_per_vp_stage if vpp > 1 else None
        total = get_num_microbatches()
        table = (get_schedule_table(total, vpp, group) if vpp > 1
            else [(index, 0) for index in range(total)])
        manifest['native_pipeline_schedule'] = dict(pp=settings.pipeline_model_parallel_size,
            vpp=vpp, num_microbatches=total, group=group, overlap_p2p=settings.overlap_p2p_comm,
            batch_p2p_comm=settings.batch_p2p_comm,
            overlap_p2p_comm_warmup_flush=settings.overlap_p2p_comm_warmup_flush,
            schedule_table=[list(row) for row in table])
    manifest['gradient_bucket_size'] = context.state.cfg.ddp.bucket_size
    manifest['overlap_grad_reduce'] = context.state.cfg.ddp.overlap_grad_reduce
    manifest['overlap_param_gather'] = context.state.cfg.ddp.overlap_param_gather
    manifest['recompute_num_layers'] = context.state.cfg.model.recompute_num_layers
    if REQUEST.get('recompute') == 'selective':
        selected = list(REQUEST['memory_policy']['recompute_modules'])
        attributes = dict(core_attn='checkpoint_core_attention', mlp='recompute_mlp',
            moe='moe_layer_recompute', moe_act='activation_recompute',
            shared_experts='shared_experts_recompute')
        enabled, examples = {}, {}
        source_files = {}
        for target in selected:
            attribute = attributes[target]
            active = [(f'{index}:{name}', module) for index, model in enumerate(self.models)
                for name, module in model.named_modules() if getattr(module, attribute, False)]
            enabled[target] = len(active)
            examples[target] = [dict(module=name, implementation=type(module).__name__)
                for name, module in active[:8]]
            for name, module in active:
                path = Path(sys.modules[type(module).__module__].__file__)
                if str(path) not in source_files:
                    source_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if context.state.cfg.model.recompute_granularity != 'selective':
            raise ValueError('选择性重计算没有进入原生训练配置')
        manifest['native_selective_recomputation'] = dict(requested_modules=selected,
            enabled_modules=enabled, examples=examples, source_sha256=source_files,
            evidence='actual_native_module_checkpoint_flags')
    manifest['cross_entropy_loss_fusion'] = context.state.cfg.model.cross_entropy_loss_fusion
    manifest['etp'] = context.state.cfg.model.expert_tensor_parallel_size
    manifest['process_groups'] = {name: native.dist.get_process_group_ranks(group)
        for name, group in vars(self.models[0].pg_collection).items()
        if isinstance(group, native.dist.ProcessGroup)}
    manifest['dp'] = native.dist.get_world_size() // (manifest['tp'] * manifest['cp'] * manifest['pp'])
    manifest['expert_dp'] = native.dist.get_world_size() // (manifest['etp'] * manifest['ep'] * manifest['pp'])
    manifest['physical_rank'] = REQUEST.get('rank_mapping', list(range(native.dist.get_world_size())))[self.rank]
    manifest['gpu_uuid'] = str(native.torch.cuda.get_device_properties(native.torch.cuda.current_device()).uuid)
    manifest['optimizer_sharding'] = ('distributed' if context.state.cfg.optimizer.use_distributed_optimizer
        else 'fsdp' if context.state.cfg.ddp.use_megatron_fsdp else 'replicated')
    manifest['execution_backend'] = 'megatron_fsdp' if context.state.cfg.ddp.use_megatron_fsdp else 'megatron_ddp'
    manifest['megatron_fsdp_version'] = (context.state.cfg.ddp.megatron_fsdp_version
        if context.state.cfg.ddp.use_megatron_fsdp else None)
    manifest['optimizer_offload_fraction'] = context.state.cfg.optimizer.optimizer_offload_fraction
    manifest['optimizer_pin_cpu_params'] = context.state.cfg.optimizer.pin_cpu_params
    if 'optimizer_cpu_copy_mode' in REQUEST:
        manifest['optimizer_cpu_copy_mode'] = REQUEST['optimizer_cpu_copy_mode']
    manifest['checkpoint_load_optimizer'] = context.state.cfg.checkpoint.load_optim
    manifest['checkpoint_finetune'] = context.state.cfg.checkpoint.finetune
    manifest['checkpoint_fully_parallel_load'] = context.state.cfg.checkpoint.fully_parallel_load
    parameters = {id(parameter): parameter for chunk in self.models for parameter in chunk.parameters()}
    manifest['rank_model_parameter_numel'] = sum(parameter.numel() for parameter in parameters.values())
    manifest['rank_model_parameter_bytes'] = sum(parameter.numel() * parameter.element_size()
        for parameter in parameters.values())
    manifest['optimizer_transfer_overlap'] = context.state.cfg.optimizer.overlap_cpu_optimizer_d2h_h2d
    manifest['megatron_fsdp_adapter_sha256'] = REQUEST.get('megatron_fsdp_adapter_sha256')
    manifest['fsdp_sharding'] = {
        'dense': context.state.cfg.ddp.data_parallel_sharding_strategy,
        'expert': context.state.cfg.ddp.expert_data_parallel_sharding_strategy,
        'outer': context.state.cfg.ddp.outer_dp_sharding_strategy,
        'expert_outer': context.state.cfg.ddp.expert_outer_dp_sharding_strategy,
        'double_buffer': context.state.cfg.ddp.fsdp_double_buffer,
        'communication_unit_size': context.state.cfg.ddp.suggested_communication_unit_size}
    manifest['precision_aware_optimizer'] = context.state.cfg.optimizer.use_precision_aware_optimizer
    from transformer_engine.pytorch.optimizers import fused_adam
    from megatron.core.optimizer import clip_grads
    manifest['optimizer_runtime_sources'] = {module.__name__: dict(
        path=module.__file__, sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        for module in (fused_adam, clip_grads)}
    manifest['fp16_optimizer_workspace'] = dict(
        bounded_update_available=hasattr(fused_adam.FusedAdam, '_workspace_groups'),
        batch_elements=16 * 1024 * 1024
            if hasattr(fused_adam.FusedAdam, '_workspace_groups') else None)
    if context.state.cfg.ddp.use_megatron_fsdp:
        buffer_source = Path(REQUEST['megatron_fsdp_adapter_path']).parent / 'src/megatron_fsdp/param_and_grad_buffer.py'
        manifest['fsdp_parameter_buffer_source'] = dict(path=str(buffer_source),
            sha256=hashlib.sha256(buffer_source.read_bytes()).hexdigest())
    from megatron.bridge.training.utils import flop_utils
    manifest['model_flops_per_update'] = flop_utils.num_floating_point_operations(
        context.state.cfg, batch_size=REQUEST['global_batch'],
        seqlen_sum=REQUEST['global_batch'] * REQUEST['sequence'],
        seqlen_squared_sum=REQUEST['global_batch'] * REQUEST['sequence'] ** 2)
    manifest['model_flops_source'] = dict(path=flop_utils.__file__,
        sha256=hashlib.sha256(Path(flop_utils.__file__).read_bytes()).hexdigest(),
        global_batch=REQUEST['global_batch'], sequence=REQUEST['sequence'])
    manifest['optimizer_hyperparameters'] = {field: getattr(context.state.cfg.optimizer, field)
        for field in ('lr', 'min_lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps', 'clip_grad')}
    manifest['scheduler_configuration'] = {field: getattr(context.state.cfg.scheduler, field)
        for field in ('lr_decay_style', 'lr_decay_iters', 'lr_warmup_iters')}
    if 'training_data_contract' in REQUEST:
        manifest['training_data_contract'] = dict(REQUEST['training_data_contract'],
            workers=context.state.cfg.dataset.num_workers,
            persistent_workers=context.state.cfg.dataset.persistent_workers)
    manifest['clip_grad'] = context.state.cfg.optimizer.clip_grad
    if 'clip_grad' in REQUEST and manifest['clip_grad'] != REQUEST['clip_grad']:
        raise ValueError('实际Optimizer clip_grad与显式训练请求不一致')
    manifest['optimizer_state_dtypes'] = {
        'main_grads': str(context.state.cfg.optimizer.main_grads_dtype),
        'main_params': str(context.state.cfg.optimizer.main_params_dtype),
        'exp_avg': str(context.state.cfg.optimizer.exp_avg_dtype),
        'exp_avg_sq': str(context.state.cfg.optimizer.exp_avg_sq_dtype)}
    manifest['finalize_model_grads_semantics'] = {
        'share_embeddings_and_output_weights':
            context.state.cfg.model.share_embeddings_and_output_weights,
        'mtp_num_layers': context.state.cfg.model.mtp_num_layers or 0,
        'main_grads_dtype': str(context.state.cfg.optimizer.main_grads_dtype),
        'embedding_group': manifest['process_groups'].get('embd'),
        'position_embedding_group': manifest['process_groups'].get('pos_embd'),
        'source': 'actual_megatron_runtime_configuration_and_process_groups'}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if REQUEST.get('profile', False):
        import os
        import psutil
        process = psutil.Process()
        names = ('CUDA_INJECTION64_PATH', 'NVTX_INJECTION64_PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH')
        metadata = {'rank': self.rank, 'pid': process.pid,
            'injection_environment': {name: os.environ.get(name) for name in names},
            'loaded_profiler_libraries': sorted({mapping.path for mapping in process.memory_maps()
                if 'ToolsInjection' in mapping.path or 'cupti' in mapping.path.lower()})}
        (self.args.output / f'rank{self.rank}.profile-injection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if REQUEST.get('defer_embedding_wgrad', False) or REQUEST.get('output_wgrad_fusion', False):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        self.frontier_deferred_activation_peak = 0
        for model in self.models:
            output = model.output_layer if model.post_process else None
            for module in model.modules():
                if isinstance(module, ColumnParallelLinear):
                    module.config = native.copy.copy(module.config)
                    module.config.defer_embedding_wgrad_compute = module is output and REQUEST.get('defer_embedding_wgrad', False)
                    module.config.wgrad_deferral_limit = REQUEST['wgrad_deferral_limit']
            model.config.defer_embedding_wgrad_compute = REQUEST.get('defer_embedding_wgrad', False)
            model.config.wgrad_deferral_limit = REQUEST['wgrad_deferral_limit']
            if output is not None:
                if not isinstance(output, ColumnParallelLinear):
                    raise ValueError('Hybrid deferred output WGrad requires the actual native ColumnParallelLinear')
                if REQUEST.get('output_wgrad_fusion', False):
                    import fused_weight_gradient_mlp_cuda
                    if not hasattr(output.weight, 'main_grad') or not hasattr(output.weight, 'grad_added_to_main_grad'):
                        raise ValueError('Native fused output WGrad requires its distributed gradient buffer and accumulation flag')
                    output.gradient_accumulation_fusion = True
                    output.config.gradient_accumulation_fusion = True
                    model.config.gradient_accumulation_fusion = True
                if REQUEST.get('defer_embedding_wgrad', False) and not output.gradient_accumulation_fusion:
                    raise ValueError('Native deferred output WGrad requires gradient accumulation fusion')
                model.embedding_activation_buffer = []
                model.grad_output_buffer = []
                output.embedding_activation_buffer = model.embedding_activation_buffer
                output.grad_output_buffer = model.grad_output_buffer

                def record_deferred(module, inputs, result):
                    self.frontier_deferred_activation_peak = max(self.frontier_deferred_activation_peak,
                        len(module.embedding_activation_buffer))

                output.register_forward_hook(record_deferred)
    if self.args.vpp > 1:
        groups = tuple(dict.fromkeys(model.pg_collection.pp for model in self.models))
        for group in groups:
            native.dist.barrier(group=group, device_ids=[native.torch.cuda.current_device()])
    rows = []
    for index, model in enumerate(self.models):
        pp_rank = model.pg_collection.pp.rank()
        identifiers = [layer.layer_number - 1 for layer in model.decoder.layers]
        layout = context.state.cfg.model.pipeline_model_parallel_layout
        if layout is None:
            counts = REQUEST['decoder_chunks']
            if not (REQUEST.get('native_uniform_pipeline_layout', False) and self.args.vpp == 1
                    and index == 0 and len(set(counts)) == 1):
                raise ValueError('训练缺少可验证的流水线层归属')
            first = sum(counts[:pp_rank])
            expected = list(range(first, first + counts[pp_rank]))
        else:
            expected = layout.get_layer_id_list(vp_stage=index, pp_rank=pp_rank)
        complete_ids = [layer.layer_number - 1 for layer in model.decoder.layers
            if getattr(layer, 'frontier_region_kind', None) not in ('FALCON_MLP_FRAGMENT', 'HYBRID_ROUTER_FRAGMENT')]
        if complete_ids != expected:
            raise ValueError('The actual model layers differ from the complete checkpoint layer ownership')
        rows.append({'vp_stage': index, 'pp_rank': pp_rank, 'layer_ids': identifiers,
            'regions': [{'layer_id': layer.layer_number - 1,
                'kind': getattr(layer, 'frontier_region_kind', 'FULL_LAYER'),
                'parameter_keys': tuple(dict(layer.named_parameters()))} for layer in model.decoder.layers],
            'pre_process': model.pre_process, 'post_process': model.post_process})
        if REQUEST.get('profile', False):
            for layer in model.decoder.layers:
                label = f'Frontier/VP{index}/layer:{layer.layer_number - 1}/{type(layer).__name__}'

                def before(module, inputs, region=label):
                    native.torch.cuda.nvtx.range_push(region)

                def after(module, inputs, output):
                    native.torch.cuda.nvtx.range_pop()

                layer.register_forward_pre_hook(before)
                layer.register_forward_hook(after)
    path = self.args.output / f'rank{self.rank}.ownership.json'
    path.write_text(json.dumps({'rank': self.rank, 'model': str(self.args.model), 'chunks': rows}, indent=2) + '\n')
    if REQUEST.get('paramorph_parameter_ledger', False) and not REQUEST.get('paramorph', False):
        from paramorph_runtime import RuntimeEvidence
        evidence = RuntimeEvidence(native, REQUEST)
        evidence.rank = self.rank
        evidence.output = self.args.output
        evidence.write_parameter_ledger(context.model)
        evidence.install_bucket_calibration(context.model)


def record_end(self, context):
    original_evidence_end(self, context)
    if REQUEST.get('record_training_contract_inputs', False):
        from native_frontier_checkpoint import NativeMegatronTrainingCheckpoint
        checkpoint = NativeMegatronTrainingCheckpoint(self.models, context.optimizer)
        ownership = checkpoint.runtime_ownership()
        ownership['scope'] = 'MODEL'
        ownership['source'] = 'Actual native sharded parameters, gradient buffers and optimizer ownership after training'
        (self.args.output / f'rank{self.rank}.training-contract-inputs.json').write_text(
            json.dumps(ownership, indent=2) + '\n')
    if REQUEST.get('hybrid_router_cut', {}).get('boundary_dtypes', False):
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
        path = self.args.output / f'rank{self.rank}.measurements.json'
        record = json.loads(path.read_text())
        record['pipeline_transport_totals'] = P2PCommunicator._frontier_transport_totals
        record['pipeline_transport_scope'] = 'Activation and gradient payloads across all warmup and measured updates; shape headers excluded'
        if not record['pipeline_transport_totals']:
            raise ValueError('Boundary-specific transport has no actual pipeline communication evidence')
        path.write_text(json.dumps(record, indent=2) + '\n')
    if REQUEST.get('defer_embedding_wgrad', False) or REQUEST.get('output_wgrad_fusion', False):
        path = self.args.output / f'rank{self.rank}.measurements.json'
        record = json.loads(path.read_text())
        record['deferred_activation_peak'] = self.frontier_deferred_activation_peak
        record['output_wgrad_fusion'] = [model.output_layer.gradient_accumulation_fusion
            for model in self.models if model.post_process]
        if (REQUEST.get('defer_embedding_wgrad', False) and any(model.post_process for model in self.models)
                and self.frontier_deferred_activation_peak == 0):
            raise ValueError('The requested native output WGrad deferral did not execute')
        path.write_text(json.dumps(record, indent=2) + '\n')


def select_falcon_layers(self, layer_type_list):
    from frontier_hybrid_provider import FALCON_VP_STAGE
    layout = self.config.pipeline_model_parallel_layout
    self.vp_stage = FALCON_VP_STAGE.get()
    identifiers = layout.get_layer_id_list(vp_stage=self.vp_stage, pp_rank=self.pp_group.rank())
    if not identifiers or identifiers != list(range(identifiers[0], identifiers[-1] + 1)):
        raise ValueError('Falcon physical pipeline ownership must be nonempty and contiguous')
    if len(layer_type_list) != self.config.num_layers:
        raise ValueError('Falcon checkpoint layer count differs from the executable pattern')
    return identifiers[0], [layer_type_list[index] for index in identifiers]


def pretrain(config, forward_step_func, callbacks):
    from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
    from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider

    cfg = config
    attention = REQUEST.get('attention_backend_capabilities')
    if attention is not None and cfg.model.context_parallel_size > 1 and attention['mla_context_parallel_supported'] is False:
        raise ValueError('当前Attention后端不支持所请求的MLA Context Parallel配置：' + json.dumps(attention))
    if REQUEST.get('precision') == 'float16':
        cfg.mixed_precision = native.get_mixed_precision_config('fp16_mixed')
        cfg.mixed_precision.loss_scale = 128.0
        cfg.model.params_dtype = native.torch.float16
        cfg.model.pipeline_dtype = native.torch.float16
        cfg.optimizer.fp16 = True
        cfg.optimizer.bf16 = False
        cfg.optimizer.loss_scale = 128.0
    if 'optimizer_hyperparameters' in REQUEST:
        values = REQUEST['optimizer_hyperparameters']
        expected = {'lr', 'min_lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps', 'clip_grad'}
        if set(values) != expected or any(isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values.values()):
            raise ValueError('Optimizer训练设置必须完整提供有限数值')
        for field, value in values.items():
            setattr(cfg.optimizer, field, value)
        schedule = REQUEST.get('optimizer_schedule', dict(lr_decay_style='cosine',
            lr_decay_iters=REQUEST['warmup'] + REQUEST['measured'], lr_warmup_iters=0))
        if (set(schedule) != {'lr_decay_style', 'lr_decay_iters', 'lr_warmup_iters'}
                or schedule['lr_decay_style'] not in ('constant', 'cosine')
                or type(schedule['lr_decay_iters']) is not int or schedule['lr_decay_iters'] < 1
                or type(schedule['lr_warmup_iters']) is not int
                or not 0 <= schedule['lr_warmup_iters'] <= schedule['lr_decay_iters']):
            raise ValueError('学习率调度器配置不完整或取值无效')
        for field, value in schedule.items():
            setattr(cfg.scheduler, field, value)
        cfg.scheduler.start_weight_decay = values['weight_decay']
        cfg.scheduler.end_weight_decay = values['weight_decay']
        cfg.scheduler.weight_decay_incr_style = 'constant'
    cfg.model.expert_tensor_parallel_size = REQUEST.get('etp', REQUEST['tp'])
    execution_backend = REQUEST.get('execution_backend', 'megatron_ddp')
    use_fsdp = execution_backend == 'megatron_fsdp'
    fsdp_version = REQUEST.get('megatron_fsdp_version', 1)
    if use_fsdp and fsdp_version != 1:
        raise ValueError('ParaMorph固定使用Megatron-FSDP v1')
    if use_fsdp:
        adapter_path = Path(REQUEST['megatron_fsdp_adapter_path'])
        adapter_digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        if adapter_digest != REQUEST['megatron_fsdp_adapter_sha256']:
            raise ValueError('Megatron-FSDP v1适配器与已验证的PP/VPP实现不同')
    if use_fsdp and REQUEST.get('fsdp_unit', 'transformer_layer') != 'transformer_layer':
        raise ValueError('当前Megatron-FSDP配置固定使用TransformerLayer FSDP Unit')
    if use_fsdp and (REQUEST.get('dp_outer_size', 1) != 1
            or REQUEST.get('outer_sharding_strategy', 'no_shard') != 'no_shard'
            or REQUEST.get('expert_outer_sharding_strategy', 'no_shard') != 'no_shard'):
        raise ValueError('当前Megatron-FSDP配置已经关闭Outer DP分片')
    cfg.ddp.use_megatron_fsdp = use_fsdp
    cfg.ddp.megatron_fsdp_version = fsdp_version if use_fsdp else 1
    if use_fsdp:
        cfg.model.deallocate_pipeline_outputs = False
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.data_parallel_sharding_strategy = REQUEST.get('dense_sharding_strategy',
        REQUEST.get('fsdp_sharding_strategy', 'no_shard'))
    cfg.ddp.expert_data_parallel_sharding_strategy = REQUEST.get('expert_sharding_strategy',
        REQUEST.get('fsdp_sharding_strategy', 'no_shard'))
    cfg.ddp.outer_dp_sharding_strategy = REQUEST.get('outer_sharding_strategy', 'no_shard')
    cfg.ddp.expert_outer_dp_sharding_strategy = REQUEST.get('expert_outer_sharding_strategy', 'no_shard')
    cfg.ddp.fsdp_all_gather_in_start_param_sync = REQUEST.get('fsdp_prefetch', False)
    cfg.ddp.fsdp_double_buffer = REQUEST.get('fsdp_double_buffer', False)
    cfg.ddp.suggested_communication_unit_size = (REQUEST.get('fsdp_bucket_elements', 40000000)
        * REQUEST.get('fsdp_prefetch_distance', 1) if REQUEST.get('fsdp_prefetch', False) else 0)
    use_distributed_optimizer = REQUEST.get('optimizer_sharding', 'distributed') == 'distributed'
    cfg.optimizer.use_distributed_optimizer = use_distributed_optimizer
    cfg.ddp.use_distributed_optimizer = use_distributed_optimizer
    cfg.checkpoint.ckpt_format = REQUEST.get('checkpoint_format',
        'fsdp_dtensor' if use_fsdp else 'torch_dist')
    if REQUEST.get('optimizer_checkpoint_mode', 'fresh') in ('fresh', 'disabled'):
        cfg.checkpoint.load_optim = False
    cfg.checkpoint.fully_parallel_load = REQUEST.get('checkpoint_fully_parallel_load', True)
    cfg.optimizer.use_precision_aware_optimizer = REQUEST.get('precision_aware_optimizer', False)
    if use_fsdp:
        cfg.ddp.megatron_fsdp_use_decoupled_grad = cfg.optimizer.use_precision_aware_optimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        def collect_fsdp_gradients(optimizer):
            field = ('decoupled_grad' if optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                else 'grad')
            gradients = [getattr(parameter, field) for group in optimizer.optimizer.param_groups
                for parameter in group['params'] if getattr(parameter, field) is not None]
            if not gradients and not optimizer.is_stub_optimizer:
                raise RuntimeError('FSDP优化器没有取得任何真实梯度')
            return [gradient.data for gradient in gradients]
        DistributedOptimizer._collect_main_grad_data_for_unscaling = collect_fsdp_gradients
    if 'clip_grad' in REQUEST:
        clip_grad = REQUEST['clip_grad']
        if (isinstance(clip_grad, bool) or not isinstance(clip_grad, (int, float))
                or not math.isfinite(clip_grad) or clip_grad < 0):
            raise ValueError('Optimizer clip_grad需要有限非负数值')
        cfg.optimizer.clip_grad = float(clip_grad)
    if cfg.optimizer.use_precision_aware_optimizer:
        cfg.optimizer.main_grads_dtype = native.torch.bfloat16
        cfg.optimizer.main_params_dtype = native.torch.float16
        cfg.optimizer.exp_avg_dtype = native.torch.bfloat16
        cfg.optimizer.exp_avg_sq_dtype = native.torch.bfloat16
    for field in ('main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
        if field in REQUEST:
            dtype = REQUEST[field].removeprefix('torch.')
            if dtype not in ('bfloat16', 'float16', 'float32'):
                raise ValueError(f'Optimizer状态精度没有后端实现：{field}={dtype}')
            setattr(cfg.optimizer, field, getattr(native.torch, dtype))
    if use_fsdp:
        for field in ('megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype'):
            if REQUEST.get(field) is not None:
                dtype = REQUEST[field].removeprefix('torch.')
                if dtype not in ('bfloat16', 'float16', 'float32'):
                    raise ValueError(f'FSDP状态精度没有后端实现：{field}={dtype}')
                setattr(cfg.ddp, field, getattr(native.torch, dtype))
    if 'bucket_elements' in REQUEST:
        cfg.ddp.bucket_size = REQUEST['bucket_elements']
    if 'overlap_p2p' in REQUEST:
        cfg.model.overlap_p2p_comm = REQUEST['overlap_p2p']
    if REQUEST.get('record_native_schedule'):
        cfg.model.batch_p2p_comm = REQUEST.get('batch_p2p_comm', not cfg.model.overlap_p2p_comm)
        cfg.model.overlap_p2p_comm_warmup_flush = REQUEST.get('overlap_p2p_comm_warmup_flush', False)
        if (cfg.model.overlap_p2p_comm and cfg.model.batch_p2p_comm
                or cfg.model.overlap_p2p_comm_warmup_flush and not cfg.model.overlap_p2p_comm):
            raise ValueError('请求的原生P2P调度开关互斥')
    if REQUEST.get('moonlight_tokenizer_contract', False):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from tiktoken.load import load_tiktoken_bpe
        model_root = Path(REQUEST['model'])
        model_vocab = json.loads((model_root / 'config.json').read_text())['vocab_size']
        base_vocab = len(load_tiktoken_bpe(str(model_root / 'tiktoken.model')))
        special_slots = model_vocab - base_vocab
        if special_slots < 2:
            raise ValueError('Moonlight词表缺少BOS/EOS位置')
        tokenizer_class = get_class_from_dynamic_module('tokenization_moonshot.TikTokenTokenizer', str(model_root))
        tokenizer_class.num_reserved_special_tokens = special_slots - 2
        print('MOONLIGHT_TOKENIZER_CONTRACT ' + json.dumps(dict(model_vocab=model_vocab,
            base_vocab=base_vocab, special_slots=special_slots)), flush=True)
    memory = REQUEST.get('memory_policy')
    if memory is not None:
        cfg.model.fine_grained_activation_offloading = memory['activation_offload_fraction'] > 0
        cfg.model.offload_modules = list(memory['activation_offload_modules'])
        cfg.model.activation_offload_fraction = memory['activation_offload_fraction']
        cfg.model.recompute_granularity = memory['recompute_granularity']
        if memory['recompute_granularity'] == 'selective':
            cfg.model.recompute_modules = list(memory['recompute_modules'])
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
    if REQUEST.get('cross_entropy_impl') is not None:
        cfg.model.cross_entropy_loss_fusion = True
        cfg.model.cross_entropy_fusion_impl = REQUEST['cross_entropy_impl']
    if 'cross_entropy_fusion' in REQUEST:
        cfg.model.cross_entropy_loss_fusion = REQUEST['cross_entropy_fusion']
    if REQUEST.get('recompute_num_layers') is not None and cfg.model.recompute_granularity == 'full':
        cfg.model.recompute_num_layers = REQUEST['recompute_num_layers']
    activation_cpu_offload_layers = int(REQUEST.get('activation_cpu_offload_layers', 0))
    if activation_cpu_offload_layers:
        if cfg.model.pipeline_model_parallel_size > 1:
            raise ValueError('Activation CPU Offload要求PP=1')
        if cfg.model.recompute_granularity is not None:
            raise ValueError('Activation CPU Offload不能与Recomputation共同使用')
        cfg.model.cpu_offloading = True
        cfg.model.cpu_offloading_num_layers = activation_cpu_offload_layers
    pp = cfg.model.pipeline_model_parallel_size
    vpp = cfg.model.virtual_pipeline_model_parallel_size or 1
    if vpp > 1 or REQUEST.get('native_uniform_pipeline_layout', False):
        cfg.model.microbatch_group_size_per_vp_stage = REQUEST.get('group', pp)
    pattern = None
    counts = None
    if isinstance(cfg.model, native.FalconH1ModelProvider):
        from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_block import FalconH1Stack

        if cfg.model.context_parallel_size != 1:
            raise ValueError('FalconH1 regional execution requires its native CP1 state interface')
        pattern = cfg.model.hybrid_override_pattern
        counts = REQUEST.get('decoder_chunks') or native.uniform_decoder_chunks(cfg.model.num_layers, pp, vpp)
        if len(counts) != pp * vpp or min(counts) < 1 or sum(counts) != cfg.model.num_layers:
            raise ValueError('Falcon region placement must preserve all original decoder layers')
        layout = [['decoder'] * count for count in counts]
        layout[0].insert(0, 'embedding')
        layout[-1].append('loss')
        cfg.model.pipeline_model_parallel_layout = PipelineParallelLayerLayout(layout, pp)
        cfg.model.pipeline_model_parallel_layout.validate_layer_layout(cfg.model.num_layers, None)
        cfg.model.num_layers_in_first_pipeline_stage = None
        cfg.model.num_layers_in_last_pipeline_stage = None
        FalconH1Stack._select_layers_for_pipeline_parallel = select_falcon_layers
        if REQUEST.get('falcon_cut') is not None:
            from frontier_falcon_cut import install_falcon_semantic_cut
            cut = REQUEST['falcon_cut']
            install_falcon_semantic_cut(FalconH1Stack, cut['logical_layer'], cut['source_pp'], cut['source_vp'])
        if vpp > 1:
            from frontier_hybrid_provider import install_falcon_provider
            install_falcon_provider(cfg.model)
    elif isinstance(cfg.model, HybridModelProvider):
        pattern = cfg.model.hybrid_layer_pattern
        parsed = parse_hybrid_pattern(pattern)
        if '|' in parsed.main_pattern:
            raise ValueError('The source checkpoint must expose its unpartitioned hybrid pattern')
        layers = len(parsed.main_pattern)
        if layers != cfg.model.num_layers or layers < pp * vpp:
            raise ValueError('Hybrid region count must match the actual provider and nonempty stage count')
        counts = REQUEST.get('decoder_chunks') or native.uniform_decoder_chunks(layers, pp, vpp)
        if len(counts) != pp * vpp or min(counts) < 1 or sum(counts) != layers:
            raise ValueError('Hybrid region placement must preserve every original native region')
        segments = []
        offset = 0
        for count in counts:
            segments.append(parsed.main_pattern[offset:offset + count])
            offset += count
        pattern = '|'.join(segments)
        if parsed.mtp_pattern is not None:
            pattern += ''.join('/' + parsed.mtp_pattern for _ in range(parsed.mtp_num_depths))
        cfg.model.hybrid_layer_pattern = pattern
        cfg.model.num_layers_in_first_pipeline_stage = None
        cfg.model.num_layers_in_last_pipeline_stage = None
        cfg.model.num_layers_per_virtual_pipeline_stage = None
        layout = [['decoder'] * count for count in counts]
        layout[0].insert(0, 'embedding')
        layout[-1].extend(['mtp'] * (cfg.model.mtp_num_layers or 0))
        layout[-1].append('loss')
        cfg.model.pipeline_model_parallel_layout = PipelineParallelLayerLayout(layout, pp)
        cfg.model.pipeline_model_parallel_layout.validate_layer_layout(layers, cfg.model.mtp_num_layers)
        if vpp > 1:
            from frontier_hybrid_provider import install_hybrid_provider
            install_hybrid_provider(cfg.model)
        if REQUEST.get('hybrid_router_cut') is not None:
            from frontier_hybrid_router import install_hybrid_router_cut
            cut = REQUEST['hybrid_router_cut']
            install_hybrid_router_cut(cut['logical_layer'], cut['source_pp'],
                cut.get('packet_schema', 'dense'), cut.get('boundary_dtypes', False))
        elif REQUEST.get('hybrid_fp32_transport', False):
            from frontier_hybrid_router import install_hybrid_fp32_transport
            install_hybrid_fp32_transport()
        if REQUEST.get('hybrid_router_cut') is not None or REQUEST.get('hybrid_fp32_transport', False):
            cfg.model.variable_seq_lengths = True
            cfg.model.pipeline_dtype = native.torch.float32
            cfg.mixed_precision = native.get_mixed_precision_config(cfg.mixed_precision)
            cfg.mixed_precision.pipeline_dtype = native.torch.float32
    else:
        counts = REQUEST.get('decoder_chunks') or native.uniform_decoder_chunks(cfg.model.num_layers, pp, vpp)
        if len(counts) != pp * vpp or min(counts) < 1 or sum(counts) != cfg.model.num_layers:
            raise ValueError('Dense Region分区必须完整覆盖模型层并匹配PP/VPP')
        if not (REQUEST.get('native_uniform_pipeline_layout', False) and vpp == 1 and len(set(counts)) == 1):
            layout = [['decoder'] * count for count in counts]
            layout[0].insert(0, 'embedding')
            layout[-1].extend(['mtp'] * (cfg.model.mtp_num_layers or 0))
            layout[-1].append('loss')
            cfg.model.pipeline_model_parallel_layout = PipelineParallelLayerLayout(layout, pp)
            cfg.model.pipeline_model_parallel_layout.validate_layer_layout(cfg.model.num_layers, cfg.model.mtp_num_layers)
    cfg.dataset.num_workers = 0
    if 'training_data_contract' in REQUEST:
        data_contract = REQUEST['training_data_contract']
        if (set(data_contract) != {'training_index_samples', 'workers', 'persistent_workers'}
                or type(data_contract['training_index_samples']) is not int
                or data_contract['training_index_samples'] < REQUEST['global_batch'] * (REQUEST['warmup'] + REQUEST['measured'])
                or type(data_contract['workers']) is not int or not 0 <= data_contract['workers'] <= 2
                or type(data_contract['persistent_workers']) is not bool
                or (data_contract['persistent_workers'] and data_contract['workers'] == 0)):
            raise ValueError('训练数据索引范围或数据加载进程配置无效')
        cfg.dataset.num_workers = data_contract['workers']
        cfg.dataset.persistent_workers = data_contract['persistent_workers']
        bridge_pretrain = importlib.import_module('megatron.bridge.training.pretrain')
        original_provider_factory = bridge_pretrain.get_dataset_provider

        def reference_provider_factory(dataset_config):
            provider = original_provider_factory(dataset_config)

            @wraps(provider)
            def with_reference_index(train_val_test_num_samples, config, *args, **kwargs):
                observed = list(train_val_test_num_samples)
                samples = [data_contract['training_index_samples'], *observed[1:]]
                print(f'TRAINING_DATA_INDEX requested={observed} actual={samples}', flush=True)
                return provider(samples, config, *args, **kwargs)

            return with_reference_index

        bridge_pretrain.get_dataset_provider = reference_provider_factory
    optimizer_offload_fraction = float(REQUEST.get('optimizer_offload_fraction',
        (memory or {}).get('optimizer_offload_fraction', 0.0)))
    if not 0 <= optimizer_offload_fraction <= 1:
        raise ValueError('CPU Optimizer Offload比例必须位于零至一之间')
    if optimizer_offload_fraction > 0 and (use_fsdp or not use_distributed_optimizer
            or not cfg.optimizer.use_precision_aware_optimizer):
        raise ValueError('CPU Optimizer Offload要求Megatron DDP、Distributed Optimizer与Precision-Aware Optimizer')
    cfg.optimizer.optimizer_cpu_offload = optimizer_offload_fraction > 0
    cfg.optimizer.optimizer_offload_fraction = optimizer_offload_fraction
    if 'optimizer_pin_cpu_params' in REQUEST:
        if type(REQUEST['optimizer_pin_cpu_params']) is not bool:
            raise ValueError('optimizer_pin_cpu_params必须为布尔值')
        cfg.optimizer.pin_cpu_params = REQUEST['optimizer_pin_cpu_params']
    if optimizer_offload_fraction > 0:
        print(f'OPTIMIZER_OFFLOAD_POLICY fraction={optimizer_offload_fraction} '
              f'pin_cpu_params={cfg.optimizer.pin_cpu_params}', flush=True)
    cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = bool(
        REQUEST.get('optimizer_transfer_overlap', False))
    if cfg.optimizer.overlap_cpu_optimizer_d2h_h2d and optimizer_offload_fraction == 0:
        raise ValueError('Optimizer传输重叠要求启用CPU Optimizer Offload')
    cfg.ddp.overlap_grad_reduce = not use_fsdp
    cfg.ddp.overlap_param_gather = cfg.optimizer.use_distributed_optimizer and not use_fsdp
    if 'overlap_grad_reduce' in REQUEST:
        cfg.ddp.overlap_grad_reduce = REQUEST['overlap_grad_reduce']
    if 'overlap_param_gather' in REQUEST:
        cfg.ddp.overlap_param_gather = REQUEST['overlap_param_gather']
    if isinstance(cfg.model, native.FalconH1ModelProvider) and '--recompute' in sys.argv:
        cfg.ddp.overlap_grad_reduce = False
        cfg.ddp.overlap_param_gather = False
    print('FRONTIER_HYBRID_PROGRAM ' + json.dumps({'decoder_chunks': counts, 'hybrid_layer_pattern': pattern,
        'tp': cfg.model.tensor_model_parallel_size, 'cp': cfg.model.context_parallel_size,
        'pp': pp, 'vpp': vpp, 'data_workers': cfg.dataset.num_workers,
        'precision_aware_optimizer': cfg.optimizer.use_precision_aware_optimizer,
        'main_grads_dtype': str(cfg.optimizer.main_grads_dtype),
        'main_params_dtype': str(cfg.optimizer.main_params_dtype),
        'exp_avg_dtype': str(cfg.optimizer.exp_avg_dtype),
        'exp_avg_sq_dtype': str(cfg.optimizer.exp_avg_sq_dtype)}), flush=True)
    if optimizer_offload_fraction <= 0:
        return original_pretrain(config=cfg, forward_step_func=forward_step_func, callbacks=callbacks)
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    if REQUEST.get('optimizer_cpu_copy_mode') is not None:
        mode = REQUEST['optimizer_cpu_copy_mode']
        if mode not in ('direct_pageable', 'batched_pageable', 'batched_pinned', 'batched'):
            raise ValueError('未知的优化器参数复制方式')
        if mode != 'batched' and cfg.optimizer.pin_cpu_params != (mode == 'batched_pinned'):
            raise ValueError('优化器参数复制方式与CPU参数锁页设置不一致')
        HybridDeviceOptimizer._get_sub_optimizer_param_groups = optimizer_parameter_groups_with_cpu_copy
        print(f'OPTIMIZER_OFFLOAD_COPY_MODE {mode}', flush=True)
    def trace(function, name):
        @wraps(function)
        def measured(*args, **kwargs):
            started = time.monotonic()
            print(f'OPTIMIZER_OFFLOAD_INIT begin {name}', flush=True)
            result = function(*args, **kwargs)
            print(f'OPTIMIZER_OFFLOAD_INIT end {name} seconds={time.monotonic()-started:.6f}', flush=True)
            return result
        return measured

    for method_name in ('_init_sub_optimizers', '_get_sub_optimizer_param_groups'):
        original = getattr(HybridDeviceOptimizer, method_name)
        setattr(HybridDeviceOptimizer, method_name, trace(original, method_name))
    HybridDeviceOptimizer.build_cpu_optimizer_list = staticmethod(trace(
        HybridDeviceOptimizer.build_cpu_optimizer_list, 'build_cpu_optimizer_list'))
    faulthandler.dump_traceback_later(35, repeat=True, file=sys.stderr)
    try:
        return original_pretrain(config=cfg, forward_step_func=forward_step_func, callbacks=callbacks)
    finally:
        faulthandler.cancel_dump_traceback_later()


native.pretrain = pretrain
native.Evidence.on_train_start = record_ownership
native.Evidence.on_train_step_start = profile_step_start
native.Evidence.on_train_step_end = profile_step_end
native.Evidence.on_train_end = record_end
if REQUEST.get('paramorph', False):
    import paramorph_runtime
    PARAMORPH_RUNTIME = paramorph_runtime.install(native, REQUEST)
native.main()
