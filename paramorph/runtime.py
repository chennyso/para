from collections import defaultdict
from contextlib import nullcontext
import base64
import functools
import hashlib
import inspect
import json
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.pipeline_parallel import schedules, p2p_communication
from megatron.core.optimizer import optimizer as optimizer_module
from megatron.core.optimizer import clip_grads as clip_grads_module
from megatron.core.utils import to_local_if_dtensor


class RuntimeEvidence:
    def __init__(self, native, request):
        self.native = native
        self.request = request
        self.step = -1
        self.events = []
        self.windows = []
        self.identities = {}
        self.counters = defaultdict(int)
        self.profiler = None
        self.schedule_tables = []
        self.output = None
        self.active_region = None
        self.executed_plans = []
        self.plan_executor = None
        self.layer_backward_events = {}
        self.active_plan_task = None
        self.parameter_storages = set()
        self.saved_activation_storages = defaultdict(dict)
        self.cuda_origins = {}
        self.cuda_origin_host_anchors = {}
        self.iteration_peak = 0
        self.external_activation_storages = set()
        self.active_operation = None
        self.communication_iterations = []
        self.communication_initialization_records = []
        self.reference_phase = False
        self.numerical_capture = None
        self.numerical_reports = {}
        self.numerical_optimizer_audit = None
        self.memory_audit = None
        self.operator_backward_events = []
        self.clock_calibration = None
        self.optimizer_norm_context = None

    def install_optimizer_cost_calibration(self, optimizer):
        if not self.request.get('paramorph_optimizer_calibration', False):
            return
        if not self.request.get('paramorph_calibration', False):
            raise ValueError('Optimizer组件采集要求启用CUDA Event校准')
        precision = self.request.get('precision', 'bfloat16').removeprefix('torch.')
        if precision not in ('bfloat16', 'float16'):
            raise ValueError('Optimizer组件采集要求BF16或者FP16训练')
        pending, leaves = [optimizer], []
        while pending:
            component = pending.pop(0)
            chained = getattr(component, 'chained_optimizers', None)
            if chained is not None:
                pending.extend(chained)
            elif not getattr(component, 'is_stub_optimizer', False):
                leaves.append(component)
        if not leaves:
            raise ValueError('当前Rank没有可采集的本地Optimizer')
        excluded = ['gradient_norm_collectives', 'overflow_collectives',
            'parameter_gather', 'upstream_stream_waits']
        if precision == 'float16':
            excluded.append('fp16_loss_unscale_and_inf_check')

        def install(owner, attribute, kind, component_index, parameter_elements, *, synchronize=True,
                    scope='local_optimizer_update_and_copy'):
            original = getattr(owner, attribute)
            function = inspect.unwrap(original)
            source = inspect.getsource(function)
            identity = dict(cost_scope=scope,
                precision=precision,
                excluded_operations=excluded, component_index=component_index,
                parameter_elements=parameter_elements, owned_elements=parameter_elements,
                callable_module=function.__module__, callable_name=function.__qualname__,
                callable_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                synchronization_policy='cuda_device_synchronize_before_component' if synchronize
                    else 'cuda_events_on_current_stream')

            @functools.wraps(original)
            def measured(*args, **kwargs):
                if self.step < 0 or self.reference_phase:
                    return original(*args, **kwargs)
                # 将前序GPU工作排除在本地组件的计时区间之外。
                if synchronize:
                    torch.cuda.synchronize()
                return self.record(kind, attribute, original, args, kwargs, identity)

            setattr(owner, attribute, measured)

        for index, component in enumerate(leaves):
            if type(component).__module__ not in (
                    'megatron.core.optimizer.optimizer', 'megatron.core.optimizer.distrib_optimizer'):
                raise ValueError('Optimizer组件采集没有当前Megatron实现的调用边界依据')
            ddp = getattr(component, 'ddp_config', None)
            if ddp is not None and (getattr(ddp, 'fp8_param_gather', False)
                    or getattr(ddp, 'fp4_param_gather', False)):
                raise ValueError('量化参数复制可能包含通信，无法登记为纯本地复制')
            if getattr(component.config, 'reuse_grad_buf_for_mxfp8_param_ag', False):
                raise ValueError('MXFP8参数复制使用独立调用路径')
            inner = component.optimizer
            hybrid = (type(inner).__name__ == 'HybridDeviceOptimizer'
                and type(inner).__module__ == 'megatron.core.optimizer.cpu_offloading.hybrid_optimizer')
            if hybrid:
                if inner.offload_fraction != self.request.get('optimizer_offload_fraction', 0.0):
                    raise ValueError('实际HybridDeviceOptimizer卸载比例与训练配置不同')
                if inner.overlap_cpu_optimizer_d2h_h2d != self.request.get('optimizer_transfer_overlap', False):
                    raise ValueError('实际HybridDeviceOptimizer传输重叠与训练配置不同')
            elif (type(inner).__name__ not in ('FusedAdam', 'Adam', 'AdamW')
                    or not type(inner).__module__.startswith(
                        ('transformer_engine.pytorch.optimizers.', 'apex.optimizers.', 'torch.optim.'))):
                raise ValueError('当前Optimizer内部更新实现未登记为本地Adam更新')
            parameter_elements = sum(to_local_if_dtensor(parameter).numel() for group in inner.param_groups
                for parameter in group['params'])
            install(inner, 'step', 'optimizer_local_update', index, parameter_elements)
            install(component, '_copy_main_params_to_model_params',
                'optimizer_parameter_copy', index, parameter_elements)
            install(component, '_copy_model_grads_to_main_grads',
                'optimizer_prepare_grads', index, parameter_elements,
                synchronize=False, scope='local_optimizer_prepare_grads')
        self.install_optimizer_norm_calibration()

    def install_optimizer_norm_calibration(self):
        original_norm = optimizer_module.get_grad_norm_fp32
        original_clip = optimizer_module.clip_grad_by_total_norm_fp32
        original_reduce = dist.all_reduce
        norm_signature = inspect.signature(original_norm)
        clip_signature = inspect.signature(original_clip)
        reduce_signature = inspect.signature(original_reduce)
        excluded = ['gradient_norm_collectives', 'overflow_collectives',
            'parameter_gather', 'upstream_stream_waits']

        def source_identity(function):
            function = inspect.unwrap(function)
            return dict(callable_module=function.__module__, callable_name=function.__qualname__,
                callable_source_sha256=hashlib.sha256(inspect.getsource(function).encode()).hexdigest())

        norm_source, clip_source = source_identity(original_norm), source_identity(original_clip)

        def begin_segment(context):
            context['cuda_start'] = torch.cuda.Event(enable_timing=True)
            context['cuda_start'].record()
            context['host_start_ns'] = time.perf_counter_ns()
            context['allocated_before'] = torch.cuda.memory_allocated()

        def finish_segment(context, *, finalize=False):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            kind = 'optimizer_local_grad_norm_finalize' if finalize else 'optimizer_local_grad_norm'
            identity = dict(context['identity'], segment_index=context['segment_index'],
                cost_scope='local_optimizer_gradient_norm_finalize' if finalize
                    else 'local_optimizer_gradient_norm',
                collective_count=context.get('collective_count', 0))
            self.events.append(dict(step=self.step, kind=kind,
                operation=kind, identity=identity,
                cuda_start=context['cuda_start'], cuda_end=end,
                host_start_ns=context['host_start_ns'], host_end_ns=time.perf_counter_ns(),
                stream_id=torch.cuda.current_stream().cuda_stream,
                stream_identity_domain='cuda_runtime_stream_handle',
                allocated_before=context['allocated_before'], allocated_after=torch.cuda.memory_allocated()))
            context['segment_index'] += 1

        @functools.wraps(original_reduce)
        def norm_reduce(*args, **kwargs):
            context = self.optimizer_norm_context
            if context is None:
                return original_reduce(*args, **kwargs)
            finish_segment(context)
            bound = reduce_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.arguments['async_op']:
                raise ValueError('Optimizer Norm采集要求当前后端的同步AllReduce调用')
            tensor, group = bound.arguments['tensor'], bound.arguments['group']
            ranks = list(range(dist.get_world_size())) if group is None else dist.get_process_group_ranks(group)
            identity = dict(norm_source, cost_scope='optimizer_gradient_norm_collective',
                collective='all_reduce', dtype=str(tensor.dtype).removeprefix('torch.'),
                payload_bytes=tensor.numel() * tensor.element_size(), group_ranks=ranks,
                norm_call_index=context['identity']['component_index'],
                reduction=str(bound.arguments['op']),
                synchronization_policy='cuda_events_on_current_stream')
            result = self.record('optimizer_gradient_norm_all_reduce',
                'optimizer_gradient_norm_all_reduce', original_reduce, args, kwargs, identity)
            context['collective_count'] = context.get('collective_count', 0) + 1
            begin_segment(context)
            return result

        @functools.wraps(original_norm)
        def measured_norm(*args, **kwargs):
            if self.step < 0 or self.reference_phase:
                return original_norm(*args, **kwargs)
            if self.optimizer_norm_context is not None:
                raise ValueError('Optimizer Norm采集检测到嵌套Norm调用')
            bound = norm_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            gradients = bound.arguments['grads_for_norm']
            gradients = [gradients] if isinstance(gradients, torch.Tensor) else gradients
            gradients = [to_local_if_dtensor(gradient) for gradient in gradients]
            group = bound.arguments['grad_stats_parallel_group']
            ranks = list(range(dist.get_world_size())) if group is None else dist.get_process_group_ranks(group)
            counter = (self.step, 'optimizer_norm_calls')
            index = self.counters[counter]
            self.counters[counter] += 1
            identity = dict(norm_source, cost_scope='local_optimizer_gradient_norm',
                excluded_operations=excluded, component_index=index,
                norm_elements=sum(gradient.numel() for gradient in gradients),
                grad_dtypes=sorted({str(gradient.dtype) for gradient in gradients}),
                grad_stats_group_ranks=ranks, norm_type=bound.arguments['norm_type'],
                synchronization_policy='cuda_event_boundaries_around_collectives')
            context = dict(identity=identity, segment_index=0)
            self.optimizer_norm_context = context
            begin_segment(context)
            result = original_norm(*args, **kwargs)
            finish_segment(context, finalize=True)
            self.optimizer_norm_context = None
            return result

        @functools.wraps(original_clip)
        def measured_clip(*args, **kwargs):
            if self.step < 0 or self.reference_phase:
                return original_clip(*args, **kwargs)
            bound = clip_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = bound.arguments['parameters']
            parameters = [parameters] if isinstance(parameters, torch.Tensor) else parameters
            gradient_name = 'decoupled_grad' if bound.arguments['use_decoupled_grad'] else 'grad'
            gradients = [to_local_if_dtensor(getattr(parameter, gradient_name))
                for parameter in parameters if getattr(parameter, gradient_name, None) is not None]
            total_norm, clip_grad = bound.arguments['total_norm'], bound.arguments['max_norm']
            counter = (self.step, 'optimizer_clip_calls')
            index = self.counters[counter]
            self.counters[counter] += 1
            identity = dict(clip_source, cost_scope='local_optimizer_gradient_clip',
                excluded_operations=excluded, component_index=index,
                owned_elements=sum(gradient.numel() for gradient in gradients),
                grad_dtypes=sorted({str(gradient.dtype) for gradient in gradients}),
                clip_grad=clip_grad,
                clip_kernel_dispatched=isinstance(total_norm, torch.Tensor)
                    or clip_grad / (total_norm + 1e-6) < 1.0,
                synchronization_policy='cuda_events_on_current_stream')
            return self.record('optimizer_local_grad_clip', 'optimizer_local_grad_clip',
                original_clip, args, kwargs, identity)

        optimizer_module.get_grad_norm_fp32 = measured_norm
        clip_grads_module.get_grad_norm_fp32 = measured_norm
        optimizer_module.clip_grad_by_total_norm_fp32 = measured_clip
        clip_grads_module.clip_grad_by_total_norm_fp32 = measured_clip
        dist.all_reduce = norm_reduce

    def save_partial_calibration(self):
        if not self.request.get('paramorph_calibration', False):
            return
        torch.cuda.synchronize()
        events = []
        for event in self.events:
            row = {key: value for key, value in event.items() if key not in ('cuda_start', 'cuda_end')}
            row['cuda_ms'] = event['cuda_start'].elapsed_time(event['cuda_end'])
            events.append(row)
        (self.output / f'rank{self.rank}.paramorph-partial.json').write_text(json.dumps(
            dict(rank=self.rank, pp_rank=self.pp_rank, request=self.request,
                 events=events, training_completion_certified=False), separators=(',', ':'), allow_nan=False) + '\n')

    def record(self, kind, operation, function, args, kwargs, identity=None):
        if self.step < 0 or self.reference_phase:
            return function(*args, **kwargs)
        identity = identity or {}
        if self.active_plan_task is not None:
            identity = dict(identity, task_id=self.active_plan_task)
        calibrated = self.request.get('paramorph_calibration', False)
        label = json.dumps(dict(identity, step=self.step, rank=dist.get_rank(), operation=operation), sort_keys=True)
        encoded_label = base64.urlsafe_b64encode(label.encode()).decode()
        context = torch.profiler.record_function('ParaMorphJSON/' + encoded_label) if self.profiler else nullcontext()
        nsys_active = (self.request.get('paramorph_nsys_trace', False)
            and self.step >= self.request['warmup'])
        nsys_kind = {
            'forward_compute': 'F', 'backward_compute': 'B',
            'pp_p2p_exchange': 'P2P', 'dp_reduce': 'DP',
            'optimizer_step': 'OPT', 'special_compute': 'SPECIAL',
        }.get(kind, kind.upper())
        nsys_label = 'FRONTIER_NSYS:' + json.dumps(dict(kind=nsys_kind,
            rank=dist.get_rank(), iteration=self.step + 1, operation=operation, **identity),
            separators=(',', ':'), sort_keys=True)
        nsys_context = torch.cuda.nvtx.range(
            nsys_label.replace('{', '{{').replace('}', '}}')) if nsys_active else nullcontext()
        start = torch.cuda.Event(enable_timing=True) if calibrated else None
        end = torch.cuda.Event(enable_timing=True) if calibrated else None
        if start is not None:
            start.record()
        before = time.perf_counter_ns()
        allocated = torch.cuda.memory_allocated()
        previous_operation = self.active_operation
        if self.memory_audit is not None and kind not in ('layer_compute', 'operator_compute'):
            self.memory_audit.snapshot(operation + ':before')
        self.active_operation = dict(kind=kind, operation=operation, identity=identity,
                                     allocated_before_bytes=allocated)
        measure_workspace = calibrated and kind in ('forward_compute', 'backward_compute',
                                                     'dp_reduce', 'optimizer_step')
        if measure_workspace:
            self.iteration_peak = max(self.iteration_peak, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        previous_region = self.active_region
        if kind in ('forward_compute', 'backward_compute'):
            self.active_region = dict(identity, phase=kind)
        with context, nsys_context:
            result = function(*args, **kwargs)
        self.active_region = previous_region
        self.active_operation = previous_operation
        if self.memory_audit is not None and kind not in ('layer_compute', 'operator_compute'):
            self.memory_audit.snapshot(operation + ':after')
        after = time.perf_counter_ns()
        if end is not None:
            end.record()
        measure_activation = (calibrated and kind == 'forward_compute' and
            (self.plan_executor is None or self.step <= self.request.get('paramorph_serial_calibration_steps', 0)))
        if measure_activation:
            key = (self.step, identity['vp_stage'], identity['microbatch'])
            self.collect_saved_activations(result[0], key)
        self.events.append(dict(step=self.step, kind=kind, operation=operation,
            stream_id=torch.cuda.current_stream().cuda_stream,
            stream_identity_domain='cuda_runtime_stream_handle',
            identity=identity, host_start_ns=before, host_end_ns=after,
            allocated_before=allocated, allocated_after=torch.cuda.memory_allocated(),
            cuda_start=start, cuda_end=end))
        if measure_workspace:
            peak = torch.cuda.max_memory_allocated()
            self.iteration_peak = max(self.iteration_peak, peak)
            self.events[-1]['operation_peak_allocated_bytes'] = peak
            self.events[-1]['workspace_upper_bound_bytes'] = max(0, peak - allocated)
        if measure_activation:
            self.events[-1]['saved_activation_bytes'] = sum(self.saved_activation_storages[key].values())
        return result

    def write_parameter_ledger(self, models):
        if not self.request.get('paramorph_parameter_ledger', False):
            return
        rows, buffers, layouts = [], [], []
        models = models if isinstance(models, (list, tuple)) else [models]
        for vp, model in enumerate(models):
            wrapper = model
            while hasattr(model, 'module'):
                model = model.module
            layer_ids = {id(parameter): layer.layer_number - 1
                for layer in model.decoder.layers for parameter in layer.parameters()}
            for order, (name, parameter) in enumerate(model.named_parameters()):
                local = to_local_if_dtensor(parameter)
                gradient = getattr(parameter, 'main_grad', parameter.grad)
                gradient = None if gradient is None else to_local_if_dtensor(gradient)
                rows.append(dict(param_key=name, vp_stage=vp, parameter_order=order,
                    layer_id=layer_ids.get(id(parameter)), local_shape=list(local.shape),
                    local_elements=local.numel(), train_param_dtype=str(local.dtype),
                    requires_grad=parameter.requires_grad,
                    grad_dtype=None if gradient is None else str(gradient.dtype),
                    gradient_elements=None if gradient is None else gradient.numel(),
                    tensor_model_parallel=getattr(parameter, 'tensor_model_parallel', False),
                    partition_axis=getattr(parameter, 'partition_dim', None),
                    partition_stride=getattr(parameter, 'partition_stride', None),
                    partition_segments=getattr(parameter, 'partition_sizes', None),
                    expert_parallel=not getattr(parameter, 'allreduce', True),
                    shared_embedding=getattr(parameter, 'shared_embedding', False),
                    shared=getattr(parameter, 'shared', False)))
            buffers.extend(dict(param_key=name, vp_stage=vp, local_shape=list(value.shape),
                dtype=str(value.dtype), bytes=value.numel() * value.element_size())
                for name, value in model.named_buffers())
            if isinstance(wrapper, DistributedDataParallel) and wrapper.ddp_config.use_distributed_optimizer:
                names = {parameter: name for name, parameter in model.named_parameters()}
                for key, layout in wrapper.full_param_layout.layouts.items():
                    group = (wrapper.intra_expt_dp_group if key.is_expert_parallel
                        else wrapper.intra_dp_cp_group)
                    if layout.num_optimizer_shards != group.size():
                        raise ValueError('原生Optimizer布局与实际状态分片进程组不一致')
                    buckets = []
                    for bucket_id, (start, end) in enumerate(layout.bucket_indices):
                        width = (end - start) // group.size()
                        shard_start = start + group.rank() * width
                        shard_end = shard_start + width
                        parameters = [dict(param_key=names[parameter], start=first, end=last,
                            optimizer_owned_elements=max(0, min(last, shard_end) - max(first, shard_start)))
                            for parameter, (first, last, index) in layout.param_index_map.items()
                            if index == bucket_id]
                        buckets.append(dict(bucket_id=bucket_id, start=start, end=end,
                            numel_unpadded=layout.per_bucket_numel_unpadded[bucket_id],
                            optimizer_shard_start=shard_start, optimizer_shard_end=shard_end,
                            optimizer_owned_elements=sum(row['optimizer_owned_elements'] for row in parameters),
                            parameters=parameters))
                    layouts.append(dict(vp_stage=vp, family='expert' if key.is_expert_parallel else 'dense',
                        param_dtype=str(key.param_dtype), grad_dtype=str(key.grad_dtype),
                        physical_ranks=dist.get_process_group_ranks(group),
                        bucket_size=wrapper.bucket_size,
                        pad_buckets_for_high_nccl_busbw=wrapper.ddp_config.pad_buckets_for_high_nccl_busbw,
                        num_optimizer_shards=layout.num_optimizer_shards, buckets=buckets))
        document = dict(version=1, kind='ModuleParameterLedger', rank=self.rank,
            request=self.request, parameters=rows, buffers=buffers, native_buffer_layouts=layouts,
            source='actual_deployed_named_parameters_and_native_partition_attributes')
        (self.output / f'rank{self.rank}.parameter-ledger.json').write_text(
            json.dumps(document, indent=2, allow_nan=False) + '\n')

    def install_bucket_calibration(self, models):
        calibrate = self.request.get('paramorph_calibration', False)
        if not calibrate and not self.request.get('paramorph_parameter_ledger', False):
            return
        self.bucket_calibration = dict(schema_version=1, rank=self.rank, request=self.request,
            source=('actual_deployed_DDP_bucket_objects_and_collective_dispatch' if calibrate
                else 'actual_deployed_DDP_bucket_objects'),
            event_capture_scope='multi_rank_collective_groups', groups=[], events=[])
        models = models if isinstance(models, (list, tuple)) else [models]
        objects, parameter_names, identifiers = [], {}, {}
        for vp, model in enumerate(models):
            if not isinstance(model, DistributedDataParallel):
                continue
            raw = model.module
            while hasattr(raw, 'module'):
                raw = raw.module
            layer_ids = {id(parameter): layer.layer_number - 1
                for layer in raw.decoder.layers for parameter in layer.parameters()}
            names = {id(parameter): name for name, parameter in raw.named_parameters()}
            for family, groups in (('dense', model.bucket_groups),
                                   ('expert', model.expert_parallel_bucket_groups)):
                for index, group in enumerate(groups):
                    identifier = f'vp{vp}:{family}:group{index}'
                    identifiers[id(group)] = identifier
                    objects.append((group, identifier, vp, family, index, model.ddp_config))
                    for parameter in group.params:
                        name = names[id(parameter)]
                        special = ('embedding' if 'word_embeddings' in name else
                            'output' if 'output_layer' in name else
                            'final_norm' if 'final_layernorm' in name else
                            'mtp' if 'mtp' in name else 'other')
                        parameter_names[id(parameter)] = dict(name=name,
                            layer_id=layer_ids.get(id(parameter)), special_module=special,
                            elements=parameter.numel(), shape=list(parameter.shape))
        for group, identifier, vp, family, index, config in objects:
            process_group = (group.intra_distributed_optimizer_instance_group
                if config.use_distributed_optimizer else group.data_parallel_group)
            ranks = dist.get_process_group_ranks(process_group)
            next_group = group.next_param_gather_bucket_group
            previous_group = group.previous_grad_reduce_bucket_group
            buckets = []
            for position, bucket in enumerate(group.buckets):
                buckets.append(dict(bucket_id=f'{identifier}:bucket{position}',
                    backend_bucket_id=bucket.bucket_id, offset=bucket.offset,
                    numel_unpadded=bucket.numel_unpadded,
                    grad_elements=bucket.grad_data.numel(), grad_dtype=str(bucket.grad_data.dtype),
                    param_elements=None if bucket.param_data is None else bucket.param_data.numel(),
                    param_dtype=None if bucket.param_data is None else str(bucket.param_data.dtype),
                    parameters=[dict(parameter_names[id(parameter)],
                        bucket_interval=list(bucket.param_to_index[parameter]))
                        for parameter in bucket.params_list]))
            self.bucket_calibration['groups'].append(dict(group_id=identifier, vp_stage=vp,
                family=family, group_index=index, physical_ranks=ranks, buckets=buckets,
                overlap_grad_reduce=config.overlap_grad_reduce,
                overlap_param_gather=config.overlap_param_gather,
                align_param_gather=config.align_param_gather,
                use_distributed_optimizer=config.use_distributed_optimizer,
                distributed_optimizer_instances=config.num_distributed_optimizer_instances,
                next_param_gather_group_id=None if next_group is None else identifiers[id(next_group)],
                previous_grad_reduce_group_id=None if previous_group is None else identifiers[id(previous_group)]))
            if not calibrate or len(ranks) == 1:
                continue
            original_ready = group.register_grad_ready
            def ready(parameter, *args, original=original_ready, current=group, **kwargs):
                current.paramorph_last_ready_parameter = parameter_names[id(parameter)]
                return original(parameter, *args, **kwargs)
            group.register_grad_ready = ready
            for method in ('start_grad_sync', 'finish_grad_sync', 'start_param_sync', 'finish_param_sync'):
                original = getattr(group, method)
                def invoke(*args, original=original, current=group, name=method,
                           group_id=identifier, **kwargs):
                    before = time.perf_counter_ns()
                    region = None if self.active_region is None else dict(self.active_region)
                    dispatched = current.param_gather_dispatched
                    grad_handle = current.grad_reduce_handle is not None
                    result = original(*args, **kwargs)
                    if self.step >= 0:
                        self.bucket_calibration['events'].append(dict(step=self.step,
                            group_id=group_id, operation=name, host_start_ns=before,
                            host_end_ns=time.perf_counter_ns(), active_region=region,
                            active_operation=self.active_operation,
                            last_ready_parameter=getattr(current, 'paramorph_last_ready_parameter', None),
                            is_last_microbatch=current.is_last_microbatch,
                            is_first_batch=current.is_first_batch,
                            param_gather_dispatched_before=dispatched,
                            param_gather_dispatched_after=current.param_gather_dispatched,
                            grad_handle_before=grad_handle,
                            grad_handle_after=current.grad_reduce_handle is not None,
                            force_sync=bool(kwargs.get('force_sync', False)),
                            scope='host_dispatch_and_wait_markers_not_collective_service_time'))
                    return result
                setattr(group, method, invoke)
        if self.request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp' and not objects:
            raise ValueError('DDP校准没有获得包装对象及真实Bucket分组')
        self.write_bucket_calibration()

    def write_bucket_calibration(self):
        if hasattr(self, 'bucket_calibration'):
            (self.output / f'rank{self.rank}.paramorph-buckets.json').write_text(
                json.dumps(self.bucket_calibration, separators=(',', ':'), allow_nan=False) + '\n')

    def collect_saved_activations(self, output, key):
        tensors = output if isinstance(output, (tuple, list)) else [output]
        pending = [tensor.grad_fn for tensor in tensors if tensor is not None and tensor.grad_fn is not None]
        visited = set()
        excluded = self.parameter_storages | self.external_activation_storages
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            names = [name for name in dir(node) if name.startswith('_saved_')]
            if hasattr(node, 'saved_tensors'):
                names.append('saved_tensors')
            for name in names:
                value = getattr(node, name)
                saved = value if isinstance(value, (tuple, list)) else [value]
                for tensor in saved:
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                        tensor = to_local_if_dtensor(tensor)
                        storage = tensor.untyped_storage()
                        if storage.nbytes() and storage.data_ptr() not in excluded:
                            self.saved_activation_storages[key][storage.data_ptr()] = storage.nbytes()
            pending.extend(predecessor for predecessor, _ in node.next_functions if predecessor is not None)

    def register_parameter_storages(self, models):
        if self.request.get('execution_backend', 'megatron_ddp') != 'megatron_fsdp':
            self.parameter_storages = {parameter.untyped_storage().data_ptr()
                for model in models for parameter in model.parameters()}
            return
        pointers = set()
        for _, buffer in self.fsdp_model_bindings(models):
            # FSDP持久参数数据由DataParallelBuffer持有，模型Parameter可能已经释放完整存储。
            for group in buffer.parameter_groups:
                for name in ('model_weight_buffer', 'main_weight_buffer', 'transpose_weight_buffer'):
                    buffer = getattr(group, name)
                    if buffer is None:
                        continue
                    local = buffer.data
                    if local is None:
                        continue
                    storage = local.untyped_storage()
                    if storage.nbytes():
                        pointers.add(storage.data_ptr())
                if group.sharding_strategy == 'no_shard' and group.model_weight_buffer is None:
                    for parameter in group.params:
                        storage = to_local_if_dtensor(parameter).untyped_storage()
                        if storage.nbytes():
                            pointers.add(storage.data_ptr())
        self.parameter_storages = pointers

    @staticmethod
    def fsdp_model_bindings(models):
        models = models if isinstance(models, (list, tuple)) else [models]
        bindings = []
        for model in models:
            current, buffer, visited = model, None, set()
            while True:
                if id(current) in visited:
                    raise ValueError('FSDP模型包装关系存在循环')
                visited.add(id(current))
                if 'param_and_grad_buffer' in vars(current):
                    observed = current.param_and_grad_buffer
                    if buffer is not None and buffer is not observed:
                        raise ValueError('同一模型包装关系包含不同FSDP参数Buffer')
                    buffer = observed
                if not hasattr(current, 'module'):
                    break
                current = current.module
            if buffer is None:
                raise ValueError('FSDP采集必须提供包含实际参数Buffer的训练模型包装对象')
            bindings.append((current, buffer))
        return bindings

    def fsdp_layer_state_rows(self, models, optimizer):
        rows, parameter_layers, optimizer_parameters = {}, {}, {}
        seen = defaultdict(set)

        def account(layer_id, category, value):
            local = to_local_if_dtensor(value)
            if local.device.type != 'cuda' or local.numel() == 0:
                return
            identity = (local.data_ptr(), local.numel(), local.dtype)
            if identity not in seen[layer_id]:
                rows[layer_id][category] += local.numel() * local.element_size()
                seen[layer_id].add(identity)

        for model, buffer in self.fsdp_model_bindings(models):
            for layer in model.decoder.layers:
                layer_id = layer.layer_number - 1
                rows[layer_id] = dict(weight_bytes=0, gradient_bytes=0, optimizer_bytes=0)
                for parameter in layer.parameters():
                    original = parameter.orig_param if hasattr(parameter, 'orig_param') else parameter
                    parameter_layers[id(original)] = layer_id
            for name, parameter in buffer.optimizer_named_parameters:
                original = parameter.orig_param
                if id(original) in parameter_layers:
                    optimizer_parameters[id(parameter)] = (parameter_layers[id(original)], name)
            for group in buffer.parameter_groups:
                for parameter in group.params:
                    layer_id = parameter_layers.get(id(parameter))
                    if layer_id is None:
                        continue
                    weight = group.model_weight_buffer
                    if weight is None:
                        if group.sharding_strategy != 'no_shard':
                            raise ValueError('FSDP分片参数缺少实际模型权重Buffer')
                        account(layer_id, 'weight_bytes', parameter)
                    else:
                        account(layer_id, 'weight_bytes', weight.get_item(weight.param_idx[parameter]))
                    gradient = group.main_grad_buffer
                    if gradient is not None:
                        account(layer_id, 'gradient_bytes', gradient.get_item(gradient.param_idx[parameter]))
                    main = group.main_weight_buffer
                    if main is not None:
                        account(layer_id, 'optimizer_bytes', main.get_item(main.param_idx[parameter]))
        pending = [optimizer]
        while pending:
            component = pending.pop(0)
            children = getattr(component, 'chained_optimizers', None)
            if children is not None:
                pending.extend(children)
                continue
            if getattr(component, 'is_stub_optimizer', False):
                continue
            for group in component.optimizer.param_groups:
                for parameter in group['params']:
                    binding = optimizer_parameters.get(id(parameter))
                    if binding is None:
                        continue
                    layer_id, _ = binding
                    for value in component.optimizer.state.get(parameter, {}).values():
                        if isinstance(value, torch.Tensor):
                            account(layer_id, 'optimizer_bytes', value)
        return rows

    def install(self):
        original_gradient_hook = DistributedDataParallel._make_backward_post_hook
        def gradient_hook_factory(instance, parameter):
            original = original_gradient_hook(instance, parameter)
            parameter_name = next(name for name, value in instance.module.named_parameters()
                                  if value is parameter)
            def gradient_hook(*args):
                if parameter.grad is None and instance.ddp_config.overlap_grad_reduce:
                    diagnostic = dict(rank=dist.get_rank(), step=self.step,
                        parameter=parameter_name, shape=list(parameter.shape),
                        grad_added_to_main_grad=parameter.grad_added_to_main_grad,
                        zero_out_wgrad=getattr(parameter, 'zero_out_wgrad', False),
                        active_operation=self.active_operation,
                        vp_stage=parallel_state.get_virtual_pipeline_model_parallel_rank())
                    print('PARAMORPH_GRADIENT_DIAGNOSTIC ' + json.dumps(diagnostic), flush=True)
                    (self.output / f'rank{self.rank}.paramorph-gradient-diagnostic.json').write_text(
                        json.dumps(diagnostic, indent=2, allow_nan=False) + '\n')
                return original(*args)
            return gradient_hook
        DistributedDataParallel._make_backward_post_hook = gradient_hook_factory
        original_forward = schedules.forward_step
        signature = inspect.signature(original_forward)

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            values = signature.bind(*args, **kwargs).arguments
            vp = values.get('vp_stage') or 0
            mb = values.get('current_microbatch')
            if mb is None:
                mb = self.counters[(self.step, 'forward', vp)]
            self.counters[(self.step, 'forward', vp)] += 1
            identity = dict(vp_stage=vp, microbatch=mb)
            step_function = values.get('forward_step_func')
            if step_function is not None and self.request.get('paramorph_calibration', False):
                @functools.wraps(step_function)
                def measured_step(*step_args, **step_kwargs):
                    outcome = step_function(*step_args, **step_kwargs)
                    if not isinstance(outcome, tuple) or len(outcome) < 2 or not callable(outcome[1]):
                        return outcome
                    loss_function = outcome[1]

                    @functools.wraps(loss_function)
                    def measured_loss(*loss_args, **loss_kwargs):
                        return self.record('special_compute', 'loss', loss_function,
                            loss_args, loss_kwargs, dict(identity, component='loss'))

                    return (outcome[0], measured_loss, *outcome[2:])

                parameters = list(signature.parameters)
                position = parameters.index('forward_step_func')
                if position < len(args):
                    args = list(args)
                    args[position] = measured_step
                    args = tuple(args)
                else:
                    kwargs = dict(kwargs, forward_step_func=measured_step)
            result = self.record('forward_compute', 'region_forward', original_forward,
                                 args, kwargs, identity)
            if self.numerical_capture is not None:
                self.numerical_capture('forward', identity, result[0])
            tensors = result[0] if isinstance(result[0], list) else [result[0]]
            for tensor in tensors:
                if tensor is not None:
                    self.identities[id(tensor)] = identity
            return result

        original_backward = schedules.backward_step

        @functools.wraps(original_backward)
        def backward(input_tensor, output_tensor, output_tensor_grad, config):
            tensors = output_tensor if isinstance(output_tensor, list) else [output_tensor]
            tensor_ids = {id(tensor) for tensor in tensors if tensor is not None}
            ids = [self.identities[tensor_id] for tensor_id in tensor_ids]
            if not ids or any(item != ids[0] for item in ids):
                raise ValueError('Backward缺少对应Forward身份')
            result = self.record('backward_compute', 'region_backward', original_backward,
                (input_tensor, output_tensor, output_tensor_grad, config), {}, ids[0])
            if self.numerical_capture is not None:
                self.numerical_capture('dgrad', ids[0], result)
            for tensor_id in tensor_ids:
                del self.identities[tensor_id]
            return result

        schedules.forward_step = forward
        schedules.backward_step = backward
        original_table = schedules.get_schedule_table

        def table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
            expected = original_table(num_microbatches, num_model_chunks,
                                      microbatch_group_size_per_vp_stage)
            generated = self.request.get('paramorph_schedule_table')
            selected = [tuple(pair) for pair in generated] if generated is not None else expected
            if sorted(selected) != sorted(expected) or len(set(selected)) != len(expected):
                raise ValueError('ParaMorph计划必须覆盖每个microbatch和VP一次')
            if generated is not None and selected != expected:
                # 原生执行器的chunk切换和warmup依赖于分组约束。
                groups = self.request['group']
                for offset in range(0, num_microbatches, groups):
                    block = selected[offset * num_model_chunks:(offset + groups) * num_model_chunks]
                    if [vp for _, vp in block] != [vp for vp in range(num_model_chunks)
                        for _ in range(min(groups, num_microbatches - offset))]:
                        raise ValueError('计划违反原生VPP执行器的chunk分组约束')
            self.schedule_tables.append(selected)
            return selected

        schedules.get_schedule_table = table
        communicator = p2p_communication.P2PCommunicator
        original_communicate = communicator._communicate

        def communicate(instance, **kwargs):
            if self.step < 0:
                return original_communicate(instance, **kwargs)
            shape = kwargs['tensor_shape']
            identity = dict(shape=list(shape) if shape is not None else None,
                dtype=str(instance.config.pipeline_dtype),
                group_ranks=dist.get_process_group_ranks(instance.pp_group),
                send_next=kwargs['tensor_send_next'] is not None,
                send_prev=kwargs['tensor_send_prev'] is not None,
                recv_prev=kwargs['recv_prev'], recv_next=kwargs['recv_next'],
                completion_waited=kwargs.get('wait_on_reqs', True), messages=[])
            operations = [(identity['send_next'], self.rank, instance.next_rank, 'forward', 'send'),
                (identity['send_prev'], self.rank, instance.prev_rank, 'backward', 'send'),
                (identity['recv_prev'], instance.prev_rank, self.rank, 'forward', 'recv'),
                (identity['recv_next'], instance.next_rank, self.rank, 'backward', 'recv')]
            for enabled, source, destination, direction, role in operations:
                if enabled:
                    key = (self.step, source, destination, direction, role)
                    index = self.counters[key]
                    self.counters[key] += 1
                    identity['messages'].append(dict(source=source, destination=destination,
                        direction=direction, role=role,
                        message_id=f'{self.step}:{source}:{destination}:{direction}:{index}'))
            if self.memory_audit is not None:
                self.memory_audit.track_communication(
                    [kwargs['tensor_send_next'], kwargs['tensor_send_prev']], 'send')
            result = self.record('pp_p2p_exchange', 'pp_p2p_exchange', original_communicate,
                                 (instance,), kwargs, identity)
            if self.memory_audit is not None:
                self.memory_audit.track_communication(result[:2], 'receive')
                self.memory_audit.snapshot('pipeline_receive_published')
            return result

        communicator._communicate = communicate
        for name in ('forward_backward_no_pipelining', 'forward_backward_pipelining_with_interleaving',
                     'forward_backward_pipelining_without_interleaving'):
            original = getattr(schedules, name)

            def pipeline(*args, original_function=original, **kwargs):
                function = self.plan_executor or original_function
                if (self.request.get('paramorph_numerical_reference', False)
                    and self.step >= self.request['warmup']):
                    from numerical_reference import NumericalReference
                    function = functools.partial(NumericalReference(self).run,
                                                 function, original_function, args, kwargs)
                    result = self.record('pipeline', 'pipeline_update', function, (), {})
                else:
                    result = self.record('pipeline', 'pipeline_update', function, args, kwargs)
                return result

            setattr(schedules, name, pipeline)
        evidence = self.native.Evidence
        original_start, original_end = evidence.on_train_step_start, evidence.on_train_step_end
        original_train_start, original_train_end = evidence.on_train_start, evidence.on_train_end

        def train_start(callback, context):
            original_train_start(callback, context)
            self.output = callback.args.output
            if self.request.get('paramorph_memory_audit', False):
                from memory_audit import MemoryAudit
                self.memory_audit = MemoryAudit(self)
                self.memory_audit.snapshot('training_start')
            self.rank = dist.get_rank()
            self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self.ownership = json.loads((self.output / f'rank{self.rank}.ownership.json').read_text())
            self.register_parameter_storages(context.model if self.request.get(
                'execution_backend', 'megatron_ddp') == 'megatron_fsdp' else callback.models)
            original_exception_hook = sys.excepthook
            def failure_hook(error_type, error, traceback):
                if issubclass(error_type, torch.OutOfMemoryError):
                    free, capacity = torch.cuda.mem_get_info()
                    failure = dict(rank=self.rank, pp_rank=self.pp_rank, step=self.step,
                        active_operation=self.active_operation, active_region=self.active_region,
                        active_plan_task=self.active_plan_task, error_type=error_type.__name__,
                        error_message=str(error), allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved(), free_device_bytes=free,
                        device_capacity_bytes=capacity, request=self.request,
                        memory_statistics=torch.cuda.memory_stats(),
                        source='actual_uncaught_gpu_allocation_failure')
                    (self.output / f'rank{self.rank}.paramorph-failure.json').write_text(
                        json.dumps(failure, indent=2, allow_nan=False) + '\n')
                original_exception_hook(error_type, error, traceback)
            sys.excepthook = failure_hook
            if self.request.get('paramorph_execution_plan') is not None:
                from paramorph_executor import PlanExecutor
                self.plan_executor = PlanExecutor(self, self.request['paramorph_execution_plan'])
            executor = 'paramorph_task_dag' if self.plan_executor is not None else 'megatron_native'
            print('PARAMORPH_EXECUTOR ' + json.dumps(dict(rank=self.rank, executor=executor,
                decoder_chunks=self.request['decoder_chunks'],
                plan_sha256=self.plan_executor.digest if self.plan_executor is not None else None)), flush=True)
            if self.request.get('paramorph_link_calibration', False):
                from paramorph_communication_calibration import calibrate_pipeline_links
                calibrate_pipeline_links(self.output, self.request, callback.models)
            if self.request.get('paramorph_calibration', False):
                if self.request.get('paramorph_operator_calibration', False):
                    from operator_profiling import install_operator_profiling
                    install_operator_profiling(self, callback.models)
                layers = []
                for model_index, model in enumerate(callback.models):
                    for layer in model.decoder.layers:
                        parameters = list(layer.parameters())
                        gradients = [getattr(parameter, 'main_grad', None) for parameter in parameters]
                        layers.append(dict(layer_id=layer.layer_number - 1, vp_stage=model_index,
                            implementation=type(layer).__module__ + '.' + type(layer).__qualname__,
                            weight_bytes=sum(parameter.numel() * parameter.element_size() for parameter in parameters),
                            gradient_bytes=sum(gradient.numel() * gradient.element_size()
                                for gradient in gradients) if all(gradient is not None for gradient in gradients) else None,
                            parameters=[dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                                device=str(parameter.device),
                                expert_parallel=not getattr(parameter, 'allreduce', True))
                                for name, parameter in layer.named_parameters()]))
                static = dict(rank=self.rank, layers=layers, request=self.request,
                    allocator_limit_bytes=int(torch.cuda.get_per_process_memory_fraction()
                        * torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory),
                    allocated_bytes=torch.cuda.memory_allocated(),
                    device_capacity_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                    source='actual_loaded_layer_parameters_and_gradient_views')
                free, capacity = torch.cuda.mem_get_info()
                static['non_allocator_bytes'] = max(0, capacity - free - torch.cuda.memory_reserved())
                static['hardware'] = dict(device_name=torch.cuda.get_device_name(),
                    gpu_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    capability=list(torch.cuda.get_device_capability()))
                (self.output / f'rank{self.rank}.paramorph-static.json').write_text(
                    json.dumps(static, indent=2, allow_nan=False) + '\n')
            self.write_parameter_ledger(context.model)
            self.install_bucket_calibration(context.model)
            visited = set()
            for model in callback.models:
                config = model.config
                if id(config) in visited:
                    continue
                visited.add(id(config))
                finalize = config.finalize_model_grads_func
                if finalize is not None:
                    def finalize_gradients(*args, original=finalize, **kwargs):
                        return self.record('dp_reduce', 'gradient_finalize', original, args, kwargs)
                    config.finalize_model_grads_func = finalize_gradients
            if self.request.get('paramorph_calibration', False):
                for model_index, model in enumerate(callback.models):
                    for layer in model.decoder.layers:
                        original_layer = layer.forward
                        layer_id = layer.layer_number - 1

                        def layer_forward(*args, original=original_layer, identifier=layer_id,
                                          vp=model_index, **kwargs):
                            if self.active_region is None:
                                return original(*args, **kwargs)
                            phase = self.active_region['phase']
                            identity = dict(self.active_region, layer_id=identifier, vp_stage=vp)
                            operation = 'layer_forward' if phase == 'forward_compute' else 'layer_recompute_forward'
                            result = self.record('layer_compute', operation, original, args, kwargs, identity)
                            if self.request.get('paramorph_layer_backward_calibration', False) and torch.is_grad_enabled():
                                input_tensor = args[0] if args else kwargs['hidden_states']
                                output_tensor = result[0] if isinstance(result, tuple) else result
                                if not input_tensor.requires_grad or not output_tensor.requires_grad:
                                    raise ValueError('逐层Backward计时需要实际输入与输出梯度')
                                key = (identifier, vp)
                                def backward_start(gradient):
                                    if key in self.layer_backward_events:
                                        raise ValueError('逐层Backward事件重复开始')
                                    start = torch.cuda.Event(enable_timing=True)
                                    start.record()
                                    self.layer_backward_events[key] = dict(step=self.step,
                                        kind='layer_compute', operation='layer_backward', identity=identity,
                                        host_start_ns=time.perf_counter_ns(),
                                        allocated_before=torch.cuda.memory_allocated(), cuda_start=start)
                                    return gradient
                                def backward_end(gradient):
                                    event = self.layer_backward_events.pop(key)
                                    event['identity']['backward_input_gradient_observed'] = gradient is not None
                                    end = torch.cuda.Event(enable_timing=True)
                                    end.record()
                                    event.update(cuda_end=end, host_end_ns=time.perf_counter_ns(),
                                        allocated_after=torch.cuda.memory_allocated())
                                    self.events.append(event)
                                    return gradient
                                output_tensor.register_hook(backward_start)
                                input_tensor.register_hook(backward_end)
                            return result

                        layer.forward = layer_forward
            if self.request.get('paramorph_calibration', False):
                group = dist.new_group(backend='gloo', timeout=timedelta(seconds=60))
                measurements = {0: dict(offset_ns=0, uncertainty_ns=0,
                    evidence_id=self.request['run_id'] + ':clock:rank0')}
                for peer in range(1, dist.get_world_size()):
                    samples = []
                    for repetition in range(5):
                        token = torch.tensor([repetition], dtype=torch.int64)
                        reply = torch.empty(1, dtype=torch.int64)
                        if self.rank == 0:
                            start = time.perf_counter_ns()
                            dist.send(token, peer, group=group)
                            dist.recv(reply, peer, group=group)
                            end = time.perf_counter_ns()
                            samples.append(dict(offset_ns=(start + end) // 2 - reply.item(),
                                uncertainty_ns=(end - start) // 2,
                                evidence_id=f'{self.request["run_id"]}:clock:{peer}:{repetition}'))
                        elif self.rank == peer:
                            dist.recv(token, 0, group=group)
                            reply[0] = time.perf_counter_ns()
                            dist.send(reply, 0, group=group)
                    if self.rank == 0:
                        measurements[peer] = min(samples, key=lambda item: item['uncertainty_ns'])
                result = [measurements if self.rank == 0 else None]
                dist.broadcast_object_list(result, src=0, group=group)
                self.clock_calibration = result[0]
                dist.destroy_process_group(group)
            else:
                self.clock_calibration = None
            optimizer = context.optimizer
            self.optimizer = optimizer
            self.install_optimizer_cost_calibration(optimizer)
            original_step = optimizer.step

            def optimizer_step(*args, **kwargs):
                if self.numerical_optimizer_audit is None:
                    result = self.record('optimizer_step', 'optimizer_step', original_step, args, kwargs)
                else:
                    result = self.numerical_optimizer_audit.verify_optimizer(original_step, args, kwargs)
                if self.plan_executor is not None:
                    self.plan_executor.finish_optimizer()
                return result

            optimizer.step = optimizer_step

        def step_start(callback, context):
            original_start(callback, context)
            self.step = len(callback.rows)
            if (self.request.get('paramorph_nsys_trace', False)
                    and self.step >= callback.args.warmup):
                torch.cuda.nvtx.range_push('FRONTIER_NSYS:' + json.dumps(dict(
                    kind='UPDATE', rank=dist.get_rank(), iteration=self.step + 1),
                    separators=(',', ':'), sort_keys=True))
            if self.request.get('paramorph_calibration', False):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.cuda_origins[self.step] = torch.cuda.Event(enable_timing=True)
                origin_before = time.perf_counter_ns()
                self.cuda_origins[self.step].record()
                self.cuda_origins[self.step].synchronize()
                origin_after = time.perf_counter_ns()
                self.cuda_origin_host_anchors[self.step] = dict(
                    monotonic_ns=(origin_before + origin_after) // 2,
                    uncertainty_ns=(origin_after - origin_before) // 2)
                self.iteration_peak = torch.cuda.memory_allocated()
            trace_steps = self.request.get('paramorph_trace_steps', 1)
            if trace_steps < 1 or trace_steps > callback.args.measured:
                raise ValueError('CUDA Trace 步数必须位于 1 到 measured 之间')
            if (self.request.get('paramorph_trace', False)
                    and self.step == callback.args.warmup + callback.args.measured - trace_steps):
                self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False,
                    acc_events=True)
                self.profiler.__enter__()
            self.window_start = time.perf_counter_ns()
            self.persistent = torch.cuda.memory_allocated()

        def step_end(callback, context):
            if self.request.get('paramorph_calibration', False):
                torch.cuda.synchronize()
            if (self.request.get('paramorph_nsys_trace', False)
                    and self.step >= callback.args.warmup):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            end = time.perf_counter_ns()
            self.windows.append(dict(step=self.step, host_start_ns=self.window_start,
                host_end_ns=end, persistent_bytes=self.persistent,
                peak_allocated_bytes=max(self.iteration_peak, torch.cuda.max_memory_allocated())))
            if self.identities:
                raise ValueError('迭代结束后仍存在未执行Backward的Forward')
            if self.step == 0:
                self.save_partial_calibration()
            original_end(callback, context)

        def train_end(callback, context):
            self.write_bucket_calibration()
            if self.memory_audit is not None:
                if self.request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
                    self.memory_audit.inventory_fsdp(context.model, self.optimizer)
                else:
                    self.memory_audit.inventory(callback.models, self.optimizer)
                self.memory_audit.snapshot('training_end')
                self.memory_audit.save()
            if self.request.get('paramorph_calibration', False):
                static_path = self.output / f'rank{self.rank}.paramorph-static.json'
                static = json.loads(static_path.read_text())
                if self.request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
                    state_rows = self.fsdp_layer_state_rows(context.model, self.optimizer)
                    for layer in static['layers']:
                        layer.update(state_rows[layer['layer_id']])
                        layer['state_bytes'] = sum(state_rows[layer['layer_id']].values())
                    static['state_source'] = 'actual_fsdp_local_weight_gradient_buffers_and_optimizer_states'
                else:
                    from native_frontier_checkpoint import NativeMegatronTrainingCheckpoint
                    state_bindings = NativeMegatronTrainingCheckpoint(callback.models, self.optimizer).runtime_tensors()
                    state_rows = {}
                    for vp, model in enumerate(callback.models):
                        names = {id(parameter): name for name, parameter in model.named_parameters()}
                        for layer in model.decoder.layers:
                            total = 0
                            for parameter in layer.parameters():
                                identity = f'model:{vp}:{names[id(parameter)]}:'
                                states = [value for key, value in state_bindings.items()
                                    if key.startswith('optimizer:') and identity in key and not key.endswith(':gradient')]
                                # Distributed Optimizer允许当前DP Rank对某个参数不持有本地状态分片。
                                seen = set()
                                for value in states:
                                    if value.device.type == 'cuda' and value.data_ptr() != parameter.data_ptr():
                                        signature = (value.data_ptr(), value.numel(), value.dtype)
                                        if signature not in seen:
                                            total += value.numel() * value.element_size()
                                            seen.add(signature)
                            state_rows[layer.layer_number - 1] = total
                    for layer in static['layers']:
                        layer['optimizer_bytes'] = state_rows[layer['layer_id']]
                        layer['state_bytes'] = layer['weight_bytes'] + layer['gradient_bytes'] + layer['optimizer_bytes']
                    static['state_source'] = 'actual_parameter_gradient_and_optimizer_tensor_views'
                static_path.write_text(json.dumps(static, indent=2, allow_nan=False) + '\n')
            original_train_end(callback, context)
            if self.plan_executor is not None:
                self.plan_executor.communication.save()
            if self.request.get('paramorph_calibration', False):
                measurement_path = self.output / f'rank{self.rank}.measurements.json'
                measurement = json.loads(measurement_path.read_text())
                measurement['allocated_gib'] = max(window['peak_allocated_bytes'] for window in self.windows
                    if window['step'] >= self.request['warmup']) / 2**30
                measurement['peak_memory_source'] = 'actual_operation_and_iteration_allocator_peaks'
                measurement_path.write_text(json.dumps(measurement, indent=2, allow_nan=False) + '\n')
            if self.profiler is not None:
                self.profiler.__exit__(None, None, None)
                self.profiler.export_chrome_trace(str(self.output / f'rank{self.rank}.paramorph-trace.json'))
            torch.cuda.synchronize()
            operator_rows = []
            for event in self.operator_backward_events:
                operator_rows.append(dict(step=event['step'], operation=event['operation'],
                    identity=event['identity'], milliseconds=event['cuda_start'].elapsed_time(event['cuda_end']),
                    source=event['source']))
            if self.request.get('paramorph_operator_calibration', False):
                (self.output / f'rank{self.rank}.paramorph-operators.json').write_text(
                    json.dumps(dict(rank=self.rank, request=self.request, backward=operator_rows),
                               indent=2, allow_nan=False) + '\n')
            for event in self.events:
                start, end = event.pop('cuda_start'), event.pop('cuda_end')
                event['cuda_ms'] = start.elapsed_time(end) if start is not None else None
                if start is not None:
                    event['cuda_start_ms'] = self.cuda_origins[event['step']].elapsed_time(start)
                    event['cuda_end_ms'] = self.cuda_origins[event['step']].elapsed_time(end)
            vp_count = self.request['vpp']
            microbatches = self.request['global_batch'] // (self.request['dp'] * self.request['micro_batch'])
            for window in self.windows:
                for vp in range(vp_count):
                    for kind in ('forward_compute', 'backward_compute'):
                        rows = [event for event in self.events if event['step'] == window['step']
                                and event['kind'] == kind and event['identity']['vp_stage'] == vp]
                        if len(rows) != microbatches or sorted(event['identity']['microbatch']
                            for event in rows) != list(range(microbatches)):
                            raise ValueError('实际Forward/Backward没有完整覆盖执行计划')
            payload = dict(rank=self.rank, pp_rank=self.pp_rank, events=self.events,
                windows=self.windows, ownership=self.ownership, schedule_tables=self.schedule_tables,
                forward_backward_coverage_verified=True, request=self.request,
                executed_plans=self.executed_plans,
                clock_domain='per_rank_monotonic',
                clock_calibration=self.clock_calibration,
                cuda_origin_host_anchors=self.cuda_origin_host_anchors,
                cross_rank_clock_calibrated=self.clock_calibration is not None)
            (self.output / f'rank{self.rank}.paramorph-cost.json').write_text(
                json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
            if self.executed_plans:
                print('PARAMORPH_TASK_ORDER ' + json.dumps(dict(rank=self.rank,
                    iterations=self.executed_plans)), flush=True)

        evidence.on_train_start = train_start
        evidence.on_train_step_start = step_start
        evidence.on_train_step_end = step_end
        evidence.on_train_end = train_end


def install(native, request):
    recorder = RuntimeEvidence(native, request)
    recorder.install()
    return recorder
