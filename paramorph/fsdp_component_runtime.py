from collections import Counter
import functools
import inspect
import json
import time

import torch
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import GradReducePipeline

from paramorph_runtime_base import RuntimeEvidence


def install(native, request):
    assert request['execution_backend'] == 'megatron_fsdp'
    assert request.get('paramorph_fsdp_component_services') is True
    assert request.get('fsdp_prefetch', False) is False
    isolated = request.get('paramorph_fsdp_component_isolation', False)
    recorder = RuntimeEvidence(native, request)
    recorder.install()
    pending, counters = [], Counter()
    active = None
    original_reduce = GradReducePipeline.reduce_gradients
    original_wait = GradReducePipeline.wait_for_previous_grad_reduce
    original_gather = MegatronFSDP.all_gather_and_wait_parameters_ready

    def measured_wait(operation, function, arguments, keywords):
        if recorder.step < 0:
            return function(*arguments, **keywords)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter_ns()
        allocated_before = torch.cuda.memory_allocated()
        start.record()
        result = function(*arguments, **keywords)
        end.record()
        recorder.events.append(dict(step=recorder.step, kind='fsdp_service_wait', operation=operation,
            identity=dict(recorder.active_region or {}), host_start_ns=before,
            host_end_ns=time.perf_counter_ns(), cuda_start=start, cuda_end=end,
            allocated_before=allocated_before, allocated_after=torch.cuda.memory_allocated(),
            cost_scope='explicit_fsdp_collective_wait_excluded_from_compute_service'))
        counters[operation] += 1
        return result

    @functools.wraps(original_wait)
    def wait(pipeline, *arguments, **keywords):
        if not pipeline.grad_reduce_queue:
            return original_wait(pipeline, *arguments, **keywords)
        return measured_wait('gradient_reduce_wait', original_wait, (pipeline, *arguments), keywords)

    @functools.wraps(original_gather)
    def gather(model, *arguments, **keywords):
        return measured_wait('parameter_gather_wait', original_gather, (model, *arguments), keywords)

    def flush():
        if not pending:
            return
        calls = list(pending)
        pending.clear()
        pipelines = {}
        for pipeline, params, arguments, keywords in calls:
            original_reduce(pipeline, params, *arguments, **keywords)
            pipelines[id(pipeline)] = pipeline
        for pipeline in pipelines.values():
            pipeline.wait_for_previous_grad_reduce(0)
            assert not pipeline.grad_reduce_queue
        counters['completed_boundary_flushes'] += 1

    @functools.wraps(original_reduce)
    def reduce(pipeline, params, *arguments, **keywords):
        params = list(params)
        counters['gradient_parameters'] += len(params)
        if isolated and active is not None:
            pending.append((pipeline, params, arguments, keywords))
            counters['deferred_gradient_calls'] += 1
            return None
        result = original_reduce(pipeline, params, *arguments, **keywords)
        if isolated:
            pipeline.wait_for_previous_grad_reduce(0)
        return result

    GradReducePipeline.reduce_gradients = reduce
    GradReducePipeline.wait_for_previous_grad_reduce = wait
    MegatronFSDP.all_gather_and_wait_parameters_ready = gather
    original_register_hook = torch.Tensor.register_hook

    @functools.wraps(original_register_hook)
    def register_hook(tensor, hook):
        qualified = getattr(hook, '__qualname__', '')
        selected = getattr(hook, '__module__', '') == 'paramorph_runtime_base'
        if selected and qualified.endswith('layer_forward.<locals>.backward_start'):
            @functools.wraps(hook)
            def start(gradient):
                nonlocal active
                assert active is None
                if isolated:
                    flush()
                active = inspect.getclosurevars(hook).nonlocals['key']
                counters['layer_backward_starts'] += 1
                return hook(gradient)
            return original_register_hook(tensor, start)
        if selected and qualified.endswith('layer_forward.<locals>.backward_end'):
            @functools.wraps(hook)
            def end(gradient):
                nonlocal active
                assert active == inspect.getclosurevars(hook).nonlocals['key']
                result = hook(gradient)
                active = None
                counters['layer_backward_ends'] += 1
                if isolated:
                    flush()
                return result
            return original_register_hook(tensor, end)
        return original_register_hook(tensor, hook)

    torch.Tensor.register_hook = register_hook
    original_record = recorder.record

    @functools.wraps(original_record)
    def record(kind, operation, function, arguments, keywords, identity=None):
        if isolated and active is None and kind in ('layer_compute', 'forward_compute', 'backward_compute'):
            flush()
        if kind == 'optimizer_step':
            assert active is None and not pending
        return original_record(kind, operation, function, arguments, keywords, identity)

    recorder.record = record
    original_end = native.Evidence.on_train_end

    def end_training(callback, context):
        assert active is None and not pending
        assert counters['layer_backward_starts'] == counters['layer_backward_ends'] > 0
        if isolated:
            assert counters['deferred_gradient_calls'] > 0 and counters['completed_boundary_flushes'] > 0
        original_end(callback, context)
        evidence = dict(protocol='native_fsdp_unit_boundary_services_v1', isolated=isolated,
            counters=dict(counters), pending_gradient_calls=0,
            overlap_grad_reduce=context.state.cfg.ddp.overlap_grad_reduce,
            overlap_param_gather=context.state.cfg.ddp.overlap_param_gather)
        assert evidence['overlap_grad_reduce'] is True and evidence['overlap_param_gather'] is True
        path = recorder.output/f'rank{recorder.rank}.paramorph-fsdp-services.json'
        path.write_text(json.dumps(evidence, ensure_ascii=False)+'\n')

    native.Evidence.on_train_end = end_training
    return recorder
