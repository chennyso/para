import functools
import inspect

import torch


def install_operator_profiling(evidence, models):
    for vp, model in enumerate(models):
        layer_ids = {id(module): layer.layer_number - 1 for layer in model.decoder.layers
                     for module in layer.modules()}
        for name, module in model.named_modules():
            if not name:
                continue
            component = name.rsplit('.', 1)[-1]
            categories = dict(linear_qkv='attention_projection', linear_proj='attention_projection',
                core_attention='attention_core', linear_fc1='mlp_gemm', linear_fc2='mlp_gemm',
                router='router', experts='expert_grouped_gemm',
                output_layer='output_projection', final_layernorm='final_layernorm')
            if component not in categories:
                continue
            original = module.forward
            signature = inspect.signature(original)
            category = categories[component]
            implementation = type(module).__module__ + '.' + type(module).__name__
            parameter_shapes = [dict(name=key, shape=list(parameter.shape), dtype=str(parameter.dtype))
                                for key, parameter in module.named_parameters()]
            logical_layer = layer_ids.get(id(module))

            @functools.wraps(original)
            def forward(*args, original=original, signature=signature, module_name=name,
                        vp_stage=vp, operation=category, implementation=implementation,
                        parameter_shapes=parameter_shapes, logical_layer=logical_layer, **kwargs):
                if evidence.active_region is None:
                    return original(*args, **kwargs)
                values = signature.bind(*args, **kwargs).arguments
                tensors = [value for value in values.values() if isinstance(value, torch.Tensor)]
                identity = dict(evidence.active_region, module_name=module_name, vp_stage=vp_stage,
                    implementation=implementation,
                    inputs=[dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                 stride=list(tensor.stride()), contiguous=tensor.is_contiguous()) for tensor in tensors],
                    phase='recomputation' if evidence.active_region['phase'] == 'backward_compute' else 'forward',
                    cost_scope='actual_module_composite_with_its_internal_operations')
                identity['parameter_shapes'] = parameter_shapes
                identity['granularity'] = 'module_composite'
                identity['features'] = {f'input_{index}_axis_{axis}': value for index, tensor in enumerate(tensors)
                    for axis, value in enumerate(tensor.shape)}
                identity['features'].update({f'parameter_{index}_axis_{axis}': value
                    for index, parameter in enumerate(parameter_shapes) for axis, value in enumerate(parameter['shape'])})
                identity['layer_id'] = logical_layer
                token_counts = values.get('tokens_per_expert')
                if isinstance(token_counts, torch.Tensor):
                    identity['expert_token_counts'] = token_counts.detach().cpu().tolist()
                    identity['expert_token_source'] = 'actual_expert_dispatch_input'
                    identity['features'].update({f'expert_{index}_tokens': count
                        for index, count in enumerate(identity['expert_token_counts'])})
                result = evidence.record('operator_compute', operation, original, args, kwargs, identity)
                outputs = result if isinstance(result, (tuple, list)) else [result]
                if operation == 'router':
                    routing = [tensor for tensor in outputs if isinstance(tensor, torch.Tensor)
                               and tensor.dtype == torch.bool and tensor.ndim == 2]
                    if len(routing) == 1:
                        identity['expert_token_counts'] = routing[0].sum(dim=0).cpu().tolist()
                        identity['expert_token_source'] = 'actual_router_routing_map'
                inputs_with_grad = [tensor for tensor in tensors if tensor.requires_grad]
                outputs_with_grad = [tensor for tensor in outputs
                                     if isinstance(tensor, torch.Tensor) and tensor.requires_grad]
                if torch.is_grad_enabled() and len(inputs_with_grad) == 1 and len(outputs_with_grad) == 1:
                    timing = {}

                    def start_backward(gradient):
                        start = torch.cuda.Event(enable_timing=True)
                        start.record()
                        timing['start'] = start
                        return gradient

                    def finish_backward(gradient):
                        end = torch.cuda.Event(enable_timing=True)
                        end.record()
                        evidence.operator_backward_events.append(dict(step=evidence.step,
                            operation=operation, identity=identity, cuda_start=timing['start'], cuda_end=end,
                            source='actual_module_output_to_input_gradient_cuda_events'))
                        return gradient

                    outputs_with_grad[0].register_hook(start_backward)
                    inputs_with_grad[0].register_hook(finish_backward)
                return result

            module.forward = forward
