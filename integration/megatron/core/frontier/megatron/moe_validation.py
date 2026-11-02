from collections import Counter

import torch.distributed as dist


def install_region_routing_validation(native_module):
    from megatron.core import parallel_state

    original_class = native_module.MoEExecutionValidation

    class RegionMoEExecutionValidation(original_class):
        def __init__(self, request, output, models):
            self.request, self.output, self.rank = request, output, dist.get_rank()
            self.step = None
            self.routes, self.communication, self.handles = [], [], []
            self.counts = Counter()
            self.coordinates = {'tp': parallel_state.get_tensor_model_parallel_rank(),
                                'dp': parallel_state.get_data_parallel_rank(),
                                'cp': parallel_state.get_context_parallel_rank()}
            for model in models:
                actual = []
                for layer in model.decoder.layers:
                    kind = getattr(layer, 'frontier_region_kind', None)
                    if kind == 'ATTENTION':
                        if 'mlp' in dict(layer.named_children()):
                            raise ValueError('attention-only region retains an unregistered MoE module')
                        continue
                    if kind == 'MOE_DISPATCH_EXPERT_COMBINE':
                        if 'router' in dict(layer.mlp.named_children()):
                            raise ValueError('expert consumer still owns the relocated router')
                        continue
                    actual.append(layer.layer_number - 1)
                    self.handles.append(layer.mlp.router.register_forward_hook(self.router_hook(layer.layer_number - 1)))
                expected = sum(region.endswith(':ROUTER') for region in model.decoder.frontier_region_ids)
                if len(actual) != expected or len(set(actual)) != expected:
                    raise ValueError('actual routing hooks differ from compiled semantic router ownership')
            self.original_alltoall = dist.all_to_all_single

            def alltoall(output_tensor, input_tensor, *args, **kwargs):
                group = kwargs.get('group', args[2] if len(args) > 2 else None)
                result = self.original_alltoall(output_tensor, input_tensor, *args, **kwargs)
                if self.step is not None:
                    self.communication.append({'group': dist.get_process_group_ranks(group),
                                               'input_elements': input_tensor.numel(),
                                               'output_elements': output_tensor.numel(),
                                               'dtype': str(input_tensor.dtype),
                                               'async_op': kwargs.get('async_op', args[3] if len(args) > 3 else False)})
                return result

            dist.all_to_all_single = alltoall

    native_module.MoEExecutionValidation = RegionMoEExecutionValidation
