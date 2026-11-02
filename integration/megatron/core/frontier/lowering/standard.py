import torch
import torch.distributed as dist

from megatron.core.frontier.lowering.base import LoweringContext, LOWERING_REGISTRY
from megatron.core.frontier.state import ObligationKind, PlacementKind


class StandardNCCLLowering:
    lowering_id = 'standard_nccl'

    def supports(self, transition, context: LoweringContext):
        source, target = transition.source, transition.target
        obligation = source.obligation
        if (obligation is None
                or obligation.kind not in (ObligationKind.REDUCE, ObligationKind.REPARTITION)
                or source.mesh_id != target.mesh_id or source.logical_shape != target.logical_shape
                or source.dtype != target.dtype or obligation.target != target.placement):
            return False
        key = context.schema.pg_key
        if not key or context.process_groups is None or key not in context.process_groups:
            return False
        if obligation.contributor_group != key:
            return False
        if source.placement.kind is PlacementKind.SHARD:
            return source.placement.dim == 0 and target.placement.kind is PlacementKind.REPLICATE and obligation.kind is ObligationKind.REPARTITION
        if source.placement.kind is not PlacementKind.PARTIAL:
            return False
        return target.placement.kind is PlacementKind.REPLICATE or (
            target.placement.kind is PlacementKind.SHARD and target.placement.dim == 0)

    def _group(self, context):
        key = context.schema.pg_key
        if not key or context.process_groups is None or key not in context.process_groups:
            raise ValueError('NCCL lowering requires an explicitly bound process group')
        group = context.process_groups[key]
        if not dist.is_initialized() or group is None:
            raise ValueError('NCCL lowering requires an initialized concrete process group')
        if dist.get_backend(group) != 'nccl':
            raise ValueError('The actual collective process group must use NCCL')
        actual = tuple(dist.get_process_group_ranks(group))
        if actual != context.schema.source_ranks:
            raise ValueError('Actual collective ranks differ from the compiled tensor contract')
        if context.schema.destination_ranks != actual:
            raise ValueError('Same-mesh collective output ranks differ from the actual process group')
        obligation = context.schema.source_state.obligation
        if (obligation is None or obligation.contributor_group != key
                or context.schema.source_state.placement.kind not in (PlacementKind.PARTIAL, PlacementKind.SHARD)
                or context.schema.source_state.mesh_id != context.schema.target_state.mesh_id
                or obligation.target != context.schema.target_state.placement):
            raise ValueError('NCCL lowering requires a complete same-mesh reduction contract')
        source = context.schema.source_state.placement
        if source.kind is PlacementKind.SHARD and (source.dim != 0
                or context.schema.target_state.placement.kind is not PlacementKind.REPLICATE
                or obligation.kind is not ObligationKind.REPARTITION):
            raise ValueError('AllGather requires an explicit dimension-zero shard and replicated target')
        return group

    def forward(self, tensor, context):
        group = self._group(context)
        if not tensor.is_cuda or str(tensor.dtype).removeprefix('torch.') != context.schema.source_state.dtype:
            raise ValueError('NCCL forward requires an actual CUDA tensor with the declared precision')
        if tuple(tensor.shape) != context.schema.tensor_shape:
            raise ValueError('Actual source tensor shape differs from its compiled collective contract')
        target = context.schema.target_state.placement
        if context.schema.source_state.placement.kind is PlacementKind.SHARD:
            result = torch.empty((tensor.shape[0] * dist.get_world_size(group), *tensor.shape[1:]),
                dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(result, tensor.contiguous(), group=group)
            return result
        if target.kind is PlacementKind.REPLICATE:
            result = tensor.contiguous().clone()
            dist.all_reduce(result, group=group)
            return result
        if target.kind is not PlacementKind.SHARD or target.dim != 0:
            raise ValueError('This NCCL reduction supports replicated output or a dimension-zero shard')
        world = dist.get_world_size(group)
        if tensor.shape[0] % world:
            raise ValueError('ReduceScatter input must divide evenly over its actual process group')
        result = torch.empty((tensor.shape[0] // world, *tensor.shape[1:]),
            dtype=tensor.dtype, device=tensor.device)
        dist.reduce_scatter_tensor(result, tensor.contiguous(), group=group)
        return result

    def backward(self, gradient, context):
        group = self._group(context)
        if not gradient.is_cuda:
            raise ValueError('NCCL backward requires an actual CUDA gradient tensor')
        target = context.schema.target_state.placement
        if context.schema.source_state.placement.kind is PlacementKind.SHARD:
            shape = context.schema.tensor_shape
            expected = (shape[0] * dist.get_world_size(group), *shape[1:])
            if tuple(gradient.shape) != expected:
                raise ValueError('AllGather adjoint gradient differs from its complete replicated output')
            result = torch.empty(shape, dtype=gradient.dtype, device=gradient.device)
            dist.reduce_scatter_tensor(result, gradient.contiguous(), group=group)
            return result
        if target.kind is PlacementKind.REPLICATE:
            if tuple(gradient.shape) != context.schema.tensor_shape:
                raise ValueError('AllReduce adjoint gradient differs from the source logical shape')
            result = gradient.contiguous().clone()
            dist.all_reduce(result, group=group)
            return result
        if target.kind is not PlacementKind.SHARD or target.dim != 0:
            raise ValueError('Unsupported NCCL reduction adjoint')
        world = dist.get_world_size(group)
        expected = (context.schema.tensor_shape[0] // world, *context.schema.tensor_shape[1:])
        if tuple(gradient.shape) != expected:
            raise ValueError('ReduceScatter adjoint gradient differs from its actual local shard')
        result = torch.empty(context.schema.tensor_shape, dtype=gradient.dtype, device=gradient.device)
        dist.all_gather_into_tensor(result, gradient.contiguous(), group=group)
        return result

    def profile_key(self, transition, context):
        group = self._group(context)
        return (self.lowering_id, transition.source.dtype, context.schema.tensor_shape,
            transition.source.placement, transition.target.placement,
            tuple(dist.get_process_group_ranks(group)))


class StandardNCCLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, context):
        ctx.context = context
        return StandardNCCLLowering().forward(tensor, context)

    @staticmethod
    def backward(ctx, gradient):
        return StandardNCCLLowering().backward(gradient, ctx.context), None


LOWERING_REGISTRY.register(StandardNCCLLowering())
