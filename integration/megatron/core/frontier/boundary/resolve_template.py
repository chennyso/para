from dataclasses import dataclass
from math import prod

from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram, BufferSpecification, ExecutableWitness
from megatron.core.frontier.ir.contract import PlacementKind
from megatron.core.frontier.runtime.resolve_handoff import PeerSlice, ResolveHandoffPlan


@dataclass(frozen=True)
class CompiledHandoffTemplate:
    source_ranks: tuple[int, ...]
    destination_ranks: tuple[int, ...]
    full_shape: tuple[int, ...]
    shard_dimension: int
    group_key: str
    dtype: str
    witness: ExecutableWitness

    @property
    def shard_shape(self):
        shape = list(self.full_shape)
        shape[self.shard_dimension] //= len(self.destination_ranks)
        return tuple(shape)

    def plan_for_rank(self, rank):
        length = self.shard_shape[self.shard_dimension]
        if rank in self.source_ranks:
            slices = tuple(PeerSlice(peer, index * length, (index + 1) * length)
                           for index, peer in enumerate(self.destination_ranks))
            return ResolveHandoffPlan(slices, (), self.full_shape, self.shard_dimension, self.group_key)
        if rank in self.destination_ranks:
            return ResolveHandoffPlan((), self.source_ranks, self.shard_shape, self.shard_dimension, self.group_key)
        raise ValueError('rank is absent from the compiled handoff witness')

    def instantiate(self, rank, process_group, device):
        import torch
        import torch.distributed as dist

        from megatron.core.frontier.runtime.buffer_pool import BoundaryBufferPool
        from megatron.core.frontier.runtime.resolve_handoff import ResolveOnHandoffRuntime

        if dist.get_process_group_ranks(process_group) != sorted(self.source_ranks + self.destination_ranks):
            raise ValueError('runtime process group differs from compiled tensor ownership')
        plan = self.plan_for_rank(rank)
        count = len(plan.send_slices) if plan.send_slices else len(plan.recv_sources)
        pool = BoundaryBufferPool()
        pool.register_slot(0, count, self.shard_shape, getattr(torch, self.dtype), device)
        return ResolveOnHandoffRuntime(plan, process_group, pool)


def compile_resolve_handoff(boundary_id, producer_region, consumer_region, source_contract,
                            destination_contract, source_ranks, destination_ranks, group_key):
    source_ranks, destination_ranks = tuple(source_ranks), tuple(destination_ranks)
    if not source_ranks or not destination_ranks or set(source_ranks) & set(destination_ranks):
        raise ValueError('handoff requires distinct source and destination rank sets')
    if len(set(source_ranks)) != len(source_ranks) or len(set(destination_ranks)) != len(destination_ranks):
        raise ValueError('handoff rank ownership contains duplicates')
    if source_contract.placement.kind is not PlacementKind.PARTIAL or source_contract.placement.reduce_op != 'sum':
        raise ValueError('ResolveOnHandoff requires partial additive contributions')
    if destination_contract.placement.kind is not PlacementKind.SHARD or destination_contract.obligation is not None:
        raise ValueError('ResolveOnHandoff requires a fully resolved consumer shard')
    if (source_contract.logical_value != destination_contract.logical_value
            or source_contract.logical_shape != destination_contract.logical_shape
            or source_contract.dtype != destination_contract.dtype):
        raise ValueError('source and consumer must observe the same logical value and precision')
    if (source_contract.presence_policy != 'required' or destination_contract.presence_policy != 'required'
            or not source_contract.gradient_eligible or not destination_contract.gradient_eligible):
        raise ValueError('this lowering requires a present differentiable training value')
    if source_contract.logical_slice is not None or destination_contract.logical_slice is not None:
        raise ValueError('compiled projection requires complete logical coordinates before rank partitioning')
    if source_contract.dtype != 'float32':
        raise ValueError('this capability requires the verified float32 GPU equivalence profile')
    dimension = destination_contract.placement.dim
    shape = source_contract.logical_shape
    if dimension != 0 or any(not isinstance(value, int) for value in shape):
        raise ValueError('this lowering requires a concrete verified projection dimension')
    if shape[dimension] % len(destination_ranks):
        raise ValueError('destination shards must divide the complete logical tensor')
    shard_shape = list(shape)
    shard_shape[dimension] //= len(destination_ranks)
    shard_shape = tuple(shard_shape)
    length = shard_shape[dimension]
    maps = tuple((source, destination, dimension, index * length, (index + 1) * length)
                 for source in source_ranks for index, destination in enumerate(destination_ranks))
    buffers = []
    for rank in source_ranks + destination_ranks:
        count = len(destination_ranks) if rank in source_ranks else len(source_ranks)
        buffers.extend(BufferSpecification(rank, f'projection:{index}', shard_shape, source_contract.dtype, 0)
                       for index in range(count))
        buffers.append(BufferSpecification(rank, 'output', shape if rank in source_ranks else shard_shape,
                                            source_contract.dtype, 1))
    witness = ExecutableWitness(((group_key, tuple(sorted(source_ranks + destination_ranks))),), (), maps,
                                 ('SOURCE_PROJECT', 'GROUPED_PP_SEND', 'DESTINATION_SUM'),
                                 ('DESTINATION_GRAD_SEND', 'SOURCE_ADJOINT_ASSEMBLY'), tuple(buffers),
                                 ('static_process_group', 'single_inflight_value_per_runtime',
                                  'retain_destination_until_backward', 'cuda_readiness_event'), 'resolve_on_handoff')
    op = BoundaryOp(f'{boundary_id}:resolve_on_handoff', BoundaryOpKind.FUSED,
                    source_contract, destination_contract, 'resolve_on_handoff')
    program = BoundaryProgram(boundary_id, producer_region, consumer_region, source_contract, destination_contract,
                               (op,), transferred_bytes=prod(shape) * 4 * len(source_ranks),
                               executable_witness_key=witness.identifier(), executable_witness=witness)
    return program, CompiledHandoffTemplate(source_ranks, destination_ranks, tuple(shape), dimension,
                                           group_key, source_contract.dtype, witness)
