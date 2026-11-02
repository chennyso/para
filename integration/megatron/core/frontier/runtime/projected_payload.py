from dataclasses import asdict, dataclass
from math import prod

import torch
import torch.distributed as dist

from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram, BufferSpecification, ExecutableWitness
from megatron.core.frontier.ir.contract import ObligationKind, PendingObligation, Placement, PlacementKind, ValueContract
from megatron.core.frontier.runtime.buffer_pool import BoundaryBufferPool


def contract_metadata(contract):
    result = asdict(contract)
    result['placement']['kind'] = contract.placement.kind.name
    if contract.obligation is not None:
        result['obligation']['kind'] = contract.obligation.kind.name
        result['obligation']['target']['kind'] = contract.obligation.target.kind.name
    return result


@dataclass(frozen=True)
class ProjectedPayloadPlan:
    source_ranks: tuple[int, ...]
    destination_ranks: tuple[int, ...]
    shard_shape: tuple[int, ...]
    lifetime_capacity: int
    microbatches: int
    logical_layer: int
    variant: str
    dtype: str = 'bfloat16'
    reduction_ring: tuple[int, ...] = ()

    def __post_init__(self):
        if (len(self.source_ranks) != len(self.destination_ranks) or not self.source_ranks
                or set(self.source_ranks) & set(self.destination_ranks)
                or len(set(self.source_ranks + self.destination_ranks)) != 2 * len(self.source_ranks)):
            raise ValueError('projected payload requires disjoint equally sized actual TP domains')
        if len(self.shard_shape) != 3 or min(self.shard_shape) < 1 or min(self.lifetime_capacity, self.microbatches) < 1:
            raise ValueError('projected payload requires concrete dimensions and a compiled lifetime capacity')
        if self.variant not in ('canonical', 'project') or self.dtype != 'bfloat16':
            raise ValueError('this model capability requires its registered BF16 canonical or PROJECT program')
        if self.variant == 'project' and (len(self.reduction_ring) != len(self.source_ranks)
                                         or set(self.reduction_ring) != set(self.source_ranks)):
            raise ValueError('PROJECT requires the certified actual source NCCL reduction ring')
        if self.variant == 'project' and len(self.source_ranks) != 4:
            raise ValueError('registered fused BF16 model PROJECT requires its certified four-contributor domain')

    def reduction_order(self, destination_rank):
        if self.variant != 'project' or destination_rank not in self.destination_ranks:
            raise ValueError('reduction order belongs to an actual PROJECT consumer')
        source = self.source_ranks[self.destination_ranks.index(destination_rank)]
        position = self.reduction_ring.index(source)
        ranks = tuple(self.reduction_ring[(position + step) % len(self.reduction_ring)]
                      for step in range(1, len(self.reduction_ring) + 1))
        return tuple(self.source_ranks.index(rank) for rank in ranks)

    @property
    def full_shape(self):
        return (self.shard_shape[0] * len(self.source_ranks),) + self.shard_shape[1:]

    @property
    def source_packet_shape(self):
        factor = len(self.source_ranks) + 1 if self.variant == 'project' else 2
        return (self.shard_shape[0] * factor,) + self.shard_shape[1:]

    @property
    def destination_packet_shape(self):
        return (self.shard_shape[0] * 2,) + self.shard_shape[1:]

    def witness(self):
        rows = self.shard_shape[0]
        maps = tuple((source, destination, 0, index * rows, (index + 1) * rows)
                     for source in self.source_ranks for index, destination in enumerate(self.destination_ranks))
        if self.variant == 'canonical':
            maps = tuple((source, destination, 0, index * rows, (index + 1) * rows)
                         for index, (source, destination) in enumerate(zip(self.source_ranks, self.destination_ranks, strict=True)))
        specs = []
        for rank in self.source_ranks + self.destination_ranks:
            shape = self.source_packet_shape if rank in self.source_ranks else self.destination_packet_shape
            for slot in range(self.lifetime_capacity):
                specs.append(BufferSpecification(rank, 'gradient_packet' if rank in self.source_ranks else 'input_packet',
                                                  shape, self.dtype, slot))
                if self.variant == 'project' and rank in self.destination_ranks:
                    for contributor in range(len(self.source_ranks)):
                        specs.append(BufferSpecification(rank, f'projected:{contributor}', self.shard_shape, self.dtype, slot))
        return ExecutableWitness((('SOURCE_TP', self.source_ranks), ('DESTINATION_TP', self.destination_ranks),
                                  ('HANDOFF', self.source_ranks + self.destination_ranks)),
                                 ((f'layer:{self.logical_layer}:MLP_ROW', self.source_ranks),), maps,
                                 ('SOURCE_PROJECT', 'PP_HANDOFF', 'DESTINATION_SUM', 'RESIDUAL_TRANSFER')
                                 if self.variant == 'project' else ('SOURCE_TP_RESOLVE', 'TYPED_PP_TRANSFER'),
                                 ('GRADIENT_FANOUT', 'SOURCE_GRADIENT_ASSEMBLY', 'RESIDUAL_ADJOINT')
                                 if self.variant == 'project' else ('TYPED_PP_GRADIENT', 'SOURCE_TP_GRADIENT_GATHER', 'RESIDUAL_ADJOINT'), tuple(specs),
                                 ('static_rank_maps', 'native_1F1B', 'combined_forward_backward_requests',
                                  'CUDA_event_before_reuse', 'global_checkpoint_layer_ownership',
                                  'BF16_source_ring:' + ','.join(map(str, self.reduction_ring))),
                                 'resolve_on_handoff_model_bf16' if self.variant == 'project' else 'canonical_row_boundary')

    def program(self):
        witness = self.witness()
        shard = Placement(PlacementKind.SHARD, 0)
        partial = Placement(PlacementKind.PARTIAL, reduce_op='sum')
        source = ValueContract(f'layer:{self.logical_layer}:MLP_ROW_OUTPUT', self.full_shape,
                               partial if self.variant == 'project' else shard, 'source_tp',
                               PendingObligation(ObligationKind.REDUCE, shard, 'SOURCE_TP')
                               if self.variant == 'project' else None,
                               adjoint_key='gradient_fanout' if self.variant == 'project' else 'identity', process_group_key='SOURCE_TP')
        destination = ValueContract(source.logical_value, self.full_shape, shard, 'destination_tp',
                                    adjoint_key='gradient_fanout' if self.variant == 'project' else 'identity', process_group_key='DESTINATION_TP')
        operation = BoundaryOp('row_project_resolve' if self.variant == 'project' else 'row_canonical_transfer',
                               BoundaryOpKind.PROJECT if self.variant == 'project' else BoundaryOpKind.TRANSFER,
                               source, destination, witness.implementation_key)
        residual_source = ValueContract(f'layer:{self.logical_layer}:MLP_RESIDUAL', self.full_shape, shard,
                                        'source_tp', adjoint_key='identity', process_group_key='SOURCE_TP')
        residual_destination = ValueContract(residual_source.logical_value, self.full_shape, shard,
                                             'destination_tp', adjoint_key='identity', process_group_key='DESTINATION_TP')
        residual = BoundaryOp('residual_handoff', BoundaryOpKind.TRANSFER, residual_source, residual_destination,
                              'typed_residual_identity_handoff')
        bytes_per_endpoint = prod(self.source_packet_shape) * 2
        return BoundaryProgram(2 * self.logical_layer + 1, 2 * self.logical_layer + 1,
                               2 * self.logical_layer + 2, source, destination, (operation, residual),
                               transferred_bytes=bytes_per_endpoint * len(self.source_ranks),
                               executable_witness_key=witness.identifier(), executable_witness=witness)


class ProjectedPayloadRuntime:
    def __init__(self, plan, handoff_group, diagnostic_directory=None, profiling_origin=None):
        if not dist.is_initialized() or not torch.cuda.is_available() or dist.get_backend(handoff_group) != 'nccl':
            raise RuntimeError('model PROJECT requires actual CUDA and NCCL')
        members = tuple(dist.get_process_group_ranks(handoff_group))
        if members != tuple(sorted(plan.source_ranks + plan.destination_ranks)):
            raise ValueError('actual NCCL members differ from compiled handoff ownership')
        self.rank = dist.get_rank()
        self.source = self.rank in plan.source_ranks
        if self.rank not in members:
            raise ValueError('projected runtime rank is outside its compiled domains')
        self.plan, self.group = plan, handoff_group
        self.program = plan.program()
        self.diagnostic_directory = diagnostic_directory
        self.diagnostic_written = False
        self.profiling_origin = profiling_origin
        self.profile_events, self.profile_rows = [], []
        self.dtype, self.device = getattr(torch, plan.dtype), torch.device('cuda', torch.cuda.current_device())
        self.peer = (plan.destination_ranks[plan.source_ranks.index(self.rank)] if self.source else
                     plan.source_ranks[plan.destination_ranks.index(self.rank)])
        self.pool = BoundaryBufferPool()
        if plan.variant == 'project' and not self.source:
            for slot in range(plan.lifetime_capacity):
                self.pool.register_slot(slot * 2, len(plan.source_ranks), plan.shard_shape, self.dtype, self.device)
        shape = plan.source_packet_shape if self.source else plan.destination_packet_shape
        self.packets = tuple(torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(plan.lifetime_capacity))
        self.live = {}
        self.forward_count = self.backward_count = 0
        self.completed_updates = 0
        self.sent_forward_bytes = self.sent_backward_bytes = 0
        self._stream = torch.cuda.Stream(device=self.device)
        self._ready = torch.cuda.Event()

    def _check(self, tensor, shape):
        if (tuple(tensor.shape) != shape or tensor.dtype != self.dtype or tensor.device != self.device
                or not tensor.is_contiguous()):
            raise ValueError('actual tensor differs from the compiled typed PROJECT packet')

    def _slot(self, forward):
        index = self.forward_count if forward else self.backward_count
        if index >= self.plan.microbatches:
            raise RuntimeError('boundary call exceeds its compiled microbatch coverage')
        slot = index % self.plan.lifetime_capacity
        if forward:
            if slot in self.live:
                raise RuntimeError('schedule attempts to overwrite a live model payload')
            self.live[slot] = index
            self.forward_count += 1
        else:
            if self.live.get(slot) != index:
                raise RuntimeError('adjoint does not match the retained model microbatch')
            self.backward_count += 1
        return slot

    def communicate(self, *, forward_tensor=None, backward_tensor=None, receive_forward=False, receive_backward=False):
        if self.source and (backward_tensor is not None or receive_forward):
            raise ValueError('source endpoint received a destination operation')
        if not self.source and (forward_tensor is not None or receive_backward):
            raise ValueError('destination endpoint received a source operation')
        forward = forward_tensor is not None or receive_forward
        backward = backward_tensor is not None or receive_backward
        fslot = self._slot(True) if forward else None
        bslot = self._slot(False) if backward else None
        operations, keepalive = [], []
        rows = self.plan.shard_shape[0]
        full_rows = self.plan.full_shape[0]
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(self._stream), torch.cuda.nvtx.range('Frontier/PROJECT_PAYLOAD'):
            if self.profiling_origin is not None:
                begin_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin_event.record(self._stream)
            if self.plan.variant == 'canonical':
                if forward_tensor is not None:
                    self._check(forward_tensor, self.plan.source_packet_shape)
                    operations.append(dist.P2POp(dist.isend, forward_tensor, self.peer, self.group))
                    self.sent_forward_bytes += forward_tensor.numel() * forward_tensor.element_size()
                elif receive_forward:
                    operations.append(dist.P2POp(dist.irecv, self.packets[fslot], self.peer, self.group))
                if backward_tensor is not None:
                    self._check(backward_tensor, self.plan.destination_packet_shape)
                    operations.append(dist.P2POp(dist.isend, backward_tensor, self.peer, self.group))
                    self.sent_backward_bytes += backward_tensor.numel() * backward_tensor.element_size()
                elif receive_backward:
                    operations.append(dist.P2POp(dist.irecv, self.packets[bslot], self.peer, self.group))
            else:
                for is_forward, slot, tensor, receiving in ((True, fslot, forward_tensor, receive_forward),
                                                           (False, bslot, backward_tensor, receive_backward)):
                    if slot is None:
                        continue
                    peers = self.plan.destination_ranks if self.source else self.plan.source_ranks
                    if tensor is not None:
                        expected = self.plan.source_packet_shape if self.source else self.plan.destination_packet_shape
                        self._check(tensor, expected)
                        projected = tuple(tensor.narrow(0, index * rows if is_forward else 0, rows)
                                          for index in range(len(peers)))
                        residual = tensor.narrow(0, full_rows if is_forward else rows, rows)
                        for peer, buffer in zip(peers, projected, strict=True):
                            operations.append(dist.P2POp(dist.isend, buffer, peer, self.group))
                        operations.append(dist.P2POp(dist.isend, residual, self.peer, self.group))
                        count = sum(buffer.numel() * buffer.element_size() for buffer in projected) + residual.numel() * residual.element_size()
                        if is_forward:
                            self.sent_forward_bytes += count
                        else:
                            self.sent_backward_bytes += count
                    elif receiving:
                        if is_forward:
                            projected = self.pool.acquire_many(slot * 2, len(peers), self.plan.shard_shape)
                            residual = self.packets[slot].narrow(0, rows, rows)
                            keepalive.append((slot, projected))
                        else:
                            projected = tuple(self.packets[slot].narrow(0, index * rows, rows) for index in range(len(peers)))
                            residual = self.packets[slot].narrow(0, full_rows, rows)
                        for peer, buffer in zip(peers, projected, strict=True):
                            operations.append(dist.P2POp(dist.irecv, buffer, peer, self.group))
                        operations.append(dist.P2POp(dist.irecv, residual, self.peer, self.group))
            if not operations:
                raise ValueError('typed boundary call contains no actual communication')
            for request in dist.batch_isend_irecv(operations):
                request.wait()
            for slot, projected in keepalive:
                if receive_forward and slot == fslot:
                    from megatron.core.frontier.runtime.cuda_ring_reduce import reduce_projected_ring

                    order = self.plan.reduction_order(self.rank)
                    reduce_projected_ring(self.packets[slot].narrow(0, 0, rows), projected, order)
                self.pool.release(slot * 2)
            self._ready.record(self._stream)
            if self.profiling_origin is not None:
                end_event.record(self._stream)
                self.profile_events.append((begin_event, end_event, {
                    'rank': self.rank, 'forward_microbatch': self.forward_count - 1 if forward else None,
                    'backward_microbatch': self.backward_count - 1 if backward else None,
                    'send_forward': forward_tensor is not None, 'receive_forward': receive_forward,
                    'send_backward': backward_tensor is not None, 'receive_backward': receive_backward}))
        torch.cuda.current_stream(self.device).wait_event(self._ready)
        if backward:
            del self.live[bslot]
        incoming = self.packets[fslot].detach().requires_grad_(True) if receive_forward else None
        gradient = self.packets[bslot] if receive_backward else None
        if receive_forward and self.diagnostic_directory is not None and not self.diagnostic_written:
            torch.save(incoming.narrow(0, 0, rows).detach().cpu(),
                       self.diagnostic_directory / f'rank{self.rank}.projected-row.pt')
            self.diagnostic_written = True
        return incoming, gradient

    def finish_update(self):
        if self.live or self.forward_count != self.plan.microbatches or self.backward_count != self.plan.microbatches:
            raise ValueError('model PROJECT update did not complete every payload and adjoint lifetime')
        for begin, end, row in self.profile_events:
            end.synchronize()
            self.profile_rows.append(row | {'update': self.completed_updates + 1,
                                            'begin_ms': self.profiling_origin.elapsed_time(begin),
                                            'consumer_ready_ms': self.profiling_origin.elapsed_time(end) if row['receive_forward'] else None,
                                            'end_ms': self.profiling_origin.elapsed_time(end),
                                            'communication_and_transform_ms': begin.elapsed_time(end),
                                            'clock_scope': 'GPU_local_update_origin'})
        self.profile_events = []
        self._ready.record(torch.cuda.current_stream(self.device))
        self.completed_updates += 1
        self.forward_count = self.backward_count = 0

    def evidence(self):
        witness = self.program.executable_witness
        allocated = sum(packet.numel() * packet.element_size() for packet in self.packets)
        if self.plan.variant == 'project' and not self.source:
            allocated += self.plan.lifetime_capacity * len(self.plan.source_ranks) * prod(self.plan.shard_shape) * 2
        return {'plan': asdict(self.plan), 'executable_witness': asdict(witness),
                'boundary_program': {'boundary_id': self.program.boundary_id,
                                     'source_region': self.program.source_region,
                                     'destination_region': self.program.destination_region,
                                     'source_contract': contract_metadata(self.program.source_contract),
                                     'destination_contract': contract_metadata(self.program.destination_contract),
                                     'implementation_key': self.program.ops[0].implementation_key,
                                     'operations': [{'kind': operation.kind.name, 'implementation_key': operation.implementation_key,
                                                     'source_contract': contract_metadata(operation.input_contract),
                                                     'destination_contract': contract_metadata(operation.output_contract)}
                                                    for operation in self.program.ops],
                                     'transferred_bytes': self.program.transferred_bytes},
                'boundary_program_id': witness.identifier(), 'completed_updates': self.completed_updates,
                'forward_sent_bytes': self.sent_forward_bytes, 'backward_sent_bytes': self.sent_backward_bytes,
                'preallocated_bytes': allocated, 'precision': self.plan.dtype,
                'profile_rows': self.profile_rows,
                'scope': 'actual Qwen row-parallel output and its residual field'}
