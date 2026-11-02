from dataclasses import asdict, dataclass
from math import prod

import torch
import torch.distributed as dist

from megatron.core.frontier.boundary.payload import BoundaryPayload, PayloadField
from megatron.core.frontier.ir.contract import Placement, PlacementKind
from megatron.core.frontier.ir.contract import ValueContract
from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram, BufferSpecification, ExecutableWitness


@dataclass(frozen=True)
class TypedBoundaryBinding:
    logical_layer: int
    source_pp: int
    local_shape: tuple[int, ...]
    microbatches: int
    dtype: str = 'bfloat16'
    tp_size: int = 1
    virtual_pipeline_size: int = 1
    source_vp: int = 0

    def __post_init__(self):
        if (min(self.logical_layer, self.source_pp, self.source_vp) < 0
                or min(self.microbatches, self.tp_size, self.virtual_pipeline_size) < 1
                or self.source_vp >= self.virtual_pipeline_size):
            raise ValueError('typed boundary requires valid region and schedule coordinates')
        if len(self.local_shape) != 3 or min(self.local_shape) < 1:
            raise ValueError('dense region payload must have sequence, microbatch and hidden dimensions')
        if self.dtype not in ('bfloat16', 'float32'):
            raise ValueError('typed training transport requires a registered tensor precision')

    @property
    def payload(self):
        logical_shape = (self.local_shape[0] * self.tp_size,) + self.local_shape[1:]
        placement = Placement(PlacementKind.SHARD, 0) if self.tp_size > 1 else Placement(PlacementKind.REPLICATE)
        field = PayloadField(f'layer:{self.logical_layer}:attention_output', logical_shape, self.dtype,
                             placement, True, False, 'identity')
        return BoundaryPayload((field,))

    @property
    def native_shape(self):
        return self.local_shape

    @property
    def native_dtype(self):
        return self.dtype

    def payload_metadata(self):
        return {'fields': [{'name': field.name, 'logical_shape': field.shape, 'dtype': field.dtype,
                            'placement': {'kind': field.placement.kind.name, 'dimension': field.placement.dim},
                            'gradient_eligible': field.gradient_eligible, 'optional': field.optional,
                            'adjoint_key': field.adjoint_key} for field in self.payload.fields]}

    def program_for_group(self, pp_ranks, tp_rank):
        if not 0 <= tp_rank < self.tp_size or self.source_pp + 1 >= len(pp_ranks):
            raise ValueError('compiled typed boundary is outside the actual parallel groups')
        source_rank, destination_rank = pp_ranks[self.source_pp:self.source_pp + 2]
        field = self.payload.fields[0]
        source = ValueContract(field.name, field.shape, field.placement, f'pp:{self.source_pp}:tp',
                               dtype=self.dtype, process_group_key='PP_TRANSPORT')
        destination = ValueContract(field.name, field.shape, field.placement, f'pp:{self.source_pp + 1}:tp',
                                    dtype=self.dtype, process_group_key='PP_TRANSPORT')
        start = tp_rank * self.local_shape[0]
        specs = tuple(BufferSpecification(rank, name, self.local_shape, self.dtype, microbatch)
                      for rank, name in ((source_rank, 'attention_output_gradient'),
                                         (destination_rank, 'attention_output'))
                      for microbatch in range(self.microbatches))
        witness = ExecutableWitness((('PP_TRANSPORT', tuple(pp_ranks)),),
                                    ((f'layer:{self.logical_layer}:ATTENTION', (source_rank,)),
                                     (f'layer:{self.logical_layer}:MLP', (destination_rank,))),
                                    ((source_rank, destination_rank, 0, start, start + self.local_shape[0]),),
                                    ('TYPED_PP_SEND', 'TYPED_PP_RECV'), ('TYPED_GRAD_SEND', 'TYPED_GRAD_RECV'),
                                    specs, ('native_interleaved' if self.virtual_pipeline_size > 1 else 'native_1F1B',
                                            f'producer_vp:{self.source_vp}', 'one_slot_per_microbatch', 'identity_adjoint',
                                            'cuda_event_before_slot_reuse'), 'typed_dense_pp')
        operation = BoundaryOp('typed_attention_output', BoundaryOpKind.TRANSFER, source, destination, 'typed_dense_pp')
        return BoundaryProgram(2 * self.logical_layer, 2 * self.logical_layer, 2 * self.logical_layer + 1,
                               source, destination, (operation,),
                               transferred_bytes=prod(self.local_shape) * (2 if self.dtype == 'bfloat16' else 4),
                               executable_witness_key=witness.identifier(), executable_witness=witness)


class StaticPayloadP2P:
    def __init__(self, communicator, binding):
        if not dist.is_initialized() or dist.get_backend(communicator.pp_group) != 'nccl':
            raise RuntimeError('typed PP transport requires an initialized NCCL process group')
        self.communicator = communicator
        self.binding = binding
        self.stage = dist.get_rank(communicator.pp_group)
        if self.stage not in (binding.source_pp, binding.source_pp + 1):
            raise ValueError('typed transport instance must own a boundary endpoint')
        if communicator.config.variable_seq_lengths or communicator.config.mtp_standalone:
            raise ValueError('typed region transport requires its compiled static payload shape')
        self.dtype = getattr(torch, binding.dtype)
        from megatron.core import parallel_state

        if dist.get_world_size(parallel_state.get_tensor_model_parallel_group()) != binding.tp_size:
            raise ValueError('typed payload partition differs from actual TP membership')
        self.program = binding.program_for_group(dist.get_process_group_ranks(communicator.pp_group),
                                                  parallel_state.get_tensor_model_parallel_rank())
        if communicator.config.pipeline_dtype != getattr(torch, binding.native_dtype):
            raise ValueError('native pipeline precision differs from the registered model capability')
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.buffers = tuple(torch.empty(binding.local_shape, dtype=self.dtype, device=self.device)
                             for _ in range(binding.microbatches))
        self.counts = dict(forward_send=0, forward_recv=0, backward_send=0, backward_recv=0)
        self.completed_updates = 0
        self._release_event = torch.cuda.Event()
        self._released = False

    def _check(self, tensor):
        if tuple(tensor.shape) != self.binding.local_shape or tensor.dtype != self.dtype or tensor.device != self.device:
            raise ValueError('actual tensor differs from the compiled region payload contract')
        if not tensor.is_contiguous():
            raise ValueError('typed PP transfer requires contiguous payload storage')

    def _count(self, direction):
        index = self.counts[direction]
        if index >= self.binding.microbatches:
            raise RuntimeError('typed boundary exceeded its compiled microbatch schedule')
        self.counts[direction] += 1
        return index

    def communicate(self, *, tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                    tensor_shape, wait_on_reqs=True):
        from megatron.core.pipeline_parallel.p2p_communication import _batched_p2p_ops, _p2p_ops

        if not wait_on_reqs:
            raise ValueError('this registered typed capability requires completed PP requests')
        if (tensor_shape is not None and tuple(tensor_shape) != self.binding.native_shape
                or (recv_prev or recv_next) and tensor_shape is None):
            raise ValueError('receive metadata differs from the compiled typed payload')
        if self._released:
            torch.cuda.current_stream(self.device).wait_event(self._release_event)
            self._released = False
        source = self.stage == self.binding.source_pp
        if ((source and (recv_prev or tensor_send_prev is not None))
                or (not source and (recv_next or tensor_send_next is not None))):
            raise ValueError('static payload call crosses an additional unregistered boundary')
        recv_forward = recv_backward = None
        with torch.cuda.nvtx.range(f'Frontier/typed_PP/layer:{self.binding.logical_layer}'):
            if tensor_send_next is not None and source:
                self._check(tensor_send_next)
                self._count('forward_send')
            if tensor_send_prev is not None and not source:
                self._check(tensor_send_prev)
                self._count('backward_send')
            if recv_prev:
                recv_forward = self.buffers[self._count('forward_recv')].detach().requires_grad_(True)
            if recv_next:
                recv_backward = self.buffers[self._count('backward_recv')].detach().requires_grad_(True)
            operation = _batched_p2p_ops if self.communicator.config.batch_p2p_comm else _p2p_ops
            requests = operation(tensor_send_prev=tensor_send_prev, tensor_recv_prev=recv_forward,
                                 tensor_send_next=tensor_send_next, tensor_recv_next=recv_backward,
                                 group=self.communicator.pp_group,
                                 prev_pipeline_rank=self.communicator.prev_rank,
                                 next_pipeline_rank=self.communicator.next_rank)
            for request in requests if isinstance(requests, list) else requests.values():
                request.wait()
            if self.communicator.config.batch_p2p_comm and self.communicator.config.batch_p2p_sync:
                torch.cuda.synchronize(self.device)
        return recv_forward, recv_backward, None

    def finish_update(self):
        expected = ('forward_send', 'backward_recv') if self.stage == self.binding.source_pp else ('forward_recv', 'backward_send')
        if any(self.counts[key] != self.binding.microbatches for key in expected):
            raise ValueError('actual forward and adjoint calls do not cover every compiled microbatch')
        self._release_event.record(torch.cuda.current_stream(self.device))
        self._released = True
        self.completed_updates += 1
        self.counts = dict.fromkeys(self.counts, 0)

    def evidence(self):
        return {'binding': asdict(self.binding), 'payload': self.binding.payload_metadata(),
                'boundary_program': {'boundary_id': self.program.boundary_id,
                                     'producer_region': self.program.source_region,
                                     'consumer_region': self.program.destination_region,
                                     'operation': self.program.ops[0].implementation_key,
                                     'transferred_bytes_per_microbatch': self.program.transferred_bytes,
                                     'executable_witness': asdict(self.program.executable_witness),
                                     'executable_witness_key': self.program.executable_witness_key},
                'pp_ranks': dist.get_process_group_ranks(self.communicator.pp_group),
                'stage': self.stage, 'completed_updates': self.completed_updates,
                'microbatches_per_update': self.binding.microbatches,
                'preallocated_bytes': sum(buffer.numel() * buffer.element_size() for buffer in self.buffers),
                'adjoint': 'identity_PP_gradient_transport', 'transport': 'native_NCCL_p2p_ops'}


class VirtualPayloadP2P(StaticPayloadP2P):
    def __init__(self, communicator, binding):
        from megatron.core.pipeline_parallel.schedules import get_schedule_table

        super().__init__(communicator, binding)
        size = communicator.config.virtual_pipeline_model_parallel_size
        if size != binding.virtual_pipeline_size or communicator.config.use_ring_exchange_p2p:
            raise ValueError('virtual payload transport requires its actual native interleaved NCCL schedule')
        self.schedule_table = tuple(get_schedule_table(binding.microbatches, size,
                                                       communicator.config.microbatch_group_size_per_vp_stage))
        self.virtual_counts = dict.fromkeys(self.counts, 0)

    def _selected(self, direction):
        index = self.virtual_counts[direction]
        if index >= len(self.schedule_table):
            raise ValueError('actual interleaved transport exceeded its native virtual microbatch table')
        self.virtual_counts[direction] += 1
        _, chunk = self.schedule_table[index]
        if direction.startswith('backward'):
            chunk = self.binding.virtual_pipeline_size - chunk - 1
        return chunk == self.binding.source_vp

    def communicate(self, *, tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                    tensor_shape, wait_on_reqs=True):
        from megatron.core.pipeline_parallel.p2p_communication import _batched_p2p_ops, _p2p_ops

        config = self.communicator.config
        if ((recv_prev or recv_next) and tuple(tensor_shape or ()) != self.binding.native_shape
                or config.batch_p2p_comm and not wait_on_reqs):
            raise ValueError('virtual payload operation differs from the actual native shape or batching policy')
        if self._released:
            torch.cuda.current_stream(self.device).wait_event(self._release_event)
            self._released = False
        source = self.stage == self.binding.source_pp
        selected_send_next = source and tensor_send_next is not None and self._selected('forward_send')
        selected_send_prev = not source and tensor_send_prev is not None and self._selected('backward_send')
        selected_recv_prev = not source and recv_prev and self._selected('forward_recv')
        selected_recv_next = source and recv_next and self._selected('backward_recv')
        for selected, tensor, direction in ((selected_send_next, tensor_send_next, 'forward_send'),
                                             (selected_send_prev, tensor_send_prev, 'backward_send')):
            if selected:
                self._check(tensor)
                self._count(direction)
        def receive(selected, direction):
            if selected:
                return self.buffers[self._count(direction)].detach().requires_grad_(True)
            return torch.empty(self.binding.native_shape, dtype=config.pipeline_dtype,
                               device=self.device, requires_grad=True)
        forward = receive(selected_recv_prev, 'forward_recv') if recv_prev else None
        backward = receive(selected_recv_next, 'backward_recv') if recv_next else None
        operation = _batched_p2p_ops if config.batch_p2p_comm else _p2p_ops
        requests = operation(tensor_send_prev=tensor_send_prev, tensor_recv_prev=forward,
                             tensor_send_next=tensor_send_next, tensor_recv_next=backward,
                             group=self.communicator.pp_group,
                             prev_pipeline_rank=self.communicator.prev_rank,
                             next_pipeline_rank=self.communicator.next_rank)
        if wait_on_reqs:
            for request in requests if isinstance(requests, list) else requests.values():
                request.wait()
            requests = None
        if config.batch_p2p_comm and config.batch_p2p_sync:
            torch.cuda.synchronize(self.device)
        return forward, backward, requests

    def finish_update(self):
        directions = ('forward_send', 'backward_recv') if self.stage == self.binding.source_pp else ('forward_recv', 'backward_send')
        if any(self.virtual_counts[name] != len(self.schedule_table) for name in directions):
            raise ValueError('actual interleaved transport omitted native virtual chunk communication')
        super().finish_update()
        self.virtual_counts = dict.fromkeys(self.virtual_counts, 0)

    def evidence(self):
        return super().evidence() | {'native_virtual_schedule_table': self.schedule_table,
                                     'virtual_pipeline_size': self.binding.virtual_pipeline_size,
                                     'producer_virtual_chunk': self.binding.source_vp,
                                     'transport': 'native_NCCL_mixed_typed_virtual_chunk_p2p'}


def install_static_payload_p2p(binding):
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    if getattr(P2PCommunicator, '_frontier_typed_binding', None) is not None:
        raise RuntimeError('typed PP capability is already installed')
    original_init = P2PCommunicator.__init__
    original_communicate = P2PCommunicator._communicate
    instances = []
    group_instances = {}

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        stage = dist.get_rank(self.pp_group)
        self._frontier_typed = None
        if stage in (binding.source_pp, binding.source_pp + 1):
            key = tuple(dist.get_process_group_ranks(self.pp_group))
            if key not in group_instances:
                transport = VirtualPayloadP2P if getattr(binding, 'virtual_pipeline_size', 1) > 1 else StaticPayloadP2P
                group_instances[key] = transport(self, binding)
                instances.append(group_instances[key])
            self._frontier_typed = group_instances[key]
            self._frontier_typed.communicator = self

    def communicate(self, **kwargs):
        if self._frontier_typed is None:
            return original_communicate(self, **kwargs)
        if isinstance(self._frontier_typed, VirtualPayloadP2P):
            return self._frontier_typed.communicate(**kwargs)
        source = self._frontier_typed.stage == binding.source_pp
        bound = (kwargs['tensor_send_next'] is not None or kwargs['recv_next']) if source else (
            kwargs['tensor_send_prev'] is not None or kwargs['recv_prev'])
        if not bound:
            return original_communicate(self, **kwargs)
        return self._frontier_typed.communicate(**kwargs)

    P2PCommunicator.__init__ = initialize
    P2PCommunicator._communicate = communicate
    P2PCommunicator._frontier_typed_binding = binding
    return instances


def install_typed_dense_p2p(binding):
    if not isinstance(binding, TypedBoundaryBinding):
        raise TypeError('dense transport requires its registered dense payload contract')
    return install_static_payload_p2p(binding)
