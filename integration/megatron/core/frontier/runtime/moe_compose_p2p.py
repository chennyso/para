from collections import deque
from dataclasses import asdict

import torch
import torch.distributed as dist

from megatron.core.frontier.runtime.moe_compose_payload import allocate_dispatch_control, dispatch_control_tensors, restore_dispatch_control


class ComposedMoEP2P:
    def __init__(self, communicator, binding, geometry, group_state):
        self.communicator, self.binding, self.geometry = communicator, binding, geometry
        self.group_state = group_state
        self.rank = dist.get_rank()
        self.source = self.rank in binding.source_expert_ranks
        ranks = binding.source_expert_ranks if self.source else binding.destination_expert_ranks
        if self.rank not in ranks or dist.get_backend(communicator.pp_group) != 'nccl':
            raise ValueError('COMPOSE transport requires an actual NCCL expert-domain endpoint')
        self.coordinate = ranks.index(self.rank)
        self.program = binding.program_for_group(dist.get_process_group_ranks(communicator.pp_group), 0)
        self.peer = (binding.destination_expert_ranks if self.source else binding.source_expert_ranks)[self.coordinate]
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.pending_source = deque()
        self.pending_receiver = deque()
        self.source_splits, self.destination_splits = [], []
        self.counts = dict(forward_send=0, forward_recv=0, backward_send=0, backward_recv=0)
        self.completed_updates = 0
        self.routes = []
        self.transferred = dict(forward=0, backward=0, control=0)
        if self.source:
            self.gradient = torch.empty(geometry.source_packet_shape(), dtype=geometry.packet_torch_dtype, device=self.device)
            self.gradient_tokens = torch.empty((geometry.permuted_tokens, geometry.hidden_shape[-1]),
                                               dtype=torch.bfloat16, device=self.device)
            self.gradient_probabilities = torch.empty(geometry.permuted_tokens, dtype=geometry.probability_torch_dtype, device=self.device)
            self.gradient_residual = torch.empty(geometry.hidden_shape, dtype=torch.bfloat16, device=self.device)
            self.receiver_buffers = ()
        else:
            # PP1 的 1F1B 最多同时保留两个 warmup 和一个当前 microbatch。
            capacity = min(binding.microbatches, communicator.config.pipeline_model_parallel_size - binding.source_pp - 1)
            if capacity < 1:
                raise ValueError('COMPOSE consumer has no legal schedule-derived payload lifetime')
            self.receiver_buffers = tuple(torch.empty(geometry.packet_shape(geometry.maximum_received_tokens),
                                                      dtype=geometry.packet_torch_dtype, device=self.device) for _ in range(capacity))
            self.received_tokens = torch.empty((geometry.maximum_received_tokens, geometry.hidden_shape[-1]),
                                              dtype=torch.bfloat16, device=self.device)
            self.received_probabilities = torch.empty(geometry.maximum_received_tokens, dtype=geometry.probability_torch_dtype, device=self.device)
            self.received_residual = torch.empty(geometry.hidden_shape, dtype=torch.bfloat16, device=self.device)

    def enqueue_source(self, control, input_splits, output_splits):
        if not self.source or len(self.pending_source) >= self.binding.microbatches:
            raise ValueError('actual native routing control has no matching source pipeline operation')
        self.pending_source.append((control, input_splits, output_splits))

    def take_receiver(self, dispatcher):
        if self.source or not self.pending_receiver:
            raise ValueError('expert consumer has no completed PP+EP routing payload')
        control, epoch, received = self.pending_receiver.popleft()
        _, output_splits = restore_dispatch_control(dispatcher, control, self.geometry, epoch)
        if sum(output_splits) != received:
            raise ValueError('actual received expert token count differs from native router metadata')
        return received

    def _next(self, name):
        index = self.counts[name]
        if index >= self.binding.microbatches:
            raise ValueError('COMPOSE exceeded its actual 1F1B microbatch count')
        self.counts[name] += 1
        return index

    def _epoch(self, index):
        return self.completed_updates * self.binding.microbatches + index

    def _operations(self, direction, tensor, splits, field):
        opposite = self.binding.destination_expert_ranks if self.source else self.binding.source_expert_ranks
        operations, offset = [], 0
        for peer, count in zip(opposite, splits, strict=True):
            if count:
                view = tensor.narrow(0, offset, count)
                operations.append(dist.P2POp(dist.isend if direction == 'send' else dist.irecv,
                                              view, peer, self.group_state['handoff']))
            offset += count
        if offset != tensor.shape[0]:
            raise ValueError('COMPOSE tensor splits omit actual routed tokens')
        return operations

    def _pair(self, direction, tensor):
        return dist.P2POp(dist.isend if direction == 'send' else dist.irecv,
                         tensor, self.peer, self.group_state['handoff'])

    def _exchange(self, operations):
        if not operations:
            raise ValueError('COMPOSE operation omitted its actual transport fields')
        for request in dist.batch_isend_irecv(operations):
            request.wait()

    def _check_packet(self, packet, rows):
        if (not packet.is_cuda or packet.device != self.device or packet.dtype != self.geometry.packet_torch_dtype
                or tuple(packet.shape) != self.geometry.packet_shape(rows) or not packet.is_contiguous()):
            raise ValueError('COMPOSE model packet differs from its actual forward or adjoint token layout')

    def communicate(self, *, tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                    tensor_shape, wait_on_reqs=True):
        if (not wait_on_reqs or self.communicator.config.variable_seq_lengths
                or self.communicator.config.mtp_standalone
                or tensor_shape is not None and tuple(tensor_shape) != self.binding.native_shape
                or (recv_prev or recv_next) and tensor_shape is None):
            raise ValueError('COMPOSE requires its registered static native shape and completed P2P requests')
        if ((self.source and (recv_prev or tensor_send_prev is not None))
                or (not self.source and (recv_next or tensor_send_next is not None))):
            raise ValueError('COMPOSE call crosses an additional unregistered pipeline edge')
        geometry = self.geometry
        hidden = geometry.hidden_shape[-1]
        operations, retained = [], []
        forward_packet = backward_packet = None
        forward_index = backward_index = None
        if self.source and tensor_send_next is not None:
            forward_index = self._next('forward_send')
            self._check_packet(tensor_send_next, geometry.permuted_tokens)
            if not self.pending_source:
                raise ValueError('COMPOSE source packet omitted its actual native dispatch metadata')
            control, input_splits, output_splits = self.pending_source.popleft()
            self.source_splits.append(input_splits)
            operations.extend(self._pair('send', field) for field in dispatch_control_tensors(control))
            retained.append(control)
        elif not self.source and recv_prev:
            forward_index = self._next('forward_recv')
            if forward_index >= len(self.receiver_buffers):
                released = forward_index - len(self.receiver_buffers)
                released_count = self.counts['backward_send'] + int(tensor_send_prev is not None)
                if released_count <= released:
                    raise ValueError('COMPOSE consumer reuses a payload before its true backward lifetime ends')
            control = allocate_dispatch_control(geometry, self.device)
            operations.extend(self._pair('recv', field) for field in dispatch_control_tensors(control))
            retained.append(control)
        if operations:
            self._exchange(operations)
            self.transferred['control'] += geometry.control_bytes
        operations = []
        if not self.source and recv_prev:
            header = dispatch_control_tensors(control)[0][:16 + 2 * geometry.expert_size].cpu().tolist()
            epoch = self._epoch(forward_index)
            if header[0] != epoch:
                raise ValueError('COMPOSE received control belongs to a different actual microbatch')
            input_splits = tuple(header[16:16 + geometry.expert_size])
            output_splits = tuple(header[16 + geometry.expert_size:])
            from megatron.core.frontier.runtime.moe_compose_payload import validate_split_counts

            validate_split_counts(input_splits, output_splits, geometry)
            self.destination_splits.append(output_splits)
            received = sum(output_splits)
            self.pending_receiver.append((control, epoch, received))
            self.routes.append({'epoch': epoch, 'input_splits': input_splits, 'output_splits': output_splits})
            tokens = self.received_tokens[:received]
            probabilities = self.received_probabilities[:received]
            operations.extend(self._operations('recv', tokens, output_splits, 'tokens'))
            operations.extend(self._operations('recv', probabilities, output_splits, 'probabilities'))
            operations.append(self._pair('recv', self.received_residual))
            forward_packet = self.receiver_buffers[forward_index % len(self.receiver_buffers)][:received + geometry.tokens].detach()
        elif self.source and tensor_send_next is not None:
            tokens = tensor_send_next[:geometry.permuted_tokens, :hidden].to(torch.bfloat16).contiguous()
            probabilities = tensor_send_next[:geometry.permuted_tokens, hidden].to(geometry.probability_torch_dtype).contiguous()
            residual = tensor_send_next[geometry.permuted_tokens:, :hidden].to(torch.bfloat16).contiguous().reshape(geometry.hidden_shape)
            retained.extend((tokens, probabilities, residual))
            operations.extend(self._operations('send', tokens, input_splits, 'tokens'))
            operations.extend(self._operations('send', probabilities, input_splits, 'probabilities'))
            operations.append(self._pair('send', residual))
        if self.source and recv_next:
            backward_index = self._next('backward_recv')
            if backward_index >= len(self.source_splits):
                raise ValueError('COMPOSE adjoint precedes its actual source token dispatch')
            splits = self.source_splits[backward_index]
            operations.extend(self._operations('recv', self.gradient_tokens, splits, 'token_gradient'))
            operations.extend(self._operations('recv', self.gradient_probabilities, splits, 'probability_gradient'))
            operations.append(self._pair('recv', self.gradient_residual))
            backward_packet = self.gradient.detach()
        elif not self.source and tensor_send_prev is not None:
            backward_index = self._next('backward_send')
            if backward_index >= len(self.destination_splits):
                raise ValueError('expert adjoint has no matching actual dispatch token ownership')
            splits = self.destination_splits[backward_index]
            received_gradient = sum(splits)
            self._check_packet(tensor_send_prev, received_gradient)
            gradient_tokens = tensor_send_prev[:received_gradient, :hidden].to(torch.bfloat16).contiguous()
            gradient_probabilities = tensor_send_prev[:received_gradient, hidden].to(geometry.probability_torch_dtype).contiguous()
            gradient_residual = tensor_send_prev[received_gradient:, :hidden].to(torch.bfloat16).contiguous().reshape(geometry.hidden_shape)
            retained.extend((gradient_tokens, gradient_probabilities, gradient_residual))
            operations.extend(self._operations('send', gradient_tokens, splits, 'token_gradient'))
            operations.extend(self._operations('send', gradient_probabilities, splits, 'probability_gradient'))
            operations.append(self._pair('send', gradient_residual))
        with torch.cuda.nvtx.range('Frontier/MoE/COMPOSE_PP_EP'):
            self._exchange(operations)
            if forward_packet is not None:
                forward_packet[:received, :hidden].copy_(tokens)
                forward_packet[:received, hidden].copy_(probabilities)
                forward_packet[received:, :hidden].copy_(self.received_residual.reshape(geometry.tokens, hidden))
                forward_packet[received:, hidden].zero_()
                forward_packet.requires_grad_(True)
            if backward_packet is not None:
                backward_packet[:geometry.permuted_tokens, :hidden].copy_(self.gradient_tokens)
                backward_packet[:geometry.permuted_tokens, hidden].copy_(self.gradient_probabilities)
                backward_packet[geometry.permuted_tokens:, :hidden].copy_(self.gradient_residual.reshape(geometry.tokens, hidden))
                backward_packet[geometry.permuted_tokens:, hidden].zero_()
            if self.communicator.config.batch_p2p_sync:
                torch.cuda.synchronize(self.device)
        if forward_index is not None:
            rows = geometry.permuted_tokens if self.source else received
            self.transferred['forward'] += rows * (hidden * 2 + geometry.probability_itemsize) + geometry.tokens * hidden * 2
        if backward_index is not None:
            rows = geometry.permuted_tokens if self.source else received_gradient
            self.transferred['backward'] += rows * (hidden * 2 + geometry.probability_itemsize) + geometry.tokens * hidden * 2
        return forward_packet, backward_packet, None

    def finish_update(self):
        expected = ('forward_send', 'backward_recv') if self.source else ('forward_recv', 'backward_send')
        if (any(self.counts[key] != self.binding.microbatches for key in expected)
                or self.pending_source or self.pending_receiver):
            raise ValueError('COMPOSE actual forward and adjoint omit compiled microbatch coverage')
        self.completed_updates += 1
        self.counts = dict.fromkeys(self.counts, 0)
        self.source_splits.clear()
        self.destination_splits.clear()

    def evidence(self):
        return {'binding': asdict(self.binding), 'geometry': asdict(self.geometry), 'rank': self.rank,
                'boundary_program': {'executable_witness_key': self.program.executable_witness_key,
                                     'executable_witness': asdict(self.program.executable_witness),
                                     'operation': self.program.ops[0].implementation_key},
                'transport': 'native_NCCL_batched_PP_EP_direct_routing', 'completed_updates': self.completed_updates,
                'actual_routes': self.routes, 'actual_endpoint_transferred_bytes': self.transferred,
                'consumer_payload_capacity': len(self.receiver_buffers),
                'adjoint': 'inverse_actual_router_splits_for_native_precision_tokens_probabilities_and_paired_residual'}


def install_composed_moe_p2p(binding, geometry, group_state):
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    if getattr(P2PCommunicator, '_frontier_typed_binding', None) is not None:
        raise ValueError('one actual MoE boundary must own its pipeline transport')
    original_init, original_communicate = P2PCommunicator.__init__, P2PCommunicator._communicate
    instances, by_group = [], {}

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._frontier_composed = None
        if dist.get_rank() in binding.source_expert_ranks + binding.destination_expert_ranks:
            key = tuple(dist.get_process_group_ranks(self.pp_group))
            if key not in by_group:
                by_group[key] = ComposedMoEP2P(self, binding, geometry, group_state)
                instances.append(by_group[key])
            self._frontier_composed = by_group[key]
            self._frontier_composed.communicator = self

    def communicate(self, **kwargs):
        instance = self._frontier_composed
        if instance is None:
            return original_communicate(self, **kwargs)
        bound = (kwargs['tensor_send_next'] is not None or kwargs['recv_next']) if instance.source else (
            kwargs['tensor_send_prev'] is not None or kwargs['recv_prev'])
        if not bound:
            return original_communicate(self, **kwargs)
        return instance.communicate(**kwargs)

    P2PCommunicator.__init__, P2PCommunicator._communicate = initialize, communicate
    P2PCommunicator._frontier_typed_binding = binding
    return instances
