from dataclasses import dataclass
from math import prod

import numpy as np
import torch

from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram, BufferSpecification, DynamicRoutingMap, ExecutableWitness
from megatron.core.frontier.ir.contract import Placement, PlacementKind, ValueContract


@dataclass(frozen=True)
class MoEComposeGeometry:
    hidden_shape: tuple[int, int, int]
    num_experts: int
    topk: int
    expert_size: int
    receiver_token_capacity: int | None = None
    probability_dtype: str = 'bfloat16'
    control_encoding: str = 'int64_carrier'

    def __post_init__(self):
        if (len(self.hidden_shape) != 3 or min(self.hidden_shape) < 1
                or not 0 < self.topk <= self.num_experts or self.expert_size < 2
                or self.num_experts % self.expert_size):
            raise ValueError('COMPOSE requires actual dropless token and expert geometry')
        if self.receiver_token_capacity is not None and not self.permuted_tokens <= self.receiver_token_capacity <= self.permuted_tokens * self.expert_size:
            raise ValueError('COMPOSE receiver allocation must fit its explicit runtime routing guard')
        if self.probability_dtype not in ('bfloat16', 'float32'):
            raise ValueError('COMPOSE requires the actual registered router probability precision')
        if self.control_encoding not in ('int64_carrier', 'native_typed_fields'):
            raise ValueError('COMPOSE requires a registered native dispatch control encoding')

    @property
    def probability_torch_dtype(self):
        return getattr(torch, self.probability_dtype)

    @property
    def probability_itemsize(self):
        return 2 if self.probability_dtype == 'bfloat16' else 4

    @property
    def packet_dtype(self):
        return self.probability_dtype

    @property
    def packet_torch_dtype(self):
        return getattr(torch, self.packet_dtype)

    @property
    def packet_itemsize(self):
        return self.probability_itemsize

    @property
    def tokens(self):
        return prod(self.hidden_shape[:2])

    @property
    def permuted_tokens(self):
        return self.tokens * self.topk

    @property
    def local_experts(self):
        return self.num_experts // self.expert_size

    @property
    def maximum_received_tokens(self):
        return self.permuted_tokens * self.expert_size if self.receiver_token_capacity is None else self.receiver_token_capacity

    @property
    def reverse_map_capacity(self):
        return self.tokens * (2 * self.num_experts + 1)

    @property
    def control_elements(self):
        # header、原生 split、expert count、routing map 和原生逆置换索引。
        return (16 + 2 * self.expert_size + self.local_experts + self.num_experts
                + self.tokens * self.num_experts + self.reverse_map_capacity)

    @property
    def header_elements(self):
        return 16 + 2 * self.expert_size + self.local_experts + self.num_experts

    @property
    def control_bytes(self):
        if self.control_encoding == 'native_typed_fields':
            return self.header_elements * 8 + self.tokens * self.num_experts + self.reverse_map_capacity * 4
        return self.control_elements * 8

    def packet_shape(self, dispatched_tokens):
        if not 0 <= dispatched_tokens <= self.maximum_received_tokens:
            raise ValueError('actual dispatched token count exceeds the dropless routing bound')
        return (dispatched_tokens + self.tokens, self.hidden_shape[-1] + 1)

    def source_packet_shape(self):
        return self.packet_shape(self.permuted_tokens)


def pack_compose_fields(tokens, probabilities, residual, geometry):
    if (not tokens.is_cuda or tokens.dtype != torch.bfloat16 or tokens.ndim != 2
            or tuple(tokens.shape) != (geometry.permuted_tokens, geometry.hidden_shape[-1])
            or probabilities.dtype != geometry.probability_torch_dtype or tuple(probabilities.shape) != (geometry.permuted_tokens,)
            or residual.dtype != torch.bfloat16 or tuple(residual.shape) != geometry.hidden_shape
            or probabilities.device != tokens.device or residual.device != tokens.device):
        raise ValueError('actual COMPOSE fields differ from their native token and probability precision contracts')
    permuted = torch.cat((tokens.to(geometry.packet_torch_dtype), probabilities.unsqueeze(1)), dim=1)
    residual_rows = residual.reshape(geometry.tokens, -1).to(geometry.packet_torch_dtype)
    padding = residual_rows.new_zeros((geometry.tokens, 1))
    return torch.cat((permuted, torch.cat((residual_rows, padding), dim=1)), dim=0)


class _UnpackComposedFields(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packet, received_tokens, geometry):
        ctx.geometry, ctx.received_tokens = geometry, received_tokens
        hidden = geometry.hidden_shape[-1]
        tokens = packet[:received_tokens, :hidden].to(torch.bfloat16).contiguous()
        probabilities = packet[:received_tokens, hidden].to(geometry.probability_torch_dtype).contiguous()
        residual = packet[received_tokens:, :hidden].to(torch.bfloat16).contiguous().reshape(geometry.hidden_shape)
        return tokens, probabilities, residual

    @staticmethod
    def backward(ctx, token_gradient, probability_gradient, residual_gradient):
        if any(value is None for value in (token_gradient, probability_gradient, residual_gradient)):
            raise ValueError('COMPOSE adjoint requires all actual differentiable payload fields')
        geometry, rows = ctx.geometry, ctx.received_tokens
        gradient = torch.empty(geometry.packet_shape(rows), dtype=geometry.packet_torch_dtype, device=token_gradient.device)
        hidden = geometry.hidden_shape[-1]
        gradient[:rows, :hidden].copy_(token_gradient)
        gradient[:rows, hidden].copy_(probability_gradient)
        gradient[rows:, :hidden].copy_(residual_gradient.reshape(geometry.tokens, hidden))
        gradient[rows:, hidden].zero_()
        return gradient, None, None


def unpack_compose_fields(packet, received_tokens, geometry):
    if (not packet.is_cuda or packet.dtype != geometry.packet_torch_dtype
            or tuple(packet.shape) != geometry.packet_shape(received_tokens)):
        raise ValueError('COMPOSE receiver packet differs from its actual routed token count')
    return _UnpackComposedFields.apply(packet, received_tokens, geometry)


@dataclass(frozen=True)
class NativeDispatchControl:
    header: torch.Tensor
    routing_map: torch.Tensor
    inverse_permutation: torch.Tensor


def allocate_dispatch_control(geometry, device):
    if geometry.control_encoding == 'native_typed_fields':
        return NativeDispatchControl(torch.empty(geometry.header_elements, dtype=torch.int64, device=device),
                                     torch.empty(geometry.tokens * geometry.num_experts, dtype=torch.bool, device=device),
                                     torch.empty(geometry.reverse_map_capacity, dtype=torch.int32, device=device))
    return torch.empty(geometry.control_elements, dtype=torch.int64, device=device)


def dispatch_control_tensors(control):
    if isinstance(control, NativeDispatchControl):
        return control.header, control.routing_map, control.inverse_permutation
    return (control,)


def capture_dispatch_control(dispatcher, geometry, epoch):
    if (dispatcher.tp_size != 1 or dispatcher.ep_size != geometry.expert_size
            or dispatcher.drop_and_pad or dispatcher.config.batch_invariant_mode
            or dispatcher.config.moe_router_padding_for_quantization
            or dispatcher.batch_invariant_inverse_permutation_mapping is not None):
        raise ValueError('COMPOSE requires the actual TP1 dropless native alltoall dispatcher')
    dispatcher.tokens_per_expert = dispatcher._maybe_dtoh_and_synchronize(
        'before_ep_alltoall', dispatcher.tokens_per_expert)
    reverse = dispatcher.reversed_local_input_permutation_mapping
    routing = dispatcher.routing_map
    if (tuple(dispatcher.hidden_shape) != geometry.hidden_shape
            or tuple(routing.shape) != (geometry.tokens, geometry.num_experts)
            or routing.dtype != torch.bool or not routing.is_cuda
            or reverse.dtype not in (torch.int32, torch.int64) or not reverse.is_cuda
            or not 1 <= reverse.ndim <= 4 or reverse.numel() > geometry.reverse_map_capacity
            or dispatcher.num_out_tokens != geometry.permuted_tokens):
        raise ValueError({'message': 'native dispatch control differs from the registered model token layout',
                          'hidden_shape': tuple(dispatcher.hidden_shape), 'expected_hidden_shape': geometry.hidden_shape,
                          'routing_shape': tuple(routing.shape), 'routing_dtype': str(routing.dtype),
                          'routing_device': str(routing.device), 'reverse_shape': tuple(reverse.shape),
                          'reverse_dtype': str(reverse.dtype), 'reverse_device': str(reverse.device),
                          'reverse_elements': reverse.numel(), 'reverse_capacity': geometry.reverse_map_capacity,
                          'num_out_tokens': int(dispatcher.num_out_tokens), 'expected_out_tokens': geometry.permuted_tokens})
    input_splits = tuple(int(value) for value in dispatcher.input_splits)
    output_splits = tuple(int(value) for value in dispatcher.output_splits)
    validate_split_counts(input_splits, output_splits, geometry)
    counts = dispatcher.num_global_tokens_per_local_expert
    if tuple(counts.shape) != (geometry.expert_size, geometry.local_experts):
        raise ValueError('native expert counts differ from actual expert-domain ownership')
    header = torch.tensor((epoch, reverse.ndim, 32 if reverse.dtype == torch.int32 else 64, reverse.numel(),
                           *reverse.shape, *([0] * (4 - reverse.ndim)), *([0] * 8)),
                          dtype=torch.int64, device=routing.device)
    header_pieces = (header, torch.as_tensor(input_splits + output_splits, dtype=torch.int64, device=routing.device),
              dispatcher.tokens_per_expert.to(device=routing.device, dtype=torch.int64),
              counts.to(device=routing.device, dtype=torch.int64).reshape(-1))
    if geometry.control_encoding == 'native_typed_fields':
        if (not dispatcher.config.moe_permute_fusion or reverse.dtype != torch.int32
                or reverse.numel() != geometry.reverse_map_capacity
                or not reverse.is_contiguous() or not routing.is_contiguous()):
            raise ValueError('typed dispatch control requires the actual contiguous TE int32 inverse mapping and bool routing map')
        control = NativeDispatchControl(torch.cat(header_pieces), routing.reshape(-1), reverse.reshape(-1))
        return control, input_splits, output_splits
    pieces = (*header_pieces, routing.to(torch.int64).reshape(-1), reverse.to(torch.int64).reshape(-1))
    used = sum(value.numel() for value in pieces)
    padding = torch.zeros(geometry.control_elements - used, dtype=torch.int64, device=routing.device)
    return torch.cat((*pieces, padding)), input_splits, output_splits


def validate_split_counts(input_splits, output_splits, geometry):
    if (len(input_splits) != geometry.expert_size or len(output_splits) != geometry.expert_size
            or any(value < 0 for value in input_splits + output_splits)
            or sum(input_splits) != geometry.permuted_tokens
            or sum(output_splits) > geometry.maximum_received_tokens):
        raise ValueError('native routing splits violate actual dropless token conservation')


def restore_dispatch_control(dispatcher, control, geometry, expected_epoch):
    typed = geometry.control_encoding == 'native_typed_fields'
    if typed:
        if (not isinstance(control, NativeDispatchControl)
                or any(not value.is_cuda or value.device != control.header.device or not value.is_contiguous()
                       for value in dispatch_control_tensors(control))
                or control.header.dtype != torch.int64 or control.routing_map.dtype != torch.bool
                or control.inverse_permutation.dtype != torch.int32
                or control.header.shape != (geometry.header_elements,)
                or control.routing_map.shape != (geometry.tokens * geometry.num_experts,)
                or control.inverse_permutation.shape != (geometry.reverse_map_capacity,)):
            raise ValueError('native dispatch control fields differ from their actual precision and token geometry')
        metadata = control.header
    else:
        if (not isinstance(control, torch.Tensor) or not control.is_cuda or control.dtype != torch.int64
                or tuple(control.shape) != (geometry.control_elements,)):
            raise ValueError('COMPOSE metadata must use its complete non-gradient CUDA control buffer')
        metadata = control
    header = metadata[:16].cpu().tolist()
    epoch, ndim, width, elements = header[:4]
    reverse_shape = tuple(header[4:4 + ndim])
    if (epoch != expected_epoch or not 1 <= ndim <= 4 or width not in (32, 64)
            or prod(reverse_shape) != elements or not 0 < elements <= geometry.reverse_map_capacity):
        raise ValueError('COMPOSE control epoch or native inverse permutation shape is invalid')
    if typed and (width != 32 or elements != geometry.reverse_map_capacity):
        raise ValueError('typed dispatch control changed its certified native inverse mapping')
    offset = 16
    splits = metadata[offset:offset + 2 * geometry.expert_size].cpu().tolist()
    input_splits, output_splits = tuple(splits[:geometry.expert_size]), tuple(splits[geometry.expert_size:])
    validate_split_counts(input_splits, output_splits, geometry)
    dispatcher.input_splits = np.asarray(input_splits, dtype=np.int64)
    dispatcher.output_splits = np.asarray(output_splits, dtype=np.int64)
    dispatcher.output_splits_tp = np.asarray((sum(output_splits),), dtype=np.int64)
    offset += 2 * geometry.expert_size
    dispatcher.tokens_per_expert = metadata[offset:offset + geometry.local_experts].cpu()
    offset += geometry.local_experts
    counts = metadata[offset:offset + geometry.num_experts].reshape(geometry.expert_size, geometry.local_experts)
    dispatcher.num_global_tokens_per_local_expert = counts if dispatcher.config.moe_permute_fusion else counts.cpu()
    offset += geometry.num_experts
    if typed:
        dispatcher.routing_map = control.routing_map.reshape(geometry.tokens, geometry.num_experts)
        dispatcher.reversed_local_input_permutation_mapping = control.inverse_permutation.reshape(reverse_shape)
    else:
        dispatcher.routing_map = control[offset:offset + geometry.tokens * geometry.num_experts].bool().reshape(
            geometry.tokens, geometry.num_experts)
        offset += geometry.tokens * geometry.num_experts
        dispatcher.reversed_local_input_permutation_mapping = control[offset:offset + elements].to(
            torch.int32 if width == 32 else torch.int64).reshape(reverse_shape)
    dispatcher.hidden_shape = geometry.hidden_shape
    dispatcher.hidden_shape_before_permute = (geometry.tokens, geometry.hidden_shape[-1])
    dispatcher.num_out_tokens = geometry.permuted_tokens
    dispatcher.batch_invariant_inverse_permutation_mapping = None
    dispatcher.cuda_sync_point = 'no_sync'
    # metadata 已完成真实 DtoH，consumer 不会再次读取 source 的同步事件。
    dispatcher.cuda_dtoh_point = 'before_permutation_1'
    return input_splits, output_splits


def build_compose_program(binding, pp_ranks):
    geometry = MoEComposeGeometry(binding.native_shape, binding.num_experts, binding.topk, len(binding.source_expert_ranks),
                                  binding.compose_receiver_token_capacity, binding.probability_dtype, binding.compose_control_encoding)
    source = ValueContract(f'layer:{binding.logical_layer}:routed_token_packet',
                           (geometry.permuted_tokens * geometry.expert_size, geometry.hidden_shape[-1] + 1),
                           Placement(PlacementKind.SHARD, 0), 'source_token_owners', dtype=geometry.packet_dtype,
                           adjoint_key='inverse_native_router_splits', process_group_key='PP_EP_HANDOFF')
    destination = ValueContract(source.logical_value, source.logical_shape, source.placement,
                                'destination_expert_owners', dtype=geometry.packet_dtype, adjoint_key=source.adjoint_key,
                                process_group_key='PP_EP_HANDOFF')
    specs = []
    for rank in binding.source_expert_ranks:
        specs.append(BufferSpecification(rank, 'source_packet_gradient', geometry.source_packet_shape(), geometry.packet_dtype, 0))
        specs.append(BufferSpecification(rank, 'native_dispatch_control',
                                         (geometry.header_elements if geometry.control_encoding == 'native_typed_fields' else geometry.control_elements,), 'int64', 0))
        if geometry.control_encoding == 'native_typed_fields':
            specs.extend((BufferSpecification(rank, 'native_routing_map', (geometry.tokens * geometry.num_experts,), 'bool', 0),
                          BufferSpecification(rank, 'native_inverse_permutation', (geometry.reverse_map_capacity,), 'int32', 0)))
    capacity = min(binding.microbatches, len(pp_ranks) - binding.source_pp - 1)
    for rank in binding.destination_expert_ranks:
        specs.extend(BufferSpecification(rank, 'expert_input_packet', geometry.packet_shape(geometry.maximum_received_tokens),
                                          geometry.packet_dtype, slot) for slot in range(capacity))
        specs.extend(BufferSpecification(rank, 'native_dispatch_control',
                                          (geometry.header_elements if geometry.control_encoding == 'native_typed_fields' else geometry.control_elements,), 'int64', slot)
                     for slot in range(capacity))
        if geometry.control_encoding == 'native_typed_fields':
            specs.extend(BufferSpecification(rank, field, shape, dtype, slot)
                         for field, shape, dtype in (('native_routing_map', (geometry.tokens * geometry.num_experts,), 'bool'),
                                                     ('native_inverse_permutation', (geometry.reverse_map_capacity,), 'int32'))
                         for slot in range(capacity))
    routing = DynamicRoutingMap(binding.source_expert_ranks, binding.destination_expert_ranks,
                                 'native_dispatch_control', geometry.permuted_tokens, geometry.maximum_received_tokens,
                                 'native_input_splits_to_destination_expert_coordinate_in_source_rank_order',
                                 'reverse_actual_source_destination_splits_with_preserved_token_order')
    ownership = tuple((f'layer:{binding.logical_layer}:{kind}', binding.source_expert_ranks)
                      for kind in ('ATTENTION', 'PRE_DISPATCH', 'ROUTER'))
    ownership += tuple((f'layer:{binding.logical_layer}:{kind}', binding.destination_expert_ranks)
                       for kind in ('DISPATCH', 'EXPERT', 'COMBINE'))
    implementation = 'native_moe_COMPOSE_PP_EP_direct_routing'
    witness = ExecutableWitness((('PP_TRANSPORT', tuple(pp_ranks)), ('SOURCE_EP', binding.source_expert_ranks),
                                 ('DESTINATION_EP', binding.destination_expert_ranks),
                                 ('PP_EP_HANDOFF', binding.source_expert_ranks + binding.destination_expert_ranks)),
                                ownership, (),
                                ('SOURCE_ATTENTION_PRENORM_ROUTE', 'NATIVE_PERMUTE1', 'COMPOSED_PP_EP_ROUTING',
                                 'NATIVE_EXPERT_SORT_COMPUTE_COMBINE_UNPERMUTE'),
                                ('NATIVE_EXPERT_COMBINE_ADJOINT', 'INVERSE_COMPOSED_PP_EP_ROUTING',
                                 'NATIVE_PERMUTE1_ROUTER_NORM_ATTENTION_ADJOINT'), tuple(specs),
                                ('native_1F1B', 'actual_dynamic_router_splits', 'matching_microbatch_epoch',
                                 'consumer_schedule_lifetime_before_reuse', 'original_EP4_expert_parameter_ownership'),
                                implementation, (routing,))
    operation = BoundaryOp('composed_PP_EP_dispatch', BoundaryOpKind.FUSED, source, destination, implementation)
    source_bytes = geometry.permuted_tokens * (geometry.hidden_shape[-1] * 2 + geometry.probability_itemsize)
    source_bytes += geometry.tokens * geometry.hidden_shape[-1] * 2 + geometry.control_bytes
    return BoundaryProgram(6 * binding.logical_layer, 6 * binding.logical_layer + 2, 6 * binding.logical_layer + 3,
                           source, destination, (operation,), transferred_bytes=source_bytes,
                           executable_witness_key=witness.identifier(), executable_witness=witness)
