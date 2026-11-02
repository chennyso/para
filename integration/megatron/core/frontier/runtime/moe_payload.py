from dataclasses import dataclass
from math import prod

import torch

from megatron.core.frontier.boundary.payload import BoundaryPayload, PayloadField
from megatron.core.frontier.ir.boundary import BoundaryOp, BoundaryOpKind, BoundaryProgram, BufferSpecification, ExecutableWitness
from megatron.core.frontier.ir.contract import Placement, PlacementKind, ValueContract


@dataclass(frozen=True)
class MoERouterPayloadBinding:
    logical_layer: int
    source_pp: int
    native_shape: tuple[int, ...]
    microbatches: int
    num_experts: int
    topk: int
    context_parallel_size: int
    source_expert_ranks: tuple[int, ...]
    destination_expert_ranks: tuple[int, ...]
    context_coordinates: tuple[int, ...]
    variant: str
    tp_size: int = 1
    native_dtype: str = 'bfloat16'
    compose_receiver_token_capacity: int | None = None
    probability_dtype: str = 'bfloat16'
    virtual_pipeline_size: int = 1
    source_vp: int = 0
    compose_control_encoding: str = 'int64_carrier'

    def __post_init__(self):
        if (self.variant not in ('canonical', 'move', 'compose') or self.tp_size != 1
                or self.native_dtype != 'bfloat16' or self.context_parallel_size != 2):
            raise ValueError('registered MoE router cut requires BF16 TP1 CP2 and an executable router boundary')
        if (len(self.native_shape) != 3 or min(self.native_shape) < 1 or self.native_shape[0] % 2
                or self.microbatches < 1 or self.logical_layer < 0 or self.source_pp < 0
                or not 0 < self.topk <= self.num_experts):
            raise ValueError('MoE payload requires actual token, expert and training geometry')
        if (len(self.source_expert_ranks) != len(self.destination_expert_ranks)
                or not self.source_expert_ranks or set(self.source_expert_ranks) & set(self.destination_expert_ranks)
                or len(set(self.source_expert_ranks + self.destination_expert_ranks)) != 2 * len(self.source_expert_ranks)):
            raise ValueError('MoE payload requires disjoint equally sized actual expert domains')
        if (len(self.context_coordinates) != len(self.source_expert_ranks)
                or any(not 0 <= coordinate < self.context_parallel_size for coordinate in self.context_coordinates)):
            raise ValueError('MoE payload requires the actual CP coordinate of every expert-domain source')
        if self.compose_receiver_token_capacity is not None and self.variant != 'compose':
            raise ValueError('dynamic expert-token allocation guard belongs to the COMPOSE transport')
        if self.probability_dtype not in ('bfloat16', 'float32'):
            raise ValueError('MoE boundary requires a registered actual router probability precision')
        if not 0 <= self.source_vp < self.virtual_pipeline_size:
            raise ValueError('MoE boundary requires a valid producer virtual chunk')
        if self.variant == 'compose' and self.virtual_pipeline_size != 1:
            raise ValueError('COMPOSE requires its certified single-chunk collective schedule')
        if self.compose_control_encoding not in ('int64_carrier', 'native_typed_fields'):
            raise ValueError('MoE boundary requires a registered native dispatch control representation')

    @property
    def dtype(self):
        if self.variant == 'compose':
            return self.probability_dtype
        return self.native_dtype if self.variant == 'canonical' else 'float32'

    @property
    def local_shape(self):
        sequence, batch, hidden = self.native_shape
        if self.variant == 'compose':
            return (sequence * batch * (self.topk + 1), hidden + 1)
        return (sequence, batch, hidden * 2 + self.num_experts * 2) if self.variant == 'move' else self.native_shape

    @property
    def payload(self):
        sequence, batch, hidden = self.native_shape
        shape = (sequence * self.context_parallel_size, batch, hidden)
        placement = Placement(PlacementKind.SHARD, 0)
        prefix = f'layer:{self.logical_layer}:'
        if self.variant == 'compose':
            from megatron.core.frontier.runtime.moe_compose_payload import MoEComposeGeometry

            geometry = MoEComposeGeometry(self.native_shape, self.num_experts, self.topk, len(self.source_expert_ranks),
                                          probability_dtype=self.probability_dtype, control_encoding=self.compose_control_encoding)
            rows = geometry.permuted_tokens * geometry.expert_size
            return BoundaryPayload((
                PayloadField(prefix + 'permuted_tokens', (rows, hidden), 'bfloat16', placement, True, False, 'inverse_router_splits'),
                PayloadField(prefix + 'permuted_probabilities', (rows,), self.probability_dtype, placement, True, False, 'inverse_router_splits'),
                PayloadField(prefix + 'residual', shape, 'bfloat16', placement, True, False, 'paired_PP'),
                PayloadField(prefix + 'native_dispatch_control',
                             (geometry.header_elements if geometry.control_encoding == 'native_typed_fields' else geometry.control_elements,), 'int64',
                             placement, False, False, 'none'),
                *((PayloadField(prefix + 'native_routing_map', (geometry.tokens * geometry.num_experts,), 'bool', placement, False, False, 'none'),
                   PayloadField(prefix + 'native_inverse_permutation', (geometry.reverse_map_capacity,), 'int32', placement, False, False, 'none'))
                  if geometry.control_encoding == 'native_typed_fields' else ())))
        if self.variant == 'canonical':
            return BoundaryPayload((PayloadField(prefix + 'attention_output', shape, 'bfloat16', placement,
                                                  True, False, 'identity'),))
        routing_shape = shape[:2] + (self.num_experts,)
        return BoundaryPayload((PayloadField(prefix + 'normalized_hidden', shape, 'bfloat16', placement, True, False, 'identity'),
                                PayloadField(prefix + 'residual', shape, 'bfloat16', placement, True, False, 'identity'),
                                PayloadField(prefix + 'router_probabilities', routing_shape, self.probability_dtype, placement, True, False, 'identity'),
                                PayloadField(prefix + 'routing_map', routing_shape, 'bool', placement, False, False, 'none')))

    def payload_metadata(self):
        return {'storage_dtype': self.dtype, 'storage_shape': self.local_shape,
                'fields': [{'name': field.name, 'logical_shape': field.shape, 'dtype': field.dtype,
                            'placement': {'kind': field.placement.kind.name, 'dimension': field.placement.dim},
                            'gradient_eligible': field.gradient_eligible, 'adjoint_key': field.adjoint_key}
                           for field in self.payload.fields],
                'encoding': {'canonical': 'native_BF16_attention_output',
                             'move': 'lossless_native_fields_in_FP32_carrier_with_exact_boolean_values',
                             'compose': 'native_BF16_routed_tokens_native_precision_probabilities_and_int64_control'}[self.variant],
                'context_partition': 'native_CP_two_mirrored_sequence_intervals'}

    def program_for_group(self, pp_ranks, tp_rank):
        if tp_rank != 0 or self.source_pp + 1 >= len(pp_ranks):
            raise ValueError('registered MoE payload differs from actual TP1/PP ownership')
        source, destination = pp_ranks[self.source_pp:self.source_pp + 2]
        if (source not in self.source_expert_ranks or destination not in self.destination_expert_ranks
                or self.source_expert_ranks.index(source) != self.destination_expert_ranks.index(destination)):
            raise ValueError('actual PP pairs must preserve the registered MoE expert-domain coordinate')
        if self.variant == 'compose':
            from megatron.core.frontier.runtime.moe_compose_payload import build_compose_program

            return build_compose_program(self, pp_ranks)
        shape = (self.native_shape[0] * self.context_parallel_size,) + self.local_shape[1:]
        placement = Placement(PlacementKind.SHARD, 0)
        name = f'layer:{self.logical_layer}:moe_boundary_packet'
        source_contract = ValueContract(name, shape, placement, 'source_cp', dtype=self.dtype, process_group_key='PP_TRANSPORT')
        destination_contract = ValueContract(name, shape, placement, 'destination_cp', dtype=self.dtype, process_group_key='PP_TRANSPORT')
        specs = tuple(BufferSpecification(rank, field, self.local_shape, self.dtype, slot)
                      for rank, field in ((source, 'gradient_packet'), (destination, 'input_packet'))
                      for slot in range(self.microbatches))
        source_kinds = ('ATTENTION', 'PRE_DISPATCH', 'ROUTER') if self.variant == 'move' else ('ATTENTION',)
        destination_kinds = ('DISPATCH', 'EXPERT', 'COMBINE') if self.variant == 'move' else (
            'PRE_DISPATCH', 'ROUTER', 'DISPATCH', 'EXPERT', 'COMBINE')
        ownership = tuple((f'layer:{self.logical_layer}:{kind}', (source,)) for kind in source_kinds)
        ownership += tuple((f'layer:{self.logical_layer}:{kind}', (destination,)) for kind in destination_kinds)
        implementation = 'native_moe_router_MOVE_typed_PP' if self.variant == 'move' else 'native_moe_attention_PP_cut'
        context_rank = self.context_coordinates[self.source_expert_ranks.index(source)]
        half = self.native_shape[0] // 2
        global_sequence = self.native_shape[0] * self.context_parallel_size
        intervals = ((context_rank * half, (context_rank + 1) * half),
                     (global_sequence - (context_rank + 1) * half, global_sequence - context_rank * half))
        witness = ExecutableWitness((('PP_TRANSPORT', tuple(pp_ranks)), ('SOURCE_EP', self.source_expert_ranks),
                                     ('DESTINATION_EP', self.destination_expert_ranks)), ownership,
                                    tuple((source, destination, 0, start, end) for start, end in intervals),
                                    ('SOURCE_ATTENTION', 'SOURCE_PRENORM_ROUTE', 'TYPED_PP_HANDOFF', 'DESTINATION_EP_DISPATCH',
                                     'DESTINATION_EXPERT_COMBINE') if self.variant == 'move' else (
                                        'SOURCE_ATTENTION', 'TYPED_PP_HANDOFF', 'DESTINATION_PRENORM_ROUTE', 'DESTINATION_EP_DISPATCH',
                                        'DESTINATION_EXPERT_COMBINE'),
                                    ('EXPERT_COMBINE_ADJOINT', 'PROBABILITY_ADJOINT', 'TYPED_PP_GRADIENT', 'SOURCE_ROUTER_NORM_ADJOINT')
                                    if self.variant == 'move' else ('EXPERT_COMBINE_ADJOINT', 'DESTINATION_ROUTER_NORM_ADJOINT',
                                                                  'TYPED_PP_GRADIENT', 'SOURCE_ATTENTION_ADJOINT'),
                                    specs, ('native_1F1B' if self.virtual_pipeline_size == 1 else 'native_interleaved_1F1B',
                                            f'producer_virtual_chunk:{self.source_vp}', 'lossless_typed_fields', 'one_slot_per_microbatch',
                                            'global_checkpoint_ownership', 'actual_CP_token_order'), implementation)
        operation = BoundaryOp('moe_router_handoff', BoundaryOpKind.ROUTE if self.variant == 'move' else BoundaryOpKind.TRANSFER,
                               source_contract, destination_contract, implementation,
                               movable_before_region=6 * self.logical_layer + 3 if self.variant == 'move' else None)
        return BoundaryProgram(6 * self.logical_layer, 6 * self.logical_layer + len(source_kinds) - 1,
                               6 * self.logical_layer + len(source_kinds), source_contract, destination_contract,
                               (operation,), transferred_bytes=prod(self.local_shape) * (4 if self.variant == 'move' else 2),
                               executable_witness_key=witness.identifier(), executable_witness=witness)


def pack_router_fields(normalized, residual, probabilities, routing_map, binding):
    if (not normalized.is_cuda or normalized.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16
            or tuple(normalized.shape) != binding.native_shape or residual.shape != normalized.shape
            or probabilities.dtype != getattr(torch, binding.probability_dtype) or routing_map.dtype != torch.bool
            or tuple(probabilities.shape) != (normalized.shape[0] * normalized.shape[1], binding.num_experts)
            or routing_map.shape != probabilities.shape or binding.variant != 'move'
            or any(value.device != normalized.device for value in (residual, probabilities, routing_map))):
        raise ValueError({'message': 'actual MoE router tensors differ from their lossless typed boundary fields',
                          'actual': [(tuple(value.shape), str(value.dtype), str(value.device))
                                     for value in (normalized, residual, probabilities, routing_map)],
                          'expected_shape': binding.native_shape, 'probability_dtype': binding.probability_dtype})
    rows = binding.native_shape[:2] + (binding.num_experts,)
    return torch.cat((normalized.float(), residual.float(), probabilities.reshape(rows).float(),
                      routing_map.reshape(rows).float()), dim=-1)


def unpack_router_fields(packet, binding):
    if not packet.is_cuda or tuple(packet.shape) != binding.local_shape or packet.dtype != torch.float32:
        raise ValueError('actual MoE model packet differs from its compiled physical storage')
    hidden, experts = binding.native_shape[-1], binding.num_experts
    normalized = packet.narrow(-1, 0, hidden).to(torch.bfloat16)
    residual = packet.narrow(-1, hidden, hidden).to(torch.bfloat16)
    probabilities = packet.narrow(-1, hidden * 2, experts).reshape(-1, experts).to(
        getattr(torch, binding.probability_dtype)).contiguous()
    routing_map = packet.narrow(-1, hidden * 2 + experts, experts).reshape(-1, experts).bool()
    return normalized, residual, probabilities, routing_map
