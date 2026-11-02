from dataclasses import dataclass
from types import MethodType

import torch
import torch.distributed as dist

from megatron.core.frontier.megatron.checkpoint_ownership import reindex_local_layer_keys
from megatron.core.frontier.megatron.dense_partition import _retain_region
from megatron.core.frontier.megatron.model_adapter import build_layer_region_graph
from megatron.core.frontier.runtime.moe_payload import pack_router_fields, unpack_router_fields
from megatron.core.frontier.runtime.moe_compose_payload import capture_dispatch_control, pack_compose_fields, unpack_compose_fields


@dataclass(frozen=True)
class MoERouterCut:
    logical_layer: int
    source_pp: int
    variant: str
    source_vp: int = 0

    def __post_init__(self):
        if min(self.logical_layer, self.source_pp, self.source_vp) < 0 or self.variant not in ('canonical', 'move', 'compose'):
            raise ValueError('MoE router cut requires registered region coordinates and transformation')

    def expected_regions(self, decoder_ids, pp_stage, vp_stage=0):
        kinds = ('ATTENTION', 'PRE_DISPATCH', 'ROUTER', 'DISPATCH', 'EXPERT', 'COMBINE')
        source_count = 1 if self.variant == 'canonical' else 3
        result = []
        if pp_stage == self.source_pp + 1 and vp_stage == self.source_vp:
            if not decoder_ids or decoder_ids[0] != self.logical_layer + 1:
                raise ValueError('MoE consumer must immediately follow the split logical layer')
            result.extend(f'layer:{self.logical_layer}:{kind}' for kind in kinds[source_count:])
        for layer in decoder_ids:
            owned = kinds[:source_count] if layer == self.logical_layer and pp_stage == self.source_pp and vp_stage == self.source_vp else kinds
            result.extend(f'layer:{layer}:{kind}' for kind in owned)
        if pp_stage == self.source_pp and vp_stage == self.source_vp and (not decoder_ids or decoder_ids[-1] != self.logical_layer):
            raise ValueError('MoE source stage must end at its compiled attention/router producer')
        return tuple(result)


def compile_moe_region_placement(graph, placement):
    regions = {region.region_id: region for region in graph.regions}
    if set(placement) != set(regions) or graph.scope != 'OPERATOR':
        raise ValueError('Region编译需要完整的实际层图和模块放置')
    coordinates = sorted(set(placement.values()))
    if (len(coordinates) != 2 or coordinates[1][0] != coordinates[0][0] + 1
            or coordinates[0][1] != coordinates[1][1]):
        raise ValueError('当前模块执行能力要求同一Virtual Chunk的相邻PP阶段')
    source, destination = coordinates
    source_regions = tuple(region for region in graph.regions if placement[region.region_id] == source)
    destination_regions = tuple(region for region in graph.regions if placement[region.region_id] == destination)
    if graph.regions != source_regions + destination_regions:
        raise ValueError('模块放置不满足实际计算顺序')
    source_kinds = tuple(region.region_kind.name for region in source_regions)
    variants = {('ATTENTION',): 'canonical', ('ATTENTION', 'PRE_DISPATCH', 'ROUTER'): 'move'}
    if source_kinds not in variants:
        raise ValueError('当前执行器没有该Region切分的Tensor与梯度实现')
    crossing = []
    for connection in graph.connections():
        producer = regions[connection.producer]
        consumer = regions[connection.consumer]
        output = producer.output_contracts[connection.producer_output]
        required = consumer.input_contracts[connection.consumer_input]
        if output != required:
            raise ValueError('Region连接的Tensor契约不一致')
        if placement[connection.producer] != placement[connection.consumer]:
            if placement[connection.producer] != source or placement[connection.consumer] != destination:
                raise ValueError('Region切分产生反向Forward依赖')
            crossing.append(output)
    values = tuple(dict.fromkeys(crossing))
    expected = ('attention_output',) if source_kinds == ('ATTENTION',) else (
        'normalized_hidden', 'residual', 'routing_probabilities', 'routing_map')
    if set(value.logical_value for value in values) != set(expected):
        raise ValueError('跨阶段Tensor集合没有覆盖实际残差和动态路由依赖')
    layers = {region.logical_layer_id for region in graph.regions}
    if len(layers) != 1:
        raise ValueError('层内部切分包含不一致的逻辑层身份')
    cut = MoERouterCut(next(iter(layers)), source[0], variants[source_kinds], source[1])
    return cut, values


def enumerate_moe_region_placements(graph, source_pp, source_vp):
    result = []
    for index, region in enumerate(graph.regions[:-1], 1):
        if not region.transportable_output:
            continue
        placement = {node.region_id: (source_pp if position < index else source_pp + 1, source_vp)
                     for position, node in enumerate(graph.regions)}
        cut, values = compile_moe_region_placement(graph, placement)
        result.append((placement, cut, values))
    return tuple(result)


def source_router_forward(self, hidden_states, attention_mask=None, context=None, context_mask=None,
                          rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                          rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
                          packed_seq_params=None, sequence_len_offset=None, padding_mask=None):
    if context is not None or inference_context is not None or packed_seq_params is not None:
        raise ValueError('registered MoE router MOVE requires actual eager unpacked training tokens')
    hidden, context = self._forward_attention(hidden_states, attention_mask=attention_mask,
                                              context_mask=context_mask, rotary_pos_emb=rotary_pos_emb,
                                              rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                                              rotary_pos_cos_sin=rotary_pos_cos_sin, attention_bias=attention_bias,
                                              sequence_len_offset=sequence_len_offset, padding_mask=padding_mask)
    with torch.cuda.nvtx.range('Frontier/MoE/PRE_DISPATCH_ROUTER_MOVE'):
        normalized, residual, mlp_state = self._pre_mlp_layernorm_and_residual(hidden)
        if mlp_state:
            raise ValueError('registered MoE cut does not own additional residual mixing state')
        mask = padding_mask.transpose(0, 1).bool() if padding_mask is not None else None
        probabilities, routing_map = self.mlp.route(normalized, mask)
        if self.frontier_payload_binding.variant == 'compose':
            manager = self.frontier_compose_manager()
            permuted, probabilities = self.mlp.preprocess(normalized, probabilities, routing_map)
            control, input_splits, output_splits = capture_dispatch_control(
                self.mlp.token_dispatcher, manager.geometry, manager._epoch(manager.counts['forward_send']))
            manager.enqueue_source(control, input_splits, output_splits)
            return pack_compose_fields(permuted, probabilities, residual, manager.geometry), context
        return pack_router_fields(normalized, residual, probabilities, routing_map, self.frontier_payload_binding), context


def consumer_expert_forward(self, hidden_states, attention_mask=None, context=None, context_mask=None,
                            rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                            rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
                            packed_seq_params=None, sequence_len_offset=None, padding_mask=None):
    if self.frontier_router_fields is None or context is not None or inference_context is not None or packed_seq_params is not None:
        raise ValueError('MoE expert consumer requires every actual typed routing field')
    residual, probabilities, routing_map = self.frontier_router_fields
    self.frontier_router_fields = None
    with torch.cuda.nvtx.range('Frontier/MoE/DISPATCH_EXPERT_COMBINE'):
        if self.frontier_payload_binding.variant == 'compose':
            dispatched = self.frontier_composed_input
            self.frontier_composed_input = None
            if dispatched is None:
                raise ValueError('COMPOSE expert consumer omitted its actual dispatched tokens')
        else:
            permuted, probabilities = self.mlp.preprocess(hidden_states, probabilities, routing_map)
            dispatched, probabilities = self.mlp.dispatch(permuted, probabilities)
        output, bias = self.mlp.routed_experts_compute(dispatched, probabilities)
        if bias is not None:
            raise ValueError('registered MoE consumer requires actual bias-free expert output')
        combined = self.mlp.combine(output)
        output = self.mlp.postprocess(combined, None)
        return self._apply_mlp_bda_step((output, bias), residual, ()), context


def retain_moe_region(layer, source, binding):
    original = set(dict(layer.named_parameters()))
    if source:
        expected = {key for key in original if not key.startswith('mlp.') or key.startswith('mlp.router.')}
        del layer.mlp.experts
        layer.forward = MethodType(source_router_forward, layer)
        layer.frontier_region_kind = 'MOE_ATTENTION_PRE_DISPATCH_ROUTER'
    else:
        expected = {key for key in original if key.startswith('mlp.') and not key.startswith('mlp.router.')}
        del layer.mlp.router
        del layer.pre_mlp_layernorm
        for name, child in tuple(layer.named_children()):
            if name != 'mlp' and tuple(child.parameters()):
                delattr(layer, name)
        layer.forward = MethodType(consumer_expert_forward, layer)
        layer.frontier_region_kind = 'MOE_DISPATCH_EXPERT_COMBINE'
        layer.frontier_router_fields = None
        layer.frontier_composed_input = None
    if not expected or set(dict(layer.named_parameters())) != expected:
        raise ValueError('actual router/expert parameter ownership differs from the MOVE program')
    layer.frontier_payload_binding = binding
    layer.frontier_original_parameter_keys = tuple(sorted(original))
    return layer


def moe_region_ids(block):
    kinds = ('ATTENTION', 'PRE_DISPATCH', 'ROUTER', 'DISPATCH', 'EXPERT', 'COMBINE')
    owned = {'ATTENTION': kinds[:1], 'MLP': kinds[1:],
             'MOE_ATTENTION_PRE_DISPATCH_ROUTER': kinds[:3], 'MOE_DISPATCH_EXPERT_COMBINE': kinds[3:]}
    result = []
    for layer in block.layers:
        kind = getattr(layer, 'frontier_region_kind', None)
        selected = kinds if kind is None else owned[kind]
        result.extend(f'layer:{layer.layer_number - 1}:{region}' for region in selected)
    return tuple(result)


def install_moe_router_partition(cut, binding, compose_instances=None):
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if (cut.variant != binding.variant or cut.logical_layer != binding.logical_layer or cut.source_pp != binding.source_pp
            or getattr(TransformerBlock, '_frontier_moe_cut', None) is not None):
        raise ValueError('one concrete MoE program must own model construction and typed transport')
    if (cut.variant == 'compose') != (compose_instances is not None):
        raise ValueError('COMPOSE model construction requires its actual shared transport instances')

    def manager():
        if len(compose_instances) != 1:
            raise ValueError('actual MoE model must own exactly one compiled COMPOSE communicator')
        return compose_instances[0]
    original_build = TransformerBlock._build_layers
    original_sharded = TransformerBlock.sharded_state_dict
    original_input = GPTModel.set_input_tensor

    def build(self):
        config = self.config
        probability_dtype = {None: 'bfloat16', 'fp32': 'float32'}[config.moe_router_dtype]
        if probability_dtype != binding.probability_dtype or config.fp32_residual_connection:
            raise ValueError('typed MoE fields must preserve actual router and residual precision')
        if (config.num_moe_experts != binding.num_experts or config.moe_router_topk != binding.topk
                or config.tensor_model_parallel_size != 1 or config.expert_tensor_parallel_size != 1
                or config.context_parallel_size != binding.context_parallel_size
                or config.moe_token_dispatcher_type != 'alltoall' or config.moe_input_jitter_eps is not None
                or config.moe_shared_expert_intermediate_size is not None or config.moe_latent_size
                or config.moe_expert_capacity_factor is not None or config.moe_router_force_load_balancing
                or config.overlap_dispatch_backward_with_experts_wgrad or config.enable_mhc_connections
                or config.fp8 or config.fp4 or not config.bf16 or config.hidden_dropout != 0
                or config.attention_dropout != 0 or config.recompute_granularity is not None
                or config.fine_grained_activation_offloading
                or (config.virtual_pipeline_model_parallel_size or 1) != binding.virtual_pipeline_size
                or config.cuda_graph_impl not in (None, 'none')):
            raise ValueError('MoE cut capability requires actual bias-free dropless TP1/ETP1/CP2 eager training')
        original_build(self)
        stage = dist.get_rank(self.pg_collection.pp)
        original_layers = [layer.layer_number - 1 for layer in self.layers]
        vp_stage = self.vp_stage or 0
        expected = cut.expected_regions(original_layers, stage, vp_stage)
        self.frontier_semantic_graphs = tuple(build_layer_region_graph(layer, config.seq_length,
            binding.native_shape[1], 'native_model_groups') for layer in self.layers)
        if stage == cut.source_pp and vp_stage == cut.source_vp:
            graph = self.frontier_semantic_graphs[-1]
            candidates = enumerate_moe_region_placements(graph, stage, vp_stage)
            selected = [item for item in candidates if item[1] == cut]
            if len(selected) != 1:
                raise ValueError('实际模型图没有对应的合法Region放置')
            self.frontier_compiled_region_placement = selected[0]
            if cut.variant == 'canonical':
                _retain_region(self.layers[-1], 'ATTENTION')
            else:
                retain_moe_region(self.layers[-1], True, binding)
        elif stage == cut.source_pp + 1 and vp_stage == cut.source_vp:
            layer = build_module(self.submodules.layer_specs[0], config=config, layer_number=cut.logical_layer + 1,
                                 pg_collection=self.pg_collection, vp_stage=self.vp_stage, add_layer_offset=False)
            self.layers.insert(0, _retain_region(layer, 'MLP') if cut.variant == 'canonical' else retain_moe_region(layer, False, binding))
            self.num_layers_per_pipeline_rank = len(self.layers)
        self.frontier_region_ids = moe_region_ids(self)
        if self.frontier_region_ids != expected:
            raise ValueError('constructed MoE regions differ from compiled global logical ownership')
        if cut.variant == 'compose' and stage in (cut.source_pp, cut.source_pp + 1) and vp_stage == cut.source_vp:
            self.layers[-1 if stage == cut.source_pp else 0].frontier_compose_manager = manager

    def sharded(self, prefix='', sharded_offsets=(), metadata=None):
        state = original_sharded(self, prefix, sharded_offsets, metadata)
        offset = get_transformer_layer_offset(self.config, self.vp_stage, dist.get_rank(self.pg_collection.pp))
        return reindex_local_layer_keys(state, self.layers, offset, prefix)

    def set_input(self, input_tensor):
        if (cut.variant == 'canonical' or dist.get_rank(self.decoder.pg_collection.pp) != cut.source_pp + 1
                or (self.decoder.vp_stage or 0) != cut.source_vp):
            return original_input(self, input_tensor)
        packets = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if len(packets) != 1 or packets[0] is None:
            raise ValueError('MoE consumer requires exactly one complete typed model packet')
        consumer = self.decoder.layers[0]
        if consumer.frontier_router_fields is not None or consumer.frontier_region_kind != 'MOE_DISPATCH_EXPERT_COMBINE':
            raise ValueError('actual expert consumer is unavailable for its next routing payload')
        if cut.variant == 'compose':
            transport = manager()
            received = transport.take_receiver(consumer.mlp.token_dispatcher)
            dispatched, probabilities, residual = unpack_compose_fields(packets[0], received, transport.geometry)
            consumer.frontier_router_fields = residual, probabilities, consumer.mlp.token_dispatcher.routing_map
            consumer.frontier_composed_input = dispatched
            return original_input(self, [residual])
        normalized, residual, probabilities, routing_map = unpack_router_fields(packets[0], binding)
        consumer.frontier_router_fields = residual, probabilities, routing_map
        return original_input(self, [normalized])

    TransformerBlock._build_layers = build
    TransformerBlock.sharded_state_dict = sharded
    TransformerBlock._frontier_moe_cut = cut
    GPTModel.set_input_tensor = set_input
    return cut
