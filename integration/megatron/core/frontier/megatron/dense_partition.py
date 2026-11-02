from dataclasses import dataclass
from types import MethodType


@dataclass(frozen=True)
class DenseRegionCut:
    logical_layer: int
    source_pp: int
    source_vp: int = 0

    def __post_init__(self):
        if min(self.logical_layer, self.source_pp, self.source_vp) < 0:
            raise ValueError('region cut coordinates must be nonnegative')

    def expected_regions(self, decoder_ids, pp_stage, vp_stage):
        regions = []
        if (pp_stage, vp_stage) == (self.source_pp + 1, self.source_vp):
            if not decoder_ids or decoder_ids[0] != self.logical_layer + 1:
                raise ValueError('consumer chunk must immediately follow the split layer')
            regions.append(f'layer:{self.logical_layer}:MLP')
        for logical_layer in decoder_ids:
            regions.append(f'layer:{logical_layer}:ATTENTION')
            if (logical_layer, pp_stage, vp_stage) != (self.logical_layer, self.source_pp, self.source_vp):
                regions.append(f'layer:{logical_layer}:MLP')
        if (pp_stage, vp_stage) == (self.source_pp, self.source_vp):
            if not decoder_ids or decoder_ids[-1] != self.logical_layer:
                raise ValueError('source chunk must end with the split attention region')
        return tuple(regions)


def _attention_forward(self, hidden_states, attention_mask=None, context=None, context_mask=None,
                       rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                       rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
                       packed_seq_params=None, sequence_len_offset=None, padding_mask=None):
    import torch
    from megatron.core.utils import make_viewless_tensor

    if inference_context is not None or context is not None:
        raise ValueError('registered region execution requires self-attention training')
    with torch.cuda.nvtx.range(f'Frontier/layer:{self.layer_number - 1}:ATTENTION'):
        output, updated_context = self._forward_attention(hidden_states, attention_mask=attention_mask, context=context,
                                                          context_mask=context_mask, rotary_pos_emb=rotary_pos_emb,
                                                          rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                                                          rotary_pos_cos_sin=rotary_pos_cos_sin, attention_bias=attention_bias,
                                                          packed_seq_params=packed_seq_params,
                                                          sequence_len_offset=sequence_len_offset, padding_mask=padding_mask)
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        repair = getattr(self, 'frontier_output_repair', None)
        if repair is not None:
            executor, connection = repair
            output = executor.forward(connection, output)
        return output, updated_context


def _mlp_forward(self, hidden_states, attention_mask=None, context=None, context_mask=None,
                 rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                 rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
                 packed_seq_params=None, sequence_len_offset=None, padding_mask=None):
    import torch

    if inference_context is not None or context is not None:
        raise ValueError('registered region execution requires self-attention training')
    with torch.cuda.nvtx.range(f'Frontier/layer:{self.layer_number - 1}:MLP'):
        output = self._forward_mlp(hidden_states, padding_mask=padding_mask,
                                 packed_seq_params=packed_seq_params)
        repair = getattr(self, 'frontier_output_repair', None)
        if repair is not None:
            executor, connection = repair
            output = executor.forward(connection, output)
        return output, context


def _retain_region(layer, kind):
    owned = ('mlp', 'pre_mlp_layernorm')
    original = dict(layer.named_parameters())
    expected = {key for key in original if key.startswith(tuple(name + '.' for name in owned))}
    if kind == 'ATTENTION':
        expected = set(original) - expected
    for name, child in tuple(layer.named_children()):
        if (name in owned) != (kind == 'MLP') and (name in owned or tuple(child.parameters())):
            delattr(layer, name)
    actual = dict(layer.named_parameters())
    if not actual or set(actual) != expected:
        raise ValueError('region parameter ownership differs from the actual transformer modules')
    layer.frontier_region_kind = kind
    layer.frontier_original_parameter_keys = tuple(sorted(original))
    layer.forward = MethodType(_attention_forward if kind == 'ATTENTION' else _mlp_forward, layer)
    return layer


def region_ids(block):
    regions = []
    for layer in block.layers:
        logical_layer = layer.layer_number - 1
        kind = getattr(layer, 'frontier_region_kind', None)
        regions.extend((f'layer:{logical_layer}:{kind}',) if kind else
                       (f'layer:{logical_layer}:ATTENTION', f'layer:{logical_layer}:MLP'))
    return tuple(regions)


def install_dense_region_partition(cut):
    import torch.distributed as dist

    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_block import TransformerBlock

    if getattr(TransformerBlock, '_frontier_region_cut', None) is not None:
        raise RuntimeError('one compiled region partition must own the model builder')
    original_build = TransformerBlock._build_layers
    original_sharded = TransformerBlock.sharded_state_dict

    def build(self):
        config = self.config
        if (config.num_moe_experts is not None or config.fp8 or config.fp4
                or config.enable_mhc_connections or config.recompute_granularity == 'full'
                or config.cuda_graph_impl not in (None, 'none')):
            raise ValueError('region capability requires dense BF16 eager transformer callables')
        original_build(self)
        stage = dist.get_rank(self.pg_collection.pp)
        vp_stage = 0 if self.vp_stage is None else self.vp_stage
        expected = [layer.layer_number - 1 for layer in self.layers]
        cut.expected_regions(expected, stage, vp_stage)
        if (stage, vp_stage) == (cut.source_pp, cut.source_vp):
            _retain_region(self.layers[-1], 'ATTENTION')
        elif (stage, vp_stage) == (cut.source_pp + 1, cut.source_vp):
            layer = build_module(self.submodules.layer_specs[0], config=config,
                                 layer_number=cut.logical_layer + 1, pg_collection=self.pg_collection,
                                 vp_stage=self.vp_stage, add_layer_offset=False)
            self.layers.insert(0, _retain_region(layer, 'MLP'))
            self.num_layers_per_pipeline_rank = len(self.layers)
        self.frontier_region_ids = region_ids(self)
        if self.frontier_region_ids != cut.expected_regions(expected, stage, vp_stage):
            raise ValueError('constructed model ownership differs from the compiled region partition')

    def sharded(self, prefix='', sharded_offsets=(), metadata=None):
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        from megatron.core.frontier.megatron.checkpoint_ownership import reindex_local_layer_keys

        result = original_sharded(self, prefix, sharded_offsets, metadata)
        offset = get_transformer_layer_offset(self.config, self.vp_stage, dist.get_rank(self.pg_collection.pp))
        return reindex_local_layer_keys(result, self.layers, offset, prefix)

    TransformerBlock._build_layers = build
    TransformerBlock.sharded_state_dict = sharded
    TransformerBlock._frontier_region_cut = cut
    return cut
