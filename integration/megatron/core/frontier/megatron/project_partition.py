from types import MethodType

import torch
import torch.distributed as dist

from megatron.core.frontier.megatron.checkpoint_ownership import reindex_local_layer_keys


class ResidualBDALayer(torch.nn.Module):
    def __init__(self, config, layer_number, bda):
        super().__init__()
        self.config, self.layer_number = config, layer_number
        self.bda = bda
        self.residual = None
        self.frontier_region_kind = 'MLP_RESOLVE_BDA'

    def forward(self, hidden_states, attention_mask=None, context=None, context_mask=None,
                rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
                packed_seq_params=None, sequence_len_offset=None, padding_mask=None):
        from megatron.core.utils import make_viewless_tensor

        if self.residual is None or context is not None or inference_context is not None:
            raise ValueError('consumer BDA requires the actual residual field from its typed model packet')
        if self.residual.shape != hidden_states.shape or self.residual.dtype != hidden_states.dtype:
            raise ValueError('resolved row output and its actual residual have incompatible contracts')
        with torch.enable_grad(), torch.cuda.nvtx.range('Frontier/MLP_RESOLVE_BDA'):
            output = self.bda(self.training, self.config.bias_dropout_fusion)(
                (hidden_states, None), self.residual, self.config.hidden_dropout)
        self.residual = None
        return make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True), context

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        if tuple(self.parameters()):
            raise ValueError('residual BDA region must retain its actual parameter-free ownership')
        return {}


def row_packet_forward(self, hidden_states, attention_mask=None, context=None, context_mask=None,
                       rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                       rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
                       packed_seq_params=None, sequence_len_offset=None, padding_mask=None):
    if context is not None or inference_context is not None:
        raise ValueError('registered partial row boundary requires dense self-attention training')
    hidden, context = self._forward_attention(hidden_states, attention_mask=attention_mask,
                                              context_mask=context_mask, rotary_pos_emb=rotary_pos_emb,
                                              rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
                                              rotary_pos_cos_sin=rotary_pos_cos_sin, attention_bias=attention_bias,
                                              packed_seq_params=packed_seq_params,
                                              sequence_len_offset=sequence_len_offset, padding_mask=padding_mask)
    with torch.cuda.nvtx.range('Frontier/MLP_ROW_OUTPUT'):
        (output, bias), residual = self._forward_mlp_output_with_bias(hidden, padding_mask=padding_mask,
                                                                      packed_seq_params=packed_seq_params)
        if bias is not None or output.dtype != residual.dtype or output.shape[1:] != residual.shape[1:]:
            raise ValueError('registered row packet requires its actual bias-free BF16 output and residual')
        if self.mlp_norm_manager is not None:
            output = self.mlp_norm_manager.group_offload(output)
            self.mlp_norm_manager = None
        if getattr(self, 'frontier_native_row_diagnostic', None) is not None:
            with torch.no_grad():
                expected = torch.empty_like(residual)
                dist.reduce_scatter_tensor(expected, output.detach().contiguous(), group=self.mlp.linear_fc2.tp_group)
                torch.save(expected.cpu(), self.frontier_native_row_diagnostic / f'rank{dist.get_rank()}.native-row.pt')
            self.frontier_native_row_diagnostic = None
        return torch.cat((output, residual), dim=0), context


def install_project_partition(logical_layer, source_pp, variant, shard_shape):
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if variant not in ('canonical', 'project') or getattr(TransformerBlock, '_frontier_row_cut', None) is not None:
        raise ValueError('one registered canonical or PROJECT row program must own the model builder')
    original_build = TransformerBlock._build_layers
    original_sharded = TransformerBlock.sharded_state_dict
    original_input = GPTModel.set_input_tensor

    def build(self):
        config = self.config
        if (config.num_moe_experts is not None or config.fp8 or config.fp4 or config.hidden_dropout != 0
                or config.fp32_residual_connection or config.enable_mhc_connections
                or config.recompute_granularity is not None or config.fine_grained_activation_offloading
                or config.cuda_graph_impl not in (None, 'none') or config.virtual_pipeline_model_parallel_size is not None):
            raise ValueError('row boundary capability requires actual dense bias-free BF16 eager 1F1B')
        original_build(self)
        stage = dist.get_rank(self.pg_collection.pp)
        if stage == source_pp:
            layer = self.layers[-1]
            if layer.layer_number - 1 != logical_layer:
                raise ValueError('row producer must be the actual last layer of its compiled source stage')
            row = layer.mlp.linear_fc2
            if (row.parallel_mode != 'row' or row.use_bias or row.ub_overlap_rs_fprop
                    or row.ub_overlap_ag_dgrad or not row.sequence_parallel):
                raise ValueError('actual TE row kernel does not match the registered partial-output capability')
            layer.frontier_defer_row_resolve = variant == 'project'
            layer.frontier_region_kind = 'MLP_ROW_PARTIAL' if variant == 'project' else 'MLP_ROW_RESOLVED'
            layer.forward = MethodType(row_packet_forward, layer)
        elif stage == source_pp + 1:
            if self.layers[0].layer_number - 1 != logical_layer + 1:
                raise ValueError('row consumer must immediately follow the compiled source layer')
            operation = build_module(self.submodules.layer_specs[0].submodules.mlp_bda)
            self.layers.insert(0, ResidualBDALayer(config, logical_layer + 1, operation))
            self.num_layers_per_pipeline_rank = len(self.layers)

    def sharded(self, prefix='', sharded_offsets=(), metadata=None):
        state = original_sharded(self, prefix, sharded_offsets, metadata)
        offset = get_transformer_layer_offset(self.config, self.vp_stage, dist.get_rank(self.pg_collection.pp))
        return reindex_local_layer_keys(state, self.layers, offset, prefix)

    def set_input(self, input_tensor):
        stage = dist.get_rank(self.decoder.pg_collection.pp)
        if stage != source_pp + 1:
            return original_input(self, input_tensor)
        packets = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if len(packets) != 1 or packets[0] is None:
            raise ValueError('consumer model requires exactly one typed row/residual packet')
        packet = packets[0]
        expected = (shard_shape[0] * 2,) + shard_shape[1:]
        if tuple(packet.shape) != expected or packet.dtype != torch.bfloat16:
            raise ValueError('consumer model input differs from the compiled row/residual fields')
        consumer = self.decoder.layers[0]
        if not isinstance(consumer, ResidualBDALayer) or consumer.residual is not None:
            raise ValueError('model input does not have an available BDA payload consumer')
        consumer.residual = packet.narrow(0, shard_shape[0], shard_shape[0])
        return original_input(self, [packet.narrow(0, 0, shard_shape[0])])

    TransformerBlock._build_layers = build
    TransformerBlock.sharded_state_dict = sharded
    TransformerBlock._frontier_row_cut = logical_layer
    GPTModel.set_input_tensor = set_input


def activate_project_rows(models, diagnostic_directory=None):
    for model in models:
        for layer in model.decoder.layers:
            if getattr(layer, 'frontier_defer_row_resolve', False):
                row = layer.mlp.linear_fc2
                if row.parallel_mode != 'row' or getattr(row.weight, 'partition_dim', None) != 1:
                    raise ValueError('materialized row weights must retain their actual input-axis sharding')
                row.parallel_mode = None
                layer.frontier_native_row_diagnostic = diagnostic_directory
