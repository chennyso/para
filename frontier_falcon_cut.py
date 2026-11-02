from types import MethodType

import torch

from megatron.core.transformer.spec_utils import build_module


def mlp_forward(self, hidden_states, *args, **kwargs):
    residual = hidden_states.to(torch.float32) if self.residual_in_fp32 else hidden_states
    output = self.mlp(hidden_states.to(self.config.params_dtype))
    with self.bias_dropout_add_exec_handler():
        return self.mlp_bda(training=self.training, fused=self.config.bias_dropout_fusion)(
            output, residual, self.hidden_dropout)


def install_falcon_semantic_cut(stack_type, logical_layer, source_pp, source_vp):
    if getattr(stack_type, '_frontier_semantic_cut', None) is not None:
        raise ValueError('One executable Semantic Cut must own the actual Falcon model construction')
    original = stack_type.__init__

    def construct(self, config, submodules, *args, **kwargs):
        original(self, config, submodules, *args, **kwargs)
        stage, vp = self.pp_group.rank(), self.vp_stage
        if (stage, vp) == (source_pp, source_vp):
            layer = self.layers[-1]
            if layer.layer_number - 1 != logical_layer or not layer.use_mlp:
                raise ValueError('Falcon Semantic Cut must match the actual terminal mixed region')
            before = set(dict(layer.named_parameters()))
            layer.mlp = None
            layer.mlp_bda = None
            layer.use_mlp = False
            layer.frontier_region_kind = 'FALCON_MIXER'
            expected = {key for key in before if not key.startswith('mlp.')}
            if set(dict(layer.named_parameters())) != expected:
                raise ValueError('Falcon source fragment lost unrelated native model parameters')
        elif (stage, vp) == (source_pp + 1, source_vp):
            if not self.layers or self.layers[0].layer_number - 1 != logical_layer + 1:
                raise ValueError('Falcon MLP consumer must precede the next original logical layer')
            layer = build_module(submodules.falconh1_layer, config=config,
                layer_number=logical_layer + 1, pg_collection=kwargs['pg_collection'],
                residual_in_fp32=True, use_mamba=False, use_attention=False, use_mlp=True)
            layer.norm = None
            layer.frontier_region_kind = 'FALCON_MLP_FRAGMENT'
            layer.forward = MethodType(mlp_forward, layer)
            keys = tuple(dict(layer.named_parameters()))
            if not keys or any(not key.startswith('mlp.') for key in keys):
                raise ValueError('Falcon consumer fragment must own exactly the original MLP parameters')
            self.layers.insert(0, layer)
            self.num_layers_per_pipeline_rank = len(self.layers)

    stack_type.__init__ = construct
    stack_type._frontier_semantic_cut = (logical_layer, source_pp, source_vp)
