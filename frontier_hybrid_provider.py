from types import MethodType
from contextvars import ContextVar

import torch.distributed as dist

FALCON_VP_STAGE = ContextVar('frontier_falcon_vp_stage', default=0)


def install_falcon_provider(provider):
    native_globals = provider.provide.__func__.__globals__
    model_type = native_globals['FalconH1Model']
    pad_vocab = native_globals['calculate_padded_vocab_size']
    module_spec = native_globals['ModuleSpec']

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        if self.share_embeddings_and_output_weights:
            raise ValueError('Falcon Hybrid VPP requires its native untied embedding and output parameters')
        specification = self.falconh1_stack_spec
        if not isinstance(specification, module_spec):
            specification = specification()
        vocab = pad_vocab(self.vocab_size, self.make_vocab_size_divisible_by,
            self.tensor_model_parallel_size) if self.should_pad_vocab else self.vocab_size
        token = FALCON_VP_STAGE.set(0 if vp_stage is None else vp_stage)
        try:
            model = model_type(config=self, falconh1_stack_spec=specification,
                vocab_size=vocab, max_sequence_length=self.seq_length,
                hybrid_attention_ratio=self.hybrid_attention_ratio, falconh1_ratio=self.falconh1_ratio,
                hybrid_mlp_ratio=self.hybrid_mlp_ratio, hybrid_override_pattern=self.hybrid_override_pattern,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy, logit_dtype=self.logit_dtype,
                parallel_output=self.parallel_output, share_embeddings_and_output_weights=False,
                position_embedding_type=self.position_embedding_type, rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base, seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process, post_process=post_process)
        finally:
            FALCON_VP_STAGE.reset(token)
        model.vp_stage = vp_stage
        return model

    provider.provide = MethodType(provide, provider)


def install_hybrid_provider(provider):
    native_globals = provider.provide.__func__.__globals__
    model_type = native_globals['MCoreHybridModel']
    pad_vocab = native_globals['calculate_padded_vocab_size']
    logit_dtype_kwarg = native_globals['logit_dtype_kwarg']

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        groups = self._pg_collection
        if groups is None:
            raise ValueError('Hybrid region execution requires actual native process groups')
        pp_rank = dist.get_rank(groups.pp)
        vp = 0 if vp_stage is None else vp_stage
        vp_size = self.virtual_pipeline_model_parallel_size or 1
        if not 0 <= vp < vp_size:
            raise ValueError('Hybrid virtual stage is outside the executable layout')
        expected_pre = pp_rank == 0 and vp == 0
        expected_post = pp_rank == dist.get_world_size(groups.pp) - 1 and vp == vp_size - 1
        if pre_process != expected_pre or post_process != expected_post:
            raise ValueError('Hybrid native preprocessing ownership differs from PP/VPP placement')
        vocab = pad_vocab(self.vocab_size, self.make_vocab_size_divisible_by,
            self.tensor_model_parallel_size) if self.should_pad_vocab else self.vocab_size
        specification = self._resolve_hybrid_stack_spec()
        self._pg_collection = None
        try:
            model = model_type(config=self, hybrid_stack_spec=specification,
                vocab_size=vocab, max_sequence_length=self.seq_length,
                hybrid_layer_pattern=self.hybrid_layer_pattern,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(model_type, self.logit_dtype),
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent, rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process, post_process=post_process,
                pg_collection=groups, vp_stage=vp_stage)
        finally:
            self._pg_collection = groups
        if model.config is not self or model.pg_collection is not groups or model.vp_stage != vp_stage:
            raise ValueError('Native Hybrid model lost its actual config, group or virtual-stage identity')
        return model

    provider.provide = MethodType(provide, provider)
