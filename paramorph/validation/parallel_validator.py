from predictor.state_sharding import state_sharding_errors
from scheduling.native_microbatches import valid_microbatch_group
from graph.native_recomputation import selective_modules


class ParallelValidator:
    def __init__(self, model_spec, hardware):
        self.model_spec = model_spec
        self.hardware = hardware

    def validate_config(self, config):
        errors = []
        if not self.hardware.validate() or config.world_size != self.hardware.num_gpus:
            errors.append('TP×CP×PP×DP与实际GPU数量不一致')
        if config.total_vp_stages > self.model_spec.num_layers:
            errors.append('非空Region数量超过模型层数')
        if self.model_spec.num_attention_heads % config.tp:
            errors.append('Attention heads不能被TP整除')
        if self.model_spec.hidden_size % config.tp or self.model_spec.hidden_size % config.etp:
            errors.append('Hidden维度不能被TP或者ETP整除')
        if self.model_spec.ffn_hidden_size is not None and self.model_spec.ffn_hidden_size % config.etp:
            errors.append('FFN维度不能被ETP整除')
        if self.model_spec.ffn_hidden_size is not None and self.model_spec.ffn_hidden_size % config.tp:
            errors.append('FFN维度不能被TP整除')
        if self.model_spec.num_query_groups is not None and (
            self.model_spec.num_query_groups % config.tp and config.tp % self.model_spec.num_query_groups):
            errors.append('KV Head分组无法对应实际TP分区或者复制规则')
        if config.sequence_length % (2 * config.cp) and config.cp > 1:
            errors.append('序列长度不满足双向CP分块要求')
        if config.ep > 1 and (self.model_spec.num_experts is None or self.model_spec.num_experts % config.ep):
            errors.append('模型专家数量不满足EP分区要求')
        if self.model_spec.num_experts is None and config.etp not in (1, config.tp):
            errors.append('Dense模型没有独立ETP执行区域')
        if not valid_microbatch_group(config.num_microbatches, config.pp, config.vpp,
                config.microbatch_group):
            errors.append('microbatch数量或分组不满足当前Megatron VPP执行器要求')
        if config.pp == 1 and config.vpp > 1:
            errors.append('当前Megatron虚拟Pipeline初始化要求PP至少为两个阶段')
        request = dict(config.execution_implementation)
        if request.get('recompute') == 'selective-mlp' and self.model_spec.num_experts:
            errors.append('当前selective-mlp入口只验证稠密MLP；MoE使用selective并显式选择原生模块')
        if request.get('recompute') == 'selective':
            unknown = set(config.memory_policy.recompute_modules) - set(selective_modules(self.model_spec))
            if unknown:
                errors.append(f'当前模型没有对应的原生重计算模块：{sorted(unknown)}')
        request.update(tp=config.tp, pp=config.pp, cp=config.cp, dp=config.dp, ep=config.ep,
            etp=config.etp, node_count=self.hardware.num_nodes, gpus_per_node=self.hardware.gpus_per_node,
            offload=config.optimizer_offload)
        errors.extend(state_sharding_errors(request, config))
        return not errors, errors

    def validate_region_layout(self, config, regions):
        errors = []
        if len(regions) != config.total_vp_stages:
            errors.append('Region数量与PP×VPP不一致')
        offset = 0
        for index, region in enumerate(regions):
            if (region.region_id != index or region.pp_rank != index % config.pp or
                region.vp_rank != index // config.pp or region.layer_start != offset or
                region.layer_end <= offset):
                errors.append(f'Region{index}的顺序、设备或层范围不合法')
            offset = region.layer_end
        if offset != self.model_spec.num_layers:
            errors.append('Region没有完整覆盖全部模型层')
        return not errors, errors


def validate_candidate(candidate_config, model_spec, hardware, region_layout=None):
    validator = ParallelValidator(model_spec, hardware)
    valid, errors = validator.validate_config(candidate_config)
    if region_layout is not None:
        region_valid, region_errors = validator.validate_region_layout(candidate_config, region_layout)
        valid = valid and region_valid
        errors += region_errors
    return valid, errors
