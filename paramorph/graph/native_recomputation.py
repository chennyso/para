def selective_modules(model):
    kinds = {kind.value for kind in model.layer_types}
    supported = {}
    if kinds.intersection({'attention', 'mlp', 'moe'}):
        supported['core_attn'] = '重计算Attention核心；Fused Attention下需要比较实际保存收益与重复计算'
    if kinds.intersection({'attention', 'mlp'}):
        supported['mlp'] = '重计算稠密MLP'
    if 'moe' in kinds:
        supported['moe_act'] = '重计算专家MLP激活函数；不重新执行整个专家路由和Token交换'
        supported['moe'] = '重计算完整MoE前向；需要检查重新执行路由和Token交换的代价'
        if model.shared_expert_size:
            supported['shared_experts'] = '重计算共享专家；受共享专家通信重叠设置约束'
    return supported
