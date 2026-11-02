import math
from dataclasses import dataclass, field

from core_data_models import LayerType


class AnalyticalModelError(ValueError):
    pass


def _analytical_missing(identity, reason):
    return dict(status='NEEDS_PROFILE', milliseconds=None, identity=identity,
                cost_scope=identity.get('granularity', 'module_composite'),
                source='hardware_analytical_model', reason=reason,
                calibration=None, assumptions=[])


def _required_values(values, names, identity, description):
    missing = [name for name in names if name not in values]
    if missing:
        return _analytical_missing(identity, f'{description}需要明确提供：' + ', '.join(missing))
    return None


def _positive_number(name, value, *, allow_zero=False, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AnalyticalModelError(f'{name}必须为有限数值')
    if value < 0 or (value == 0 and not allow_zero):
        raise AnalyticalModelError(f'{name}必须为' + ('非负数值' if allow_zero else '正数'))
    if integer and (type(value) is not int):
        raise AnalyticalModelError(f'{name}必须为整数')
    return value


def _nonempty_source(name, value):
    if not isinstance(value, str) or not value.strip():
        raise AnalyticalModelError(f'{name}需要明确的数据来源')
    return value


def _dtype_size(identity, spec):
    if 'dtype_bytes' in spec:
        return _positive_number('dtype_bytes', spec['dtype_bytes'], integer=True)
    dtype = identity.get('dtype', identity.get('precision'))
    sizes = {'float64': 8, 'float32': 4, 'float16': 2, 'bfloat16': 2,
             'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1,
             'float8_e4m3fn': 1, 'float8_e5m2': 1}
    if isinstance(dtype, str):
        return sizes.get(dtype.removeprefix('torch.'))
    return None


def _matrix_block(m, n, k, dtype_bytes, name):
    return dict(name=name, flops=2 * m * n * k,
                traffic_bytes=dtype_bytes * (m * k + k * n + m * n),
                elementwise_operations=0)


def _gemm_blocks(features, phase, dtype_bytes, identity):
    missing = _required_values(features, ('M', 'N', 'K'), identity, 'GEMM局部矩阵形状')
    if missing:
        return missing
    m, n, k = [_positive_number(name, features[name], integer=True) for name in ('M', 'N', 'K')]
    if phase in ('forward', 'recomputation'):
        return [_matrix_block(m, n, k, dtype_bytes, 'gemm_forward')]
    if phase == 'dgrad':
        return [_matrix_block(m, k, n, dtype_bytes, 'gemm_dgrad')]
    if phase == 'wgrad':
        return [_matrix_block(k, n, m, dtype_bytes, 'gemm_wgrad')]
    if phase == 'backward':
        return [_matrix_block(m, k, n, dtype_bytes, 'gemm_dgrad'),
                _matrix_block(k, n, m, dtype_bytes, 'gemm_wgrad')]
    return _analytical_missing(identity, 'GEMM解析模型缺少适用的Forward或Backward阶段')


def _grouped_gemm_blocks(features, phase, dtype_bytes, identity):
    missing = _required_values(features, ('expert_token_counts', 'expert_token_source', 'N', 'K'),
                               identity, 'Grouped GEMM局部Expert形状')
    if missing:
        return missing
    _nonempty_source('expert_token_source', features['expert_token_source'])
    counts = features['expert_token_counts']
    if not isinstance(counts, list) or not counts:
        raise AnalyticalModelError('Grouped GEMM需要逐个本地Expert的Token数量列表')
    for count in counts:
        _positive_number('expert_token_counts', count, allow_zero=True, integer=True)
    n = _positive_number('N', features['N'], integer=True)
    k = _positive_number('K', features['K'], integer=True)
    active = [count for count in counts if count > 0]
    if phase not in ('forward', 'recomputation', 'dgrad', 'wgrad', 'backward'):
        return _analytical_missing(identity, 'Grouped GEMM解析模型缺少适用的计算阶段')
    phases = ['dgrad', 'wgrad'] if phase == 'backward' else [phase]
    blocks = []
    for part in phases:
        local = [_gemm_blocks(dict(M=count, N=n, K=k), part, dtype_bytes, identity)[0]
                 for count in active]
        blocks.append(dict(name=f'grouped_gemm_{part}',
                           flops=sum(block['flops'] for block in local),
                           traffic_bytes=sum(block['traffic_bytes'] for block in local),
                           elementwise_operations=0, expert_token_counts=counts,
                           active_expert_count=len(active), expert_token_source=features['expert_token_source']))
    return blocks


def _attention_blocks(features, phase, dtype_bytes, identity):
    names = ('batch', 'query_heads', 'kv_heads', 'query_length', 'key_length',
             'head_dimension', 'value_dimension', 'causal', 'materializes_attention_scores')
    missing = _required_values(features, names, identity, 'Attention局部Tensor形状及实现语义')
    if missing:
        return missing
    for name in names[:7]:
        _positive_number(name, features[name], integer=True)
    for name in names[7:]:
        if type(features[name]) is not bool:
            raise AnalyticalModelError(f'{name}必须为明确的布尔值')
    batch, heads, kv_heads, queries, keys, dimension, value_dimension = (
        features[name] for name in names[:7])
    if heads % kv_heads:
        raise AnalyticalModelError('Attention的Query Heads必须被KV Heads整除')
    if phase not in ('forward', 'recomputation', 'backward'):
        return _analytical_missing(identity, 'Attention解析模型只支持完整Forward、Recomputation或Backward')
    count_name = 'softmax_backward_operations_per_element' if phase == 'backward' else 'softmax_operations_per_element'
    missing = _required_values(features, (count_name, 'softmax_operation_source'), identity,
                               'Attention Softmax解析计算量')
    if missing:
        return missing
    operations = _positive_number(count_name, features[count_name])
    _nonempty_source('softmax_operation_source', features['softmax_operation_source'])
    pairs = queries * keys
    if features['causal'] and queries == keys:
        pairs = queries * (queries + 1) // 2
    pairs *= batch * heads
    q_bytes = dtype_bytes * batch * heads * queries * dimension
    k_bytes = dtype_bytes * batch * kv_heads * keys * dimension
    v_bytes = dtype_bytes * batch * kv_heads * keys * value_dimension
    o_bytes = dtype_bytes * batch * heads * queries * value_dimension
    score_bytes = dtype_bytes * batch * heads * queries * keys
    if phase == 'backward':
        flops = 4 * pairs * (dimension + value_dimension)
        traffic = 2 * (q_bytes + k_bytes + v_bytes + o_bytes)
        if features['materializes_attention_scores']:
            traffic += 5 * score_bytes
        return [dict(name='attention_backward', flops=flops, traffic_bytes=traffic,
                     elementwise_operations=operations * pairs)]
    if features['materializes_attention_scores']:
        return [dict(name='attention_qk', flops=2 * pairs * dimension,
                     traffic_bytes=q_bytes + k_bytes + score_bytes, elementwise_operations=0),
                dict(name='attention_softmax', flops=0, traffic_bytes=2 * score_bytes,
                     elementwise_operations=operations * pairs),
                dict(name='attention_av', flops=2 * pairs * value_dimension,
                     traffic_bytes=score_bytes + v_bytes + o_bytes, elementwise_operations=0)]
    return [dict(name='attention_fused', flops=2 * pairs * (dimension + value_dimension),
                 traffic_bytes=q_bytes + k_bytes + v_bytes + o_bytes,
                 elementwise_operations=operations * pairs)]


def estimate_hardware_cost(identity, analytical_spec):
    if analytical_spec is None:
        return _analytical_missing(identity, '缺少计算任务的解析形状和硬件参数')
    if not isinstance(analytical_spec, dict):
        raise AnalyticalModelError('解析计算参数必须为字典')
    spec = analytical_spec
    missing = _required_values(spec, ('operation', 'hardware'), identity, '计算解析模型')
    if missing:
        return missing
    hardware = spec['hardware']
    if not isinstance(hardware, dict):
        raise AnalyticalModelError('解析硬件参数必须为字典')
    missing = _required_values(hardware, ('identity', 'dtype', 'source', 'throughput_kind', 'assumptions',
                                         'memory_bandwidth_bytes_per_second'), identity, '计算硬件能力')
    if missing:
        return missing
    gpu = identity.get('gpu', identity.get('hardware'))
    dtype = identity.get('dtype', identity.get('precision'))
    if gpu is None or dtype is None:
        return _analytical_missing(identity, '计算成本身份需要明确GPU与精度')
    if hardware['identity'] != gpu or hardware['dtype'] != dtype:
        return _analytical_missing(identity, '解析硬件参数与计算成本的GPU或者精度条件不一致')
    _nonempty_source('hardware.source', hardware['source'])
    if hardware['throughput_kind'] not in ('theoretical_peak', 'calibrated_effective'):
        raise AnalyticalModelError('硬件吞吐量需要标明theoretical_peak或者calibrated_effective')
    assumptions = hardware['assumptions']
    if not isinstance(assumptions, list) or any(not isinstance(value, str) or not value.strip() for value in assumptions):
        raise AnalyticalModelError('硬件解析假设必须为明确的文本列表')
    bandwidth = _positive_number('memory_bandwidth_bytes_per_second', hardware['memory_bandwidth_bytes_per_second'])
    calibration_ids = hardware.get('calibration_evidence_ids', [])
    if not isinstance(calibration_ids, list) or any(not isinstance(value, str) or not value.strip() for value in calibration_ids):
        raise AnalyticalModelError('硬件标定证据身份必须为明确的文本列表')
    if hardware['throughput_kind'] == 'calibrated_effective' and not calibration_ids:
        return _analytical_missing(identity, '有效硬件吞吐量缺少实际标定证据身份')
    if spec['operation'] == 'composite':
        components = spec.get('components')
        if not isinstance(components, list) or not components:
            raise AnalyticalModelError('复合计算任务需要完整的解析组件列表')
        estimates = []
        for component in components:
            if not isinstance(component, dict):
                raise AnalyticalModelError('复合计算任务的每个组件必须为字典')
            child = dict(component)
            child.setdefault('hardware', hardware)
            child.setdefault('phase', spec.get('phase', identity.get('phase')))
            if 'dtype_bytes' in spec:
                child.setdefault('dtype_bytes', spec['dtype_bytes'])
            estimate = estimate_hardware_cost(identity, child)
            if estimate['status'] == 'NEEDS_PROFILE':
                return dict(estimate, component_operation=child.get('operation'))
            estimates.append(estimate)
        return dict(status='ANALYTICAL', milliseconds=sum(item['milliseconds'] for item in estimates),
                    identity=identity, cost_scope=identity.get('granularity', 'module_composite'),
                    source='hardware_analytical_composite', components=estimates,
                    calibration=dict(kind=hardware['throughput_kind'], source=hardware['source'],
                                     evidence_ids=hardware.get('calibration_evidence_ids', [])),
                    assumptions=list(dict.fromkeys(assumptions +
                        ['按组件依赖顺序累计解析时间，未声明组件融合或者组件并发'] +
                        [value for item in estimates for value in item['assumptions']])))
    features = spec.get('features', identity.get('features'))
    if not isinstance(features, dict):
        return _analytical_missing(identity, '解析计算模型缺少该操作的局部Tensor形状')
    phase = spec.get('phase', identity.get('phase'))
    dtype_bytes = _dtype_size(identity, spec)
    if dtype_bytes is None:
        return _analytical_missing(identity, '解析计算模型需要明确Tensor精度对应的元素字节数')
    operation = spec['operation']
    local_assumptions = ['局部Tensor形状直接来自候选计划的任务定义',
                         '计算与显存访问按Roofline约束组合，未指定利用率修正系数',
                         f'输入与输出Tensor均按每个元素{dtype_bytes}字节计算，混合精度转换需要独立任务描述']
    if operation == 'gemm':
        blocks = _gemm_blocks(features, phase, dtype_bytes, identity)
        local_assumptions += ['每次矩阵操作读取两个输入并写入一个输出，未计入输出累加读取或者额外Workspace访问']
    elif operation == 'grouped_gemm':
        blocks = _grouped_gemm_blocks(features, phase, dtype_bytes, identity)
        local_assumptions += ['按本地Expert的明确Token分布累计计算与字节数',
                              '有效Grouped GEMM成本仍受Kernel调度影响，解析估计未包含Expert之间的占用差异',
                              '零Token Expert未执行矩阵操作，其初始化或者路由开销由对应独立任务描述']
    elif operation == 'attention':
        blocks = _attention_blocks(features, phase, dtype_bytes, identity)
        local_assumptions += ['Attention仅包含QK、Softmax及AV，Projection和Dropout由对应独立任务描述',
                              '融合Attention的显存访问按Q、K、V和输出各访问一次计算，未估计Kernel内部Tile重读',
                              'Backward计算dQ、dK、dV与中间梯度，未包含重算Attention概率的额外Forward']
        if features.get('causal'):
            local_assumptions.append('等长因果Attention按三角形有效元素计算；其他长度按完整矩阵计算，Mask和Tile开销需要独立标定')
    elif operation == 'elementwise':
        missing = _required_values(features, ('operation_count', 'traffic_bytes', 'operation_count_source', 'traffic_source'),
                                   identity, 'Elementwise解析操作数量与显存访问')
        if missing:
            return missing
        _nonempty_source('operation_count_source', features['operation_count_source'])
        _nonempty_source('traffic_source', features['traffic_source'])
        blocks = [dict(name='elementwise', flops=0,
                       traffic_bytes=_positive_number('traffic_bytes', features['traffic_bytes'], allow_zero=True),
                       elementwise_operations=_positive_number('operation_count', features['operation_count'], allow_zero=True))]
        local_assumptions += ['Elementwise使用独立的操作吞吐率，操作数量与字节数由任务定义提供']
    else:
        return _analytical_missing(identity, f'操作{operation}缺少受支持的硬件解析公式')
    if isinstance(blocks, dict):
        return blocks
    needs_compute = any(block['flops'] > 0 for block in blocks)
    needs_elementwise = any(block['elementwise_operations'] > 0 for block in blocks)
    if needs_compute and 'compute_throughput_flops_per_second' not in hardware:
        return _analytical_missing(identity, '矩阵解析模型缺少对应精度的硬件计算吞吐率')
    if needs_elementwise and any(name not in hardware for name in
                                ('elementwise_throughput_operations_per_second', 'elementwise_throughput_source')):
        return _analytical_missing(identity, '非矩阵操作需要独立Elementwise吞吐率与数据来源')
    compute_rate = _positive_number('compute_throughput_flops_per_second',
                                   hardware['compute_throughput_flops_per_second']) if needs_compute else None
    elementwise_rate = None
    if needs_elementwise:
        elementwise_rate = _positive_number('elementwise_throughput_operations_per_second',
                                           hardware['elementwise_throughput_operations_per_second'])
        _nonempty_source('elementwise_throughput_source', hardware['elementwise_throughput_source'])
    launch_ms = spec.get('launch_ms', 0)
    _positive_number('launch_ms', launch_ms, allow_zero=True)
    if 'launch_ms' in spec:
        if 'launch_source' not in spec:
            return _analytical_missing(identity, 'Kernel启动时延缺少来源')
        _nonempty_source('launch_source', spec['launch_source'])
        local_assumptions.append('启动时延使用显式参数：' + spec['launch_source'])
    else:
        local_assumptions.append('未提供Kernel启动时延，解析结果不包含启动开销')
    estimates = []
    for block in blocks:
        matrix_seconds = block['flops'] / compute_rate if block['flops'] else 0
        elementwise_seconds = block['elementwise_operations'] / elementwise_rate if block['elementwise_operations'] else 0
        memory_seconds = block['traffic_bytes'] / bandwidth
        estimate = dict(block, matrix_time_ms=matrix_seconds * 1000,
                        elementwise_time_ms=elementwise_seconds * 1000,
                        memory_time_ms=memory_seconds * 1000,
                        milliseconds=launch_ms + max(matrix_seconds + elementwise_seconds, memory_seconds) * 1000)
        estimates.append(estimate)
    if hardware['throughput_kind'] == 'theoretical_peak':
        local_assumptions.append('理论峰值参数给出理想执行条件下的时间估计，未包含真实Kernel效率与额外访存')
    return dict(status='ANALYTICAL', milliseconds=sum(item['milliseconds'] for item in estimates),
                identity=identity, cost_scope=identity.get('granularity', 'module_composite'),
                source='hardware_analytical_roofline', operation=operation, phase=phase, blocks=estimates,
                calibration=dict(kind=hardware['throughput_kind'], source=hardware['source'],
                                 evidence_ids=hardware.get('calibration_evidence_ids', []),
                                 compute_throughput_flops_per_second=compute_rate,
                                 memory_bandwidth_bytes_per_second=bandwidth,
                                 elementwise_throughput_operations_per_second=elementwise_rate),
                assumptions=list(dict.fromkeys(assumptions + local_assumptions)))


@dataclass(frozen=True)
class AnalyticalEstimate:
    estimated_ms: float | None
    confidence: float | None
    source: str
    scope: str
    assumptions: tuple = ()
    metadata: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(estimated_ms=self.estimated_ms, confidence=self.confidence,
                    source=self.source, scope=self.scope,
                    assumptions=list(self.assumptions), metadata=self.metadata)


def _validate_positive(**values):
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AnalyticalModelError(f'{name}必须为正的有限数值')


def layer_flops_tokens(model_spec, layer_index, sequence_length, micro_batch_size,
                       tp=1, cp=1, recompute=False):
    """估计单层前向的 FLOPs（按 elementwise 计），供形状缩放使用。

    只依赖模型结构与批次形状，与 pp/vpp/region 无关——这正是复用性的来源。
    """
    _validate_positive(sequence_length=sequence_length, micro_batch_size=micro_batch_size,
                       tp=tp, cp=cp)
    if not 0 <= layer_index < model_spec.num_layers:
        raise AnalyticalModelError('层索引超出模型范围')
    hidden = model_spec.hidden_size
    tokens = sequence_length * micro_batch_size
    layer_type = model_spec.get_layer_type(layer_index)
    if layer_type in (LayerType.HYBRID, LayerType.MAMBA):
        raise AnalyticalModelError('异构模块需要实际模块形状或者独立Profiling')
    if model_spec.ffn_hidden_size is None:
        raise AnalyticalModelError('解析计算量需要实际FFN维度')
    heads = model_spec.num_attention_heads
    dimension = model_spec.head_dimension or hidden // heads
    query_groups = model_spec.num_query_groups or heads
    projection = 2 * tokens * hidden * (heads * dimension + 2 * query_groups * dimension)
    projection += 2 * tokens * heads * dimension * hidden
    attention = 4 * micro_batch_size * heads * sequence_length**2 * dimension
    mlp = 6 * tokens * hidden * model_spec.ffn_hidden_size
    if layer_type == LayerType.MOE:
        if model_spec.moe_top_k is None:
            raise AnalyticalModelError('Expert计算量需要实际Top-K或者Router Token统计')
        mlp *= model_spec.moe_top_k
        mlp += 6 * tokens * hidden * (model_spec.shared_expert_size or 0)
    forward = (projection + attention + mlp) / (tp * cp)
    # 反向约为前向的 2 倍（dX + dW）；完整重计算再补一次前向。
    total = forward * 3.0
    if recompute:
        total += forward
    return total


def scale_layer_cost(base_cost_ms, base_shape, target_shape):
    """按形状比例缩放逐层成本。

    base_shape / target_shape 是 dict，至少含 (sequence_length, micro_batch_size,
    tp, cp)。计算量近似线性于 tokens，反比于切分度。
    """
    required = ('sequence_length', 'micro_batch_size', 'tp', 'cp')
    for name in required:
        if name not in base_shape or name not in target_shape:
            raise AnalyticalModelError(f'形状缩放需要 {name}')
    if base_cost_ms <= 0:
        raise AnalyticalModelError('基准层成本必须为正')
    base_tokens = base_shape['sequence_length'] * base_shape['micro_batch_size']
    target_tokens = target_shape['sequence_length'] * target_shape['micro_batch_size']
    base_shard = base_shape['tp'] * base_shape['cp']
    target_shard = target_shape['tp'] * target_shape['cp']
    if base_tokens <= 0 or base_shard <= 0:
        raise AnalyticalModelError('基准形状必须为正')
    ratio = (target_tokens / base_tokens) * (base_shard / target_shard)
    return base_cost_ms * ratio


def estimate_link_cost(base_ms, base_message_bytes, target_message_bytes, bandwidth_gbps):
    """按消息规模缩放 P2P 通信成本。

    通信时间 ≈ 消息字节 / 有效带宽 + 固定延迟。固定延迟取实测基准与带宽估计的残差。
    """
    if base_message_bytes <= 0 or target_message_bytes <= 0:
        raise AnalyticalModelError('消息规模必须为正')
    if not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0:
        raise AnalyticalModelError('链路带宽必须为正的有限数值')
    bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1e3
    if base_ms <= 0:
        raise AnalyticalModelError('基准链路成本必须为正')
    fixed_ms = max(0.0, base_ms - base_message_bytes / bytes_per_ms)
    return fixed_ms + target_message_bytes / bytes_per_ms


def analytical_layer_estimate(model_spec, layer_index, config, base_layer_cost_ms=None,
                              base_shape=None):
    """给出一层的解析估计。

    有基准实测时按形状缩放（精度最高）；无基准时用 FLOPs 与参考吞吐给出量级估计。
    """
    sequence = config.sequence_length
    micro_batch = config.micro_batch_size
    recompute = bool(config.use_activation_recomputation or config.memory_policy.recompute_granularity)
    if base_layer_cost_ms is not None and base_shape is not None:
        target_shape = dict(sequence_length=sequence, micro_batch_size=micro_batch,
                            tp=config.tp, cp=config.cp)
        estimate = scale_layer_cost(base_layer_cost_ms, base_shape, target_shape)
        return AnalyticalEstimate(estimate, None, 'analytical_scaled',
            '按序列长度、批次与切分度线性缩放的实测基准',
            ('假设层计算量线性于 token 数', '假设切分度反比缩放'),
            dict(base_cost_ms=base_layer_cost_ms, base_shape=base_shape,
                 target_shape=target_shape))
    flops = layer_flops_tokens(model_spec, layer_index, sequence, micro_batch,
                               config.tp, config.cp, recompute)
    # 缺少基准时无法给出可信数值，只报告规模供上层决定是否 profiling。
    return AnalyticalEstimate(None, None, 'unmeasured',
        '缺少该配置的基准实测，无法给出可信估计',
        ('需要至少一次同族配置的逐层 Profiling',),
        dict(flops=flops, requires_profiling=True))
