from pydantic import Field

from contracts import Contract


def tensor_parallel_limit(workload, model_name=''):
    world = workload.get('node_count', 1) * workload.get('gpus_per_node', 4)
    if type(world) is not int or world < 1:
        raise ValueError('训练设备数量必须为正整数')
    requested = workload.get('max_tensor_parallel_size', world)
    if type(requested) is not int or requested < 1:
        raise ValueError('Tensor并行上限必须为正整数')
    return min(requested, world)


class Capability(Contract):
    operation: str
    supported: bool
    supported_regions: list[str]
    preconditions: list[str]
    transformation_semantics: str
    backend_callable: str | None
    gradient_semantics: str
    correctness_tests: list[str]
    limitations: list[str] = Field(default_factory=list)


class CapabilityRegistry:
    def __init__(self, model_config):
        self.model_config = model_config
        supported = {
            'RepartitionRegion': ('Region连续区间及PP/VPP归属', 'ProgramAgent.lower_native'),
            'SetPipelineLayout': ('完整PP/VPP分区与原生交错调度表', 'ProgramAgent.compile'),
            'SetParallelism': ('重建原生Dense与Expert进程组', 'hybrid_training_entry.pretrain'),
            'SetMicrobatch': ('MBS与梯度累积次数', 'ProgramAgent.lower_native'),
            'SelectPipelineSchedule': ('选择后端原生1F1B或Interleaved 1F1B', 'megatron.core.pipeline_parallel.schedules.get_forward_backward_func'),
            'SetRecomputation': ('完整激活重计算及其真实反向路径', 'native_hybrid_train.main'),
            'SetCommunicationPolicy': ('原生P2P重叠与梯度Bucket', 'ProgramAgent.lower_native')}
        supported.update(
            SetRankMapping=('逻辑Rank到实际Node/GPU的完整映射', 'hybrid_training_entry.rank_mapping'),
            SetOptimizerSharding=('原生分布式或者复制Optimizer状态及Precision-Aware状态精度', 'hybrid_training_entry.pretrain'),
            JointTransform=('受支持变量的联合变化与硬约束补全', 'CompleteAndRewrite.complete_and_rewrite'))
        self.entries = {name: Capability(operation=name, supported=True,
            supported_regions=['actual_megatron_model_modules'],
            preconditions=['合法并行组', '完整模型覆盖', '真实后端参数验证'],
            transformation_semantics=semantics, backend_callable=callable_name,
            gradient_semantics='保持有效输入、损失归一化、梯度累积与Optimizer Update语义',
            correctness_tests=['paramorph/tests/test_strategy_transformations.py', 'actual_gpu_numerical_reference'])
            for name, (semantics, callable_name) in supported.items()}
        for name in ('SetRegionSharding', 'SplitBackward', 'ChangeExecutionOrder'):
            self.entries[name] = Capability(operation=name, supported=False,
                supported_regions=[], preconditions=['需要对应模块的独立后端执行接口'],
                transformation_semantics='独立算子布局、反向任务或执行顺序修改',
                backend_callable=None, gradient_semantics='需要独立数值认证', correctness_tests=[],
                limitations=['当前原生模型入口未提供该操作的通用执行接口'])

    def require(self, operation):
        if operation not in self.entries or not self.entries[operation].supported:
            raise ValueError(f'当前后端不支持图变换：{operation}')
        return self.entries[operation]

    def parallel_domains(self, model, topology, workload):
        divisors = [value for value in range(1, topology.num_gpus + 1)
                    if topology.num_gpus % value == 0]
        maximum_tp = tensor_parallel_limit(workload, model.name)
        hybrid = any(kind.value in ('attention_mamba_composite', 'mamba', 'gdn') for kind in model.layer_types)
        return dict(tp=[value for value in divisors
                        if value <= maximum_tp and model.num_attention_heads % value == 0
                        and model.hidden_size % value == 0
                        and (model.num_query_groups or model.num_attention_heads) % value == 0],
            etp=[value for value in divisors if value <= maximum_tp and model.hidden_size % value == 0 and
                 (model.ffn_hidden_size is None or model.ffn_hidden_size % value == 0)],
            pp=[value for value in divisors if value <= model.num_layers], dp=divisors, expert_dp=divisors,
            cp=[1] if hybrid else [value for value in divisors
                if value == 1 or workload['sequence'] % (2 * value) == 0],
            ep=[value for value in divisors if model.num_experts is not None and model.num_experts % value == 0] or [1],
            vpp=list(range(1, model.num_layers + 1)),
            micro_batch=[value for value in range(1, workload['global_batch'] + 1)
                         if workload['global_batch'] % value == 0],
            group=list(range(1, workload['global_batch'] + 1)))

    def describe(self):
        return {key: row.model_dump(mode='json') for key, row in self.entries.items()}
