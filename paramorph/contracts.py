import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cost.runtime_calibration import configuration


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class WorkloadSpec(Contract):
    model_structure: dict = Field(default_factory=dict)
    model_type: Literal['Dense', 'MoE', 'Hybrid'] = 'Dense'
    model_signature: str
    model_revision: str | None
    sequence_length: int = Field(gt=0)
    global_batch_size: int = Field(gt=0)
    precision: str
    optimizer: str
    optimizer_config: dict
    data_signature: str
    input_specification: dict
    loss_normalization: str
    hardware_signature: str
    backend_signature: str
    random_seed: int
    training_step_boundary: list[str]

    @model_validator(mode='after')
    def validate_batch(self):
        if 'optimizer_update' not in self.training_step_boundary:
            raise ValueError('训练步骤测量边界需要包含Optimizer Update')
        return self

    @property
    def effective_tokens_per_step(self):
        return self.global_batch_size * self.sequence_length


class EffectiveTrainingPlan(Contract):
    requested_config: dict
    effective_config: dict
    workload_fingerprint: str
    model_fingerprint: str
    hardware_fingerprint: str
    backend_fingerprint: str
    backend_source_hashes: dict[str, str] = Field(default_factory=dict)
    schedule_identity: str
    optimizer_semantics: dict

    @property
    def effective_plan_hash(self):
        return fingerprint(dict(model=self.model_fingerprint, hardware=self.hardware_fingerprint,
            workload=self.workload_fingerprint, backend=self.backend_fingerprint,
            effective_config=self.effective_config, schedule_identity=self.schedule_identity))


class TrainingPlan(Contract):
    identity_revision: Literal['v2', 'v3', 'v4'] = 'v2'
    plan_id: str
    parent_plan_id: str | None
    model_fingerprint: str
    hardware_fingerprint: str
    workload_fingerprint: str
    backend_fingerprint: str
    parallelism: dict
    region_partition: list[int]
    region_layouts: dict
    rank_mapping: list[int]
    schedule: str
    memory_policy: dict
    backend_config: dict
    transformations: list[dict]
    expert_parallelism: dict = Field(default_factory=dict)
    process_groups: dict = Field(default_factory=dict)
    special_module_placement: dict = Field(default_factory=dict)
    optimizer_sharding: Literal['distributed', 'replicated', 'fsdp'] = 'distributed'
    state_sharding: dict = Field(default_factory=dict)
    optimizer_configuration: dict = Field(default_factory=dict)
    checkpoint_policy: dict = Field(default_factory=dict)
    num_microbatches: int | None = None

    @model_validator(mode='before')
    @classmethod
    def normalize_optimizer_configuration(cls, value):
        if not isinstance(value, dict):
            return value
        from predictor.profile_database import resolved_clip_grad
        clip_grad = resolved_clip_grad(value['backend_config'])
        optimizer = dict(value.get('optimizer_configuration', {}))
        if 'clip_grad' in optimizer and resolved_clip_grad(optimizer) != clip_grad:
            raise ValueError('Optimizer配置与后端请求的clip_grad不一致')
        return dict(value, optimizer_configuration=dict(optimizer, clip_grad=clip_grad))

    @model_validator(mode='after')
    def validate_plan(self):
        request = self.backend_config
        config = configuration(request)
        if config.world_size != request['node_count'] * request['gpus_per_node']:
            raise ValueError('计划并行进程组与物理设备数量不一致')
        if self.region_partition != request['decoder_chunks'] or len(self.region_partition) != config.total_vp_stages:
            raise ValueError('计划Region分区与后端配置不一致')
        if any(type(value) is not int or value < 1 for value in self.region_partition):
            raise ValueError('Region层数必须为正整数')
        if sorted(self.rank_mapping) != list(range(config.world_size)):
            raise ValueError('物理Rank Mapping必须完整覆盖每个设备一次')
        if self.rank_mapping != request.get('rank_mapping', list(range(config.world_size))):
            raise ValueError('计划物理映射与真实启动配置不同')
        expert = dict(ep=config.ep, etp=config.etp,
            expert_dp=config.world_size // (config.etp * config.ep * config.pp))
        if self.expert_parallelism and self.expert_parallelism != expert:
            raise ValueError('计划Expert并行参数与后端进程组不同')
        if self.num_microbatches is not None and self.num_microbatches != config.num_microbatches:
            raise ValueError('计划Microbatch数量与Batch训练语义不同')
        expected = {key: getattr(config, key) for key in ('tp', 'pp', 'cp', 'ep', 'dp', 'vpp')}
        if self.parallelism != expected:
            raise ValueError('计划并行参数与后端配置不一致')
        from predictor.state_sharding import state_sharding_config
        expected_sharding = state_sharding_config(request, config)
        if self.state_sharding and self.state_sharding != expected_sharding:
            raise ValueError('计划状态分片与实际后端配置不同')
        return self

    @property
    def effective_plan(self):
        from predictor.profile_database import effective_execution_config
        from scheduling.native_microbatches import native_schedule_settings

        request = self.backend_config
        effective = effective_execution_config(request) | dict(decoder_chunks=self.region_partition)
        return EffectiveTrainingPlan(requested_config=request, effective_config=effective,
            workload_fingerprint=self.workload_fingerprint, model_fingerprint=self.model_fingerprint,
            hardware_fingerprint=self.hardware_fingerprint, backend_fingerprint=self.backend_fingerprint,
            backend_source_hashes=request.get('backend_source_hashes', {}),
            schedule_identity=fingerprint(native_schedule_settings(request)),
            optimizer_semantics=self.optimizer_configuration)

    @property
    def effective_plan_hash(self):
        return self.effective_plan.effective_plan_hash

    @property
    def plan_hash(self):
        if self.identity_revision == 'v4':
            return self.effective_plan_hash
        from predictor.profile_database import normalized_execution
        payload = self.model_dump(exclude={'plan_id', 'parent_plan_id', 'transformations', 'identity_revision'})
        payload['backend_config'] = normalized_execution(self.backend_config)
        if self.identity_revision == 'v2' and 'cuda_device_max_connections' not in self.backend_config:
            payload['backend_config']['cuda_device_max_connections'] = 1
        if payload['backend_config']['execution_backend'] == 'megatron_ddp':
            payload['state_sharding'] = dict(execution_backend='megatron_ddp',
                optimizer_sharding=payload['backend_config']['optimizer_sharding'])
        return fingerprint(payload)


class ModuleRelation(Contract):
    relation: Literal['same_virtual', 'same_physical', 'different_physical', 'allowed_virtual']
    module_ids: list[int] = Field(min_length=1)
    virtual_slots: list[int] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_relation(self):
        if any(type(value) is not int or value < 0 for value in self.module_ids + self.virtual_slots):
            raise ValueError('模块编号和虚拟区间编号必须为非负整数')
        if len(set(self.module_ids)) != len(self.module_ids):
            raise ValueError('模块关系包含重复编号')
        if self.relation == 'allowed_virtual':
            if not self.virtual_slots or len(set(self.virtual_slots)) != len(self.virtual_slots):
                raise ValueError('允许的虚拟区间必须非空且没有重复')
        elif self.virtual_slots or len(self.module_ids) != 2:
            raise ValueError('共同归属及分离关系需要两个模块且不填写虚拟区间')
        return self


class RewriteIntent(Contract):
    operation: Literal['SetParallelism', 'RepartitionRegion', 'SetMicrobatch', 'SetRecomputation',
        'SetCommunicationPolicy', 'SetPipelineLayout', 'SelectPipelineSchedule',
        'SetOptimizerSharding', 'SetRankMapping', 'JointTransform']
    values: dict
    adjustable: list[str]
    preserve: list[str]
    hypothesis: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    target_task_ids: list[str] = Field(default_factory=list)
    target_flow_ids: list[str] = Field(default_factory=list)
    interaction_evidence_ids: list[str] = Field(default_factory=list)
    negative_effects: list[str] = Field(default_factory=list)
    parent_plan_id: str | None = None
    target_bottleneck: dict | None = None
    proposed_actions: list[dict] = Field(default_factory=list)
    affected_regions_or_flows: dict = Field(default_factory=dict)
    expected_effect: str | None = None
    possible_side_effects: list[str] = Field(default_factory=list)
    allowed_ranges: dict = Field(default_factory=dict)
    module_relations: list[ModuleRelation] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def merge_actions(cls, value):
        if not isinstance(value, dict):
            return value
        values = dict(value.get('values', {}))
        for action in value.get('proposed_actions', []):
            for name, target in action['values'].items():
                if target is None:
                    continue
                if name in values and values[name] != target:
                    raise ValueError('联合动作对同一变量给出了不同目标')
                values[name] = target
        bottleneck = value.get('target_bottleneck') or {}
        return dict(value, values=values,
            target_task_ids=sorted(set(value.get('target_task_ids', [])) | set(bottleneck.get('task_ids', []))),
            target_flow_ids=sorted(set(value.get('target_flow_ids', [])) | set(bottleneck.get('flow_ids', []))),
            negative_effects=list(dict.fromkeys(value.get('negative_effects', []) + value.get('possible_side_effects', []))))


class SearchBudget(Contract):
    max_agent_calls: int = Field(gt=0)
    max_agent_rounds: int = Field(gt=0)
    max_cpu_predictions: int = Field(gt=0)
    max_profiling_gpu_hours: float = Field(ge=0, allow_inf_nan=False)
    max_full_training_trials: int = Field(ge=0)
    max_total_gpu_hours: float = Field(ge=0, allow_inf_nan=False)
    max_walltime_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_no_improvement_rounds: int = Field(gt=0)
    estimated_trial_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_agent_tokens: int = Field(default=1000000, gt=0)
    max_benchmarks_per_round: int = Field(default=1, gt=0)
    max_candidates_per_round: int = Field(default=48, gt=0)
    profiling_warmup: int = Field(default=2, ge=1)
    profiling_measured: int = Field(default=4, ge=3)


class ProfilingRequest(Contract):
    request_id: str
    trial_ids: list[str]
    cost_keys: list[str]
    measurement_type: Literal['native_training_cost_profile', 'nccl', 'compute_communication', 'memory']
    execution_request: dict
    reason: str
    estimated_gpu_hours: float = Field(ge=0)
    priority: int = Field(ge=0)
    state: Literal['PENDING', 'APPROVED', 'RUNNING', 'COMPLETED', 'FAILED', 'DEFERRED',
        'OUT_OF_BUDGET', 'CANCELLED'] = 'PENDING'
    result: dict | None = None


class SystemConfig(Contract):
    request: dict
    model_config_path: str
    model_parameter_metadata_path: str | None = None
    profile_database: str
    profile_id: str | None = None
    baseline_directory: str
    output_directory: str
    continuation_history_path: str | None = None
    pods: list[str]
    cluster_context: str = 'prd'
    namespace: str = 'default'
    agent_model: str
    agent_reasoning_effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] | None = None
    agent_execution_enabled: bool = False
    external_agent_authorization_path: str | None = None
    max_llm_calls_per_round: int = Field(default=4, ge=0, le=4,
        description='每轮允许启动的不同Agent角色数量；工具追问的实际Provider请求受max_agent_calls控制')
    fast_agent_rounds: bool = False
    initial_cost_calibration_enabled: bool = True
    boundary_cost_calibration_enabled: bool = False
    llm_provider: Literal['codex', 'direct_api'] = 'direct_api'
    agent_cli_context_profile: Literal['default', 'structured'] = 'structured'
    agent_cli_model_instructions_file: str | None = None
    agent_request_profile: Literal['full', 'decision', 'compact'] = 'full'
    agent_recent_runtime_failure_evidence: bool = False
    fsdp_load_validation_seconds: int = Field(default=120, ge=60, le=900)
    constraint_review_policy: Literal['always', 'native_complete'] = 'always'
    single_decision_round: bool = False
    budgeted_investigation_enabled: bool = False
    compact_results: bool = False
    agent_token_budget_enabled: bool = True
    round_input_token_limit: int = Field(default=10000, gt=0)
    round_total_token_limit: int = Field(default=10000, gt=0)
    llm_base_url: str | None = None
    tokenizer_cache_directory: str | None = None
    budget: SearchBudget
    candidate_limit: int = Field(default=6, ge=1, le=16)
    capability_domains: dict[str, list[int]]
    gpu_lock_path: str
    evaluation_mode: Literal['full', 'config_only', 'graph_aware', 'graph_impact', 'multi_agent', 'seed', 'random_region'] = 'full'
    agent_ablation: Literal['full', 'without_dag_evidence', 'without_dag_model', 'without_agent',
        'without_joint_search', 'without_bottleneck_agent',
        'without_search_memory', 'single_agent'] = 'full'
    execution_evidence_format: Literal['legacy', 'execution_view', 'flat_records'] = 'legacy'
    run_namespace: str = ''
    selection_policy: Literal['investigation', 'topk', 'diversity'] = 'investigation'
    intent_count: int = Field(default=6, ge=1, le=32)
    use_backend_global_domains: bool = True
    hardware_topology_paths: list[str] = Field(default_factory=list)
    gpu_execution_enabled: bool = True
    benchmark_warmup: int = Field(default=10, ge=1)
    benchmark_measured: int = Field(default=30, ge=1)
    max_tool_rounds: int = Field(default=3, ge=0)
    initial_seed_limit: int = Field(default=6, ge=1)
    initial_configuration_seed: int = 0
    initial_request_from_config: bool = False
    shared_baseline_required: bool = False
    exploration_weight: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    ranking_top_k: int = Field(default=3, ge=2, le=16)
    uncalibrated_exploration_trials: int = Field(default=1, ge=0)
    predictor_cost_mode: Literal['hybrid', 'measured_only', 'analytical_only'] = 'hybrid'
    predictor_hardware_parameters: dict = Field(default_factory=dict)
    conditional_partition_max_states: int = Field(default=16, ge=1)
    conditional_partition_max_expansions: int = Field(default=20000, ge=1)
    conditional_partition_max_layouts: int = Field(default=64, ge=1)
    conditional_partition_max_replays: int = Field(default=32, ge=0)
    continuation_donor_selection: Literal['round_robin', 'coverage', 'anchored_coverage', 'direct_time', 'random'] = 'round_robin'
    continuation_donor_limit: int | None = Field(default=None, ge=1)
    planning_control_enabled: bool = False
    planning_selector: Literal['agent', 'deterministic_scope', 'deterministic_restart', 'deterministic_portfolio'] = 'agent'
    planning_portfolio_enabled: bool = False
    planning_full_strategy_enabled: bool = False
    planning_partition_engine: Literal['prefix', 'complete', 'path'] = 'prefix'
    planning_budget_allocation: Literal['equal_share', 'completion_reserve'] = 'equal_share'
    planning_execution_context_enabled: bool = False
    planning_initial_prompt_limit: int = Field(default=10000, ge=1000)
    agent_context_max_tokens: int = Field(default=24000, ge=1000)
    planning_runtime_reference: Literal['parent', 'initial'] = 'parent'
    planning_runtime_scope: Literal['open', 'fixed'] = 'open'
    planning_fixed_parent_id: str | None = None
    continuation_evaluation_timing: Literal['late', 'fixed_depth', 'early_independent', 'paired', 'budgeted_paired', 'terminal_paired', 'revisit_paired'] = 'late'
    continuation_early_fraction: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    investigation_max_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    reference_backend_source_hashes: dict[str, str] | None = None

    @model_validator(mode='after')
    def validate_allocation(self):
        per_node = self.request.get('gpus_per_node', 8)
        per_worker = self.request.get('gpus_per_worker', per_node)
        if type(per_worker) is not int or per_worker < 1 or per_node % per_worker:
            raise ValueError('每个Worker的GPU数必须整除单台物理节点的GPU数')
        expected = self.request['node_count'] * per_node // per_worker
        if len(self.pods) != expected or len(set(self.pods)) != expected:
            raise ValueError('Pod数量必须覆盖实际训练Worker且身份唯一')
        if not self.agent_model:
            raise ValueError('真实Agent调用需要明确模型名称')
        return self
