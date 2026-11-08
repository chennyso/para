from datetime import datetime, timezone
from copy import deepcopy
from collections import Counter
from itertools import combinations, count
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import random
import subprocess
from typing import Annotated, Literal, get_args

from pydantic import Field, ValidationError, create_model
import tiktoken
from openai.lib._pydantic import to_strict_json_schema

from contracts import Contract, ModuleRelation, ProfilingRequest, RewriteIntent, SystemConfig, TrainingPlan, fingerprint
from agents.single_round import ScopeRoundDecision, ScopeRow
from core_data_models import HardwareTopology, MemoryPolicy
from graph.capability_registry import CapabilityRegistry, tensor_parallel_limit
from graph.complete_rewrite import CompleteAndRewrite
from graph.model_graph import read_model, build_operator_graph
from predictor import FastPredictor
from predictor.dag_projection import PROJECTION_REVISION
from predictor.fast_predictor import ranges
from cost.runtime_calibration import configuration
from storage.experiment_store import ExperimentStore, prediction_evidence
from search.seed_planner import SeedPlanner
from search.joint_planning import JointPlanning
from search.execution_comparison import compare_execution_reports
from search.strategy_evidence import (collective_summary, compare_strategy_structure,
    execution_structure, measured_strategy_memory, proposed_strategy_structure, complete_strategy_comparisons,
    strategy_decision_evidence, strategy_decision_values)
from search.execution_view import execution_view, VIEW_REVISION
from search.execution_decisions import execution_analysis, candidate_decisions
from search.graph_exploration import (execution_shape, pipeline_investigations,
    exploration_representative, physical_parallel_investigations)
from search.measurement_frontier import profile_decisions, ranking_sensitive_candidates
from search.native_strategy_effects import native_strategy_effects
from search.schedule_controls import effective_schedule_execution, schedule_control_options
from search.experiment_value import budgeted_selection_values, experiment_value
from scheduling.native_microbatches import native_schedule_settings, schedule_delta
from agents.evidence_policy import AblationEvidencePolicy
from search.investigation_control import InvestigationControl, PlanningInvestigationBatch
from search.communication_placement import placement_evidence
from search.family_state import FamilyLedger, GLOBAL_FIELDS, global_identity, global_values
from adapters.megatron_adapter import MegatronAdapter
from agents.context_budget import pack_context
from megatron.core.frontier.agents.direct_provider import DirectAPIProvider
from graph.process_groups import process_groups
from graph.native_recomputation import selective_modules
from graph.execution_impact import execution_impact
from predictor.profile_database import (signature, plan_signature, same_training_plan,
    normalized_execution)
from search.training_semantics import require_fixed_training_semantics
from predictor.state_sharding import state_sharding_config
from memory_failure import oom_allocation_site, write_memory_failure_witness


TOKEN_ENCODING = None
TOKEN_ENCODING_NAME = 'o200k_base'
LAYERED_CONTEXT_VERSION = 'native_policy_and_execution_evidence_v48_execution_revision'
AGENT_ROLE_NAMES = {
    'PerformanceAnalysisAgent': 'Bottleneck Analysis Agent',
    'JointStrategyAgent': 'Joint Strategy Generation Agent',
    'ConstraintValidationAgent': 'Constraint and Validation Agent',
    'ExperimentInvestigationAgent': 'Profiling and Evaluation Agent',
    'SingleAgent': 'Single Agent',
}
EvidenceIdentifier = Annotated[str, Field(min_length=1, max_length=256)]


class AgentConstraintConflict(ValueError):
    def __init__(self, result):
        self.result = dict(result, status='CONSTRAINT_CONFLICT')
        super().__init__(self.result)


def token_count(value):
    global TOKEN_ENCODING
    if TOKEN_ENCODING is None:
        TOKEN_ENCODING = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return len(TOKEN_ENCODING.encode(text))


def bounded_observation(value, limit=16):
    if isinstance(value, list):
        return dict(total_count=len(value), items=[bounded_observation(item, limit) for item in value[:limit]])
    if isinstance(value, dict):
        return {key: bounded_observation(item, limit) for key, item in value.items()}
    return value


def gpu_uuid(device):
    value = device.get('uuid', device.get('gpu_uuid'))
    return str(value).removeprefix('GPU-').lower() if value is not None else None


def profile_hardware_matches(profile, devices):
    actual = {row['physical_rank']: gpu_uuid(row) for row in devices}
    mapping = profile['request'].get('rank_mapping', list(range(len(devices))))
    expected = [actual.get(rank) for rank in mapping]
    observed = [gpu_uuid(row) for row in profile.get('hardware', [])]
    return bool(expected) and all(value is not None for value in expected + observed) and expected == observed


def select_hardware_profile(database, request, devices, preferred_id=None):
    compatible = []
    for identifier, encoded in database.connection.execute(
            'SELECT id, document FROM profiles WHERE signature=? ORDER BY id', (signature(request),)):
        document = json.loads(encoded)
        if profile_hardware_matches(document, devices):
            compatible.append((identifier, document))
    if not compatible:
        return None, None
    preferred = next((row for row in compatible if row[0] == preferred_id), None)
    selected = preferred or max(compatible, key=lambda row: (
        row[1]['request']['decoder_chunks'] == request['decoder_chunks'], row[1].get('anchor') is not None,
        sum(len(windows) for windows in row[1].get('measured_windows', {}).values())))
    return selected[0], database.get(selected[0], request)


class AnalysisToolRequest(Contract):
    history_view: Literal['summary', 'structure', 'notes', 'rounds'] = 'summary'
    tool: Literal['inspect_critical_path', 'inspect_dag_subgraph', 'inspect_measured_trace',
        'inspect_task_dependencies', 'inspect_region_cost', 'inspect_communication_flow',
        'inspect_rank_topology', 'inspect_shared_resources', 'query_affected_decisions', 'get_trial_history']
    candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16, ge=1, le=32)
    region_id: int | None = Field(default=None, ge=0)
    task_ids: list[str] = Field(default_factory=list, max_length=8)
    dependency_depth: int = Field(default=1, ge=1, le=3)
    dependency_direction: Literal['predecessors', 'successors', 'both'] = 'both'


class BottleneckDecision(Contract):
    bottleneck_type: Literal['MEASURED', 'PREDICTED', 'HYPOTHESIS'] = 'HYPOTHESIS'
    affected_regions: list[int] = Field(default_factory=list)
    affected_tasks: list[EvidenceIdentifier] = Field(default_factory=list)
    related_variables: list[str] = Field(default_factory=list)
    hypotheses: list[str]
    evidence_refs: list[EvidenceIdentifier]
    target_regions: list[int]
    requested_metrics: list[str]
    tool_requests: list[AnalysisToolRequest] = Field(default_factory=list, max_length=4)


class RewriteValues(Contract):
    memory_policy: MemoryPolicy | None = None
    tp: int | None = None
    pp: int | None = None
    cp: int | None = None
    ep: int | None = None
    etp: int | None = None
    expert_dp: int | None = None
    dp: int | None = None
    vpp: int | None = None
    micro_batch: int | None = None
    group: int | None = None
    decoder_chunks: list[int] | None = None
    recompute: Literal['none', 'full', 'selective-mlp', 'selective'] | None = None
    bucket_elements: int | None = Field(default=None, gt=0)
    overlap_p2p: bool | None = None
    rank_mapping: list[int] | None = None
    pipeline_schedule: Literal['1f1b', 'interleaved_1f1b'] | None = None
    optimizer_sharding: Literal['distributed', 'replicated', 'fsdp'] | None = None
    precision_aware_optimizer: bool | None = None
    optimizer_offload_fraction: float | None = Field(default=None, ge=0, le=1)
    optimizer_transfer_overlap: bool | None = None
    optimizer_pin_cpu_params: bool | None = None
    execution_backend: Literal['megatron_ddp', 'megatron_fsdp'] | None = None
    megatron_fsdp_version: Literal[0, 1] | None = None
    fsdp_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] | None = None
    dense_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] | None = None
    expert_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] | None = None
    fsdp_unit: Literal['transformer_layer', 'model_chunk'] | None = None
    fsdp_prefetch: bool | None = None
    fsdp_prefetch_distance: int | None = Field(default=None, ge=1)
    fsdp_reshard_after_forward: bool | None = None
    fsdp_double_buffer: bool | None = None
    fsdp_bucket_elements: int | None = Field(default=None, gt=0)
    dp_inner_size: int | None = Field(default=None, gt=0)
    dp_outer_size: int | None = Field(default=None, gt=0)
    expert_dp_inner_size: int | None = Field(default=None, gt=0)
    expert_dp_outer_size: int | None = Field(default=None, gt=0)
    outer_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] | None = None
    expert_outer_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] | None = None


StrategyVariable = Literal['tp', 'pp', 'cp', 'ep', 'etp', 'expert_dp', 'dp', 'vpp',
    'micro_batch', 'group', 'decoder_chunks', 'recompute', 'bucket_elements', 'overlap_p2p',
    'rank_mapping', 'pipeline_schedule', 'optimizer_sharding', 'precision_aware_optimizer',
    'optimizer_offload_fraction', 'optimizer_transfer_overlap', 'optimizer_pin_cpu_params', 'memory_policy',
    'execution_backend', 'fsdp_sharding_strategy', 'dense_sharding_strategy',
    'expert_sharding_strategy', 'fsdp_unit', 'fsdp_prefetch', 'fsdp_prefetch_distance', 'fsdp_reshard_after_forward',
    'fsdp_double_buffer', 'fsdp_bucket_elements', 'megatron_fsdp_version',
    'dp_inner_size', 'dp_outer_size', 'expert_dp_inner_size', 'expert_dp_outer_size',
    'outer_sharding_strategy', 'expert_outer_sharding_strategy']


class ProposedAction(Contract):
    operation: Literal['SetParallelism', 'RepartitionRegion', 'SetMicrobatch', 'SetRecomputation',
        'SetCommunicationPolicy', 'SetPipelineLayout', 'SelectPipelineSchedule',
        'SetOptimizerSharding', 'SetRankMapping']
    values: RewriteValues
    rationale: str


class BottleneckTarget(Contract):
    source: Literal['MEASURED', 'PREDICTED', 'HYPOTHESIS']
    task_ids: list[str] = Field(default_factory=list)
    flow_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    description: str


class AffectedScope(Contract):
    region_ids: list[int] = Field(default_factory=list)
    flow_ids: list[str] = Field(default_factory=list)


class AllowedRanges(Contract):
    memory_policy: list[MemoryPolicy] | None = None
    tp: list[int] | None = None
    pp: list[int] | None = None
    cp: list[int] | None = None
    ep: list[int] | None = None
    etp: list[int] | None = None
    expert_dp: list[int] | None = None
    dp: list[int] | None = None
    vpp: list[int] | None = None
    micro_batch: list[int] | None = None
    group: list[int] | None = None
    recompute: list[Literal['none', 'full', 'selective-mlp', 'selective']] | None = None
    optimizer_sharding: list[Literal['distributed', 'replicated', 'fsdp']] | None = None
    pipeline_schedule: list[Literal['1f1b', 'interleaved_1f1b']] | None = None
    overlap_p2p: list[bool] | None = None
    bucket_elements: list[int] | None = None
    decoder_chunks: list[list[int]] | None = None
    rank_mapping: list[list[int]] | None = None
    precision_aware_optimizer: list[bool] | None = None
    optimizer_offload_fraction: list[float] | None = None
    optimizer_transfer_overlap: list[bool] | None = None
    optimizer_pin_cpu_params: list[bool] | None = None
    execution_backend: list[Literal['megatron_ddp', 'megatron_fsdp']] | None = None
    fsdp_sharding_strategy: list[Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params']] | None = None
    dense_sharding_strategy: list[Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params']] | None = None
    expert_sharding_strategy: list[Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params']] | None = None
    fsdp_unit: list[Literal['transformer_layer', 'model_chunk']] | None = None
    fsdp_prefetch: list[bool] | None = None
    fsdp_prefetch_distance: list[int] | None = None
    fsdp_reshard_after_forward: list[bool] | None = None
    fsdp_double_buffer: list[bool] | None = None
    fsdp_bucket_elements: list[int] | None = None
    megatron_fsdp_version: list[Literal[0, 1]] | None = None
    dp_inner_size: list[int] | None = None
    dp_outer_size: list[int] | None = None
    expert_dp_inner_size: list[int] | None = None
    expert_dp_outer_size: list[int] | None = None
    outer_sharding_strategy: list[Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params']] | None = None
    expert_outer_sharding_strategy: list[Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params']] | None = None


class NativeScopeFixedValues(RewriteValues):
    optimizer_sharding: Literal['distributed', 'replicated'] | None = None


class NativeScopeAllowedRanges(AllowedRanges):
    optimizer_sharding: list[Literal['distributed', 'replicated']] | None = None


class NativeScopeRow(ScopeRow):
    parallel: list[Annotated[int, Field(ge=1)]] = Field(min_length=5, max_length=5)
    adjustable: list[StrategyVariable] = Field(max_length=len(get_args(StrategyVariable)))
    allowed_ranges: NativeScopeAllowedRanges
    fixed_values: NativeScopeFixedValues


class NativeScopeRoundDecision(Contract):
    candidates: list[NativeScopeRow] = Field(min_length=1, max_length=2)
    measurement_preference: Literal['training', 'missing_execution_costs'] = 'training'


LOCAL_SCOPE_GLOBAL_FIELDS = frozenset(('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping'))
LocalScopeVariable = Literal[tuple(name for name in get_args(StrategyVariable) if name not in LOCAL_SCOPE_GLOBAL_FIELDS)]
LocalScopeFixed = create_model('LocalScopeFixed', __base__=Contract, **{
    key: (field.annotation, field) for key, field in NativeScopeFixedValues.model_fields.items()
    if key not in LOCAL_SCOPE_GLOBAL_FIELDS})
LocalScopeRanges = create_model('LocalScopeRanges', __base__=Contract, **{
    key: (field.annotation, field) for key, field in NativeScopeAllowedRanges.model_fields.items()
    if key not in LOCAL_SCOPE_GLOBAL_FIELDS})


class LocalNativeScopeRow(NativeScopeRow):
    adjustable: list[LocalScopeVariable] = Field(max_length=len(get_args(LocalScopeVariable)))
    fixed_values: LocalScopeFixed
    allowed_ranges: LocalScopeRanges


class LocalNativeScopeRoundDecision(NativeScopeRoundDecision):
    candidates: list[LocalNativeScopeRow] = Field(min_length=1, max_length=2)


def native_geometry_choices(families):
    axes = ('tp', 'pp', 'cp', 'ep', 'etp')
    return sorted({state['execution_backend'] + ':' + '/'.join(str(family['values'][axis]) for axis in axes)
        for family in families for state in family['state_variants']})


def native_geometry_decision_schema(choices, world):
    if not choices:
        raise ValueError('本轮硬件和训练后端没有共同支持的并行组合')
    def local_fields(model):
        return {key: (field.annotation, field) for key, field in model.model_fields.items()
            if key != 'execution_backend'}
    fixed = create_model('HardwareLocalFixedValues', __base__=Contract, **local_fields(LocalScopeFixed))
    ranges = create_model('HardwareLocalAllowedRanges', __base__=Contract, **local_fields(LocalScopeRanges))
    adjustable = Literal[tuple(name for name in get_args(LocalScopeVariable) if name != 'execution_backend')]
    rank = Literal[tuple(range(world))]
    row = create_model('HardwareNativeScopeRow', __base__=LocalNativeScopeRow,
        parallel=(Literal[tuple(choices)], ...), ranks=(list[rank], Field(min_length=world, max_length=world)),
        adjustable=(list[adjustable], Field(max_length=len(get_args(adjustable)))),
        fixed_values=(fixed, ...), allowed_ranges=(ranges, ...))
    return create_model('HardwareNativeScopeRoundDecision', __base__=LocalNativeScopeRoundDecision,
        candidates=(list[row], Field(min_length=1, max_length=2)))


class AgentRewriteIntent(Contract):
    proposal_kind: Literal['conditional_intent', 'complete_strategy'] = 'conditional_intent'
    operation: Literal['SetParallelism', 'RepartitionRegion', 'SetMicrobatch', 'SetRecomputation',
        'SetCommunicationPolicy', 'SetPipelineLayout', 'SelectPipelineSchedule',
        'SetOptimizerSharding', 'SetRankMapping', 'JointTransform']
    values: RewriteValues
    adjustable: list[StrategyVariable]
    preserve: list[StrategyVariable]
    hypothesis: str = Field(min_length=1)
    evidence_refs: list[EvidenceIdentifier] = Field(min_length=1)
    target_task_ids: list[EvidenceIdentifier] = Field(default_factory=list)
    target_flow_ids: list[EvidenceIdentifier] = Field(default_factory=list)
    interaction_evidence_ids: list[EvidenceIdentifier] = Field(default_factory=list)
    negative_effects: list[str] = Field(default_factory=list)
    parent_plan_id: EvidenceIdentifier | None = None
    target_bottleneck: BottleneckTarget | None = None
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    affected_regions_or_flows: AffectedScope = Field(default_factory=AffectedScope)
    expected_effect: str | None = None
    possible_side_effects: list[str] = Field(default_factory=list)
    allowed_ranges: AllowedRanges = Field(default_factory=AllowedRanges)
    module_relations: list[ModuleRelation] = Field(default_factory=list)
    family_id: str | None = None

    def executable(self):
        payload = self.model_dump(exclude={'family_id', 'proposal_kind'}, exclude_none=True)
        payload['allowed_ranges'] = {name: values for name, values in
            payload.get('allowed_ranges', {}).items() if values}
        return RewriteIntent.model_validate(payload)


class IntentBatch(Contract):
    intents: list[AgentRewriteIntent] = Field(default_factory=list, max_length=32)
    reason: str
    evidence_refs: list[EvidenceIdentifier] = Field(default_factory=list)
    tool_requests: list['InvestigationToolRequest'] = Field(default_factory=list, max_length=4)


class InvestigationToolRequest(Contract):
    history_view: Literal['summary', 'structure', 'notes', 'rounds'] = 'summary'
    tool: Literal['inspect_execution_graph', 'evaluate_memory', 'compare_candidates', 'inspect_prediction_error',
        'evaluate_candidate', 'inspect_critical_path', 'inspect_region', 'inspect_communication',
        'inspect_memory', 'get_trial_history', 'get_best_plan', 'get_supported_transforms', 'get_budget_status',
        'inspect_training_graph', 'inspect_dag_subgraph', 'inspect_measured_trace', 'inspect_task_dependencies', 'inspect_region_cost',
        'inspect_communication_flow', 'inspect_rank_topology', 'inspect_shared_resources',
        'query_affected_decisions', 'preview_transformation', 'compare_candidate_graph_delta', 'inspect_trial_history',
        'analyze_transformation_interactions', 'inspect_global_families', 'inspect_policy_frontier', 'inspect_execution_impact']
    candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=2)
    family_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=2)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16, ge=1, le=32)
    region_id: int | None = Field(default=None, ge=0)
    task_ids: list[str] = Field(default_factory=list, max_length=8)
    dependency_depth: int = Field(default=1, ge=1, le=3)
    dependency_direction: Literal['predecessors', 'successors', 'both'] = 'both'
    transformation_intent: AgentRewriteIntent | None = None
    transformation_intents: list[AgentRewriteIntent] = Field(default_factory=list, max_length=3)


BottleneckDecision.model_rebuild()
IntentBatch.model_rebuild()


class SearchBudgetExhausted(RuntimeError):
    pass


class EvaluationToolRequest(Contract):
    history_view: Literal['summary', 'structure', 'notes', 'rounds'] = 'summary'
    tool: Literal['evaluate_candidate', 'inspect_execution_graph', 'inspect_critical_path',
        'inspect_dag_subgraph', 'inspect_task_dependencies', 'evaluate_memory', 'inspect_memory',
        'compare_candidates', 'inspect_prediction_error', 'get_trial_history', 'get_best_plan',
        'get_budget_status', 'inspect_global_families', 'inspect_communication_flow', 'inspect_rank_topology']
    candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=2)
    family_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=2)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8, ge=1, le=32)
    region_id: int | None = Field(default=None, ge=0)
    task_ids: list[str] = Field(default_factory=list, max_length=8)
    dependency_depth: int = Field(default=1, ge=1, le=3)
    dependency_direction: Literal['predecessors', 'successors', 'both'] = 'both'


class CandidateAssessment(Contract):
    trial_id: EvidenceIdentifier
    expected_gain: str = Field(min_length=1)
    added_cost: str = Field(min_length=1)
    unresolved_costs: list[str]
    next_action: Literal['BENCHMARK', 'PROFILE', 'DEFER', 'REUSE']
    reason: str = Field(min_length=1)


class InvestigationSelection(Contract):
    accepted_candidate_ids: list[EvidenceIdentifier]
    evidence_refs: list[EvidenceIdentifier]
    reason: str
    missing_measurements: list[str]
    tool_requests: list[EvaluationToolRequest] = Field(default_factory=list, max_length=4)
    accepted_profiling_request_ids: list[EvidenceIdentifier] = Field(default_factory=list)
    load_validation_candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list)
    exploration_candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list)
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list, max_length=8)


class InvestigationDecision(Contract):
    conclusions: list[str]
    evidence_refs: list[EvidenceIdentifier]
    next_focus: list[str]
    requested_profiles: list[str]


class ConstraintValidationToolRequest(Contract):
    tool: Literal['complete_training_plans', 'solve_conditional_partition', 'validate_training_plans']
    parent_trial_id: EvidenceIdentifier
    strategy_evidence_id: EvidenceIdentifier
    intent_indices: list[int] = Field(default_factory=list, max_length=8)
    candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=16)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4, ge=1, le=8)


class ConstraintIntentDecision(Contract):
    intent_index: int = Field(ge=0)
    status: Literal['COMPLETED', 'CONSTRAINT_CONFLICT', 'UNKNOWN']
    candidate_ids: list[EvidenceIdentifier] = Field(default_factory=list, max_length=16)
    evidence_refs: list[EvidenceIdentifier] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1)


class ConstraintValidationDecision(Contract):
    intent_results: list[ConstraintIntentDecision] = Field(default_factory=list, max_length=8)
    evidence_refs: list[EvidenceIdentifier] = Field(min_length=1, max_length=24)
    reason: str = Field(min_length=1)
    tool_requests: list[ConstraintValidationToolRequest] = Field(default_factory=list, max_length=4)


class EvidenceGuidedSearch(InvestigationControl, JointPlanning):
    def __init__(self, config, *, prepare_only=False):
        self.config = SystemConfig.model_validate(config)
        self.prepare_only = prepare_only
        if prepare_only and self.config.planning_control_enabled:
            raise ValueError('首轮预规划需要使用正式训练搜索入口')
        if self.config.tokenizer_cache_directory is not None:
            tokenizer_cache = Path(self.config.tokenizer_cache_directory)
            tokenizer_cache.mkdir(parents=True, exist_ok=True)
            os.environ['TIKTOKEN_CACHE_DIR'] = str(tokenizer_cache)
        if self.config.agent_execution_enabled:
            token_count('')
        self.output = Path(self.config.output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        self.model, self.model_graph = read_model(self.config.model_config_path,
            parameter_metadata_path=self.config.model_parameter_metadata_path)
        self.model_configuration = json.loads(Path(self.config.model_config_path).read_text())
        self.registry = CapabilityRegistry(self.model_graph['module_structure'])
        request = self.config.request
        self.adapter = MegatronAdapter(self.config)
        self.predictor = FastPredictor(self.config.profile_database)
        self.profile_id = self.config.profile_id or self.predictor.database.select(request)
        self.profile = self.predictor.database.get(self.profile_id, request) if self.profile_id else None
        if self.profile is not None and not self.profile.get('source_directory'):
            profile_directory = Path(self.config.profile_database).parent.parent / 'profiling' / 'runs' / self.profile['request']['run_id']
            if profile_directory.is_dir():
                self.predictor.database.register_source(self.profile_id, profile_directory)
                self.profile = self.predictor.database.get(self.profile_id, request)
        hardware = []
        for path in self.config.hardware_topology_paths:
            document = json.loads(Path(path).read_text())
            hardware.extend(document if isinstance(document, list) else [document])
        hardware_path = self.output / 'hardware-topology.json'
        if not hardware and hardware_path.is_file():
            hardware = json.loads(hardware_path.read_text())
        if self.config.gpu_execution_enabled and not self.prepare_only:
            if hardware:
                snapshot = self.output / ('hardware-topology-' + fingerprint(hardware) + '.json')
                if not snapshot.exists():
                    snapshot.write_text(json.dumps(hardware, indent=2, allow_nan=False) + '\n')
            hardware = self.adapter.discover_hardware()
            hardware_path.write_text(json.dumps(hardware, indent=2, allow_nan=False) + '\n')
        elif not hardware:
            raise ValueError('关闭GPU执行时需要提供已有硬件拓扑文件')
        devices = []
        network = []
        for index, node in enumerate(hardware):
            devices.extend(device | dict(node=node.get('node', node['hostname']),
                node_index=index, physical_rank=index * request['gpus_per_node'] + device['index'])
                for device in node['devices'])
            network.extend(adapter | dict(node_index=index) for adapter in node['adapters'])
        self.profile_id, self.profile = select_hardware_profile(
            self.predictor.database, request, devices, self.profile_id)
        if self.profile is not None:
            self.profile = self.predictor._special_module_breakdown(self.profile)
        capacity = (min(row['capacity'] for row in self.profile['memory'].values()) / 2**30
                    if self.profile else min(device['capacity_bytes'] for device in devices) / 2**30)
        software = hardware[0]['software']
        if any(node['software'] != software for node in hardware):
            raise ValueError('当前训练计划要求所有参与节点具有相同的实测Torch、CUDA和NCCL版本')
        self.topology = HardwareTopology(request['node_count'] * request['gpus_per_node'],
            request['node_count'], request['gpus_per_node'], capacity, devices=devices,
            network_paths=network, software=software,
            environment_per_node={str(index): node.get('launch_environment', node['environment'])
                for index, node in enumerate(hardware)},
            environment_per_rank={key: value for node in hardware
                for key, value in node.get('environment_per_rank', {}).items()})
        self.hardware_identity = dict(devices=devices, network=network,
            software=[node['software'] for node in hardware])
        self.execution_resources = {index: node.get('pod_resources') for index, node in enumerate(hardware)}
        self.backend_identity = self.profile['source_hashes'] if self.profile else {
            name: fingerprint((self.adapter.runner.ROOT / name).read_text())
            for name in ('hybrid_training_entry.py', 'frontier_hybrid_provider.py')}
        if self.config.reference_backend_source_hashes is not None:
            reference_sources = self.config.reference_backend_source_hashes
            if self.config.gpu_execution_enabled:
                raise ValueError('归档后端身份仅用于关闭GPU执行的成本回放')
            registered_sources = [json.loads(row[0])['source_hashes'] for row in
                self.predictor.database.connection.execute('SELECT document FROM profiles')]
            if reference_sources not in registered_sources or request.get('backend_fingerprint') != fingerprint(reference_sources):
                raise ValueError('归档后端身份必须来自实际成本且匹配请求指纹')
            self.backend_identity = reference_sources
        self.domains = (self.registry.parallel_domains(self.model, self.topology, request)
                        if self.config.use_backend_global_domains else self.config.capability_domains)
        maximum_tp = tensor_parallel_limit(request, self.model.name)
        self.domains = dict(self.domains, tp=[value for value in self.domains['tp'] if value <= maximum_tp])
        self.rewriter = CompleteAndRewrite(self.model, self.topology, self.registry, self.domains)
        source_root = Path(__file__).resolve().parents[1]
        self.constraint_implementation = fingerprint(dict(
            sources={name: fingerprint((source_root / name).read_text())
            for name in ('contracts.py', 'search/system_search.py', 'search/joint_planning.py',
                'search/conditional_partition.py', 'search/continuation.py', 'search/seed_planner.py',
                'search/joint_scope.py', 'search/policy_frontier.py', 'search/execution_comparison.py',
                'graph/execution_impact.py', 'graph/recompute_effect.py',
                'search/communication_placement.py', 'search/strategy_evidence.py', 'search/investigation_control.py',
                'search/planning_portfolio.py', 'search/complete_partition.py',
                'search/execution_path_model.py', 'search/execution_view.py', 'scheduling/native_microbatches.py',
                'validation/parallel_validator.py', 'agents/program_agent.py',
                'agents/strategy_transformations.py', 'agents/architect_agent.py',
                'predictor/fast_predictor.py', 'predictor/measured_modules.py',
                'predictor/dag_projection.py',
                'graph/complete_rewrite.py', 'graph/native_recomputation.py', 'scheduling/task_graph_builder.py',
                'predictor/state_sharding.py')},
            partition_budget=dict(states=self.config.conditional_partition_max_states,
                expansions=self.config.conditional_partition_max_expansions,
                layouts=self.config.conditional_partition_max_layouts,
                replays=self.config.conditional_partition_max_replays)))
        predictor_parameters = dict(self.config.predictor_hardware_parameters)
        if all(node.get('launch_environment') is not None for node in hardware):
            environments = {str(index): node['launch_environment'] for index, node in enumerate(hardware)}
            network_parameters = dict(predictor_parameters.get('network', {}))
            network_parameters.setdefault('environment_per_node', environments)
            if len(hardware) == 1:
                network_parameters.setdefault('environment', hardware[0]['launch_environment'])
            predictor_parameters['network'] = network_parameters
        self.predictor.configure_projection(self.model, self.model_graph, self.topology,
            predictor_parameters, self.config.predictor_cost_mode)
        self.predictor.set_preferred_profile(self.profile_id)
        self.store = ExperimentStore(self.output / 'experiments.sqlite')
        self.family_ledger = FamilyLedger(self.store)
        self.cost_revision = self.predictor.database.cost_database_revision()
        self.family_ledger.invalidate_costs(self.cost_revision)
        for existing_trial in self.store.trials():
            self.family_ledger.record_trial(existing_trial, self.cost_revision)
        self.dag_cache = {}
        self.global_family_cache = {}
        self.provider = None
        self.started = time.perf_counter()
        self.session_id = str(time.time_ns())
        self.store.set_cost_clock(self.session_id, self.started)
        self.store.record_session_time(self.session_id, 0)

    def predict_request(self, request, candidate_graph=None, parent_request=None):
        return self.predictor.evaluate(request, profile_id=self.profile_id,
            candidate_graph=candidate_graph, parent_plan=parent_request, persist=False)

    def compare_native_schedules(self, parent, target_request, target_prediction=None, *, allow_replay=True):
        source = parent['plan']['backend_config']
        result = schedule_delta(source, target_request)
        if not hasattr(self, '_schedule_comparison_predictions'):
            self._schedule_comparison_predictions = {}

        def timed(request, existing):
            if existing and existing.get('cost_database_revision') == self.cost_revision:
                graph = self.load_prediction_graph(existing)
                if graph and graph.get('tasks'):
                    return dict(existing, execution_graph=graph)
            if not allow_replay:
                return None
            key = (self.cost_revision, plan_signature(request), parent['trial_id'])
            if key not in self._schedule_comparison_predictions:
                if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                    return None
                self.check_walltime()
                candidate = self.rewriter.candidate(request)
                structural = self.rewriter.compiler.observation(candidate, [], [])['training_graph']
                predicted = self.predict_request(request, structural, self.projection_parent_context(source))
                tag = 'native-schedule-preview:' + str(time.time_ns())
                self.store.charge(tag, 'cpu_predictions', 1)
                self.store.charge(tag + ':seconds', 'cpu_seconds', predicted['evaluation_seconds'])
                self._schedule_comparison_predictions[key] = self.persist_prediction_graph(predicted)
            cached = self._schedule_comparison_predictions[key]
            return dict(cached, execution_graph=self.load_prediction_graph(cached))

        comparison = compare_execution_reports(source, timed(source, parent.get('prediction')),
            target_request, timed(target_request, target_prediction), limit=2)
        before, after = comparison.get('before', {}), comparison.get('after', {})
        result['execution_comparison'] = dict(status=comparison['status'], reason=comparison.get('reason'),
            predicted_step_delta_ms=comparison.get('predicted_step_delta_ms'),
            exposed_communication_before_ms=before.get('modeled_exposed_communication_ms'),
            exposed_communication_after_ms=after.get('modeled_exposed_communication_ms'),
            estimated_peak_before_bytes=before.get('estimated_peak_bytes'),
            estimated_peak_after_bytes=after.get('estimated_peak_bytes'),
            critical_path_tasks_before=before.get('critical_path_task_count'),
            critical_path_tasks_after=after.get('critical_path_task_count'),
            p2p_timeline_before=before.get('p2p_timeline_samples'),
            p2p_timeline_after=after.get('p2p_timeline_samples'),
            numeric_comparison_eligible=comparison.get('numeric_comparison_eligible', False))
        result['timing_evidence'] = ('SIMULATED' if result['execution_comparison']['numeric_comparison_eligible']
            else 'UNKNOWN')
        return result

    def persist_prediction_graph(self, prediction):
        graph = prediction.get('execution_graph')
        partial = prediction.get('partial_execution_graph')
        costs = prediction.get('evidence', {}).get('costs', [])
        if (graph is None or 'tasks' not in graph) and partial is None and not costs:
            return prediction
        directory = self.output / 'prediction-graphs'
        directory.mkdir(parents=True, exist_ok=True)
        artifact = dict(execution_graph=graph, partial_execution_graph=partial, costs=costs)
        artifact_id = fingerprint(artifact)
        path = directory / f'{artifact_id}.json.gz'
        if path.is_file():
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                if fingerprint(json.load(stream)) != artifact_id:
                    raise ValueError('执行图文件与内容身份不同')
        else:
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                json.dump(artifact, stream,
                          ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'))
        output = dict(prediction)
        if partial is not None and 'tasks' in partial:
            output['partial_execution_graph'] = {key: value for key, value in partial.items() if key != 'tasks'}
            output['partial_execution_graph'].update(storage_path=str(path), storage_format='gzip_json')
        if graph is not None and 'tasks' in graph:
            critical = prediction.get('diagnosis', {}).get('critical_tasks', [])[:16]
            summary = {key: value for key, value in graph.items() if key != 'tasks'}
            summary.update(storage_path=str(path), storage_format='gzip_json',
                critical_task_records={key: graph['tasks'][key] for key in critical if key in graph['tasks']})
            output['execution_graph'] = summary
        if costs:
            evidence = dict(output['evidence'])
            evidence['cost_record_count'] = len(costs)
            evidence['cost_status_counts'] = dict(Counter(
                row.get('status', 'UNKNOWN') for row in costs))
            evidence['cost_source_counts'] = dict(Counter(
                row.get('time_source', row.get('source', 'UNKNOWN')) for row in costs))
            evidence['cost_records_artifact'] = str(path)
            evidence.pop('costs')
            output['evidence'] = evidence
        return output

    @staticmethod
    def load_prediction_graph(prediction):
        graph = prediction.get('execution_graph')
        if graph is None or 'tasks' in graph:
            return graph
        path = graph.get('storage_path')
        if path is None:
            return None
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)['execution_graph']

    @staticmethod
    def load_decision_graph(prediction):
        graph = EvidenceGuidedSearch.load_prediction_graph(prediction)
        if graph is not None:
            return graph
        partial = prediction.get('partial_execution_graph')
        if partial is None or 'tasks' in partial:
            return partial
        with gzip.open(partial['storage_path'], 'rt', encoding='utf-8') as stream:
            return json.load(stream)['partial_execution_graph']

    def projection_parent_context(self, parent_request=None, parent_trial_id=None):
        parent = None
        if parent_trial_id is not None:
            parent = next((row for row in self.store.trials()
                if row['trial_id'] == parent_trial_id), None)
        if parent is None and parent_request is not None:
            parent = next((row for row in self.store.trials()
                if row['plan']['backend_config']['run_id'] == parent_request.get('run_id')), None)
        if parent is None:
            return parent_request
        context = dict(parent['plan']['backend_config'])
        measurement = parent.get('measurement')
        if measurement is not None and measurement.get('status') == 'PASS':
            context['_measured_step_time_ms'] = measurement['step_seconds'] * 1000
            context['_measurement_run_id'] = measurement['run_id']
        prediction = parent.get('prediction')
        if (prediction is not None and prediction['performance'].get('step_time_ms') is not None
                and prediction.get('cost_database_revision') == self.cost_revision):
            context['_predicted_raw_step_time_ms'] = prediction['performance'].get(
                'raw_resource_aware_step_time_ms', prediction['performance']['step_time_ms'])
            context['_prediction_id'] = prediction['prediction_id']
        return context

    def predict_trial(self, trial, candidate_graph=None, parent_request=None, force=False):
        self.synchronize_cost_revision()
        if (trial['prediction'] is not None and not force
                and trial['prediction'].get('cost_database_revision') == self.cost_revision
                and trial['prediction'].get('evidence', {}).get('predictor_revision') == PROJECTION_REVISION):
            return trial
        self.check_walltime()
        parent_context = self.projection_parent_context(parent_request,
            trial['plan'].get('parent_plan_id'))
        cache_key = fingerprint(dict(cost_revision=self.cost_revision,
            candidate=plan_signature(trial['plan']['backend_config']), parent=parent_context))
        prediction = None if force else getattr(self, '_continuation_predictions', {}).get(cache_key)
        cached = prediction is not None
        if cached:
            prediction = deepcopy(prediction)
            source_prediction_id = prediction['prediction_id']
            prediction['candidate_id'] = trial['plan']['backend_config']['run_id']
            prediction['prediction_id'] = fingerprint(dict(source_prediction=source_prediction_id,
                candidate_id=prediction['candidate_id'], plan_signature=prediction['plan_signature']))
            prediction['continuation_reuse'] = dict(prediction_id=source_prediction_id,
                same_execution=True, additional_evaluations=0)
        if not cached:
            if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                raise SearchBudgetExhausted('CPU预测预算已经用完')
            prediction = self.predict_request(trial['plan']['backend_config'], candidate_graph, parent_context)
        prediction = self.persist_prediction_graph(prediction)
        prediction['model_prediction_id'] = prediction.get('model_prediction_id', prediction['prediction_id'])
        observation = prediction_evidence(prediction)
        observation.pop('prediction_id', None)
        prediction['prediction_id'] = fingerprint(observation)
        identifier = 'cpu-evaluation:' + str(time.time_ns())
        self.store.add_evidence(prediction['prediction_id'], 'PREDICTED', prediction)
        if not cached:
            self.store.charge(identifier + ':count', 'cpu_predictions', 1)
            self.store.charge(identifier + ':seconds', 'cpu_seconds', prediction['evaluation_seconds'])
        if prediction.get('prediction_status') == 'NEEDS_PROFILE':
            self.request_profiles(trial, prediction)
        elif any(row.get('kind') in ('nccl', 'memory', 'compute_communication_overlap')
                 for row in prediction['evidence'].get('profiling_requests', [])):
            self.request_profiles(trial, prediction)
        updated = self.store.update_trial(trial['trial_id'], prediction=prediction,
            validation=dict(trial.get('validation') or {}, **prediction['validation']),
            state=trial['state'] if trial['state'] in ('MEASURED', 'FAILED', 'LOAD_VALIDATED') else 'PROJECTED')
        self.family_ledger.record_trial(updated, self.cost_revision)
        return updated

    def counterfactual_prediction(self, request, execution_graph, parent_request):
        parent_context = self.projection_parent_context(parent_request)
        prediction = self.predictor.evaluate_projection(request, execution_graph, parent_context,
            cost_mode=self.config.predictor_cost_mode, persist=False)
        prediction = self.persist_prediction_graph(prediction)
        charge_id = 'counterfactual:' + prediction['prediction_id']
        if self.store.charged(charge_id + ':count') is None:
            if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                raise SearchBudgetExhausted('反事实评价达到CPU预测预算')
            self.store.charge(charge_id + ':count', 'cpu_predictions', 1)
            self.store.charge(charge_id + ':seconds', 'cpu_seconds', prediction['evaluation_seconds'])
            self.store.add_evidence(prediction['prediction_id'], 'PREDICTED', prediction)
        return prediction

    @staticmethod
    def combine_counterfactual_intents(first, second):
        left = first.executable() if isinstance(first, AgentRewriteIntent) else first
        right = second.executable() if isinstance(second, AgentRewriteIntent) else second
        values = dict(left.values)
        conflicts = [name for name, value in right.values.items()
            if name in values and values[name] != value]
        if conflicts:
            return None, dict(status='CONSTRAINT_COUPLED', conflicts=conflicts,
                reason='两个动作对同一变量给出了不同目标')
        values.update(right.values)
        preserve = set(left.preserve) | set(right.preserve)
        changed_preserved = sorted(set(values) & preserve)
        if changed_preserved:
            return None, dict(status='CONSTRAINT_COUPLED', conflicts=changed_preserved,
                reason='一个动作修改了另一个动作要求保持的变量')
        adjustable = (set(left.adjustable) | set(right.adjustable)) - set(values) - preserve
        parent_ids = {value for value in (left.parent_plan_id, right.parent_plan_id) if value is not None}
        if len(parent_ids) > 1:
            return None, dict(status='CONSTRAINT_COUPLED', conflicts=sorted(parent_ids),
                reason='两个动作的父计划不同')
        return RewriteIntent(operation='JointTransform', values=values,
            adjustable=sorted(adjustable), preserve=sorted(preserve),
            hypothesis=left.hypothesis + '；' + right.hypothesis,
            evidence_refs=sorted(set(left.evidence_refs) | set(right.evidence_refs)),
            target_task_ids=sorted(set(left.target_task_ids) | set(right.target_task_ids)),
            target_flow_ids=sorted(set(left.target_flow_ids) | set(right.target_flow_ids)),
            interaction_evidence_ids=sorted(set(left.interaction_evidence_ids) |
                set(right.interaction_evidence_ids)), negative_effects=sorted(
                    set(left.negative_effects) | set(right.negative_effects)),
            parent_plan_id=next(iter(parent_ids), None)), None

    def counterfactual_interactions(self, trial, intents):
        parent_request = trial['plan']['backend_config']
        intents = [self.resolve_agent_intent(intent, parent_request) for intent in intents]
        immutable_fields = ('model', 'checkpoint', 'data', 'global_batch', 'sequence',
            'precision', 'optimizer', 'node_count', 'gpus_per_node')
        completion_policy = 'deterministic_minimum_constraint_changes_then_solver_order_v1'
        parent_candidate = self.rewriter.candidate(parent_request)
        parent_graph = self.rewriter.compiler.observation(parent_candidate, [], [])['training_graph']
        baseline = self.counterfactual_prediction(parent_request, parent_graph, parent_request)
        baseline_time = baseline['performance']['step_time_ms']
        baseline_cost_basis = dict(prediction_status=baseline['prediction_status'],
            profile_id=baseline['evidence'].get('profile_id'),
            cost_mode=baseline['evidence'].get('cost_mode'),
            cost_granularity=baseline['evidence'].get('cost_granularity'))
        output = []
        for left_index, right_index in combinations(range(len(intents)), 2):
            first, second = intents[left_index], intents[right_index]
            combined, coupling = self.combine_counterfactual_intents(first, second)
            if coupling is not None:
                output.append(dict(left=left_index, right=right_index, interaction_status='CONSTRAINT_COUPLED',
                    constraint_coupling=coupling, predicted_interaction_ms=None))
                continue
            plans = []
            conflicts = []
            for label, intent in (('left', first), ('right', second),
                                  ('joint', combined)):
                try:
                    rewrite = self.rewriter.complete_and_rewrite(parent_request, intent, limit=1)[0]
                except ValueError as error:
                    if not error.args or not isinstance(error.args[0], dict) or error.args[0].get(
                            'status') != 'CONSTRAINT_CONFLICT':
                        raise
                    conflicts.append(dict(plan=label, conflict=error.args[0]))
                    continue
                if any(rewrite['request'].get(key) != parent_request.get(key) for key in immutable_fields):
                    raise ValueError('反事实候选改变了固定训练语义')
                prediction = self.counterfactual_prediction(
                    rewrite['request'], rewrite['execution_graph'], parent_request)
                cost_basis = dict(prediction_status=prediction['prediction_status'],
                    profile_id=prediction['evidence'].get('profile_id'),
                    cost_mode=prediction['evidence'].get('cost_mode'),
                    cost_granularity=prediction['evidence'].get('cost_granularity'))
                plans.append(dict(plan=label, run_id=rewrite['request']['run_id'],
                    prediction_id=prediction['prediction_id'],
                    prediction_status=prediction['prediction_status'],
                    step_time_ms=prediction['performance']['step_time_ms'],
                    completion_policy=completion_policy,
                    candidate_selection='first_solution_under_fixed_solver_order',
                    intended_values=intent.values,
                    constraint_propagated_changes={key: value for key, value in
                        rewrite['applied_changes'].items() if key not in intent.values},
                    immutable_semantics={key: rewrite['request'].get(key) for key in immutable_fields},
                    cost_basis=cost_basis,
                    graph_delta=dict(config_changes=rewrite['graph_delta']['config_changes'],
                        changed_task_count=len(rewrite['graph_delta']['affected_task_ids']),
                        changed_dependency_count=len(rewrite['graph_delta']['added_edges']) +
                            len(rewrite['graph_delta']['removed_edges']),
                        changed_rank_groups=sorted(rewrite['graph_delta']['process_group_changes']),
                        changed_memory_count=len(rewrite['graph_delta']['buffer_lifetime_changes'])),
                    critical_path=prediction.get('diagnosis', {}).get('critical_tasks', [])[:12]))
            times = {row['plan']: row['step_time_ms'] for row in plans}
            cost_bases = {json.dumps(row['cost_basis'], sort_keys=True) for row in plans}
            cost_bases.add(json.dumps(baseline_cost_basis, sort_keys=True))
            if conflicts or baseline_time is None or set(times) != {'left', 'right', 'joint'} or any(
                    value is None for value in times.values()) or len(cost_bases) != 1:
                output.append(dict(left=left_index, right=right_index,
                    interaction_status='INDEPENDENT_PLAN_UNAVAILABLE', conflicts=conflicts,
                    plans=plans, cost_basis_comparable=len(cost_bases) == 1,
                    baseline_prediction_id=baseline['prediction_id'],
                    baseline_prediction_status=baseline['prediction_status'],
                    baseline_step_time_ms=baseline_time,
                    predicted_interaction_ms=None))
                continue
            interaction = times['left'] + times['right'] - times['joint'] - baseline_time
            output.append(dict(left=left_index, right=right_index, interaction_status='PREDICTED',
                baseline_prediction_id=baseline['prediction_id'], baseline_step_time_ms=baseline_time,
                plans=plans, predicted_interaction_ms=interaction,
                completion_policy=completion_policy, cost_basis_comparable=True,
                interpretation='SYNERGISTIC' if interaction > 0 else
                    'OFFSETTING' if interaction < 0 else 'ADDITIVE',
                scope='counterfactual_complete_plan_prediction'))
        return dict(source='complete_and_rewrite_plus_full_dag_predictor', pairs=output,
            formula='T(left)+T(right)-T(joint)-T(parent)')

    def reference_trial(self):
        return self.store.best_measured() or next(iter(self.store.trials()), None)

    def check_walltime(self):
        self.store.record_session_time(self.session_id, time.perf_counter() - self.started)
        if self.store.search_seconds() >= self.config.budget.max_walltime_seconds:
            raise SearchBudgetExhausted('搜索累计运行时间达到预算')

    def round_call_cost_report(self, checkpoint, best):
        start = checkpoint.get('usage_at_start', {})
        usage = self.store.usage()
        metrics = self.store.agent_context_metrics()[checkpoint.get('context_metric_start', 0):]
        delta = lambda name: usage.get(name, 0) - start.get(name, 0)
        return dict(llm_calls=delta('agent_calls'), input_tokens=delta('agent_input_tokens'),
            output_tokens=delta('agent_output_tokens'),
            schema_tokens=sum(row.get('strict_schema_tokens', 0) for row in metrics),
            tool_result_tokens=sum(row.get('current_tool_return_tokens', 0) for row in metrics),
            by_role=self.agent_role_costs(metrics),
            cpu_prediction_count=delta('cpu_predictions'), gpu_hours=delta('gpu_hours'),
            cpu_seconds=delta('cpu_seconds'), profiling_gpu_hours=delta('profiling_gpu_hours'),
            llm_seconds=delta('llm_seconds'),
            best_real_tokens_per_s=(best['measurement']['tokens_per_second']
                if best is not None and best.get('measurement') is not None
                and best['measurement'].get('performance_eligible', False) else None))

    @staticmethod
    def agent_role_costs(metrics):
        output = {}
        for role, name in AGENT_ROLE_NAMES.items():
            rows = [row for row in metrics if row.get('role') == role]
            output[role] = dict(role_name=name,
                provider_calls=sum(row.get('provider_attempt_count',
                    int('provider' in row and not row.get('replayed_verified_response', False)))
                    for row in rows),
                cached_response_count=sum(row.get('replayed_verified_response', False) for row in rows),
                completed_responses=sum('provider' in row for row in rows),
                input_tokens=sum(row.get('actual_input_tokens', 0) for row in rows),
                output_tokens=sum(row.get('actual_output_tokens', 0) for row in rows),
                schema_tokens=sum(row.get('strict_schema_tokens', 0) for row in rows),
                tool_result_tokens=sum(row.get('current_tool_return_tokens', 0) for row in rows))
        return output

    def complete_round(self, round_index, best, **fields):
        checkpoint = self.store.get_round(round_index)
        checkpoint = self.store.save_round(round_index, stage='COMPLETE',
            call_cost_report=self.round_call_cost_report(checkpoint, best), **fields)
        if self.config.single_decision_round:
            self.write_round_record(checkpoint)
        return checkpoint

    def evidence_interface_fault(self, checkpoint):
        if (not checkpoint or checkpoint.get('measured_candidate_ids')
                or not checkpoint.get('single_wire_evidence_id')):
            return []
        visible = set(checkpoint.get('single_reference_aliases', {}).values())
        rows = self.store.connection.execute(
            "SELECT document FROM evidence WHERE kind='FAILURE' "
            "AND json_extract(document, '$.model_evidence_id')=?",
            (checkpoint['single_wire_evidence_id'],))
        missing = {identifier for (document,) in rows
            for row in json.loads(document).get('rejected_wire_rows', [])
            if row.get('status') == 'INVALID_EVIDENCE_REFERENCE'
            for identifier in row.get('unknown_refs', []) if identifier in visible}
        return sorted(identifier for identifier in missing if self.store.connection.execute(
            'SELECT 1 FROM evidence WHERE id=?', (identifier,)).fetchone())

    def write_round_record(self, checkpoint):
        wire_id = checkpoint.get('single_wire_evidence_id')
        wire = (self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
            (wire_id,)).fetchone() if wire_id else None)
        measured = set(checkpoint.get('measured_candidate_ids', []))
        candidates = set(checkpoint.get('candidate_ids', [])) | measured
        document = dict(round=checkpoint['round_index'] + 1, variant=self.config.agent_ablation,
            model=self.config.agent_model, effort=self.config.agent_reasoning_effort,
            model_evidence_id=wire_id, proposals=json.loads(wire[0]) if wire else None,
            selection=checkpoint.get('selection'), constraints=checkpoint.get('constraint_decision'),
            **({'experiment_options': checkpoint.get('budgeted_experiment_options', []),
                'experiment_exclusions': checkpoint.get('budgeted_experiment_exclusions', []),
                'experiment_selection_policy': checkpoint.get('budgeted_experiment_policy', 'v1')}
                if self.config.budgeted_investigation_enabled else {}),
            execution_revisions=checkpoint.get('observation', {}).get('execution_revisions', []),
            candidates=[dict(candidate_id=trial['trial_id'], strategy=trial['plan']['backend_config'],
                measured_this_round=trial['trial_id'] in measured, measurement=trial.get('measurement'),
                failure=trial.get('failure')) for trial in self.store.trials() if trial['trial_id'] in candidates],
            actual_cost=checkpoint['call_cost_report'], stop_reason=checkpoint.get('stop_reason'))
        if self.config.budgeted_investigation_enabled:
            by_id = {row['candidate_id']: row for row in document['experiment_options']}
            for row in document['candidates']:
                option = by_id.get(row['candidate_id'])
                result = row['measurement'] or {}
                if not option or not row['measured_this_round'] or not result.get('performance_eligible'):
                    continue
                interval = option['throughput_interval']
                estimate = option['predicted_step_ms']
                row['prediction_validation'] = dict(
                    measured_step_ms=result['step_seconds'] * 1000,
                    measured_throughput=result['tokens_per_second'],
                    residual_ms=None if estimate is None else result['step_seconds'] * 1000 - estimate,
                    predicted_gain=option['predicted_gain'],
                    inside_interval=None if interval is None else interval[0] <= result['tokens_per_second'] <= interval[1])
        if checkpoint.get('interface_fault_evidence_ids'):
            document['interface_fault_evidence_ids'] = checkpoint['interface_fault_evidence_ids']
            document['counts_as_strategy_round'] = False
        directory = self.output / 'round-records'
        directory.mkdir(exist_ok=True)
        path = directory / f'round-{document["round"]:04d}.json'
        if path.exists():
            if json.loads(path.read_text()) != document:
                raise ValueError('已保存的逐轮策略记录内容发生变化')
        else:
            path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(event='AGENT_ROUND_COMPLETED', round=document['round'],
            variant=document['variant'], actual_cost=document['actual_cost'], record=str(path)),
            ensure_ascii=False, allow_nan=False), flush=True)

    def visible_observation(self, value):
        mode = self.config.evaluation_mode
        hidden = set()
        if mode == 'config_only':
            hidden |= {'model_graph', 'measured_cost_profile', 'graph_delta', 'critical_path_graph',
                'execution_graph', 'critical_tasks', 'critical_regions', 'critical_path_tasks',
                'affected_tasks', 'target_task_ids', 'target_flow_ids', 'affected_task_ids',
                'target_impact', 'changed_target_ids', 'effect_preview', 'rewrite', 'detected_topology'}
        if mode in ('config_only', 'graph_aware'):
            hidden |= {'impact_index', 'hard_constraint_dependencies', 'active_decisions',
                'completed_decisions', 'effect_preview'}
        if mode in ('config_only', 'graph_aware', 'graph_impact'):
            hidden |= {'transformation_interactions', 'interaction_evidence_id', 'interaction_evidence_ids',
                'previews', 'edges', 'measured_interactions'}
        def project(item):
            if isinstance(item, dict):
                return {key: project(entry) for key, entry in item.items() if key not in hidden}
            if isinstance(item, list):
                return [project(entry) for entry in item]
            return item
        scoped = AblationEvidencePolicy.current_trial_ids(value) if isinstance(value, dict) else set()
        current = scoped or getattr(self, '_visible_trial_ids', set())
        historical = ({row['trial_id'] for row in self.store.trials()}
            if self.config.agent_ablation == 'without_search_memory' else set())
        return AblationEvidencePolicy(self.config.agent_ablation, current, historical).observation(project(value))

    def backend_dag(self, trial):
        prediction = trial.get('prediction') or {}
        projected = self.load_decision_graph(prediction) if prediction else None
        projected_tasks = {} if projected is None else projected.get('tasks', {})
        cache_key = (trial['plan_hash'], self.profile_id, self.cost_revision, prediction.get('prediction_id'))
        if cache_key in self.dag_cache:
            return self.dag_cache[cache_key]
        candidate = self.rewriter.candidate(trial['plan']['backend_config'])
        graph = self.rewriter.compiler.observation(candidate, [], [])['training_graph']
        regions = {row['region_id']: row for row in graph['regions']}
        exact_profile = (self.profile is not None and
            plan_signature(self.profile['request']) == plan_signature(trial['plan']['backend_config']))
        task_profile = self.profile if exact_profile else None
        if task_profile is None:
            selected_profile = self.predictor.database.select(trial['plan']['backend_config'])
            if selected_profile is not None:
                selected_document = self.predictor.database.get(selected_profile, trial['plan']['backend_config'])
                if (plan_signature(selected_document['request']) == plan_signature(trial['plan']['backend_config'])
                        and profile_hardware_matches(selected_document, self.topology.devices)):
                    task_profile = selected_document
        exact_profile = task_profile is not None
        tasks = {}
        flows = set()
        for task_id, task in graph['tasks'].items():
            record = dict(task)
            operation = task['operation']
            physical_rank = task['metadata']['physical_rank']
            profile_rank = task['metadata'].get('logical_rank', task['rank'])
            cost = None
            evidence_refs = []
            if exact_profile and operation in ('forward_compute', 'backward_compute'):
                vp_rank = regions[task['region_id']]['vp_rank']
                region = regions[task['region_id']]
                identifiers = list(range(region['layer_start'], region['layer_end']))
                measured_cost = self.predictor._region_cost(task_profile, profile_rank,
                    vp_rank, operation, identifiers)
                cost = dict(measured_cost, duration_ms=measured_cost['duration'])
            elif exact_profile and operation == 'dp_reduce':
                value = task_profile.get('gradient_finalize', {}).get(str(profile_rank))
                cost = None if value is None else dict(duration_ms=value)
            elif exact_profile and operation == 'optimizer_step':
                value = task_profile.get('optimizer', {}).get(str(profile_rank))
                cost = None if value is None else dict(duration_ms=value)
            elif exact_profile and operation in ('pp_p2p_send', 'pp_p2p_recv'):
                group = task['metadata'].get('physical_group_ranks', [])
                if len(group) == 2:
                    direction = task['metadata']['direction']
                    source, destination = group if direction == 'forward' else list(reversed(group))
                    value = task_profile.get('links', {}).get(f'{source}:{destination}:{direction}')
                    cost = None if value is None else dict(duration_ms=value)
            if cost is not None:
                evidence_refs.append(task_profile['id'])
            flow_id = None
            if 'channel' in task['metadata']:
                flow_id = f'flow:{task["metadata"]["channel"]}:mb:{task["microbatch"]}'
            elif operation == 'dp_reduce':
                flow_id = f'flow:gradient_sync:rank:{task["rank"]}'
            if flow_id is not None:
                flows.add(flow_id)
            cost_status = 'MEASURED' if cost is not None else 'UNRESOLVED'
            projected_task = projected_tasks.get(task_id)
            if (cost is None and projected_task is not None
                    and projected_task.get('duration_ms') is not None
                    and projected_task['rank'] == physical_rank):
                cost = dict(duration_ms=projected_task['duration_ms'], prediction_id=prediction['prediction_id'])
                cost_status = 'PREDICTED'
                evidence_refs.append(prediction['prediction_id'])
            record.update(cost=cost, cost_status=cost_status,
                evidence_refs=evidence_refs, flow_id=flow_id)
            tasks[task_id] = record
        output = dict(graph_sha256=graph['graph_sha256'], representation=graph['representation'],
            config=graph['config'], regions=graph['regions'], tasks=tasks,
            task_count=len(tasks), edge_count=sum(len(task['dependencies']) for task in tasks.values()),
            flow_count=len(flows), opaque_operations=graph['opaque_operations'],
            semantic_sources=graph['semantic_sources'], data_gaps=graph['data_gaps'],
            structural_evidence='deterministic_megatron_schedule_compiler',
            cost_evidence_refs=[task_profile['id']] if exact_profile else [],
            baseline_evidence_ref=(trial['measurement'] or {}).get('run_id'))
        graph_evidence = dict(kind='structural_executable_dag', graph_sha256=graph['graph_sha256'],
                plan_hash=trial['plan_hash'], task_count=output['task_count'],
                edge_count=output['edge_count'], flow_count=output['flow_count'],
                representation=output['representation'], semantic_sources=output['semantic_sources'],
                cost_evidence_refs=output['cost_evidence_refs'],
                storage='complete_graph_retained_in_backend_dag_cache')
        output['graph_evidence_id'] = fingerprint(graph_evidence)
        self.store.add_evidence(output['graph_evidence_id'], 'PREDICTED', graph_evidence)
        self.dag_cache[cache_key] = output
        return output

    def native_execution_view(self, trial, region_ids=()):
        document = execution_view(trial['plan']['backend_config'], self.backend_dag(trial),
            self.topology.devices, self.predictor._native_stage_sequence,
            self.config.execution_evidence_format, region_ids)
        identifier = fingerprint(dict(trial_id=trial['trial_id'], execution_view=document))
        self.store.add_evidence(identifier, 'PREDICTED', dict(
            kind='native_execution_view', trial_id=trial['trial_id'], execution_view=document))
        return dict(evidence_id=identifier, trial_id=trial['trial_id'], **document)

    def dag_index(self, trial):
        graph = self.backend_dag(trial)
        operation_counts = Counter(task['operation'] for task in graph['tasks'].values())
        rank_counts = Counter(task['rank'] for task in graph['tasks'].values())
        cost_counts = Counter(task['cost_status'] for task in graph['tasks'].values())
        samples = {}
        for operation in sorted(operation_counts):
            samples[operation] = [task_id for task_id, task in graph['tasks'].items()
                if task['operation'] == operation][:2]
        return dict(graph_sha256=graph['graph_sha256'], evidence_id=graph['graph_evidence_id'],
            task_count=graph['task_count'],
            edge_count=graph['edge_count'], flow_count=graph['flow_count'],
            operation_counts=dict(operation_counts), rank_task_counts=dict(sorted(rank_counts.items())),
            cost_status_counts=dict(cost_counts), region_count=len(graph['regions']),
            regions=[{key: row[key] for key in ('region_id', 'pp_rank', 'vp_rank', 'layer_start', 'layer_end')}
                for row in graph['regions']], stable_task_id_samples=samples,
            query_contract='完整DAG保存在后端；使用inspect_dag_subgraph、inspect_task_dependencies、'
                'query_affected_decisions和preview_transformation按页读取')

    def compact_plan(self, trial):
        plan = trial['plan']
        request = plan['backend_config']
        schedule = native_schedule_settings(request)
        workload_fields = ('model', 'sequence', 'global_batch', 'micro_batch', 'precision',
            'precision_aware_optimizer', 'recompute',
            'pipeline_schedule', 'optimizer_sharding', 'execution_backend',
            'group', 'overlap_p2p', 'batch_p2p_comm', 'overlap_p2p_comm_warmup_flush', 'bucket_elements')
        return dict(plan_id=plan['plan_id'], plan_hash=trial['plan_hash'],
            parallelism=plan['parallelism'], expert_parallelism=plan['expert_parallelism'],
            region_partition=plan['region_partition'], rank_mapping=plan['rank_mapping'],
            schedule=plan['schedule'], num_microbatches=plan['num_microbatches'],
            memory_policy=plan['memory_policy'], state_sharding=plan.get('state_sharding', {}),
            optimizer_configuration=plan.get('optimizer_configuration', {}),
            checkpoint_policy=plan.get('checkpoint_policy', {}),
            workload={key: (schedule[key] if key in
                ('group', 'overlap_p2p', 'batch_p2p_comm', 'overlap_p2p_comm_warmup_flush')
                else request.get(key)) for key in workload_fields},
            process_group_sizes={key: sorted({len(group) for group in groups})
                for key, groups in plan['process_groups']['logical'].items()})

    @staticmethod
    def global_family_values(request):
        return global_values(request)

    @classmethod
    def global_family_id(cls, request):
        return global_identity(request)

    def global_family_catalog(self, trial, representatives=()):
        return self.family_catalog(trial, representatives)

    @staticmethod
    def apply_family_to_intent(intent, parent_request, families):
        executable = intent.executable()
        if intent.family_id is None or intent.proposal_kind == 'complete_strategy':
            EvidenceGuidedSearch.validate_bound_batch(executable, parent_request)
            return executable
        family = next((row for row in families if row['family_id'] == intent.family_id), None)
        if family is None:
            raise AgentConstraintConflict(dict(conflicts=['family_id'],
                reason='选择的全局家族不在当前合法家族目录中',
                selected_family_id=intent.family_id, family_values=None, preserve_values={}))
        family_fields = set(family['values'])
        preserved_changes = sorted(name for name in set(executable.preserve) & family_fields
            if family['values'][name] != (parent_request[name] if name in parent_request
                else normalized_execution(parent_request)[name]))
        if preserved_changes:
            raise AgentConstraintConflict(dict(conflicts=preserved_changes,
                reason='所选家族要求修改这些字段，同时当前意图要求保持父计划中的取值',
                selected_family_id=intent.family_id,
                family_values={name: family['values'][name] for name in preserved_changes},
                preserve_values={name: parent_request[name] if name in parent_request
                    else normalized_execution(parent_request)[name] for name in preserved_changes},
                revision_owner='JointStrategyAgent'))
        supplied_family_fields = family_fields.intersection(executable.values)
        mismatched = sorted(name for name in supplied_family_fields
            if executable.values[name] != family['values'][name])
        if mismatched:
            raise AgentConstraintConflict(dict(conflicts=mismatched,
                reason='显式请求值与所选家族的强制取值不同', selected_family_id=intent.family_id,
                family_values={name: family['values'][name] for name in mismatched},
                requested_values={name: executable.values[name] for name in mismatched},
                preserve_values={}, revision_owner='JointStrategyAgent'))
        values = dict(family['values'])
        values.update(executable.values)
        resolved = executable.model_copy(update=dict(values=values,
            adjustable=sorted(set(executable.adjustable) - family_fields),
            preserve=sorted(set(executable.preserve) - family_fields)))
        EvidenceGuidedSearch.validate_bound_batch(resolved, parent_request, intent.family_id)
        return resolved

    @staticmethod
    def validate_bound_batch(intent, parent_request, family_id=None):
        bound = set(intent.values) | set(intent.preserve)
        effective = dict(parent_request, **intent.values)
        if not {'dp', 'micro_batch'} <= bound or not all(
                name in effective for name in ('global_batch', 'dp', 'micro_batch')):
            return
        divisor = effective['dp'] * effective['micro_batch']
        if effective['global_batch'] % divisor:
            raise AgentConstraintConflict(dict(conflicts=['dp', 'micro_batch'],
                reason='固定GBS不能被DP与MBS的乘积整除，当前意图没有合法的整数Microbatch数量',
                relation='GBS % (DP * MBS) == 0',
                effective_values={name: effective[name] for name in ('global_batch', 'dp', 'micro_batch')},
                numerator=effective['global_batch'], denominator=divisor,
                remainder=effective['global_batch'] % divisor,
                selected_family_id=family_id,
                family_values={name: intent.values[name] for name in ('dp', 'micro_batch') if name in intent.values},
                preserve_values={name: parent_request[name] for name in ('dp', 'micro_batch')
                    if name in intent.preserve},
                revision_direction='保持GBS和Sequence，联合选择合法DP与其他并行维度，或调整MBS并重新验证',
                revision_owner='JointStrategyAgent'))

    def canonical_dense_scope(self, parent_request, values, ranges):
        if self.model.num_experts is not None or not {'tp', 'pp', 'ep', 'etp', 'expert_dp'} <= values.keys():
            return values
        if 'execution_backend' in values:
            backend = values['execution_backend']
        elif 'execution_backend' in ranges:
            choices = ranges['execution_backend']
            if len(choices) != 1:
                return values
            backend = choices[0]
        else:
            backend = parent_request.get('execution_backend', 'megatron_ddp')
        if backend not in ('megatron_ddp', 'megatron_fsdp'):
            return values
        etp = values['tp'] if backend == 'megatron_ddp' else 1
        pp = values['pp']
        world = parent_request['node_count'] * parent_request['gpus_per_node']
        if world % (etp * pp):
            return values
        return dict(values, ep=1, etp=etp, expert_dp=world // (etp * pp))

    def resolve_agent_intent(self, intent, parent_request):
        if intent.family_id is not None and intent.proposal_kind == 'conditional_intent':
            executable = intent.executable()
            values = executable.values
            canonical = self.canonical_dense_scope(parent_request, values, executable.allowed_ranges)
            if canonical != values and intent.family_id == global_identity(dict(parent_request, **values)):
                intent = intent.model_copy(update=dict(values=RewriteValues.model_validate(canonical),
                    family_id=global_identity(dict(parent_request, **canonical))))
        if intent.family_id is None:
            resolved = self.apply_family_to_intent(intent, parent_request, [])
        else:
            families = [*self.joint_global_families(parent_request),
                *SeedPlanner(self.rewriter, self.profile).global_families(parent_request)]
            resolved = self.apply_family_to_intent(intent, parent_request, families)
        return self.bind_execution_targets(resolved, parent_request)

    def bind_execution_targets(self, intent, parent_request):
        if not intent.target_task_ids:
            return intent
        structural = self.rewriter.compiler.graph(self.rewriter.candidate(parent_request))
        unresolved = set(intent.target_task_ids) - structural.tasks.keys()
        if not unresolved:
            return intent
        parent = next((trial for trial in self.store.trials()
            if same_training_plan(trial['plan']['backend_config'], parent_request)
            and trial.get('prediction') is not None), None)
        prediction = None if parent is None else parent['prediction']
        graph = None if prediction is None else self.load_decision_graph(prediction)
        tasks = {} if graph is None else graph.get('tasks', {})
        bindings = {}
        for identifier in sorted(unresolved):
            record = tasks.get(identifier)
            if record is None:
                raise AgentConstraintConflict(dict(conflicts=['target_task_ids'],
                    reason='执行目标未出现在同一父配置的实际执行图中',
                    task_id=identifier, parent_plan_id=intent.parent_plan_id,
                    revision_owner='JointStrategyAgent'))
            origin = record.get('metadata', {}).get('origin_task')
            if origin in structural.tasks:
                task = structural.tasks[origin]
                if (record.get('region', record.get('region_id')) != task.region_id
                        or record.get('microbatch', record.get('microbatch_id')) != task.microbatch_id):
                    raise ValueError('执行任务与原始结构任务的模块区间或微批身份不同')
            bindings[identifier] = dict(origin_task=origin,
                structural_origin_available=origin in structural.tasks)
        evidence = dict(kind='execution_task_reference_binding',
            parent_trial_id=parent['trial_id'], prediction_id=prediction['prediction_id'],
            original_target_ids=intent.target_task_ids, bindings=bindings,
            source='actual_parent_prediction_and_projection_task_metadata',
            scope='native_legality_with_unchanged_execution_targets')
        identifier = fingerprint(evidence)
        self.store.add_evidence(identifier, 'PREDICTED', evidence)
        self.rewriter.register_execution_reference(parent_request, graph, identifier)
        return intent.model_copy(update=dict(
            evidence_refs=list(dict.fromkeys([*intent.evidence_refs, identifier]))))

    def validate_intent_evidence(self, intent):
        rows = {identifier: json.loads(encoded) for identifier, encoded in
            self.store.connection.execute('SELECT id, document FROM evidence')}
        refs = set(intent.evidence_refs) | set(intent.interaction_evidence_ids)
        refs.update(reference for row in intent.module_relations for reference in row.evidence_refs)
        if intent.target_bottleneck is not None:
            refs.update(intent.target_bottleneck.evidence_refs)
        unknown = sorted(refs - set(rows))
        incorrect = sorted(reference for reference in intent.interaction_evidence_ids
            if reference in rows and not self.is_interaction_evidence(rows[reference]))
        if unknown or incorrect:
            raise AgentConstraintConflict(dict(conflicts=['evidence_refs'],
                reason='意图引用缺少真实证据，或把其他类型证据声明为变换交互证据',
                selected_family_id=intent.family_id, family_values={}, preserve_values={},
                unknown_evidence_refs=unknown, incorrect_interaction_evidence_refs=incorrect,
                available_interaction_evidence=[dict(evidence_id=identifier,
                    tool=row['tool']['tool']) for identifier, row in rows.items()
                    if self.is_interaction_evidence(row)][-8:],
                revision_owner='JointStrategyAgent'))

    @staticmethod
    def strategy_effect_identity(intent):
        values = intent.executable().model_dump(mode='json')['values']
        required = {'tp', 'pp', 'cp', 'dp', 'ep', 'etp'}
        if (intent.proposal_kind == 'complete_strategy'
                and values.get('execution_backend') == 'megatron_ddp' and required <= set(values)):
            world = values['tp'] * values['pp'] * values['cp'] * values['dp']
            expert_dp = world // (values['etp'] * values['ep'] * values['pp'])
            for field, value in dict(dp_inner_size=values['dp'] * values['cp'], dp_outer_size=1,
                    expert_dp_inner_size=expert_dp, expert_dp_outer_size=1).items():
                values.setdefault(field, value)
        return fingerprint(dict(parent_plan_id=intent.parent_plan_id,
            family_id=None if intent.proposal_kind == 'complete_strategy' else intent.family_id,
            operation=intent.operation, values=values))

    @staticmethod
    def preview_attests_strategy(intent, receipt, evidence):
        body = receipt.get('result', {})
        if body.get('status') == 'PAGE_TOO_LARGE':
            complete = evidence[body['complete_result_id']]
            if complete['tool'] != receipt['tool']:
                raise ValueError('分页预览与完整后端收据的查询身份不同')
            receipt = complete
            body = receipt['result']
        tool = receipt.get('tool', {})
        submitted = tool.get('transformation_intent')
        if (tool.get('tool') != 'preview_transformation' or submitted is None
                or not isinstance(body.get('strategy_effect_identity'), str)):
            return False
        original = AgentRewriteIntent.model_validate(submitted)
        return EvidenceGuidedSearch.strategy_effect_identity(original) == EvidenceGuidedSearch.strategy_effect_identity(intent)

    @staticmethod
    def is_interaction_evidence(receipt):
        return (receipt.get('tool', {}).get('tool') in (
                'preview_transformation', 'analyze_transformation_interactions')
            and receipt.get('result', {}).get('status') != 'CONSTRAINT_CONFLICT')

    def strategy_validation_feedback(self, response):
        issues = []
        if len(response.intents) != self.config.intent_count:
            issues.append(dict(conflicts=['intent_count'],
                reason='生成的意图数量与当前任务要求不同',
                expected=self.config.intent_count, actual=len(response.intents)))
        known = {row[0] for row in self.store.connection.execute('SELECT id FROM evidence')}
        unknown = sorted(set(response.evidence_refs) - known)
        if unknown or not response.evidence_refs:
            issues.append(dict(conflicts=['evidence_refs'], reason='批次需要引用已登记的真实证据',
                unknown_evidence_refs=unknown))
        reference = self.reference_trial()
        for index, intent in enumerate(response.intents):
            try:
                self.validate_intent_evidence(intent)
            except AgentConstraintConflict as error:
                issues.append(dict(error.result, intent_index=index))
            if reference is not None:
                try:
                    self.resolve_agent_intent(intent, reference['plan']['backend_config'])
                except AgentConstraintConflict as error:
                    issues.append(dict(error.result, intent_index=index))
                except ValidationError as error:
                    issues.append(dict(conflicts=['joint_action_targets'], intent_index=index,
                        reason='联合动作包含互相冲突的变量目标',
                        validation_errors=[dict(type=row['type'], location=list(row['loc']),
                            message=row['msg']) for row in error.errors()]))
        if not issues:
            return None
        result = dict(status='REVISION_REQUIRED', revision_owner='JointStrategyAgent',
            reason='确定性校验发现可修订的生成合同或策略约束问题', issues=issues,
            instruction='依据逐项原因修订意图，保留合法方向；原始证据身份、GBS和Sequence保持有效。')
        tool = dict(tool='validate_agent_strategy', role='JointStrategyAgent')
        identifier = fingerprint(dict(tool=tool, response=response.model_dump(mode='json'), result=result))
        receipt = dict(evidence_id=identifier, tool=tool, result=result,
            result_tokens=token_count(result), token_encoding=TOKEN_ENCODING_NAME,
            evidence_scope='deterministic_agent_output_validation')
        self.store.add_evidence(identifier, 'FAILURE', receipt)
        return receipt

    def prepare_global_family_candidates(self, parent):
        return self.prepare_joint_candidates(parent)

    def topology_summary(self):
        nodes = []
        for node_index in range(self.topology.num_nodes):
            devices = sorted((row for row in self.topology.devices if row['node_index'] == node_index),
                key=lambda row: row['physical_rank'])
            nodes.append(dict(node_index=node_index, node=devices[0]['node'],
                pod_resources=self.execution_resources[node_index],
                physical_ranks=[row['physical_rank'] for row in devices],
                local_gpu_indices=[row['index'] for row in devices],
                pci_bus_ids=[row.get('pci_bus_id') for row in devices],
                numa_nodes=[row.get('numa_node') for row in devices],
                gpu_uuid_fingerprint=fingerprint([gpu_uuid(row) for row in devices])))
        network = []
        for node_index in range(self.topology.num_nodes):
            adapters = [row for row in self.hardware_identity['network'] if row['node_index'] == node_index]
            network.append(dict(node_index=node_index, adapter_count=len(adapters),
                adapter_names=[row.get('name', row.get('device')) for row in adapters[:8]],
                adapter_name_count=len({row.get('name', row.get('device')) for row in adapters}),
                numa_nodes=sorted({row.get('numa_node') for row in adapters
                    if row.get('numa_node') is not None}),
                topology_fingerprint=fingerprint(adapters)))
        return dict(num_gpus=self.topology.num_gpus, num_nodes=self.topology.num_nodes,
            gpus_per_node=self.topology.gpus_per_node, memory_per_gpu_gib=self.topology.memory_per_gpu_gb,
            gpu_models=sorted({row.get('name', row.get('device_name')) for row in self.topology.devices}),
            nodes=nodes,
            network=network, software=self.hardware_identity['software'])

    def performance_summary(self, trial):
        measurement = trial['measurement']
        baseline = None if measurement is None else {key: measurement.get(key) for key in (
            'run_id', 'status', 'performance_eligible', 'step_seconds', 'tokens_per_second',
            'peak_allocated_gib', 'numerical_equivalence_certified', 'numerical_equivalence_scope')}
        if baseline is not None and trial.get('failure') is not None:
            baseline['memory_failure_witness'] = self.failure_summary(trial)
        if self.profile is None:
            return dict(baseline=baseline, cost_profile=dict(status='NEEDS_PROFILE'))
        request = trial['plan']['backend_config']
        lanes = request['tp'] * request['cp'] * request['dp']
        profile_request = self.profile['request']
        profile_config = configuration(profile_request)
        profile_regions = ranges(profile_request['decoder_chunks'])
        profile_lanes = profile_config.world_size // profile_config.pp
        ranks = []
        for rank in range(self.topology.num_gpus):
            prefix = str(rank) + ':'
            rows = [row for key, row in self.profile['regions'].items() if key.startswith(prefix)]
            memory = self.profile['memory'].get(str(rank), {})
            pp_rank = rank // profile_lanes

            def measured_total(kind):
                field = 'forward_ms' if kind == 'forward_compute' else 'backward_ms'
                total = 0.0
                deviations = []
                components = {}
                for vp in range(profile_config.vpp):
                    region = vp * profile_config.pp + pp_rank
                    row = self.profile['regions'][f'{rank}:{vp}:{kind}']
                    total += sum(self.profile['layers'][str(layer)][field]
                                 for layer in profile_regions[region]) + row['overhead_ms']
                    deviations.append(row['residual_ms'])
                    phase = 'forward' if kind == 'forward_compute' else 'backward'
                    breakdown = self.profile.get('fixed_cost_breakdown', {}).get(
                        f'{rank}:{vp}:{phase}', {}).get('components', {})
                    for name, value in breakdown.items():
                        components[name] = components.get(name, 0.0) + value
                return total, math.sqrt(sum(value * value for value in deviations)), components

            forward, forward_stddev, forward_fixed = measured_total('forward_compute')
            backward, backward_stddev, backward_fixed = measured_total('backward_compute')
            ranks.append(dict(rank=rank, stage=pp_rank,
                forward_region_ms=forward, backward_region_ms=backward,
                forward_sampling_stddev_ms=forward_stddev,
                backward_sampling_stddev_ms=backward_stddev,
                forward_fixed_costs_ms=forward_fixed, backward_fixed_costs_ms=backward_fixed,
                measured_region_entries=len(rows), peak_bytes=memory.get('peak'),
                persistent_bytes=memory.get('persistent'), capacity_bytes=memory.get('capacity')))
        stages = []
        for stage in range(profile_config.pp):
            members = [row for row in ranks if row['stage'] == stage]
            stages.append(dict(stage=stage, ranks=[row['rank'] for row in members],
                forward_region_ms_min=min(row['forward_region_ms'] for row in members),
                forward_region_ms_max=max(row['forward_region_ms'] for row in members),
                backward_region_ms_min=min(row['backward_region_ms'] for row in members),
                backward_region_ms_max=max(row['backward_region_ms'] for row in members),
                peak_bytes_max=max(row['peak_bytes'] for row in members)))
        matches = same_training_plan(request, profile_request)
        result = dict(baseline=baseline, cost_profile=dict(profile_id=self.profile_id,
            source='MEASURED', region_cost_count=len(self.profile['regions']),
            layer_cost_count=len(self.profile['layers']), link_cost_count=len(self.profile.get('links', {})),
            target_plan_matches_profile=matches,
            measured_configuration={name: profile_request.get(name) for name in
                ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'micro_batch', 'decoder_chunks',
                 'recompute', 'execution_backend', 'precision')}),
            stage_summary=stages if matches else [], rank_summary=ranks if matches else [])
        if not matches:
            result.update(source_profile_stage_summary=stages, source_profile_rank_summary=ranks,
                transfer_scope='source_configuration_measurements_require_projection_for_current_layout')
        return result

    def execution_decision_report(self, trial):
        self.synchronize_cost_revision()
        cache = getattr(self, '_execution_decision_reports', None)
        if cache is None:
            self._execution_decision_reports = cache = {}
        key = (trial['plan_hash'], self.cost_revision,
            (trial.get('measurement') or {}).get('run_id'),
            (trial.get('prediction') or {}).get('prediction_id'))
        if key in cache:
            return cache[key]
        viewed = trial
        if trial.get('prediction') is None and self.store.usage().get('cpu_predictions', 0) < self.config.budget.max_cpu_predictions:
            prediction = self.predict_request(trial['plan']['backend_config'],
                parent_request=self.projection_parent_context(trial['plan']['backend_config']))
            evaluation_seconds = prediction['evaluation_seconds']
            prediction = self.persist_prediction_graph(prediction)
            registered = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                (prediction['prediction_id'],)).fetchone()
            if registered is None:
                self.store.add_evidence(prediction['prediction_id'], 'PREDICTED', prediction)
            else:
                original = json.loads(registered[0])
                assert all(original.get(name) == prediction.get(name) for name in
                    ('plan_signature', 'cost_database_revision', 'prediction_status'))
                prediction = original
            charge_id = 'decision-report:' + str(time.time_ns())
            self.store.charge(charge_id, 'cpu_predictions', 1)
            self.store.charge(charge_id + ':seconds',
                'cpu_seconds', evaluation_seconds)
            viewed = dict(trial, prediction=prediction)
        prediction = viewed.get('prediction') or {}
        graph = self.load_decision_graph(prediction) if prediction else None
        candidate = self.rewriter.candidate(trial['plan']['backend_config'])
        report = execution_analysis(viewed, self.backend_dag(viewed), graph,
            self.rewriter.compiler.decision_impact_index(candidate),
            configuration(trial['plan']['backend_config']),
            self.predictor.projector.parameters.get('resource_replay', {}))
        identifier = fingerprint(report)
        self.store.add_evidence(identifier, 'PREDICTED', dict(kind='execution_decision_report', report=report))
        report = dict(report, evidence_id=identifier)
        if report['bottlenecks']:
            self.store.charge('decision-report-impact:' + identifier,
                'counterfactual_replays', report['replay_count'])
        cache[key] = report
        return report

    def revision_prediction(self, trial):
        self.synchronize_cost_revision()
        key = (trial['plan_hash'], self.cost_revision, PROJECTION_REVISION,
            (trial.get('measurement') or {}).get('run_id'))
        if not hasattr(self, '_revision_predictions'):
            self._revision_predictions = {}
        if key in self._revision_predictions:
            return self._revision_predictions[key]
        prediction = trial.get('prediction')
        if (prediction is None or prediction.get('cost_database_revision') != self.cost_revision
                or prediction.get('evidence', {}).get('predictor_revision') != PROJECTION_REVISION):
            if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                return None
            self.check_walltime()
            prediction = self.predict_request(trial['plan']['backend_config'],
                parent_request=self.projection_parent_context(trial['plan']['backend_config']))
            seconds = prediction['evaluation_seconds']
            prediction = self.persist_prediction_graph(prediction)
            document = prediction_evidence(prediction)
            document.pop('prediction_id', None)
            prediction['prediction_id'] = fingerprint(document)
            self.store.add_evidence(prediction['prediction_id'], 'PREDICTED', prediction)
            charge = 'revision-evaluation:' + str(time.time_ns())
            self.store.charge(charge, 'cpu_predictions', 1)
            self.store.charge(charge + ':seconds', 'cpu_seconds', seconds)
        graph = self.load_prediction_graph(prediction)
        viewed = dict(prediction, execution_graph=graph)
        self._revision_predictions[key] = viewed
        return viewed

    def execution_revision_reports(self, parent, candidates):
        if self.config.agent_ablation == 'without_dag_evidence':
            return []
        before = self.revision_prediction(parent)
        rows = []
        for trial in candidates[:3]:
            after = self.revision_prediction(trial)
            comparison = compare_execution_reports(parent['plan']['backend_config'], before,
                trial['plan']['backend_config'], after, limit=self.topology.num_gpus)
            outcomes = candidate_decisions(parent, [trial])['candidates'][0]
            document = dict(kind='execution_revision', parent_trial_id=parent['trial_id'],
                candidate_id=trial['trial_id'], comparison=comparison, outcome=outcomes,
                execution_impact=execution_impact(parent, trial, before, after, comparison, self.cost_revision).__dict__,
                measured_before=parent.get('measurement'), measured_after=trial.get('measurement'),
                cost_coverage=[dict(status=(prediction or {}).get('prediction_status'),
                    missing=(prediction or {}).get('evidence', {}).get('missing_profiles', [])[:2])
                    for prediction in (before, after)],
                cost_database_revision=self.cost_revision,
                attribution='complete_joint_strategy',
                source='this_search_database_only')
            identifier = fingerprint(document)
            self.store.add_evidence(identifier, 'PREDICTED', document)
            rows.append(dict(document, evidence_id=identifier))
        return rows

    def recent_execution_revisions(self):
        if self.config.agent_ablation in ('without_dag_evidence', 'without_search_memory'):
            return []
        completed = int(self.store.usage().get('agent_rounds', 0))
        trials = {trial['trial_id']: trial for trial in self.store.trials()}
        pending = None
        for index in range(completed - 1, max(-1, completed - 5), -1):
            row = self.store.get_round(index)
            if row is None or row.get('stage') != 'COMPLETE':
                continue
            identifiers = row.get('measured_candidate_ids') or row.get('candidate_ids', [])
            candidates = [trials[key] for key in identifiers if key in trials][:2]
            if row.get('measured_candidate_ids') and candidates:
                return self.execution_revision_reports(trials[row['parent_trial_id']], candidates)
            if candidates and pending is None:
                pending = (trials[row['parent_trial_id']], candidates)
        if pending is not None:
            return self.execution_revision_reports(*pending)
        return []

    def proposal_archive(self):
        if self.config.agent_ablation == 'without_search_memory':
            return []
        from agents.single_round import parent_row
        identifiers = set()
        for encoded, in self.store.connection.execute('SELECT document FROM search_rounds ORDER BY id'):
            row = json.loads(encoded)
            for intent in row.get('constraint_decision', {}).get('intent_results', []):
                identifiers.update(intent['candidate_ids'])
        trials = [trial for trial in self.store.trials() if trial['trial_id'] in identifiers
            and trial.get('measurement') is None and trial.get('failure') is None
            and (trial.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM']
        return [dict(candidate_id=trial['trial_id'], strategy=parent_row(trial['plan']['backend_config']),
            predicted_step_ms=(trial.get('prediction') or {}).get('performance', {}).get('step_time_ms'),
            prediction_id=(trial.get('prediction') or {}).get('prediction_id'),
            cost_database_revision=(trial.get('prediction') or {}).get('cost_database_revision'),
            measured=False) for trial in trials[-2:]]

    def critical_path_summary(self, trial):
        prediction = trial['prediction']
        if prediction is None or prediction['performance'].get('step_time_ms') is None:
            report = self.execution_decision_report(trial)
            return dict(status=report['status'], evidence_id=report['evidence_id'],
                critical_tasks=[row['task_id'] for row in report['critical_tasks']],
                known_cost_tasks=report['known_cost_tasks'],
                unresolved_task_count=report['unresolved_task_count'],
                measured_step_time_ms=report['measured_step_ms'],
                simulated_step_time_ms=report.get('raw_replayed_step_ms'))
        diagnosis = prediction['diagnosis']
        task_ids = diagnosis.get('critical_tasks', [])
        graph = self.load_prediction_graph(prediction)
        records = {} if graph is None else graph.get('tasks', {})
        selected = sorted(task_ids, key=lambda key: (
            -(records.get(key, {}).get('duration_ms') or 0.0), key))[:12]
        regions = diagnosis.get('critical_regions', [])
        return dict(critical_tasks=selected, critical_task_count=len(task_ids),
            critical_regions=regions[:16], critical_region_count=len(regions),
            task_selection='largest_predicted_duration_on_full_critical_path',
            detail_access=dict(tool='inspect_dag_subgraph', candidate_ids=[trial['trial_id']]),
            **{key: diagnosis.get(key) for key in (
                'simulated_step_time_ms', 'pipeline_idle_ms', 'modeled_exposed_communication_ms')})

    def compact_history(self, limit=4):
        if self.config.agent_ablation == 'without_search_memory':
            return []
        records = self.store.trials()
        observed = [trial for trial in records
            if trial['measurement'] is not None
            or (trial.get('validation') or {}).get('load_validation') is not None
            or trial.get('failure') is not None]
        pending = [trial for trial in records if trial['measurement'] is None
            and trial.get('prediction') is not None and trial not in observed]
        observed_count = min(len(observed), max(1, limit // 2)) if pending else limit
        selected = {trial['trial_id'] for trial in observed[-observed_count:]}
        remaining = limit - len(selected)
        if remaining:
            selected.update(trial['trial_id'] for trial in pending[-remaining:])
        trials = [trial for trial in records if trial['trial_id'] in selected]
        return [dict(trial_id=trial['trial_id'], state=trial['state'],
            configuration={key: trial['plan']['backend_config'].get(key) for key in
                ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'micro_batch',
                 'decoder_chunks', 'execution_backend', 'optimizer_sharding')},
            measurement=None if trial['measurement'] is None else {key: trial['measurement'].get(key) for key in
                ('run_id', 'status', 'step_seconds', 'tokens_per_second', 'peak_allocated_gib')},
            load_validation=None if self.load_validation_summary(trial) is None else {
                key: self.load_validation_summary(trial)[key] for key in ('status', 'returncodes')},
            failure=self.failure_summary(trial),
            prediction=None if trial['prediction'] is None else dict(
                prediction_id=trial['prediction']['prediction_id'],
                prediction_status=trial['prediction'].get('prediction_status'),
                step_time_ms=trial['prediction']['performance'].get('step_time_ms'),
                memory_admission=trial['prediction']['memory'].get('admission')))
            for trial in trials]

    def compact_round_history(self, limit=2):
        if self.config.agent_ablation == 'without_search_memory':
            return []
        completed = int(self.store.usage().get('agent_rounds', 0))
        rows = []
        for index in range(max(0, completed - limit), completed):
            checkpoint = self.store.get_round(index)
            if checkpoint is None or checkpoint.get('stage') != 'COMPLETE':
                continue
            rows.append(dict(round_index=index,
                feedback_evidence_id=checkpoint.get('feedback_evidence_id'),
                feedback=checkpoint.get('feedback'),
                candidate_ids=checkpoint.get('candidate_ids', [])[:8],
                measured_candidate_ids=checkpoint.get('measured_candidate_ids', []),
                stop_reason=checkpoint.get('stop_reason')))
        return rows

    def shared_context(self, trial):
        note_evidence = sorted({reference for note in self.store.notes()[-4:]
            for reference in note['evidence_refs']})
        current_plan = self.compact_plan(trial)
        existing = self.store.connection.execute('SELECT kind, document FROM evidence WHERE id=?',
            (trial['plan_hash'],)).fetchone()
        plan_evidence = dict(kind='complete_training_plan', plan_hash=trial['plan_hash'],
            plan=current_plan, source='deterministic_complete_and_rewrite')
        if existing is None:
            self.store.add_evidence(trial['plan_hash'], 'PREDICTED', plan_evidence)
        elif existing[0] != 'PREDICTED' or json.loads(existing[1]) != plan_evidence:
            raise ValueError('TrainingPlan证据身份与当前完整计划不同')
        decision_report = self.execution_decision_report(trial)
        if decision_report['status'] == 'COSTED_EXECUTION_GRAPH':
            diagnostic_evidence = dict(level='PREDICTED_EXECUTION',
                evidence_id=decision_report['evidence_id'],
                permitted_conclusions='可以依据完整资源回放比较关键任务、资源等待与反事实影响。',
                unavailable_conclusions=[])
        else:
            diagnostic_evidence = dict(level='STRUCTURAL_AND_MEASURED_GLOBAL',
                evidence_id=decision_report['evidence_id'],
                permitted_conclusions='训练图用于模块归属、依赖可达性、通信组和共驻关系；完整训练测量用于全局吞吐和显存。',
                unavailable_conclusions=['不能把未赋成本的任务、通信或区域描述为已确认的性能瓶颈。',
                    '任务级优先级只能作为需要训练或成本测量检验的HYPOTHESIS。'])
        context = dict(context_version=LAYERED_CONTEXT_VERSION,
            model_structure=dict(source='current_model_config_json_and_model_graph',
                configuration={key: self.model_configuration[key] for key in
                    ('model_type', 'num_hidden_layers', 'hidden_size', 'vocab_size', 'num_attention_heads',
                     'num_key_value_heads', 'head_dim', 'intermediate_size', 'moe_intermediate_size',
                     'n_routed_experts', 'num_local_experts', 'num_experts', 'num_experts_per_tok',
                     'n_shared_experts', 'mamba_n_heads', 'mamba_n_groups', 'tie_word_embeddings')
                    if self.model_configuration.get(key) is not None},
                module_types=[kind.value for kind in self.model.layer_types]),
            current_training_plan=current_plan, dag_index=self.dag_index(trial),
            diagnostic_evidence=diagnostic_evidence,
            shared_memory=dict(training_plan_id=trial['trial_id'],
                executable_dag_id=self.backend_dag(trial)['graph_sha256'],
                graph_delta_candidate_id=trial['trial_id'] if trial.get('graph_delta') else None,
                cost_profile_id=self.profile_id,
                measurements_access='get_trial_history',
                search_bounds_access='inspect_global_families',
                full_records_location='backend_experiment_store'),
            evidence_catalog=dict(baseline=(trial['measurement'] or {}).get('run_id'),
                cost_profile=self.profile_id, recent_note_evidence_refs=note_evidence),
            recent_history=self.compact_history(),
            recent_search_rounds=self.compact_round_history(),
            execution_permissions=dict(gpu_execution_enabled=self.config.gpu_execution_enabled,
                full_training_trials=self.config.budget.max_full_training_trials,
                profiling_gpu_hours=self.config.budget.max_profiling_gpu_hours,
                exploration_trial_limit=self.config.uncalibrated_exploration_trials,
                exploration_trials_remaining=max(0, self.config.uncalibrated_exploration_trials
                    - self.store.usage().get('exploration_trials', 0))),
            recent_profiling_results=[dict(request_id=row['request_id'], state=row['state'],
                measurement_type=row['measurement_type'],
                configuration={key: row['execution_request'].get(key) for key in
                    ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'decoder_chunks')},
                result={key: (row.get('result') or {}).get(key) for key in
                    ('status', 'returncodes')})
                for row in self.store.profile_requests()
                if row['state'] in ('COMPLETED', 'FAILED')][-4:],
            budget=dict(limits=self.config.budget.model_dump(), usage=self.store.usage(),
                elapsed_search_seconds=self.store.search_seconds(),
                remaining_search_seconds=max(0,
                    self.config.budget.max_walltime_seconds - self.store.search_seconds())))
        if self.config.agent_ablation == 'without_search_memory':
            context['shared_memory'] = dict(
                training_plan_id=trial['trial_id'],
                cost_profile_id=self.profile_id,
                search_bounds_access='inspect_global_families')
            context['evidence_catalog']['recent_note_evidence_refs'] = []
            context['recent_profiling_results'] = []
        if self.config.execution_evidence_format != 'legacy':
            context['execution_view'] = self.native_execution_view(trial)
            context['dag_index'] = {name: value for name, value in context['dag_index'].items()
                if name in ('graph_sha256', 'task_count', 'edge_count', 'flow_count',
                    'region_count', 'stable_task_id_samples', 'query_contract')}
        context['execution_analysis'] = self.execution_decision_report(trial)
        context['parent_native_execution_effects'] = self.native_strategy_summary(trial['plan']['backend_config'])
        if self.config.single_decision_round:
            if self.config.continuation_history_path and self.config.agent_ablation != 'without_search_memory':
                prior = json.loads(Path(self.config.continuation_history_path).read_text())
                identifier = fingerprint(dict(kind='cross_node_strategy_history', history=prior))
                self.store.add_evidence(identifier, 'HYPOTHESIS', dict(kind='cross_node_strategy_history', history=prior))
                context['prior_episode'] = dict(prior, evidence_id=identifier)
            context['execution_revisions'] = self.recent_execution_revisions()
            context['proposal_archive'] = self.proposal_archive()
            if self.config.agent_ablation != 'without_search_memory':
                policy_rows = self.store.connection.execute(
                    "SELECT id,document FROM evidence WHERE json_extract(document,'$.kind')='policy_frontier' ORDER BY rowid DESC LIMIT 1").fetchall()
                context['policy_comparisons'] = [dict(evidence_id=key,
                    declared_recompute_modes=document['scope']['allowed_ranges'].get('recompute'),
                    comparisons=[dict(layout_id=row['layout_id'], status=row['status'],
                        pending_policy_count=len(row['pending_policies'])) for row in document['layouts']],
                    evaluated=[{name: row[name] for name in ('policy', 'step_time_ms', 'status')}
                        for row in document['evaluations'][:3]])
                    for key, encoded in policy_rows for document in [json.loads(encoded)]]
        if self.config.agent_ablation != 'without_search_memory':
            context['strategy_decision_evidence'] = strategy_decision_evidence(
                trial, self.store.trials(), self.domains)
        return context

    def native_strategy_summary(self, request):
        if not hasattr(self, '_native_strategy_summaries'):
            self._native_strategy_summaries = {}
        key = plan_signature(request)
        if key not in self._native_strategy_summaries:
            started = time.perf_counter()
            invocation = key + ':' + str(time.time_ns())
            self._native_strategy_summaries[key] = native_strategy_effects(
                self.rewriter, request, self.predictor._native_stage_sequence)
            self.store.charge('native-summary:' + invocation, 'native_structure_compilations', 1)
            self.store.charge('native-summary-cpu:' + invocation, 'native_structure_cpu_seconds',
                time.perf_counter() - started)
        return self._native_strategy_summaries[key]

    def deterministic_bottleneck_decision(self, trial):
        performance = self.performance_summary(trial)
        stages = performance.get('stage_summary', [])
        ranks = performance.get('rank_summary', [])
        slowest_stage = max(stages, key=lambda row: (
            row.get('forward_region_ms_max', 0) + row.get('backward_region_ms_max', 0),
            row.get('pp_rank', -1))) if stages else None
        peak_rank = max(ranks, key=lambda row: row.get('peak_bytes', 0)) if ranks else None
        reference = ((trial.get('measurement') or {}).get('run_id')
            or (trial.get('prediction') or {}).get('prediction_id'))
        evidence = dict(kind='deterministic_basic_performance_summary',
            slowest_stage=slowest_stage, highest_memory_rank=peak_rank,
            step_seconds=(trial.get('measurement') or {}).get('step_seconds'))
        evidence_id = fingerprint(evidence)
        self.store.add_evidence(evidence_id, 'MEASURED', evidence)
        diagnosis = BottleneckDecision(
            bottleneck_type='MEASURED' if trial.get('measurement') else 'PREDICTED',
            hypotheses=['根据最慢 Stage、最高显存 Rank 和 Step Time 展开候选'],
            evidence_refs=[evidence_id] + ([reference] if reference else []),
            target_regions=[], requested_metrics=[])
        return diagnosis, evidence_id

    def constraint_agent_context(self, trial, strategy, strategy_id):
        parent_request = trial['plan']['backend_config']
        intents = [dict(intent_index=index, family_id=intent.family_id,
            operation=intent.operation, values={key: value for key, value in
                intent.values.model_dump(exclude_none=True).items() if parent_request.get(key) != value},
            adjustable=intent.adjustable, preserve=intent.preserve,
            allowed_ranges=intent.allowed_ranges.model_dump(exclude_none=True),
            evidence_refs=intent.evidence_refs)
            for index, intent in enumerate(strategy.intents)]
        shared = dict(current_training_plan=dict(plan_id=trial['trial_id'],
            parallelism=trial['plan']['parallelism'], region_partition=trial['plan']['region_partition']),
            fixed_training_contract={key: parent_request.get(key) for key in
                ('model', 'checkpoint', 'data', 'sequence', 'global_batch', 'precision', 'optimizer',
                 'optimizer_hyperparameters', 'main_params_dtype', 'main_grads_dtype',
                 'exp_avg_dtype', 'exp_avg_sq_dtype', 'node_count', 'gpus_per_node')},
            budget=dict(limits=self.config.budget.model_dump(), usage=self.store.usage()))
        validation = self.strategy_validation_feedback(strategy)
        completion = self.constraint_tool(ConstraintValidationToolRequest(tool='complete_training_plans',
            parent_trial_id=trial['trial_id'], strategy_evidence_id=strategy_id,
            intent_indices=list(range(len(strategy.intents))), limit=8))
        return shared | dict(role_context='constraint_validation_v3_compiled_receipts',
            parent_trial_id=trial['trial_id'], strategy_evidence_id=strategy_id,
            intents=intents, backend_domains=self.domains,
            generation_validation=validation,
            tool_history=[completion],
            constraint_authority='deterministic_complete_and_rewrite_and_conditional_partition_solver',
            instructions='完整TrainingPlan、DAG与GraphDelta保留在后端；引用工具返回的candidate_ids与evidence_id。'
                '内存UNKNOWN不能升级为FEASIBLE；单份候选失败不能证明整个家族不可行。')

    @staticmethod
    def constraint_instructions():
        return ('逐项引用已有完成回执的status、candidate_ids和evidence_id。'
            '缺页或需要修复时调用确定性工具；最终intent_results覆盖全部意图。')

    @staticmethod
    def current_proposal_profiles(pending_profiles, intent_results, available_ids=None):
        proposed_ids = {identifier for entry in intent_results for identifier in entry['candidate_ids']}
        if available_ids is not None:
            proposed_ids.intersection_update(available_ids)
        return {identifier: row for identifier, row in pending_profiles.items()
            if proposed_ids.intersection(row['trial_ids'])}

    def constraint_tool(self, request):
        self.check_walltime()
        trials = {trial['trial_id']: trial for trial in self.store.trials()}
        if request.parent_trial_id not in trials:
            raise ValueError('约束工具引用了不存在的父计划')
        parent = trials[request.parent_trial_id]
        row = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
            (request.strategy_evidence_id,)).fetchone()
        if row is None:
            raise ValueError('约束工具引用了不存在的联合意图证据')
        strategy = IntentBatch.model_validate(json.loads(row[0]))
        proposal_count = max(1, len(strategy.intents))
        base_quota, extra_quota = divmod(self.config.budget.max_candidates_per_round, proposal_count)
        candidate_output_limits = [max(1, base_quota + int(index < extra_quota))
                                   for index in range(proposal_count)]
        indices = request.intent_indices or list(range(len(strategy.intents)))
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(strategy.intents)
                for index in indices):
            raise ValueError('约束工具的意图编号无效')
        cost_revision = self.predictor.database.cost_database_revision()
        identity = fingerprint(dict(tool=request.model_dump(mode='json'),
            parent_plan_hash=parent['plan_hash'], cost_revision=cost_revision,
            constraint_implementation=self.constraint_implementation,
            domains=self.domains, candidate_limit=self.config.candidate_limit,
            complete_strategy_output_limit=candidate_output_limits,
            cpu_prediction_budget=self.config.budget.max_cpu_predictions))
        cached = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
            (identity,)).fetchone()
        if cached is not None:
            return json.loads(cached[0])
        if request.tool == 'validate_training_plans':
            if not request.candidate_ids or not set(request.candidate_ids) <= set(trials):
                raise ValueError('完整计划验证需要提供已登记的候选身份')
            selected_ids = request.candidate_ids[request.offset:request.offset + request.limit]
            validations = []
            for identifier in selected_ids:
                candidate = trials[identifier]
                plan = TrainingPlan.model_validate(candidate['plan'])
                self.rewriter.candidate(plan.backend_config)
                self.require_fixed_training_semantics(parent['plan']['backend_config'], plan.backend_config)
                prediction = candidate.get('prediction')
                validations.append(dict(candidate_id=identifier, plan_hash=plan.plan_hash,
                    status='PASS', scope='deterministic_backend_and_training_semantics',
                    memory_admission=None if prediction is None else prediction['memory']['admission'],
                    numerical_validation='requires_actual_training'))
            total = len(request.candidate_ids)
            result = dict(validations=validations, total_count=total, offset=request.offset,
                next_offset=request.offset + request.limit
                    if request.offset + request.limit < total else None)
        else:
            selected_indices = indices[request.offset:request.offset + request.limit]
            reports = []
            primary_predictions = {}
            primary_requests = {}
            for proposal_index, proposal in enumerate(strategy.intents):
                if proposal.proposal_kind != 'complete_strategy':
                    continue
                from search.complete_strategy import compile_complete_strategy
                self.validate_intent_evidence(proposal)
                complete = self.resolve_agent_intent(proposal, parent['plan']['backend_config'])
                try:
                    original = compile_complete_strategy(self.rewriter, parent['plan']['backend_config'], complete)
                    self.require_fixed_training_semantics(parent['plan']['backend_config'], original['request'])
                except ValueError as error:
                    if not isinstance(error.args[0], dict) or error.args[0].get('status') != 'CONSTRAINT_CONFLICT':
                        raise
                    continue
                self.register_constraint_candidate(original, parent, complete)
                complete_request = original['request']
                primary_requests[proposal_index] = complete_request
                self.require_fixed_training_semantics(parent['plan']['backend_config'], complete_request)
                prediction = self.evaluate_planning_request(complete_request, parent['plan']['backend_config'])
                if prediction is not None:
                    primary_predictions[proposal_index] = prediction['prediction_id']
            for index in selected_indices:
                candidate_output_limit = candidate_output_limits[index]
                operation_id = fingerprint(dict(strategy=request.strategy_evidence_id, intent_index=index,
                    parent=parent['plan_hash'], tool=request.tool, cost_revision=cost_revision,
                    constraint_implementation=self.constraint_implementation,
                    domains=self.domains, candidate_limit=self.config.candidate_limit,
                    complete_strategy_output_limit=candidate_output_limit,
                    cpu_prediction_budget=self.config.budget.max_cpu_predictions))
                saved = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                    (operation_id,)).fetchone()
                if saved is not None:
                    reports.append(json.loads(saved[0]))
                    continue
                try:
                    self.validate_intent_evidence(strategy.intents[index])
                    intent = self.resolve_agent_intent(strategy.intents[index], parent['plan']['backend_config'])
                    self.require_fixed_training_semantics(parent['plan']['backend_config'],
                        parent['plan']['backend_config'] | intent.values)
                    if strategy.intents[index].proposal_kind == 'complete_strategy':
                        continuation_requests = list(primary_requests.values())
                        completion = self.expand_complete_strategy(
                            parent['plan']['backend_config'], intent, candidate_output_limit,
                            continuation_requests=continuation_requests)
                        rewrites = completion['candidates']
                    elif (request.tool == 'solve_conditional_partition' or intent.module_relations
                            or strategy.intents[index].proposal_kind == 'conditional_intent'):
                        family_values = self.global_family_values(parent['plan']['backend_config'])
                        family_values.update({key: value for key, value in intent.values.items()
                            if key in GLOBAL_FIELDS})
                        family = dict(family_id=strategy.intents[index].family_id,
                            values=family_values)
                        completion = self.solve_joint_scope(parent['plan']['backend_config'],
                            family, intent, candidate_output_limit)
                        rewrites = completion['candidates']
                    else:
                        rewrites = self.rewriter.complete_and_rewrite(
                            parent['plan']['backend_config'], intent, limit=self.config.candidate_limit)
                        completion = dict(status='UNKNOWN', search_complete=False,
                            termination='bounded_constraint_completion', proofs=[], missing_cost_requests=[])
                except ValueError as error:
                    conflict = error.args[0]
                    if not isinstance(conflict, dict) or conflict.get('status') != 'CONSTRAINT_CONFLICT':
                        raise
                    report = dict(intent_index=index, status='CONSTRAINT_CONFLICT', candidate_ids=[],
                        constraint_evidence_id=operation_id, failure_constraints=conflict,
                        family_admission='UNKNOWN', family_infeasibility_proven=False,
                        revision_owner='JointStrategyAgent', reason=conflict.get('reason', str(conflict['conflicts'])))
                    self.store.add_evidence(operation_id, 'FAILURE', report)
                    reports.append(report)
                    continue
                candidate_ids = []
                for rewrite in rewrites:
                    candidate = self.register_constraint_candidate(rewrite, parent, intent)
                    candidate_ids.append(candidate['trial_id'])
                report = dict(intent_index=index,
                    status='COMPLETED' if candidate_ids else 'UNKNOWN',
                    candidate_ids=sorted(set(candidate_ids)), constraint_evidence_id=operation_id,
                    family_admission=completion['status'],
                    search_complete=completion['search_complete'], termination=completion['termination'],
                    proof_count=len(completion.get('proofs', [])),
                    missing_cost_count=len(completion.get('missing_cost_requests', [])))
                report['candidate_output_limit'] = (candidate_output_limit
                    if strategy.intents[index].proposal_kind == 'complete_strategy'
                    else self.config.candidate_limit)
                if index in primary_predictions:
                    report['original_proposal_prediction_id'] = primary_predictions[index]
                if 'continuation_search' in completion:
                    continuation = completion['continuation_search']
                    report['continuation_evidence_id'] = completion['continuation_evidence_id']
                    report['continuation_evaluations'] = continuation['evaluations']
                    report['retention_changes'] = len(continuation['retention_changes'])
                    report['continuation_source_plan_ids'] = completion.get('continuation_source_plan_ids', [])
                if completion.get('placement_pair') is not None:
                    report['placement_pair'] = completion['placement_pair']
                self.store.add_evidence(operation_id, 'PREDICTED', report)
                reports.append(report)
            total = len(indices)
            result = dict(intent_results=reports, total_count=total, offset=request.offset,
                next_offset=request.offset + request.limit
                    if request.offset + request.limit < total else None,
                complete_plan_access='validate_training_plans',
                revision_requests=[dict(intent_index=row['intent_index'],
                    **row['failure_constraints']) for row in reports if row['status'] == 'CONSTRAINT_CONFLICT'])
        result_tokens = token_count(result)
        if result_tokens > 4000:
            full_identity = fingerprint(dict(tool=request.model_dump(mode='json'), result=result))
            self.store.add_evidence(full_identity, 'PREDICTED', dict(
                tool=request.model_dump(mode='json'), result=result, result_tokens=result_tokens,
                evidence_scope='complete_constraint_tool_result_retained_in_backend'))
            result = dict(status='PAGE_TOO_LARGE', complete_result_id=full_identity,
                complete_result_tokens=result_tokens, maximum_result_tokens=4000,
                suggested_query=request.model_copy(update=dict(limit=max(1, request.limit // 2))).model_dump(mode='json'))
            identity = fingerprint(dict(tool=request.model_dump(mode='json'), result=result))
            result_tokens = token_count(result)
        receipt = dict(evidence_id=identity, tool=request.model_dump(mode='json'), result=result,
            result_tokens=result_tokens, token_encoding=TOKEN_ENCODING_NAME,
            evidence_scope='deterministic_constraint_solver')
        self.store.add_evidence(identity, 'PREDICTED', receipt)
        return receipt

    require_fixed_training_semantics = staticmethod(require_fixed_training_semantics)

    def scoped_candidate_request(self, request):
        request = dict(request)
        if self.config.request.get('max_allocation_seconds') is not None:
            request['max_allocation_seconds'] = self.config.request['max_allocation_seconds']
        if self.config.request.get('attention_backend_capabilities') is not None:
            request['attention_backend_capabilities'] = self.config.request['attention_backend_capabilities']
        identity = fingerprint(dict(output=str(self.output.resolve()),
            plan_hash=self.plan(request).plan_hash))
        request['run_id'] = 'candidate-' + identity[:24]
        return request

    def register_constraint_candidate(self, rewrite, parent, intent):
        row = dict(rewrite, request=self.scoped_candidate_request(rewrite['request']))
        if row.get('proposal_kind') == 'placement_control':
            intent = RewriteIntent.model_validate(row['control_intent'])
        elif row.get('intent') is not None:
            intent = RewriteIntent.model_validate(row['intent'])
        candidate_source = row.get('proposal_kind',
            'conditional_partition' if 'partition_evidence' in row else 'constraint_completion')
        plan = self.plan(row['request'], parent['trial_id'], [intent.model_dump(mode='json')])
        self.require_fixed_training_semantics(parent['plan']['backend_config'], plan.backend_config)
        directory = self.output / 'constraint-plans'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (plan.plan_hash + '.json.gz')
        if not path.exists():
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                json.dump(row, stream, ensure_ascii=False, allow_nan=False)
        trial, _ = self.store.put_trial(plan, intent.model_dump(mode='json'), row['graph_delta'],
            rewrite=dict(artifact=str(path), constraint_result=row['constraint_result'],
                candidate_source=candidate_source))
        return trial

    def validate_constraint_decision(self, response, strategy, parent, strategy_id):
        indices = [row.intent_index for row in response.intent_results]
        if sorted(indices) != list(range(len(strategy.intents))):
            raise ValueError('约束Agent必须逐项报告全部联合意图，编号不得重复')
        accepted_ids = set()
        for report in response.intent_results:
            matching = []
            for identifier in report.evidence_refs:
                row = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                    (identifier,)).fetchone()
                if row is None:
                    raise ValueError('约束Agent引用了不存在的完成证据')
                receipt = json.loads(row[0])
                query = receipt.get('tool', {})
                if (query.get('tool') in ('complete_training_plans', 'solve_conditional_partition')
                        and query.get('parent_trial_id') == parent['trial_id']
                        and query.get('strategy_evidence_id') == strategy_id):
                    matching.extend(item for item in receipt['result']['intent_results']
                        if item['intent_index'] == report.intent_index)
            if not any(item['status'] == report.status
                    and set(item['candidate_ids']) == set(report.candidate_ids) for item in matching):
                raise ValueError('约束Agent结论与确定性求解工具证据不同')
            accepted_ids.update(report.candidate_ids)
        return sorted(accepted_ids)

    def constraint_decision_feedback(self, response, observation):
        expected = {}
        for identifier, encoded in self.store.connection.execute('SELECT id, document FROM evidence'):
            receipt = json.loads(encoded)
            query = receipt.get('tool', {})
            if (query.get('tool') not in ('complete_training_plans', 'solve_conditional_partition')
                    or query.get('parent_trial_id') != observation['parent_trial_id']
                    or query.get('strategy_evidence_id') != observation['strategy_evidence_id']):
                continue
            for report in receipt['result']['intent_results']:
                expected[report['intent_index']] = dict(report, evidence_refs=[identifier])
        missing = sorted(set(range(len(observation['intents']))) - set(expected))
        if missing:
            return None, [ConstraintValidationToolRequest(tool='complete_training_plans',
                parent_trial_id=observation['parent_trial_id'],
                strategy_evidence_id=observation['strategy_evidence_id'],
                intent_indices=missing, limit=min(len(missing), 6))]
        reports = response.intent_results
        valid = sorted(row.intent_index for row in reports) == sorted(expected)
        valid = valid and all(row.status == expected[row.intent_index]['status']
            and set(row.candidate_ids) == set(expected[row.intent_index]['candidate_ids'])
            and bool(set(row.evidence_refs) & set(expected[row.intent_index]['evidence_refs']))
            for row in reports)
        if valid:
            return None, []
        result = dict(status='REVISION_REQUIRED', revision_owner='ConstraintValidationAgent',
            reason='逐项使用完成工具返回的status、candidate_ids和evidence_refs',
            expected_intent_results=[expected[index] for index in sorted(expected)])
        tool = dict(tool='validate_constraint_decision')
        identifier = fingerprint(dict(tool=tool, response=response.model_dump(mode='json'), result=result))
        self.store.add_evidence(identifier, 'FAILURE', dict(tool=tool, result=result))
        return dict(evidence_id=identifier, tool=tool, result=result,
            result_tokens=token_count(result)), []

    def performance_agent_context(self, trial):
        return self.shared_context(trial) | dict(model=dict(name=self.model.name,
            num_layers=self.model.num_layers, hidden_size=self.model.hidden_size,
            num_attention_heads=self.model.num_attention_heads, num_experts=self.model.num_experts),
            hardware_topology=self.topology_summary(), performance=self.performance_summary(trial),
            critical_path=self.critical_path_summary(trial), recent_history=self.compact_history())

    def strategy_agent_context(self, trial, diagnosis, diagnosis_id, global_family_search):
        placement = placement_evidence(trial['plan']['backend_config'], self.topology.devices)
        placement_id = fingerprint(dict(kind='native_group_placement',
            plan_hash=trial['plan_hash'], placement=placement))
        self.store.add_evidence(placement_id, 'PREDICTED', dict(
            kind='native_group_placement', placement=placement))
        compact_diagnosis = {key: value for key, value in diagnosis.model_dump(mode='json').items()
            if key in ('bottleneck_type', 'affected_regions', 'affected_tasks',
                       'related_variables', 'hypotheses', 'evidence_refs', 'target_regions')}
        compact_diagnosis['affected_tasks'] = compact_diagnosis.get('affected_tasks', [])[:12]
        compact_diagnosis['hypotheses'] = compact_diagnosis.get('hypotheses', [])
        families = global_family_search.get('representative_families', [])[:16]
        mappings = {fingerprint(row['values']['rank_mapping'])[:8]: row['values']['rank_mapping']
            for row in families}
        columns = ['family_id', *[key for key in GLOBAL_FIELDS if key != 'rank_mapping'],
            'rank_mapping_id', 'status', 'candidate_count', 'best_predicted_step_ms',
            'best_measured_tokens_per_s']
        catalog = dict(total_legal_families=global_family_search['total_legal_families'],
            family_status_counts=global_family_search['family_status_counts'], columns=columns,
            rows=[[row['family_id'], *[row['values'][key] for key in GLOBAL_FIELDS
                if key != 'rank_mapping'], fingerprint(row['values']['rank_mapping'])[:8],
                row['status'], row['candidate_count'], row['best_predicted_step_ms'],
                row['best_measured_tokens_per_s']] for row in families],
            rank_mapping_catalog=mappings, next_offset=len(families)
                if len(families) < global_family_search['total_legal_families'] else None,
            catalog_access=global_family_search['catalog_access'],
            ordering_semantics='identity_order_without_performance_preference',
            proposal_permission='complete_strategy_can_use_family_id_null_and_native_validation')
        from search.communication_placement import compact_placement_evidence
        context = self.shared_context(trial) | dict(diagnosis=compact_diagnosis,
            performance=self.performance_summary(trial),
            diagnosis_evidence_id=diagnosis_id,
            legal_actions={name: value.transformation_semantics for name, value in self.registry.entries.items()
                if value.supported},
            backend_domains=self.domains, global_family_search=catalog,
            continuation_semantics=dict(source='all_complete_proposals_in_current_batch',
                compatibility='same_effective_execution_settings_and_module_count',
                generated_object='ordered_complete_module_layouts',
                use='complete_partial_layouts_then_compare_actual_execution',
                retention='best_evaluated_completion_with_donor_coverage_and_unknown_state'),
            communication_placement=dict(evidence_id=placement_id, **compact_placement_evidence(placement)),
            backend_state_access=dict(allowed_execution_backends=trial['plan']['backend_config'].get(
                'allowed_execution_backends', ['megatron_ddp']),
                tool='inspect_global_families', field='state_variants'),
            native_recomputation_options=dict(modes=['none', 'full']
                    + (['selective'] if selective_modules(self.model) else []),
                semantics=dict(none='保存反向所需激活，消除重计算引入的前向重放',
                    full='重放完整Transformer层，减少保存激活',
                    selective='重放明确指定的模块，在重复计算与激活保存之间取舍'),
                selective_policy=dict(recompute_granularity='selective',
                    recompute_modules='从supported_modules选择明确的非空模块集合'),
                supported_modules=selective_modules(self.model),
                runtime_verification='每个Rank记录实际模块上的原生checkpoint开关，完整训练检查成功更新',
                source='Megatron TransformerConfig.recompute_modules and native module checkpoint implementations'),
            parallel_legality=dict(world_size=self.topology.num_gpus,
                equations=['world_size=tp*cp*pp*dp', 'world_size=etp*ep*expert_dp*pp',
                    'GBS=dp*micro_batch*num_microbatches', 'regions=pp*vpp',
                    'sum(decoder_chunks)=num_layers'],
                native_interleaved_schedule=['vpp>1 requires pp>1',
                    'pp<=group<=num_microbatches',
                    'current_training_entry_requires_num_microbatches%group=0']))
        context['cross_shape_investigations'] = self.graph_investigations(trial, context['execution_analysis'])
        if self.config.agent_ablation != 'without_dag_evidence':
            request = trial['plan']['backend_config']
            context['physical_parallel_alternatives'] = physical_parallel_investigations(request,
                self.joint_global_families(request), context['execution_analysis'], self.topology.devices,
                self.attempted_execution_shapes())
            controls = schedule_control_options(trial, self.store.trials(), self.domains.get('group', ()))
            if controls:
                measured_peak = trial['measurement']['peak_allocated_gib']
                context['schedule_controls'] = dict(evidence_id=context['execution_analysis']['evidence_id'],
                    measured_peak_gib=measured_peak, device_capacity_gib=self.topology.memory_per_gpu_gb,
                    candidates=controls, performance_status='REQUIRES_REAL_MEASUREMENT')
        context['runtime_failure'] = self.failure_summary(trial)
        if self.config.agent_recent_runtime_failure_evidence:
            failures = []
            for row in reversed(self.store.trials()):
                if row['trial_id'] == trial['trial_id'] or not row.get('failure'):
                    continue
                report = self.failure_summary(row)
                if not report or not report.get('observed'):
                    continue
                request = row['plan']['backend_config']
                failures.append(dict(candidate_id=row['trial_id'], status=report['status'],
                    strategy={key: request.get(key) for key in ('tp', 'pp', 'dp', 'cp', 'ep', 'etp',
                        'expert_dp', 'vpp', 'micro_batch', 'recompute', 'execution_backend',
                        'optimizer_offload_fraction')}, observed=report['observed'],
                    evidence_refs=report['evidence_refs']))
                if len(failures) == 2:
                    break
            if failures:
                context['recent_candidate_runtime_failures'] = failures
        init_witness_path = self.output / 'optimizer-init-witness.json'
        if init_witness_path.is_file():
            witness = json.loads(init_witness_path.read_text())
            if witness['model'] != trial['plan']['backend_config']['name']:
                raise ValueError('优化器初始化观测与当前模型不一致')
            witness_id = fingerprint(witness)
            self.store.add_evidence(witness_id, 'MEASURED', witness)
            context['optimizer_init_witness'] = dict(evidence_id=witness_id, **witness)
        recent_runtime = []
        for row in reversed(self.store.trials()):
            result = row.get('measurement') or {}
            if not result.get('execution_deadline_reached'):
                continue
            plan = row['plan']['backend_config']
            recent_runtime.append(dict(status='EXECUTION_TIME_LIMIT',
                gpu_allocation_seconds=result.get('gpu_allocation_seconds'),
                strategy={key: plan.get(key) for key in ('tp', 'pp', 'cp', 'ep', 'etp',
                    'execution_backend', 'optimizer_offload_fraction', 'optimizer_transfer_overlap',
                    'decoder_chunks')}))
            if len(recent_runtime) == 2:
                break
        if recent_runtime:
            context['recent_runtime_timeouts'] = recent_runtime
        failed_profiles = []
        for row in reversed(self.store.profile_requests()):
            result = row.get('result') or {}
            if row['state'] != 'FAILED' or not result.get('run_id'):
                continue
            seconds = result.get('gpu_allocation_seconds')
            limit = result.get('max_allocation_seconds')
            deadline = result.get('execution_deadline_reached', False) or (
                result.get('status') == 'FAILED' and seconds is not None and limit is not None
                and seconds >= limit and bool(result.get('returncodes'))
                and all(code == -15 for code in result['returncodes']))
            failed_profiles.append(dict(
                type=row['measurement_type'], result='EXECUTION_TIME_LIMIT' if deadline else result.get('status'),
                strategy={key: result.get(key) for key in ('tp', 'pp', 'cp', 'ep', 'etp',
                    'execution_backend', 'optimizer_offload_fraction', 'decoder_chunks')},
                gpu_allocation_seconds=seconds))
            if len(failed_profiles) == 2:
                break
        if failed_profiles:
            context['recent_profile_failures'] = failed_profiles
        context['dag_index'] = {key: value for key, value in context['dag_index'].items()
            if key in ('graph_sha256', 'task_count', 'edge_count', 'flow_count',
                       'region_count', 'regions', 'query_contract')}
        revisions = []
        for identifier, encoded in self.store.connection.execute(
                "SELECT id, document FROM evidence WHERE json_extract(document, '$.tool.tool') "
                "IN ('complete_training_plans', 'solve_conditional_partition') ORDER BY rowid DESC LIMIT 8"):
            receipt = json.loads(encoded)
            revisions.extend(dict(item, validation_evidence_id=identifier)
                for item in receipt.get('result', {}).get('revision_requests', []))
        if revisions:
            context['revision_requests'] = revisions[:8]
        failed_trials = [row for row in self.compact_history(limit=8) if row.get('failure')]
        context['action_evidence_contract'] = dict(
            required_for_each_intent=['parent_plan_id', 'target_bottleneck', 'values',
                'evidence_refs', 'interaction_evidence_ids', 'expected_effect',
                'possible_side_effects'],
            action_query='preview_transformation 或 query_affected_decisions',
            query_result_must_cover=['changed_parallelism_or_layout', 'affected_regions_or_flows',
                'communication_or_memory_interaction', 'missing_cost_or_memory_condition'],
            candidate_identity='完整生效配置、训练语义与模块区间共同确定候选身份')
        context['conditional_failure_memory'] = failed_trials[:4]
        if self.config.agent_ablation != 'without_search_memory':
            context['measured_strategy_memory'] = measured_strategy_memory(
                self.store.trials(), trial['plan']['backend_config'], self.topology.devices, limit=8, model=self.model)
        context['execution_structure'] = execution_structure(
            trial['plan']['backend_config'], self.topology.devices, self.model)
        return context

    @staticmethod
    def failure_summary(trial):
        failure = trial.get('failure')
        if failure is None:
            result = trial.get('measurement') or (trial.get('validation') or {}).get('load_validation')
            if result is None or result.get('status') == 'PASS':
                return None
            errors = []
            for path in sorted(Path(result['raw_artifacts']).glob('node*.log')):
                errors.extend(line.strip()[:800] for line in path.read_text(errors='replace').splitlines()
                    if 'Error:' in line or 'AssertionError' in line or 'Multiple ranks detected' in line
                    or 'Failed to CUDA calloc' in line or 'Failed to CUDA malloc' in line)
            return dict(status=result['status'], observed_error_lines=list(dict.fromkeys(errors))[:4],
                execution_deadline_reached=result.get('execution_deadline_reached', False),
                host_memory_failures=result.get('host_memory_failures', []),
                evidence_refs=[result['run_id']], source=result.get('source', 'actual_training_log'))
        observed = failure.get('observed') or {}
        site = observed.get('allocation_site')
        source = observed.get('source_log')
        if site is None and source and Path(source).is_file():
            site = oom_allocation_site(Path(source).read_text(errors='replace'))
        return dict(witness_id=failure.get('witness_id'), status=failure.get('status'),
            observed={key: observed.get(key) for key in
                ('first_exception_rank', 'phase', 'requested_bytes', 'allocated_bytes',
                 'reserved_bytes', 'device_capacity_bytes', 'device_free_bytes')} | dict(allocation_site=site),
            evidence_refs=[failure.get('witness_id', trial['trial_id'])],
            detail_access=dict(tool='inspect_memory', candidate_ids=[trial['trial_id']]))

    @staticmethod
    def load_validation_summary(trial):
        result = (trial.get('validation') or {}).get('load_validation')
        if result is None:
            return None
        return {key: result.get(key) for key in
            ('run_id', 'status', 'returncodes', 'validation_scope')}

    @staticmethod
    def available_for_joint_measurement(trial):
        return (trial.get('measurement') is None and not trial.get('failure')
            and (EvidenceGuidedSearch.load_validation_summary(trial) or {}).get('status')
                not in ('FAILED', 'CUDA_OOM', 'HOST_MEMORY_BUDGET_EXCEEDED')
            and (trial.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM')

    def candidate_summary(self, trial):
        prediction = trial['prediction']
        delta = trial['graph_delta'] or {}
        return dict(trial_id=trial['trial_id'], global_family_id=self.global_family_id(
                trial['plan']['backend_config']), plan=self.compact_plan(trial), state=trial['state'],
            measurement=None if trial['measurement'] is None else {key: trial['measurement'].get(key)
                for key in ('run_id', 'status', 'step_seconds', 'tokens_per_second',
                    'peak_allocated_gib', 'performance_eligible')},
            load_validation=self.load_validation_summary(trial),
            failure=self.failure_summary(trial),
            prediction=None if prediction is None else dict(prediction_id=prediction['prediction_id'],
                prediction_status=prediction.get('prediction_status'), performance=prediction['performance'],
                memory={key: prediction['memory'].get(key) for key in ('admission', 'peak_per_rank_bytes',
                    'memory_headroom_per_rank_bytes', 'oom_risk', 'missing_profiles')},
                cost_coverage=prediction['evidence'].get('calibration_coverage'),
                missing_profiles=prediction['evidence'].get('missing_profiles', [])),
            graph_delta=dict(config_changes=delta.get('config_changes', {}),
                counts={key: len(value) for key, value in delta.items() if isinstance(value, list)},
                affected_task_ids=delta.get('affected_task_ids', [])[:8],
                changed_region_ids=delta.get('changed_regions', [])[:8]),
            hypothesis=trial['hypothesis'])

    def investigation_candidate_summary(self, trial):
        prediction = trial['prediction']
        delta = trial['graph_delta'] or {}
        plan = trial['plan']
        hypothesis = trial['hypothesis'] or {}
        headroom = {} if prediction is None else prediction['memory'].get(
            'memory_headroom_per_rank_bytes', {})
        headroom_summary = None if not headroom else dict(
            minimum_bytes=min(headroom.values()), maximum_bytes=max(headroom.values()),
            minimum_rank=min(headroom, key=headroom.get))
        return dict(trial_id=trial['trial_id'], plan_hash=trial['plan_hash'],
            global_family_id=self.global_family_id(plan['backend_config']),
            plan=dict(parallelism=plan['parallelism'], expert_parallelism=plan['expert_parallelism'],
                region_partition=plan['region_partition'], schedule=plan['schedule'],
                num_microbatches=plan['num_microbatches'], rank_mapping=plan['rank_mapping'],
                micro_batch=plan['backend_config']['micro_batch'], group=plan['backend_config']['group'],
                precision_aware_optimizer=plan['backend_config'].get('precision_aware_optimizer', False),
                execution_backend=plan['backend_config'].get('execution_backend', 'megatron_ddp'),
                state_sharding=plan.get('state_sharding', {}),
                recompute=plan['backend_config']['recompute'],
                overlap_p2p=plan['backend_config'].get('overlap_p2p'),
                optimizer_sharding=plan['backend_config'].get('optimizer_sharding')),
            prediction=None if prediction is None else dict(prediction_id=prediction['prediction_id'],
                prediction_status=prediction['prediction_status'], performance=prediction['performance'],
                memory_admission=prediction['memory']['admission'],
                memory_headroom=headroom_summary,
                cost_coverage=prediction['evidence'].get('calibration_coverage'),
                missing_profiles=prediction['evidence'].get('missing_profiles', [])),
            graph_delta=dict(config_changes=delta.get('config_changes', {}),
                changed_counts={key: len(value) for key, value in delta.items() if isinstance(value, list)}),
            intent=dict(operation=hypothesis.get('operation'), hypothesis=hypothesis.get('hypothesis'),
                evidence_refs=hypothesis.get('evidence_refs', []),
                target_task_ids=hypothesis.get('target_task_ids', [])[:4]))

    def trial_history_summary(self, trial):
        prediction = trial.get('prediction') or {}
        measurement = trial.get('measurement') or {}
        missing = prediction.get('evidence', {}).get('missing_profiles', [])
        return dict(trial_id=trial['trial_id'], state=trial['state'],
            effective_decisions=strategy_decision_values(trial['plan']['backend_config']),
            measurement={key: measurement.get(key) for key in ('run_id', 'status',
                'tokens_per_second', 'step_seconds', 'peak_allocated_gib', 'performance_eligible')},
            failure=self.failure_summary(trial), load_validation=self.load_validation_summary(trial),
            prediction=dict(prediction_id=prediction.get('prediction_id'),
                status=prediction.get('prediction_status'),
                step_time_ms=prediction.get('performance', {}).get('step_time_ms'),
                memory_admission=prediction.get('memory', {}).get('admission'),
                missing_profile_count=len(missing), missing_profile_samples=missing[:2]),
            details_access=dict(tool='evaluate_candidate', candidate_ids=[trial['trial_id']]))

    def investigation_candidate_table(self, candidates):
        columns = ['trial_id', 'global_family_id', 'tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'micro_batch',
            'group', 'decoder_chunks', 'recompute', 'precision_aware_optimizer', 'execution_backend',
            'dense_sharding_strategy', 'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size',
            'schedule', 'overlap_p2p',
            'prediction_status', 'step_time_ms', 'prediction_interval_ms',
            'prediction_uncertainty', 'ranking_sensitivity_scenario_ms', 'memory_admission',
            'minimum_memory_headroom_bytes', 'cost_condition_id', 'config_changes',
            'graph_delta_counts', 'intent_operation', 'evidence_refs']
        catalog = {}
        rows = []
        for trial in candidates:
            request = trial['plan']['backend_config']
            prediction = trial['prediction']
            hypothesis = trial['hypothesis'] or {}
            headroom = {} if prediction is None else prediction['memory'].get(
                'memory_headroom_per_rank_bytes', {})
            cost_condition = None if prediction is None else dict(
                coverage=prediction['evidence'].get('calibration_coverage'),
                missing_profiles=prediction['evidence'].get('missing_profiles', []))
            condition_id = None if cost_condition is None else fingerprint(cost_condition)
            if condition_id is not None:
                catalog[condition_id] = cost_condition
            delta = trial['graph_delta'] or {}
            rows.append([trial['trial_id'], self.global_family_id(request), request['tp'], request['pp'], request['cp'], request['ep'],
                request.get('etp'), request['dp'], request['vpp'], request['micro_batch'], request['group'],
                request['decoder_chunks'], request['recompute'], request.get('precision_aware_optimizer', False),
                request.get('execution_backend', 'megatron_ddp'),
                request.get('dense_sharding_strategy', 'no_shard'),
                request.get('expert_sharding_strategy', 'no_shard'),
                request.get('dp_inner_size', request['dp']), request.get('dp_outer_size', 1),
                request.get('pipeline_schedule'),
                request.get('overlap_p2p'), None if prediction is None else prediction['prediction_status'],
                None if prediction is None else prediction['performance']['step_time_ms'],
                None if prediction is None else prediction['performance'].get('prediction_interval_ms'),
                None if prediction is None else prediction['performance'].get('prediction_uncertainty'),
                None if prediction is None else prediction['performance'].get(
                    'ranking_sensitivity_scenario_ms'),
                None if prediction is None else prediction['memory']['admission'],
                min(headroom.values()) if headroom else None, condition_id, delta.get('config_changes', {}),
                {key: len(value) for key, value in delta.items() if isinstance(value, list)},
                hypothesis.get('operation'), hypothesis.get('evidence_refs', [])])
        return dict(columns=columns, rows=rows, total_count=len(rows), cost_condition_catalog=catalog)

    def investigation_agent_context(self, trial, candidates, strategy, interactions, interaction_id):
        table = self.investigation_candidate_table(candidates)
        columns = ['trial_id', 'global_family_id', 'tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp',
            'micro_batch', 'group', 'decoder_chunks', 'recompute', 'execution_backend',
            'dense_sharding_strategy', 'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size',
            'step_time_ms', 'prediction_status', 'memory_admission',
            'minimum_memory_headroom_bytes', 'cost_condition_id']
        indices = [table['columns'].index(name) for name in columns]
        overview_columns = ['trial_id', 'global_family_id', 'step_time_ms', 'memory_admission']
        overview_indices = [table['columns'].index(name) for name in overview_columns]
        candidate_table = dict(columns=columns,
            rows=[[row[index] for index in indices] for row in table['rows'][:6]],
            overview_columns=overview_columns,
            overview=[[row[index] for index in overview_indices] for row in table['rows']],
            total_count=table['total_count'], detail_access='evaluate_candidate',
            cost_condition_catalog={key: dict(coverage=value['coverage'],
                missing_profiles=value['missing_profiles'][:4],
                missing_profile_count=len(value['missing_profiles']))
                for key, value in table['cost_condition_catalog'].items()})
        metrics = [{key: row[key] for key in ('trial_id', 'predicted_improvement',
            'structural_distance', 'estimated_gpu_hours', 'cost_sufficient_for_ranking')}
            for row in self.selection_metrics(candidates)[:6]]
        candidate_ids = {row['trial_id'] for row in candidates} | {trial['trial_id']}
        pending = [row for row in self.store.profile_requests()
            if row['state'] in ('PENDING', 'APPROVED', 'DEFERRED')
            and candidate_ids.intersection(row['trial_ids'])]
        pending.sort(key=lambda row: (-len(candidate_ids.intersection(row['trial_ids'])),
            -row['priority'], row['estimated_gpu_hours'], row['request_id']))
        preferred = self.select_unranked_profile(pending, candidates, trial)
        if preferred is not None:
            pending = [preferred, *[row for row in pending if row['request_id'] != preferred['request_id']]]
        context = self.shared_context(trial) | dict(role_context='investigation_v11',
            selection_limits=self.selection_limits(),
            candidates=candidate_table,
            complete_strategy_comparison=complete_strategy_comparisons(
                trial, candidates[:6], self.topology.devices, self.model),
            candidate_decision_report=candidate_decisions(trial, candidates),
            strategy_batch_reason=strategy.reason,
            candidate_hypotheses=[dict(trial_id=candidate['trial_id'],
                candidate_source=(candidate.get('rewrite') or {}).get('candidate_source'),
                hypothesis=(candidate.get('hypothesis') or {}).get('hypothesis'),
                evidence_refs=(candidate.get('hypothesis') or {}).get('evidence_refs', []),
                rank_mapping=candidate['plan']['backend_config']['rank_mapping'])
                for candidate in candidates],
            selection_metrics=metrics,
            ranking_sensitivity=self.ranking_sensitivity(candidates),
            strategy_evidence_refs=strategy.evidence_refs,
            interaction_summary=dict(evidence_id=interaction_id, edge_count=len(interactions['edges']),
                details_access='compare_candidates'),
            pending_profiling_request_count=len(pending),
            pending_profiling_requests=[dict(
                    request_id=row['request_id'], trial_ids=row['trial_ids'], cost_keys=row['cost_keys'][:2],
                    cost_key_count=len(row['cost_keys']),
                    measurement_type=row['measurement_type'], reason=row['reason'],
                    estimated_gpu_hours=row['estimated_gpu_hours'], state=row['state'])
                for row in pending[:6]])
        sensitive = ranking_sensitive_candidates(context['ranking_sensitivity'])
        decisions = profile_decisions(pending, candidates, trial, sensitive, self.attempted_execution_plans())
        by_request = {row['request_id']: row for row in pending}
        if decisions:
            suggested = [by_request[row['request_id']] for row in decisions[:6]]
            suggested_ids = {row['request_id'] for row in suggested}
            pending = suggested + [row for row in pending if row['request_id'] not in suggested_ids]
            by_decision = {row['request_id']: row for row in decisions}
            context['pending_profiling_requests'] = [dict(request_id=row['request_id'],
                trial_ids=row['trial_ids'], cost_key_count=len(row['cost_keys']),
                measurement_type=row['measurement_type'], reason=row['reason'],
                decision_reasons=by_decision.get(row['request_id'], {}).get('reasons', []),
                decision_candidate_ids=by_decision.get(row['request_id'], {}).get('candidate_ids', []),
                estimated_gpu_hours=row['estimated_gpu_hours'], state=row['state'])
                for row in pending[:6]]
        context['dag_index'] = {key: value for key, value in context['dag_index'].items()
            if key in ('graph_sha256', 'task_count', 'edge_count', 'flow_count',
                       'region_count', 'regions', 'query_contract')}
        context['parent_execution_structure'] = execution_structure(
            trial['plan']['backend_config'], self.topology.devices, self.model)
        context['candidate_execution_effects'] = [dict(trial_id=row['trial_id'],
            native_execution_effects=self.native_strategy_summary(row['plan']['backend_config']),
            **{name: value for name, value in compare_strategy_structure(
                trial['plan']['backend_config'], row['plan']['backend_config'],
                self.topology.devices, self.model).items() if name in ('changed_settings', 'candidate')})
            for row in candidates[:6]]
        if self.config.agent_ablation != 'without_search_memory':
            context['measured_strategy_memory'] = measured_strategy_memory(
                self.store.trials(), trial['plan']['backend_config'], self.topology.devices, limit=8, model=self.model)
        return context

    def selection_limits(self):
        names = ('max_full_training_trials', 'max_total_gpu_hours', 'max_profiling_gpu_hours',
            'max_agent_calls', 'max_agent_tokens', 'max_benchmarks_per_round', 'max_cpu_predictions')
        return dict({name: getattr(self.config.budget, name) for name in names},
            uncalibrated_exploration_trials=self.config.uncalibrated_exploration_trials)

    def feedback_agent_context(self, previous_best, measured, profiled, candidates=(),
            agent_selection=None, selection=None):
        outcomes = []
        current_trials = {row['trial_id']: row for row in self.store.trials()}
        for original in candidates:
            trial = current_trials[original['trial_id']]
            request = trial['plan']['backend_config']
            prediction = trial.get('prediction') or {}
            outcomes.append(dict(trial_id=trial['trial_id'],
                configuration={name: request.get(name) for name in
                    ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'micro_batch',
                     'group', 'decoder_chunks', 'recompute', 'execution_backend')},
                changed_fields=sorted((trial.get('graph_delta') or {}).get('config_changes', {})),
                prediction_id=prediction.get('prediction_id'),
                prediction_status=prediction.get('prediction_status'),
                load_validation=self.load_validation_summary(trial),
                step_time_ms=prediction.get('performance', {}).get('step_time_ms'),
                memory_admission=prediction.get('memory', {}).get('admission'),
                missing_cost_count=len(prediction.get('evidence', {}).get('missing_profiles', [])),
                cost_status_counts=prediction.get('evidence', {}).get('cost_status_counts', {})))
        changes = {name: value for name, value in (selection or {}).items()
            if value != (agent_selection or {}).get(name)}
        revisions = (self.execution_revision_reports(previous_best,
            [current_trials[row['trial_id']] for row in candidates]) if self.config.single_decision_round else [])
        return self.shared_context(previous_best) | dict(previous_best_trial_id=previous_best['trial_id'],
            execution_revisions=revisions,
            measured=[self.candidate_summary(trial) for trial in measured], profiling_results=profiled,
            candidate_decision_report=candidate_decisions(previous_best,
                [current_trials[row['trial_id']] for row in candidates]),
            candidate_outcomes=outcomes,
            agent_selection=agent_selection, execution_selection_changes=changes,
            recent_history=self.compact_history())

    @staticmethod
    def performance_instructions():
        return ('依据真实测量、图依赖与已有预测提出性能假设。引用提供的evidence_id，明确缺少的测量。'
            '禁止编造耗时、收益或数值认证。使用上下文已经提供的关键路径、Stage/Rank摘要、稳定Task ID和DAG证据。'
            'cost_profile给出实际测量配置；target_plan_matches_profile为false时，source_profile摘要属于其原始配置，'
            '当前布局的阶段判断需要相容成本投影或新的测量支持。'
            'diagnostic_evidence.level为STRUCTURAL_AND_MEASURED_GLOBAL时，图只支持结构、依赖、通信组和共驻关系判断；'
            '任务级性能优先级必须写为HYPOTHESIS，并在requested_metrics中列出需要的成本或训练测量。'
            'diagnostic_evidence.level为PREDICTED_EXECUTION时，才可以将完整资源回放的关键路径写为PREDICTED。'
            '需要确认跨Rank、跨Stage依赖时使用DAG查询工具；完成调查后将tool_requests设为空列表。')

    @staticmethod
    def strategy_instructions():
        return ('提交conditional_intent联合调查，明确全局并行家族、开放变量与允许范围；完整配置使用complete_strategy。'
            '按调查优先级排列策略；剩余候选扩展配额依此分配。'
            '结合model_structure、实测Stage成本和显存占用，比较重计算、Microbatch、全局并行与模块布局的联合改变。'
            '实测显存余量只适用于已经测量的完整配置。减少重计算、增加Microbatch或改变共驻布局后，'
            '必须重新检查激活保存与缓冲区需求；原配置的空闲显存不能证明修改后的容量可行性。'
            '实测配置能在显存约束内关闭重计算且未发生激活显存不足时，优先从该配置探索通信、调度和计算粒度；'
            '仅在容量证据支持时开启重计算，并选择能满足显存约束的最小范围。'
            '梯度缓冲区初始化或优化器步骤的显存失败，应优先调整对应的参数、梯度或优化器状态占用。'
            '比较本次已测成功与失败配置，提出能够区分计算收益、调度变化和容量变化的联合假设；'
            '缺少相容成本时，可以请求实际成本测量，明确哪些候选暂时无法按预测性能排序。'
            '有限候选配额应覆盖具有不同执行机制的策略；相近布局的比较需要给出具体计算、通信或显存依据。'
            '减小PP时核查每个GPU的参数、Optimizer、激活和新增DP通信；MoE同时核查EP与ETP的专家归属和All-to-All。'
            '改变TP或设备映射时比较实际进程组的通信位置与计算粒度。结构距离描述探索覆盖，不代表性能收益。'
            '完整策略的values填写tp、pp、cp、dp、ep、etp、vpp、micro_batch、group、decoder_chunks、recompute、'
            'pipeline_schedule、overlap_p2p、rank_mapping、execution_backend和optimizer_sharding。'
            'decoder_chunks必须包含pp*vpp个正整数，并按模型原始模块顺序完整覆盖全部层。'
            '改变PP或VPP时重新给出完整分区；改变DP或Microbatch时重新检查固定Global Batch对应的微批数量。'
            '完整策略可以直接提出原生支持的全局组合，family_id允许使用null；程序依据完整取值核验合法性并计算候选身份。'
            '联合调查使用adjustable声明开放变量，用allowed_ranges给出范围，并通过preserve保留必要设置。'
            '同一字段只声明一次：开放字段通过allowed_ranges给值，values对应字段使用null，preserve中不重复。'
            '完整策略的adjustable、preserve、allowed_ranges和module_relations保持为空。'
            '需要补充决策依据时，可以查询preview_transformation或query_affected_decisions。'
            '已经取得取值一致的预览时，在interaction_evidence_ids引用其真实收据；'
            '否则保留空列表，后端将对最终完整取值执行真实预览并绑定回执，参数取值保持原样。'
            'validate_agent_strategy的校验反馈使用evidence_refs引用。'
            'expected_effect说明预期缩短的等待、降低的容量压力或改善的计算粒度；'
            'possible_side_effects说明可能转移的Stage、通信组或显存峰值。'
            'conditional_failure_memory中的失败仅在相同模型、训练语义、硬件和相关并行条件下可复用；'
            '遇到OOM时，提出包含主导存储对象和关联变量的联合修复，不得重复同一完整生效配置。'
            '依据失败发生的加载、前向、反向或更新阶段选择修复对象；结合实际支持范围自主选择后端与状态分片。'
            '区分已经修复的执行错误和当前仍成立的容量限制，引用对应源码版本及逐Rank记录。'
            '同一批完整策略会共同提供接续布局：相同生效配置下的不同有序模块安排可用于比较与重组。'
            '依据模块结构、通信和保存需求解释各项候选之间的机制差别，选择有调查价值的完整安排。'
            'compare_candidate_graph_delta同时返回已有预测中的逐Rank工作与显存变化，先检查成本版本和容量状态。'
            'measured_strategy_memory保存本次同负载的最好实测、不同并行家族和失败条件。'
            '使用preview_transformation返回的execution_effects比较专家归属、物理通信组和微批数量。'
            '说明哪些设置具有直接实测依据，哪些沿用设置仍缺少替代取值的对照；'
            '利用strategy_decision_evidence共同选择可重新调查的变量，解释新增代价和预期收益；'
            '发现代价缺少补偿或违反已知容量条件时，修改完整策略并再次预览。'
            '联合测量结果适用于该完整配置，不得把全部差异归因于单个参数。'
            '查询或预览具体修改后，在interaction_evidence_ids引用返回的真实交互证据。'
            'hypothesis每项不超过300字，批次reason不超过500字；已有执行证据使用标识引用。'
            '训练后端、Optimizer分片和数值精度保持父策略定义的可执行范围。'
            '查询阶段允许intents为空；最终提交时tool_requests为空。')

    def validate_strategy_batch(self, strategy, parent_trial):
        for intent in strategy.intents:
            try:
                self.validate_intent_evidence(intent)
            except AgentConstraintConflict as error:
                raise ValueError(dict(error.result, status='REVISION_REQUIRED')) from error
        if (not strategy.intents or (len(strategy.intents) > self.config.intent_count
                if self.config.single_decision_round else len(strategy.intents) != self.config.intent_count)):
            raise ValueError(dict(status='REVISION_REQUIRED', conflicts=['intent_count'],
                expected=self.config.intent_count, actual=len(strategy.intents), reason=strategy.reason))
        parent_request = parent_trial['plan']['backend_config']
        signatures = {}
        plan_sets = {}
        legal_counts = {}
        capacity_checks = {}
        global_family_intents = []
        joint_intents = []
        complete_strategy_changes = []
        legal_changes = []
        parent_signature = plan_signature(parent_request)
        counterfactual_refs = set()
        known_counterfactual = {identifier for identifier, encoded in self.store.connection.execute(
            'SELECT id, document FROM evidence') if self.is_interaction_evidence(json.loads(encoded))}
        family_fields = SeedPlanner.family_fields()
        for index, intent in enumerate(strategy.intents):
            try:
                executable = self.resolve_agent_intent(intent, parent_request)
            except ValueError as error:
                legal_counts[index] = dict(count=0, conflict=error.args[0])
                plan_sets[index] = set()
                continue
            semantic = dict(values=executable.values, adjustable=sorted(executable.adjustable),
                preserve=sorted(executable.preserve), allowed_ranges=executable.allowed_ranges,
                module_relations=[row.model_dump(mode='json') for row in executable.module_relations],
                proposed_actions=[dict(operation=row['operation'], values=row['values'])
                    for row in executable.proposed_actions])
            signatures.setdefault(fingerprint(semantic), []).append(index)
            previous_values = normalized_execution(parent_request)
            proposed_values = normalized_execution(dict(parent_request, **executable.values))
            active_values = {key: value for key, value in executable.values.items()
                if previous_values.get(key, parent_request.get(key))
                != proposed_values.get(key, value)}
            if (len(active_values) > 1 or len(executable.proposed_actions) > 1
                    or len(executable.adjustable) > 1):
                joint_intents.append(index)
            if intent.family_id is not None or intent.proposal_kind == 'complete_strategy':
                global_family_intents.append(index)
            elif any(key in active_values for key in family_fields):
                legal_counts[index] = dict(count=0, conflict=dict(
                    status='GLOBAL_FAMILY_ID_REQUIRED', conflicts=sorted(
                        set(active_values).intersection(family_fields))))
                plan_sets[index] = set()
                continue
            counterfactual_refs.update(executable.interaction_evidence_ids)
            try:
                if intent.proposal_kind == 'complete_strategy':
                    from search.complete_strategy import compile_complete_strategy
                    rewrites = [compile_complete_strategy(self.rewriter, parent_request, executable)]
                else:
                    family_values = dict(self.global_family_values(parent_request),
                        **{key: value for key, value in executable.values.items() if key in GLOBAL_FIELDS})
                    completion = self.solve_joint_scope(parent_request,
                        dict(family_id=global_identity(dict(parent_request, **family_values)), values=family_values),
                        executable, self.config.candidate_limit)
                    rewrites = completion['candidates']
                    if not rewrites:
                        legal_counts[index] = dict(count=0, conflict=dict(
                            status=('PREDICTED_CAPACITY_REJECTED' if completion.get('capacity_rejections')
                                else 'EMPTY_DECLARED_SCOPE'), reason=completion.get('termination'),
                            capacity_rejections=completion.get('capacity_rejections', []),
                            layout_rejections=completion.get('layout_rejections', [])[:4],
                            policy_rejections=[rejection for layout in
                                completion['policy_frontier']['layouts'] for rejection in layout['rejected_policies']][:4]))
                        plan_sets[index] = set()
                        continue
                plan_sets[index] = {self.plan(row['request'], parent_trial['trial_id']).plan_hash
                    for row in rewrites}
            except ValueError as error:
                legal_counts[index] = dict(count=0, conflict=error.args[0])
                plan_sets[index] = set()
                continue
            capacity_checks[index] = []
            for row in rewrites:
                if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                    capacity_checks[index].append(dict(admission='NOT_EVALUATED_BUDGET'))
                    continue
                trial = self.predict_candidate(row, parent_trial, intent)
                memory = trial['prediction']['memory']
                largest = sorted(memory.get('peak_per_rank_bytes', {}),
                    key=lambda rank: memory['peak_per_rank_bytes'][rank], reverse=True)[:2]
                components = memory.get('breakdown_per_rank', {})
                capacity_checks[index].append(dict(trial_id=trial['trial_id'],
                    prediction_id=trial['prediction']['prediction_id'],
                    capacity_per_gpu_bytes=int(self.topology.memory_per_gpu_gb * 2**30),
                    **{key: memory[key] for key in ('admission', 'persistent_per_rank_bytes',
                        'peak_per_rank_bytes', 'memory_budget_per_rank_bytes', 'rejected_ranks',
                        'rejection_basis') if key in memory},
                    largest_rank_components={rank: {key: value for key, value in
                        components.get(rank, components.get(str(rank), {})).items()
                        if key in ('parameter_bytes', 'gradient_bytes', 'optimizer_bytes',
                            'persistent_bytes', 'host_optimizer_bytes', 'measured_persistent_and_runtime_bytes')}
                        for rank in largest}))
            if any(plan_signature(row['request']) != parent_signature for row in rewrites):
                legal_changes.append(index)
                if intent.proposal_kind == 'complete_strategy':
                    complete_strategy_changes.append(index)
            legal_counts[index] = dict(count=len(plan_sets[index]), conflict=None)
        duplicate_intents = [indices for indices in signatures.values() if len(indices) > 1]
        equivalent_intents = []
        for left, right in combinations(range(len(strategy.intents)), 2):
            if plan_sets[left] and plan_sets[left] == plan_sets[right]:
                equivalent_intents.append([left, right])
        legal_global = [index for index in global_family_intents if plan_sets[index]]
        if duplicate_intents:
            raise ValueError(dict(status='REVISION_REQUIRED', conflicts=['duplicate_intents'],
                intent_groups=duplicate_intents,
                reason='这些意图具有相同的变换语义，需要生成不同的策略',
                revision_owner='JointStrategyAgent'))
        if equivalent_intents:
            raise ValueError(dict(status='REVISION_REQUIRED', conflicts=['equivalent_plan_sets'],
                intent_groups=equivalent_intents,
                reason='这些意图生成相同的可执行计划集合，需要改变并行配置或执行布局',
                revision_owner='JointStrategyAgent'))
        if not legal_changes and not any(plan_sets.values()):
            raise ValueError(dict(status='NO_EXECUTABLE_GLOBAL_FAMILY_INTENT',
                global_family_intents=global_family_intents, legal_counts=legal_counts))
        if not legal_changes:
            raise ValueError(dict(status='REVISION_REQUIRED', conflicts=['no_executable_strategy_change'],
                legal_counts=legal_counts, revision_owner='JointStrategyAgent',
                reason='策略批次没有产生新的可执行安排'))
        if capacity_checks and all(checks and all(check['admission'] == 'REJECT_OOM'
                for check in checks) for checks in capacity_checks.values()):
            feedback = dict(status='REVISION_REQUIRED', revision_owner='JointStrategyAgent',
                conflicts=['all_proposed_strategies_exceed_capacity'],
                capacity_checks=capacity_checks,
                reason='全部完整提议超过容量，返回逐项容量证据后修改联合策略。',
                instruction='核对专家参数实际归属与状态分片范围，联合考虑TP、PP、EP、ETP、DP、'
                    'Optimizer卸载及模块分配。由Agent重新提交完整取值，保留训练负载和精度。')
            self.store.add_evidence(fingerprint(feedback), 'FAILURE', feedback)
            raise ValueError(feedback)
        if (self.config.agent_execution_enabled and self.config.evaluation_mode != 'config_only'
                and (not counterfactual_refs or not counterfactual_refs <= known_counterfactual)):
            raise ValueError(dict(status='REVISION_REQUIRED',
                conflicts=['counterfactual_evidence_reference'],
                submitted_evidence_ids=sorted(counterfactual_refs),
                allowed_evidence_ids=sorted(known_counterfactual),
                reason='策略需要在interaction_evidence_ids中引用真实变换预览或交互查询的证据标识',
                revision_owner='JointStrategyAgent'))
        return dict(intent_signatures={key: value for key, value in signatures.items()},
            legal_counts=legal_counts, capacity_checks=capacity_checks, duplicate_intents=duplicate_intents,
            equivalent_intents=equivalent_intents, joint_intents=joint_intents, legal_changes=legal_changes,
            complete_strategy_changes=complete_strategy_changes,
            global_family_intents=global_family_intents,
            legal_global_family_intents=legal_global,
            counterfactual_evidence_refs=sorted(counterfactual_refs))

    def bind_strategy_preview_evidence(self, strategy, strategy_id):
        references = set(strategy.evidence_refs)
        for intent in strategy.intents:
            references.update(intent.evidence_refs)
            if intent.target_bottleneck is not None:
                references.update(intent.target_bottleneck.evidence_refs)
        for identifier in references:
            if self.store.connection.execute('SELECT 1 FROM evidence WHERE id=?', (identifier,)).fetchone():
                continue
            record = self.store.connection.execute('SELECT document FROM trials WHERE id=?',
                (identifier,)).fetchone()
            if record is not None:
                trial = json.loads(record[0])
                self.store.add_evidence(identifier, 'PREDICTED', dict(
                    kind='registered_candidate_record', trial_id=identifier, plan_hash=trial['plan_hash'],
                    plan=self.compact_plan(trial),
                    prediction_id=(trial.get('prediction') or {}).get('prediction_id'),
                    source='actual_backend_experiment_store'))
        evidence = {identifier: json.loads(document) for identifier, document in
            self.store.connection.execute('SELECT id, document FROM evidence WHERE '
                'json_extract(document, "$.tool.tool") = "preview_transformation"')}
        for receipt in list(evidence.values()):
            body = receipt.get('result', {})
            if body.get('status') == 'PAGE_TOO_LARGE':
                identifier = body['complete_result_id']
                record = self.store.connection.execute(
                    'SELECT document FROM evidence WHERE id=?', (identifier,)).fetchone()
                if record is None:
                    raise ValueError('策略预览缺少完整证据记录')
                evidence[identifier] = json.loads(record[0])
        intents, bindings = list(strategy.intents), []
        for index, intent in enumerate(intents):
            if intent.interaction_evidence_ids and all(identifier in evidence
                    and self.preview_attests_strategy(intent, evidence[identifier], evidence)
                    for identifier in intent.interaction_evidence_ids):
                continue
            matches = sorted(identifier for identifier, receipt in evidence.items()
                if self.preview_attests_strategy(intents[index], receipt, evidence)
                and receipt.get('result', {}).get('status') != 'CONSTRAINT_CONFLICT')
            if not matches:
                started = time.perf_counter()
                preview = self.investigate(InvestigationToolRequest(tool='preview_transformation',
                    transformation_intent=intent, limit=6), self.store.trials())
                identifier = preview['evidence_id']
                record = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                    (identifier,)).fetchone()
                if record is None:
                    raise ValueError('完整策略预览没有登记真实收据')
                evidence[identifier] = json.loads(record[0])
                body = evidence[identifier]['result']
                if body.get('status') == 'PAGE_TOO_LARGE':
                    complete_id = body['complete_result_id']
                    complete = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                        (complete_id,)).fetchone()
                    if complete is None:
                        raise ValueError('完整策略预览缺少分页原始记录')
                    evidence[complete_id] = json.loads(complete[0])
                    body = evidence[complete_id]['result']
                if (body.get('status') != 'CONSTRAINT_CONFLICT'
                        and self.preview_attests_strategy(intent, evidence[identifier], evidence)):
                    matches = [identifier]
                charge_id = 'final-strategy-preview:' + identifier
                if self.store.charged(charge_id) is None:
                    self.store.charge(charge_id, 'cpu_seconds', time.perf_counter() - started)
                preview_origin = dict(kind='final_strategy_backend_preview',
                    original_agent_strategy_id=strategy_id, intent_index=index,
                    preview_evidence_id=identifier, decision_values_changed=False)
                self.store.add_evidence(fingerprint(preview_origin), 'PREDICTED', preview_origin)
            if matches:
                intents[index] = intents[index].model_copy(update=dict(interaction_evidence_ids=[matches[0]]))
                bindings.append(dict(intent_index=index, preview_evidence_id=matches[0],
                    strategy_effect_identity=self.strategy_effect_identity(intents[index])))
        if not bindings:
            return strategy, strategy_id
        compiled = strategy.model_copy(update=dict(intents=intents))
        compiled_id = fingerprint(compiled.model_dump(mode='json'))
        self.store.add_evidence(compiled_id, 'PREDICTED', compiled.model_dump(mode='json'))
        receipt = dict(kind='strategy_preview_evidence_binding',
            original_agent_strategy_id=strategy_id, compiled_strategy_id=compiled_id,
            decision_values_changed=False, bindings=bindings)
        self.store.add_evidence(fingerprint(receipt), 'PREDICTED', receipt)
        return compiled, compiled_id

    def validate_or_repair_strategy(self, strategy, strategy_id, parent_trial,
            diagnosis, diagnosis_id, global_family_search):
        strategy, strategy_id = self.bind_strategy_preview_evidence(strategy, strategy_id)
        try:
            validation = self.validate_strategy_batch(strategy, parent_trial)
            return strategy, strategy_id, validation
        except ValueError as error:
            feedback = error.args[0]
            if not isinstance(feedback, dict) or feedback.get('status') not in (
                    'NO_EXECUTABLE_GLOBAL_FAMILY_INTENT', 'REVISION_REQUIRED'):
                raise
        if self.config.single_decision_round:
            return strategy, strategy_id, dict(status='REJECTED', native_validation=feedback)
        compact_intents = [dict(index=index, family_id=intent.family_id,
            values=intent.values.model_dump(exclude_none=True), adjustable=intent.adjustable,
            preserve=intent.preserve, interaction_evidence_ids=intent.interaction_evidence_ids)
            for index, intent in enumerate(strategy.intents)]
        selected_family_ids = {intent.family_id for intent in strategy.intents
            if intent.family_id is not None}
        planner = SeedPlanner(self.rewriter, self.profile)
        family_base_plans = []
        base_fields = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping',
            'vpp', 'micro_batch', 'group', 'decoder_chunks', 'recompute',
            'pipeline_schedule', 'overlap_p2p', 'execution_backend', 'megatron_fsdp_version',
            'optimizer_sharding', 'dense_sharding_strategy', 'expert_sharding_strategy',
            'optimizer_offload_fraction', 'optimizer_transfer_overlap', 'precision_aware_optimizer')
        for family in planner.global_families(parent_trial['plan']['backend_config']):
            identifier = self.global_family_id(family['representative_request'])
            if identifier in selected_family_ids:
                family_base_plans.append(dict(family_id=identifier,
                    values={name: family['representative_request'][name] for name in base_fields}))
        observation = dict(role_context='joint_strategy_constraint_repair_v1',
            model_module_sequence=[dict(module_id=index, kind=kind.value)
                for index, kind in enumerate(self.model.layer_types)],
            current_plan=self.compact_plan(parent_trial),
            required_parent_plan_id=parent_trial['plan']['plan_id'],
            diagnosis=dict(hypotheses=diagnosis.hypotheses,
                affected_tasks=diagnosis.affected_tasks, evidence_refs=diagnosis.evidence_refs),
            diagnosis_evidence_id=diagnosis_id,
            backend_domains=self.domains,
            cross_shape_investigations=self.graph_investigations(parent_trial),
            intent_count=self.config.intent_count,
            global_family_search={key: global_family_search.get(key) for key in
                ('total_legal_families', 'family_fields', 'catalog_access', 'evidence_id')},
            family_base_plans=family_base_plans[:8],
            rejected_intents=compact_intents, constraint_feedback=feedback,
            valid_evidence_ids=[row[0] for row in self.store.connection.execute(
                'SELECT id FROM evidence ORDER BY rowid DESC LIMIT 24')])
        instructions = (self.strategy_instructions() +
            '修正constraint_feedback指定的字段。范围式意图选择合法family_id，完整策略的family_id使用null，'
            '所有意图的parent_plan_id必须逐字复制required_parent_plan_id。'
            'family_base_plans提供结构参照。完整策略自主选择全部最终取值，family_id使用null，'
            '根据原生支持条件选择相容的VPP、Microbatch、Group、Pipeline Schedule和全部模块区间。'
            '完整提交的值由Agent明确填写，程序只执行核验。'
            '没有范围限制的字段使用null，allowed_ranges保持对象结构。')

        def complete_constraint_owned_fields(batch):
            family_values = {row['family_id']: set(row['values']) for row in family_base_plans}
            completed = []
            for intent in batch.intents:
                if intent.proposal_kind == 'complete_strategy':
                    completed.append(intent)
                    continue
                values = intent.values.model_dump()
                for name in family_values.get(intent.family_id, set()):
                    values[name] = None
                completed.append(intent.model_copy(update=dict(
                    parent_plan_id=parent_trial['plan']['plan_id'],
                    values=RewriteValues.model_validate(values))))
            return batch.model_copy(update={'intents': completed})

        repaired, repaired_id = self.invoke_with_tools(
            'JointStrategyAgent', instructions, observation, IntentBatch)
        repaired = complete_constraint_owned_fields(repaired)
        repaired, repaired_id = self.bind_strategy_preview_evidence(repaired, repaired_id)
        for repair_round in range(10):
            try:
                validation = self.validate_strategy_batch(repaired, parent_trial)
                return repaired, repaired_id, validation
            except ValueError as error:
                feedback = error.args[0]
                if (repair_round == 9 or not isinstance(feedback, dict)
                        or feedback.get('status') not in (
                            'NO_EXECUTABLE_GLOBAL_FAMILY_INTENT', 'REVISION_REQUIRED')):
                    raise
                observation['rejected_intents'] = [dict(index=index,
                    family_id=intent.family_id,
                    values=intent.values.model_dump(exclude_none=True),
                    adjustable=intent.adjustable, preserve=intent.preserve,
                    interaction_evidence_ids=intent.interaction_evidence_ids)
                    for index, intent in enumerate(repaired.intents)]
                observation['constraint_feedback'] = feedback
                repaired, repaired_id = self.invoke_with_tools(
                    'JointStrategyAgent', instructions, observation, IntentBatch)
                repaired = complete_constraint_owned_fields(repaired)
                repaired, repaired_id = self.bind_strategy_preview_evidence(repaired, repaired_id)
        raise RuntimeError('联合策略修复循环异常终止')

    def trial_observation(self, trial):
        prediction = trial['prediction']
        projected = None
        if prediction is not None:
            projected = {key: prediction[key] for key in
                ('prediction_id', 'validation', 'performance', 'memory', 'diagnosis', 'evidence')
                if key in prediction}
            projected['memory'] = {key: value for key, value in prediction['memory'].items()
                if key in ('admission', 'peak_per_rank_bytes', 'memory_budget_per_rank_bytes',
                    'memory_headroom_per_rank_bytes', 'oom_risk', 'scope', 'missing_profiles')}
            projected['diagnosis'] = {key: value for key, value in prediction['diagnosis'].items()
                if key in ('critical_tasks', 'critical_regions', 'simulated_step_time_ms',
                    'pipeline_idle_ms', 'modeled_exposed_communication_ms')}
            graph = prediction.get('execution_graph')
            if graph is not None:
                critical = prediction['diagnosis'].get('critical_tasks', [])[:16]
                records = (graph.get('tasks') or graph.get('critical_task_records', {}))
                projected['critical_path_graph'] = dict(task_count=graph['task_count'],
                    tasks={key: records[key] for key in critical if key in records})
        measurement = trial['measurement']
        measured = None if measurement is None else {key: measurement[key] for key in
            ('run_id', 'status', 'performance_eligible', 'step_seconds', 'tokens_per_second',
             'peak_allocated_gib', 'numerical_equivalence_certified', 'numerical_equivalence_scope') if key in measurement}
        delta = trial['graph_delta']
        if delta is not None:
            delta = {key: value[:16] if isinstance(value, list) else value for key, value in delta.items()}
            delta['full_list_counts'] = {key: len(value) for key, value in trial['graph_delta'].items()
                if isinstance(value, list)}
            if 'impact' in delta:
                delta['impact'] = {key: value[:16] if isinstance(value, list) else value
                    for key, value in delta['impact'].items()}
        return dict(trial_id=trial['trial_id'], plan_hash=trial['plan_hash'],
            request=trial['plan']['backend_config'], graph_delta=delta,
            hypothesis=trial['hypothesis'], prediction=bounded_observation(projected), measurement=measured,
            state=trial['state'])

    def evidence_observation(self):
        rows = []
        for identifier, kind, encoded in self.store.connection.execute('SELECT * FROM evidence'):
            document = json.loads(encoded)
            if document.get('tool', {}).get('tool') not in (None, *self.available_query_tools()):
                continue
            if (document.get('source') == 'actual_dag_dependencies_and_decision_impact_index'
                    and self.config.evaluation_mode in ('config_only', 'graph_aware', 'graph_impact')):
                continue
            if document.get('kind') == 'measured_cost_profile' and self.config.evaluation_mode == 'config_only':
                continue
            if kind in ('PROJECTED', 'PREDICTED') and 'performance' in document:
                document = {key: document[key] for key in
                    ('prediction_id', 'prediction_status', 'validation', 'performance') if key in document}
            elif kind == 'MEASURED':
                if document.get('kind') == 'measured_cost_profile':
                    document = {key: document[key] for key in
                        ('kind', 'id', 'hardware', 'source_hashes', 'layers') if key in document}
                else:
                    document = {key: document[key] for key in
                        ('run_id', 'status', 'performance_eligible', 'step_seconds',
                         'tokens_per_second', 'peak_allocated_gib', 'numerical_equivalence_certified', 'numerical_equivalence_scope') if key in document}
            rows.append(dict(evidence_id=identifier, kind=kind,
                document=self.visible_observation(bounded_observation(document))))
        return rows

    @staticmethod
    def compact_task(task_id, task):
        metadata_fields = ('physical_rank', 'group_ranks', 'physical_group_ranks', 'shape', 'dtype',
            'channel', 'direction', 'producer_task', 'input_task', 'layer_ids', 'vp_stage', 'cuda_stream')
        return dict(task_id=task_id, rank=task['rank'], microbatch=task['microbatch'],
            operation=task['operation'], region_id=task['region_id'],
            dependency_ids=task['dependencies'], flow_id=task.get('flow_id'),
            cost=task.get('cost'), cost_status=task.get('cost_status'),
            evidence_refs=task.get('evidence_refs', []),
            metadata={key: task['metadata'][key] for key in metadata_fields if key in task['metadata']})

    def dag_page(self, graph, request):
        successors = {task_id: [] for task_id in graph['tasks']}
        for task_id, task in graph['tasks'].items():
            for dependency in task['dependencies']:
                successors[dependency].append(task_id)
        if request.task_ids:
            selected = set(request.task_ids)
            frontier = set(request.task_ids)
            for _ in range(request.dependency_depth):
                predecessors = {dependency for task_id in frontier
                    for dependency in graph['tasks'][task_id]['dependencies']}
                following = {successor for task_id in frontier for successor in successors[task_id]}
                if request.dependency_direction == 'predecessors':
                    adjacent = predecessors
                elif request.dependency_direction == 'successors':
                    adjacent = following
                else:
                    adjacent = predecessors | following
                frontier = adjacent - selected
                selected.update(adjacent)
            identities = sorted(selected, key=lambda key: (key not in request.task_ids, key))
        else:
            identities = sorted(task_id for task_id, task in graph['tasks'].items()
                if request.region_id is None or task['region_id'] == request.region_id)
        page_limit = min(request.limit, 8)
        page = identities[request.offset:request.offset + page_limit]
        page_set = set(page)
        selected_set = set(identities)
        internal_edges = [(dependency, task_id) for task_id in page
            for dependency in graph['tasks'][task_id]['dependencies'] if dependency in page_set]
        external_edges = [(dependency, task_id) for task_id in page
            for dependency in graph['tasks'][task_id]['dependencies'] if dependency not in page_set]
        external_edges.extend((task_id, successor) for task_id in page
            for successor in successors[task_id] if successor not in page_set)
        cross_rank_edges = [(dependency, task_id) for task_id in selected_set
            for dependency in graph['tasks'][task_id]['dependencies']
            if dependency in selected_set
            and graph['tasks'][dependency]['rank'] != graph['tasks'][task_id]['rank']]
        return dict(graph_sha256=graph['graph_sha256'], full_task_count=graph['task_count'],
            full_edge_count=graph['edge_count'], full_flow_count=graph['flow_count'],
            offset=request.offset, returned_count=len(page), total_matching_nodes=len(identities),
            next_offset=request.offset + len(page) if request.offset + len(page) < len(identities) else None,
            nodes={task_id: self.compact_task(task_id, graph['tasks'][task_id]) for task_id in page},
            internal_edges=internal_edges[:page_limit * 4],
            external_edge_refs=sorted(set(external_edges))[:page_limit * 4],
            external_edge_ref_count=len(set(external_edges)),
            expanded_cross_rank_edges=sorted(set(cross_rank_edges))[:page_limit * 2],
            expanded_cross_rank_edge_count=len(set(cross_rank_edges)),
            structural_source=graph['structural_evidence'], cost_evidence_refs=graph['cost_evidence_refs'])

    @staticmethod
    def paged_mapping(values, offset, limit):
        entries = sorted(values.items())
        page = entries[offset:offset + limit]
        return dict(total_count=len(entries), offset=offset, items=dict(page),
            next_offset=offset + len(page) if offset + len(page) < len(entries) else None)

    @staticmethod
    def paged_delta(delta, offset, limit):
        output = {}
        for key, value in delta.items():
            if isinstance(value, list):
                output[key] = dict(total_count=len(value), offset=offset,
                    items=value[offset:offset + limit],
                    next_offset=offset + min(limit, max(0, len(value) - offset))
                        if offset + limit < len(value) else None)
            elif isinstance(value, dict):
                output[key] = EvidenceGuidedSearch.paged_mapping(value, offset, limit)
            else:
                output[key] = value
        return output

    @staticmethod
    def bottleneck_decisions(operations):
        mapping = {
            'forward_compute': {'tp', 'cp', 'pp', 'vpp', 'decoder_chunks', 'micro_batch',
                'recompute', 'rank_mapping'},
            'backward_compute': {'tp', 'cp', 'pp', 'vpp', 'decoder_chunks', 'micro_batch',
                'recompute', 'rank_mapping'},
            'pp_p2p_send': {'pp', 'vpp', 'decoder_chunks', 'micro_batch', 'group',
                'overlap_p2p', 'rank_mapping', 'pipeline_schedule'},
            'pp_p2p_recv': {'pp', 'vpp', 'decoder_chunks', 'micro_batch', 'group',
                'overlap_p2p', 'rank_mapping', 'pipeline_schedule'},
            'dp_reduce': {'dp', 'expert_dp', 'bucket_elements', 'optimizer_sharding', 'rank_mapping'},
            'optimizer_step': {'dp', 'expert_dp', 'optimizer_sharding', 'recompute'},
            'cuda_sync': {'pipeline_schedule', 'optimizer_sharding', 'overlap_p2p'}}
        selected = set()
        for operation in operations:
            selected.update(mapping.get(operation, ()))
        return selected

    @staticmethod
    def decision_effects(name):
        compute_shape = {'tp', 'cp', 'ep', 'etp', 'micro_batch'}
        compute_placement = {'pp', 'vpp', 'decoder_chunks', 'rank_mapping'}
        communication = {'tp', 'cp', 'ep', 'etp', 'dp', 'expert_dp', 'bucket_elements',
            'rank_mapping', 'overlap_p2p'}
        readiness = {'pp', 'vpp', 'decoder_chunks', 'micro_batch', 'group',
            'pipeline_schedule', 'overlap_p2p', 'recompute'}
        return dict(changes_primary_cost=sorted(
            ({'compute_shape'} if name in compute_shape else set()) |
            ({'compute_placement'} if name in compute_placement else set()) |
            ({'communication_volume_or_path'} if name in communication else set())),
            changes_dependency_or_resource=sorted(
                ({'task_readiness_or_schedule'} if name in readiness else set()) |
                ({'rank_or_network_resource'} if name == 'rank_mapping' else set())))

    @staticmethod
    def compact_interaction_edges(edges, limit):
        output = []
        for edge in edges[:limit]:
            evidence = {}
            for key, value in edge['evidence'].items():
                evidence[key] = (dict(total_count=len(value), items=value[:8])
                    if isinstance(value, list) else value)
            output.append({key: edge[key] for key in ('left', 'right', 'kind', 'evidence_scope',
                'performance_synergy')} | dict(evidence=evidence))
        return output

    @staticmethod
    def compact_counterfactual(result):
        catalog = {}
        pairs = []
        for pair in result['pairs']:
            plan_ids = []
            for plan in pair.get('plans', []):
                plan_ids.append(plan['run_id'])
                catalog[plan['run_id']] = dict(plan=plan['plan'],
                    prediction_status=plan['prediction_status'],
                    step_time_ms=plan['step_time_ms'], intended_values=plan['intended_values'],
                    constraint_propagated_fields=sorted(plan['constraint_propagated_changes']),
                    graph_delta=plan['graph_delta'], cost_basis=plan['cost_basis'])
            pairs.append({key: value for key, value in pair.items() if key not in ('plans',)} |
                dict(plan_ids=plan_ids))
        return dict(source=result['source'], formula=result['formula'], pairs=pairs,
            plan_catalog=catalog)

    def investigate(self, request, candidates):
        policy = AblationEvidencePolicy(self.config.agent_ablation)
        withheld = policy.intercept(request, getattr(self, '_visible_trial_ids', set()))
        if withheld is not None:
            identifier = fingerprint(dict(tool=request.model_dump(mode='json'), result=withheld))
            receipt = dict(evidence_id=identifier, tool=request.model_dump(mode='json'), result=withheld,
                result_tokens=token_count(withheld), evidence_scope='ablation_evidence_visibility')
            self.store.add_evidence(identifier, 'PREDICTED', receipt)
            return receipt
        try:
            return self._investigate(request, candidates)
        except ValueError as error:
            conflict = error.args[0] if error.args else None
            if isinstance(conflict, dict) and conflict.get('status') == 'NEEDS_PROFILE':
                identifier = fingerprint(dict(tool=request.model_dump(), result=conflict))
                receipt = dict(evidence_id=identifier, tool=request.model_dump(), result=conflict,
                    result_tokens=token_count(conflict), token_encoding=TOKEN_ENCODING_NAME,
                    evidence_scope='actual_cost_evidence_availability')
                self.store.add_evidence(identifier, 'FAILURE', receipt)
                return receipt
            if not isinstance(conflict, dict) or conflict.get('status') != 'CONSTRAINT_CONFLICT':
                raise
            result = dict(conflict)
            result.setdefault('revision_owner', 'JointStrategyAgent')
            result.setdefault('reason', '确定性约束检查确认当前变换意图存在冲突')
            transformation = getattr(request, 'transformation_intent', None)
            result.setdefault('selected_family_id', transformation.family_id
                if transformation is not None else None)
            identifier = fingerprint(dict(tool=request.model_dump(), result=result))
            receipt = dict(evidence_id=identifier, tool=request.model_dump(), result=result,
                result_tokens=token_count(result), token_encoding=TOKEN_ENCODING_NAME,
                evidence_scope='deterministic_intent_constraint_validation')
            self.store.add_evidence(identifier, 'FAILURE', receipt)
            return receipt

    def _investigate(self, request, candidates):
        if request.tool not in self.available_query_tools():
            return dict(status='UNAVAILABLE', tool=request.tool,
                reason='当前上下文模式未提供该查询工具',
                available_tools=self.available_query_tools())
        if not request.candidate_ids:
            best = self.reference_trial()
            if best is not None and best['trial_id'] not in {trial['trial_id'] for trial in candidates}:
                candidates = candidates + [best]
        by_id = {trial['trial_id']: trial for trial in candidates}
        if not set(request.candidate_ids) <= set(by_id):
            raise AgentConstraintConflict(dict(conflicts=['candidate_ids'],
                reason='candidate_ids仅接受已有trial_id；全局配置查询使用family_ids或offset',
                unknown_candidate_ids=sorted(set(request.candidate_ids) - set(by_id)),
                available_candidate_ids=sorted(by_id)[:12],
                revision_owner='requesting_agent'))
        selected = [by_id[identifier] for identifier in request.candidate_ids]
        implicit_current_plan_tools = {'evaluate_candidate', 'inspect_critical_path',
            'inspect_region', 'inspect_communication', 'inspect_execution_graph',
            'evaluate_memory', 'inspect_memory', 'inspect_prediction_error'}
        if not selected and request.tool in implicit_current_plan_tools:
            reference = self.reference_trial()
            if reference is None:
                raise ValueError('候选调查工具缺少当前参考计划')
            selected = [reference]
        if request.tool in ('compare_candidate_graph_delta', 'compare_candidates', 'inspect_execution_impact') and (
                len(selected) != 2 or len(set(request.candidate_ids)) != 2):
            raise AgentConstraintConflict(dict(conflicts=['candidate_ids'],
                reason='比较操作需要两个不同的已登记trial_id；未登记策略使用preview_transformation预览。',
                supplied_candidate_ids=request.candidate_ids,
                available_candidate_ids=sorted(by_id)[:12], revision_owner='requesting_agent'))
        structural = ('inspect_training_graph', 'inspect_dag_subgraph', 'inspect_measured_trace', 'inspect_task_dependencies',
            'inspect_region_cost', 'inspect_communication_flow', 'inspect_rank_topology',
            'inspect_shared_resources', 'query_affected_decisions', 'preview_transformation',
            'compare_candidate_graph_delta', 'analyze_transformation_interactions', 'inspect_execution_impact', 'inspect_policy_frontier')
        if request.tool in structural:
            trial = selected[0] if selected else self.reference_trial()
            if trial is None:
                raise ValueError('图查询缺少真实参考计划')
            candidate = self.rewriter.candidate(trial['plan']['backend_config'])
            graph = self.backend_dag(trial)
            task_ids = set(graph['tasks'])
            if request.tool in ('inspect_dag_subgraph', 'inspect_task_dependencies'):
                task_ids = set(self.bottleneck_graph(trial)[0].records)
            if not set(request.task_ids) <= task_ids:
                raise AgentConstraintConflict(dict(conflicts=['task_ids'],
                    reason='图查询引用了不存在的稳定任务ID',
                    unknown_task_ids=sorted(set(request.task_ids) - task_ids),
                    available_task_ids=sorted(task_ids)[:16],
                    revision_owner='requesting_agent'))
            if request.tool == 'analyze_transformation_interactions':
                if len(request.transformation_intents) < 2:
                    raise AgentConstraintConflict(dict(conflicts=['transformation_intents'],
                        reason='变换交互查询需要至少两个原子意图', revision_owner='requesting_agent'))
                interactions = self.rewriter.compiler.transformation_interactions(candidate,
                    [self.resolve_agent_intent(intent, trial['plan']['backend_config'])
                        for intent in request.transformation_intents], request.task_ids)
                result = dict(source=interactions['source'], measured_interactions=interactions['measured_interactions'],
                    preview_count=len(interactions['previews']), edge_count=len(interactions['edges']),
                    previews=[dict(affected_task_count=len(row['affected_task_ids']),
                        affected_region_ids=row['affected_region_ids'],
                        affected_flow_ids=row['affected_flow_ids'][:8],
                        resource_ids=row['resource_ids'],
                        hard_constraint_dependencies=row['hard_constraint_dependencies'])
                        for row in interactions['previews']],
                    edges=self.compact_interaction_edges(
                        interactions['edges'][request.offset:], request.limit),
                    next_edge_offset=request.offset + request.limit
                        if request.offset + request.limit < len(interactions['edges']) else None,
                    counterfactual=self.compact_counterfactual(self.counterfactual_interactions(
                        trial, request.transformation_intents)))
            elif request.tool == 'preview_transformation':
                if request.transformation_intent is None:
                    raise AgentConstraintConflict(dict(conflicts=['transformation_intent'],
                        reason='Effect Preview需要完整结构化优化意图', revision_owner='requesting_agent'))
                preview = self.rewriter.preview_transformation(trial['plan']['backend_config'],
                    self.resolve_agent_intent(
                        request.transformation_intent, trial['plan']['backend_config']))
                result = self.paged_delta(preview, request.offset, request.limit)
                result['strategy_effect_identity'] = self.strategy_effect_identity(request.transformation_intent)
                if request.transformation_intent.proposal_kind == 'complete_strategy':
                    from search.complete_strategy import complete_request
                    resolved = self.resolve_agent_intent(
                        request.transformation_intent, trial['plan']['backend_config'])
                    try:
                        proposed = complete_request(trial['plan']['backend_config'], resolved.values)
                        result['execution_effects'] = proposed_strategy_structure(
                            trial['plan']['backend_config'], dict(resolved.values, offload=proposed['offload']),
                            self.topology.devices, self.model)
                        if 'expert_dp' not in resolved.values:
                            proposed.pop('expert_dp', None)
                        result['execution_effects']['native_execution_effects'] = self.native_strategy_summary(proposed)
                        result['graph_delta'] = self.compare_native_schedules(trial, proposed)
                    except ValueError as error:
                        raise AgentConstraintConflict(dict(conflicts=['complete_strategy'],
                            reason=error.args[0], revision_owner='requesting_agent',
                            instruction='核查完整并行形状、Expert DP、模块覆盖与微批整除关系后重新提交预览。')) from error
                    result['strategy_effect_identity'] = self.strategy_effect_identity(request.transformation_intent)
            elif request.tool == 'query_affected_decisions':
                index = self.rewriter.compiler.decision_impact_index(candidate)
                target_ids = set(request.task_ids)
                if not target_ids and request.region_id is not None:
                    target_ids = {task_id for task_id, task in graph['tasks'].items()
                        if task['region_id'] == request.region_id}
                operations = {graph['tasks'][task_id]['operation'] for task_id in target_ids}
                primary = self.bottleneck_decisions(operations)
                coupled = {dependency for name in primary for dependency in
                    index[name]['hard_constraint_dependencies'].get(name, []) if name in index}
                priority = {'PRIMARY': 0, 'CONSTRAINT_COUPLED': 1, 'GLOBAL_EXPLORATION': 2}
                matching = []
                for name, row in index.items():
                    if name in primary:
                        relevance = 'PRIMARY'
                    elif name in coupled:
                        relevance = 'CONSTRAINT_COUPLED'
                    else:
                        relevance = 'GLOBAL_EXPLORATION'
                    overlap = sorted(target_ids & set(row['affected_task_ids']))
                    matching.append((name, row, relevance, overlap))
                matching.sort(key=lambda item: (priority[item[2]], -len(item[3]), item[0]))
                page_limit = min(request.limit, 4)
                page = matching[request.offset:request.offset + page_limit]
                result = dict(total_count=len(matching), offset=request.offset,
                    next_offset=request.offset + len(page)
                        if request.offset + len(page) < len(matching) else None,
                    target_task_ids=sorted(target_ids)[:8], target_operations=sorted(operations),
                    decisions={name: dict(affected_task_count=len(row['affected_task_ids']),
                        relevance=relevance, target_overlap_count=len(overlap),
                        affected_target_task_ids=overlap[:4],
                        affected_region_ids=row['affected_region_ids'],
                        affected_flow_ids=row['affected_flow_ids'][:4],
                        resource_ids=row['resource_ids'],
                        hard_constraint_dependencies=row['hard_constraint_dependencies'],
                        **self.decision_effects(name),
                        constraint_propagation=row['hard_constraint_dependencies'].get(name, []),
                        potential_placement_changes=row['potential_placement_changes'],
                        potential_tensor_shape_changes=row['potential_tensor_shape_changes'],
                        potential_process_group_changes=row['potential_process_group_changes'],
                        potential_buffer_lifetime_changes=row['potential_buffer_lifetime_changes'],
                        source=row['source']) for name, row, relevance, overlap in page})
            elif request.tool == 'inspect_task_dependencies':
                if not request.task_ids:
                    raise AgentConstraintConflict(dict(conflicts=['task_ids'],
                        reason='任务依赖查询需要至少一个稳定Task ID',
                        available_task_ids=sorted(task_ids)[:16], revision_owner='requesting_agent'))
                query, query_scope = self.bottleneck_graph(trial)
                closure = query.closure(request.task_ids,
                    dependency_depth=request.dependency_depth, include_resources=True,
                    include_predecessors=request.dependency_direction in ('predecessors', 'both'),
                    include_successors=request.dependency_direction in ('successors', 'both'))
                result = query.page(closure, offset=request.offset,
                    limit=min(request.limit, 16), max_chars=9000)
                result['query_scope'] = query_scope
            elif request.tool == 'inspect_shared_resources':
                ranks = {graph['tasks'][key]['rank'] for key in request.task_ids}
                matching = [(key, task) for key, task in sorted(graph['tasks'].items()) if task['rank'] in ranks]
                page = matching[request.offset:request.offset + request.limit]
                result = dict(total_count=len(matching), offset=request.offset,
                    next_offset=request.offset + len(page)
                        if request.offset + len(page) < len(matching) else None,
                    tasks={key: self.compact_task(key, task) for key, task in page},
                    resources=[f'gpu:{rank}' for rank in sorted(ranks)], scope='actual_rank_ownership',
                    network_contention='requires_hardware_and_concurrency_measurements')
            elif request.tool == 'inspect_rank_topology':
                groups = process_groups(trial['plan']['backend_config'])
                result = dict(process_groups={name: dict(group_count=len(rows),
                    group_sizes=sorted({len(row) for row in rows}), samples=rows[:2])
                    for name, rows in groups['physical'].items()}, hardware=self.topology_summary(),
                    missing_topology=[] if self.topology.devices else ['GPU_UUID_PCIe_NUMA_NIC_paths'])
                if len(selected) == 2:
                    result = dict(execution_effects=compare_strategy_structure(
                        selected[0]['plan']['backend_config'], selected[1]['plan']['backend_config'],
                        self.topology.devices, self.model))
                else:
                    result['execution_structure'] = execution_structure(
                        trial['plan']['backend_config'], self.topology.devices, self.model)
                result['collective_tasks'] = [dict(trial_id=item['trial_id'],
                    **collective_summary(self.backend_dag(item), self.topology.num_gpus))
                    for item in (selected or [trial])]
            elif request.tool == 'inspect_execution_impact':
                result = dict(impacts=self.execution_revision_reports(selected[0], [selected[1]]))
            elif request.tool == 'inspect_policy_frontier':
                records = [json.loads(value) for value, in self.store.connection.execute(
                    "SELECT document FROM evidence WHERE json_extract(document,'$.kind')='policy_frontier' ORDER BY rowid DESC")]
                if request.family_ids:
                    records = [row for row in records if row['scope']['family_id'] in request.family_ids]
                if selected:
                    identities = {plan_signature(row['plan']['backend_config']) for row in selected}
                    records = [row for row in records if any(item['plan_signature'] in identities for item in row['evaluations'])]
                page = records[request.offset:request.offset + request.limit]
                result = dict(total_count=len(records), offset=request.offset, comparisons=[dict(
                    scope=row['scope'], layouts=row['layouts'], retention=row['retention'],
                    evaluations=[{key: item[key] for key in
                        ('plan_signature', 'policy', 'prediction_id', 'step_time_ms', 'status', 'missing_profiles')}
                        for item in row['evaluations']]) for row in page])
            elif request.tool == 'compare_candidate_graph_delta':
                delta = self.rewriter.compiler.graph_delta(
                    self.rewriter.candidate(selected[0]['plan']['backend_config']),
                    self.rewriter.candidate(selected[1]['plan']['backend_config']))
                result = self.paged_delta(delta, request.offset, min(request.limit, 4))
                comparisons = []
                for item in selected:
                    prediction = item.get('prediction')
                    existing_graph = self.load_prediction_graph(prediction) if prediction is not None else None
                    comparisons.append(dict(prediction, execution_graph=existing_graph)
                        if prediction is not None and existing_graph is not None else prediction)
                result['execution_comparison'] = compare_execution_reports(
                    selected[0]['plan']['backend_config'], comparisons[0],
                    selected[1]['plan']['backend_config'], comparisons[1],
                    offset=request.offset, limit=min(request.limit, 4))
                result['execution_effect_query'] = 'inspect_rank_topology with the same candidate_ids'
                result['native_pipeline_delta'] = schedule_delta(
                    selected[0]['plan']['backend_config'], selected[1]['plan']['backend_config'])
                if result['execution_comparison']['status'] == 'UNAVAILABLE':
                    result['execution_comparison'] = self.compare_native_schedules(selected[0],
                        selected[1]['plan']['backend_config'])['execution_comparison']
            elif request.tool == 'inspect_measured_trace':
                from trace.trace_mapper import TraceMapper
                profile_id = self.predictor.database.select(trial['plan']['backend_config'])
                profile = self.predictor.database.get(profile_id, trial['plan']['backend_config']) if profile_id else None
                if profile is None or not profile.get('source_directory'):
                    result = dict(status='UNAVAILABLE', measurement_type='training_trace',
                        reason='当前计划没有登记可查询的实际Trace目录')
                else:
                    trace = TraceMapper().map_runtime(profile['source_directory'])
                    events = [event for rank in trace['ranks'].values() for event in rank['tasks']
                        if (not request.task_ids or event['task_id'] in request.task_ids)
                        and (request.region_id is None or event['region_id'] == request.region_id)]
                    page_limit = min(request.limit, 4)
                    selected_events = events[request.offset:request.offset + page_limit]
                    result = dict(trace=dict(events=selected_events, total_matching_events=len(events),
                        offset=request.offset,
                        next_offset=request.offset + page_limit
                            if request.offset + page_limit < len(events) else None,
                        cross_rank_clock_alignment=trace['cross_rank_clock_alignment'],
                        cross_rank_wait_ms=trace['cross_rank_wait_ms'], source=trace['source'],
                        missing_measurements=trace['missing_measurements']),
                        measured_plan_signature=plan_signature(profile['request']),
                        candidate_plan_signature=plan_signature(trial['plan']['backend_config']),
                        exact_candidate_trace=profile['request']['decoder_chunks'] == trial['plan']['backend_config']['decoder_chunks'],
                        source='MEASURED', evidence_id=profile_id)
            elif request.tool == 'inspect_region_cost':
                matching = [(key, task) for key, task in sorted(graph['tasks'].items())
                    if (request.region_id is None or task['region_id'] == request.region_id)
                    and task['operation'] in
                    ('forward_compute', 'backward_compute')]
                page_limit = min(request.limit, 8)
                page = matching[request.offset:request.offset + page_limit]
                result = dict(region_id=request.region_id, total_count=len(matching), offset=request.offset,
                    next_offset=request.offset + len(page)
                        if request.offset + len(page) < len(matching) else None,
                    tasks={key: self.compact_task(key, task) for key, task in page},
                    source='MEASURED' if graph['cost_evidence_refs'] else 'UNRESOLVED',
                    evidence_refs=graph['cost_evidence_refs'])
            elif request.tool == 'inspect_communication_flow':
                matching = [(key, task) for key, task in sorted(graph['tasks'].items())
                    if task['flow_id'] is not None]
                if request.task_ids:
                    flows = {graph['tasks'][key]['flow_id'] for key in request.task_ids}
                    matching = [(key, task) for key, task in matching if task['flow_id'] in flows]
                page = matching[request.offset:request.offset + request.limit]
                result = dict(total_count=len(matching), offset=request.offset,
                    next_offset=request.offset + len(page)
                        if request.offset + len(page) < len(matching) else None,
                    tasks={key: self.compact_task(key, task) for key, task in page},
                    flow_ids=sorted({task['flow_id'] for key, task in page}),
                    source_graph_sha256=graph['graph_sha256'])
            elif request.tool == 'inspect_dag_subgraph' and request.task_ids:
                query, query_scope = self.bottleneck_graph(trial)
                closure = query.closure(request.task_ids,
                    dependency_depth=request.dependency_depth, include_resources=True,
                    include_predecessors=request.dependency_direction in ('predecessors', 'both'),
                    include_successors=request.dependency_direction in ('successors', 'both'))
                result = query.page(closure, offset=request.offset,
                    limit=min(request.limit, 16), max_chars=9000)
                result['query_scope'] = query_scope
                if query_scope == 'PREDICTED_COSTED_DAG':
                    charge_id = fingerprint(dict(graph=query.graph_id,
                        closure=closure['closure_id'], epsilon=0.1))
                    cached = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                        (charge_id,)).fetchone()
                    if cached is None:
                        impact = query.epsilon_impact(closure['task_ids'], 0.1,
                            resource_model=self.predictor.projector.parameters.get('resource_replay', {}))
                        self.store.add_evidence(charge_id, 'PREDICTED', impact)
                        self.store.charge('impact:' + charge_id, 'counterfactual_replays', 2)
                    else:
                        impact = json.loads(cached[0])
                    result['counterfactual_impact'] = {key: impact[key] for key in (
                        'epsilon', 'baseline_step_time_ms', 'counterfactual_step_time_ms',
                        'impact_ms', 'evidence_type', 'measured_strategy_gain', 'scope')}
                    result['counterfactual_impact'].update(evidence_id=charge_id,
                        selected_task_count=len(closure['task_ids']))
            elif request.tool in ('inspect_training_graph', 'inspect_dag_subgraph'):
                result = self.dag_page(graph, request)
                if request.tool == 'inspect_training_graph':
                    operators = build_operator_graph(self.model, candidate.parallel_config)
                    entries = sorted(operators['nodes'].items())
                    operator_page = entries[request.offset:request.offset + min(request.limit, 4)]
                    result['operator_index'] = dict(total_count=len(entries), offset=request.offset,
                        nodes=dict(operator_page),
                        next_offset=request.offset + len(operator_page)
                            if request.offset + len(operator_page) < len(entries) else None,
                        source=operators['source'], cost_annotations=False)
            else:
                tasks = [(key, task) for key, task in graph['tasks'].items()
                    if task['operation'] in ('pp_p2p_send', 'pp_p2p_recv', 'dp_reduce')]
                result = dict(graph_sha256=graph['graph_sha256'], full_task_count=len(tasks),
                    tasks={key: self.compact_task(key, task)
                        for key, task in tasks[request.offset:request.offset + request.limit]},
                    opaque_operations=graph['opaque_operations'])
        elif request.tool == 'inspect_global_families':
            reference = self.reference_trial()
            if reference is None:
                raise ValueError('全局并行家族查询缺少当前参考计划')
            families = self.joint_global_families(reference['plan']['backend_config'])
            requested_families = getattr(request, 'family_ids', [])
            if requested_families:
                known_families = {row['family_id'] for row in families}
                if not set(requested_families) <= known_families:
                    raise AgentConstraintConflict(dict(conflicts=['family_ids'],
                        reason='family_ids必须引用当前合法全局配置目录中的标识',
                        unknown_family_ids=sorted(set(requested_families) - known_families)))
                families = [row for row in families if row['family_id'] in requested_families]
            records = {row['family_id']: row for row in self.family_ledger.records()}
            page = families[request.offset:request.offset + min(request.limit, 8)]
            result = dict(total_count=len(families), offset=request.offset,
                next_offset=request.offset + len(page)
                    if request.offset + len(page) < len(families) else None,
                family_fields=list(GLOBAL_FIELDS),
                families=[dict(family_id=row['family_id'], values=row['values'],
                    parent_distance=row['parent_distance'], state_variant_count=row['state_variant_count'],
                    state_variants=row['state_variants'],
                    status=records.get(row['family_id'], {}).get('status', 'UNKNOWN'),
                    candidate_count=len(records.get(row['family_id'], {}).get('candidate_ids', [])),
                    best_predicted=records.get(row['family_id'], {}).get('best_predicted'),
                    best_measured=records.get(row['family_id'], {}).get('best_measured'),
                    lower_bound_ms=records.get(row['family_id'], {}).get('lower_bound_ms'))
                    for row in page],
                source='deterministic_model_hardware_batch_and_backend_constraints')
        elif request.tool == 'get_budget_status':
            result = dict(limits=self.config.budget.model_dump(), usage=self.store.usage())
        elif request.tool == 'get_supported_transforms':
            result = self.registry.describe()
        elif request.tool == 'get_best_plan':
            measured = self.store.best_measured()
            result = dict(best_measured=self.candidate_summary(measured) if measured else None)
        elif request.tool in ('get_trial_history', 'inspect_trial_history'):
            trials = selected if request.candidate_ids else self.store.trials()
            page_limit = min(request.limit, 4)
            result = dict(total_count=len(trials), offset=request.offset,
                next_offset=request.offset + page_limit if request.offset + page_limit < len(trials) else None,
                trials=[self.trial_history_summary(trial)
                    for trial in trials[request.offset:request.offset + page_limit]],
                recent_search_rounds=self.compact_round_history())
            if self.config.agent_request_profile == 'compact':
                if request.history_view != 'summary':
                    result = dict(total_count=len(trials), offset=request.offset,
                        next_offset=request.offset + page_limit if request.offset + page_limit < len(trials) else None)
                result['configuration_structures'] = []
                notes = [note for note in reversed(self.store.notes())
                    if not request.candidate_ids or note['trial_id'] in request.candidate_ids]
                result['notes'] = notes[request.offset:request.offset + page_limit]
                result['notes_next_offset'] = (request.offset + page_limit
                    if request.offset + page_limit < len(notes) else None)
                for item in trials[request.offset:request.offset + page_limit]:
                    structure = execution_structure(item['plan']['backend_config'],
                        self.topology.devices, self.model)
                    result['configuration_structures'].append(dict(trial_id=item['trial_id'],
                        **{key: structure[key] for key in ('process_groups', 'module_ownership')
                            if key in structure}))
                if request.history_view != 'structure':
                    result.pop('configuration_structures')
                if request.history_view != 'notes':
                    result.pop('notes')
                    result.pop('notes_next_offset')
                if request.history_view == 'rounds':
                    result['recent_search_rounds'] = self.compact_round_history()
                elif request.history_view == 'summary':
                    result.pop('recent_search_rounds')
        elif not selected:
            raise ValueError('候选调查工具需要至少一个候选身份')
        elif request.tool == 'evaluate_candidate':
            result = {trial['trial_id']: self.candidate_summary(trial) for trial in selected}
        elif request.tool == 'inspect_critical_path':
            result = {trial['trial_id']: bounded_observation(trial['prediction']['diagnosis'], request.limit)
                if trial['prediction'] is not None else dict(status='NEEDS_PROFILE',
                    reason='当前参考计划缺少完整任务成本，尚未生成预测关键路径') for trial in selected}
        elif request.tool in ('inspect_region', 'inspect_communication'):
            result = {}
            for trial in selected:
                graph = self.backend_dag(trial)
                if request.tool == 'inspect_region' and request.region_id is None:
                    raise ValueError('Region调查需要指定region_id')
                tasks = [(key, task) for key, task in graph['tasks'].items()
                    if task['region_id'] == request.region_id] if request.tool == 'inspect_region' else [
                    (key, task) for key, task in graph['tasks'].items()
                    if task['operation'] in ('pp_p2p_send', 'pp_p2p_recv', 'dp_reduce')]
                result[trial['trial_id']] = dict(total_count=len(tasks),
                    offset=request.offset,
                    next_offset=request.offset + request.limit
                        if request.offset + request.limit < len(tasks) else None,
                    tasks={key: self.compact_task(key, task)
                        for key, task in tasks[request.offset:request.offset + request.limit]},
                    source_graph_sha256=graph['graph_sha256'])
        elif request.tool == 'inspect_execution_graph':
            result = {}
            for trial in selected:
                result[trial['trial_id']] = self.dag_page(self.backend_dag(trial), request)
        elif request.tool in ('evaluate_memory', 'inspect_memory'):
            result = {trial['trial_id']: {key: trial['prediction']['memory'].get(key) for key in
                ('admission', 'peak_per_rank_bytes', 'memory_budget_per_rank_bytes',
                 'memory_headroom_per_rank_bytes', 'oom_risk', 'scope', 'missing_profiles')}
                if trial['prediction'] is not None
                else dict(status='NEEDS_PROFILE', measured_peak_allocated_gib=(trial['measurement'] or {}).get(
                    'peak_allocated_gib'), scope='measured_global_peak_without_task_lifetime_breakdown')
                for trial in selected}
            for trial in selected:
                if trial.get('failure') is not None:
                    result[trial['trial_id']]['runtime_failure'] = {key: trial['failure'].get(key)
                        for key in ('witness_id', 'run_id', 'status', 'observed',
                                    'hypotheses', 'evidence_refs', 'evidence_scope')}
        elif request.tool == 'inspect_prediction_error':
            result = {trial['trial_id']: dict(validation_residuals=self.predictor.database.validation_residuals(
                trial['plan']['backend_config']), missing_profiles=trial['prediction']['evidence'].get(
                    'missing_profiles', []) if trial['prediction'] is not None else ['当前计划尚无预测记录'])
                for trial in selected}
        else:
            if len(selected) != 2 or len(set(request.candidate_ids)) != 2:
                raise ValueError('候选比较需要两个不同的候选')
            left, right = selected
            times = [trial['prediction']['performance']['step_time_ms']
                if trial['prediction'] is not None else None for trial in (left, right)]
            result = dict(left=left['trial_id'], right=right['trial_id'],
                performance_comparison_status='AVAILABLE' if all(value is not None for value in times) else 'UNAVAILABLE',
                step_time_delta_ms=times[1] - times[0] if all(value is not None for value in times) else None,
                left_critical_path=bounded_observation(left['prediction']['diagnosis'])
                    if left['prediction'] is not None else None,
                right_critical_path=bounded_observation(right['prediction']['diagnosis'])
                    if right['prediction'] is not None else None,
                left_memory={key: left['prediction']['memory'].get(key) for key in
                    ('admission', 'peak_per_rank_bytes', 'memory_budget_per_rank_bytes',
                     'memory_headroom_per_rank_bytes', 'oom_risk', 'scope', 'missing_profiles')}
                    if left['prediction'] is not None else None,
                right_memory={key: right['prediction']['memory'].get(key) for key in
                    ('admission', 'peak_per_rank_bytes', 'memory_budget_per_rank_bytes',
                     'memory_headroom_per_rank_bytes', 'oom_risk', 'scope', 'missing_profiles')}
                    if right['prediction'] is not None else None,
                left_runtime_failure=self.failure_summary(left), right_runtime_failure=self.failure_summary(right),
                left_measurement=self.candidate_summary(left)['measurement'],
                right_measurement=self.candidate_summary(right)['measurement'])
        if (self.config.execution_evidence_format != 'legacy'
                and request.tool in ('inspect_training_graph', 'inspect_execution_graph',
                    'inspect_dag_subgraph', 'inspect_shared_resources')):
            viewed_trials = selected or [self.reference_trial()]
            views = []
            for item in viewed_trials:
                graph = self.backend_dag(item)
                scope = ([request.region_id] if request.region_id is not None else
                    sorted({graph['tasks'][key]['region_id'] for key in request.task_ids
                        if key in graph['tasks'] and graph['tasks'][key]['region_id'] is not None}))
                if not scope:
                    regions = sorted(row['region_id'] for row in graph['regions'])
                    width = min(request.limit, 2)
                    scope = regions[request.offset:request.offset + width]
                    if not scope:
                        views.append(dict(trial_id=item['trial_id'], status='END_OF_RECORDS',
                            total_regions=len(regions), offset=request.offset))
                        continue
                    document = self.native_execution_view(item, scope)
                    document.update(total_regions=len(regions), offset=request.offset,
                        next_offset=request.offset + len(scope)
                            if request.offset + len(scope) < len(regions) else None)
                else:
                    document = self.native_execution_view(item, scope)
                views.append(document)
            result = dict(execution_views=views, query=request.model_dump(mode='json'))
        identifier = fingerprint(dict(tool=request.model_dump(), result=result))
        result_tokens = token_count(result)
        maximum_result_tokens = (12000 if self.config.execution_evidence_format != 'legacy'
            and 'execution_views' in result else 4000)
        if result_tokens > maximum_result_tokens:
            full_identity = identifier
            self.store.add_evidence(full_identity, 'PREDICTED', dict(
                tool=request.model_dump(mode='json'), result=result, result_tokens=result_tokens,
                evidence_scope='complete_tool_result_retained_in_backend'))
            binding = {key: result[key] for key in ('strategy_effect_identity', 'proposal_identity') if key in result}
            if 'strategy_effect_identity' in binding:
                binding['execution_effects'] = dict(changed_settings=result['execution_effects']['changed_settings'])
            result = dict(status='PAGE_TOO_LARGE', complete_result_id=full_identity,
                complete_result_tokens=result_tokens, maximum_result_tokens=maximum_result_tokens,
                suggested_query=request.model_copy(update=dict(limit=max(1, request.limit // 2))).model_dump(mode='json'),
                instruction='缩小limit或限定task_ids、region_ids后再次查询；完整结果保存在后端。', **binding)
            identifier = fingerprint(dict(tool=request.model_dump(mode='json'), result=result))
            result_tokens = token_count(result)
        self.store.add_evidence(identifier, 'PREDICTED', dict(tool=request.model_dump(), result=result,
            result_tokens=result_tokens, token_encoding='o200k_base',
            evidence_scope='query_result_with_explicit_cost_and_trace_sources'))
        return dict(evidence_id=identifier, tool=request.model_dump(), result=result,
            result_tokens=result_tokens, token_encoding='o200k_base')

    def execution_graph_observation(self, trial):
        graph = self.load_prediction_graph(trial['prediction']) if trial['prediction'] else None
        if graph is not None:
            return dict(graph, source='PREDICTED')
        candidate = self.rewriter.candidate(trial['plan']['backend_config'])
        graph = self.rewriter.compiler.observation(candidate, [], [])['training_graph']
        return dict(task_count=len(graph['tasks']), source='STRUCTURAL',
            tasks={key: dict(rank=task['rank'], region=task['region_id'], kind=task['operation'],
                dependencies=task['dependencies'], metadata=task['metadata'], duration_ms=None)
                for key, task in graph['tasks'].items()})

    def available_query_tools(self):
        names = set(InvestigationToolRequest.model_json_schema()['properties']['tool']['enum'])
        graph_tools = {'inspect_training_graph', 'inspect_dag_subgraph', 'inspect_measured_trace', 'inspect_task_dependencies',
            'inspect_region_cost', 'inspect_communication_flow', 'inspect_shared_resources',
            'inspect_execution_graph', 'inspect_critical_path', 'inspect_region',
            'inspect_communication', 'compare_candidate_graph_delta', 'inspect_execution_impact'}
        impact_tools = {'query_affected_decisions', 'preview_transformation'}
        interaction_tools = {'analyze_transformation_interactions'}
        if self.config.evaluation_mode == 'config_only':
            names -= graph_tools | impact_tools | interaction_tools
        elif self.config.evaluation_mode == 'graph_aware':
            names -= impact_tools | interaction_tools
        elif self.config.evaluation_mode == 'graph_impact':
            names -= interaction_tools
        return sorted(names)

    def plan(self, request, parent=None, transformations=None):
        if self.model.num_experts is None and request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp':
            # 稠密DDP没有专家参数，补齐未使用的专家Mesh描述以匹配实际进程组。
            request = dict(request, expert_dp_inner_size=process_groups(request)['expert_dp_size'],
                expert_dp_outer_size=1)
        config = self.rewriter.candidate(request).parallel_config
        sharding = state_sharding_config(request, config)
        stored = self.store.connection.execute('SELECT document FROM trials WHERE id=?',
            (request['run_id'],)).fetchone()
        revision = json.loads(stored[0])['plan'].get('identity_revision', 'v2') if stored else 'v3'
        return TrainingPlan(identity_revision=revision, plan_id=request['run_id'], parent_plan_id=parent,
            model_fingerprint=self.model_graph['model_fingerprint'],
            hardware_fingerprint=fingerprint(self.hardware_identity),
            backend_fingerprint=fingerprint(self.backend_identity),
            workload_fingerprint=fingerprint({key: request.get(key, {'precision': 'bfloat16',
                'optimizer': 'adam', 'precision_aware_optimizer': False}.get(key)) for key in
                ('model', 'checkpoint', 'data', 'global_batch', 'sequence', 'precision', 'optimizer',
                 'precision_aware_optimizer')}),
            parallelism={key: getattr(config, key) for key in ('tp', 'pp', 'cp', 'ep', 'dp', 'vpp')},
            region_partition=request['decoder_chunks'], region_layouts={},
            rank_mapping=request.get('rank_mapping', list(range(config.world_size))),
            schedule=request.get('pipeline_schedule', 'interleaved_1f1b' if config.vpp > 1 else '1f1b'),
            expert_parallelism=dict(ep=config.ep, etp=config.etp,
                expert_dp=config.world_size // (config.etp * config.ep * config.pp)),
            process_groups=process_groups(request), num_microbatches=config.num_microbatches,
            optimizer_sharding=request.get('optimizer_sharding', 'distributed'),
            state_sharding=sharding,
            optimizer_configuration=dict(optimizer=request.get('optimizer', 'adam'),
                precision_aware=request.get('precision_aware_optimizer', False),
                state_owner=sharding['execution_backend'],
                distributed_optimizer=sharding['optimizer_sharding'] == 'distributed'),
            checkpoint_policy=dict(format=sharding['checkpoint_format'],
                reshard=sharding['checkpoint_reshard'], source=request.get('checkpoint'),
                optimizer_mode=sharding['optimizer_checkpoint_mode'],
                optimizer_checkpoint_supported=(sharding['execution_backend'] != 'megatron_fsdp'
                    or request.get('fsdp_optimizer_checkpoint_supported', False))),
            special_module_placement=dict(embedding=0, output=len(request['decoder_chunks']) - 1,
                loss=len(request['decoder_chunks']) - 1, tied_parameters='native_embedding_gradient_finalize'),
            memory_policy=request.get('memory_policy', dict(recompute=request['recompute'], offload=request['offload'])),
            backend_config=request, transformations=transformations or [])

    def invoke(self, role, instructions, observation, schema, context_metric):
        self.check_walltime()
        if not self.config.agent_execution_enabled:
            raise ValueError('当前搜索配置关闭外部Agent执行')
        if self.config.external_agent_authorization_path is None:
            raise ValueError('外部Agent调用缺少限定范围的授权文件')
        authorization_path = Path(self.config.external_agent_authorization_path)
        authorization = json.loads(authorization_path.read_text())
        required_scope = {'training_plan', 'performance_summary', 'task_ids',
            'local_dag_subgraphs', 'candidate_predictions', 'search_history'}
        required_exclusions = {'model_weights', 'training_data_content', 'credentials',
            'checkpoints', 'complete_trace', 'complete_dag'}
        expires_at = datetime.fromisoformat(authorization['expires_at'])
        if (authorization.get('project') != 'ParaMorph'
                or authorization.get('provider') != self.config.llm_provider
                or not required_scope <= set(authorization.get('allowed_data', []))
                or not required_exclusions <= set(authorization.get('excluded_data', []))
                or expires_at <= datetime.now(timezone.utc)):
            raise ValueError('外部Agent授权范围、接收端或有效期不满足当前调用')
        if self.provider is None:
            if self.config.llm_provider == 'direct_api':
                self.provider = DirectAPIProvider(self.output / 'agents', model=self.config.agent_model,
                    base_url=self.config.llm_base_url,
                    reasoning_effort=self.config.agent_reasoning_effort)
            else:
                from megatron.core.frontier.agents.provider import CodexJSONProvider
                self.provider = CodexJSONProvider(self.output / 'agents', model=self.config.agent_model,
                    base_url=self.config.llm_base_url, timeout_seconds=600,
                    reasoning_effort=self.config.agent_reasoning_effort or 'medium',
                    structured_context=self.config.agent_cli_context_profile == 'structured',
                    model_instructions_file=self.config.agent_cli_model_instructions_file)
        usage = self.store.usage()
        prompt_names = dict(PerformanceAnalysisAgent='performance_analysis_agent.md',
            JointStrategyAgent='joint_strategy_agent.md',
            ConstraintValidationAgent='constraint_validation_agent.md',
            ExperimentInvestigationAgent='experiment_investigation_agent.md')
        prompt_directory = Path(__file__).resolve().parents[1] / 'agents' / 'prompts'
        provider_role = 'SingleAgent' if self.config.agent_ablation == 'single_agent' else role
        role_prompt = (prompt_directory / prompt_names[role]).read_text()
        if role == 'JointStrategyAgent' and issubclass(schema, PlanningInvestigationBatch):
            role_prompt = (prompt_directory / ('planning_portfolio_agent.md'
                if self.config.planning_portfolio_enabled else 'planning_investigation_agent.md')).read_text()
        if provider_role == 'SingleAgent':
            role_prompt = ('你是 ParaMorph Single Agent，在相同工具和总推理预算下依次承担瓶颈分析、'
                '联合策略生成、约束解释和实验选择职责。当前调用职责如下。\n' + role_prompt)
        shared_knowledge = ('' if role in ('JointStrategyAgent', 'ConstraintValidationAgent') else
            (prompt_directory / 'training_optimization_knowledge.md').read_text() + '\n')
        if self.config.agent_request_profile == 'compact':
            compact_prompts = json.loads((prompt_directory / 'compact_roles.json').read_text())
            role_prompt = compact_prompts[role]
            shared_knowledge = compact_prompts['common']
        if role == 'ExperimentInvestigationAgent' and schema.__name__ == 'BudgetedExperimentDecision':
            role_prompt = ('你负责从options选择下一项实验，按指定JSON结构返回candidate_id和reason。'
                '有效配置、预算、容量状态和预测分别记录；可以用合法配置的可验证变化选择探索测量。')
        instructions = role_prompt + '\n' + shared_knowledge + instructions
        instructions += '\ncolumnar_records_v1使用columns定义字段顺序，每项rows保存一条完整记录的对应字段值。'
        instructions += ('\n决策解释保持简洁：reason控制在600个中文字符以内；'
            '每项hypothesis和expected_effect各控制在180个中文字符以内。'
            '说明影响选择的执行证据、具体修改和需要检验的结果；'
            '完整配置、证据编号和工具请求通过对应结构化字段提交，说明文字避免重复这些字段。')
        if self.config.execution_evidence_format != 'legacy':
            instructions += ('\n执行视图记录当前候选的实际模块归属、原生F/B执行顺序、通信组及成本状态。'
                '依据这些事实自主决定联合并行参数、模块布局、运行设置及测量次序。'
                '比较不同MBS、PP和VPP时检查固定GBS下的微批数量、原生调度和通信粒度；'
                '事件数量不能直接代替总计算量或训练时间。'
                '局部查询保留物理阶段共驻区间和外部依赖；完成前向至开始反向的序列区间'
                '表示待反向分块，实际激活字节与峰值显存仍使用成本和实测证据。'
                '有效加载验证支持选择完整Benchmark；策略收益由实际训练检验。')
        policy = AblationEvidencePolicy(self.config.agent_ablation)
        if self.config.single_decision_round and role == 'JointStrategyAgent':
            from agents.single_round import INSTRUCTIONS
            instructions = INSTRUCTIONS
        instructions += '\n' + policy.instructions
        visible = self.visible_observation(observation)
        if self.config.agent_request_profile == 'decision':
            from agents.context_budget import role_observation
            visible = role_observation(role, visible)
        elif self.config.agent_request_profile == 'compact':
            from agents.context_budget import compact_role_observation
            visible = compact_role_observation(role, visible)
        visible['ablation_contract'] = policy.contract
        specification = to_strict_json_schema(schema)
        input_limit = (min(self.config.planning_initial_prompt_limit, self.config.agent_context_max_tokens)
            if context_metric['tool_round'] == 0 else self.config.agent_context_max_tokens)
        if issubclass(schema, NativeScopeRoundDecision):
            legacy_schema_tokens = token_count(to_strict_json_schema(ScopeRoundDecision))
            input_limit += max(0, token_count(specification) - legacy_schema_tokens)
            if self.config.agent_token_budget_enabled:
                input_limit = min(self.config.round_input_token_limit, input_limit)
        fixed_tokens = token_count(instructions) + token_count(specification) + 256
        if fixed_tokens >= input_limit:
            raise ValueError(dict(status='CONTEXT_SCHEMA_BUDGET_EXCEEDED',
                instruction_and_schema_tokens=fixed_tokens, input_limit=input_limit))
        visible, packing = pack_context(visible, input_limit - fixed_tokens, token_count)
        visible = self.visible_observation(visible)
        prompt = ('Use only the supplied observation. Request backend tools through the JSON tool_requests field. '
                  'Do not access files, edit source code, or invent measurements. '
                  'Return one object matching the output schema.\n'
                  + instructions + '\n' + json.dumps(visible, ensure_ascii=False, allow_nan=False,
                    separators=(',', ':')))
        estimated_prompt_tokens = token_count(prompt)
        metric = self.store.record_agent_context(context_metric['context_id'],
            role=provider_role, role_name=AGENT_ROLE_NAMES[provider_role], context_version=LAYERED_CONTEXT_VERSION,
            tool_round=context_metric['tool_round'], token_encoding='o200k_base',
            observation_tokens=token_count(visible), strict_schema_tokens=token_count(specification),
            context_packing=packing, input_token_limit=input_limit,
            estimated_prompt_tokens=estimated_prompt_tokens,
            current_tool_return_tokens=context_metric['current_tool_return_tokens'],
            cumulative_tool_return_tokens=context_metric['cumulative_tool_return_tokens'],
            cumulative_estimated_context_tokens=context_metric['cumulative_estimated_context_tokens']
                + estimated_prompt_tokens)
        if estimated_prompt_tokens + metric['strict_schema_tokens'] > input_limit:
            raise ValueError(f'Agent Prompt与严格Schema合计超过{input_limit} tokens：'
                f'{estimated_prompt_tokens + metric["strict_schema_tokens"]}')
        remaining_tokens = (int(self.config.budget.max_agent_tokens - usage.get('agent_input_tokens', 0)
                               - usage.get('agent_output_tokens', 0)) if self.config.agent_token_budget_enabled else 32768)
        audit = dict(timestamp=datetime.now(timezone.utc).isoformat(), role=role,
            cli_context_profile=self.config.agent_cli_context_profile,
            execution_evidence_format=self.config.execution_evidence_format,
            execution_view_revision=VIEW_REVISION,
            provider=self.config.llm_provider, base_url=self.config.llm_base_url,
            authorization_path=str(authorization_path), authorization_id=authorization['authorization_id'],
            observation_top_level_fields=sorted(visible), prompt_sha256=fingerprint(prompt),
            estimated_prompt_tokens=estimated_prompt_tokens,
            schema_tokens=metric['strict_schema_tokens'])
        with (self.output / 'external-agent-audit.jsonl').open('a') as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + '\n')
        if hasattr(self.provider, 'timeout_seconds'):
            self.store.record_session_time(self.session_id, time.perf_counter() - self.started)
            self.provider.timeout_seconds = min(self.provider.timeout_seconds,
                max(0.001, self.config.budget.max_walltime_seconds - self.store.search_seconds()))
        for output_revision in range(3):
            try:
                response, provenance = self.provider.generate(provider_role, instructions, visible, schema,
                    on_start=lambda identifier: self.charge_agent_attempt(identifier, provider_role, context_metric['context_id']),
                    on_finish=lambda identifier, seconds: self.store.charge(identifier + ':seconds', 'llm_seconds', seconds),
                    on_usage=self.charge_agent_usage, max_output_tokens=max(1, min(32768, remaining_tokens)))
                break
            except ValidationError as error:
                failure = dict(status='SCHEMA_REVISION_REQUIRED', revision_owner=role,
                    errors=error.errors(include_input=False, include_context=False, include_url=False),
                    instruction='修正字段校验错误，重新提交符合Schema的完整决策。')
                self.store.add_evidence(fingerprint(failure), 'FAILURE', failure)
                if output_revision == 2:
                    raise
                visible = dict(visible, schema_validation_feedback=failure)
                visible, packing = pack_context(visible, input_limit - fixed_tokens, token_count)
            except subprocess.TimeoutExpired:
                self.check_walltime()
                raise
        if role == 'JointStrategyAgent' and schema.__name__.startswith('ConfiguredCompleteStrategyBatch'):
            response = self.canonical_complete_batch(response.model_dump(mode='json', exclude_none=True))
        self.store.record_agent_context(context_metric['context_id'],
            provider=provenance['provider'], provider_prompt_sha256=provenance['prompt_sha256'],
            actual_input_tokens=provenance['usage'].get('input_tokens', 0),
            actual_output_tokens=provenance['usage'].get('output_tokens', 0),
            cumulative_actual_input_tokens=context_metric['cumulative_actual_input_tokens']
                + provenance['usage'].get('input_tokens', 0),
            replayed_verified_response=provenance.get('replayed_verified_response', False),
            cumulative_estimated_context_tokens=metric['cumulative_estimated_context_tokens'])
        known = {row[0] for row in self.store.connection.execute('SELECT id FROM evidence')}
        if role == 'ExperimentInvestigationAgent' and not isinstance(response, InvestigationDecision):
            known.update(trial['trial_id'] for trial in self.store.trials())
        refs = list(getattr(response, 'evidence_refs', []))
        refs.extend(ref for intent in getattr(response, 'intents', []) for ref in intent.evidence_refs)
        refs.extend(ref for action in getattr(response, 'actions', []) for ref in action.evidence_refs)
        refs.extend(ref for intent in getattr(response, 'intents', []) for ref in intent.interaction_evidence_ids)
        refs.extend(ref for intent in getattr(response, 'intents', []) if intent.target_bottleneck is not None
                    for ref in intent.target_bottleneck.evidence_refs)
        refs.extend(ref for report in getattr(response, 'intent_results', []) for ref in report.evidence_refs)
        identifier = fingerprint(dict(role=role, prompt=provenance['prompt_sha256'], response=provenance['response_sha256']))
        self.store.add_evidence(identifier, 'HYPOTHESIS', response.model_dump(mode='json'))
        self.store.add_search_state('AGENT_RESPONSE', dict(role=role,
            wire_schema=schema.__name__, evidence_id=identifier,
            response=response.model_dump(mode='json')))
        return response, identifier

    def evidence_reference_feedback(self, role, response):
        if role == 'JointStrategyAgent':
            return None
        known = {row[0] for row in self.store.connection.execute('SELECT id FROM evidence')}
        if role == 'ExperimentInvestigationAgent' and not isinstance(response, InvestigationDecision):
            known.update(trial['trial_id'] for trial in self.store.trials())
        refs = list(getattr(response, 'evidence_refs', []))
        refs.extend(ref for report in getattr(response, 'intent_results', []) for ref in report.evidence_refs)
        unknown = sorted(set(refs) - known)
        valid_tasks = set()
        unknown_tasks = []
        if isinstance(response, BottleneckDecision) and response.affected_tasks:
            reference = self.reference_trial()
            valid_tasks = set(self.rewriter.compiler.graph(
                self.rewriter.candidate(reference['plan']['backend_config'])).tasks)
            unknown_tasks = sorted(set(response.affected_tasks) - valid_tasks)
        if refs and not unknown and not unknown_tasks:
            return None
        result = dict(status='REVISION_REQUIRED', revision_owner=role,
            reason='证据编号及任务编号必须存在于当前后端记录', unknown_evidence_refs=unknown,
            unknown_task_ids=unknown_tasks, valid_task_id_examples=sorted(valid_tasks)[:24],
            valid_evidence_ids=sorted(known)[:24],
            instruction='依据已有证据修订引用；文件路径不能作为evidence_id。'
                'affected_tasks引用当前执行图中的真实任务；缺少任务证据时保留空列表并说明需要的测量。')
        tool = dict(tool='validate_evidence_references', role=role)
        identifier = fingerprint(dict(tool=tool, response=response.model_dump(mode='json'), result=result))
        self.store.add_evidence(identifier, 'FAILURE', dict(tool=tool, result=result))
        return dict(evidence_id=identifier, tool=tool, result=result,
            result_tokens=token_count(result))

    def selection_admission_feedback(self, response, candidate_ids=()):
        by_id = {row['trial_id']: row for row in self.store.trials()}
        conflicts = []
        assessments = {row.trial_id: row for row in response.candidate_assessments}
        if len(assessments) != len(response.candidate_assessments):
            conflicts.append(dict(reason='同一候选存在重复比较'))
        missing = sorted(set(candidate_ids) - assessments.keys())
        if missing:
            conflicts.append(dict(candidate_ids=missing, reason='需要逐项比较完整候选的收益、代价及未知成本'))
        for identifier in assessments.keys() - by_id.keys():
            conflicts.append(dict(trial_id=identifier, reason='比较引用了未登记的候选'))
        for identifier in response.accepted_candidate_ids:
            trial = by_id.get(identifier)
            if trial is None:
                conflicts.append(dict(trial_id=identifier, reason='候选身份未登记'))
                continue
            if trial.get('measurement') is not None:
                continue
            if candidate_ids and (identifier not in assessments
                    or assessments[identifier].next_action != 'BENCHMARK'):
                conflicts.append(dict(trial_id=identifier, reason='完整训练选择需要对应BENCHMARK比较结论'))
            prediction = trial.get('prediction') or {}
            admission = prediction.get('memory', {}).get('admission')
            missing_time = prediction.get('performance', {}).get('step_time_ms') is None
            if admission == 'REJECT_OOM':
                conflicts.append(dict(trial_id=identifier, reason='当前完整显存模型拒绝容量准入'))
            elif (admission != 'WITHIN_ESTIMATED_CAPACITY' or missing_time) and (
                    identifier not in response.exploration_candidate_ids):
                conflicts.append(dict(trial_id=identifier, memory_admission=admission,
                    complete_time_available=not missing_time,
                    reason='完整训练需要显式选择探索配额'))
        if not conflicts:
            return None
        result = dict(status='REVISION_REQUIRED', revision_owner='ExperimentInvestigationAgent',
            conflicts=conflicts,
            instruction='依据当前预测和预算重新选择。完整时间或容量未知的完整训练，'
                '需要在exploration_candidate_ids中显式授权；先做独立加载及单步验证时，'
                '使用load_validation_candidate_ids并从accepted_candidate_ids移除。'
                '也可以选择补测成本或取消本次训练。candidate_assessments覆盖当前候选，'
                '分别填写完整策略的潜在收益、新增代价、未知成本和后续动作；'
                '当前父策略的瓶颈用于提出假设，候选优先级依据完整迭代的收益与测量价值判断。')
        tool = dict(tool='validate_training_admission')
        identifier = fingerprint(dict(tool=tool, response=response.model_dump(mode='json'), result=result))
        self.store.add_evidence(identifier, 'FAILURE', dict(tool=tool, result=result))
        return dict(evidence_id=identifier, tool=tool, result=result, result_tokens=token_count(result))

    @staticmethod
    def canonical_complete_batch(payload):
        def canonical_intent(intent):
            return dict(intent, adjustable=[], preserve=[], allowed_ranges={},
                module_relations=[], proposed_actions=[])

        payload = dict(payload, intents=[canonical_intent(intent) for intent in payload['intents']])
        queries = []
        for original in payload.get('tool_requests', []):
            query = dict(original)
            if query.get('transformation_intent') is not None:
                query['transformation_intent'] = canonical_intent(query['transformation_intent'])
            query['transformation_intents'] = [canonical_intent(intent)
                for intent in query.get('transformation_intents', [])]
            queries.append(query)
        payload['tool_requests'] = queries
        return IntentBatch.model_validate(payload)

    def charge_agent_attempt(self, identifier, role=None, context_id=None):
        self.check_walltime()
        if self.config.single_decision_round:
            index = int(self.store.usage().get('agent_rounds', 0))
            checkpoint = self.store.get_round(index)
            start = checkpoint['usage_at_start']
            limit = 2 if self.config.budgeted_investigation_enabled else 1
            permitted = ('JointStrategyAgent', 'ExperimentInvestigationAgent') if limit == 2 else ('JointStrategyAgent',)
            if role not in permitted or self.store.usage().get('agent_calls', 0) - start.get('agent_calls', 0) >= limit:
                raise SearchBudgetExhausted('本轮已用完模型决策次数')
        if self.store.usage().get('agent_calls', 0) >= self.config.budget.max_agent_calls:
            raise SearchBudgetExhausted('实际Provider请求次数达到预算')
        usage = self.store.usage()
        if (self.config.agent_token_budget_enabled
                and usage.get('agent_input_tokens', 0) + usage.get('agent_output_tokens', 0) >= self.config.budget.max_agent_tokens):
            raise SearchBudgetExhausted('实际Provider Token消耗达到预算')
        self.store.charge(identifier, 'agent_calls', 1)
        if context_id is not None:
            metric = self.store.record_agent_context(context_id)
            attempt_ids = sorted(set(metric.get('provider_attempt_ids', [])) | {identifier})
            self.store.record_agent_context(context_id, role=role,
                role_name=AGENT_ROLE_NAMES[role], provider_attempt_ids=attempt_ids,
                provider_attempt_count=len(attempt_ids))

    def charge_agent_usage(self, identifier, usage):
        for name in ('input_tokens', 'output_tokens'):
            self.store.charge(identifier + ':' + name, 'agent_' + name, usage.get(name, 0))

    def agent_session_identity(self, role, instructions, observation, schema):
        stable_observation = self.visible_observation(self.agent_context(observation, role))
        if 'budget' in stable_observation:
            stable_observation = dict(stable_observation,
                budget=dict(limits=stable_observation['budget']['limits']))
        return fingerprint(dict(role=role, instructions=instructions,
            model=self.config.agent_model, reasoning_effort=self.config.agent_reasoning_effort,
            request_profile=self.config.agent_request_profile,
            model_instructions_sha256=(fingerprint(Path(self.config.agent_cli_model_instructions_file).read_text())
                if self.config.agent_cli_model_instructions_file else None),
            observation=stable_observation,
            schema=schema.model_json_schema()))

    def agent_context(self, observation, role=None):
        tools = (list(ConstraintValidationToolRequest.model_json_schema()['properties']['tool']['enum'])
            if role == 'ConstraintValidationAgent' else self.available_query_tools())
        if role == 'PerformanceAnalysisAgent':
            tools = sorted(set(tools) & set(AnalysisToolRequest.model_json_schema()['properties']['tool']['enum']))
        elif role == 'ExperimentInvestigationAgent':
            tools = sorted(set(tools) & set(EvaluationToolRequest.model_json_schema()['properties']['tool']['enum']))
        return dict(observation, available_tools=tools,
            intent_count=self.config.intent_count,
            tool_usage='通过tool_requests查询后端完整DAG、成本、历史和变换影响；'
                '每页使用offset与limit，继续展开使用next_offset；引用返回的evidence_id')

    @staticmethod
    def expand_preview_query(query):
        if query.tool != 'preview_transformation' or not query.transformation_intents:
            return [query]
        intents = ([query.transformation_intent] if query.transformation_intent is not None else [])
        intents.extend(query.transformation_intents)
        unique = {}
        for intent in intents:
            identity = EvidenceGuidedSearch.strategy_effect_identity(intent)
            unique.setdefault(identity, intent)
        return [query.model_copy(update=dict(transformation_intent=intent,
            transformation_intents=[])) for intent in unique.values()]

    def invoke_with_tools(self, role, instructions, observation, schema):
        self._visible_trial_ids = AblationEvidencePolicy.current_trial_ids(observation)
        if self.config.single_decision_round:
            if role != 'JointStrategyAgent' or schema is not IntentBatch:
                raise ValueError('单次决策模式仅由策略角色调用模型')
            from agents.single_round import (RoundDecision,
                observation as compact_input, complete_values, reference_view)
            def read_wire(encoded):
                document = json.loads(encoded)
                first = document['candidates'][0]
                wire_schema = (geometry_schema if isinstance(first.get('parallel'), str)
                    else NativeScopeRoundDecision if isinstance(first.get('allowed_ranges'), dict)
                    else RoundDecision)
                return wire_schema.model_validate(document)
            parent_id = observation['current_training_plan']['plan_id']
            parent = next(trial for trial in self.store.trials() if trial['plan']['plan_id'] == parent_id)
            request = parent['plan']['backend_config']
            thin = compact_input(self.visible_observation(observation), request, self.config.intent_count)
            geometry_schema = native_geometry_decision_schema(
                native_geometry_choices(self.joint_global_families(request)), len(request['rank_mapping']))
            thin['hardware_limits'] = dict(nodes=self.topology.num_nodes,
                gpus_per_node=self.topology.gpus_per_node, memory_per_gpu_gib=self.topology.memory_per_gpu_gb,
                gpu_models=sorted({device.get('name') or device.get('device_name', 'unknown')
                    for device in self.topology.devices}),
                rank_to_node={str(device['physical_rank']): device['node_index'] for device in self.topology.devices})
            thin['round_index'] = int(self.store.usage().get('agent_rounds', 0)) + 1
            metric = dict(context_id=fingerprint(dict(single_decision=thin)), tool_round=0,
                current_tool_return_tokens=0, cumulative_tool_return_tokens=0,
                cumulative_estimated_context_tokens=0, cumulative_actual_input_tokens=0)
            index = int(self.store.usage().get('agent_rounds', 0))
            checkpoint = self.store.get_round(index)
            aliases = checkpoint.get('single_reference_aliases', {})
            if checkpoint.get('single_wire_evidence_id'):
                identifier = checkpoint['single_wire_evidence_id']
                wire = self.store.connection.execute('SELECT document FROM evidence WHERE id=?', (identifier,)).fetchone()
                decision = read_wire(wire[0])
            elif self.store.usage().get('agent_calls', 0) > checkpoint['usage_at_start'].get('agent_calls', 0):
                metrics = self.store.agent_context_metrics()[checkpoint['context_metric_start']:]
                hashes = {entry['provider_prompt_sha256'] for entry in metrics
                    if entry.get('role') == role and entry.get('actual_input_tokens', 0)}
                attempts = {identifier.split(':', 1)[0] for entry in metrics if entry.get('role') == role
                    for identifier in entry.get('provider_attempt_ids', [])}
                receipts = [(path.parent, json.loads(path.read_text()))
                    for path in (self.output / 'agents').glob('*/provenance.json')]
                matching = [(path, entry) for path, entry in receipts if entry['role'] == role
                    and (entry['prompt_sha256'] in hashes or path.name in attempts)]
                if len(matching) != 1:
                    raise ValueError('当前迭代的已计费模型结果缺少唯一、完整的原始回执')
                directory, previous = matching[0]
                identifier = fingerprint(dict(role=role, prompt=previous['prompt_sha256'], response=previous['response_sha256']))
                wire = self.store.connection.execute('SELECT document FROM evidence WHERE id=?', (identifier,)).fetchone()
                if wire is None:
                    raw = (directory / 'response.json').read_bytes()
                    if (hashlib.sha256(raw).hexdigest() != previous['response_sha256']
                            or hashlib.sha256((directory / 'prompt.txt').read_bytes()).hexdigest() != previous['prompt_sha256']):
                        raise ValueError('模型调用原始文件与已计费回执的摘要不同')
                    encoded = json.loads(raw)['json_object']
                    decision = read_wire(encoded)
                    attempt_metrics = [entry for entry in metrics if any(
                        attempt.split(':', 1)[0] == directory.name
                        for attempt in entry.get('provider_attempt_ids', []))]
                    if len(attempt_metrics) != 1:
                        raise ValueError('原始模型回执未对应唯一的上下文计费记录')
                    self.store.record_agent_context(attempt_metrics[0]['context_id'],
                        provider=previous['provider'], provider_prompt_sha256=previous['prompt_sha256'],
                        actual_input_tokens=previous['usage']['input_tokens'],
                        actual_output_tokens=previous['usage']['output_tokens'],
                        cumulative_actual_input_tokens=previous['usage']['input_tokens'])
                    self.store.add_evidence(identifier, 'HYPOTHESIS', decision.model_dump(mode='json'))
                    self.store.add_search_state('AGENT_RESPONSE', dict(role=role,
                        wire_schema=type(decision).__name__, evidence_id=identifier,
                        response=decision.model_dump(mode='json')))
                else:
                    decision = read_wire(wire[0])
            else:
                compact, aliases = reference_view(thin)
                self.store.save_round(index, single_reference_aliases=aliases)
                metric['context_id'] = fingerprint(dict(single_decision=compact, aliases=aliases))
                decision, identifier = self.invoke(role, '', compact, geometry_schema, metric)
            self.store.save_round(index, single_wire_evidence_id=identifier,
                measurement_preference=decision.measurement_preference)
            actual_input = self.store.usage().get('agent_input_tokens', 0) - checkpoint['usage_at_start'].get('agent_input_tokens', 0)
            actual_output = self.store.usage().get('agent_output_tokens', 0) - checkpoint['usage_at_start'].get('agent_output_tokens', 0)
            actual_total = actual_input + actual_output
            capped = self.config.agent_token_budget_enabled
            within_cap = (actual_input <= self.config.round_input_token_limit
                and actual_total <= self.config.round_total_token_limit)
            receipt = dict(kind='single_round_input_budget', round_index=index,
                actual_input_tokens=actual_input, input_limit=self.config.round_input_token_limit if capped else None,
                actual_output_tokens=actual_output, actual_total_tokens=actual_total,
                total_limit=self.config.round_total_token_limit if capped else None,
                status=('WITHIN_BUDGET' if within_cap else 'EXCEEDED') if capped else 'RECORDED_WITHOUT_CAP',
                model_evidence_id=identifier)
            self.store.add_evidence(fingerprint(receipt), 'MEASURED', receipt)
            if capped and not within_cap:
                raise SearchBudgetExhausted('本轮实际Token超过预算，费用已登记并停止追加调用')
            decision = decision.model_copy(update=dict(candidates=[row.model_copy(update=dict(
                revision_refs=[aliases.get(identifier, identifier) for identifier in row.revision_refs]))
                for row in decision.candidates]))
            if not decision.candidates or len(decision.candidates) > self.config.intent_count:
                raise ValueError('模型提交的联合调查数量超出本轮允许范围')
            refs = thin['evidence_refs']
            if not refs:
                raise ValueError('单次决策缺少实际证据编号')
            intents, rejected_rows, accepted_rows, derived_rows = [], [], [], []
            for row_index, row in enumerate(decision.candidates):
                adjustable = list(row.adjustable) if isinstance(row, ScopeRow) else []
                if not set(row.revision_refs) <= set(refs):
                    rejected_rows.append(dict(row_index=row_index, status='INVALID_EVIDENCE_REFERENCE',
                        unknown_refs=sorted(set(row.revision_refs) - set(refs)), allowed_refs=refs))
                    continue
                try:
                    if isinstance(row, ScopeRow):
                        fixed = (row.fixed_values.model_dump(mode='json', exclude_none=True)
                            if isinstance(row, NativeScopeRow) else json.loads(row.fixed_values))
                        ranges = (row.allowed_ranges.model_dump(mode='json', exclude_none=True)
                            if isinstance(row, NativeScopeRow) else json.loads(row.allowed_ranges))
                        if not isinstance(fixed, dict) or not isinstance(ranges, dict):
                            raise ValueError('联合调查的固定值和范围需要JSON对象')
                        if isinstance(row, NativeScopeRow):
                            if isinstance(row.parallel, str):
                                backend, geometry = row.parallel.split(':', 1)
                                tp, pp, cp, ep, etp = map(int, geometry.split('/'))
                                if (fixed.get('execution_backend', backend) != backend
                                        or ranges.get('execution_backend', [backend]) != [backend]):
                                    rejected_rows.append(dict(row_index=row_index, status='CONFLICTING_EXECUTION_BACKEND',
                                        selected_backend=backend))
                                    continue
                                fixed['execution_backend'] = backend
                                ranges.pop('execution_backend', None)
                                adjustable = [name for name in adjustable if name != 'execution_backend']
                            else:
                                tp, pp, cp, ep, etp = row.parallel
                            world = len(request['rank_mapping'])
                            if sorted(row.ranks) != list(range(world)):
                                rejected_rows.append(dict(row_index=row_index, status='INVALID_PHYSICAL_RANK_MAPPING',
                                    world_size=world))
                                continue
                            dense, expert = tp * cp * pp, etp * ep * pp
                            if world % dense or (self.model.num_experts is not None and world % expert):
                                rejected_rows.append(dict(row_index=row_index, status='INCOMPATIBLE_PARALLEL_MESH',
                                    world_size=world, dense_mesh=dense, expert_mesh=expert))
                                continue
                            global_values = dict(tp=tp, pp=pp, cp=cp, dp=world // dense,
                                ep=ep, etp=etp, expert_dp=world // expert, rank_mapping=row.ranks)
                        else:
                            global_values = dict(zip(('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp'),
                                row.parallel, strict=True), rank_mapping=row.ranks)
                        duplicates = LOCAL_SCOPE_GLOBAL_FIELDS & (set(fixed) | set(ranges) | set(adjustable))
                        conflicts = sorted(name for name in duplicates if
                            (name in fixed and fixed[name] != global_values[name])
                            or (name in ranges and ranges[name] != [global_values[name]]))
                        if conflicts:
                            rejected_rows.append(dict(row_index=row_index, status='CONFLICTING_GLOBAL_FIELD',
                                fields=conflicts, required_source='parallel_and_ranks'))
                            continue
                        for name in duplicates:
                            fixed.pop(name, None)
                            ranges.pop(name, None)
                        adjustable = [name for name in adjustable if name not in duplicates]
                        scope_values = self.canonical_dense_scope(parent_request=request,
                            values=dict(global_values, **fixed), ranges=ranges)
                        changes = {key: [value, scope_values[key]] for key, value in global_values.items()
                            if scope_values[key] != value}
                        if changes:
                            derived_rows.append(dict(row_index=row_index, native_derived_fields=changes))
                        global_values = {key: scope_values[key] for key in global_values}
                        values = RewriteValues.model_validate(scope_values)
                        allowed = AllowedRanges.model_validate(ranges)
                    else:
                        values = RewriteValues.model_validate(complete_values(row, request))
                except ValidationError as error:
                    rejected_rows.append(dict(row_index=row_index,
                        errors=error.errors(include_url=False, include_context=False, include_input=False)))
                    continue
                intents.append(AgentRewriteIntent(proposal_kind='conditional_intent' if isinstance(row, ScopeRow) else 'complete_strategy',
                    operation='JointTransform', values=values,
                    adjustable=adjustable, preserve=[],
                    allowed_ranges=allowed if isinstance(row, ScopeRow) else AllowedRanges(),
                    parent_plan_id=parent_id, family_id=global_identity(dict(request, **global_values)) if isinstance(row, ScopeRow) else None,
                    hypothesis=row.reason + ('；验证对象：' + row.test_target if row.test_target else ''),
                    evidence_refs=list(dict.fromkeys([*row.revision_refs, *refs]))))
                accepted_rows.append(row_index)
            if rejected_rows:
                self.store.add_evidence(fingerprint(dict(model_id=identifier, rejected=rejected_rows)),
                    'FAILURE', dict(model_evidence_id=identifier, rejected_wire_rows=rejected_rows))
            strategy = IntentBatch(intents=intents,
                reason=('按模型给出的联合调查顺序展开本轮搜索' if intents
                    else json.dumps(rejected_rows, ensure_ascii=False)), evidence_refs=refs)
            expanded_id = fingerprint(strategy.model_dump(mode='json'))
            self.store.add_evidence(expanded_id, 'PREDICTED', strategy.model_dump(mode='json'))
            materialization = dict(kind='compact_strategy_materialization',
                model_evidence_id=identifier, expanded_strategy_id=expanded_id,
                numeric_completion=('joint_scope_native_solver_and_policy_frontier'
                    if any(isinstance(row, ScopeRow) for row in decision.candidates) else 'native_derived_dp_groups_only'),
                accepted_wire_rows=accepted_rows, rejected_wire_rows=rejected_rows, derived_wire_rows=derived_rows,
                agent_controls=('global_family_open_fields_ranges_and_measurement_priority'
                    if any(isinstance(row, ScopeRow) for row in decision.candidates)
                    else 'parallelism_layout_memory_communication_mapping_and_backend'))
            self.store.add_evidence(fingerprint(materialization), 'PREDICTED', materialization)
            return self.bind_strategy_preview_evidence(strategy, expanded_id)
        if role == 'JointStrategyAgent' and schema is IntentBatch:
            value_type = RewriteValues
            if not (self.config.request.get('megatron_fsdp_adapter_path')
                    and self.config.request.get('megatron_fsdp_adapter_sha256')
                    and 'megatron_fsdp' in self.config.request.get('allowed_execution_backends',
                        ('megatron_ddp', 'megatron_fsdp'))):
                value_names = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp',
                    'micro_batch', 'group', 'decoder_chunks', 'recompute', 'rank_mapping', 'memory_policy',
                    'pipeline_schedule', 'overlap_p2p', 'bucket_elements', 'optimizer_sharding',
                    'precision_aware_optimizer', 'optimizer_offload_fraction', 'optimizer_transfer_overlap',
                    'dp_inner_size', 'dp_outer_size', 'expert_dp_inner_size', 'expert_dp_outer_size')
                value_type = create_model('ConfiguredDDPStrategyValues', __base__=Contract,
                    **{name: (RewriteValues.model_fields[name].annotation,
                        deepcopy(RewriteValues.model_fields[name])) for name in value_names},
                    execution_backend=(Literal['megatron_ddp'], 'megatron_ddp'))
            complete_intent = create_model('ConfiguredJointStrategyIntent', __base__=AgentRewriteIntent,
                values=(value_type, ...))
            complete_query = create_model('CompleteStrategyQuery', __base__=InvestigationToolRequest,
                tool=(Literal[tuple(name for name in InvestigationToolRequest.model_json_schema()['properties']['tool']['enum']
                    if name not in ('preview_transformation', 'analyze_transformation_interactions'))], ...),
                transformation_intent=(complete_intent | None, None),
                transformation_intents=(list[complete_intent], Field(default_factory=list, max_length=3)))
            preview_query = create_model('CompleteStrategyPreviewQuery', __base__=InvestigationToolRequest,
                tool=(Literal['preview_transformation'], ...),
                transformation_intent=(complete_intent, ...),
                transformation_intents=(list[complete_intent], Field(default_factory=list, max_length=3)))
            interaction_query = create_model('CompleteStrategyInteractionQuery', __base__=InvestigationToolRequest,
                tool=(Literal['analyze_transformation_interactions'], ...),
                transformation_intent=(complete_intent | None, None),
                transformation_intents=(list[complete_intent], Field(min_length=2, max_length=3)))
            schema = create_model('ConfiguredCompleteStrategyBatch', __base__=IntentBatch,
                intents=(list[complete_intent], Field(default_factory=list,
                    max_length=self.config.intent_count)),
                tool_requests=(list[complete_query | preview_query | interaction_query],
                    Field(default_factory=list, max_length=min(self.config.intent_count, 4))))
        session_schema = IntentBatch if role == 'JointStrategyAgent' and issubclass(schema, IntentBatch) else schema
        identity = self.agent_session_identity(role, instructions, observation, schema)
        saved = self.store.get_agent_session(identity)
        context = self.agent_context(observation, role)
        final_schema = create_model(schema.__name__ + 'FinalDecision', __base__=schema,
            tool_requests=(schema.model_fields['tool_requests'].annotation,
                Field(default_factory=list, max_length=0))) if 'tool_requests' in schema.model_fields else schema

        def execute_query(query):
            if role == 'ConstraintValidationAgent':
                if (query.parent_trial_id != observation['parent_trial_id']
                        or query.strategy_evidence_id != observation['strategy_evidence_id']):
                    raise ValueError('约束工具必须使用当前角色上下文中的父计划与联合意图')
                return self.constraint_tool(query)
            return self.investigate(query, self.store.trials())
        if saved is None:
            cumulative_context_tokens = 0
            cumulative_tool_tokens = 0
            cumulative_actual_tokens = 0
            self.store.save_agent_session(identity, dict(context=context, rounds=0,
                cumulative_context_tokens=0, cumulative_tool_tokens=0, cumulative_actual_tokens=0))
            metric = dict(context_id=fingerprint(dict(session=identity, tool_round=0, context=context)),
                tool_round=0, current_tool_return_tokens=0, cumulative_tool_return_tokens=0,
                cumulative_estimated_context_tokens=0, cumulative_actual_input_tokens=0)
            response, identifier = self.invoke(role, instructions, context, schema, metric)
            recorded = self.store.record_agent_context(metric['context_id'])
            cumulative_context_tokens = recorded['cumulative_estimated_context_tokens']
            cumulative_actual_tokens = recorded['cumulative_actual_input_tokens']
            rounds = 0
        else:
            context, rounds = saved['context'], saved['rounds']
            cumulative_context_tokens = saved.get('cumulative_context_tokens', 0)
            cumulative_tool_tokens = saved.get('cumulative_tool_tokens', 0)
            cumulative_actual_tokens = saved.get('cumulative_actual_tokens', 0)
            if 'response' in saved:
                response = (self.canonical_complete_batch(saved['response'])
                    if role == 'JointStrategyAgent' and schema.__name__.startswith('ConfiguredCompleteStrategyBatch')
                    else session_schema.model_validate(saved['response']))
                identifier = saved['evidence_id']
            else:
                current_tool_tokens = saved.get('current_tool_tokens', 0)
                if rounds and context.get('previous_response') is not None and not context.get('tool_history'):
                    previous = session_schema.model_validate(context['previous_response'])
                    context = {key: value for key, value in context.items()
                               if key not in ('tool_history', 'previous_response')}
                    results = [execute_query(item) for query in previous.tool_requests
                        for item in self.expand_preview_query(query)]
                    current_tool_tokens = token_count(results)
                    cumulative_tool_tokens += current_tool_tokens
                    context.update(tool_history=results,
                        previous_response=previous.model_dump(mode='json', exclude_none=True))
                pending_instructions = instructions + ('\n根据工具返回的真实证据继续调查或输出最终结果。' if rounds else '')
                metric = dict(context_id=fingerprint(dict(session=identity, tool_round=rounds, context=context)),
                    tool_round=rounds, current_tool_return_tokens=current_tool_tokens if rounds else 0,
                    cumulative_tool_return_tokens=cumulative_tool_tokens,
                    cumulative_estimated_context_tokens=cumulative_context_tokens,
                    cumulative_actual_input_tokens=cumulative_actual_tokens)
                response, identifier = self.invoke(role, pending_instructions, context, schema, metric)
                recorded = self.store.record_agent_context(metric['context_id'])
                cumulative_context_tokens = recorded['cumulative_estimated_context_tokens']
                cumulative_actual_tokens = recorded['cumulative_actual_input_tokens']
        self.store.save_agent_session(identity, dict(context=context, rounds=rounds,
            cumulative_context_tokens=cumulative_context_tokens, cumulative_tool_tokens=cumulative_tool_tokens,
            cumulative_actual_tokens=cumulative_actual_tokens,
            response=response.model_dump(mode='json'), evidence_id=identifier))
        while True:
            requests = list(getattr(response, 'tool_requests', []))
            if requests and rounds >= self.config.max_tool_rounds:
                metric = dict(context_id=fingerprint(dict(session=identity, tool_round=rounds,
                    final_decision=True, context=context)), tool_round=rounds,
                    current_tool_return_tokens=0, cumulative_tool_return_tokens=cumulative_tool_tokens,
                    cumulative_estimated_context_tokens=cumulative_context_tokens,
                    cumulative_actual_input_tokens=cumulative_actual_tokens)
                response, identifier = self.invoke(role,
                    instructions + '\n调查配额已经用完。使用已取得的真实证据完成当前决策，tool_requests必须为空列表。',
                    context, final_schema, metric)
                recorded = self.store.record_agent_context(metric['context_id'])
                cumulative_context_tokens = recorded['cumulative_estimated_context_tokens']
                cumulative_actual_tokens = recorded['cumulative_actual_input_tokens']
                self.store.save_agent_session(identity, dict(context=context, rounds=rounds,
                    cumulative_context_tokens=cumulative_context_tokens,
                    cumulative_tool_tokens=cumulative_tool_tokens,
                    cumulative_actual_tokens=cumulative_actual_tokens,
                    response=response.model_dump(mode='json'), evidence_id=identifier))
                requests = []
            if role == 'JointStrategyAgent' and isinstance(response, IntentBatch) and not requests:
                response, identifier = self.bind_strategy_preview_evidence(response, identifier)
            feedback = (self.planning_validation_feedback(response)
                if isinstance(response, PlanningInvestigationBatch) and not requests else
                self.strategy_validation_feedback(response)
                if role == 'JointStrategyAgent' and isinstance(response, IntentBatch) and not requests
                else self.evidence_reference_feedback(role, response))
            if (not requests and feedback is None and role == 'ExperimentInvestigationAgent'
                    and isinstance(response, InvestigationSelection)):
                table = observation.get('candidates', {})
                current_ids = [row[table['columns'].index('trial_id')] for row in table.get('rows', [])]
                feedback = self.selection_admission_feedback(response, current_ids)
            if role == 'ConstraintValidationAgent' and not requests:
                constraint_feedback, required_queries = self.constraint_decision_feedback(response, observation)
                requests = required_queries
                if constraint_feedback is not None:
                    feedback = constraint_feedback
            interaction_refs = {ref for intent in getattr(response, 'intents', [])
                for ref in intent.interaction_evidence_ids}
            known_interaction_refs = {identifier for identifier, encoded in
                self.store.connection.execute('SELECT id, document FROM evidence')
                if self.is_interaction_evidence(json.loads(encoded))}
            known_family_ids = set()
            if role == 'JointStrategyAgent':
                reference = self.reference_trial()
                if reference is not None:
                    known_family_ids = {row['family_id'] for row in self.joint_global_families(
                        reference['plan']['backend_config'])}
            if (not requests and feedback is None and role == 'JointStrategyAgent' and response.intents
                    and any(intent.proposal_kind != 'complete_strategy' for intent in response.intents)
                    and not any(intent.family_id in known_family_ids for intent in response.intents)
                    and rounds < self.config.max_tool_rounds):
                requests = [InvestigationToolRequest(tool='inspect_global_families', limit=8)]
            elif (not requests and feedback is None and role == 'JointStrategyAgent' and response.intents
                    and (not interaction_refs or not interaction_refs <= known_interaction_refs)
                    and rounds < self.config.max_tool_rounds):
                preview_intents = response.intents[:3]
                target_tasks = list(dict.fromkeys(task_id for intent in preview_intents
                    for task_id in intent.target_task_ids))[:8]
                requests = [InvestigationToolRequest(tool='analyze_transformation_interactions',
                    task_ids=target_tasks, transformation_intents=preview_intents, limit=6)]
            if not requests and feedback is None:
                break
            if not requests and feedback is not None and rounds >= self.config.max_tool_rounds:
                revisions = context.get('final_validation_revisions', 0)
                if revisions >= 2:
                    raise AgentConstraintConflict(dict(conflicts=['final_response'],
                        reason='最终响应经过两次修订仍未通过真实证据校验',
                        validation_feedback=feedback, revision_owner=role))
                context = dict(context, final_validation_revisions=revisions + 1,
                    tool_history=[feedback], previous_response=response.model_dump(mode='json', exclude_none=True))
                metric = dict(context_id=fingerprint(dict(session=identity,
                    final_validation_revision=revisions + 1, context=context)), tool_round=rounds,
                    current_tool_return_tokens=token_count(feedback),
                    cumulative_tool_return_tokens=cumulative_tool_tokens + token_count(feedback),
                    cumulative_estimated_context_tokens=cumulative_context_tokens,
                    cumulative_actual_input_tokens=cumulative_actual_tokens)
                response, identifier = self.invoke(role,
                    instructions + '\n查询配额已经用完。依据明确的校验反馈修订最终响应，保留真实证据编号，tool_requests必须为空列表。',
                    context, final_schema, metric)
                recorded = self.store.record_agent_context(metric['context_id'])
                cumulative_context_tokens = recorded['cumulative_estimated_context_tokens']
                cumulative_actual_tokens = recorded['cumulative_actual_input_tokens']
                cumulative_tool_tokens += token_count(feedback)
                self.store.save_agent_session(identity, dict(context=context, rounds=rounds,
                    cumulative_context_tokens=cumulative_context_tokens,
                    cumulative_tool_tokens=cumulative_tool_tokens,
                    cumulative_actual_tokens=cumulative_actual_tokens,
                    response=response.model_dump(mode='json'), evidence_id=identifier))
                continue
            if rounds >= self.config.max_tool_rounds and role == 'JointStrategyAgent':
                if feedback is not None:
                    context = dict(context, unresolved_validation_feedback=feedback)
                    self.store.save_agent_session(identity, dict(context=context, rounds=rounds,
                        cumulative_context_tokens=cumulative_context_tokens,
                        cumulative_tool_tokens=cumulative_tool_tokens,
                        cumulative_actual_tokens=cumulative_actual_tokens,
                        response=response.model_dump(mode='json'), evidence_id=identifier))
                break
            rounds += 1
            if rounds > self.config.max_tool_rounds:
                raise ValueError('当前角色超过配置的图查询轮数')
            results = [execute_query(item) for query in requests
                for item in self.expand_preview_query(query)]
            if feedback is not None:
                results.append(feedback)
            if role == 'JointStrategyAgent':
                preview_facts = dict(context.get('available_preview_evidence', {}))
                for receipt in results:
                    body = receipt.get('result', {})
                    effect_identity = body.get('strategy_effect_identity')
                    if effect_identity is None:
                        continue
                    preview_facts.pop(effect_identity, None)
                    preview_facts[effect_identity] = dict(evidence_id=receipt['evidence_id'],
                        changed_settings=body['execution_effects']['changed_settings'],
                        parent_plan_id=receipt['tool']['transformation_intent']['parent_plan_id'])
                preview_budget = 1800 if self.config.agent_request_profile == 'compact' else 3000
                while preview_facts and token_count(preview_facts) > preview_budget:
                    preview_facts.pop(next(iter(preview_facts)))
                context['available_preview_evidence'] = preview_facts
                family_facts = dict(context.get('queried_global_families', {}))
                for receipt in results:
                    if (not isinstance(receipt, dict)
                            or not isinstance(receipt.get('tool'), dict)
                            or receipt['tool'].get('tool') != 'inspect_global_families'):
                        continue
                    for family in receipt['result'].get('families', []):
                        family_facts.pop(family['family_id'], None)
                        family_facts[family['family_id']] = dict(family,
                            evidence_id=receipt['evidence_id'])
                family_budget = 2000 if self.config.agent_request_profile == 'compact' else 4000
                while family_facts and token_count(family_facts) > family_budget:
                    family_facts.pop(next(iter(family_facts)))
                context['queried_global_families'] = family_facts
            current_tool_tokens = token_count(results)
            cumulative_tool_tokens += current_tool_tokens
            history = list(context.get('tool_history', [])) + results
            receipts = list(context.get('query_receipts', []))
            history_budget = 2400 if self.config.agent_request_profile == 'compact' else 7000
            minimum_history = 1 if self.config.agent_request_profile == 'compact' else 0
            while len(history) > minimum_history and token_count(history) > history_budget:
                removed = history.pop(0)
                reference = dict(evidence_id=removed['evidence_id'], tool=removed['tool']['tool'],
                    result_tokens=removed['result_tokens'])
                if self.config.agent_request_profile == 'compact':
                    reference['result_access'] = removed['tool']
                    reference['status'] = 'DETAILS_DEFERRED_WITH_ORIGINAL_QUERY'
                receipts.append(reference)
            context.update(tool_history=history, query_receipts=receipts[-16:],
                previous_response=response.model_dump(mode='json', exclude_none=True))
            self.store.save_agent_session(identity, dict(context=context, rounds=rounds,
                current_tool_tokens=current_tool_tokens,
                cumulative_context_tokens=cumulative_context_tokens, cumulative_tool_tokens=cumulative_tool_tokens,
                cumulative_actual_tokens=cumulative_actual_tokens))
            metric = dict(context_id=fingerprint(dict(session=identity, tool_round=rounds, context=context)),
                tool_round=rounds, current_tool_return_tokens=current_tool_tokens,
                cumulative_tool_return_tokens=cumulative_tool_tokens,
                cumulative_estimated_context_tokens=cumulative_context_tokens,
                cumulative_actual_input_tokens=cumulative_actual_tokens)
            continuation = ('\n根据工具返回的真实证据继续调查或输出最终结果。'
                + ('确定性校验给出了修订要求。请逐项修订数量、证据引用及冲突变量；保持GBS和Sequence。'
                   if feedback is not None else '')
                + ('已达到工具调查轮数上限，本次必须输出最终结果并将tool_requests设为空列表。'
                   if rounds == self.config.max_tool_rounds else ''))
            response, identifier = self.invoke(role,
                instructions + continuation,
                context, final_schema if rounds == self.config.max_tool_rounds else schema, metric)
            recorded = self.store.record_agent_context(metric['context_id'])
            cumulative_context_tokens = recorded['cumulative_estimated_context_tokens']
            cumulative_actual_tokens = recorded['cumulative_actual_input_tokens']
            self.store.save_agent_session(identity, dict(context=context, rounds=rounds,
                cumulative_context_tokens=cumulative_context_tokens, cumulative_tool_tokens=cumulative_tool_tokens,
                cumulative_actual_tokens=cumulative_actual_tokens,
                response=response.model_dump(mode='json'), evidence_id=identifier))
        if self.evidence_reference_feedback(role, response) is not None:
            raise ValueError('Agent最终响应的证据引用未通过校验')
        return response, identifier

    def initial_configuration(self):
        messages = self.store.search_states()
        initial = [row for row in messages if row['kind'] == 'INITIAL_CONFIGURATION']
        if messages and not initial:
            raise ValueError('Search State Repository已有历史记录但缺少初始配置')
        if len(initial) > 1:
            raise ValueError('Search State Repository包含多个初始配置')
        if initial:
            document = initial[0]['document']
            request = document['request']
            self.require_fixed_training_semantics(self.config.request, request)
            recorded = [row for row in self.store.trials()
                if row['plan_hash'] == document['plan_hash']]
            if len(recorded) != 1:
                raise ValueError('Search State Repository中的初始配置缺少唯一的训练计划')
            plan = TrainingPlan.model_validate(recorded[0]['plan'])
            current = self.plan(request)
            if (plan.plan_hash != document['plan_hash']
                    or plan.plan_id != document['plan_id']
                    or plan.backend_config != current.backend_config):
                raise ValueError('Search State Repository中的初始配置身份校验失败')
            if current.model_copy(update={'backend_fingerprint': plan.backend_fingerprint}) != plan:
                raise ValueError('当前模型、硬件或训练语义与保存的初始计划不同')
            return request, dict(document, state_id=initial[0]['state_id'],
                plan=plan.model_dump(mode='json'))
        planner = SeedPlanner(self.rewriter)
        if self.config.initial_request_from_config:
            request = dict(self.config.request)
            planner.random_initial_selection = dict(
                random_seed=self.config.initial_configuration_seed,
                selected_index=None,
                legal_candidate_count=len(planner.global_families(self.config.request)),
                selection_policy='fixed_shared_initial_configuration',
                prediction_used=False, measurement_used=False)
        else:
            request = planner.random_initial_plan(
                self.config.request, self.config.initial_configuration_seed)
        plan = self.plan(request)
        document = dict(request=request, plan_id=plan.plan_id, plan_hash=plan.plan_hash,
            plan=plan.model_dump(mode='json'),
            source=('fixed_shared_initial_configuration' if self.config.initial_request_from_config
                else 'uniform_random_legal_global_parallel_configuration'),
            **planner.random_initial_selection)
        state_id = self.store.add_search_state('INITIAL_CONFIGURATION', document)
        return request, dict(document, state_id=state_id)

    def record_initial_configuration_result(self, trial):
        measurement = trial.get('measurement')
        prediction = trial.get('prediction')
        if measurement is not None:
            status = measurement['status']
            evidence_id = measurement['run_id']
        elif prediction is not None:
            status = 'PREDICTED'
            evidence_id = prediction['prediction_id']
        else:
            status = trial['state']
            evidence_id = None
        return self.store.add_search_state('INITIAL_CONFIGURATION_RESULT', dict(
            trial_id=trial['trial_id'], plan_hash=trial['plan_hash'], status=status,
            evidence_id=evidence_id, failure=trial.get('failure')))

    def initialize(self):
        initial_request, initial_record = self.initial_configuration()
        initial_plan = TrainingPlan.model_validate(initial_record['plan'])
        initial_trial, _ = self.store.put_trial(initial_plan, hypothesis=dict(
            kind='uniform_random_legal_global_parallel_configuration',
            search_state_id=initial_record.get('state_id'),
            random_seed=initial_record['random_seed'],
            legal_candidate_count=initial_record['legal_candidate_count']))
        directory = Path(self.config.baseline_directory)
        verified = None
        if (directory / 'derived-baseline.json').exists():
            from experiments.derive_effective_execution import load_verified_baseline
            verified = load_verified_baseline(directory)
        if verified is not None or (directory / 'result.json').is_file():
            measured_request = verified['request'] if verified is not None else json.loads(
                (directory / 'request.json').read_text())
            manifests = [directory / 'artifacts' / f'rank{rank}.manifest.json'
                for rank in range(self.topology.num_gpus)]
            if all(path.is_file() for path in manifests):
                actual = [json.loads(path.read_text()) for path in manifests]
                bound = {}
                for field in ('precision_aware_optimizer', 'optimizer_transfer_overlap', 'optimizer_offload_fraction'):
                    values = {row[field] for row in actual if field in row}
                    if len(values) > 1:
                        raise ValueError('Baseline各Rank的实际运行设置不同：' + field)
                    if len(values) == 1:
                        bound[field] = values.pop()
                precision = {row['params_dtype'].removeprefix('torch.') for row in actual}
                if len(precision) != 1:
                    raise ValueError('Baseline各Rank的参数精度不同')
                bound['precision'] = precision.pop()
                for field in ('main_grads', 'main_params', 'exp_avg', 'exp_avg_sq'):
                    values = {row['optimizer_state_dtypes'][field].removeprefix('torch.') for row in actual}
                    if len(values) != 1:
                        raise ValueError('Baseline各Rank的Optimizer状态精度不同：' + field)
                    bound[field + '_dtype'] = values.pop()
                measured_request = dict(measured_request, **bound)
                self.store.add_search_state('BASELINE_RUNTIME_BINDING', dict(
                    source='current_baseline_actual_rank_manifests', run_id=measured_request['run_id'],
                    bound_fields=bound, source_files=[str(path) for path in manifests]))
            measured_plan = self.plan(measured_request)
            persisted_measurement = initial_trial.get('measurement')
            baseline_measurement = (verified['result'] if verified is not None
                else json.loads((directory / 'result.json').read_text()))
            if (persisted_measurement is not None
                    and {key: value for key, value in persisted_measurement.items()
                        if key != 'memory_failure_witness'} == baseline_measurement
                    and normalized_execution(measured_request) == normalized_execution(initial_request)):
                measured_plan = measured_plan.model_copy(update=dict(
                    backend_fingerprint=initial_plan.backend_fingerprint))
                if baseline_measurement['status'] != 'PASS':
                    # 后端能力目录不改变既有失败配置的生效状态设置。
                    measured_state = {key: value for key, value in measured_plan.state_sharding.items()
                        if not key.startswith('supported_') and not key.endswith('_supported')}
                    initial_state = {key: value for key, value in initial_plan.state_sharding.items()
                        if not key.startswith('supported_') and not key.endswith('_supported')}
                    if measured_state == initial_state:
                        measured_plan = measured_plan.model_copy(update=dict(
                            state_sharding=initial_plan.state_sharding))
            reusable = measured_plan.plan_hash == initial_plan.plan_hash
            if self.config.shared_baseline_required and not reusable:
                initial_fields = initial_plan.model_dump(exclude={'plan_id', 'parent_plan_id', 'transformations', 'backend_config'})
                measured_fields = measured_plan.model_dump(exclude={'plan_id', 'parent_plan_id', 'transformations', 'backend_config'})
                raise ValueError(dict(status='COMMON_BASELINE_MISMATCH',
                    backend_differences={key: dict(initial=value,
                        measured=normalized_execution(measured_request).get(key))
                        for key, value in normalized_execution(initial_request).items()
                        if normalized_execution(measured_request).get(key) != value},
                    differences={key: dict(initial=value, measured=measured_fields[key])
                        for key, value in initial_fields.items() if measured_fields[key] != value}))
            self.store.add_search_state('BASELINE_REUSE_DECISION', dict(
                initial_plan_hash=initial_plan.plan_hash,
                measured_plan_hash=measured_plan.plan_hash,
                reusable=reusable))
            if reusable:
                measurement = verified['result'] if verified is not None else json.loads(
                    (directory / 'result.json').read_text())
                witness = None
                if 'OOM' in measurement['status']:
                    if initial_trial.get('measurement') is not None and initial_trial.get('failure') is not None:
                        measurement = initial_trial['measurement']
                        witness = initial_trial['failure']
                    else:
                        witness = write_memory_failure_witness(measurement,
                            measured_plan.model_dump(mode='json'), directory, world_size=self.topology.num_gpus)
                    measurement['memory_failure_witness'] = witness
                    self.store.add_evidence(witness['witness_id'], 'FAILURE', witness)
                if measurement['status'] == 'PASS' and not measurement['performance_eligible']:
                    raise ValueError('初始配置的复用测量需要通过正式性能测量')
                if (measurement['status'] == 'PASS'
                        and not self.baseline_hardware_matches(measurement, measured_request, directory)):
                    raise ValueError('初始配置的真实测量缺少匹配当前GPU与Rank放置的硬件证据')
                evidence_kind = 'MEASURED' if measurement['status'] == 'PASS' else 'FAILURE'
                self.store.add_evidence(measurement['run_id'], evidence_kind, measurement)
                if initial_trial['measurement'] is None:
                    initial_trial = self.store.update_trial(initial_trial['trial_id'], measurement=measurement,
                        state='MEASURED' if measurement['status'] == 'PASS' else 'FAILED',
                        failure=witness)
                if (measurement['status'] != 'PASS'
                        and self.store.charged(initial_plan.plan_hash + ':trial') is None):
                    self.store.charge(initial_plan.plan_hash + ':trial', 'full_training_trials', 1)
                    self.store.charge(initial_plan.plan_hash + ':gpu', 'gpu_hours',
                        self.topology.num_gpus * measurement['gpu_allocation_seconds'] / 3600)
        initial_trial = next(row for row in self.store.trials()
            if row['trial_id'] == initial_trial['trial_id'])
        if initial_trial['measurement'] is None:
            if self.config.shared_baseline_required:
                raise ValueError('共同基线尚未成功复用，停止启动新的基线训练')
            if not self.config.gpu_execution_enabled:
                initial_trial = self.predict_trial(initial_trial)
                self.record_initial_configuration_result(initial_trial)
                return initial_trial
            self.store.save_round(-1, stage='INITIAL_CONFIGURATION_TRAINING',
                current_trial_id=initial_trial['trial_id'])
            measured = self.execute_trial(initial_trial)
            if measured is None:
                return None
            initial_trial = measured
        self.record_initial_configuration_result(initial_trial)
        completed_rounds = int(self.store.usage().get('agent_rounds', 0))
        best = self.store.best_measured() if completed_rounds else initial_trial
        if best is None:
            best = initial_trial
        if best['measurement'] is not None and best['measurement']['status'] != 'PASS':
            self.store.save_round(-1, stage='COMPLETE', best_trial_id=best['trial_id'])
            return best
        request = best['plan']['backend_config']
        self.profile_id, self.profile = select_hardware_profile(
            self.predictor.database, request, self.topology.devices, self.profile_id)
        if self.profile is None:
            missing = self.predict_request(request)
            missing = dict(missing, evidence=dict(missing['evidence'], partition_specific=True))
            pending = self.request_profiles(best, missing, native_only=True)
            self.store.add_search_state('COST_MEASUREMENT_AVAILABLE_TO_AGENT', dict(
                trial_id=best['trial_id'], profiling_request_id=pending['request_id'],
                reason='保留实测策略，成本补测与新策略调查由评价Agent共同选择'))
            if (self.config.initial_cost_calibration_enabled and self.config.gpu_execution_enabled
                    and completed_rounds == 0):
                self.store.save_round(-1, stage='INITIAL_COMPUTE_COSTS',
                    current_trial_id=best['trial_id'], profiling_request_id=pending['request_id'])
                measured = self.execute_profile(pending['request_id'])
                if measured['state'] == 'OUT_OF_BUDGET':
                    raise SearchBudgetExhausted('初始成本测量预算不足')
                if measured['state'] != 'COMPLETED':
                    raise RuntimeError('初始成本采集未完成：' + measured['request_id']
                        + '，状态为' + measured['state'])
                self.profile_id, self.profile = select_hardware_profile(
                    self.predictor.database, request, self.topology.devices, self.profile_id)
                if self.profile is None:
                    raise ValueError('初始成本采集结果缺少适用于当前硬件的Profile')
            else:
                self.store.save_round(-1, stage='COMPLETE', best_trial_id=best['trial_id'])
                return best
        if self.profile is not None:
            self.profile = self.predictor._special_module_breakdown(self.profile)
            profile_evidence = dict(
                kind='measured_cost_profile', id=self.profile['id'],
                hardware=self.profile['hardware'], source_hashes=self.profile['source_hashes'],
                regions=self.profile['regions'], layers=self.profile['layers'])
            existing = self.store.connection.execute(
                'SELECT kind, document FROM evidence WHERE id=?', (self.profile['id'],)).fetchone()
            if existing is None:
                self.store.add_evidence(self.profile['id'], 'MEASURED', profile_evidence)
            elif existing[0] != 'MEASURED' or json.loads(existing[1]).get('id') != self.profile['id']:
                raise ValueError('已登记成本证据身份与当前Profile不同')
        if self.profile is not None:
            best = self.predict_trial(self.store.best_measured())
        if self.profile is not None and self.config.gpu_execution_enabled and not completed_rounds:
            pending = [row for row in self.store.profile_requests()
                if row['measurement_type'] == 'nccl'
                and best['trial_id'] in row['trial_ids']
                and row['state'] in ('PENDING', 'APPROVED', 'RUNNING', 'DEFERRED')
                and max(row['execution_request'].get('nccl_group_sizes', [1])) > 1]
            for dtype in sorted({row['execution_request'].get('nccl_dtype', 'bfloat16') for row in pending}):
                selected = next(row for row in pending
                    if row['execution_request'].get('nccl_dtype', 'bfloat16') == dtype)
                self.store.save_round(-1, stage='INITIAL_COMMUNICATION_COSTS',
                    current_trial_id=best['trial_id'], profiling_request_id=selected['request_id'])
                self.execute_profile(selected['request_id'])
        self.store.save_round(-1, stage='COMPLETE', best_trial_id=best['trial_id'])
        return self.store.best_measured()

    def baseline_hardware_matches(self, measurement, measured_request, directory):
        artifacts = Path(measurement.get('raw_artifacts', directory / 'artifacts'))
        mapping = measured_request.get('rank_mapping', list(range(self.topology.num_gpus)))
        actual = {row['physical_rank']: gpu_uuid(row) for row in self.topology.devices}
        manifests = [artifacts / f'rank{rank}.manifest.json' for rank in range(self.topology.num_gpus)]
        if all(path.is_file() for path in manifests):
            rows = [json.loads(path.read_text()) for path in manifests]
            observed = [gpu_uuid(row) for row in rows]
            expected = [actual.get(rank) for rank in mapping]
            if all(value is not None for value in observed):
                return all(value is not None for value in expected) and observed == expected
        if self.profile is not None and profile_hardware_matches(self.profile, self.topology.devices):
            anchor = self.profile.get('anchor')
            return anchor is not None and anchor.get('run_id') == measurement['run_id']
        return False

    def calibrate_measured_parent(self, trial):
        if (self.prepare_only or not self.config.initial_cost_calibration_enabled
                or not self.config.gpu_execution_enabled
                or (trial.get('measurement') or {}).get('status') != 'PASS'):
            return trial
        request = trial['plan']['backend_config']
        if self.profile is not None and same_training_plan(self.profile['request'], request):
            if (self.profile.get('anchor') or {}).get('run_id') == trial['measurement']['run_id']:
                return trial
            anchor_directory = self.measured_anchor_directory(request)
            if anchor_directory is None:
                raise ValueError('已测父配置缺少相容的完整训练基准')
            profile_id = self.predictor.ingest(self.profile['source_directory'], anchor_directory=anchor_directory)
            self.synchronize_cost_revision()
            self.profile_id, self.profile = select_hardware_profile(
                self.predictor.database, request, self.topology.devices, profile_id)
            self.profile = self.predictor._special_module_breakdown(self.profile)
            self.predictor.set_preferred_profile(self.profile_id)
            return self.predict_trial(trial, force=True)
        prediction = self.predict_request(request)
        prediction = dict(prediction, evidence=dict(prediction['evidence'], partition_specific=True))
        pending = self.request_profiles(trial, prediction, native_only=True)
        finished = self.execute_profile(pending['request_id'])
        self.store.add_search_state('MEASURED_PARENT_COST_CALIBRATION', dict(
            trial_id=trial['trial_id'], measurement_run_id=trial['measurement']['run_id'],
            profiling_request_id=finished['request_id'], state=finished['state'],
            scope='current_measured_parent_exact_training_configuration'))
        if finished['state'] != 'COMPLETED':
            return trial
        profile_id = finished['result']['profile_id']
        self.profile_id, self.profile = select_hardware_profile(
            self.predictor.database, request, self.topology.devices, profile_id)
        if self.profile is None or not same_training_plan(self.profile['request'], request):
            raise ValueError('父配置成本采集结果与实际完整训练配置不同')
        self.profile = self.predictor._special_module_breakdown(self.profile)
        self.predictor.set_preferred_profile(self.profile_id)
        trial = self.predict_trial(trial, force=True)
        pending_nccl = [row for row in self.store.profile_requests()
            if row['measurement_type'] == 'nccl' and trial['trial_id'] in row['trial_ids']
            and row['state'] in ('PENDING', 'APPROVED', 'DEFERRED')
            and max(row['execution_request'].get('nccl_group_sizes', [1])) > 1]
        for dtype in sorted({row['execution_request'].get('nccl_dtype', 'bfloat16')
                for row in pending_nccl}):
            selected = next(row for row in pending_nccl
                if row['execution_request'].get('nccl_dtype', 'bfloat16') == dtype)
            self.execute_profile(selected['request_id'])
        return next(row for row in self.store.trials() if row['trial_id'] == trial['trial_id'])

    def predict_candidate(self, row, parent, hypothesis, force=False):
        row = dict(row, request=self.scoped_candidate_request(row['request']))
        plan = self.plan(row['request'], parent['trial_id'], [hypothesis.model_dump(mode='json')])
        trial, added = self.store.put_trial(plan, hypothesis.model_dump(mode='json'), row['graph_delta'],
            rewrite={key: row[key] for key in ('effect_preview', 'active_decisions',
                'completed_decisions', 'constraint_result', 'layout_conversions', 'process_groups',
                'candidate_source', 'proposal_identity', 'proposal_kind', 'global_family_id') if key in row}
                | (dict(candidate_source='complete_strategy') if row.get('proposal_kind') == 'complete_strategy' else {}))
        return self.predict_trial(trial, row.get('execution_graph'), parent['plan']['backend_config'], force=force)

    def request_profiles(self, trial, prediction, native_only=False):
        descriptions = prediction['evidence'].get('profiling_requests', [])
        root = Path(__file__).resolve().parents[1]
        runtime_revision = fingerprint({name: fingerprint((root / name).read_text())
            for name in ('runtime.py', 'communication_calibration.py')})
        output = []
        optimizer_items = []
        for item in descriptions:
            identity = item.get('identity', {})
            if item.get('kind') == 'optimizer_component' or (item.get('kind') == 'operator_cost' and (
                    identity.get('operation') in ('local_optimizer_update_and_copy', 'optimizer_step')
                    or {'physical_rank', 'parameter_count', 'optimizer_sharding'} <= set(identity))):
                optimizer_items.append(item)
        if optimizer_items and not native_only:
            request = dict(trial['plan']['backend_config'])
            plan_key = plan_signature(request)
            identifier = 'optimizer-profile-' + fingerprint(dict(plan_key=plan_key,
                runtime_revision=runtime_revision)) + '-' + fingerprint(str(self.output.resolve()))[:12]
            request.update(run_id=identifier, warmup=self.config.budget.profiling_warmup,
                measured=self.config.budget.profiling_measured,
                port=35000 + int(plan_key[-4:], 16) % 1000,
                paramorph_calibration=True, paramorph_optimizer_calibration=True,
                paramorph_operator_calibration=False, paramorph_memory_audit=False)
            keys = sorted({item.get('cost_key') or fingerprint(item['identity']) for item in optimizer_items})
            output.append(self.store.put_profile_request(ProfilingRequest(request_id=identifier,
                trial_ids=[trial['trial_id']], cost_keys=keys,
                measurement_type='native_training_cost_profile', execution_request=request,
                reason='在候选原始完整计划下统一采集全部Rank缺失的本地Optimizer组件成本',
                estimated_gpu_hours=self.config.budget.estimated_trial_seconds * self.topology.num_gpus / 3600,
                priority=1)))
        if not native_only:
            unique_descriptions = {item.get('cost_key') or fingerprint(item.get('identity', item)): item
                                   for item in descriptions}
            for item in unique_descriptions.values():
                kind = item.get('kind')
                identity = item.get('identity', item)
                if kind not in ('nccl', 'memory', 'compute_communication_overlap',
                        'representative_module'):
                    continue
                request = dict(self.config.request if kind == 'nccl' else trial['plan']['backend_config'])
                if kind == 'nccl':
                    ranks = identity.get('physical_ranks', identity.get('ranks'))
                    payload = identity.get('payload_bytes')
                    if not ranks or payload is None:
                        continue
                    if any(type(rank) is not int or not 0 <= rank < self.topology.num_gpus for rank in ranks):
                        raise ValueError('NCCL请求引用了未知物理Rank')
                    dtype = identity.get('dtype', 'bfloat16').removeprefix('torch.')
                    element_bytes = {'bfloat16': 2, 'float16': 2, 'float32': 4, 'uint8': 1}[dtype]
                    if payload % element_bytes:
                        raise ValueError('NCCL请求的消息字节数与dtype不一致')
                    request.update(nccl_profile=True, nccl_rank_groups=[ranks], nccl_group_sizes=[len(ranks)],
                        nccl_payloads_bytes=[payload],
                        nccl_payloads_mib=[1], nccl_dtype=dtype)
                elif kind == 'memory':
                    request['paramorph_memory_audit'] = True
                elif kind == 'representative_module':
                    for name in ('tp', 'cp', 'ep', 'etp', 'micro_batch'):
                        if name in identity:
                            request[name] = identity[name]
                    world = request['node_count'] * request['gpus_per_node']
                    if request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp':
                        legal_pp = [pp for pp in self.domains['pp']
                            if pp <= self.model.num_layers
                            and world % (request['tp'] * request['cp'] * pp) == 0
                            and world % (request.get('etp', request['tp']) * request['ep'] * pp) == 0]
                        if not legal_pp:
                            raise ValueError('代表模块找不到可执行的Pipeline并行配置')
                        request['pp'] = max(legal_pp)
                        request['dp'] = world // (request['tp'] * request['cp'] * request['pp'])
                        request['expert_dp'] = world // (
                            request.get('etp', request['tp']) * request['ep'] * request['pp'])
                    request.update(vpp=1, group=1, pipeline_schedule='1f1b', overlap_p2p=False,
                        decoder_chunks=[self.model.num_layers // request['pp'] +
                            (index < self.model.num_layers % request['pp'])
                            for index in range(request['pp'])],
                        paramorph_memory_audit=True, paramorph_serial_calibration_steps=0)
                else:
                    ranks = identity.get('physical_group_ranks')
                    payload = identity.get('payload_bytes')
                    if not ranks or payload is None:
                        continue
                    request.update(compute_communication_profile=True,
                        overlap_group_ranks=ranks,
                        overlap_collective=identity['collective'],
                        overlap_payload_bytes=payload,
                        overlap_compute_shape=dict(
                            M=request['micro_batch'] * request['sequence'] // request['cp'],
                            N=self.model.ffn_hidden_size // request['tp'],
                            K=self.model.hidden_size))
                key = item.get('cost_key', fingerprint(identity or request))
                if kind == 'representative_module':
                    condition = dict(model=request['model'],
                        runtime_revision=runtime_revision,
                        checkpoint=request['checkpoint'], precision=request.get('precision', 'bfloat16'),
                        execution_signature=plan_signature(request),
                        recompute=request['recompute'], execution_backend=request.get(
                            'execution_backend', 'megatron_ddp'), optimizer_sharding=request.get(
                            'optimizer_sharding', 'distributed'), dense_sharding_strategy=request.get(
                            'dense_sharding_strategy', 'no_shard'), expert_sharding_strategy=request.get(
                            'expert_sharding_strategy', 'no_shard'), dp_inner_size=request.get('dp_inner_size'),
                        expert_dp_inner_size=request.get('expert_dp_inner_size'))
                    condition_key = fingerprint(condition)
                else:
                    condition_key = (key if kind == 'nccl' else fingerprint(dict(
                        cost_key=key, plan_signature=plan_signature(request))))
                condition_key = fingerprint(dict(cost_condition=condition_key,
                    execution_request={name: value for name, value in request.items()
                        if name not in ('run_id', 'port', 'warmup', 'measured')},
                    warmup=self.config.budget.profiling_warmup,
                    measured=self.config.budget.profiling_measured))
                identifier = kind + '-' + condition_key + '-' + fingerprint(str(self.output.resolve()))[:12]
                request.update(run_id=identifier, warmup=self.config.budget.profiling_warmup,
                    measured=self.config.budget.profiling_measured,
                    port=35000 + int(condition_key[-4:], 16) % 1000)
                measurement_type = ('compute_communication' if kind == 'compute_communication_overlap'
                    else 'native_training_cost_profile' if kind == 'representative_module' else kind)
                output.append(self.store.put_profile_request(ProfilingRequest(request_id=identifier,
                    trial_ids=[trial['trial_id']], cost_keys=[key], measurement_type=measurement_type,
                    execution_request=request,
                    reason=(item.get('reason') or
                        (prediction['evidence'].get('missing_profiles', ['补充适用成本'])[0]
                         if prediction['evidence'].get('missing_profiles') else '补充适用成本')),
                    estimated_gpu_hours=self.config.budget.estimated_trial_seconds * self.topology.num_gpus / 3600,
                    priority=3 if kind == 'representative_module' else 2)))
        if not native_only:
            return output[0] if output else None
        if output:
            return output[0]
        request = dict(trial['plan']['backend_config'])
        slots = request['pp'] * request['vpp']
        layers = self.model.num_layers
        exact_partition = prediction['evidence'].get('partition_specific', False)
        if not exact_partition:
            request['decoder_chunks'] = [layers // slots + (index < layers % slots) for index in range(slots)]
        identity = 'profile-' + fingerprint(dict(
            plan=plan_signature(request) if exact_partition else signature(request),
            runtime_revision=runtime_revision, optimizer_calibration=True,
            operator_calibration=self.config.boundary_cost_calibration_enabled,
            operator_scope='boundary' if self.config.boundary_cost_calibration_enabled else 'all')) + '-' + fingerprint(str(self.output.resolve()))[:12]
        profile_port = (36500 + int(fingerprint(identity)[-4:], 16) % 1000
            if 'rendezvous-retry' in self.config.run_namespace
            else 35000 + int(signature(request)[-4:], 16) % 1000)
        request.update(run_id=identity, warmup=self.config.budget.profiling_warmup,
            measured=self.config.budget.profiling_measured,
            port=profile_port,
            paramorph_calibration=True, paramorph_optimizer_calibration=True,
            paramorph_operator_calibration=self.config.boundary_cost_calibration_enabled,
            paramorph_operator_scope='boundary' if self.config.boundary_cost_calibration_enabled else 'all',
            paramorph_memory_audit=True, paramorph_serial_calibration_steps=0)
        document = ProfilingRequest(request_id=identity, trial_ids=[trial['trial_id']],
            cost_keys=prediction['evidence'].get('missing_profiles') or ['默认计划的代表性计算、通信与显存成本'],
            measurement_type='native_training_cost_profile', execution_request=request,
            reason='候选缺少适用的计算、通信、显存和原生调度成本',
            estimated_gpu_hours=self.config.budget.estimated_trial_seconds * self.topology.num_gpus / 3600,
            priority=1)
        return self.store.put_profile_request(document)

    def execute_profile(self, identifier):
        if self.prepare_only:
            raise ValueError('首轮预规划阶段需要已有成本，GPU成本测量等待资源分配')
        pending = {row['request_id']: row for row in self.store.profile_requests()}
        if identifier not in pending:
            raise ValueError('Agent选择了不存在的Profiling请求')
        request = pending[identifier]
        if request['state'] == 'COMPLETED':
            self.refresh_profile_trials(request)
            return next(row for row in self.store.profile_requests() if row['request_id'] == identifier)
        if request['state'] in ('FAILED', 'CANCELLED'):
            return request
        if not self.config.gpu_execution_enabled:
            return self.store.update_profile_request(identifier, state='DEFERRED',
                reason='当前配置关闭GPU执行，保留成本测量请求')
        self.check_walltime()
        usage, budget = self.store.usage(), self.config.budget
        estimate = request['estimated_gpu_hours']
        is_nccl = request['measurement_type'] == 'nccl'
        execution = dict(request['execution_request'])
        stored_run_id = execution['run_id']
        if self.config.run_namespace:
            stored_run_id += '-' + self.config.run_namespace
        stored = self.output / ('training' if is_nccl else 'profiling') / 'runs' / stored_run_id / 'request.json'
        remaining = min(budget.max_profiling_gpu_hours - usage.get('profiling_gpu_hours', 0),
            budget.max_total_gpu_hours - usage.get('gpu_hours', 0))
        remaining_search_seconds = budget.max_walltime_seconds - self.store.search_seconds()
        insufficient = (remaining * 3600 / self.topology.num_gpus <= 30
            if is_nccl else estimate > remaining)
        insufficient = insufficient or remaining_search_seconds < (
            30 if is_nccl else budget.estimated_trial_seconds)
        if not stored.is_file() and insufficient:
            return self.store.update_profile_request(identifier, state='OUT_OF_BUDGET',
                reason='剩余成本测量时间或者GPU预算不足')
        self.store.update_profile_request(identifier, state='APPROVED')
        self.store.update_profile_request(identifier, state='RUNNING')
        if stored.is_file():
            previous = json.loads(stored.read_text())
            execution['max_allocation_seconds'] = previous.get('max_allocation_seconds')
        else:
            remaining = min(budget.max_profiling_gpu_hours - usage.get('profiling_gpu_hours', 0),
                            budget.max_total_gpu_hours - usage.get('gpu_hours', 0))
            execution['max_allocation_seconds'] = max(1, min(remaining * 3600 / self.topology.num_gpus - 30,
                budget.max_walltime_seconds - self.store.search_seconds()))
        if request['measurement_type'] == 'memory':
            execution['paramorph_memory_audit'] = True
        if is_nccl:
            execution['nccl_profile'] = True
        result, directory = self.adapter.execute(execution, profile=not is_nccl)
        result = json.loads((directory / 'result.json').read_text())
        seconds = result['gpu_allocation_seconds']
        hours = self.topology.num_gpus * seconds / 3600
        self.store.charge(identifier + ':gpu', 'gpu_hours', hours)
        self.store.charge(identifier + ':profile', 'profiling_gpu_hours', hours)
        self.store.add_evidence(identifier, 'MEASURED' if result['status'] == 'PASS' else 'FAILURE', result)
        if result['status'] != 'PASS':
            return self.store.update_profile_request(identifier, state='FAILED', result=result)
        if is_nccl:
            self.predictor.database.add_communication(identifier, result['nccl_benchmark'])
            measured_keys = set(result.get('nccl_batch_cost_keys', []))
            measured_requests = set(result.get('nccl_batch_profile_request_ids', []))
            for sibling in self.store.profile_requests():
                sibling_id = sibling['request_id']
                if (sibling_id == identifier or sibling_id not in measured_requests
                        or sibling['measurement_type'] != 'nccl'
                        or sibling['state'] not in ('PENDING', 'APPROVED', 'RUNNING', 'DEFERRED')
                        or not set(sibling['cost_keys']) <= measured_keys):
                    continue
                shared = dict(status='PASS', shared_measurement_request_id=identifier,
                    shared_measurement_run_id=result['run_id'],
                    covered_cost_keys=sibling['cost_keys'], additional_gpu_hours=0,
                    measurement_source=str(directory / 'result.json'))
                self.store.add_evidence(sibling_id, 'MEASURED', shared)
                self.store.update_profile_request(sibling_id, state='COMPLETED', result=shared)
            finished = self.store.update_profile_request(identifier, state='COMPLETED', result=result)
            self.refresh_profile_trials(finished)
            return next(row for row in self.store.profile_requests() if row['request_id'] == identifier)
        if request['measurement_type'] == 'compute_communication':
            self.predictor.register_overlap_profile(identifier, result['overlap_benchmark'])
            finished = self.store.update_profile_request(identifier, state='COMPLETED', result=result)
            self.refresh_profile_trials(finished)
            return next(row for row in self.store.profile_requests() if row['request_id'] == identifier)
        profile_id = self.predictor.ingest(directory,
            anchor_directory=self.measured_anchor_directory(request['execution_request']))
        self.store.add_evidence(profile_id, 'MEASURED', dict(kind='measured_cost_profile',
            id=profile_id, source_directory=str(directory), run_id=result['run_id']))
        self.store.update_profile_request(identifier, state='COMPLETED',
            result=dict(measurement=result, profile_id=profile_id))
        finished = next(row for row in self.store.profile_requests() if row['request_id'] == identifier)
        self.refresh_profile_trials(finished)
        return next(row for row in self.store.profile_requests() if row['request_id'] == identifier)

    def measured_anchor_directory(self, request):
        for trial in reversed(self.store.trials()):
            measurement = trial.get('measurement') or {}
            if (measurement.get('status') != 'PASS' or measurement.get('performance_eligible') is not True
                    or not same_training_plan(trial['plan']['backend_config'], request)):
                continue
            directory = Path(measurement['raw_artifacts']).parent
            executed = json.loads((directory / 'request.json').read_text())
            if not same_training_plan(executed, request):
                raise ValueError('训练基准的实际配置与已登记父策略不同')
            return directory
        return None

    def refresh_profile_trials(self, request):
        previous_revision = self.cost_revision
        self.synchronize_cost_revision()
        completed = set((request.get('result') or {}).get('refreshed_trial_ids', []))
        requested = set(request['trial_ids'])
        cost_keys = set(request.get('cost_keys', []))
        trials = self.store.trials()
        affected = {trial['trial_id'] for trial in trials
            if trial.get('prediction') is not None and cost_keys.intersection(self.trial_cost_keys(trial))}
        if self.cost_revision != previous_revision:
            affected.update(trial['trial_id'] for trial in trials
                if trial.get('prediction') is not None and trial.get('measurement') is None)
        current = self.store.get_round(int(self.store.usage().get('agent_rounds', 0))) or {}
        active = set(current.get('candidate_ids') or []) | set(current.get('constraint_candidate_ids') or [])
        active.update(current.get('global_family_candidate_ids') or [])
        active.update((current.get('selection') or {}).get('accepted_candidate_ids', []))
        best = self.store.best_measured()
        if best is not None:
            active.add(best['trial_id'])
        targets = (requested | affected) & active
        trials.sort(key=lambda row: (row['trial_id'] != (best or {}).get('trial_id'),
            (row.get('rewrite') or {}).get('candidate_source') != 'complete_strategy'))
        for trial in trials:
            if trial['trial_id'] not in targets or trial['trial_id'] in completed:
                continue
            if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                self.store.update_trial(trial['trial_id'], state='PROFILED_AWAITING_PREDICTION')
                continue
            candidate = self.rewriter.candidate(trial['plan']['backend_config'])
            graph = self.rewriter.compiler.observation(candidate, [], [])['training_graph']
            self.predict_trial(trial, graph, force=True)
            completed.add(trial['trial_id'])
            self.store.update_profile_request(request['request_id'], result=dict(request['result'],
                refreshed_trial_ids=sorted(completed)))

    def execute_load_validation(self, trial):
        if self.prepare_only:
            raise ValueError('首轮预规划阶段不执行GPU加载验证')
        current = trial.get('validation') or {}
        if current.get('load_validation') is not None:
            return current['load_validation']
        if not self.config.gpu_execution_enabled:
            return dict(status='DEFERRED', reason='当前配置关闭GPU执行')
        self.check_walltime()
        usage, budget = self.store.usage(), self.config.budget
        world = self.topology.num_gpus
        validation_seconds = self.config.fsdp_load_validation_seconds
        estimate = min(120.0, budget.estimated_trial_seconds) * world / 3600
        if (usage.get('profiling_gpu_hours', 0) + estimate > budget.max_profiling_gpu_hours
                or usage.get('gpu_hours', 0) + estimate > budget.max_total_gpu_hours):
            return dict(status='OUT_OF_BUDGET')
        request = dict(trial['plan']['backend_config'])
        request.update(run_id=request['run_id'] + '-load-validation', warmup=1, measured=1,
            validation_mode='checkpoint_load_and_single_training_step',
            port=24600 + int(trial['plan_hash'][:6], 16) % 1000,
            max_allocation_seconds=max(1,
                min(validation_seconds, request.get('max_allocation_seconds', validation_seconds),
                    (budget.max_total_gpu_hours - usage.get('gpu_hours', 0)) * 3600 / world - 10,
                    (budget.max_profiling_gpu_hours - usage.get('profiling_gpu_hours', 0)) * 3600 / world - 10,
                    budget.max_walltime_seconds - self.store.search_seconds() - 15)))
        result, directory = self.adapter.execute(request)
        result = json.loads((directory / 'result.json').read_text())
        result['validation_scope'] = 'checkpoint_load_initialization_forward_backward_optimizer_two_steps'
        witness = None
        if 'OOM' in result['status']:
            witness = write_memory_failure_witness(result, trial['plan'], directory,
                trial['prediction'], world)
            result['memory_failure_witness'] = witness
            self.store.add_evidence(witness['witness_id'], 'FAILURE', witness)
        hours = world * result['gpu_allocation_seconds'] / 3600
        identity = trial['plan_hash'] + ':load-validation'
        self.store.charge(identity + ':gpu', 'gpu_hours', hours)
        self.store.charge(identity + ':profile', 'profiling_gpu_hours', hours)
        evidence_id = result['run_id']
        self.store.add_evidence(evidence_id,
            'MEASURED' if result['status'] == 'PASS' else 'FAILURE', result)
        validation = dict(current, load_validation=result)
        self.store.update_trial(trial['trial_id'], validation=validation,
            state='LOAD_VALIDATED' if result['status'] == 'PASS' else 'FAILED',
            failure=witness)
        return result

    def execute_trial(self, trial, exploration=False):
        if self.prepare_only:
            raise ValueError('首轮预规划阶段不执行GPU训练')
        if trial['measurement'] is not None:
            return trial
        if not self.config.gpu_execution_enabled:
            self.store.update_trial(trial['trial_id'], state='DEFERRED',
                failure=dict(reason='当前配置关闭GPU执行，保留完整候选'))
            return None
        if trial['prediction'] is None:
            trial = self.predict_trial(trial)
        memory = trial['prediction']['memory']
        if memory['admission'] == 'REJECT_OOM':
            self.store.add_search_state('TRAINING_MEMORY_REJECTED', dict(
                trial_id=trial['trial_id'], prediction_id=trial['prediction']['prediction_id'],
                memory_admission=memory['admission']))
            return None
        training_peaks = memory.get('gpu_training_peak_per_rank_bytes',
            memory.get('peak_per_rank_bytes', {}))
        predicted_over = any(value > self.topology.memory_per_gpu_gb * 2**30
            for value in training_peaks.values())
        if predicted_over:
            validation = self.execute_load_validation(trial)
            self.store.add_search_state('CAPACITY_VALIDATION_BEFORE_BENCHMARK', dict(
                trial_id=trial['trial_id'], prediction_id=trial['prediction']['prediction_id'],
                analytical_training_peak_per_rank_bytes=training_peaks,
                validation_status=validation['status']))
            if validation['status'] != 'PASS':
                return None
        self.check_walltime()
        usage = self.store.usage()
        budget = self.config.budget
        world = self.topology.num_gpus
        estimate = budget.estimated_trial_seconds * world / 3600
        remaining_search_seconds = budget.max_walltime_seconds - self.store.search_seconds()
        if remaining_search_seconds < budget.estimated_trial_seconds:
            return None
        reserved = self.store.charged(trial['plan_hash'] + ':trial') is not None
        if not reserved and (usage.get('full_training_trials', 0) >= budget.max_full_training_trials
            or usage.get('gpu_hours', 0) + estimate > budget.max_total_gpu_hours):
            return None
        request = dict(trial['plan']['backend_config'])
        request.update(warmup=self.config.benchmark_warmup, measured=self.config.benchmark_measured)
        request['port'] = 23600 + int(trial['plan_hash'][:6], 16) % 1000
        remaining_gpu_seconds = max(1,
            (budget.max_total_gpu_hours - usage.get('gpu_hours', 0)) * 3600 / world - 30)
        declared_trial_seconds = request.get('max_allocation_seconds')
        request['max_allocation_seconds'] = min(remaining_search_seconds,
            remaining_gpu_seconds if declared_trial_seconds is None
            else min(declared_trial_seconds, remaining_gpu_seconds))
        if (trial['prediction'] is not None
                and trial['prediction']['plan_signature'] != plan_signature(request)):
            candidate = self.rewriter.candidate(trial['plan']['backend_config'])
            graph = self.rewriter.compiler.observation(candidate, [], [])['training_graph']
            trial = self.predict_trial(trial, graph, force=True)
        if (trial['prediction']['memory']['admission'] == 'REQUIRE_MEMORY_PROFILE'
                and trial['plan']['backend_config'].get('execution_backend', 'megatron_ddp') == 'megatron_ddp'):
            exploration = True
        if reserved:
            stored = self.output / 'training' / 'runs' / request['run_id'] / 'request.json'
            if stored.is_file():
                request['max_allocation_seconds'] = json.loads(stored.read_text())['max_allocation_seconds']
        if exploration:
            reservation = trial['plan_hash'] + ':exploration'
            if (self.store.charged(reservation) is None and self.store.usage().get('exploration_trials', 0)
                    >= self.config.uncalibrated_exploration_trials):
                return None
            self.store.charge(reservation, 'exploration_trials', 1)
        self.store.charge(trial['plan_hash'] + ':trial', 'full_training_trials', 1)
        self.store.update_trial(trial['trial_id'], state='RUNNING')
        result, directory = self.adapter.execute(request, None if exploration else trial['prediction'])
        result = json.loads((directory / 'result.json').read_text())
        witness = None
        if 'OOM' in result['status']:
            witness = write_memory_failure_witness(result, trial['plan'], directory,
                trial['prediction'], world)
            result['memory_failure_witness'] = witness
            self.store.add_evidence(witness['witness_id'], 'FAILURE', witness)
        self.store.add_evidence(result['run_id'], 'MEASURED' if result['status'] == 'PASS' else 'FAILURE', result)
        seconds = result['gpu_allocation_seconds']
        self.store.charge(trial['plan_hash'] + ':gpu', 'gpu_hours', world * seconds / 3600)
        if self.config.budgeted_investigation_enabled:
            prediction = trial['prediction']
            result['prediction_at_execution'] = dict(prediction_id=prediction['prediction_id'],
                predictor_revision=prediction['evidence'].get('predictor_revision'),
                step_time_ms=prediction['performance'].get('step_time_ms'))
        trial = self.store.update_trial(trial['trial_id'], measurement=result,
            state='MEASURED' if result['status'] == 'PASS' else 'FAILED',
            failure=witness,
            completed_at=datetime.now(timezone.utc).isoformat())
        if (result['status'] == 'PASS' and result.get('performance_eligible') is True
                and trial['prediction'] is not None and trial['prediction']['memory'].get('peak_per_rank_bytes')
                and trial['prediction']['performance']['step_time_ms'] is not None):
            executed_request = json.loads((directory / 'request.json').read_text())
            self.predictor.database.record_validation(executed_request, trial['prediction'], result)
        self.synchronize_cost_revision()
        self.family_ledger.record_trial(trial, self.cost_revision)
        return trial

    def finish(self):
        self.synchronize_cost_revision()
        for trial in self.store.trials():
            self.family_ledger.record_trial(trial, self.cost_revision)
        self.store.record_session_time(self.session_id, time.perf_counter() - self.started)
        context_metrics = self.store.agent_context_metrics()
        context_metrics_path = self.output / 'agent-context-metrics.json'
        context_metrics_path.write_text(json.dumps(
            context_metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        round_costs = [dict(round_index=index, available=row.get('usage_at_start') is not None,
            scope=('measured_round_delta' if row.get('usage_at_start') is not None
                else 'legacy_round_without_start_snapshot'),
            metrics=row.get('call_cost_report') if row.get('usage_at_start') is not None else None)
            for index in range(self.config.budget.max_agent_rounds)
            if (row := self.store.get_round(index)) is not None]
        call_cost_report = dict(rounds=round_costs, by_role=self.agent_role_costs(context_metrics), totals=dict(
            llm_calls=self.store.usage().get('agent_calls', 0),
            input_tokens=self.store.usage().get('agent_input_tokens', 0),
            output_tokens=self.store.usage().get('agent_output_tokens', 0),
            schema_tokens=sum(row.get('strict_schema_tokens', 0) for row in context_metrics),
            tool_result_tokens=sum(row.get('current_tool_return_tokens', 0) for row in context_metrics),
            cpu_prediction_count=self.store.usage().get('cpu_predictions', 0),
            gpu_hours=self.store.usage().get('gpu_hours', 0),
            best_real_tokens_per_s=(self.store.best_measured()['measurement']['tokens_per_second']
                if self.store.best_measured() is not None else None)))
        call_cost_path = self.output / 'agent-call-cost-report.json'
        call_cost_path.write_text(json.dumps(
            call_cost_report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        report = dict(best_measured=self.store.best_measured(), usage=self.store.usage(),
            trials=self.store.trials(), profiling_requests=self.store.profile_requests(),
            search_state_repository=self.store.search_states(),
            agent_roles=AGENT_ROLE_NAMES, global_families=self.store.family_records(),
            cumulative_search_seconds=self.store.search_seconds(),
            gpu_execution_enabled=self.config.gpu_execution_enabled,
            algorithms=['Multi-Agent-Guided Joint Hybrid Parallel Strategy Synthesis',
                'Conditional Multi-Candidate Partition Dynamic Programming',
                'Ranking-Aware Shared-Cost Profiling'],
            predictor='Measurement-Calibrated, Analytical-Backed DAG Performance Projection',
            predictor_cost_mode=self.config.predictor_cost_mode,
            agent_context_accounting=dict(call_count=len(context_metrics),
                actual_input_tokens=sum(row.get('actual_input_tokens', 0) for row in context_metrics),
                tool_return_tokens=sum(row.get('current_tool_return_tokens', 0) for row in context_metrics),
                strict_schema_tokens=sum(row.get('strict_schema_tokens', 0) for row in context_metrics),
                record_path=str(context_metrics_path), call_cost_report_path=str(call_cost_path)),
            numerical_equivalence_scope='see_each_real_measurement')
        if self.config.compact_results:
            report = {key: value for key, value in report.items()
                if key in ('usage', 'status', 'best_trial', 'best_measured', 'search_seconds', 'stop_reason')}
            report['record_database'] = str(self.output / 'experiments.sqlite')
        else:
            self.store.export(self.output / 'trials.jsonl')
        (self.output / 'search-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return report

    def plan_distance(self, first, second):
        left, right = first['plan'], second['plan']
        keys = ('tp', 'pp', 'cp', 'ep', 'dp', 'vpp')
        variables = [left['parallelism'][key] != right['parallelism'][key] for key in keys]
        variables.extend(left.get(name) != right.get(name) for name in
            ('expert_parallelism', 'rank_mapping', 'schedule', 'memory_policy', 'optimizer_sharding',
             'state_sharding', 'num_microbatches'))
        variables.extend(left['backend_config'].get(name) != right['backend_config'].get(name) for name in
                ('micro_batch', 'group', 'overlap_p2p', 'bucket_elements', 'precision_aware_optimizer'))
        boundaries = []
        for plan in (left, right):
            offset, positions = 0, set()
            for count in plan['region_partition'][:-1]:
                offset += count
                positions.add(offset)
            boundaries.append(positions)
        region_difference = len(boundaries[0] ^ boundaries[1]) / max(1, len(boundaries[0] | boundaries[1]))
        return (sum(variables) / len(variables) + region_difference) / 2

    def graph_investigations(self, parent, analysis=None):
        if self.config.agent_ablation == 'without_dag_evidence':
            return []
        if analysis is None:
            analysis = self.execution_decision_report(parent)
        request = dict(parent['plan']['backend_config'], num_moe_experts=self.model.num_experts or None)
        return pipeline_investigations(request, [kind.value for kind in self.model.layer_types], analysis)

    def attempted_execution_shapes(self):
        return {execution_shape(row['plan']['backend_config']) for row in self.store.trials()
            if (row.get('measurement') or {}).get('status') in ('PASS', 'CUDA_OOM', 'FAILED')}

    def attempted_execution_plans(self):
        return {plan_signature(row['plan']['backend_config']) for row in self.store.trials()
            if (row.get('measurement') or {}).get('status') in ('PASS', 'CUDA_OOM', 'FAILED')}

    def reuse_known_measurement(self, trial):
        if trial.get('measurement') is not None or trial.get('failure'):
            return trial
        request = trial['plan']['backend_config']
        for previous in reversed(self.store.trials()):
            measurement = previous.get('measurement') or {}
            other = previous['plan']['backend_config']
            if (measurement.get('status') != 'PASS' or measurement.get('performance_eligible') is not True
                    or any(request.get(name) != other.get(name) for name in ('warmup', 'measured'))
                    or plan_signature(request) != plan_signature(other)):
                continue
            reuse = dict(kind='measured_effective_configuration_reuse',
                trial_id=trial['trial_id'], source_trial_id=previous['trial_id'],
                measured_run_id=measurement['run_id'], new_gpu_execution=False)
            identifier = fingerprint(reuse)
            self.store.add_evidence(identifier, 'MEASURED', reuse)
            return self.store.update_trial(trial['trial_id'], measurement=measurement,
                measurement_reuse=reuse | dict(evidence_id=identifier), state='MEASURED')
        return trial

    def retain_candidate_portfolio(self, candidates, limit, *, parent=None, proposal_candidate_ids=()):
        if type(limit) is not int or limit < 1:
            raise ValueError('候选保留数量必须为正整数')
        candidates = list({row['trial_id']: row for row in candidates}.values())
        eligible = [row for row in candidates if not row.get('failure') and row.get('measurement') is None
            and (row.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM']
        def key(row):
            prediction = row.get('prediction') or {}
            value = prediction.get('performance', {}).get('step_time_ms')
            usable = (value is not None and prediction.get('prediction_status') != 'NEEDS_PROFILE'
                and prediction.get('cost_database_revision') == self.cost_revision
                and prediction.get('evidence', {}).get('predictor_revision') == PROJECTION_REVISION)
            return (not usable, value if usable else float('inf'), row['plan_hash'])
        ordered = sorted(eligible, key=key)
        ranked = [row for row in ordered if not key(row)[0]]
        unknown = [row for row in ordered if key(row)[0]]
        best = self.store.best_measured()
        known_families = {global_identity(row['plan']['backend_config']) for row in ranked}
        unknown.sort(key=lambda row: (
            (self.load_validation_summary(row) or {}).get('status') != 'PASS',
            global_identity(row['plan']['backend_config']) in known_families,
            (row.get('rewrite') or {}).get('candidate_source') != 'complete_strategy',
            -(self.plan_distance(row, best) if best is not None else 0), row['plan_hash']))
        representatives = {}
        for row in ranked:
            representatives.setdefault(global_identity(row['plan']['backend_config']), row)
        complete = [row for row in eligible
            if (row.get('rewrite') or {}).get('candidate_source') == 'complete_strategy']
        by_id = {row['trial_id']: row for row in eligible}
        proposed = [by_id[identifier] for identifier in dict.fromkeys(proposal_candidate_ids) if identifier in by_id]
        selected = []
        for row in [*proposed, *ranked[:1], *complete, *representatives.values(), *ranked, *unknown]:
            if len(selected) >= limit:
                break
            if row not in selected:
                selected.append(row)
        evidence = dict(kind='candidate_portfolio_retention',
            policy='current_agent_investigations_then_complete_prediction_and_family_coverage',
            candidate_count=len(candidates), retained_ids=[row['trial_id'] for row in selected],
            current_proposal_ids=list(proposal_candidate_ids),
            complete_proposal_ids=[row['trial_id'] for row in complete],
            unranked_retained_ids=[row['trial_id'] for row in selected if key(row)[0]],
            source_by_candidate={row['trial_id']: (row.get('rewrite') or {}).get('candidate_source')
                for row in selected},
            rejected_candidates=[dict(trial_id=row['trial_id'],
                candidate_source=(row.get('rewrite') or {}).get('candidate_source'),
                reason=('PREVIOUS_FAILURE' if row.get('failure') else
                    'ALREADY_MEASURED' if row.get('measurement') is not None else
                    'REJECT_OOM' if (row.get('prediction') or {}).get('memory', {}).get('admission') == 'REJECT_OOM'
                    else 'PORTFOLIO_LIMIT'),
                prediction_id=(row.get('prediction') or {}).get('prediction_id'))
                for row in candidates if row not in selected])
        self.store.add_evidence(fingerprint(evidence), 'PREDICTED', evidence)
        return selected

    def selection_metrics(self, candidates):
        best = self.store.best_measured()
        best_ms = best['measurement']['step_seconds'] * 1000 if best else None
        measured = [trial for trial in self.store.trials() if trial['measurement'] is not None]
        output = []
        for trial in candidates:
            prediction = trial['prediction']
            milliseconds = prediction['performance']['step_time_ms']
            distance = min((self.plan_distance(trial, previous) for previous in measured), default=1.0)
            expected_seconds = max(self.config.budget.estimated_trial_seconds,
                (self.config.benchmark_warmup + self.config.benchmark_measured) * milliseconds / 1000
                if milliseconds is not None else self.config.budget.estimated_trial_seconds)
            hours = expected_seconds * self.topology.num_gpus / 3600
            improvement = max(0, (best_ms - milliseconds) / best_ms) if best_ms and milliseconds is not None else None
            sufficient = (milliseconds is not None and prediction.get('prediction_status') != 'NEEDS_PROFILE'
                and prediction['memory']['admission'] == 'WITHIN_ESTIMATED_CAPACITY')
            output.append(dict(trial_id=trial['trial_id'], predicted_improvement=improvement,
                structural_distance=distance, estimated_gpu_hours=hours, estimate_source='fixed_protocol_and_budget_duration_estimate',
                prediction_status=prediction.get('prediction_status'), cost_coverage=prediction['evidence'].get('calibration_coverage'),
                missing_profiles=prediction['evidence'].get('missing_profiles', []),
                memory_admission=prediction['memory']['admission'], cost_sufficient_for_ranking=sufficient))
        return output

    def ranking_sensitivity(self, candidates):
        eligible = [trial for trial in candidates if trial['prediction'] is not None
            and trial['prediction']['performance']['step_time_ms'] is not None
            and trial['prediction']['memory']['admission'] != 'REJECT_OOM']
        eligible.sort(key=lambda trial: (
            trial['prediction']['performance']['step_time_ms'], trial['plan_hash']))
        top_k = min(self.config.ranking_top_k, len(eligible))

        def ranked(overrides=None):
            overrides = overrides or {}
            return [trial['trial_id'] for trial in sorted(eligible, key=lambda trial: (
                overrides.get(trial['trial_id'], trial['prediction']['performance']['step_time_ms']),
                trial['plan_hash']))[:top_k]]

        base = ranked()
        missing_intervals = [trial['trial_id'] for trial in eligible
            if trial['prediction']['performance'].get('prediction_interval_ms') is None]
        scenarios = {trial['trial_id']:
            (trial['prediction']['performance'].get('prediction_interval_ms')
             or trial['prediction']['performance'].get('ranking_sensitivity_scenario_ms')
             or [trial['prediction']['performance']['step_time_ms']] * 2)
            for trial in eligible}
        lower = {trial_id: scenario[0] for trial_id, scenario in scenarios.items()}
        upper = {trial_id: scenario[1] for trial_id, scenario in scenarios.items()}
        optimistic, pessimistic = ranked(lower), ranked(upper)

        estimated_hours = self.config.budget.estimated_trial_seconds * self.topology.num_gpus / 3600
        probes = []
        uncalibrated_signatures = {signature(trial['plan']['backend_config']) for trial in eligible
            if self.predictor.database.select(trial['plan']['backend_config']) is None}
        shared_results = self.shared_ranking_probes(eligible)
        shared_reports = [report for report in shared_results if report.get('status') == 'AVAILABLE']
        unavailable = [report for report in shared_results if report.get('status') == 'UNAVAILABLE']
        for report in shared_reports:
            low = report['scenarios']['low']
            high = report['scenarios']['high']
            low_top = low['ranking'][:top_k]
            high_top = high['ranking'][:top_k]
            changes = self.shared_ranking_change_summary(report, top_k)
            set_change = changes.pop('top_k_symmetric_difference')
            affected = report['affected_candidate_ids']
            scenario_times = [[report['baseline_step_time_ms'][identifier],
                low['step_time_ms'][identifier], high['step_time_ms'][identifier]]
                for identifier in affected]
            impact = max((max(values) - min(values) for values in scenario_times), default=0.0)
            probes.append(dict(cost_key=report['cost_key'], candidates=affected,
                source_profile_id=None, ranking_impact_ms=impact,
                baseline_top_k=report['baseline_ranking'][:top_k],
                low_top_k=low_top, high_top_k=high_top, **changes,
                top_k_symmetric_difference=set_change,
                estimated_gpu_hours=estimated_hours,
                gpu_hours_source='configured_targeted_profile_duration_estimate',
                reuse_count=report['reuse_count'], shared_cost_scenario=True,
                statistical_confidence_interval=False,
                priority_key=[int(changes['best_candidate_changed']),
                    changes['top_k_order_inversions'], set_change, -estimated_hours]))
        probes.sort(key=lambda row: (-int(row['best_candidate_changed']),
            -row['top_k_order_inversions'], -row['top_k_symmetric_difference'],
            row['estimated_gpu_hours'], row['cost_key']))
        return dict(top_k=top_k, base_top_k=base, optimistic_top_k=optimistic,
            pessimistic_top_k=pessimistic,
            **{'interval_' + key: value for key, value in
                self.ranking_change_summary(optimistic, pessimistic).items()},
            interval_top_k_symmetric_difference=len(set(optimistic) ^ set(pessimistic)),
            ranking_stable=optimistic == pessimistic and not missing_intervals,
            ranking_sensitive_costs=probes[:16], uncalibrated_signatures=sorted(uncalibrated_signatures),
            shared_cost_scenario_count=len(shared_reports),
            shared_cost_unavailable=unavailable,
            missing_prediction_intervals=missing_intervals,
            interval_scope='predictor_observed_variability_and_validation_residual_bounds',
            decision_order=['best_candidate_changed', 'top_k_order_inversions',
                'top_k_set_changed', 'estimated_gpu_hours'])

    @staticmethod
    def shared_ranking_change_summary(report, top_k):
        orders = [report['baseline_ranking'][:top_k],
            report['scenarios']['low']['ranking'][:top_k],
            report['scenarios']['high']['ranking'][:top_k]]
        changes = [EvidenceGuidedSearch.ranking_change_summary(left, right)
            for left, right in combinations(orders, 2)]
        return dict(best_candidate_changed=any(row['best_candidate_changed'] for row in changes),
            top_k_order_changed=any(row['top_k_order_changed'] for row in changes),
            top_k_order_inversions=max(row['top_k_order_inversions'] for row in changes),
            top_k_set_changed=any(row['top_k_set_changed'] for row in changes),
            top_k_symmetric_difference=max(len(set(left) ^ set(right))
                for left, right in combinations(orders, 2)))

    @staticmethod
    def ranking_change_summary(low_order, high_order):
        common = sorted(set(low_order) & set(high_order))
        low_position = {identity: index for index, identity in enumerate(low_order)}
        high_position = {identity: index for index, identity in enumerate(high_order)}
        inversions = sum((low_position[first] - low_position[second]) *
            (high_position[first] - high_position[second]) < 0
            for first, second in combinations(common, 2))
        return dict(best_candidate_changed=bool(low_order and high_order
                and low_order[0] != high_order[0]),
            top_k_order_changed=low_order != high_order,
            top_k_order_inversions=inversions,
            top_k_set_changed=set(low_order) != set(high_order))

    def requires_experiment_investigation(self, context, candidates):
        ranking = context['ranking_sensitivity']
        ranking_changed = any(ranking.get(name, False) for name in (
            'interval_best_candidate_changed', 'interval_top_k_order_changed',
            'interval_top_k_set_changed'))
        sensitive_cost = any(row['best_candidate_changed'] or row['top_k_order_inversions']
            or row['top_k_set_changed'] for row in ranking['ranking_sensitive_costs'])
        memory_failure = any(trial.get('failure') is not None
            or (trial.get('measurement') or {}).get('status') in ('CUDA_OOM', 'FAILED')
            for trial in candidates)
        return ranking_changed or sensitive_cost or memory_failure

    def select_joint_investigation(self, candidates, intent_results, strategy_id,
                                   measurement_preference='training', *, allow_profiles=True):
        by_id = {trial['trial_id']: trial for trial in candidates}
        intent_priority = {}
        for entry in sorted(intent_results, key=lambda row: row['intent_index']):
            for identifier in entry['candidate_ids']:
                intent_priority.setdefault(identifier, entry['intent_index'])
        ordered_ids = list(intent_priority)
        eligible = [by_id[identifier] for identifier in ordered_ids if identifier in by_id
            and self.available_for_joint_measurement(by_id[identifier])]
        conditions = {(trial['prediction'].get('cost_database_revision'),
            trial['prediction'].get('evidence', {}).get('predictor_revision'),
            trial['prediction'].get('evidence', {}).get('compatible_anchor_id')) for trial in eligible}
        comparable = (bool(eligible) and len(conditions) == 1 and all(
            trial['prediction']['performance']['step_time_ms'] is not None for trial in eligible))
        if comparable:
            eligible.sort(key=lambda trial: (intent_priority[trial['trial_id']],
                trial['prediction']['performance']['step_time_ms']))
        selected = eligible[:self.config.budget.max_benchmarks_per_round]
        planned_ids = [trial['trial_id'] for trial in selected]
        requested_profiles = []
        capacity_bootstrap = bool(selected) and self.store.best_measured() is None
        if allow_profiles and measurement_preference == 'missing_execution_costs' and not capacity_bootstrap:
            options = self.current_proposal_profiles({row['request_id']: row
                for row in self.store.profile_requests() if row['state'] in ('PENDING', 'APPROVED', 'DEFERRED')},
                intent_results, {row['trial_id'] for row in eligible})
            remaining = self.config.budget.max_profiling_gpu_hours - self.store.usage().get('profiling_gpu_hours', 0)
            options = [row for row in options.values() if 0 < row['estimated_gpu_hours'] <= remaining]
            options.sort(key=lambda row: (-len(set(planned_ids).intersection(row['trial_ids'])),
                -len(row['cost_keys']) / row['estimated_gpu_hours'], row['request_id']))
            if options:
                requested_profiles = [options[0]['request_id']]
                selected = []
        load_ids = [trial['trial_id'] for trial in selected
            if trial['plan']['backend_config'].get('execution_backend') == 'megatron_fsdp'
            and (self.load_validation_summary(trial) or {}).get('status') != 'PASS']
        ids = [trial['trial_id'] for trial in selected if trial['trial_id'] not in load_ids]
        evidence = dict(kind='joint_investigation_selection', strategy_evidence_id=strategy_id,
            agent_candidate_order=ordered_ids, planned_training_candidate_ids=planned_ids,
            selected_candidate_ids=ids, requested_profiles=requested_profiles,
            comparable_complete_predictions=comparable, profile_selection_enabled=allow_profiles,
            capacity_bootstrap=capacity_bootstrap,
            unresolved_candidates=[trial['trial_id'] for trial in eligible
                if trial['prediction']['performance']['step_time_ms'] is None])
        identifier = fingerprint(evidence)
        self.store.add_evidence(identifier, 'PREDICTED', evidence)
        return InvestigationSelection(accepted_candidate_ids=ids,
            accepted_profiling_request_ids=requested_profiles,
            exploration_candidate_ids=[trial['trial_id'] for trial in selected
                if trial['trial_id'] in ids and (trial['prediction']['performance']['step_time_ms'] is None
                    or trial['prediction']['memory']['admission'] != 'WITHIN_ESTIMATED_CAPACITY')],
            load_validation_candidate_ids=load_ids, evidence_refs=[identifier], missing_measurements=[],
            reason=('共同基线没有成功训练；本轮优先验证Agent提出的可执行候选。' if capacity_bootstrap
                else '按Agent给出的调查顺序测量；同一项调查内使用完整可比预测排列候选。')), identifier

    def choose_budgeted_experiment(self, parent, candidates, intent_results, strategy, strategy_id, index):
        checkpoint = self.store.get_round(index)
        if checkpoint.get('budgeted_experiment_decision'):
            saved = checkpoint['budgeted_experiment_decision']
            return InvestigationSelection.model_validate(saved['review']), saved['evidence_id']
        options = checkpoint.get('budgeted_experiment_options')
        exclusions = checkpoint.get('budgeted_experiment_exclusions', [])
        policy = checkpoint.get('budgeted_experiment_policy', 'v1')
        if options is None:
            policy = 'native_effective_exploration_v2'
            budget, usage = self.config.budget, self.store.usage()
            remaining = max(0, budget.max_total_gpu_hours - usage.get('gpu_hours', 0))
            world = self.topology.num_gpus
            owners = {}
            for entry in intent_results:
                for candidate_id in entry['candidate_ids']:
                    owners.setdefault(candidate_id, []).append(entry['intent_index'])
            history = self.store.trials()
            best = self.store.best_measured()
            best_throughput = None if best is None else best['measurement']['tokens_per_second']
            options = []
            measured_effects = {fingerprint(effective_schedule_execution(row['plan']['backend_config']))
                for row in history if (row.get('measurement') or {}).get('status') in
                    ('PASS', 'CUDA_OOM', 'HOST_MEMORY_BUDGET_EXCEEDED')}
            measured_effects.add(fingerprint(effective_schedule_execution(parent['plan']['backend_config'])))
            offered_effects = set()
            for trial in candidates:
                identifier = trial['trial_id']
                if identifier not in owners or not self.available_for_joint_measurement(trial):
                    continue
                request = trial['plan']['backend_config']
                effect = fingerprint(effective_schedule_execution(request))
                if effect in measured_effects or effect in offered_effects:
                    exclusions.append(dict(candidate_id=identifier, reason='原生生效配置与已有记录相同'))
                    continue
                prediction = trial['prediction']
                memory = prediction['memory']
                training_peaks = memory.get('gpu_training_peak_per_rank_bytes', memory.get('peak_per_rank_bytes', {}))
                predicted_over = any(value > self.topology.memory_per_gpu_gb * 2**30
                    for value in training_peaks.values())
                needs_load = (request.get('execution_backend') == 'megatron_fsdp' or predicted_over) and (
                    self.load_validation_summary(trial) or {}).get('status') != 'PASS'
                action = 'LOAD_VALIDATION' if needs_load else 'BENCHMARK'
                minimum = min(120, budget.estimated_trial_seconds) if needs_load else budget.estimated_trial_seconds
                value = experiment_value(request, prediction, history, best_throughput, minimum, action=action)
                required_hours = max(minimum * world / 3600, value['expected_gpu_hours'])
                if required_hours > remaining:
                    exclusions.append(dict(candidate_id=identifier, reason='预计用量超过剩余 GPU 预算',
                        expected_gpu_hours=required_hours, remaining_gpu_hours=remaining))
                    continue
                if needs_load and usage.get('profiling_gpu_hours', 0) + required_hours > budget.max_profiling_gpu_hours:
                    exclusions.append(dict(candidate_id=identifier, reason='超过加载验证预算'))
                    continue
                if not needs_load and usage.get('full_training_trials', 0) >= budget.max_full_training_trials:
                    exclusions.append(dict(candidate_id=identifier, reason='完整训练次数用尽'))
                    continue
                if (not needs_load and (value['uncertainty'] == 'UNKNOWN'
                        or memory['admission'] != 'WITHIN_ESTIMATED_CAPACITY')
                        and usage.get('exploration_trials', 0) >= self.config.uncalibrated_exploration_trials):
                    exclusions.append(dict(candidate_id=identifier, reason='探索测量次数用尽'))
                    continue
                row = dict(candidate_id=identifier, action=action,
                    hypothesis=[strategy.intents[n].hypothesis for n in owners[identifier]],
                    effective_config=self.compact_plan(trial),
                    memory_admission=memory['admission'], **value)
                if self.config.agent_ablation != 'without_dag_evidence':
                    row['graph_delta'] = self.compare_native_schedules(parent, request, prediction, allow_replay=False)
                options.append(row)
                offered_effects.add(effect)
            self.store.save_round(index, budgeted_experiment_options=options,
                budgeted_experiment_exclusions=exclusions, budgeted_experiment_policy=policy)
        values, first_probe = budgeted_selection_values(options,
            gpu_execution_enabled=policy != 'v1' and self.config.gpu_execution_enabled,
            full_training_trials=self.store.usage().get('full_training_trials', 0))
        option_evidence = dict(kind='cross_investigation_experiment_values', strategy_evidence_id=strategy_id,
            parent_trial_id=parent['trial_id'], options=options)
        if policy != 'v1':
            option_evidence.update(exclusions=exclusions, first_probe_required=bool(first_probe))
        option_id = fingerprint(option_evidence)
        self.store.add_evidence(option_id, 'PREDICTED', option_evidence)
        chosen, reason, decision_id = 'DEFER', '当前预算内没有可执行的调查候选', option_id
        if options:
            schema = create_model('BudgetedExperimentDecision', __base__=Contract,
                candidate_id=(Literal[tuple(values)], ...), reason=(str, Field(min_length=1, max_length=240)))
            observation = dict(options=options, excluded_options=exclusions, first_probe_required=bool(first_probe),
                remaining_gpu_hours=max(0,
                self.config.budget.max_total_gpu_hours - self.store.usage().get('gpu_hours', 0)),
                hardware=dict(gpus=self.topology.num_gpus, memory_per_gpu_gib=self.topology.memory_per_gpu_gb))
            context_id = fingerprint(dict(role='budgeted_experiment_selection', option_id=option_id))
            instructions = ('在所有调查中比较实验价值，自主选择一次测量。预测或通过率未知时，可以根据实际'
                '测量的信息价值选择，并说明依据。使用预算、真实反馈和当前可用的图证据。候选顺序无优先含义。'
                'UNKNOWN是待测状态，不能写成零收益或不可执行。')
            instructions += ('请选择一项通过配置和预算检查的探索测量，说明可验证的执行变化；缺少精确预测不影响测量资格。'
                if first_probe else 'DEFER表示本轮没有值得消耗GPU的候选。')
            existing = self.store.connection.execute("SELECT document FROM evidence WHERE "
                "json_extract(document, '$.kind')='budgeted_agent_selection' AND "
                "json_extract(document, '$.context_id')=?", (context_id,)).fetchone()
            if existing:
                saved = json.loads(existing[0])
                chosen, reason, decision_id = saved['candidate_id'], saved['reason'], saved['decision_id']
            else:
                choice, decision_id = self.invoke('ExperimentInvestigationAgent', instructions, observation, schema,
                    dict(context_id=context_id, tool_round=0, current_tool_return_tokens=0,
                        cumulative_tool_return_tokens=0, cumulative_estimated_context_tokens=0,
                        cumulative_actual_input_tokens=0))
                chosen, reason = choice.candidate_id, choice.reason
                receipt = dict(kind='budgeted_agent_selection', context_id=context_id,
                    candidate_id=chosen, reason=reason, decision_id=decision_id)
                self.store.add_evidence(fingerprint(receipt), 'PREDICTED', receipt)
        selected = next((row for row in options if row['candidate_id'] == chosen), None)
        train = [chosen] if selected and selected['action'] == 'BENCHMARK' else []
        validate = [chosen] if selected and selected['action'] == 'LOAD_VALIDATION' else []
        review = InvestigationSelection(accepted_candidate_ids=train, load_validation_candidate_ids=validate,
            exploration_candidate_ids=train if selected and (selected['uncertainty'] == 'UNKNOWN'
                or selected['memory_admission'] != 'WITHIN_ESTIMATED_CAPACITY') else [],
            evidence_refs=[option_id, decision_id], missing_measurements=[], reason=reason)
        self.store.save_round(index, budgeted_experiment_decision=dict(review=review.model_dump(mode='json'),
            evidence_id=decision_id))
        return review, decision_id

    def select_experiments(self, candidates):
        eligible = [trial for trial in candidates if trial['measurement'] is None and
            trial['prediction']['performance']['step_time_ms'] is not None and
            trial['prediction']['memory']['admission'] != 'REJECT_OOM']
        eligible.sort(key=lambda trial: (trial['prediction']['performance']['step_time_ms'], trial['plan_hash']))
        selected = []
        references = [trial for trial in self.store.trials() if trial['measurement'] is not None]

        def distance(left, right):
            first, second = left['plan']['backend_config'], right['plan']['backend_config']
            differences = sum(first.get(key) != second.get(key) for key in
                ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'micro_batch', 'recompute',
                 'optimizer_sharding', 'rank_mapping', 'overlap_p2p', 'bucket_elements'))
            left_bounds, right_bounds = set(), set()
            offset = 0
            for count in first['decoder_chunks'][:-1]:
                offset += count
                left_bounds.add(offset)
            offset = 0
            for count in second['decoder_chunks'][:-1]:
                offset += count
                right_bounds.add(offset)
            return differences + len(left_bounds ^ right_bounds) / max(1, self.model.num_layers)

        while eligible and len(selected) < self.config.budget.max_benchmarks_per_round:
            if self.config.selection_policy == 'diversity' and references:
                trial = max(eligible, key=lambda item: (min(distance(item, reference) for reference in references),
                    -item['prediction']['performance']['step_time_ms'], item['plan_hash']))
            else:
                trial = eligible[0]
            eligible.remove(trial)
            selected.append(trial)
            references.append(trial)
        ids = {trial['trial_id'] for trial in candidates}
        profiles = sorted((request for request in self.store.profile_requests() if
            request['state'] in ('PENDING', 'APPROVED', 'RUNNING', 'DEFERRED') and
            ids & set(request['trial_ids'])), key=lambda request: (
                -len(ids & set(request['trial_ids'])), -request['priority'],
                request['estimated_gpu_hours'], request['request_id']))
        decision = dict(policy=self.config.selection_policy,
            accepted_candidate_ids=[trial['trial_id'] for trial in selected],
            accepted_profiling_request_ids=[profiles[0]['request_id']] if profiles else [],
            candidates=[dict(trial_id=trial['trial_id'], step_time_ms=trial['prediction']['performance']['step_time_ms'],
                memory_admission=trial['prediction']['memory']['admission']) for trial in candidates],
            budget=self.config.budget.model_dump(), usage=self.store.usage(),
            reason='按预测Step Time排序' if self.config.selection_policy == 'topk' else
                '按全局变量和Region边界距离选择不同方向，使用预测Step Time处理相同距离')
        decision['selection_metrics'] = self.selection_metrics(candidates)
        identifier = fingerprint(decision)
        self.store.add_evidence(identifier, 'PREDICTED', dict(kind='experiment_selection', **decision))
        return InvestigationSelection(accepted_candidate_ids=decision['accepted_candidate_ids'],
            accepted_profiling_request_ids=decision['accepted_profiling_request_ids'],
            load_validation_candidate_ids=[trial['trial_id'] for trial in selected
                if trial['plan']['backend_config'].get('execution_backend') == 'megatron_fsdp'],
            evidence_refs=[identifier], reason=decision['reason'], missing_measurements=[]), identifier

    @staticmethod
    def select_unranked_profile(profile_requests, candidates, reference):
        candidate_ids = {trial['trial_id'] for trial in candidates}
        reference_request = reference['plan']['backend_config']
        reference_backend = reference_request.get('execution_backend', 'megatron_ddp')
        dimensions = ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'micro_batch',
            'recompute', 'pipeline_schedule', 'optimizer_sharding')
        eligible = []
        for row in profile_requests:
            covered = candidate_ids.intersection(row['trial_ids'])
            if not covered or row['state'] not in ('PENDING', 'APPROVED', 'DEFERRED'):
                continue
            request = row['execution_request']
            backend = request.get('execution_backend', 'megatron_ddp')
            if backend == 'megatron_fsdp' and request.get('dp_inner_size', request.get('dp', 1)) <= 1:
                continue
            distance = sum(request.get(name) != reference_request.get(name) for name in dimensions)
            eligible.append((row, (
                backend == reference_backend,
                row['measurement_type'] == 'native_training_cost_profile',
                request.get('optimizer_sharding', 'distributed') ==
                    reference_request.get('optimizer_sharding', 'distributed'),
                len(covered),
                -distance,
                -len(row['cost_keys']),
                row['priority'],
                -row['estimated_gpu_hours'],
                row['request_id'])))
        if not eligible:
            return None
        return max(eligible, key=lambda item: item[1])[0]

    def preserve_cross_family_measurement(self, review, candidates):
        if self.config.agent_execution_enabled and self.config.selection_policy == 'investigation':
            return review
        measured_backends = {trial['plan']['backend_config'].get('execution_backend', 'megatron_ddp')
            for trial in self.store.trials()
            if ((trial['measurement'] is not None
                 and trial['measurement'].get('status') in ('PASS', 'CUDA_OOM', 'FAILED'))
                or ((trial.get('validation') or {}).get('load_validation') or {}).get('status')
                    in ('PASS', 'CUDA_OOM', 'FAILED'))}
        fsdp_eligible = [trial for trial in candidates if trial['measurement'] is None
            and ((trial.get('validation') or {}).get('load_validation') is None
                 or (trial.get('validation') or {}).get('load_validation', {}).get('status') != 'FAILED')
            and trial['plan']['backend_config'].get('execution_backend') == 'megatron_fsdp'
            and trial['prediction'] is not None
            and trial['prediction']['memory']['admission'] != 'REJECT_OOM']
        if 'megatron_ddp' in measured_backends and 'megatron_fsdp' not in measured_backends and fsdp_eligible:
            fsdp_eligible.sort(key=lambda trial: (
                trial['prediction']['performance']['step_time_ms'] is None,
                trial['prediction']['performance']['step_time_ms'], trial['plan_hash']))
            chosen = fsdp_eligible[0]
            accepted = [chosen['trial_id'], *[identifier for identifier in review.accepted_candidate_ids
                if identifier != chosen['trial_id']]]
            exploration = [chosen['trial_id'], *[identifier for identifier in review.exploration_candidate_ids
                if identifier != chosen['trial_id']]]
            evidence = dict(kind='backend_family_measurement_reservation',
                selected_trial_id=chosen['trial_id'], selected_backend='megatron_fsdp',
                measured_backends=sorted(measured_backends))
            evidence_id = fingerprint(evidence)
            self.store.add_evidence(evidence_id, 'PREDICTED', evidence)
            return review.model_copy(update=dict(accepted_candidate_ids=accepted,
                exploration_candidate_ids=exploration,
                evidence_refs=[*review.evidence_refs, evidence_id],
                reason=review.reason + '；DDP已出现真实失败，优先测量FSDP后端家族。'))
        measured_families = {self.global_family_id(trial['plan']['backend_config'])
            for trial in self.store.trials() if trial['measurement'] is not None
            and trial['measurement'].get('status') in ('PASS', 'CUDA_OOM', 'FAILED')}
        selected = [trial for trial in candidates if trial['trial_id'] in review.accepted_candidate_ids]
        if any(self.global_family_id(trial['plan']['backend_config']) not in measured_families
                for trial in selected):
            return review
        eligible = [trial for trial in candidates if trial['measurement'] is None
            and self.global_family_id(trial['plan']['backend_config']) not in measured_families
            and trial['prediction'] is not None
            and trial['prediction']['performance']['step_time_ms'] is not None
            and trial['prediction']['memory']['admission'] == 'WITHIN_ESTIMATED_CAPACITY']
        if not eligible:
            return review
        eligible.sort(key=lambda trial: (
            trial['prediction']['performance']['step_time_ms'], trial['plan_hash']))
        chosen = eligible[0]
        accepted = [chosen['trial_id'], *[identifier for identifier in review.accepted_candidate_ids
            if identifier != chosen['trial_id']]]
        exploration = list(review.exploration_candidate_ids)
        if (chosen['prediction']['memory']['admission'] != 'REJECT_OOM'
                and self.store.usage().get('exploration_trials', 0)
                < self.config.uncalibrated_exploration_trials):
            exploration = [chosen['trial_id'], *[identifier for identifier in exploration
                if identifier != chosen['trial_id']]]
        evidence = dict(kind='cross_family_measurement_reservation',
            selected_trial_id=chosen['trial_id'],
            selected_family_id=self.global_family_id(chosen['plan']['backend_config']),
            measured_family_ids=sorted(measured_families),
            prediction_status=chosen['prediction']['prediction_status'],
            memory_admission=chosen['prediction']['memory']['admission'])
        evidence_id = fingerprint(evidence)
        self.store.add_evidence(evidence_id, 'PREDICTED', evidence)
        return review.model_copy(update=dict(accepted_candidate_ids=accepted,
            exploration_candidate_ids=exploration,
            evidence_refs=[*review.evidence_refs, evidence_id],
            reason=review.reason + '；协调器保留一个尚未实测全局家族的真实测量机会。'))

    @staticmethod
    def resolve_selection_candidate_ids(review, candidates):
        known = {trial['trial_id'] for trial in candidates}

        def resolve(identifier):
            if identifier in known:
                return identifier
            matches = [candidate_id for candidate_id in known if candidate_id.startswith(identifier)]
            return matches[0] if len(matches) == 1 else identifier

        return review.model_copy(update=dict(
            accepted_candidate_ids=[resolve(identifier) for identifier in review.accepted_candidate_ids],
            load_validation_candidate_ids=[resolve(identifier)
                for identifier in review.load_validation_candidate_ids],
            exploration_candidate_ids=[resolve(identifier)
                for identifier in review.exploration_candidate_ids]))

    def selected_candidate_pool(self, review, candidates, parent):
        selected_ids = set(review.accepted_candidate_ids) | set(review.load_validation_candidate_ids)
        current = {row['trial_id']: row for row in candidates}
        registered = {row['trial_id']: row for row in self.store.trials()}
        for identifier in sorted(selected_ids):
            if identifier not in registered:
                raise ValueError('实验选择引用了未登记的候选')
            trial = registered[identifier]
            self.require_fixed_training_semantics(parent['plan']['backend_config'], trial['plan']['backend_config'])
            self.rewriter.candidate(trial['plan']['backend_config'])
            if trial.get('failure') or (trial.get('prediction') or {}).get('memory', {}).get('admission') == 'REJECT_OOM':
                raise ValueError('实验选择引用了已确认失败的候选')
            if (trial.get('prediction') is None or
                    trial['prediction'].get('cost_database_revision') != self.cost_revision):
                trial = self.predict_trial(trial, force=True)
            current[identifier] = trial
            evidence = dict(kind='agent_selected_registered_candidate', trial_id=identifier,
                parent_trial_id=parent['trial_id'], plan_hash=trial['plan_hash'],
                load_validation=self.load_validation_summary(trial),
                selection=review.model_dump(mode='json'))
            self.store.add_evidence(fingerprint(evidence), 'PREDICTED', evidence)
        return list(current.values())

    @staticmethod
    def preserve_unranked_exploration(review, candidates, enabled, *, agent_owned=False):
        if not enabled:
            return review
        eligible = [trial for trial in candidates if trial['measurement'] is None
            and trial['prediction']['memory']['admission'] != 'REJECT_OOM'
            and trial['prediction']['performance'].get('step_time_ms') is None]
        if not eligible:
            return review
        by_id = {trial['trial_id']: trial for trial in eligible}
        ranked_families = {global_identity(trial['plan']['backend_config']) for trial in candidates
            if trial['trial_id'] in review.accepted_candidate_ids and
            trial['prediction']['performance'].get('step_time_ms') is not None}
        coverage = [trial for trial in eligible
            if global_identity(trial['plan']['backend_config']) not in ranked_families]
        chosen_id = next((identifier for identifier in review.exploration_candidate_ids
                          if identifier in by_id), None)
        if chosen_id is None and agent_owned:
            chosen_id = next((identifier for identifier in review.accepted_candidate_ids
                if identifier in by_id), None)
            if chosen_id is None:
                return review
        if chosen_id is None:
            chosen_id = min(coverage or eligible, key=lambda trial: (
                global_identity(trial['plan']['backend_config']), trial['plan_hash']))['trial_id']
        accepted = [chosen_id, *[identifier for identifier in review.accepted_candidate_ids
            if identifier != chosen_id]]
        exploration = [chosen_id, *[identifier for identifier in review.exploration_candidate_ids
            if identifier != chosen_id and identifier in accepted]]
        load_validation = list(review.load_validation_candidate_ids)
        if (by_id[chosen_id]['plan']['backend_config'].get('execution_backend') == 'megatron_fsdp'
                and chosen_id not in load_validation):
            load_validation.insert(0, chosen_id)
        return review.model_copy(update=dict(
            accepted_candidate_ids=accepted,
            exploration_candidate_ids=exploration,
            load_validation_candidate_ids=load_validation,
            reason=review.reason + '；为尚未获得完整成本的全局并行组合预留本轮真实探索测量。'))

    def execute_planning_measurements(self, parent, candidate_ids, action_ids):
        candidates = [row for row in self.store.trials() if row['trial_id'] in candidate_ids
            and row.get('measurement') is None and not row.get('failure')
            and (row.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM']
        if not candidates:
            return []
        context = self.shared_context(parent) | dict(parent_trial_id=parent['trial_id'],
            candidates=self.investigation_candidate_table(candidates),
            candidate_decision_report=candidate_decisions(parent, candidates),
            planning_action_evidence_refs=action_ids,
            profiling_requests=[dict(request_id=row['request_id'], measurement_type=row['measurement_type'],
                estimated_gpu_hours=row['estimated_gpu_hours'], state=row['state'])
                for row in self.store.profile_requests() if row['state'] in ('PENDING', 'DEFERRED')])
        selection, evidence_id = self.invoke_with_tools('ExperimentInvestigationAgent',
            '联合调查已经生成原生合法的完整候选。选择本轮实际训练及必要补测。'
            '完整成本未知时可选择探索候选；已通过加载验证的候选优先进行完整测量。',
            context, InvestigationSelection)
        selection = self.resolve_selection_candidate_ids(selection, candidates)
        self.store.add_search_state('PLANNING_GPU_SELECTION', dict(
            selection=selection.model_dump(mode='json'), evidence_id=evidence_id, action_ids=action_ids))
        for identifier in selection.accepted_profiling_request_ids:
            self.execute_profile(identifier)
        measured = []
        for identifier in selection.accepted_candidate_ids:
            if sum(measured_result['measurement'].get('performance_eligible', False)
                    for measured_result in measured) >= self.config.budget.max_benchmarks_per_round:
                break
            candidate = next(row for row in self.store.trials() if row['trial_id'] == identifier)
            if (candidate.get('prediction') or {}).get('memory', {}).get('admission') == 'REJECT_OOM':
                continue
            if candidate['plan']['backend_config'].get('execution_backend') == 'megatron_fsdp':
                validation = self.execute_load_validation(candidate)
                if validation['status'] != 'PASS':
                    continue
            result = self.execute_trial(candidate, exploration=identifier in selection.exploration_candidate_ids
                or (candidate.get('prediction') or {}).get('performance', {}).get('step_time_ms') is None)
            if result is not None:
                measured.append(result)
        self._planning_last_measured_feedback = candidate_decisions(parent, measured)
        self.store.add_search_state('PLANNING_TRAINING_FEEDBACK', self._planning_last_measured_feedback)
        return measured

    def run(self):
        try:
            return self.run_loop()
        except SearchBudgetExhausted:
            return self.finish()

    def run_loop(self):
        if self.config.planning_control_enabled and self.config.evaluation_mode in ('full', 'multi_agent'):
            return self.run_investigation_loop()
        full_agents = (self.config.agent_execution_enabled
            and self.config.evaluation_mode not in ('seed', 'random_region'))
        fast_agent_rounds = full_agents and self.config.fast_agent_rounds
        if (full_agents and self.store.usage().get('agent_rounds', 0) < self.config.budget.max_agent_rounds
                and self.config.max_llm_calls_per_round < (1 if fast_agent_rounds else 4)):
            raise ValueError('完整四Agent流程要求max_llm_calls_per_round=4；工具追问受max_agent_calls限制')
        if full_agents and self.config.max_tool_rounds < 1:
            raise ValueError('Constraint and Validation Agent需要至少一轮确定性工具调用')
        for trial in self.store.trials():
            if trial['state'] == 'RUNNING' and trial['measurement'] is None:
                self.execute_trial(trial, exploration=self.store.charged(
                    trial['plan_hash'] + ':exploration') is not None)
        best = self.initialize()
        if best is None:
            return self.finish()
        best = self.store.best_measured() or best
        for request in self.store.profile_requests():
            if request['state'] == 'RUNNING':
                self.execute_profile(request['request_id'])
        completed_rounds = int(self.store.usage().get('agent_rounds', 0))
        previous = self.store.get_round(completed_rounds - 1)
        no_improvement = previous.get('no_improvement', 0) if previous else 0
        interface_faults = sum(bool(self.evidence_interface_fault(self.store.get_round(index)))
            for index in range(completed_rounds))
        for round_index in count(completed_rounds):
            if round_index - interface_faults >= self.config.budget.max_agent_rounds:
                break
            self.check_walltime()
            if (self.config.gpu_execution_enabled
                    and self.config.budget.max_walltime_seconds - self.store.search_seconds()
                        < self.config.budget.estimated_trial_seconds):
                return self.finish()
            checkpoint = self.store.get_round(round_index)
            resume_measurements = bool(checkpoint and checkpoint['stage'] == 'MEASUREMENTS')
            if checkpoint and checkpoint['stage'] == 'COMPLETE':
                self.store.charge(f'round:{round_index}', 'agent_rounds', 1)
                interface_faults += bool(self.evidence_interface_fault(checkpoint))
                best = self.store.best_measured() or best
                continue
            usage = self.store.usage()
            reset_context = (checkpoint is None or checkpoint.get('context_version') != LAYERED_CONTEXT_VERSION
                or checkpoint.get('agent_model') != self.config.agent_model
                or checkpoint.get('agent_reasoning_effort') != self.config.agent_reasoning_effort
                or checkpoint.get('agent_request_profile', 'decision') != self.config.agent_request_profile
                or checkpoint.get('fast_agent_rounds', False) != self.config.fast_agent_rounds
                or checkpoint.get('single_decision_round', False) != self.config.single_decision_round)
            usage_at_start = usage if reset_context else checkpoint.get('usage_at_start', usage)
            context_metric_start = (len(self.store.agent_context_metrics()) if reset_context
                else checkpoint.get('context_metric_start', len(self.store.agent_context_metrics())))
            if checkpoint:
                best = next(trial for trial in self.store.trials() if trial['trial_id'] == checkpoint['parent_trial_id'])
            if reset_context:
                best = self.calibrate_measured_parent(best)
            request = best['plan']['backend_config']
            if reset_context:
                global_family_candidates, global_family_search = self.prepare_global_family_candidates(best)
            else:
                global_ids = set(checkpoint.get('global_family_candidate_ids', []))
                global_family_candidates = [trial for trial in self.store.trials()
                    if trial['trial_id'] in global_ids]
                global_family_search = checkpoint.get(
                    'global_family_search', self.global_family_catalog(best, global_family_candidates))
                if 'family_status_counts' not in global_family_search:
                    global_family_search = self.global_family_catalog(best, global_family_candidates)
                    checkpoint = self.store.save_round(round_index, global_family_search=global_family_search)
            observation = self.performance_agent_context(best)
            if reset_context:
                checkpoint = self.store.replace_round(round_index, dict(round_index=round_index,
                    parent_trial_id=best['trial_id'], context_version=LAYERED_CONTEXT_VERSION,
                    agent_model=self.config.agent_model, agent_reasoning_effort=self.config.agent_reasoning_effort,
                    agent_request_profile=self.config.agent_request_profile,
                    fast_agent_rounds=self.config.fast_agent_rounds,
                    single_decision_round=self.config.single_decision_round,
                    usage_at_start=usage_at_start, context_metric_start=context_metric_start,
                    observation=observation,
                    global_family_candidate_ids=[trial['trial_id'] for trial in global_family_candidates],
                    global_family_search=global_family_search, stage='ANALYSIS'))
            observation = checkpoint['observation']
            analysis_key = fingerprint(dict(context_version=LAYERED_CONTEXT_VERSION,
                baseline=(best['measurement'] or {}).get('run_id'), profile_id=self.profile_id,
                prediction_id=(best.get('prediction') or {}).get('prediction_id'),
                cost_revision=self.cost_revision,
                recent_history=observation.get('recent_history', []),
                recent_profiling_results=observation.get('recent_profiling_results', []),
                graph_sha256=self.backend_dag(best)['graph_sha256'],
                ablation_contract=AblationEvidencePolicy(self.config.agent_ablation).contract,
                agent_model=self.config.agent_model, reasoning_effort=self.config.agent_reasoning_effort))
            analysis_path = self.output / ('analysis-' + analysis_key + '.json')
            if 'diagnosis' in checkpoint:
                diagnosis = BottleneckDecision.model_validate(checkpoint['diagnosis'])
                diagnosis_id = checkpoint['diagnosis_evidence_id']
            elif analysis_path.exists():
                saved = json.loads(analysis_path.read_text())
                diagnosis = BottleneckDecision.model_validate(saved['diagnosis'])
                diagnosis_id = saved['evidence_id']
            else:
                if (self.config.evaluation_mode in ('seed', 'random_region')
                        or not self.config.agent_execution_enabled):
                    reference_id = best['measurement']['run_id'] if best['measurement'] else best['prediction']['prediction_id']
                    diagnosis = BottleneckDecision(hypotheses=['从当前基准证据与合法变量范围展开确定性对照候选'],
                        evidence_refs=[reference_id], target_regions=[], requested_metrics=[])
                    diagnosis_id = fingerprint(diagnosis.model_dump(mode='json'))
                    self.store.add_evidence(diagnosis_id, 'HYPOTHESIS', diagnosis.model_dump(mode='json'))
                elif self.config.agent_ablation == 'without_bottleneck_agent' or fast_agent_rounds:
                    diagnosis, diagnosis_id = self.deterministic_bottleneck_decision(best)
                else:
                    diagnosis, diagnosis_id = self.invoke_with_tools('PerformanceAnalysisAgent',
                        self.performance_instructions(), observation, BottleneckDecision)
                analysis_path.write_text(json.dumps(dict(diagnosis=diagnosis.model_dump(),
                    evidence_id=diagnosis_id), ensure_ascii=False, allow_nan=False) + '\n')
            diagnosis_feedback = self.evidence_reference_feedback('PerformanceAnalysisAgent', diagnosis)
            if (diagnosis_feedback is not None and self.config.agent_execution_enabled
                    and not fast_agent_rounds):
                diagnosis, diagnosis_id = self.invoke_with_tools('PerformanceAnalysisAgent',
                    self.performance_instructions(), dict(observation,
                        previous_response=diagnosis.model_dump(mode='json'),
                        tool_history=[diagnosis_feedback]), BottleneckDecision)
                analysis_path.write_text(json.dumps(dict(diagnosis=diagnosis.model_dump(),
                    evidence_id=diagnosis_id), ensure_ascii=False, allow_nan=False) + '\n')
            checkpoint = self.store.save_round(round_index, diagnosis=diagnosis.model_dump(mode='json'),
                diagnosis_evidence_id=diagnosis_id, stage='STRATEGY')
            if 'strategy' in checkpoint:
                strategy = IntentBatch.model_validate(checkpoint['strategy'])
                strategy_id = checkpoint['strategy_evidence_id']
            else:
                if (self.config.evaluation_mode in ('seed', 'random_region')
                        or not self.config.agent_execution_enabled):
                    planner = SeedPlanner(self.rewriter, self.profile)
                    reference_id = best['measurement']['run_id'] if best['measurement'] else best['prediction']['prediction_id']
                    rows = planner.generate(request, reference_id, self.config.budget.max_candidates_per_round)
                    intents = {fingerprint(row['intent']): row['intent'] for row in rows}
                    values = list(intents.values())
                    if self.config.evaluation_mode == 'random_region':
                        random.Random(round_index).shuffle(values)
                    else:
                        def deterministic_priority(intent):
                            active = {key: value for key, value in intent['values'].items()
                                if value is not None}
                            return ('decoder_chunks' not in active,
                                'recompute' in active, -len(active), fingerprint(intent))

                        values.sort(key=deterministic_priority)
                    for failure in planner.rejections:
                        self.store.add_evidence(fingerprint(failure), 'FAILURE', failure)
                    strategy = IntentBatch(intents=[AgentRewriteIntent.model_validate(intent) for intent in
                        values[:self.config.intent_count]], reason='同一协调器中的候选生成对照',
                        evidence_refs=[reference_id])
                    strategy_id = fingerprint(strategy.model_dump(mode='json'))
                    self.store.add_evidence(strategy_id, 'HYPOTHESIS', strategy.model_dump(mode='json'))
                else:
                    strategy_observation = self.strategy_agent_context(best, diagnosis, diagnosis_id,
                        checkpoint['global_family_search'])
                    strategy_observation['model_module_sequence'] = dict(module_id_origin=0,
                        module_id_rule='module_types数组下标是原始module_id，保留全部模块次序',
                        module_types=[kind.value for kind in self.model.layer_types])
                    strategy, strategy_id = self.invoke_with_tools('JointStrategyAgent',
                        self.strategy_instructions(), strategy_observation, IntentBatch)
                checkpoint = self.store.save_round(round_index,
                    strategy=strategy.model_dump(mode='json'),
                    strategy_evidence_id=strategy_id, stage='CONSTRAINT_VALIDATION')
            if full_agents:
                strategy, strategy_id, strategy_validation = self.validate_or_repair_strategy(
                    strategy, strategy_id, best, diagnosis, diagnosis_id, checkpoint['global_family_search'])
                checkpoint = self.store.save_round(round_index,
                    strategy=strategy.model_dump(mode='json'), strategy_evidence_id=strategy_id,
                    strategy_validation=strategy_validation)
                if self.config.single_decision_round and strategy_validation.get('status') == 'REJECTED':
                    missing = self.evidence_interface_fault(checkpoint)
                    interface_faults += bool(missing)
                    rejection_id = fingerprint(dict(strategy_id=strategy_id, validation=strategy_validation))
                    self.store.add_evidence(rejection_id, 'FAILURE', strategy_validation)
                    self.complete_round(round_index, best, no_improvement=no_improvement,
                        feedback=dict(native_validation=strategy_validation['native_validation'],
                            evidence_refs=[rejection_id]), feedback_evidence_id=rejection_id,
                        stop_reason='原生检查拒绝本轮候选，反馈进入下一轮独立决策',
                        **(dict(interface_fault_evidence_ids=missing) if missing else {}))
                    self.store.charge(f'round:{round_index}', 'agent_rounds', 1)
                    continue
            if full_agents:
                if ('constraint_decision' in checkpoint and not checkpoint.get('candidates_complete')
                        and checkpoint.get('constraint_implementation') != self.constraint_implementation):
                    checkpoint = {key: value for key, value in checkpoint.items() if key not in (
                        'constraint_decision', 'constraint_evidence_id', 'constraint_candidate_ids',
                        'candidate_ids')}
                    checkpoint['stage'] = 'CONSTRAINT_VALIDATION'
                    self.store.replace_round(round_index, checkpoint)
                if 'constraint_decision' in checkpoint:
                    constraint_decision = ConstraintValidationDecision.model_validate(checkpoint['constraint_decision'])
                    constraint_id = checkpoint['constraint_evidence_id']
                elif fast_agent_rounds:
                    completion = self.constraint_tool(ConstraintValidationToolRequest(
                        tool='complete_training_plans', parent_trial_id=best['trial_id'],
                        strategy_evidence_id=strategy_id,
                        intent_indices=list(range(len(strategy.intents))), limit=8))
                    constraint_id = completion['evidence_id']
                    constraint_decision = ConstraintValidationDecision(
                        intent_results=[ConstraintIntentDecision(intent_index=row['intent_index'],
                            status=row['status'], candidate_ids=row['candidate_ids'],
                            evidence_refs=[constraint_id], reason=(row.get('reason', row['status'])
                                if isinstance(row.get('reason', row['status']), str)
                                else json.dumps(row['reason'], ensure_ascii=False)))
                            for row in completion['result']['intent_results']],
                        evidence_refs=[constraint_id],
                        reason='使用确定性联合求解器生成的完整候选与约束结果。')
                else:
                    constraint_context = self.constraint_agent_context(best, strategy, strategy_id)
                    if ((self.config.agent_request_profile in ('decision', 'compact')
                            or self.config.constraint_review_policy == 'native_complete')
                            and all(intent.proposal_kind == 'complete_strategy' for intent in strategy.intents)):
                        constraint_context['backend_completion_receipt'] = self.constraint_tool(
                            ConstraintValidationToolRequest(tool='complete_training_plans',
                                parent_trial_id=best['trial_id'], strategy_evidence_id=strategy_id,
                                intent_indices=list(range(len(strategy.intents))), limit=8))
                        constraint_context['review_scope'] = (
                            '程序已经完成常规原生约束与候选构造。核对完整策略语义，'
                            '逐项引用backend_completion_receipt，必要时查询额外证据。')
                    receipt = constraint_context.get('backend_completion_receipt')
                    native_results = receipt['result']['intent_results'] if receipt else []
                    if (self.config.constraint_review_policy == 'native_complete'
                            and len(native_results) == len(strategy.intents)
                            and all(row['status'] == 'COMPLETED' and row['candidate_ids']
                                for row in native_results)):
                        constraint_id = receipt['evidence_id']
                        constraint_decision = ConstraintValidationDecision(
                            intent_results=[ConstraintIntentDecision(intent_index=row['intent_index'],
                                status=row['status'], candidate_ids=row['candidate_ids'],
                                evidence_refs=[constraint_id], reason=(row.get('reason', row['status'])
                                    if isinstance(row.get('reason', row['status']), str)
                                    else json.dumps(row['reason'], ensure_ascii=False)))
                                for row in native_results],
                            evidence_refs=[constraint_id],
                            reason='使用后端对全部完整策略给出的原生约束核验结果。')
                    else:
                        constraint_decision, constraint_id = self.invoke_with_tools(
                            'ConstraintValidationAgent', self.constraint_instructions(),
                            constraint_context, ConstraintValidationDecision)
                completed_ids = self.validate_constraint_decision(
                    constraint_decision, strategy, best, strategy_id)
                checkpoint = self.store.save_round(round_index,
                    constraint_decision=constraint_decision.model_dump(mode='json'),
                    constraint_evidence_id=constraint_id, constraint_candidate_ids=completed_ids,
                    constraint_implementation=(self.config.constraint_review_policy
                        if self.config.constraint_review_policy != 'always' else self.constraint_implementation),
                    stage='CANDIDATES')
            legal_intents = strategy.intents
            if full_agents:
                completed_indices = {row.intent_index for row in constraint_decision.intent_results
                    if row.status == 'COMPLETED'}
                legal_intents = [intent for index, intent in enumerate(strategy.intents)
                    if index in completed_indices]
            elif checkpoint.get('strategy_validation') is not None:
                legal_intents = [intent for index, intent in enumerate(strategy.intents)
                    if checkpoint['strategy_validation']['legal_counts'].get(str(index),
                        checkpoint['strategy_validation']['legal_counts'].get(index))['count'] > 0]
            interactions = (self.rewriter.compiler.transformation_interactions(
                self.rewriter.candidate(request), [self.resolve_agent_intent(intent, request)
                    for intent in legal_intents],
                diagnosis.affected_tasks) if self.config.agent_execution_enabled else dict(
                    source='local_deterministic_candidate_generation', measured_interactions=[],
                    previews=[], edges=[]))
            interaction_id = fingerprint(interactions)
            self.store.add_evidence(interaction_id, 'PREDICTED', interactions)
            known_candidate_ids = set(checkpoint.get('candidate_ids', [])) | set(
                checkpoint.get('global_family_candidate_ids', []))
            if full_agents:
                known_candidate_ids.update(checkpoint['constraint_candidate_ids'])
            candidates = [trial for trial in self.store.trials()
                          if trial['trial_id'] in known_candidate_ids]
            if full_agents:
                candidates = [self.reuse_known_measurement(row) for row in candidates]
                candidates.sort(key=lambda row:
                    (row.get('rewrite') or {}).get('candidate_source') != 'complete_strategy')
                for index, candidate in enumerate(candidates):
                    existing_prediction = candidate.get('prediction') or {}
                    if (candidate['measurement'] is not None or (
                            existing_prediction.get('cost_database_revision') == self.cost_revision
                            and existing_prediction.get('evidence', {}).get('predictor_revision') == PROJECTION_REVISION)):
                        continue
                    if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                        break
                    candidates[index] = self.predict_trial(candidate, parent_request=request)
                    checkpoint = self.store.save_round(round_index,
                        candidate_ids=[trial['trial_id'] for trial in candidates])
                candidates = [trial for trial in candidates if trial['prediction'] is not None
                    and trial['measurement'] is None]
            proposals = [] if full_agents or checkpoint.get('candidates_complete') else legal_intents
            for proposal in proposals:
                if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                    break
                if len({trial['trial_id'] for trial in candidates}) >= self.config.budget.max_candidates_per_round:
                    break
                try:
                    if self.config.agent_execution_enabled:
                        self.validate_intent_evidence(proposal)
                    intent = self.resolve_agent_intent(proposal, request)
                    if intent.module_relations:
                        family = dict(family_id=proposal.family_id,
                            values=self.global_family_values(request))
                        rows = self.solve_conditional_family(request, family, intent,
                            self.config.candidate_limit)['candidates']
                    else:
                        rows = self.rewriter.complete_and_rewrite(request, intent, limit=self.config.candidate_limit)
                except ValueError as exc:
                    if isinstance(exc.args[0], dict) and exc.args[0].get('status') == 'CONSTRAINT_CONFLICT':
                        failure = dict(intent=proposal.model_dump(mode='json'),
                            parent_plan_hash=best['plan_hash'], constraints=exc.args[0])
                        self.store.add_evidence(fingerprint(failure), 'FAILURE', failure)
                        (self.output / f'round-{round_index}-skipped-intents.jsonl').open('a').write(
                            json.dumps(dict(operation=proposal.operation, conflicts=exc.args[0].get('conflicts')),
                                       ensure_ascii=False, allow_nan=False) + '\n')
                        continue
                    raise
                for row in rows:
                    if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                        break
                    if len({trial['trial_id'] for trial in candidates}) >= self.config.budget.max_candidates_per_round:
                        break
                    trial = self.predict_candidate(row, best, intent)
                    if trial['measurement'] is None:
                        candidates.append(trial)
                    checkpoint = self.store.save_round(round_index,
                        candidate_ids=sorted({trial['trial_id'] for trial in candidates}))
            candidates = list({trial['trial_id']: trial for trial in candidates}.values())
            carryovers = [trial for trial in self.store.trials()
                if trial['measurement'] is None and trial['prediction'] is not None
                and trial['prediction'].get('cost_database_revision') == self.cost_revision
                and (trial['prediction']['performance']['step_time_ms'] is not None
                    or (self.load_validation_summary(trial) or {}).get('status') == 'PASS')]
            candidates = list({trial['trial_id']: trial for trial in [*candidates, *carryovers]}.values())
            candidates = self.retain_candidate_portfolio(candidates,
                self.config.budget.max_candidates_per_round, parent=best if full_agents else None,
                proposal_candidate_ids=([identifier for entry in checkpoint['constraint_decision']['intent_results']
                    for identifier in entry['candidate_ids']] if self.config.single_decision_round else ()))
            checkpoint = self.store.save_round(round_index, candidates_complete=True,
                candidate_ids=[trial['trial_id'] for trial in candidates], stage='SELECTION')
            if not candidates:
                self.complete_round(round_index, best, no_improvement=no_improvement,
                    stop_reason='当前轮次没有新增可执行候选')
                self.store.charge(f'round:{round_index}', 'agent_rounds', 1)
                if self.config.single_decision_round:
                    continue
                break
            needs_model_selection = (not self.config.single_decision_round and (not fast_agent_rounds or any(
                trial['measurement'] is None
                and trial['prediction']['memory']['admission'] != 'REJECT_OOM'
                and (trial['prediction']['performance']['step_time_ms'] is None
                    or trial['prediction']['memory']['admission'] != 'WITHIN_ESTIMATED_CAPACITY')
                for trial in candidates)))
            selection_role = ('fast_deterministic_selection_v1' if fast_agent_rounds and not needs_model_selection else
                'investigation_v11' if self.config.agent_execution_enabled else 'local_deterministic_v1')
            expected_candidate_ids = {trial['trial_id'] for trial in candidates}
            checkpoint_candidates = checkpoint.get('selection_context', {}).get('candidates', {})
            checkpoint_candidate_ids = ({row[0] for row in checkpoint_candidates.get('overview', [])}
                if isinstance(checkpoint_candidates, dict) else
                {row['trial_id'] for row in checkpoint_candidates})
            selection_context_changed = (
                checkpoint.get('selection_context', {}).get('role_context') != selection_role
                or checkpoint_candidate_ids != expected_candidate_ids
                or checkpoint.get('selection_context', {}).get('selection_limits') != self.selection_limits())
            if selection_context_changed:
                selection_context = (self.investigation_agent_context(
                    best, candidates, strategy, interactions, interaction_id)
                    if self.config.agent_execution_enabled and needs_model_selection else dict(
                        role_context=selection_role,
                        candidates=self.investigation_candidate_table(candidates),
                        selection_metrics=self.selection_metrics(candidates),
                        ranking_sensitivity=self.ranking_sensitivity(candidates)))
                checkpoint = self.store.save_round(round_index,
                    selection_context=self.visible_observation(selection_context))
            if 'selection' in checkpoint and (not selection_context_changed or
                    (resume_measurements and checkpoint.get('measurement_execution_started', False))):
                review = InvestigationSelection.model_validate(checkpoint['selection'])
                review_id = checkpoint['selection_evidence_id']
            elif self.config.budgeted_investigation_enabled and self.config.single_decision_round:
                review, review_id = self.choose_budgeted_experiment(best, candidates,
                    checkpoint['constraint_decision']['intent_results'], strategy, strategy_id, round_index)
            elif (self.config.selection_policy == 'investigation'
                    and self.config.evaluation_mode not in ('seed', 'random_region')
                    and self.config.agent_execution_enabled and needs_model_selection):
                review, review_id = self.invoke_with_tools('ExperimentInvestigationAgent',
                    '逐项比较完整候选的潜在吞吐收益、执行代价、未知成本与实际测量价值，填写candidate_assessments。当前父策略的瓶颈具有执行条件；候选依据完整迭代效果竞争预算。可使用tool_requests合并必要查询。accepted_candidate_ids按测量优先级返回，对应BENCHMARK结论。load_validation_candidate_ids用于独立加载验证；成本或容量未知的完整训练同时加入exploration_candidate_ids。使用真实证据，性能预期标记为HYPOTHESIS。',
                    checkpoint['selection_context'], InvestigationSelection)
            else:
                if self.config.single_decision_round:
                    review, review_id = self.select_joint_investigation(candidates,
                        checkpoint['constraint_decision']['intent_results'], strategy_id,
                        checkpoint.get('measurement_preference', 'training'))
                else:
                    review, review_id = self.select_experiments(candidates)
            review = self.resolve_selection_candidate_ids(review, self.store.trials())
            candidates = self.selected_candidate_pool(review, candidates, best)
            agent_selection = review.model_dump(mode='json')
            self.store.save_round(round_index, agent_selection=agent_selection)
            if self.config.gpu_execution_enabled and not self.config.single_decision_round:
                review = self.preserve_cross_family_measurement(review, candidates)
            ranking = checkpoint['selection_context']['ranking_sensitivity']
            sensitive_candidates = ranking_sensitive_candidates(ranking)
            profile_requests = self.store.profile_requests()
            root = Path(__file__).resolve().parents[1]
            runtime_hashes = {name: hashlib.sha256((root / source).read_bytes()).hexdigest()
                for name, source in (('paramorph_runtime.py', 'runtime.py'),
                    ('paramorph_communication_calibration.py', 'communication_calibration.py'))}
            def native_profile_key(row):
                execution = row['execution_request']
                return (plan_signature(execution), execution.get('paramorph_operator_calibration', True),
                    execution.get('paramorph_memory_audit', False),
                    execution.get('paramorph_serial_calibration_steps'))
            completed_native = {native_profile_key(row) for row in profile_requests
                if row['state'] == 'COMPLETED'
                and row['measurement_type'] == 'native_training_cost_profile'
                and all((row.get('result') or {}).get('measurement', {}).get(
                    'source_hashes', {}).get(name) == digest for name, digest in runtime_hashes.items())}
            pending_profiles = {row['request_id']: row for row in profile_requests
                if row['state'] in ('PENDING', 'APPROVED', 'DEFERRED')
                and (row['measurement_type'] != 'native_training_cost_profile'
                    or native_profile_key(row) not in completed_native)}
            if self.config.single_decision_round:
                pending_profiles = self.current_proposal_profiles(pending_profiles,
                    checkpoint['constraint_decision']['intent_results'],
                    {row['trial_id'] for row in candidates if self.available_for_joint_measurement(row)})
            all_unranked = all(trial['prediction']['performance'].get('step_time_ms') is None
                               for trial in candidates)
            if resume_measurements and not selection_context_changed:
                accepted_profiles = review.accepted_profiling_request_ids
            elif self.config.gpu_execution_enabled and all_unranked and pending_profiles:
                accepted_profiles = [identifier for identifier in review.accepted_profiling_request_ids
                    if identifier in pending_profiles]
                if (not accepted_profiles and not review.accepted_candidate_ids
                        and not review.load_validation_candidate_ids and not self.config.budgeted_investigation_enabled):
                    selected_request = self.select_unranked_profile(
                        pending_profiles.values(), candidates, best)
                    accepted_profiles = ([selected_request['request_id']]
                        if selected_request is not None else [])
            else:
                decisions = profile_decisions(pending_profiles.values(), candidates, best,
                    sensitive_candidates, self.attempted_execution_plans())
                allowed = {row['request_id'] for row in decisions}
                accepted_profiles = [identifier for identifier in review.accepted_profiling_request_ids
                    if identifier in allowed]
                self.store.save_round(round_index, profile_decision_gate=dict(
                    available=decisions, requested=review.accepted_profiling_request_ids,
                    accepted=accepted_profiles))
            if accepted_profiles != review.accepted_profiling_request_ids:
                review = review.model_copy(update=dict(
                    accepted_profiling_request_ids=accepted_profiles,
                    reason=review.reason + ('；全部候选缺少可排序成本时，优先测量当前实测后端内'
                        '与参考计划最接近的候选。' if all_unranked else
                        '；补测需影响现有排序，或匹配未试并行形状的明确成本缺项。')))
            known = {row['trial_id'] for row in candidates}
            if not set(review.accepted_candidate_ids) <= known:
                raise ValueError('ExperimentInvestigationAgent选择了不存在的候选')
            if len(set(review.accepted_candidate_ids)) != len(review.accepted_candidate_ids):
                raise ValueError('ExperimentInvestigationAgent重复选择了同一候选')
            if len(set(review.accepted_profiling_request_ids)) != len(review.accepted_profiling_request_ids):
                raise ValueError('ExperimentInvestigationAgent重复选择了同一Profiling请求')
            if not set(review.load_validation_candidate_ids) <= known:
                raise ValueError('Checkpoint Load验证引用了不存在的候选')
            if len(set(review.load_validation_candidate_ids)) != len(review.load_validation_candidate_ids):
                raise ValueError('Checkpoint Load验证重复选择了同一候选')
            if not set(review.exploration_candidate_ids) <= set(review.accepted_candidate_ids):
                raise ValueError('探索预算只能用于已经选择的候选')
            if (review.accepted_candidate_ids
                    and (best.get('measurement') or {}).get('status') != 'PASS'):
                profile_types = {row['request_id']: row['measurement_type'] for row in profile_requests}
                capacity_profiles = {identifier for identifier in review.accepted_profiling_request_ids
                    if profile_types[identifier] == 'native_training_cost_profile'}
                if capacity_profiles:
                    review = review.model_copy(update=dict(
                        accepted_profiling_request_ids=[identifier for identifier in
                            review.accepted_profiling_request_ids if identifier not in capacity_profiles],
                        reason=review.reason + '；容量修复候选先执行真实训练验证，可运行后再采集模块成本。'))
            self.store.save_round(round_index, selection=review.model_dump(mode='json'),
                selection_evidence_id=review_id, stage='MEASUREMENTS')
            if self.prepare_only:
                self.check_walltime()
                return dict(status='PREPARED_FOR_GPU', round_index=round_index,
                    parent_trial_id=best['trial_id'],
                    candidate_ids=[row['trial_id'] for row in candidates],
                    selection=review.model_dump(mode='json'), usage=self.store.usage(),
                    cumulative_search_seconds=self.store.search_seconds(),
                    new_gpu_execution_performed=False,
                    hardware_revalidation_required=True)
            resumed_ids = set(checkpoint.get('resumed_measurement_candidate_ids', []))
            self.store.save_round(round_index, measurement_execution_started=True)
            measured = [row for row in self.store.trials() if row['trial_id'] in resumed_ids]
            if len(measured) != len(resumed_ids) or any(not (row.get('measurement') or {}).get(
                    'performance_eligible') for row in measured):
                raise ValueError('接续测量必须引用已登记的完整训练结果')
            profiled = [self.execute_profile(identifier) for identifier in review.accepted_profiling_request_ids]
            if any(row['state'] == 'COMPLETED' for row in profiled):
                self.synchronize_cost_revision()
                candidate_ids = {row['trial_id'] for row in candidates}
                if self.config.single_decision_round:
                    proposed_ids = {identifier for entry in checkpoint['constraint_decision']['intent_results']
                        for identifier in entry['candidate_ids']}
                    candidate_ids.intersection_update(proposed_ids)
                candidates = [row for row in self.store.trials()
                    if row['trial_id'] in candidate_ids and row.get('prediction') is not None
                    and row['prediction'].get('cost_database_revision') == self.cost_revision]
                post_profile_ranking = self.ranking_sensitivity(candidates)
                if self.config.single_decision_round:
                    reranked, reranked_id = self.select_joint_investigation(candidates,
                        checkpoint['constraint_decision']['intent_results'], strategy_id, allow_profiles=False)
                elif (self.config.agent_execution_enabled and self.config.selection_policy == 'investigation'
                        and not fast_agent_rounds):
                    updated_context = self.investigation_agent_context(
                        best, candidates, strategy, interactions, interaction_id)
                    updated_context['previous_selection'] = review.model_dump(mode='json')
                    updated_context['previous_selection_evidence_id'] = review_id
                    reranked, reranked_id = self.invoke_with_tools('ExperimentInvestigationAgent',
                        '新增成本已经写入并重新预测。检查更新后的执行证据，重新选择候选和测量顺序。'
                        'accepted_candidate_ids按实际测量优先级返回；完整成本缺失时可以明确选择exploration_candidate_ids。'
                        '说明保留或改变前次选择的具体依据。', updated_context, InvestigationSelection)
                    reranked = self.resolve_selection_candidate_ids(reranked, candidates)
                else:
                    reranked, reranked_id = self.select_experiments(candidates)
                if not self.config.single_decision_round:
                    reranked = self.preserve_cross_family_measurement(reranked, candidates)
                if not reranked.accepted_candidate_ids and all(
                        row['prediction']['performance']['step_time_ms'] is None for row in candidates):
                    reranked = reranked.model_copy(update=dict(
                        accepted_candidate_ids=review.accepted_candidate_ids,
                        exploration_candidate_ids=review.exploration_candidate_ids,
                        reason='成本补测后仍缺少完整排序，继续执行Agent选择的真实探索候选。'))
                retained_exploration = [identifier for identifier in review.exploration_candidate_ids
                    if identifier in reranked.accepted_candidate_ids]
                review = reranked.model_copy(update=dict(
                    accepted_profiling_request_ids=[],
                    exploration_candidate_ids=list(dict.fromkeys(
                        [*reranked.exploration_candidate_ids, *retained_exploration])),
                    evidence_refs=list(dict.fromkeys(
                        [*review.evidence_refs, *reranked.evidence_refs])),
                    reason=reranked.reason + '；补充成本写入后已经重新预测全部受影响候选并重新选择训练计划。'))
                review_id = reranked_id
                self.store.save_round(round_index,
                    selection=review.model_dump(mode='json'), selection_evidence_id=review_id,
                    post_profile_ranking_sensitivity=post_profile_ranking)
            exploration_enabled = (self.store.usage().get('exploration_trials', 0)
                < self.config.uncalibrated_exploration_trials)
            review = self.preserve_unranked_exploration(
                review, candidates, not self.config.budgeted_investigation_enabled
                and self.config.gpu_execution_enabled and exploration_enabled and not any(
                    row['state'] == 'COMPLETED' for row in profiled),
                agent_owned=self.config.agent_execution_enabled and self.config.selection_policy == 'investigation')
            self.store.save_round(round_index,
                selection=review.model_dump(mode='json'), selection_evidence_id=review_id)
            candidates = self.selected_candidate_pool(review, candidates, best)
            by_id = {row['trial_id']: row for row in candidates}
            selected = [by_id[identifier] for identifier in review.accepted_candidate_ids]
            training_validated = [identifier for identifier in review.load_validation_candidate_ids
                if identifier in review.accepted_candidate_ids]
            if training_validated:
                self.store.add_search_state('CHECKPOINT_VALIDATION_IN_TRAINING', dict(
                    candidate_ids=training_validated,
                    warmup=self.config.benchmark_warmup, measured=self.config.benchmark_measured,
                    checks=['native_checkpoint_load', 'finite_loss', 'finite_gradient_norm',
                            'successful_optimizer_updates']))
            load_validations = {}
            selected = [next(row for row in self.store.trials() if row['trial_id'] == trial['trial_id'])
                        for trial in selected]
            for trial in selected:
                completed_benchmarks = sum(row['measurement']['status'] == 'PASS' for row in measured)
                if completed_benchmarks >= self.config.budget.max_benchmarks_per_round:
                    break
                exploration = trial['trial_id'] in review.exploration_candidate_ids
                if (trial['trial_id'] in load_validations
                        and load_validations[trial['trial_id']]['status'] != 'PASS'):
                    continue
                if trial['prediction']['memory']['admission'] == 'REJECT_OOM':
                    continue
                if (trial['prediction']['memory']['admission'] != 'WITHIN_ESTIMATED_CAPACITY'
                        and not exploration
                        and trial['plan']['backend_config'].get(
                            'execution_backend', 'megatron_ddp') == 'megatron_fsdp'):
                    continue
                previously_measured = trial.get('measurement') is not None
                result = self.execute_trial(trial, exploration=exploration)
                if result and not previously_measured:
                    measured.append(result)
            load_validations = {identifier: self.execute_load_validation(by_id[identifier])
                for identifier in review.load_validation_candidate_ids
                if identifier not in training_validated}
            checkpoint = self.store.get_round(round_index)
            if checkpoint.get('feedback_context_version') != 'candidate_outcomes_v2_comparison':
                checkpoint = self.store.save_round(round_index, stage='FEEDBACK',
                    feedback_context_version='candidate_outcomes_v2_comparison',
                    feedback_context=self.visible_observation(
                        self.feedback_agent_context(best, measured, profiled, candidates,
                            checkpoint.get('agent_selection'), review.model_dump(mode='json'))))
            saved_feedback = (InvestigationDecision.model_validate(checkpoint['feedback'])
                if 'feedback' in checkpoint else None)
            if (saved_feedback is not None and self.evidence_reference_feedback(
                    'ExperimentInvestigationAgent', saved_feedback) is None):
                feedback = saved_feedback
                feedback_id = checkpoint['feedback_evidence_id']
            elif full_agents and not fast_agent_rounds:
                feedback, feedback_id = self.invoke_with_tools('ExperimentInvestigationAgent',
                    '依据本轮实际候选、编译与预测结果、测量及预算反馈，解释原有预期受到什么支持或限制。'
                    '没有新增实测时明确说明，并根据已发生的搜索给出下一轮应调查的具体配置、模块或成本。'
                    '引用提供的证据，区分Agent原始选择与程序调整，禁止填写未测量的收益。'
                    '输出InvestigationDecision，requested_profiles记录仍需调查的成本。',
                    checkpoint['feedback_context'], InvestigationDecision)
                checkpoint = self.store.save_round(round_index,
                    feedback=feedback.model_dump(mode='json'), feedback_evidence_id=feedback_id)
            else:
                feedback = InvestigationDecision(
                    conclusions=['本轮候选、预测状态、实测结果与待执行成本请求已经保存'],
                    evidence_refs=[trial['measurement']['run_id'] for trial in measured]
                        or [best['measurement']['run_id']],
                    next_focus=['依据已调查候选与成本覆盖继续搜索'], requested_profiles=[])
                feedback_id = fingerprint(feedback.model_dump(mode='json'))
                self.store.add_evidence(feedback_id, 'HYPOTHESIS', feedback.model_dump(mode='json'))
            self.store.add_note(measured[0]['trial_id'] if measured else best['trial_id'],
                'HYPOTHESIS', feedback.model_dump(), feedback.evidence_refs)
            next_best = self.store.best_measured()
            if next_best is not None:
                if (best['measurement'] is None
                        or not best['measurement'].get('performance_eligible', False)
                        or next_best['measurement']['tokens_per_second'] > best['measurement']['tokens_per_second']):
                    no_improvement = 0
                elif any(row['measurement']['status'] == 'PASS' for row in measured):
                    no_improvement += 1
                best = next_best
            self.complete_round(round_index, best, no_improvement=no_improvement,
                measured_candidate_ids=[trial['trial_id'] for trial in measured],
                feedback=feedback.model_dump(mode='json'), feedback_evidence_id=feedback_id,
                stop_reason=('本轮Agent提议没有新增可执行测量；检查已测重复配置与容量拒绝条件，提交不同的完整生效配置。'
                    if self.config.single_decision_round and not (review.accepted_candidate_ids
                        or review.accepted_profiling_request_ids or review.load_validation_candidate_ids) else None),
                best_trial_id=best['trial_id'])
            self.store.charge(f'round:{round_index}', 'agent_rounds', 1)
            self.store.export(self.output / 'trials.jsonl')
            if no_improvement >= self.config.budget.max_no_improvement_rounds:
                break
        return self.finish()
