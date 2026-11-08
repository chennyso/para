from copy import deepcopy


STRUCTURAL_TOOLS = frozenset({
    'inspect_training_graph', 'inspect_critical_path', 'inspect_dag_subgraph',
    'inspect_measured_trace', 'inspect_task_dependencies', 'inspect_region_cost',
    'inspect_communication_flow', 'inspect_rank_topology', 'inspect_shared_resources',
    'query_affected_decisions', 'inspect_execution_graph', 'compare_candidate_graph_delta',
    'inspect_region', 'inspect_communication', 'analyze_transformation_interactions',
    'preview_transformation',
    'inspect_execution_impact',
})
HISTORY_TOOLS = frozenset({'get_trial_history', 'inspect_trial_history'})
GRAPH_FIELDS = frozenset({
    'physical_parallel_alternatives', 'cross_shape_investigations', 'schedule_controls',
    'execution_revisions',
    'execution_impact',
    'parent_native_execution_effects', 'native_execution_effects',
    'execution_view', 'execution_structure',
    'diagnostic_evidence',
    'execution_analysis', 'candidate_execution_analysis',
    'dag_index', 'model_graph', 'training_graph', 'execution_graph', 'graph_delta', 'native_pipeline_delta',
    'critical_path', 'critical_path_graph', 'critical_tasks', 'critical_regions',
    'critical_path_tasks', 'affected_tasks', 'affected_regions', 'target_regions',
    'target_task_ids', 'target_flow_ids', 'affected_tasks', 'affected_task_ids', 'changed_target_ids',
    'target_impact', 'effect_preview', 'impact_index', 'transformation_interactions',
    'interaction_summary', 'candidate_execution_effects', 'parent_execution_structure',
    'communication_placement', 'region_costs', 'stage_summary', 'source_profile_stage_summary',
    'source_profile_rank_summary',
    'rank_summary', 'stage_forward_ms', 'stage_backward_ms', 'pipeline_idle_ms',
    'modeled_exposed_communication_ms', 'tasks', 'edges', 'flows', 'regions',
    'task_ids', 'flow_ids', 'region_ids', 'task_id', 'flow_id', 'region_id',
    'available_task_ids', 'stable_task_id_samples',
    'task_count', 'edge_count', 'flow_count', 'region_count', 'graph_sha256',
    'execution_effects', 'collective_tasks', 'communication_boundary',
    'cost_key', 'cost_keys', 'cost_condition_catalog', 'region_implementations',
    'continuation_witness', 'full_records_location', 'left_critical_path', 'right_critical_path',
    'peak_per_rank_bytes', 'memory_headroom_per_rank_bytes', 'liveness_intervals', 'memory_events',
})
MEMORY_FIELDS = frozenset({
    'prior_episode',
    'proposal_archive',
    'policy_comparisons',
    'strategy_decision_evidence',
    'recent_history', 'recent_search_rounds', 'recent_profiling_results',
    'recent_note_evidence_refs', 'measured_strategy_memory', 'conditional_failure_memory',
    'family_status_counts', 'best_measured', 'best_measured_tokens_per_s',
    'best_predicted', 'best_predicted_step_ms', 'candidate_count', 'lower_bound_ms',
    'source_profile_configuration', 'source_profile_stage_summary', 'source_profile_rank_summary',
})


class AblationEvidencePolicy:
    def __init__(self, mode, current_ids=(), historical_ids=()):
        self.mode = mode
        self.current_ids = frozenset(current_ids)
        self.historical_ids = frozenset(historical_ids) - self.current_ids
        self.without_graph = mode in ('without_dag_evidence', 'without_dag_model')
        self.without_memory = mode == 'without_search_memory'

    @property
    def contract(self):
        return dict(mode=self.mode, version='agent_evidence_visibility_v2_execution_view',
            graph_evaluator='aggregate_stage_model' if self.mode == 'without_dag_model' else 'unchanged',
            native_constraints='unchanged',
            candidate_space='unchanged', scalar_scores='available')

    def withheld(self, tool):
        return dict(status='WITHHELD_BY_ABLATION', tool=tool,
            ablation_contract=self.contract, detailed_records=0)

    def intercept(self, request, current_ids):
        if self.without_memory:
            if request.tool in HISTORY_TOOLS:
                return self.withheld(request.tool)
            if request.candidate_ids and not set(request.candidate_ids) <= set(current_ids):
                return self.withheld(request.tool)
        if self.without_graph and request.tool in STRUCTURAL_TOOLS and request.tool not in (
                'preview_transformation', 'analyze_transformation_interactions'):
            return self.withheld(request.tool)
        return None

    def observation(self, value):
        if not self.without_graph and not self.without_memory:
            return value

        def visit(item):
            if isinstance(item, list):
                return [visit(row) for row in item]
            if isinstance(item, str) and self.without_memory:
                for identifier in self.historical_ids:
                    item = item.replace(identifier, 'HISTORY_WITHHELD')
                return item
            if not isinstance(item, dict):
                return item
            tool = item.get('tool')
            tool_name = tool.get('tool') if isinstance(tool, dict) else tool
            if 'result' in item and self.without_graph and tool_name in STRUCTURAL_TOOLS:
                result = item['result']
                if result.get('status') in ('CONSTRAINT_CONFLICT', 'NEEDS_PROFILE'):
                    projected = visit(result)
                else:
                    projected = self.withheld(tool_name)
                    for field in ('strategy_effect_identity', 'strategy_effect_identities',
                            'proposal_identity', 'native_validation', 'validation'):
                        if field in result:
                            projected[field] = result[field]
                    if isinstance(result.get('previews'), list):
                        projected['previews'] = [{key: row[key] for key in
                            ('strategy_effect_identity', 'proposal_identity', 'native_validation') if key in row}
                            for row in result['previews']]
                return {key: visit(entry) for key, entry in item.items() if key != 'result'} | dict(result=projected)
            if 'result' in item and self.without_memory and tool_name in HISTORY_TOOLS:
                return {key: visit(entry) for key, entry in item.items() if key != 'result'} | dict(
                    result=self.withheld(tool_name))
            if 'result' in item and self.without_memory and tool_name == 'get_best_plan':
                best = item['result'].get('best_measured')
                projected = (dict(current_parent=visit(best))
                    if best is not None and best.get('trial_id') in self.current_ids
                    else self.withheld(tool_name))
                return {key: visit(entry) for key, entry in item.items() if key != 'result'} | dict(result=projected)
            hidden = (GRAPH_FIELDS if self.without_graph else frozenset()) | (
                MEMORY_FIELDS if self.without_memory else frozenset())
            result = {key: visit(entry) for key, entry in item.items() if key not in hidden}
            if self.without_graph and isinstance(item.get('rank_summary'), list):
                result['rank_memory_summary'] = [{key: row[key] for key in
                    ('rank', 'peak_bytes', 'persistent_bytes', 'capacity_bytes') if key in row}
                    for row in item['rank_summary']]
            if self.without_graph and 'request_id' in item and 'measurement_type' in item:
                result.pop('reason', None)
            if self.without_memory and isinstance(item.get('trial_ids'), list):
                result['trial_ids'] = [identifier for identifier in item['trial_ids'] if identifier in self.current_ids]
            if self.without_memory and isinstance(item.get('columns'), list) and 'family_id' in item['columns']:
                retained = [index for index, name in enumerate(item['columns'])
                    if name not in MEMORY_FIELDS and name != 'status']
                result['columns'] = [item['columns'][index] for index in retained]
                result['rows'] = [[visit(row[index]) for index in retained] for row in item.get('rows', [])]
            if self.without_memory and 'family_id' in item and 'values' in item:
                result.pop('status', None)
            if self.without_graph and isinstance(item.get('columns'), list):
                retained = [index for index, name in enumerate(item['columns']) if name not in GRAPH_FIELDS]
                result['columns'] = [item['columns'][index] for index in retained]
                if 'rows' in item:
                    result['rows'] = [[visit(row[index]) for index in retained] for row in item['rows']]
            if isinstance(result.get('deferred_context'), list):
                result['deferred_context'] = [row for row in result['deferred_context']
                    if row.get('field', '').split('.')[0] not in hidden
                    and not (self.without_memory and row.get('access') in HISTORY_TOOLS)]
            return result

        return visit(deepcopy(value))

    @staticmethod
    def current_trial_ids(observation):
        identifiers = {observation[key] for key in ('parent_trial_id', 'previous_best_trial_id')
            if isinstance(observation.get(key), str)}
        plan = observation.get('current_training_plan') or {}
        if isinstance(plan.get('plan_id'), str):
            identifiers.add(plan['plan_id'])
        candidates = observation.get('candidates', [])
        if isinstance(candidates, dict):
            columns = candidates.get('columns', [])
            if 'trial_id' in columns:
                identifiers.update(row[columns.index('trial_id')] for row in candidates.get('rows', []))
            identifiers.update(row[0] for row in candidates.get('overview', []))
        else:
            identifiers.update(row['trial_id'] for row in candidates if 'trial_id' in row)
        for field in ('candidate_outcomes', 'measured', 'candidate_hypotheses'):
            identifiers.update(row['trial_id'] for row in observation.get(field, []) if 'trial_id' in row)
        history = observation.get('tool_history', [])
        for receipt in history if isinstance(history, list) else []:
            tool = receipt.get('tool', {})
            name = tool.get('tool') if isinstance(tool, dict) else tool
            if name not in ('complete_training_plans', 'solve_conditional_partition', 'validate_training_plans'):
                continue
            result = receipt.get('result', {})
            for row in result.get('intent_results', []):
                identifiers.update(row.get('candidate_ids', []))
            for row in result.get('validations', []):
                identifiers.add(row['candidate_id'])
        return identifiers

    @property
    def instructions(self):
        if self.without_graph:
            return ('本实验屏蔽细粒度执行图、任务、依赖、通信流和阶段成本。'
                '继续使用模型结构、完整配置、原生合法性、标量预测及真实训练结果。'
                '目标任务与Flow引用使用空列表；完整策略仍引用真实预览身份收据。'
                '结构查询返回WITHHELD_BY_ABLATION，依据可见信息自主决定完整策略及测量。')
        if self.without_memory:
            return ('本实验屏蔽早于当前轮次的显式策略、失败及搜索历史。'
                '使用当前父计划、当前候选、当前轮次工具结果及反馈，后端评价器保持原样。'
                '历史查询返回WITHHELD_BY_ABLATION。保持完整策略、原生检查及费用限制。')
        return ''
