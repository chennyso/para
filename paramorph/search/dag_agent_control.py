from collections import defaultdict

from pydantic import Field

from contracts import Contract, fingerprint
from predictor.profile_database import same_training_plan
from search.execution_decisions import candidate_decisions
from search.investigation_control import PlanningInvestigation
from search.system_search import BottleneckDecision, EvidenceGuidedSearch, InvestigationSelection


class DAGCandidateReview(Contract):
    admitted_candidate_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class DAGAgentSearch(EvidenceGuidedSearch):
    def __init__(self, config, *, prepare_only=False):
        super().__init__(config, prepare_only=prepare_only)
        if (not self.config.planning_control_enabled
                or self.config.planning_selector != 'agent'
                or self.config.planning_partition_engine != 'path'
                or not self.config.agent_execution_enabled
                or self.config.evaluation_mode not in ('full', 'multi_agent')):
            raise ValueError(
                'DAG-Agent搜索要求planning_control_enabled、planning_selector=agent、'
                'planning_partition_engine=path、agent_execution_enabled及正式多智能体模式')
        if self.config.single_decision_round:
            raise ValueError('DAG-Agent完整协作需要允许四个Agent独立执行决策')

    def dag_task_guidance(self, parent):
        prediction = parent.get('prediction')
        source, scope = self.bottleneck_graph(parent)
        revision = self.cost_revision
        key = (parent['trial_id'], prediction.get('prediction_id') if prediction else None, revision)
        cache = getattr(self, '_dag_guidance_cache', {})
        if key in cache:
            return cache[key]
        report = dict(kind='dag_task_impact_guidance', parent_trial_id=parent['trial_id'],
            parent_plan_hash=parent['plan_hash'],
            source_prediction_id=prediction.get('prediction_id') if prediction else None,
            cost_revision=revision, graph_id=source.graph_id, graph_scope=scope,
            impact_ratio=0.1, task_groups=[])
        if scope == 'PREDICTED_COSTED_DAG':
            resource_model = self.predictor.projector.parameters.get('resource_replay', {})
            baseline = source.replay(resource_model=resource_model)
            structural = self.backend_dag(parent)['tasks']
            index = self.rewriter.compiler.decision_impact_index(
                self.rewriter.candidate(parent['plan']['backend_config']))
            groups = defaultdict(list)
            for task_id in baseline['critical_task_ids']:
                task = source.records[task_id]
                if task['duration_ms'] > 0:
                    groups[(task.get('region'), task['kind'])].append(task_id)
            ordered = sorted(groups.items(), key=lambda item: (
                -sum(source.records[task_id]['duration_ms'] for task_id in item[1]),
                str(item[0])))
            for (region, operation), task_ids in ordered[:5]:
                impact = source.epsilon_impact(task_ids, 0.1,
                    resource_model=resource_model, baseline_replay=baseline)
                origin_ids = sorted({
                    origin for task_id in task_ids
                    if (origin := source.records[task_id].get('metadata', {}).get(
                        'origin_task', task_id)) in structural})
                variables = sorted(name for name, row in index.items()
                    if set(row['affected_task_ids']).intersection(origin_ids))
                report['task_groups'].append(dict(region_id=region, operation=operation,
                    task_ids=task_ids[:8], task_count=len(task_ids),
                    native_task_ids=origin_ids[:8], related_variables=variables,
                    impact_ms=impact['impact_ms'],
                    baseline_step_time_ms=impact['baseline_step_time_ms'],
                    modified_step_time_ms=impact['counterfactual_step_time_ms'],
                    evidence_type='PREDICTED_TASK_DURATION_IMPACT'))
            self.store.charge('dag-impact:' + fingerprint(report), 'counterfactual_replays',
                1 + len(report['task_groups']))
            report['status'] = 'AVAILABLE'
        else:
            report.update(status='STRUCTURAL_ONLY', reason='当前父配置缺少完整可回放任务成本',
                required_costs=(prediction or {}).get('evidence', {}).get('profiling_requests', [])[:4])
        identifier = fingerprint(report)
        self.store.add_evidence(identifier, 'PREDICTED', report)
        report = dict(report, evidence_id=identifier)
        cache[key] = report
        self._dag_guidance_cache = cache
        return report

    def planning_observation(self, parent):
        observation = super().planning_observation(parent)
        if self.config.agent_ablation in ('without_dag_evidence', 'without_dag_model'):
            return observation
        guidance = self.dag_task_guidance(parent)
        observation['dag_task_guidance'] = guidance
        observation['evidence_ids'] = list(dict.fromkeys(
            [*observation['evidence_ids'], guidance['evidence_id']]))
        if self.config.agent_ablation in ('without_agent', 'without_bottleneck_agent'):
            return observation
        key = (parent['trial_id'], guidance['evidence_id'])
        cache = getattr(self, '_dag_analysis_cache', {})
        if key not in cache:
            analysis, evidence_id = self.invoke_with_tools('PerformanceAnalysisAgent',
                '依据dag_task_guidance的任务影响、实际依赖和资源等待选择调查问题。'
                'affected_tasks仅引用当前父配置结构DAG的真实Task ID，'
                '可直接使用native_task_ids；性能未知时仅提出结构假设。'
                '说明涉及哪些关联配置变量及还需要什么成本测量。',
                observation, BottleneckDecision)
            cache[key] = dict(decision=analysis.model_dump(mode='json'),
                evidence_id=evidence_id)
            self._dag_analysis_cache = cache
            self.store.add_search_state('DAG_ANALYSIS_AGENT',
                dict(parent_trial_id=parent['trial_id'], guidance_id=guidance['evidence_id'],
                    **cache[key]))
        observation['analysis_agent'] = cache[key]
        observation['evidence_ids'] = list(dict.fromkeys(
            [*observation['evidence_ids'], cache[key]['evidence_id']]))
        return observation

    def validate_investigation(self, action):
        result = super().validate_investigation(action)
        if (self.config.agent_ablation in ('without_dag_evidence', 'without_dag_model')
                or not self.config.agent_execution_enabled):
            return result
        if not isinstance(action, PlanningInvestigation):
            action = PlanningInvestigation.model_validate(action)
        documents = []
        for identifier in action.evidence_refs:
            row = self.store.connection.execute(
                'SELECT document FROM evidence WHERE id=?', (identifier,)).fetchone()
            if row is not None:
                import json
                documents.append(json.loads(row[0]))
        if not any(row.get('kind') == 'dag_task_impact_guidance'
                and row.get('parent_trial_id') == action.parent_trial_id
                for row in documents):
            raise ValueError(dict(status='REVISION_REQUIRED',
                reason='规划动作需要引用同一父配置的DAG任务影响证据',
                parent_trial_id=action.parent_trial_id,
                required_evidence='dag_task_guidance.evidence_id'))
        return result

    def execute_planning_measurements(self, parent, candidate_ids, action_ids):
        candidates = [row for row in self.store.trials()
            if row['trial_id'] in candidate_ids
            and row.get('prediction') is not None
            and row.get('measurement') is None and not row.get('failure')
            and row['prediction']['memory'].get('admission') != 'REJECT_OOM']
        if not candidates:
            return []
        candidate_evidence = [
            row['prediction'].get('planning_evidence_id') or row['prediction']['prediction_id']
            for row in candidates]
        context = self.shared_context(parent) | dict(
            parent_trial_id=parent['trial_id'], planning_action_evidence_refs=action_ids,
            candidate_ids=[row['trial_id'] for row in candidates],
            candidate_evidence_refs=candidate_evidence,
            native_completed_candidates=[dict(candidate_id=row['trial_id'],
                plan_signature=row['prediction']['plan_signature'],
                prediction_status=row['prediction']['prediction_status'],
                memory_admission=row['prediction']['memory'].get('admission'),
                step_time_ms=row['prediction']['performance'].get('step_time_ms'),
                required_profiles=row['prediction'].get('evidence', {}).get(
                    'profiling_requests', [])[:3]) for row in candidates])
        review, review_id = self.invoke_with_tools('ConstraintValidationAgent',
            '原生约束求解器已经完成所有提交配置的结构合法性检查。'
            '从candidate_ids中选择允许交给Evaluation Agent的完整候选，'
            '只能引用已有candidate_id与candidate_evidence_refs。'
            '显存或耗时未知的候选可以保留，交由后续成本测量及训练步验证。'
            '不得声称任何候选通过了未发生的实际训练。',
            context, DAGCandidateReview)
        valid_ids = {row['trial_id'] for row in candidates}
        admitted = set(review.admitted_candidate_ids)
        if len(review.admitted_candidate_ids) != len(admitted) or not admitted <= valid_ids:
            raise ValueError('Validation Agent引用了无效或者重复的完整候选')
        self.store.add_search_state('DAG_VALIDATION_AGENT', dict(
            parent_trial_id=parent['trial_id'], action_ids=action_ids,
            admitted_candidate_ids=sorted(admitted),
            decision_evidence_id=review_id, reason=review.reason))
        if not admitted:
            return []
        def active_candidates():
            return [row for row in self.store.trials() if row['trial_id'] in admitted
                and not row.get('measurement') and not row.get('failure')
                and (row.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM']
        def selection_context():
            rows = active_candidates()
            return self.shared_context(parent) | dict(
                parent_trial_id=parent['trial_id'],
                candidates=self.investigation_candidate_table(rows),
                candidate_decision_report=candidate_decisions(parent, rows),
                planning_action_evidence_refs=action_ids,
                validation_decision_evidence_id=review_id,
                profiling_requests=[dict(request_id=row['request_id'],
                    measurement_type=row['measurement_type'],
                    estimated_gpu_hours=row['estimated_gpu_hours'], state=row['state'],
                    affected_candidate_ids=sorted(admitted.intersection(row['trial_ids'])))
                    for row in self.store.profile_requests()
                    if row['state'] in ('PENDING', 'DEFERRED')
                    and admitted.intersection(row['trial_ids'])])
        selection, selection_id = self.invoke_with_tools('ExperimentInvestigationAgent',
            '依据经过原生约束检查的候选、DAG影响证据、缺失成本和GPU费用，'
            '自主选择组件测量及完整训练。显存不确定的配置需经过训练步验证。',
            selection_context(), InvestigationSelection)
        selection = self.resolve_selection_candidate_ids(selection, active_candidates())
        if not set(selection.accepted_candidate_ids) <= admitted:
            raise ValueError('Evaluation Agent选择了未通过Validation Agent审查的候选')
        self.store.add_search_state('DAG_GPU_SELECTION', dict(
            selection=selection.model_dump(mode='json'), evidence_id=selection_id,
            validation_evidence_id=review_id, action_ids=action_ids))
        completed_profile_ids = []
        for identifier in selection.accepted_profiling_request_ids:
            request = next(row for row in self.store.profile_requests()
                if row['request_id'] == identifier)
            if not admitted.intersection(request['trial_ids']):
                raise ValueError('Profile请求未关联本轮已审查的候选')
            if request['measurement_type'] == 'native_training_cost_profile':
                for trial in self.store.trials():
                    if (trial['trial_id'] in request['trial_ids']
                            and same_training_plan(trial['plan']['backend_config'],
                                request['execution_request'])
                            and (trial.get('prediction') or {}).get('memory', {}).get(
                                'admission') == 'REQUIRE_MEMORY_PROFILE'):
                        validation = self.execute_load_validation(trial)
                        if validation['status'] != 'PASS':
                            self.store.update_profile_request(identifier,
                                state='DEFERRED' if validation['status'] == 'OUT_OF_BUDGET'
                                else 'FAILED',
                                reason='完整配置训练步验证状态：' + validation['status'])
                            break
                else:
                    result = self.execute_profile(identifier)
                    if result['state'] == 'COMPLETED':
                        completed_profile_ids.append(identifier)
                continue
            result = self.execute_profile(identifier)
            if result['state'] == 'COMPLETED':
                completed_profile_ids.append(identifier)
        if completed_profile_ids:
            self.synchronize_cost_revision()
            for trial in active_candidates():
                if trial['prediction'].get('cost_database_revision') != self.cost_revision:
                    if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                        break
                    self.predict_trial(trial, force=True)
            updated, updated_id = self.invoke_with_tools('ExperimentInvestigationAgent',
                '组件Profile已完成并更新全部可评价候选。重新决定本轮正式训练对象。'
                '新增Profile请求保留供后续轮次，不得沿用已经失效的性能排序。',
                selection_context() | dict(
                    completed_profile_request_ids=completed_profile_ids,
                    previous_selection_evidence_id=selection_id),
                InvestigationSelection)
            selection = self.resolve_selection_candidate_ids(updated, active_candidates())
            selection_id = updated_id
            self.store.add_search_state('DAG_POST_PROFILE_SELECTION', dict(
                selection=selection.model_dump(mode='json'),
                evidence_id=selection_id, completed_profiles=completed_profile_ids))
        measured = []
        for identifier in selection.accepted_candidate_ids:
            if identifier not in admitted:
                raise ValueError('GPU训练候选未通过本轮约束审查')
            if len(measured) >= self.config.budget.max_benchmarks_per_round:
                break
            trial = next(row for row in self.store.trials()
                if row['trial_id'] == identifier)
            if trial.get('measurement') is not None or trial.get('failure'):
                continue
            prediction = trial.get('prediction') or {}
            if prediction.get('memory', {}).get('admission') == 'REJECT_OOM':
                continue
            if (prediction.get('memory', {}).get('admission') != 'WITHIN_ESTIMATED_CAPACITY'
                    or trial['plan']['backend_config'].get(
                        'execution_backend') == 'megatron_fsdp'):
                validation = self.execute_load_validation(trial)
                if validation['status'] != 'PASS':
                    continue
                trial = next(row for row in self.store.trials()
                    if row['trial_id'] == identifier)
            result = self.execute_trial(trial,
                exploration=identifier in selection.exploration_candidate_ids
                or prediction.get('performance', {}).get('step_time_ms') is None)
            if result is not None:
                measured.append(result)
        self._planning_last_measured_feedback = candidate_decisions(parent, measured)
        self.store.add_search_state('PLANNING_TRAINING_FEEDBACK',
            self._planning_last_measured_feedback)
        return measured
