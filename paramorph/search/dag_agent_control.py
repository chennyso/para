import json
from collections import defaultdict

from pydantic import Field

from contracts import Contract, fingerprint
from predictor.profile_database import same_training_plan
from search.execution_decisions import candidate_decisions
from search.investigation_control import PlanningInvestigation
from search.system_search import (
    BottleneckDecision, EvidenceGuidedSearch, InvestigationSelection, token_count,
)


class DAGCandidateReview(Contract):
    admitted_candidate_ids: list[str] = Field(min_length=1, max_length=8)
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
                'DAG-Agent搜索需要原生Planning、Agent选择、执行路径CP-SAT和完整Agent执行')
        if self.config.single_decision_round:
            raise ValueError('DAG-Agent联合搜索需要允许Analysis和Validation独立决策')

    def dag_task_guidance(self, parent):
        prediction = parent.get('prediction') or {}
        source, scope = self.bottleneck_graph(parent)
        key = (
            parent['trial_id'], prediction.get('prediction_id'), self.cost_revision,
            source.graph_id,
        )
        cache = getattr(self, '_dag_guidance_cache', {})
        if key in cache:
            return cache[key]
        report = dict(
            kind='dag_task_impact_guidance', parent_trial_id=parent['trial_id'],
            parent_plan_hash=parent['plan_hash'],
            source_prediction_id=prediction.get('prediction_id'),
            cost_revision=self.cost_revision, graph_id=source.graph_id,
            graph_scope=scope, impact_ratio=0.1, task_groups=[],
        )
        if (scope == 'PREDICTED_COSTED_DAG'
                and prediction.get('prediction_status') != 'NEEDS_PROFILE'):
            resource_model = self.predictor.projector.parameters.get(
                'resource_replay', {})
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
                str(item[0]),
            ))
            for (region, operation), task_ids in ordered[:5]:
                impact = source.epsilon_impact(
                    task_ids, 0.1, resource_model=resource_model,
                    baseline_replay=baseline,
                )
                origin_ids = sorted({
                    origin for task_id in task_ids
                    if (origin := source.records[task_id].get(
                        'metadata', {}).get('origin_task', task_id)) in structural
                })
                variables = sorted(
                    name for name, row in index.items()
                    if set(row['affected_task_ids']).intersection(origin_ids)
                )
                report['task_groups'].append(dict(
                    region_id=region, operation=operation, task_ids=task_ids[:8],
                    native_task_ids=origin_ids[:8], task_count=len(task_ids),
                    related_variables=variables, impact_ms=impact['impact_ms'],
                    baseline_step_time_ms=impact['baseline_step_time_ms'],
                    modified_step_time_ms=impact['counterfactual_step_time_ms'],
                    evidence_type='PREDICTED_TASK_DURATION_IMPACT',
                ))
            self.store.charge(
                'dag-impact:' + fingerprint(report),
                'counterfactual_replays', 1 + len(report['task_groups']),
            )
            report['status'] = 'AVAILABLE'
        else:
            report.update(
                status='STRUCTURAL_ONLY',
                reason='父策略尚无完整可排序执行时间，任务影响仍待取得相容成本',
                required_costs=prediction.get('evidence', {}).get(
                    'profiling_requests', [])[:4],
            )
        identifier = fingerprint(report)
        self.store.add_evidence(identifier, 'PREDICTED', report)
        saved = dict(report, evidence_id=identifier)
        cache[key] = saved
        self._dag_guidance_cache = cache
        return saved

    def planning_observation(self, parent):
        observation = super().planning_observation(parent)
        if self.config.agent_ablation in (
                'without_dag_evidence', 'without_dag_model'):
            return observation
        guidance = self.dag_task_guidance(parent)
        observation['dag_task_guidance'] = guidance
        observation['dag_guidance_contract'] = dict(
            parent_trial_id=parent['trial_id'],
            required_evidence_ref=guidance['evidence_id'],
            instruction='每项针对当前父策略的调查需引用DAG影响证据；'
                '根据任务影响自主选择关联变量与模块范围；'
                '成本未知时允许提出结构假设并记录缺失测量。',
        )
        observation['evidence_ids'] = list(dict.fromkeys([
            *observation['evidence_ids'], guidance['evidence_id'],
        ]))
        key = (parent['trial_id'], guidance['evidence_id'])
        cache = getattr(self, '_dag_analysis_cache', {})
        if key not in cache:
            analysis, identifier = self.invoke_with_tools(
                'PerformanceAnalysisAgent',
                '依据dag_task_guidance和实际依赖选择本轮调查的任务与执行瓶颈。'
                'affected_tasks必须引用当前父策略的真实结构Task ID；'
                '可以使用native_task_ids。性能成本缺失时只能提交结构假设。'
                '给出相关配置变量和需要的测量。',
                observation, BottleneckDecision,
            )
            cache[key] = dict(
                decision=analysis.model_dump(mode='json'),
                evidence_id=identifier,
            )
            self._dag_analysis_cache = cache
            self.store.add_search_state('DAG_ANALYSIS_AGENT', dict(
                parent_trial_id=parent['trial_id'],
                guidance_id=guidance['evidence_id'], **cache[key],
            ))
        observation['analysis_agent'] = cache[key]
        observation['evidence_ids'] = list(dict.fromkeys([
            *observation['evidence_ids'], cache[key]['evidence_id'],
        ]))
        return observation

    def validate_investigation(self, action):
        action, parent, shapes, rejected = super().validate_investigation(action)
        if self.config.agent_ablation in (
                'without_dag_evidence', 'without_dag_model'):
            return action, parent, shapes, rejected
        evidence = [
            json.loads(row[0])
            for identifier in action.evidence_refs
            if (row := self.store.connection.execute(
                'SELECT document FROM evidence WHERE id=?',
                (identifier,),
            ).fetchone()) is not None
        ]
        if not any(
                record.get('kind') == 'dag_task_impact_guidance'
                and record.get('parent_trial_id') == action.parent_trial_id
                for record in evidence):
            raise ValueError(dict(
                status='REVISION_REQUIRED',
                reason='联合调查需要引用当前父配置的DAG任务影响证据',
                parent_trial_id=action.parent_trial_id,
                required_evidence='dag_task_guidance.evidence_id',
            ))
        return action, parent, shapes, rejected

    def execute_investigation(self, action):
        receipt = super().execute_investigation(action)
        evaluated = set(receipt['evaluated_candidate_ids'])
        for trial in self.store.trials():
            if trial['trial_id'] not in evaluated:
                continue
            prediction = trial.get('prediction')
            if prediction is None or prediction['prediction_status'] == 'UNSUPPORTED':
                continue
            if (prediction['prediction_status'] == 'NEEDS_PROFILE'
                    or prediction.get('evidence', {}).get('profiling_requests')):
                profile = self.request_profiles(trial, prediction)
                if profile is None and prediction['prediction_status'] == 'NEEDS_PROFILE':
                    self.request_profiles(trial, prediction, native_only=True)
        self.store.add_search_state('DAG_INVESTIGATION_CANDIDATE_COVERAGE', dict(
            action_id=receipt['action_id'],
            evaluated_candidate_ids=receipt['evaluated_candidate_ids'],
            retained_candidate_ids=receipt['candidate_ids'],
            missing_cost_candidates=[
                trial['trial_id'] for trial in self.store.trials()
                if trial['trial_id'] in evaluated
                and (trial.get('prediction') or {}).get(
                    'prediction_status') == 'NEEDS_PROFILE'
            ],
            registered_profiles=[
                row['request_id'] for row in self.store.profile_requests()
                if evaluated.intersection(row['trial_ids'])
            ],
        ))
        return receipt

    def constraint_decision_feedback(self, response, observation):
        if not isinstance(response, DAGCandidateReview):
            return super().constraint_decision_feedback(response, observation)
        valid = set(observation['candidate_ids'])
        invalid = sorted(set(response.admitted_candidate_ids) - valid)
        duplicate = len(set(response.admitted_candidate_ids)) != len(
            response.admitted_candidate_ids)
        if not invalid and not duplicate:
            return None, []
        feedback = dict(
            status='REVISION_REQUIRED',
            revision_owner='ConstraintValidationAgent',
            invalid_candidate_ids=invalid, duplicated_candidate_ids=duplicate,
            permitted_candidate_ids=sorted(valid),
            instruction='从程序已完成原生约束检查的完整候选中重新选择。',
        )
        identifier = fingerprint(feedback)
        self.store.add_evidence(identifier, 'FAILURE', feedback)
        return dict(
            evidence_id=identifier, tool=dict(tool='validate_dag_candidate_review'),
            result=feedback, result_tokens=token_count(feedback),
        ), []

    def execute_planning_measurements(self, parent, candidate_ids, action_ids):
        candidates = [
            row for row in self.store.trials()
            if row['trial_id'] in candidate_ids
            and row.get('prediction') is not None
            and not row.get('measurement') and not row.get('failure')
            and row['prediction']['memory'].get('admission') != 'REJECT_OOM'
        ]
        if not candidates:
            self.store.add_search_state('DAG_NO_MEASURABLE_CANDIDATE', dict(
                parent_trial_id=parent['trial_id'], action_ids=action_ids,
                reason='本轮没有结构合法且通过必要容量下界检查的完整候选',
                requested_candidate_ids=sorted(candidate_ids),
            ))
            return []

        decision = self.shared_context(parent) | dict(
            parent_trial_id=parent['trial_id'],
            candidate_ids=[row['trial_id'] for row in candidates],
            planning_action_evidence_refs=action_ids,
            candidates=self.investigation_candidate_table(candidates),
            candidate_decision_report=candidate_decisions(parent, candidates),
        )
        review, review_id = self.invoke_with_tools(
            'ConstraintValidationAgent',
            '原生约束求解器已经完成候选构造。'
            '从candidate_ids中选出值得继续评价的合法完整配置。'
            '每项输出必须引用已有trial_id，候选性能或容量未知时保留未知状态。'
            '不允许声明尚未执行的GPU验证已经通过。',
            decision, DAGCandidateReview,
        )
        admitted = set(review.admitted_candidate_ids)
        self.store.add_search_state('DAG_VALIDATION_AGENT', dict(
            parent_trial_id=parent['trial_id'], decision_evidence_id=review_id,
            admitted_candidate_ids=sorted(admitted), reason=review.reason,
        ))

        def current_candidates():
            return [
                row for row in self.store.trials()
                if row['trial_id'] in admitted
                and not row.get('measurement') and not row.get('failure')
                and (row.get('prediction') or {}).get(
                    'memory', {}).get('admission') != 'REJECT_OOM'
            ]

        def selection_context():
            rows = current_candidates()
            pending = [
                dict(
                    request_id=row['request_id'],
                    trial_ids=sorted(admitted.intersection(row['trial_ids'])),
                    cost_keys=row['cost_keys'][:4],
                    measurement_type=row['measurement_type'],
                    estimated_gpu_hours=row['estimated_gpu_hours'],
                    state=row['state'],
                )
                for row in self.store.profile_requests()
                if row['state'] in ('PENDING', 'DEFERRED', 'APPROVED')
                and admitted.intersection(row['trial_ids'])
            ]
            return self.shared_context(parent) | dict(
                parent_trial_id=parent['trial_id'],
                candidates=self.investigation_candidate_table(rows),
                candidate_decision_report=candidate_decisions(parent, rows),
                planning_action_evidence_refs=action_ids,
                validation_decision_evidence_id=review_id,
                profiling_requests=pending,
            )

        selection, selection_id = self.invoke_with_tools(
            'ExperimentInvestigationAgent',
            '依据完整候选的DAG性能、成本缺项、显存状态与GPU费用，'
            '自主选择组件校准、短程训练步验证或完整训练。'
            'accepted_candidate_ids对应实际训练优先级。',
            selection_context(), InvestigationSelection,
        )
        selection = self.resolve_selection_candidate_ids(
            selection, current_candidates())
        self.store.add_search_state('DAG_GPU_SELECTION', dict(
            selection=selection.model_dump(mode='json'),
            evidence_id=selection_id, validation_evidence_id=review_id,
        ))

        profiles = []
        for identifier in selection.accepted_profiling_request_ids:
            request = next(
                row for row in self.store.profile_requests()
                if row['request_id'] == identifier
            )
            if not admitted.intersection(request['trial_ids']):
                raise ValueError('GPU组件测量未关联本轮Validation候选')
            validation_failed = False
            if request['measurement_type'] == 'native_training_cost_profile':
                for trial in current_candidates():
                    if (trial['trial_id'] not in request['trial_ids']
                            or not same_training_plan(
                                trial['plan']['backend_config'],
                                request['execution_request'])
                            or trial['prediction']['memory'].get(
                                'admission') == 'WITHIN_ESTIMATED_CAPACITY'):
                        continue
                    result = self.execute_load_validation(trial)
                    if result['status'] != 'PASS':
                        self.store.update_profile_request(
                            identifier,
                            state='DEFERRED' if result['status'] == 'OUT_OF_BUDGET'
                            else 'FAILED',
                            reason='完整配置训练步验证状态：' + result['status'],
                        )
                        validation_failed = True
                        break
            if validation_failed:
                continue
            result = self.execute_profile(identifier)
            if result['state'] == 'COMPLETED':
                profiles.append(identifier)

        if profiles:
            self.synchronize_cost_revision()
            for row in current_candidates():
                if row['prediction'].get(
                        'cost_database_revision') != self.cost_revision:
                    if self.store.usage().get(
                            'cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                        break
                    self.predict_trial(row, force=True)
            selection, selection_id = self.invoke_with_tools(
                'ExperimentInvestigationAgent',
                '组件测量已经完成并更新候选。依据最新DAG预测和费用，'
                '重新决定本轮需要执行的完整训练对象。',
                selection_context() | dict(
                    completed_profile_request_ids=profiles,
                    previous_selection_evidence_id=selection_id,
                ), InvestigationSelection,
            )
            selection = self.resolve_selection_candidate_ids(
                selection, current_candidates())
            self.store.add_search_state('DAG_POST_PROFILE_SELECTION', dict(
                selection=selection.model_dump(mode='json'),
                evidence_id=selection_id, completed_profiles=profiles,
            ))

        measured = []
        for identifier in selection.accepted_candidate_ids:
            if identifier not in admitted:
                raise ValueError('完整训练候选未通过本轮Validation审查')
            if len(measured) >= self.config.budget.max_benchmarks_per_round:
                break
            trial = next(
                row for row in self.store.trials()
                if row['trial_id'] == identifier
            )
            if trial.get('measurement') or trial.get('failure'):
                continue
            prediction = trial.get('prediction') or {}
            memory = prediction.get('memory', {}).get('admission')
            if memory == 'REJECT_OOM':
                continue
            if (memory != 'WITHIN_ESTIMATED_CAPACITY'
                    or trial['plan']['backend_config'].get(
                        'execution_backend') == 'megatron_fsdp'):
                result = self.execute_load_validation(trial)
                if result['status'] != 'PASS':
                    continue
                trial = next(
                    row for row in self.store.trials()
                    if row['trial_id'] == identifier
                )
            result = self.execute_trial(
                trial,
                exploration=identifier in selection.exploration_candidate_ids
                or prediction.get('performance', {}).get('step_time_ms') is None,
            )
            if result is not None:
                measured.append(result)

        self._planning_last_measured_feedback = candidate_decisions(
            parent, measured)
        self.store.add_search_state(
            'PLANNING_TRAINING_FEEDBACK',
            self._planning_last_measured_feedback,
        )
        return measured
