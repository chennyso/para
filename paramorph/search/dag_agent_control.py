import json
from collections import defaultdict

from pydantic import Field

from contracts import Contract, fingerprint
from predictor.profile_database import same_training_plan
from search.execution_decisions import candidate_decisions
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
                or self.config.planning_portfolio_enabled
                or self.config.planning_partition_engine != 'path'
                or not self.config.agent_execution_enabled
                or self.config.evaluation_mode not in ('full', 'multi_agent')):
            raise ValueError(
                'DAG-Agent需要直接Agent调查、原生Planning、路径CP-SAT和正式Agent执行')
        if self.config.single_decision_round:
            raise ValueError('DAG-Agent搜索需要允许独立的分析、策略及评价决策')

    def record_dag_event(self, kind, document):
        event = dict(kind=kind, **document)
        identifier = fingerprint(event)
        self.store.add_evidence(identifier, 'PREDICTED', event)
        return identifier

    def dag_task_guidance(self, parent):
        prediction = parent.get('prediction') or {}
        graph, scope = self.bottleneck_graph(parent)
        cache = getattr(self, '_dag_guidance_cache', {})
        key = (
            parent['trial_id'], prediction.get('prediction_id'),
            self.cost_revision, graph.graph_id,
        )
        if key in cache:
            return cache[key]
        report = dict(
            kind='dag_task_impact_guidance',
            parent_trial_id=parent['trial_id'],
            parent_plan_hash=parent['plan_hash'],
            graph_id=graph.graph_id, graph_scope=scope,
            source_prediction_id=prediction.get('prediction_id'),
            cost_revision=self.cost_revision, impact_ratio=0.1,
            task_groups=[],
        )
        structural = self.backend_dag(parent)['tasks']
        if (scope == 'PREDICTED_COSTED_DAG'
                and prediction.get('prediction_status') != 'NEEDS_PROFILE'):
            resource_model = self.predictor.projector.parameters.get(
                'resource_replay', {})
            baseline = graph.replay(resource_model=resource_model)
            index = self.rewriter.compiler.decision_impact_index(
                self.rewriter.candidate(parent['plan']['backend_config']))
            groups = defaultdict(list)
            for task_id in baseline['critical_task_ids']:
                task = graph.records[task_id]
                if task['duration_ms'] > 0:
                    groups[(task.get('region'), task['kind'])].append(task_id)
            ordered = sorted(groups.items(), key=lambda item: (
                -sum(graph.records[task_id]['duration_ms'] for task_id in item[1]),
                str(item[0]),
            ))
            for (region, operation), task_ids in ordered[:6]:
                impact = graph.epsilon_impact(
                    task_ids, 0.1, resource_model=resource_model,
                    baseline_replay=baseline,
                )
                native_ids = sorted({
                    origin for task_id in task_ids
                    if (origin := graph.records[task_id].get(
                        'metadata', {}).get('origin_task', task_id)) in structural
                })
                variables = sorted(name for name, row in index.items()
                    if set(row['affected_task_ids']).intersection(native_ids)
                    or region is not None and region in row['affected_region_ids'])
                report['task_groups'].append(dict(
                    region_id=region, operation=operation,
                    task_ids=task_ids[:8], native_task_ids=native_ids[:8],
                    related_variables=variables,
                    task_count=len(task_ids), impact_ms=impact['impact_ms'],
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
            report['status'] = 'STRUCTURAL_ONLY'
            report['reason'] = '完整训练成本不足，允许根据真实任务依赖提出待验证假设'
            report['structural_task_ids'] = sorted(structural)[:16]
            report['required_costs'] = prediction.get('evidence', {}).get(
                'profiling_requests', [])[:4]
        identifier = fingerprint(report)
        self.store.add_evidence(identifier, 'PREDICTED', report)
        cache[key] = dict(report, evidence_id=identifier)
        self._dag_guidance_cache = cache
        return cache[key]

    def evidence_reference_feedback(self, role, response):
        if role != 'PerformanceAnalysisAgent' or not isinstance(
                response, BottleneckDecision):
            return super().evidence_reference_feedback(role, response)
        parent_id = getattr(self, '_dag_analysis_parent_id', None)
        parents = [row for row in self.store.trials()
            if row['trial_id'] == parent_id]
        if len(parents) != 1:
            raise ValueError('DAG分析需要唯一的父训练配置')
        parent = parents[0]
        graph, _ = self.bottleneck_graph(parent)
        valid = set(graph.records) | set(self.backend_dag(parent)['tasks'])
        known = {row[0] for row in self.store.connection.execute(
            'SELECT id FROM evidence')}
        unknown_tasks = sorted(set(response.affected_tasks) - valid)
        unknown_refs = sorted(set(response.evidence_refs) - known)
        if response.evidence_refs and not unknown_tasks and not unknown_refs:
            return None
        result = dict(
            status='REVISION_REQUIRED', revision_owner=role,
            parent_trial_id=parent_id,
            unknown_task_ids=unknown_tasks,
            unknown_evidence_refs=unknown_refs,
            valid_task_id_examples=sorted(valid)[:24],
            instruction='引用当前父配置结构DAG或预测DAG中的实际任务；'
                '成本未知时允许仅提交结构假设及所需测量。',
        )
        receipt = dict(
            tool=dict(tool='validate_dag_analysis_references'),
            result=result,
        )
        identifier = fingerprint(dict(response=response.model_dump(
            mode='json'), receipt=receipt))
        self.store.add_evidence(identifier, 'FAILURE', receipt)
        return dict(
            evidence_id=identifier, **receipt,
            result_tokens=token_count(result),
        )

    def planning_observation(self, parent):
        observation = super().planning_observation(parent)
        guidance = self.dag_task_guidance(parent)
        observation['dag_task_guidance'] = guidance
        observation['evidence_ids'] = list(dict.fromkeys([
            *observation['evidence_ids'], guidance['evidence_id'],
        ]))
        self._dag_analysis_parent_id = parent['trial_id']
        cache = getattr(self, '_dag_analysis_cache', {})
        key = (parent['trial_id'], guidance['evidence_id'])
        if key not in cache:
            analysis, identifier = self.invoke_with_tools(
                'PerformanceAnalysisAgent',
                '依据dag_task_guidance中的任务、完整迭代时间影响和原生依赖，'
                '自主选择需要调查的瓶颈。真实执行任务可使用task_ids；'
                '结构任务可使用native_task_ids。'
                '当性能成本缺失时仅提出结构假设并指出需要的测量。'
                '列出与任务直接相关的联合配置变量。',
                observation, BottleneckDecision,
            )
            cache[key] = dict(
                evidence_id=identifier,
                decision=analysis.model_dump(mode='json'),
            )
            self._dag_analysis_cache = cache
            self.record_dag_event('dag_analysis_decision', dict(
                parent_trial_id=parent['trial_id'],
                guidance_evidence_id=guidance['evidence_id'],
                analysis_evidence_id=identifier,
                decision=cache[key]['decision'],
            ))
        observation['analysis_agent'] = cache[key]
        observation['evidence_ids'].append(cache[key]['evidence_id'])
        observation['dag_strategy_guidance'] = (
            '使用Analysis选择的执行问题和dag_task_guidance的关联变量'
            '自主生成完整联合修改。保留原生可执行的Rank映射，'
            '没有数值成本时保留NEEDS_PROFILE并选择相容测量。'
            'evidence_refs引用已有证据标识；实际候选合法性由工具判定。'
        )
        return observation

    def execute_investigation(self, action):
        receipt = super().execute_investigation(action)
        evaluated = set(receipt['evaluated_candidate_ids'])
        unresolved, requested = [], []
        for trial in self.store.trials():
            if trial['trial_id'] not in evaluated:
                continue
            prediction = trial.get('prediction')
            if prediction is None:
                raise ValueError('Planning调查登记的候选缺少预测记录')
            if prediction['prediction_status'] == 'UNSUPPORTED':
                continue
            if prediction['prediction_status'] == 'NEEDS_PROFILE' or (
                    prediction.get('evidence', {}).get('profiling_requests')):
                self.request_profiles(trial, prediction)
                links = [row for row in self.store.profile_requests()
                    if trial['trial_id'] in row['trial_ids']
                    and row['state'] not in ('FAILED', 'CANCELLED')]
                if prediction['prediction_status'] == 'NEEDS_PROFILE' and not links:
                    self.request_profiles(trial, prediction, native_only=True)
                    links = [row for row in self.store.profile_requests()
                        if trial['trial_id'] in row['trial_ids']
                        and row['state'] not in ('FAILED', 'CANCELLED')]
                if prediction['prediction_status'] == 'NEEDS_PROFILE':
                    unresolved.append(trial['trial_id'])
                    if not links:
                        raise ValueError(
                            '合法候选缺少完整成本且没有可执行的Profile请求')
                requested.extend(row['request_id'] for row in links)
        self.record_dag_event('dag_candidate_cost_coverage', dict(
            action_id=receipt['action_id'],
            evaluated_candidate_ids=receipt['evaluated_candidate_ids'],
            retained_candidate_ids=receipt['candidate_ids'],
            unknown_cost_candidate_ids=sorted(unresolved),
            profile_request_ids=sorted(set(requested)),
        ))
        return receipt

    def constraint_decision_feedback(self, response, observation):
        if not isinstance(response, DAGCandidateReview):
            return super().constraint_decision_feedback(response, observation)
        allowed = set(observation['candidate_ids'])
        submitted = response.admitted_candidate_ids
        conflicts = sorted(set(submitted) - allowed)
        duplicates = len(set(submitted)) != len(submitted)
        if not conflicts and not duplicates:
            return None, []
        feedback = dict(
            status='REVISION_REQUIRED',
            revision_owner='ConstraintValidationAgent',
            invalid_candidate_ids=conflicts,
            duplicated_candidate_ids=duplicates,
            permitted_candidate_ids=sorted(allowed),
            instruction='只能选择已经通过真实原生约束检查的完整候选。',
        )
        identifier = self.record_dag_event(
            'dag_constraint_decision_feedback', feedback)
        return dict(
            evidence_id=identifier,
            tool=dict(tool='validate_dag_candidates'),
            result=feedback,
            result_tokens=token_count(feedback),
        ), []

    def execute_planning_measurements(self, parent, candidate_ids, action_ids):
        eligible = [row for row in self.store.trials()
            if row['trial_id'] in candidate_ids
            and row.get('prediction') is not None
            and row.get('measurement') is None
            and not row.get('failure')
            and row['prediction']['memory'].get('admission') != 'REJECT_OOM']
        if not eligible:
            self.record_dag_event('dag_no_eligible_candidates', dict(
                parent_trial_id=parent['trial_id'],
                planning_action_ids=action_ids,
                proposed_candidate_ids=sorted(candidate_ids),
            ))
            return []
        candidates = self.retain_candidate_portfolio(
            eligible, 8, parent=parent)
        prediction_refs = {
            row['trial_id']: row['prediction']['planning_evidence_id']
            for row in candidates
        }
        review_context = self.shared_context(parent) | dict(
            parent_trial_id=parent['trial_id'],
            candidate_ids=[row['trial_id'] for row in candidates],
            candidate_prediction_evidence=prediction_refs,
            candidates=self.investigation_candidate_table(candidates),
            planning_action_evidence_refs=action_ids,
        )
        review, review_id = self.invoke_with_tools(
            'ConstraintValidationAgent',
            '原生约束求解器已生成合法候选。依据真实候选预测和容量状态'
            '选择需要继续交给Evaluation Agent的完整策略，'
            '允许NEEDS_PROFILE候选进入组件成本测量。'
            '每项admitted_candidate_ids必须引用candidate_ids，'
            'evidence_refs引用candidate_prediction_evidence中的实际标识。',
            review_context, DAGCandidateReview,
        )
        admitted = set(review.admitted_candidate_ids)
        self.record_dag_event('dag_validation_decision', dict(
            parent_trial_id=parent['trial_id'],
            validation_evidence_id=review_id,
            admitted_candidate_ids=sorted(admitted),
            excluded_candidate_ids=sorted(
                set(prediction_refs) - admitted),
            reason=review.reason,
        ))

        def current_candidates():
            return [row for row in self.store.trials()
                if row['trial_id'] in admitted
                and row.get('measurement') is None
                and not row.get('failure')
                and (row.get('prediction') or {}).get(
                    'memory', {}).get('admission') != 'REJECT_OOM']

        def selection_context():
            current = current_candidates()
            pending = [dict(
                request_id=row['request_id'],
                trial_ids=sorted(admitted.intersection(row['trial_ids'])),
                cost_keys=row['cost_keys'][:4],
                measurement_type=row['measurement_type'],
                estimated_gpu_hours=row['estimated_gpu_hours'],
                state=row['state'],
            ) for row in self.store.profile_requests()
                if row['state'] in (
                    'PENDING', 'APPROVED', 'DEFERRED', 'OUT_OF_BUDGET')
                and admitted.intersection(row['trial_ids'])]
            return self.shared_context(parent) | dict(
                parent_trial_id=parent['trial_id'],
                candidates=self.investigation_candidate_table(current),
                candidate_decision_report=candidate_decisions(parent, current),
                planning_action_evidence_refs=action_ids,
                validation_evidence_id=review_id,
                profiling_requests=pending,
            )

        selection, selection_id = self.invoke_with_tools(
            'ExperimentInvestigationAgent',
            '根据真实任务影响、候选性能、显存状态和缺失成本决定'
            '组件校准、单步验证及正式训练。'
            '缺少完整时间的合法候选仍可选择补测；'
            '正式训练必须满足容量验证和实验权限。',
            selection_context(), InvestigationSelection,
        )
        selection = self.resolve_selection_candidate_ids(
            selection, current_candidates())
        self.record_dag_event('dag_measurement_decision', dict(
            selection_evidence_id=selection_id,
            validation_evidence_id=review_id,
            selection=selection.model_dump(mode='json'),
        ))

        for identifier in selection.load_validation_candidate_ids:
            if identifier not in admitted:
                raise ValueError('短程验证引用了未通过原生校验的候选')
            trial = next(row for row in self.store.trials()
                if row['trial_id'] == identifier)
            self.execute_load_validation(trial)

        completed_profiles = []
        for identifier in selection.accepted_profiling_request_ids:
            request = next(row for row in self.store.profile_requests()
                if row['request_id'] == identifier)
            linked = admitted.intersection(request['trial_ids'])
            if not linked:
                raise ValueError('Profile请求没有关联本轮合法候选')
            if request['measurement_type'] == 'native_training_cost_profile':
                validation_failed = False
                for trial in current_candidates():
                    if trial['trial_id'] not in linked or not same_training_plan(
                            trial['plan']['backend_config'],
                            request['execution_request']):
                        continue
                    if trial['prediction']['memory'].get(
                            'admission') == 'WITHIN_ESTIMATED_CAPACITY':
                        continue
                    validation = self.execute_load_validation(trial)
                    if validation['status'] != 'PASS':
                        self.store.update_profile_request(identifier,
                            state='DEFERRED' if validation['status'] == 'OUT_OF_BUDGET'
                            else 'FAILED',
                            reason='短程显存验证状态：' + validation['status'])
                        validation_failed = True
                        break
                if validation_failed:
                    continue
            result = self.execute_profile(identifier)
            if result['state'] == 'COMPLETED':
                completed_profiles.append(identifier)

        if completed_profiles:
            self.synchronize_cost_revision()
            for trial in current_candidates():
                if (trial['prediction'].get('cost_database_revision')
                        == self.cost_revision):
                    continue
                if (self.store.usage().get('cpu_predictions', 0)
                        >= self.config.budget.max_cpu_predictions):
                    break
                self.predict_trial(trial, force=True)
            selection, selection_id = self.invoke_with_tools(
                'ExperimentInvestigationAgent',
                '组件成本测量已写入共享数据库。依据新的完整DAG预测、'
                '实际显存及GPU费用重新选择正式训练候选。',
                selection_context() | dict(
                    completed_profile_request_ids=completed_profiles,
                    previous_selection_evidence_id=selection_id,
                ), InvestigationSelection,
            )
            selection = self.resolve_selection_candidate_ids(
                selection, current_candidates())
            self.record_dag_event('dag_post_profile_selection', dict(
                selection_evidence_id=selection_id,
                completed_profile_request_ids=completed_profiles,
                selection=selection.model_dump(mode='json'),
            ))

        measured = []
        for identifier in selection.accepted_candidate_ids:
            if identifier not in admitted:
                raise ValueError('完整训练候选未通过原生约束校验')
            if len(measured) >= self.config.budget.max_benchmarks_per_round:
                break
            trial = next(row for row in self.store.trials()
                if row['trial_id'] == identifier)
            if trial.get('measurement') is not None or trial.get('failure'):
                continue
            prediction = trial.get('prediction') or {}
            admission = prediction.get('memory', {}).get('admission')
            if admission == 'REJECT_OOM':
                continue
            if (admission != 'WITHIN_ESTIMATED_CAPACITY'
                    or trial['plan']['backend_config'].get(
                        'execution_backend') == 'megatron_fsdp'):
                validation = self.execute_load_validation(trial)
                if validation['status'] != 'PASS':
                    continue
                trial = next(row for row in self.store.trials()
                    if row['trial_id'] == identifier)
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
