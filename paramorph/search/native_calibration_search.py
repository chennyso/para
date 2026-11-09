from contracts import fingerprint
from predictor.profile_database import backend_reuse_evidence, same_training_plan
from search.agent_decision_protocol import (
    EvidencedDAGReview, EvidencedExperimentDecision,
    experiment_conflicts, review_conflicts,
)
from search.communication_placement import placement_families
from search.dag_agent_control import DAGAgentSearch, DAGCandidateReview
from search.native_calibration_admission import admit_exact_native_calibration
from search.system_search import (
    InvestigationSelection, profile_hardware_matches, token_count,
)


class CalibratedDAGAgentSearch(DAGAgentSearch):
    def predict_request(self, request, candidate_graph=None, parent_request=None):
        report = super().predict_request(request, candidate_graph, parent_request)
        if report['prediction_status'] != 'NEEDS_PROFILE':
            return report
        if (not (report.get('execution_graph') or {}).get('tasks')
                or report.get('performance', {}).get(
                    'raw_resource_aware_step_time_ms') is None):
            return report
        matching = []
        for profile_id, profile in self.predictor.database.measured_profiles():
            if (not same_training_plan(profile['request'], request)
                    or not profile_hardware_matches(profile, self.topology.devices)):
                continue
            backend = backend_reuse_evidence(request, profile)
            if backend['status'] == 'MISMATCHED':
                continue
            windows = profile.get('measured_windows', {})
            if len(windows) != self.topology.num_gpus:
                continue
            count = sum(len(rows) for rows in windows.values())
            matching.append((
                backend['status'] == 'MATCHED',
                not profile['request'].get('paramorph_trace', False),
                count, profile_id, profile,
            ))
        if not matching:
            return report
        _, _, _, profile_id, profile = max(matching, key=lambda item: item[:-1])
        return admit_exact_native_calibration(report, request, profile_id, profile)

    def dag_task_guidance(self, parent):
        prediction = parent.get('prediction') or {}
        calibration = prediction.get('evidence', {}).get(
            'native_calibration_admission')
        if calibration is None:
            return super().dag_task_guidance(parent)
        graph, scope = self.bottleneck_graph(parent)
        report = dict(
            kind='dag_task_impact_guidance',
            parent_trial_id=parent['trial_id'],
            parent_plan_hash=parent['plan_hash'],
            graph_id=graph.graph_id, graph_scope=scope,
            source_prediction_id=prediction['prediction_id'],
            cost_revision=self.cost_revision,
            status='MEASURED_TOTAL_TIME_WITH_UNCALIBRATED_PATHS',
            task_groups=[],
            native_calibration_profile_id=calibration['profile_id'],
            critical_task_ids=prediction.get('diagnosis', {}).get(
                'critical_tasks', [])[:12],
            related_task_evidence_scope='structural_and_raw_dag_diagnostic',
            performance_effect_scope='not_calibrated_for_task_level_impact',
        )
        identifier = fingerprint(report)
        self.store.add_evidence(identifier, 'PREDICTED', report)
        return dict(report, evidence_id=identifier)

    def planning_observation(self, parent):
        observation = super().planning_observation(parent)
        parent_request = parent['plan']['backend_config']
        observation['native_placement_choices'] = [
            dict(axis_order=row['axis_order'],
                 rank_mapping=row['request']['rank_mapping'])
            for row in placement_families(parent_request)
        ]
        history = []
        for trial in self.store.trials()[-12:]:
            prediction = trial.get('prediction') or {}
            measurement = trial.get('measurement') or {}
            failure = trial.get('failure') or {}
            request = trial['plan']['backend_config']
            history.append(dict(
                trial_id=trial['trial_id'],
                parallelism={name: request.get(name) for name in (
                    'tp', 'pp', 'dp', 'cp', 'ep', 'etp', 'vpp')},
                decoder_chunks=request['decoder_chunks'],
                prediction_status=prediction.get('prediction_status'),
                predicted_step_ms=prediction.get('performance', {}).get(
                    'step_time_ms'),
                memory_admission=prediction.get('memory', {}).get('admission'),
                measured_tokens_per_second=measurement.get('tokens_per_second')
                    if measurement.get('performance_eligible') else None,
                failure_reason=failure.get('reason'),
                failure_retry_eligibility=failure.get('retry_eligibility'),
            ))
        observation['recent_complete_trial_ledger'] = history
        observation['recent_profile_failures'] = [
            dict(request_id=row['request_id'], state=row['state'],
                 reason=row['reason'], trial_ids=row['trial_ids'])
            for row in self.store.profile_requests()
            if row['state'] in ('FAILED', 'OUT_OF_BUDGET', 'DEFERRED')
        ][-8:]
        best = self.store.best_measured()
        if best is not None:
            observation['measured_incumbent'] = dict(
                trial_id=best['trial_id'],
                tokens_per_second=best['measurement']['tokens_per_second'],
                plan_hash=best['plan_hash'])
        return observation

    def agent_context(self, observation, role=None):
        context = super().agent_context(observation, role)
        if (role == 'ConstraintValidationAgent'
                and 'candidate_prediction_evidence' in observation):
            context['available_tools'] = []
            context['tool_usage'] = (
                '本轮原生编译与完整约束检查已经完成。'
                '逐项审查candidate_ids的原始预测、显存、成本缺失，'
                '返回candidate_verifications并使用真实证据编号。'
                '执行验证、组件测量和正式训练交给Evaluation Agent。')
        return context

    def invoke_with_tools(self, role, instructions, observation, schema):
        if role == 'ConstraintValidationAgent' and schema is DAGCandidateReview:
            schema = EvidencedDAGReview
            instructions += (
                '\n每项candidate_verifications都必须引用'
                'candidate_prediction_evidence对应的真实预测编号，'
                '准确复述预测和显存状态，并说明推荐补测、训练步验证或正式训练的原因。'
                'admitted_candidate_ids包含全部结构合法候选。')
        elif (role == 'ExperimentInvestigationAgent'
                and schema is InvestigationSelection):
            schema = EvidencedExperimentDecision
            instructions += (
                '\n选择的Profile必须存在于profiling_requests。'
                '存在合法候选、性能成本缺项和可执行Profile时，'
                '应选择实际测量。没有选择任何操作时，'
                '必须在stop_reason提供客观停止原因；不得重复空轮次。'
                '在Profile更新后重新判断正式训练对象。')
        return super().invoke_with_tools(
            role, instructions, observation, schema)

    def constraint_decision_feedback(self, response, observation):
        feedback, required = super().constraint_decision_feedback(
            response, observation)
        if feedback is not None or required or not isinstance(
                response, EvidencedDAGReview):
            return feedback, required
        conflicts = review_conflicts(
            response, observation, self.store.trials())
        if not conflicts:
            return None, []
        result = dict(
            status='REVISION_REQUIRED',
            revision_owner='ConstraintValidationAgent',
            conflicts=conflicts,
            instruction='依据原生候选的真实预测和显存记录逐项修订，'
                '不得宣称尚未完成的成本采集或容量验证已经通过。')
        tool = dict(tool='verify_dag_candidate_evidence')
        receipt = dict(tool=tool, result=result)
        identifier = fingerprint(dict(
            response=response.model_dump(mode='json'), receipt=receipt))
        self.store.add_evidence(identifier, 'FAILURE', receipt)
        return dict(
            evidence_id=identifier, tool=tool, result=result,
            result_tokens=token_count(result)), []

    def selection_admission_feedback(self, response, candidate_ids=()):
        feedback = super().selection_admission_feedback(
            response, candidate_ids)
        if feedback is not None or not isinstance(
                response, EvidencedExperimentDecision):
            return feedback
        observation = getattr(self, '_dag_evaluation_observation', None)
        if observation is None:
            raise ValueError('DAG评价缺少本轮完整候选及Profile目录')
        usage = self.store.usage()
        budget = self.config.budget
        remaining = budget.max_total_gpu_hours - usage.get('gpu_hours', 0)
        conflicts = experiment_conflicts(
            response, observation, self.store.trials(),
            gpu_enabled=self.config.gpu_execution_enabled,
            remaining_gpu_hours=remaining)
        if not conflicts:
            return None
        result = dict(
            status='REVISION_REQUIRED',
            revision_owner='ExperimentInvestigationAgent',
            conflicts=conflicts,
            remaining_gpu_hours=remaining,
            instruction='只选择真实存在并关联本轮候选的操作。'
                '成本缺失时优先选择相容的Profile；'
                '停止前必须给出能够由费用账本和候选状态验证的原因。')
        tool = dict(tool='verify_dag_experiment_decision')
        receipt = dict(tool=tool, result=result)
        identifier = fingerprint(dict(
            response=response.model_dump(mode='json'), receipt=receipt))
        self.store.add_evidence(identifier, 'FAILURE', receipt)
        return dict(
            evidence_id=identifier, tool=tool, result=result,
            result_tokens=token_count(result))

    def retain_candidate_portfolio(self, candidates, limit, *, parent=None,
                                   proposal_candidate_ids=()):
        prioritized = list(dict.fromkeys([
            *proposal_candidate_ids,
            *getattr(self, '_dag_current_proposal_ids', ()),
        ]))
        return super().retain_candidate_portfolio(
            candidates, limit, parent=parent,
            proposal_candidate_ids=prioritized)

    def execute_planning_measurements(self, parent, candidate_ids, action_ids):
        self._dag_current_proposal_ids = set(candidate_ids)
        selected = set(candidate_ids)
        unresolved_profiles = [
            row for row in self.store.profile_requests()
            if row['state'] in ('PENDING', 'APPROVED', 'DEFERRED',
                                'OUT_OF_BUDGET')]
        pending_trial_ids = {
            identifier for row in unresolved_profiles
            for identifier in row['trial_ids']
        }
        for trial in self.store.trials():
            if (trial['trial_id'] in pending_trial_ids
                    and not trial.get('failure')
                    and trial.get('measurement') is None
                    and (trial.get('prediction') or {}).get(
                        'memory', {}).get('admission') != 'REJECT_OOM'):
                selected.add(trial['trial_id'])
        return super().execute_planning_measurements(
            parent, selected, action_ids)

    def execute_investigation(self, action):
        receipt = super().execute_investigation(action)
        evaluated = set(receipt['evaluated_candidate_ids'])
        outstanding = []
        for trial in self.store.trials():
            if trial['trial_id'] not in evaluated:
                continue
            report = trial.get('prediction') or {}
            if report.get('prediction_status') != 'NEEDS_PROFILE':
                continue
            replay = report.get('diagnosis', {}).get(
                'resource_aware_replay', {})
            unresolved = replay.get('missing_overlap_profiles', [])
            if not any(row.get('condition') is None for row in unresolved):
                continue
            native = self.request_profiles(trial, report, native_only=True)
            if native is None:
                raise ValueError(
                    '复合Region重叠成本缺失时必须登记原生校准请求')
            outstanding.append(dict(
                trial_id=trial['trial_id'],
                native_profile_request_id=native['request_id'],
                missing_composite_overlap_count=sum(
                    row.get('condition') is None for row in unresolved),
            ))
        if outstanding:
            self.record_dag_event(
                'native_composite_overlap_measurement', dict(
                    action_id=receipt['action_id'],
                    unresolved_candidates=outstanding,
                    source='actual_unresolved_composite_dag_overlap_conditions'))
        return receipt
