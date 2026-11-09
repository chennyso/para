from predictor.profile_database import backend_reuse_evidence, same_training_plan
from search.dag_agent_control import DAGAgentSearch
from search.native_calibration_admission import admit_exact_native_calibration
from search.system_search import profile_hardware_matches


class CalibratedDAGAgentSearch(DAGAgentSearch):
    def predict_request(self, request, candidate_graph=None, parent_request=None):
        report = super().predict_request(request, candidate_graph, parent_request)
        if report['prediction_status'] != 'NEEDS_PROFILE':
            return report
        if (not (report.get('execution_graph') or {}).get('tasks')
                or report.get('performance', {}).get('raw_resource_aware_step_time_ms') is None):
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
            observed = sum(len(rows) for rows in windows.values())
            matching.append((
                backend['status'] == 'MATCHED',
                not profile['request'].get('paramorph_trace', False),
                observed, profile_id, profile,
            ))
        if not matching:
            return report
        _, _, _, profile_id, profile = max(matching, key=lambda item: item[:-1])
        return admit_exact_native_calibration(report, request, profile_id, profile)

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
                raise ValueError('复合Region重叠成本缺失时必须登记原生校准请求')
            outstanding.append(dict(
                trial_id=trial['trial_id'],
                native_profile_request_id=native['request_id'],
                missing_composite_overlap_count=sum(
                    row.get('condition') is None for row in unresolved),
            ))
        if outstanding:
            self.record_dag_event('native_composite_overlap_measurement', dict(
                action_id=receipt['action_id'],
                unresolved_candidates=outstanding,
                source='actual_unresolved_composite_dag_overlap_conditions',
            ))
        return receipt
