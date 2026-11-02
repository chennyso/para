import unittest

from search.system_search import (AgentRewriteIntent, EvidenceGuidedSearch,
    InvestigationSelection, RewriteValues)


class SearchAlgorithmTests(unittest.TestCase):
    def test_selection_resolves_unique_truncated_candidate_id(self):
        review = InvestigationSelection(
            accepted_candidate_ids=['candidate-123'],
            load_validation_candidate_ids=['candidate-123'],
            exploration_candidate_ids=['candidate-123'],
            evidence_refs=['selection'], reason='Agent选择', missing_measurements=[])
        candidates = [dict(trial_id='candidate-1234'), dict(trial_id='candidate-5678')]
        resolved = EvidenceGuidedSearch.resolve_selection_candidate_ids(review, candidates)
        self.assertEqual(resolved.accepted_candidate_ids, ['candidate-1234'])
        self.assertEqual(resolved.load_validation_candidate_ids, ['candidate-1234'])
        self.assertEqual(resolved.exploration_candidate_ids, ['candidate-1234'])

    def test_unranked_candidates_preserve_one_real_exploration(self):
        review = InvestigationSelection(accepted_candidate_ids=['candidate-b'],
            evidence_refs=['selection'], reason='Agent选择', missing_measurements=['MTP'])
        candidates = [
            dict(trial_id='candidate-a', plan_hash='a', measurement=None,
                plan=dict(backend_config=dict(execution_backend='megatron_ddp')),
                prediction=dict(performance=dict(step_time_ms=None),
                    memory=dict(admission='WITHIN_ESTIMATED_CAPACITY'))),
            dict(trial_id='candidate-b', plan_hash='b', measurement=None,
                plan=dict(backend_config=dict(execution_backend='megatron_ddp')),
                prediction=dict(performance=dict(step_time_ms=None),
                    memory=dict(admission='REQUIRE_MEMORY_PROFILE')))]
        preserved = EvidenceGuidedSearch.preserve_unranked_exploration(
            review, candidates, enabled=True)
        self.assertEqual(preserved.accepted_candidate_ids[0], 'candidate-b')
        self.assertEqual(preserved.exploration_candidate_ids, ['candidate-b'])

    def test_unranked_fsdp_exploration_requires_load_validation(self):
        review = InvestigationSelection(accepted_candidate_ids=[], evidence_refs=['selection'],
            reason='Agent未选择', missing_measurements=['FSDP'])
        candidates = [dict(trial_id='candidate-fsdp', plan_hash='a', measurement=None,
            plan=dict(backend_config=dict(execution_backend='megatron_fsdp')),
            prediction=dict(performance=dict(step_time_ms=None),
                memory=dict(admission='REQUIRE_MEMORY_PROFILE')))]
        preserved = EvidenceGuidedSearch.preserve_unranked_exploration(
            review, candidates, enabled=True)
        self.assertEqual(preserved.exploration_candidate_ids, ['candidate-fsdp'])
        self.assertEqual(preserved.load_validation_candidate_ids, ['candidate-fsdp'])

    def test_unranked_exploration_rejects_predicted_oom(self):
        review = InvestigationSelection(accepted_candidate_ids=[], evidence_refs=['selection'],
            reason='Agent未选择', missing_measurements=['memory'])
        candidates = [dict(trial_id='candidate-oom', plan_hash='a', measurement=None,
            plan=dict(backend_config=dict(execution_backend='megatron_ddp')),
            prediction=dict(performance=dict(step_time_ms=None),
                memory=dict(admission='REJECT_OOM')))]
        preserved = EvidenceGuidedSearch.preserve_unranked_exploration(
            review, candidates, enabled=True)
        self.assertEqual(preserved.accepted_candidate_ids, [])
        self.assertEqual(preserved.exploration_candidate_ids, [])

    def test_ranking_sensitivity_distinguishes_best_order_and_membership(self):
        changes = EvidenceGuidedSearch.ranking_change_summary(
            ['plan-a', 'plan-b', 'plan-c'], ['plan-b', 'plan-a', 'plan-d'])
        self.assertTrue(changes['best_candidate_changed'])
        self.assertTrue(changes['top_k_order_changed'])
        self.assertEqual(changes['top_k_order_inversions'], 1)
        self.assertTrue(changes['top_k_set_changed'])

    def test_counterfactual_combination_rejects_conflicting_targets(self):
        first = AgentRewriteIntent(operation='SetParallelism', values=RewriteValues(pp=2),
            adjustable=['dp'], preserve=[], hypothesis='检查PP变换', evidence_refs=['evidence'])
        second = AgentRewriteIntent(operation='SetParallelism', values=RewriteValues(pp=4),
            adjustable=['dp'], preserve=[], hypothesis='检查PP变换', evidence_refs=['evidence'])
        combined, conflict = EvidenceGuidedSearch.combine_counterfactual_intents(first, second)
        self.assertIsNone(combined)
        self.assertEqual(conflict['status'], 'CONSTRAINT_COUPLED')
        self.assertEqual(conflict['conflicts'], ['pp'])

    def test_unranked_profile_prefers_current_backend_and_ignores_failed_requests(self):
        candidates = [dict(trial_id='ddp-candidate'), dict(trial_id='fsdp-candidate')]
        reference = dict(plan=dict(backend_config=dict(execution_backend='megatron_ddp',
            tp=2, pp=4, cp=1, ep=1, etp=2, dp=1, vpp=2, micro_batch=1,
            recompute='full', pipeline_schedule='interleaved_1f1b',
            optimizer_sharding='distributed')))
        base = dict(state='PENDING', priority=1, estimated_gpu_hours=0.5,
            measurement_type='native_training_cost_profile',
            cost_keys=['MTP', 'NCCL'])
        requests = [
            base | dict(request_id='failed-ddp', state='FAILED', trial_ids=['ddp-candidate'],
                execution_request=reference['plan']['backend_config']),
            base | dict(request_id='fsdp', trial_ids=['fsdp-candidate'], execution_request=dict(
                reference['plan']['backend_config'], execution_backend='megatron_fsdp',
                dp=4, dp_inner_size=4)),
            base | dict(request_id='ddp', trial_ids=['ddp-candidate'],
                execution_request=reference['plan']['backend_config'])]
        selected = EvidenceGuidedSearch.select_unranked_profile(requests, candidates, reference)
        self.assertEqual(selected['request_id'], 'ddp')

    def test_unranked_profile_rejects_fsdp_without_effective_sharding_group(self):
        reference = dict(plan=dict(backend_config=dict(execution_backend='megatron_fsdp',
            tp=2, pp=4, cp=1, ep=1, etp=2, dp=1, vpp=2, micro_batch=1,
            recompute='full', pipeline_schedule='interleaved_1f1b',
            optimizer_sharding='distributed')))
        requests = [dict(request_id='fsdp', state='PENDING', trial_ids=['candidate'],
            cost_keys=['FSDP'], priority=1, estimated_gpu_hours=0.5,
            execution_request=reference['plan']['backend_config'] | dict(dp_inner_size=1))]
        selected = EvidenceGuidedSearch.select_unranked_profile(
            requests, [dict(trial_id='candidate')], reference)
        self.assertIsNone(selected)

    def test_unranked_profile_preserves_current_optimizer_ownership(self):
        reference_request = dict(execution_backend='megatron_ddp', optimizer_sharding='distributed',
            tp=2, pp=4, cp=1, ep=1, etp=2, dp=1, vpp=2, micro_batch=1,
            recompute='full', pipeline_schedule='interleaved_1f1b')
        reference = dict(plan=dict(backend_config=reference_request))
        base = dict(state='PENDING', priority=1, estimated_gpu_hours=0.5, cost_keys=['MTP'],
            measurement_type='native_training_cost_profile')
        requests = [
            base | dict(request_id='replicated', trial_ids=['replicated-candidate'],
                execution_request=reference_request | dict(optimizer_sharding='replicated')),
            base | dict(request_id='distributed', trial_ids=['distributed-candidate'],
                execution_request=reference_request | dict(vpp=1))]
        candidates = [dict(trial_id='replicated-candidate'), dict(trial_id='distributed-candidate')]
        selected = EvidenceGuidedSearch.select_unranked_profile(requests, candidates, reference)
        self.assertEqual(selected['request_id'], 'distributed')

    def test_family_id_expands_to_complete_global_family(self):
        parent = dict(tp=2, pp=4, cp=1, dp=1, ep=1, etp=2, expert_dp=1,
            rank_mapping=list(range(8)), execution_backend='megatron_ddp',
            megatron_fsdp_version=0, optimizer_sharding='distributed',
            precision_aware_optimizer=False, dense_sharding_strategy='no_shard',
            expert_sharding_strategy='no_shard', dp_inner_size=1, dp_outer_size=1,
            expert_dp_inner_size=1, expert_dp_outer_size=1,
            outer_sharding_strategy='no_shard', expert_outer_sharding_strategy='no_shard')
        family_values = parent | dict(tp=1, pp=4, dp=2, etp=1, expert_dp=2,
            dp_inner_size=2, expert_dp_inner_size=2)
        intent = AgentRewriteIntent(operation='JointTransform', values=RewriteValues(vpp=2),
            adjustable=['vpp'], preserve=['recompute'], hypothesis='选择全局家族并修改VPP',
            evidence_refs=['evidence'], family_id='family-a')
        resolved = EvidenceGuidedSearch.apply_family_to_intent(
            intent, parent, [dict(family_id='family-a', values=family_values)])
        self.assertEqual(resolved.values['tp'], 1)
        self.assertEqual(resolved.values['dp'], 2)
        self.assertEqual(resolved.values['vpp'], 2)

    def test_family_id_rejects_conflicting_global_value(self):
        parent = dict(tp=2)
        intent = AgentRewriteIntent(operation='JointTransform', values=RewriteValues(tp=2),
            adjustable=[], preserve=[], hypothesis='冲突家族变量', evidence_refs=['evidence'],
            family_id='family-a')
        with self.assertRaisesRegex(ValueError, 'tp'):
            EvidenceGuidedSearch.apply_family_to_intent(
                intent, parent, [dict(family_id='family-a', values=dict(tp=1))])

    def test_family_id_respects_preserved_optimizer_precision(self):
        intent = AgentRewriteIntent(operation='JointTransform', values=RewriteValues(vpp=2),
            adjustable=['vpp'], preserve=['precision_aware_optimizer'],
            hypothesis='保持优化器精度', evidence_refs=['evidence'], family_id='family-a')
        family = dict(family_id='family-a', values=dict(precision_aware_optimizer=True))
        with self.assertRaisesRegex(ValueError, 'precision_aware_optimizer'):
            EvidenceGuidedSearch.apply_family_to_intent(
                intent, dict(precision_aware_optimizer=False), [family])

    def test_agent_empty_allowed_ranges_mean_unspecified(self):
        intent = AgentRewriteIntent(operation='JointTransform', values=RewriteValues(vpp=2),
            adjustable=['vpp'], preserve=[], hypothesis='空范围规范化', evidence_refs=['evidence'],
            allowed_ranges=dict(vpp=[], overlap_p2p=[]))
        executable = intent.executable()
        self.assertEqual(executable.allowed_ranges, {})


if __name__ == '__main__':
    unittest.main()
