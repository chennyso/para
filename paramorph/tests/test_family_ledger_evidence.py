import unittest

from search.family_state import FamilyLedger
from storage.experiment_store import ExperimentStore


class FamilyLedgerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore(':memory:')
        self.ledger = FamilyLedger(self.store)
        self.request = dict(
            tp=2, pp=4, cp=1, dp=2, ep=1, etp=2, expert_dp=2,
            rank_mapping=list(range(16)), node_count=2, gpus_per_node=8,
            vpp=2, micro_batch=1, global_batch=16, group=4,
            decoder_chunks=[5] * 8, recompute='full', overlap_p2p=True,
        )

    def tearDown(self):
        self.store.connection.close()

    def trial(self, identifier, *, step_ms, state='PROJECTED', measurement=None, failure=None,
              chunks=None, revision='cost-revision-1'):
        request = dict(self.request)
        if chunks is not None:
            request['decoder_chunks'] = chunks
        return dict(
            trial_id=identifier,
            plan=dict(backend_config=request),
            state=state,
            prediction=dict(
                prediction_id='prediction-' + identifier,
                prediction_status='PROJECTED',
                performance=dict(step_time_ms=step_ms),
                memory=dict(admission='WITHIN_ESTIMATED_CAPACITY'),
                cost_database_revision=revision,
            ),
            measurement=measurement,
            failure=failure,
        )

    def test_observed_oom_revokes_predicted_winner_without_discarding_evidence(self):
        candidate = self.trial('qwen-selective-recompute', step_ms=1900.0)
        self.ledger.record_trial(candidate, 'cost-revision-1')
        self.assertEqual(self.ledger.get(self.request)['best_predicted']['trial_id'],
            candidate['trial_id'])

        candidate['state'] = 'FAILED'
        candidate['measurement'] = dict(status='TRAIN_OOM', run_id='qwen-oom-rank1')
        candidate['failure'] = dict(
            status='TRAIN_OOM', witness_id='memory-failure-qwen-rank1',
            observed=dict(first_exception_rank=1, phase='forward_backward'))
        self.ledger.record_trial(candidate, 'cost-revision-1')

        family = self.ledger.get(self.request)
        self.assertIsNone(family['best_predicted'])
        self.assertIsNone(family['best_measured'])
        self.assertTrue(family['candidates'][candidate['trial_id']]['observed_oom'])
        self.assertEqual(family['candidates'][candidate['trial_id']]['failure_witness_id'],
            'memory-failure-qwen-rank1')
        observations = self.ledger.observations(self.request)
        self.assertEqual(len(observations), 2)
        self.assertEqual([record['state'] for record in observations],
            ['PROJECTED', 'FAILED'])
        self.ledger.record_trial(candidate, 'cost-revision-1')
        self.assertEqual(len(self.ledger.observations(self.request)), 2)

    def test_real_measurement_remains_winner_when_faster_projection_ooms(self):
        measured = self.trial(
            'qwen-vpp2-measured', step_ms=2600.0, state='MEASURED',
            measurement=dict(status='PASS', performance_eligible=True,
                run_id='qwen-vpp2-training', step_seconds=2.545, tokens_per_second=12870.44))
        oom = self.trial(
            'qwen-selective-oom', step_ms=1900.0, state='FAILED',
            chunks=[4, 6, 5, 5, 5, 5, 5, 5],
            measurement=dict(status='TRAIN_OOM', run_id='qwen-selective-training'),
            failure=dict(status='TRAIN_OOM', witness_id='memory-failure-qwen-selective'))
        self.ledger.record_trial(measured, 'cost-revision-1')
        self.ledger.record_trial(oom, 'cost-revision-1')

        family = self.ledger.get(self.request)
        self.assertEqual(family['best_predicted']['trial_id'], measured['trial_id'])
        self.assertEqual(family['best_measured']['trial_id'], measured['trial_id'])
        self.assertEqual(family['best_measured']['measured_tokens_per_s'], 12870.44)

    def test_infrastructure_error_does_not_prove_configuration_infeasible(self):
        candidate = self.trial(
            'qwen-api-timeout', step_ms=2500.0, state='FAILED',
            measurement=dict(status='API_TIMEOUT', run_id='qwen-api-timeout-run'))
        self.ledger.record_trial(candidate, 'cost-revision-1')
        entry = self.ledger.get(self.request)['best_predicted']
        self.assertEqual(entry['trial_id'], candidate['trial_id'])
        self.assertFalse(entry['excluded_from_prediction'])

    def test_ineligible_success_cannot_rank_as_valid_candidate(self):
        candidate = self.trial(
            'qwen-invalid-window', step_ms=1800.0, state='MEASURED',
            measurement=dict(status='PASS', performance_eligible=False,
                run_id='qwen-invalid-window-run'))
        self.ledger.record_trial(candidate, 'cost-revision-1')
        family = self.ledger.get(self.request)
        self.assertIsNone(family['best_predicted'])
        self.assertIsNone(family['best_measured'])

    def test_old_prediction_revision_is_not_compared_to_new_costs(self):
        candidate = self.trial('qwen-old-costs', step_ms=2500.0)
        self.ledger.record_trial(candidate, 'cost-revision-1')
        self.ledger.invalidate_costs('cost-revision-2')
        family = self.ledger.get(self.request)
        self.assertIsNone(family['best_predicted'])
        self.assertEqual(len(self.ledger.observations(self.request)), 1)


if __name__ == '__main__':
    unittest.main()
