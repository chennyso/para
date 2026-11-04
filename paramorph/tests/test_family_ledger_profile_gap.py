import unittest

from search.family_state import FamilyLedger
from storage.experiment_store import ExperimentStore


class FamilyLedgerProfileGapTests(unittest.TestCase):
    def test_missing_cost_does_not_become_a_predicted_winner(self):
        store = ExperimentStore(':memory:')
        ledger = FamilyLedger(store)
        request = dict(
            tp=2, pp=4, cp=1, dp=2, ep=1, etp=2, expert_dp=2,
            rank_mapping=list(range(16)), node_count=2, gpus_per_node=8,
            vpp=2, micro_batch=1, global_batch=16, group=4,
            decoder_chunks=[5] * 8, recompute='full', overlap_p2p=True,
        )
        trial = dict(
            trial_id='moonlight-pending-all-to-all-cost',
            plan=dict(backend_config=request),
            state='PROJECTED',
            prediction=dict(
                prediction_id='moonlight-unresolved-evaluation',
                prediction_status='NEEDS_PROFILE',
                cost_database_revision='cost-revision-1',
                performance=dict(step_time_ms=1000.0),
                memory=dict(admission='WITHIN_ESTIMATED_CAPACITY'),
                evidence=dict(profiling_requests=[
                    dict(cost_key='expert-all-to-all-topology-cost')]),
            ),
            measurement=None,
            failure=None,
        )
        ledger.record_trial(trial, 'cost-revision-1')
        family = ledger.get(request)
        self.assertIsNone(family['best_predicted'])
        self.assertEqual(family['candidates'][trial['trial_id']]['missing_cost_keys'],
            ['expert-all-to-all-topology-cost'])
        self.assertEqual(ledger.observations(request)[0]['prediction_status'], 'NEEDS_PROFILE')
        store.connection.close()


if __name__ == '__main__':
    unittest.main()
