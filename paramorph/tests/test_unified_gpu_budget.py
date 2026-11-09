import unittest

from contracts import SearchBudget
from search.gpu_budget_policy import unified_gpu_budget
from storage.experiment_store import ExperimentStore


class UnifiedGPUBudgetTests(unittest.TestCase):
    @staticmethod
    def budget():
        return SearchBudget(
            max_agent_calls=20,
            max_agent_rounds=12,
            max_cpu_predictions=200,
            max_profiling_gpu_hours=0.65,
            max_full_training_trials=16,
            max_total_gpu_hours=6.0,
            max_walltime_seconds=7200.0,
            max_no_improvement_rounds=6,
            estimated_trial_seconds=180.0,
        )

    def test_calibration_can_start_after_real_short_validation_cost(self):
        original = self.budget()
        unified = unified_gpu_budget(original)
        self.assertEqual(original.max_profiling_gpu_hours, 0.65)
        self.assertEqual(unified.max_profiling_gpu_hours, 6.0)
        self.assertEqual(unified.max_total_gpu_hours, 6.0)
        self.assertEqual(unified_gpu_budget(unified), unified)

        store = ExperimentStore(':memory:')
        short_hours = 46 * 8 / 3600
        recorded_before_validation = 0.5983 - short_hours
        store.charge('earlier:gpu', 'gpu_hours', recorded_before_validation)
        store.charge('earlier:profile', 'profiling_gpu_hours', recorded_before_validation)

        self.assertTrue(store.reserve_gpu_execution(
            'short-validation', 'short-validation-run', 'validation',
            short_hours, unified.max_total_gpu_hours,
            unified.max_profiling_gpu_hours,
        ))
        store.record_gpu_receipt(
            'short-validation', 'short-validation-run', 'validation', 8,
            dict(run_id='short-validation-run', gpu_allocation_seconds=46),
        )
        self.assertAlmostEqual(store.usage()['gpu_hours'], 0.5983)
        self.assertAlmostEqual(store.usage()['profiling_gpu_hours'], 0.5983)
        self.assertEqual(
            store.connection.execute(
                'SELECT COUNT(*) FROM gpu_execution_reservations WHERE active=1',
            ).fetchone()[0],
            0,
        )

        self.assertFalse(store.reserve_gpu_execution(
            'old-cap-calibration', 'old-cap-run', 'profiling', 0.2,
            original.max_total_gpu_hours, original.max_profiling_gpu_hours,
        ))
        self.assertTrue(store.reserve_gpu_execution(
            'calibration', 'calibration-run', 'profiling', 0.2,
            unified.max_total_gpu_hours, unified.max_profiling_gpu_hours,
        ))
        store.record_gpu_receipt(
            'calibration', 'calibration-run', 'profiling', 8,
            dict(run_id='calibration-run', gpu_allocation_seconds=90),
        )
        self.assertAlmostEqual(store.usage()['gpu_hours'], 0.7983)
        self.assertAlmostEqual(store.usage()['profiling_gpu_hours'], 0.7983)
        self.assertEqual(
            store.connection.execute(
                'SELECT COUNT(*) FROM gpu_execution_reservations WHERE active=1',
            ).fetchone()[0],
            0,
        )
        store.connection.close()

    def test_real_ledger_prevents_exceeding_shared_total_and_releases_reservations(self):
        unified = unified_gpu_budget(self.budget())
        store = ExperimentStore(':memory:')
        store.charge('earlier:gpu', 'gpu_hours', 0.5983)
        store.charge('earlier:profile', 'profiling_gpu_hours', 0.5983)

        self.assertTrue(store.reserve_gpu_execution(
            'long-profile', 'long-profile-run', 'profiling', 5.3,
            unified.max_total_gpu_hours, unified.max_profiling_gpu_hours,
        ))
        self.assertFalse(store.reserve_gpu_execution(
            'extra-profile', 'extra-profile-run', 'profiling', 0.2,
            unified.max_total_gpu_hours, unified.max_profiling_gpu_hours,
        ))
        store.release_gpu_execution('long-profile')
        self.assertTrue(store.reserve_gpu_execution(
            'extra-profile', 'extra-profile-run', 'profiling', 0.2,
            unified.max_total_gpu_hours, unified.max_profiling_gpu_hours,
        ))
        self.assertEqual(
            store.connection.execute(
                'SELECT COUNT(*) FROM gpu_execution_reservations WHERE active=1',
            ).fetchone()[0],
            1,
        )
        store.connection.close()


if __name__ == '__main__':
    unittest.main()
