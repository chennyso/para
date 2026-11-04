import unittest

from predictor.graph_delta import graph_delta


class GraphDeltaEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.parent = dict(
            tp=2, pp=4, cp=1, dp=2, ep=1, etp=2, expert_dp=2,
            node_count=2, gpus_per_node=8, rank_mapping=list(range(16)),
            vpp=2, micro_batch=1, global_batch=16, group=4,
            decoder_chunks=[5] * 8, sequence=2048, precision='float16',
            recompute='full', overlap_p2p=True,
        )

    def test_cross_parallel_configuration_preserves_compute_only_when_compatible(self):
        candidate = dict(self.parent, pp=2, dp=4, decoder_chunks=[10] * 4)
        report = graph_delta(self.parent, candidate)
        self.assertTrue(report['reuse_layer_costs'])
        self.assertFalse(report['reuse_link_costs'])
        self.assertFalse(report['reuse_schedule'])
        self.assertTrue(report['requires_new_profiling'])
        self.assertEqual(report['changed_execution_fields'],
            sorted(report['changed_execution_fields']))
        self.assertEqual(report['reuse_assessment']['changed_organization_fields'],
            ['decoder_chunks', 'dp', 'pp'])

    def test_vpp_change_rebuilds_schedule_with_atomic_costs(self):
        candidate = dict(self.parent, vpp=1, decoder_chunks=[10] * 4)
        report = graph_delta(self.parent, candidate)
        self.assertTrue(report['reuse_layer_costs'])
        self.assertTrue(report['reuse_link_costs'])
        self.assertFalse(report['reuse_schedule'])
        self.assertFalse(report['requires_new_profiling'])

    def test_module_boundary_change_preserves_costs_but_rebuilds_schedule(self):
        candidate = dict(self.parent, decoder_chunks=[4, 6, 5, 5, 5, 5, 5, 5])
        report = graph_delta(self.parent, candidate)
        self.assertTrue(report['reuse_layer_costs'])
        self.assertTrue(report['reuse_link_costs'])
        self.assertFalse(report['reuse_schedule'])
        self.assertEqual(report['changed_regions'], [0, 1])
        self.assertEqual(report['changed_execution_fields'], ['decoder_chunks'])

    def test_expert_parallel_change_invalidates_compute_and_link_evidence(self):
        candidate = dict(self.parent, ep=2)
        report = graph_delta(self.parent, candidate)
        self.assertFalse(report['reuse_layer_costs'])
        self.assertFalse(report['reuse_link_costs'])
        self.assertTrue(report['requires_new_profiling'])


if __name__ == '__main__':
    unittest.main()
