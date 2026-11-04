import unittest

from cost.decomposition import CostDecompositionError, assess_reuse, atomic_link_key


class CostReusePathsTests(unittest.TestCase):
    def setUp(self):
        self.parent = dict(
            tp=2, pp=4, cp=1, dp=2, ep=1, etp=2, vpp=2,
            micro_batch=1, sequence=2048, precision='float16',
            rank_mapping=list(range(16)), node_count=2, gpus_per_node=8,
            decoder_chunks=[5] * 8, group=4, overlap_p2p=True,
            recompute='full',
        )

    def test_pp_dp_repartition_requires_physical_communication_evidence(self):
        candidate = dict(self.parent, pp=2, dp=4, decoder_chunks=[10] * 4)
        assessment = assess_reuse(self.parent, candidate)
        self.assertTrue(assessment.reuse_layer_costs)
        self.assertFalse(assessment.reuse_link_costs)
        self.assertTrue(assessment.rebuild_schedule)
        self.assertTrue(assessment.requires_new_profiling)
        self.assertIn('pp', assessment.changed_organization_fields)

    def test_vpp_partition_reuses_independent_costs_and_rebuilds_schedule(self):
        candidate = dict(self.parent, vpp=1, decoder_chunks=[10] * 4)
        assessment = assess_reuse(self.parent, candidate)
        self.assertTrue(assessment.reuse_layer_costs)
        self.assertTrue(assessment.reuse_link_costs)
        self.assertTrue(assessment.rebuild_schedule)
        self.assertFalse(assessment.requires_new_profiling)

    def test_rank_mapping_change_invalidates_link_reuse(self):
        candidate = dict(self.parent, rank_mapping=list(reversed(self.parent['rank_mapping'])))
        assessment = assess_reuse(self.parent, candidate)
        self.assertTrue(assessment.reuse_layer_costs)
        self.assertFalse(assessment.reuse_link_costs)
        self.assertTrue(assessment.requires_new_profiling)

    def test_expert_parallel_change_requires_new_compute_and_communication_costs(self):
        candidate = dict(self.parent, ep=2)
        assessment = assess_reuse(self.parent, candidate)
        self.assertFalse(assessment.reuse_layer_costs)
        self.assertFalse(assessment.reuse_link_costs)
        self.assertTrue(assessment.requires_new_profiling)

    def test_physical_communication_key_is_sensitive_to_endpoint_and_context(self):
        base = dict(source_stage=0, destination_stage=1, direction='fwd',
            shape=(2048, 5120), dtype='float16',
            source_physical_rank=0, destination_physical_rank=8,
            topology_signature='nvlink-plus-network',
            communication_backend='nccl',
            concurrent_execution_signature='overlap-p2p')
        identity = atomic_link_key(**base)
        self.assertEqual(identity, atomic_link_key(**base))
        for change in (
                dict(destination_physical_rank=1),
                dict(topology_signature='nvlink-only'),
                dict(concurrent_execution_signature='no-overlap')):
            changed = dict(base, **change)
            self.assertNotEqual(identity, atomic_link_key(**changed))

    def test_incomplete_physical_link_identity_is_rejected(self):
        with self.assertRaises(CostDecompositionError):
            atomic_link_key(0, 1, 'fwd', source_physical_rank=0)
        with self.assertRaises(CostDecompositionError):
            atomic_link_key(0, 1, 'fwd', source_physical_rank=0, destination_physical_rank=8)


if __name__ == '__main__':
    unittest.main()
