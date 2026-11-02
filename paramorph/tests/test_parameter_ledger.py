import json
from pathlib import Path
import unittest

from cost.runtime_calibration import configuration
from graph.model_graph import read_model
from predictor.parameter_metadata import candidate_rank_ownership, checkpoint_parameter_counts
from predictor.native_parameter_layout import candidate_buffer_layouts
from predictor.state_sharding import parameter_state_ownership
from region_partition.region_partitioner import RegionPartitioner


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / 'outputs/heterogeneous-context-agent-search/model-configs'
RUN = ROOT / ('outputs/heterogeneous-context-agent-search/execution/runs/'
    'agent-l30-s4096-tp4-pp4-ep2-numerical_smoke-20261012-b0/artifacts')


class ParameterLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, graph = read_model(CONFIGS / 'nemotron-lightning30.json',
            CONFIGS / 'lightning30-checkpoint-shapes.json')
        cls.ledger = graph['parameter_metadata']
        cls.request = json.loads((RUN / 'request.json').read_text())
        cls.request.update(precision='bfloat16', main_grads_dtype='float32')

    def owners(self, request):
        config = configuration(request)
        regions = RegionPartitioner(self.model, config).from_counts(request['decoder_chunks'])
        return config, regions, candidate_rank_ownership(self.ledger, config, regions, request)

    def test_historical_parameters_match_all_actual_ranks(self):
        config, regions, _ = self.owners(self.request)
        counts = checkpoint_parameter_counts(self.ledger, config, regions, self.request)
        for rank in range(config.world_size):
            manifest = json.loads((RUN / f'rank{rank}.manifest.json').read_text())
            self.assertEqual(counts['dense'][rank] + counts['expert'][rank],
                manifest['parameter_elements'], f'rank={rank}')

    def test_expert_ids_follow_candidate_ep_groups(self):
        request = dict(self.request, tp=2, pp=2, dp=4, ep=4, etp=2,
            expert_dp=1, vpp=1, decoder_chunks=[26, 26])
        _, _, owners = self.owners(request)
        for rank in range(8):
            ids = {row['expert_id'] for row in owners[rank] if row['layer_id'] == 1
                and row['family'] == 'expert'}
            self.assertEqual(ids, set(range((rank // 2) * 32, (rank // 2 + 1) * 32)))
        config = configuration(request)
        regions = RegionPartitioner(self.model, config).from_counts(request['decoder_chunks'])
        counts = checkpoint_parameter_counts(self.ledger, config, regions, request)
        for rank, rows in owners.items():
            for family in ('dense', 'expert'):
                self.assertEqual(counts[family][rank], sum(row['local_elements']
                    for row in rows if row['family'] == family))

    def test_mamba_packed_slices_cover_full_tensor(self):
        _, _, owners = self.owners(self.request)
        name = 'backbone.layers.0.mixer.in_proj.weight'
        spans = [part for rank in range(4) for row in owners[rank] if row['name'] == name
            for part in row['global_slices']]
        cursor = 0
        for span in sorted(spans, key=lambda row: row['global_start']):
            self.assertEqual(span['global_start'], cursor)
            cursor = span['global_end']
        self.assertEqual(cursor, 10304)

    def test_router_bias_is_buffer_and_mtp_embedding_is_physical_copy(self):
        _, _, owners = self.owners(self.request)
        self.assertTrue(all(not row['requires_grad'] for row in self.ledger['tensors']
            if row['name'].endswith('e_score_correction_bias')))
        embeddings = [row for rank in (0, 12) for row in owners[rank]
            if row['component'] == 'embedding']
        self.assertEqual(len(embeddings), 2)
        self.assertEqual({row['shared_weight_id'] for row in embeddings}, {'word_embedding'})
        self.assertEqual({row['physical_copy_id'] for row in embeddings},
            {'input_embedding', 'mtp_embedding'})

    def test_native_buffer_layout_matches_all_observed_parameters_and_shards(self):
        self.assert_native_runtime('lightning-parameter-ledger-tp4-pp4-ep2-20261012')

    def test_native_buffer_layout_matches_changed_tp_pp_ep_dp(self):
        self.assert_native_runtime('lightning-parameter-ledger-tp2-pp2-ep4-20261012')

    def assert_native_runtime(self, run_id):
        validation = ROOT / 'outputs/heterogeneous-agent-dag/parameter-ledger-validation'
        model, graph = read_model(CONFIGS / 'nemotron-lightning30.json',
            validation / 'verified-checkpoint-shapes.json')
        artifacts = ROOT / 'outputs/heterogeneous-agent-dag/lightning-parameter-runtime/training/runs' / run_id / 'artifacts'
        request = json.loads((artifacts / 'request.json').read_text())
        config = configuration(request)
        regions = RegionPartitioner(model, config).from_counts(request['decoder_chunks'])
        layouts = candidate_buffer_layouts(graph['parameter_metadata'], config, regions, request)
        counts = checkpoint_parameter_counts(graph['parameter_metadata'], config, regions, request)
        states = parameter_state_ownership(counts, config, request)
        for rank in range(config.world_size):
            runtime = json.loads((artifacts / f'rank{rank}.parameter-ledger.json').read_text())
            self.assertEqual(layouts[rank], runtime['native_buffer_layouts'])
            for family in ('dense', 'expert'):
                buffers = [bucket for layout in layouts[rank] if layout['family'] == family
                    for bucket in layout['buckets']]
                self.assertEqual(states[rank]['components'][family]['persistent_gradient_bytes'],
                    sum(bucket['end'] - bucket['start'] for bucket in buffers) * 4)
                self.assertEqual(states[rank]['components'][family]['host_optimizer_bytes'],
                    sum(bucket['optimizer_owned_elements'] for bucket in buffers) * 12)


if __name__ == '__main__':
    unittest.main()
