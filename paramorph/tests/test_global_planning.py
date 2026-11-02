import unittest

from contracts import RewriteIntent
from core_data_models import HardwareTopology, LayerType, ModelSpec, ParallelConfig
from graph.capability_registry import CapabilityRegistry
from graph.complete_rewrite import CompleteAndRewrite
from graph.process_groups import process_groups
from region_partition.region_partitioner import RegionPartitioner
from predictor.ddp_buckets import bucket_layout_signature
from predictor.dag_projection import compute_profile_quality, exact_profile_anchor, optimizer_profile_key
from predictor.profile_database import normalized_execution
from search.seed_planner import SeedPlanner
from storage.experiment_store import ExperimentStore


class GlobalPlanningTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelSpec('constraint_fixture', 12, 64, 16,
            [LayerType.MOE] * 12, list(range(12)), num_experts=8, ffn_hidden_size=128)
        self.topology = HardwareTopology(8, 1, 8, 32)
        self.request = dict(run_id='parent', tp=2, cp=1, pp=2, dp=2, ep=1, vpp=1,
            group=2, sequence=128, micro_batch=1, global_batch=16, offload=False,
            recompute='full', decoder_chunks=[6, 6], node_count=1, gpus_per_node=8)
        self.rewriter = CompleteAndRewrite(self.model, self.topology, CapabilityRegistry({}),
            dict(tp=[1, 2, 4, 8], cp=[1, 2], pp=[1, 2, 4], dp=[1, 2, 4, 8],
                 ep=[1, 2, 4, 8], vpp=[1, 2], micro_batch=[1, 2], group=[1, 2, 4, 8, 16]))

    def intent(self, values, adjustable=(), preserve=()):
        return RewriteIntent(operation='JointTransform', values=values,
            adjustable=list(adjustable), preserve=list(preserve),
            hypothesis='验证确定性约束与执行结构', evidence_refs=['constraint_fixture'])

    def test_cp_closes_dense_dp_and_preserves_batch(self):
        rows = self.rewriter.complete_and_rewrite(self.request,
            self.intent(dict(cp=2), adjustable=['dp']), limit=1)
        request = rows[0]['request']
        self.assertEqual(request['dp'], 1)
        self.assertEqual(rows[0]['candidate']['config']['global_batch_size'], 16)
        self.assertIn('dp', rows[0]['completed_decisions'])

    def test_expert_folding_uses_independent_etp(self):
        request = dict(self.request, etp=1, ep=4, expert_dp=1)
        groups = process_groups(request)
        self.assertEqual(groups['expert_dp_size'], 1)
        self.assertTrue(all(len(row) == 4 for row in groups['logical']['ep']))
        self.assertEqual(len(groups['logical']['tp'][0]), 2)

    def test_preserved_dp_rejects_incompatible_cp(self):
        with self.assertRaises(ValueError):
            self.rewriter.complete(self.request,
                self.intent(dict(cp=2), preserve=['dp']), 1)

    def test_explicit_values_obey_search_domains(self):
        with self.assertRaises(ValueError) as failure:
            self.rewriter.complete(self.request,
                self.intent(dict(cp=4), adjustable=['dp', 'pp']), 1)
        self.assertEqual(failure.exception.args[0]['status'], 'CONSTRAINT_CONFLICT')

    def test_pp_vpp_change_generates_complete_regions(self):
        rows = self.rewriter.complete_and_rewrite(self.request,
            self.intent(dict(pp=4, vpp=2), adjustable=['dp', 'decoder_chunks', 'group']), limit=1)
        self.assertEqual(len(rows[0]['request']['decoder_chunks']), 8)
        self.assertEqual(sum(rows[0]['request']['decoder_chunks']), self.model.num_layers)
        self.assertTrue(rows[0]['graph_delta']['added_task_ids'])

    def test_multiple_legal_completions_are_returned(self):
        rows = self.rewriter.complete_and_rewrite(self.request,
            self.intent(dict(cp=2), adjustable=['dp', 'micro_batch', 'group']), limit=4)
        self.assertGreater(len(rows), 1)
        self.assertEqual(len({row['request']['run_id'] for row in rows}), len(rows))

    def test_mapping_changes_real_resource_binding(self):
        mapping = list(reversed(range(8)))
        rows = self.rewriter.complete_and_rewrite(self.request,
            self.intent(dict(rank_mapping=mapping)), limit=1)
        task = next(iter(rows[0]['execution_graph']['tasks'].values()))
        self.assertEqual(task['metadata']['physical_rank'], mapping[task['rank']])
        self.assertEqual(rows[0]['process_groups']['rank_mapping'], mapping)

    def test_stable_flow_identity_survives_constraint_closed_rewrite(self):
        parent = self.rewriter.candidate(self.request)
        flows = self.rewriter.flow_tasks(self.rewriter.compiler.graph(parent))
        flow_id = next(identity for identity in sorted(flows) if identity.startswith('flow:forward:'))
        intent = RewriteIntent(operation='JointTransform', values=dict(rank_mapping=list(reversed(range(8)))),
            adjustable=[], preserve=[], hypothesis='验证稳定Flow身份',
            evidence_refs=['constraint_fixture'], target_flow_ids=[flow_id])
        row = self.rewriter.complete_and_rewrite(self.request, intent, limit=1)[0]
        self.assertIn(flow_id, row['graph_delta']['target_impact']['target_ids'])
        self.assertIn(flow_id, row['graph_delta']['target_impact']['changed_target_ids'])
        self.assertEqual(row['graph_delta']['target_impact']['status'], 'AFFECTED')

    def test_multi_boundary_generation_scales_with_model(self):
        for layers in (12, 36, 52, 72):
            model = ModelSpec('partition_fixture', layers, 64, 16,
                [LayerType.ATTENTION] * layers, list(range(layers)))
            partitioner = RegionPartitioner(model, ParallelConfig(2, 4, 1, vpp=2))
            rows = partitioner.generate_diverse_regions(24)
            self.assertTrue(all(sum(region.num_layers for region in row) == layers for row in rows))
            uniform = rows[0]
            self.assertTrue(any(sum(left.layer_end != right.layer_end for left, right in zip(uniform, row)) > 2
                                for row in rows[1:]))

    def test_preview_and_interactions_reference_actual_tasks(self):
        parent = self.rewriter.candidate(self.request)
        intents = [self.intent(dict(cp=2)), self.intent(dict(micro_batch=2))]
        result = self.rewriter.compiler.transformation_interactions(parent, intents)
        tasks = set(self.rewriter.compiler.graph(parent).tasks)
        self.assertTrue(set(result['previews'][0]['affected_task_ids']) <= tasks)
        self.assertTrue(any(edge['kind'] == 'HardCoupling' for edge in result['edges']))
        self.assertTrue(all(edge['performance_synergy'] == 'unmeasured' for edge in result['edges']))

    def test_global_family_catalog_covers_dense_and_expert_constraints(self):
        planner = SeedPlanner(self.rewriter)
        families = planner.global_families(self.request)
        self.assertGreater(len(families), 1)
        for row in families:
            values = row['values']
            self.assertEqual(values['tp'] * values['cp'] * values['pp'] * values['dp'], 8)
            self.assertEqual(values['etp'] * values['ep'] * values['expert_dp'] * values['pp'], 8)
        self.assertGreater(len({(row['values']['tp'], row['values']['pp'], row['values']['ep'])
            for row in families}), 1)
        self.assertEqual({row['values']['execution_backend'] for row in families},
            {'megatron_ddp', 'megatron_fsdp'})
        self.assertEqual({row['values']['dense_sharding_strategy'] for row in families
            if row['values']['execution_backend'] == 'megatron_fsdp'},
            {'optim_grads_params'})
        self.assertEqual({row['values']['megatron_fsdp_version'] for row in families
            if row['values']['execution_backend'] == 'megatron_fsdp'}, {1})
        self.assertEqual({row['values']['optimizer_sharding'] for row in families
            if row['values']['execution_backend'] == 'megatron_fsdp'}, {'distributed'})

    def test_global_family_catalog_honors_explicit_ddp_distributed_scope(self):
        request = dict(self.request, allowed_execution_backends=['megatron_ddp'],
            allowed_optimizer_sharding=['distributed'])
        families = SeedPlanner(self.rewriter).global_families(request)
        self.assertTrue(families)
        self.assertEqual({row['values']['execution_backend'] for row in families},
            {'megatron_ddp'})
        self.assertEqual({row['values']['optimizer_sharding'] for row in families},
            {'distributed'})

    def test_family_selection_covers_distinct_global_parallel_groups(self):
        planner = SeedPlanner(self.rewriter)
        rows = planner.select_global_families(self.request, 4)
        global_keys = {(row['values']['tp'], row['values']['pp'], row['values']['cp'],
            row['values']['dp'], row['values']['ep'], row['values']['etp']) for row in rows}
        self.assertEqual(len(global_keys), 4)

    def test_global_family_representatives_include_complete_execution_plans(self):
        planner = SeedPlanner(self.rewriter)
        family = planner.select_global_families(self.request, 1)[0]
        rows = planner.representative_global_plans(self.request, family, 'family-evidence', limit=2)
        self.assertTrue(rows)
        for row in rows:
            candidate = row['request']
            self.assertEqual(sum(candidate['decoder_chunks']), self.model.num_layers)
            self.assertEqual(len(candidate['decoder_chunks']), candidate['pp'] * candidate['vpp'])
            self.assertEqual(candidate['tp'] * candidate['cp'] * candidate['pp'] * candidate['dp'], 8)
            self.assertEqual(candidate['etp'] * candidate['ep'] * candidate['expert_dp'] * candidate['pp'], 8)
            self.assertTrue(row['execution_graph']['tasks'])

    def test_backend_domains_enforce_tensor_parallel_limit(self):
        request = dict(self.request, max_tensor_parallel_size=2)
        domains = CapabilityRegistry({}).parallel_domains(self.model, self.topology, request)
        self.assertEqual(domains['tp'], [1, 2])
        self.assertEqual(domains['etp'], [1, 2])

    def test_tensor_parallel_hard_limit_cannot_be_expanded(self):
        domains = CapabilityRegistry({}).parallel_domains(self.model, self.topology,
            dict(self.request, max_tensor_parallel_size=8))
        self.assertEqual(domains['tp'], [1, 2, 4])
        self.assertEqual(self.rewriter.domains['tp'], [1, 2, 4])
        with self.assertRaises(ValueError):
            self.rewriter.candidate(dict(self.request, tp=8))
        with self.assertRaises(ValueError):
            self.rewriter.complete(self.request,
                self.intent(dict(tp=8), adjustable=['pp', 'dp']), 1)

    def test_30b_tensor_parallel_limit_applies_to_domains_and_explicit_intents(self):
        for name in ('Qwen3-30B-A3B', 'Nemotron-3.5-Lightning-30B-A3B'):
            request = dict(self.request, name=name, max_tensor_parallel_size=8)
            domains = CapabilityRegistry({}).parallel_domains(self.model, self.topology, request)
            self.assertEqual(domains['tp'], [1, 2])
            with self.assertRaises(ValueError):
                self.rewriter.candidate(dict(request, tp=4))
            with self.assertRaises(ValueError):
                self.rewriter.complete(request, self.intent(dict(tp=4), adjustable=['dp']), 1)

    def test_family_representatives_vary_internal_execution_strategy(self):
        planner = SeedPlanner(self.rewriter)
        family = next(row for row in planner.select_global_families(self.request, 8)
            if row['values']['execution_backend'] == 'megatron_ddp')
        rows = planner.representative_global_plans(self.request, family,
            'family-evidence', limit=4)
        self.assertGreaterEqual(len(rows), 3)
        signatures = {(row['request']['vpp'], row['request']['micro_batch'],
            tuple(row['request']['decoder_chunks']), row['request']['recompute']) for row in rows}
        self.assertEqual(len(signatures), len(rows))

    def test_fsdp_v1_representative_keeps_source_checkpoint_format(self):
        request = dict(self.request, checkpoint_format='torch_dist',
            megatron_fsdp_version=1,
            fsdp_supported_sharding_strategies=['optim_grads_params'],
            fsdp_supported_expert_sharding_strategies=['no_shard'])
        planner = SeedPlanner(self.rewriter)
        family = next(row for row in planner.global_families(request)
            if row['values']['execution_backend'] == 'megatron_fsdp'
            and row['values']['pp'] > 1)
        row = planner.representative_global_plans(request, family,
            'family-evidence', limit=1)[0]
        self.assertEqual(row['request']['checkpoint_format'], 'torch_dist')
        self.assertEqual(row['request']['fsdp_sharding_strategy'], 'optim_grads_params')

    def test_random_initial_plan_is_reproducible_and_ignores_performance(self):
        left = SeedPlanner(self.rewriter)
        right = SeedPlanner(self.rewriter)
        first = left.random_initial_plan(self.request, 17)
        second = right.random_initial_plan(self.request, 17)
        self.assertEqual(first, second)
        self.assertEqual(first['tp'] * first['cp'] * first['pp'] * first['dp'], 8)
        self.assertEqual(first['etp'] * first['ep'] * first['expert_dp'] * first['pp'], 8)
        self.assertEqual(sum(first['decoder_chunks']), self.model.num_layers)
        self.assertFalse(left.random_initial_selection['prediction_used'])
        self.assertFalse(left.random_initial_selection['measurement_used'])
        self.assertEqual(left.random_initial_selection['selection_policy'],
            'uniform_over_legal_global_parallel_configurations')

    def test_search_state_repository_persists_one_initial_configuration(self):
        store = ExperimentStore(':memory:')
        self.assertEqual(store.search_states(), [])
        document = dict(plan_hash='plan-a', request=dict(tp=1), random_seed=9,
            legal_candidate_count=4)
        identifier = store.add_search_state('INITIAL_CONFIGURATION', document)
        repeated = store.add_search_state('INITIAL_CONFIGURATION', document)
        self.assertEqual(identifier, repeated)
        states = store.search_states()
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]['kind'], 'INITIAL_CONFIGURATION')
        self.assertEqual(states[0]['document'], document)

    def test_ddp_bucket_layout_signature_ignores_batch_and_schedule_only(self):
        base = dict(self.request, etp=2, expert_dp=2, optimizer='adam',
            optimizer_sharding='distributed', execution_backend='megatron_ddp',
            precision='bfloat16', rank_mapping=list(range(8)), fsdp_bucket_elements=40000000)
        changed_batch = dict(base, micro_batch=2, group=1, overlap_p2p=True)
        self.assertEqual(bucket_layout_signature(base), bucket_layout_signature(changed_batch))
        self.assertNotEqual(bucket_layout_signature(base), bucket_layout_signature(
            dict(base, decoder_chunks=[5, 7])))
        self.assertNotEqual(bucket_layout_signature(base), bucket_layout_signature(
            dict(base, pp=4, dp=1, vpp=1, decoder_chunks=[3, 3, 3, 3])))

    def test_ddp_optimizer_defaults_match_explicit_execution_semantics(self):
        implicit = dict(self.request)
        explicit = dict(self.request, precision_aware_optimizer=False,
            main_grads_dtype='float32', main_params_dtype='torch.float32',
            exp_avg_dtype='float32', exp_avg_sq_dtype='torch.float32')
        left = normalized_execution(implicit)
        right = normalized_execution(explicit)
        for field in ('precision_aware_optimizer', 'main_grads_dtype', 'main_params_dtype',
                'exp_avg_dtype', 'exp_avg_sq_dtype'):
            self.assertEqual(left[field], right[field])

    def test_compute_profile_quality_prefers_measured_perturbation_evidence(self):
        unverified = {}
        measured = {'measurement_semantics': {
            'layer': {'scope': 'inclusive_layer_cuda_event',
                'step_comparison_available': True, 'local_service_time_verified': False},
            'paired_step_comparison': {'relative_difference': 0.08}}}
        verified = {'measurement_semantics': {
            'layer': {'scope': 'inclusive_layer_cuda_event',
                'step_comparison_available': True, 'local_service_time_verified': True},
            'paired_step_comparison': {'relative_difference': 0.12}}}
        self.assertGreater(compute_profile_quality(measured),
            compute_profile_quality(unverified))
        self.assertGreater(compute_profile_quality(verified),
            compute_profile_quality(measured))

    def test_optimizer_profile_key_reuses_independent_components_across_parallel_families(self):
        baseline = dict(self.request, model='/models/qwen3-14B', checkpoint='/models/qwen3-14B',
            optimizer='adam', optimizer_sharding='distributed', execution_backend='megatron_ddp',
            precision='bfloat16', main_grads_dtype='float32', main_params_dtype='float32',
            exp_avg_dtype='float32', exp_avg_sq_dtype='float32')
        changed_parallelism = dict(baseline, tp=1, pp=4, dp=2, vpp=2,
            etp=1, expert_dp=2, decoder_chunks=[3, 3, 3, 3])
        self.assertEqual(optimizer_profile_key(baseline),
            optimizer_profile_key(changed_parallelism))
        self.assertNotEqual(optimizer_profile_key(baseline),
            optimizer_profile_key(dict(changed_parallelism, main_grads_dtype='bfloat16')))

    def test_exact_profile_anchor_requires_same_plan_and_successful_measurement(self):
        request = dict(self.request, model='/models/qwen3-14B', checkpoint='/models/qwen3-14B')
        document = {'request': dict(request), 'anchor': {
            'run_id': 'measured-plan', 'status': 'PASS',
            'performance_eligible': True, 'step_seconds': 1.5}}
        self.assertEqual(exact_profile_anchor(document, request)['measured_step_time_ms'], 1500)
        changed = dict(request, decoder_chunks=[5, 7])
        self.assertIsNone(exact_profile_anchor(document, changed))


if __name__ == '__main__':
    unittest.main()
