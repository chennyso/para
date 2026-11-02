import unittest

from agents.strategy_transformations import GraphQuery, StrategyProposal, StrategyTransformationCompiler
from agents.program_agent import ProgramAgent
from agents.critic_agent import CriticAgent
from core_data_models import Candidate, HardwareTopology, LayerType, ModelSpec, ParallelConfig
from region_partition.region_partitioner import RegionPartitioner


class StrategyTransformationTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelSpec('Qwen3-30B-A3B', 48, 2048, 32,
            [LayerType.MOE] * 48, list(range(48)), num_experts=128)
        config = ParallelConfig(tp=1, pp=4, dp=2, cp=2, ep=4, vpp=3,
            micro_batch_size=2, microbatch_group=4, sequence_length=1024)
        self.parent = Candidate('reference', None, config,
            RegionPartitioner(self.model, config).from_counts([4] * 12), [])
        self.compiler = StrategyTransformationCompiler(self.model, HardwareTopology(16, 2, 8, 31))
        self.evidence = []

    def test_graph_contains_wrap_and_optimizer_dependencies(self):
        dag = self.compiler.graph(self.parent)
        wrap = dag.tasks['SF:0:0:4']
        self.assertEqual(wrap.rank, 12)
        self.assertEqual(wrap.metadata['group_ranks'], [12, 0])
        self.assertIn('F:0:0:3', wrap.dependencies)
        self.assertFalse(dag.has_cycle())
        self.assertTrue(any(task.task_type.value == 'optimizer_step' for task in dag.tasks.values()))

    def test_unknown_evidence_is_rejected_before_transformation(self):
        proposal = dict(name='unverified', missing_measurements=[], edits=[dict(
            operation='SetMicrobatchGroup', parameters=dict(microbatch_group=8),
            target_task_ids=['F:0:0:0'], evidence_ids=['unregistered'], hypothesis='改变分组')])
        with self.assertRaisesRegex(ValueError, '实际证据'):
            self.compiler.compile(self.parent, proposal, self.evidence)

    def test_unknown_graph_task_is_rejected(self):
        proposal = dict(name='unknown-task', missing_measurements=[], edits=[dict(
            operation='SetMicrobatchGroup', parameters=dict(microbatch_group=8),
            target_task_ids=['not-a-task'], evidence_ids=['unregistered'], hypothesis='改变分组')])
        with self.assertRaisesRegex(ValueError, '不存在的任务'):
            self.compiler.compile(self.parent, proposal, self.evidence)

    def test_native_lowering_preserves_reference_fields(self):
        program = ProgramAgent().compile(self.parent)
        reference = dict(world=16, model='Qwen3-30B-A3B', recompute='none',
                         cp_communication='p2p', overlap_p2p=True)
        lowered = ProgramAgent.lower_native(program, reference)
        self.assertEqual(lowered['decoder_chunks'], [4] * 12)
        self.assertEqual(lowered['cp_communication'], 'p2p')
        self.assertEqual(lowered['recompute'], 'none')
        self.assertEqual((lowered['global_batch'], lowered['micro_batch'], lowered['sequence']),
                         (16, 2, 1024))

    def test_bounded_context_keeps_global_dependency_interfaces(self):
        observation = self.compiler.observation(self.parent, [], [])
        bounded = self.compiler.agent_context(observation)
        graph = bounded['training_graph']
        self.assertEqual(graph['graph_sha256'], observation['training_graph']['graph_sha256'])
        self.assertLess(len(graph['tasks']), graph['full_task_count'])
        self.assertIn('SF:0:0:4', graph['tasks'])
        known = set(graph['tasks']) | set(graph['external_dependency_ids'])
        self.assertTrue(all(set(task['dependencies']) <= known for task in graph['tasks'].values()))

    def test_graph_query_returns_explicit_external_measurement_gaps(self):
        observation = self.compiler.observation(self.parent, [], [])
        result = self.compiler.query(observation, GraphQuery(operation='analyze_critical_path'))
        self.assertEqual(result['status'], 'measurement_required')
        self.assertIsNone(result['predicted_step_time_ms'])

    def test_graph_query_rejects_unknown_task(self):
        observation = self.compiler.observation(self.parent, [], [])
        with self.assertRaisesRegex(ValueError, '不存在的任务'):
            self.compiler.query(observation, GraphQuery(operation='inspect_graph', task_ids=['unknown']))

    def test_reference_identity_can_query_its_complete_graph_pages(self):
        observation = self.compiler.observation(self.parent, [], [])
        result = self.compiler.query(observation, GraphQuery(operation='inspect_graph',
            candidate_id=self.parent.candidate_id, limit=8))
        self.assertEqual(len(result['tasks']), 8)
        self.assertEqual(result['next_offset'], 8)
        self.assertEqual(result['full_task_count'], len(observation['training_graph']['tasks']))

    def test_region_change_updates_task_metadata_without_claiming_speed(self):
        config = self.parent.parallel_config
        changed = Candidate('changed', self.parent.candidate_id, config,
            RegionPartitioner(self.model, config).from_counts([3, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4]), [])
        delta = self.compiler.graph_delta(self.parent, changed)
        self.assertIn('F:0:0:0', delta['modified_task_ids'])
        self.assertEqual(delta['added_task_ids'], [])
        self.assertFalse(changed.actual_throughput)

    def test_related_programs_combine_layout_and_p2p_with_bounded_budget(self):
        evidence = [dict(evidence_id='actual-measurement')]
        proposals = [StrategyProposal.model_validate(dict(name=name, missing_measurements=[],
            edits=[dict(operation=operation, parameters=parameters,
                target_task_ids=['F:0:0:0'], evidence_ids=['actual-measurement'],
                hypothesis='检查Region边界与相邻通信的结构关联')]))
            for name, operation, parameters in [
                ('layout', 'SetRegionLayout', dict(decoder_chunks=[3,4,4,4,4,4,4,4,5,4,4,4])),
                ('overlap', 'ConfigureP2POverlap', dict(enabled=False)),
                ('bucket', 'ConfigureGradientBucket', dict(bucket_elements=1000000))]]
        results = self.compiler.related_programs(self.parent, proposals, evidence, 1)
        self.assertEqual(len(results), 1)
        record = results[0].transformation_record
        self.assertEqual(set(record['changed_variables']), {'decoder_chunks', 'overlap_p2p'})
        self.assertIn('adjacent_dependencies', record['interaction_scope']['reasons'])
        self.assertIsNone(record['search_selection']['predicted_step_time_ms'])
        self.assertFalse(results[0].actual_throughput)

    def test_same_gpu_without_dependency_does_not_establish_coupling(self):
        first = StrategyProposal.model_validate(dict(name='overlap', missing_measurements=[],
            edits=[dict(operation='ConfigureP2POverlap', parameters=dict(enabled=False),
                target_task_ids=['SF:0:0:1'], evidence_ids=['measurement'], hypothesis='调整P2P通信')]))
        second = StrategyProposal.model_validate(dict(name='bucket', missing_measurements=[],
            edits=[dict(operation='ConfigureGradientBucket', parameters=dict(bucket_elements=1000000),
                target_task_ids=['DP:0'], evidence_ids=['measurement'], hypothesis='调整梯度通信')]))
        relation = self.compiler.interaction_scope(self.parent, first, second)
        self.assertTrue(relation['shared_gpu_ranks'])
        self.assertFalse(relation['related'])

    def test_critical_frontier_requires_measured_complete_dependencies(self):
        dag = self.compiler.graph(self.parent)
        with self.assertRaisesRegex(ValueError, '真实执行依赖'):
            CriticAgent().find_critical_frontier(dag)

    def test_cp_change_derives_mbs_compensation_and_rejects_insufficient_microbatches(self):
        completed = self.compiler.complete_parallel_change(self.parent, {'cp': 1},
            ['measurement'], [dict(evidence_id='measurement')])
        candidates = completed['candidates']
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.parallel_config.dp == 4 for candidate in candidates))
        preserved_vpp = next(candidate for candidate in candidates if candidate.parallel_config.vpp == 3)
        self.assertEqual(preserved_vpp.parallel_config.micro_batch_size, 1)
        self.assertEqual(preserved_vpp.parallel_config.num_microbatches, 4)
        self.assertEqual(preserved_vpp.parallel_config.global_batch_size, 16)
        closure = preserved_vpp.transformation_record['constraint_closure']
        self.assertEqual((closure['before_local_sequence'], closure['after_local_sequence']), (512, 1024))
        self.assertEqual(closure['necessary']['expert_data_parallel'], 1)
        self.assertIsNone(closure['performance_impact']['predicted_step_time_ms'])
        rejected = [item for item in completed['rejected']
                    if item['config']['mbs'] == 2 and item['config']['vpp'] == 3]
        self.assertTrue(rejected)
        program = ProgramAgent().compile(preserved_vpp)
        lowered = ProgramAgent.lower_native(program, dict(world=16, model='Qwen3-30B-A3B'))
        self.assertEqual((lowered['cp'], lowered['micro_batch'], lowered['vpp']), (1, 1, 3))
        proposal = StrategyProposal.model_validate(preserved_vpp.transformation_record['proposal'])
        self.assertEqual(self.compiler.parameter_conflicts(proposal), [])
        compact = self.compiler.candidate_context(preserved_vpp)
        delta = compact['transformation_record']['graph_delta']
        full_delta = preserved_vpp.transformation_record['graph_delta']
        self.assertEqual(delta['affected_task_ids_count'], len(full_delta['affected_task_ids']))
        self.assertLessEqual(len(delta['affected_task_ids']), 16)
        self.assertEqual(compact['config'], preserved_vpp.to_dict()['config'])
        self.assertEqual(compact['region_layout'], preserved_vpp.to_dict()['region_layout'])

    def test_tp_change_closes_expert_folding_without_multiplying_ep_into_world(self):
        completed = self.compiler.complete_parallel_change(self.parent, {'tp': 2, 'cp': 1},
            ['measurement'], [dict(evidence_id='measurement')])
        candidate = next(item for item in completed['candidates'] if item.parallel_config.vpp == 3)
        self.assertEqual((candidate.parallel_config.tp, candidate.parallel_config.cp,
            candidate.parallel_config.dp, candidate.parallel_config.ep), (2, 1, 2, 2))
        self.assertEqual(candidate.parallel_config.world_size, 16)
        closure = candidate.transformation_record['constraint_closure']
        self.assertEqual(closure['closed_parallel_change']['ep'], 2)
        self.assertEqual(closure['handoff_token_count'][0], closure['handoff_token_count'][1])

    def test_mbs_diff_separates_message_frequency_from_total_payload(self):
        candidate = self.compiler.compile(self.parent, dict(name='mbs1', missing_measurements=[],
            edits=[dict(operation='SetMicrobatchSize', parameters=dict(micro_batch_size=1),
                target_task_ids=['F:0:0:0'], evidence_ids=['measurement'], hypothesis='检验任务数量与载荷')]),
            [dict(evidence_id='measurement')])
        delta = candidate.transformation_record['graph_delta']['workload_summary']
        self.assertEqual((delta['before']['microbatches'], delta['after']['microbatches']), (4, 8))
        self.assertEqual(delta['after']['logical_pp_transfers'], 2 * delta['before']['logical_pp_transfers'])
        self.assertEqual(delta['after']['logical_pp_tensor_bytes'], delta['before']['logical_pp_tensor_bytes'])
        snapshot = self.compiler.observation(candidate, [], [])
        result = self.compiler.query(dict(candidate_observations={candidate.candidate_id: snapshot}),
            GraphQuery(operation='inspect_candidate_diff', candidate_id=candidate.candidate_id))
        self.assertEqual(result['workload_summary'], delta)
        self.assertIsNone(result['predicted_step_time_ms'])


if __name__ == '__main__':
    unittest.main()
