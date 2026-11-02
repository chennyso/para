import unittest

from core_data_models import LayerType, ModelSpec, TaskType
from cost.runtime_calibration import configuration
from region_partition.region_partitioner import RegionPartitioner
from scheduling.schedule_simulator import topological_order
from scheduling.task_graph_builder import TaskGraphBuilder


class FinalizeModelGradsCompilerTests(unittest.TestCase):
    def request(self, **updates):
        request = dict(run_id='finalize-test', tp=2, cp=1, pp=2, dp=2, ep=1, etp=2,
            expert_dp=2, vpp=2, group=2, sequence=128, micro_batch=1, global_batch=8,
            offload=False, recompute='full', decoder_chunks=[1, 1, 1, 1],
            node_count=2, gpus_per_node=4, precision='bfloat16',
            optimizer_sharding='distributed', execution_backend='megatron_ddp',
            rank_mapping=[7, 6, 5, 4, 3, 2, 1, 0])
        request.update(updates)
        return request

    def model(self, share=False, mtp=0, position_elements=0):
        return ModelSpec('finalize-model', 4, 64, 8, [LayerType.MLP] * 4,
            list(range(4)), ffn_hidden_size=128, finalize_grad_semantics=dict(
                share_embeddings_and_output_weights=share, mtp_num_layers=mtp,
                embedding_requires_grad=True, vocab_size=128,
                position_embedding_gradient_elements=position_elements,
                non_tensor_parallel_grad_elements_per_layer=128,
                final_layernorm_grad_elements=64, qk_layernorm=False,
                source='unit_actual_model_semantics'))

    def build(self, model, request):
        config = configuration(request)
        regions = RegionPartitioner(model, config).from_counts(request['decoder_chunks'])
        return TaskGraphBuilder().build(model, config, regions, config.num_microbatches)

    def test_mtp_word_embedding_sync_uses_all_groups_and_physical_mapping(self):
        dag = self.build(self.model(share=False, mtp=1), self.request())
        rows = sorted((task for task in dag.tasks.values()
            if task.metadata.get('operation') == 'word_embedding_gradient_all_reduce'),
            key=lambda task: task.task_id)
        self.assertEqual([task.metadata['group_ranks'] for task in rows],
            [[0, 4], [2, 6], [1, 5], [3, 7]])
        self.assertEqual([task.metadata['physical_group_ranks'] for task in rows],
            [[7, 3], [5, 1], [6, 2], [4, 0]])
        self.assertTrue(all(task.metadata['payload_bytes'] == 16384
            and task.metadata['dtype'] == 'float32' for task in rows))
        self.assertTrue(all(task.dependencies == [f'DP:{rank}' for rank in task.metadata['group_ranks']]
            for task in rows))
        self.assertTrue(all(f'FINAL:{rank}' in dag.tasks[f'OPREP:{rank}'].dependencies
            for rank in range(8)))
        self.assertTrue(all(not dependency.startswith('DP:')
            for rank in range(8) for dependency in dag.tasks[f'OPT:{rank}'].dependencies))
        topological_order(dag.tasks)

    def test_optimizer_gradient_norm_waits_for_every_instance_rank(self):
        dag = self.build(self.model(), self.request())
        reduction = dag.tasks['ONORM:AR:0']
        self.assertEqual(reduction.metadata['group_ranks'], list(range(8)))
        self.assertEqual(reduction.metadata['physical_group_ranks'], list(reversed(range(8))))
        self.assertEqual(reduction.metadata['payload_bytes'], 4)
        self.assertEqual(reduction.metadata['dtype'], 'float32')
        self.assertEqual(set(reduction.dependencies), {f'ONORM:{rank}' for rank in range(8)})
        for rank in range(8):
            self.assertEqual(dag.tasks[f'ONORM:{rank}'].dependencies, [f'OPREP:{rank}'])
            self.assertEqual(dag.tasks[f'ONORM_POST:{rank}'].dependencies, ['ONORM:AR:0'])
            self.assertEqual(dag.tasks[f'OCLIP:{rank}'].dependencies, [f'ONORM_POST:{rank}'])
            self.assertEqual(dag.tasks[f'OPT:{rank}'].dependencies, [f'OCLIP:{rank}'])
        topological_order(dag.tasks)

    def test_replicated_optimizer_uses_model_parallel_statistics_groups(self):
        dag = self.build(self.model(), self.request(optimizer_sharding='replicated'))
        groups = [task.metadata['group_ranks'] for task in dag.tasks.values()
            if task.metadata.get('operation') == 'optimizer_gradient_norm_all_reduce']
        self.assertEqual(groups, [[0, 1, 4, 5], [2, 3, 6, 7]])
        topological_order(dag.tasks)

    def test_disabled_clipping_keeps_chained_optimizer_norm(self):
        dag = self.build(self.model(), self.request(clip_grad=0.0))
        self.assertIn('ONORM:AR:0', dag.tasks)
        self.assertFalse(any(task.task_type == TaskType.GRADIENT_CLIP for task in dag.tasks.values()))
        self.assertTrue(all(dag.tasks[f'OPT:{rank}'].dependencies == [f'ONORM_POST:{rank}']
            for rank in range(8)))
        topological_order(dag.tasks)

    def test_optimizer_statistics_require_supported_instance_groups(self):
        with self.assertRaisesRegex(ValueError, 'DistributedOptimizer实例'):
            self.build(self.model(), self.request(num_distributed_optimizer_instances=2))

    def test_untied_without_mtp_has_no_word_embedding_sync(self):
        dag = self.build(self.model(share=False, mtp=0), self.request())
        self.assertFalse(any(task.metadata.get('operation') == 'word_embedding_gradient_all_reduce'
            for task in dag.tasks.values()))

    def test_qk_gradient_sync_keeps_sequence_parallel_condition(self):
        model = self.model()
        model.finalize_grad_semantics.update(qk_layernorm=True,
            qk_layernorm_grad_elements_per_layer=16)
        for sequence_parallel in (False, True):
            dag = self.build(model, self.request(sequence_parallel=sequence_parallel))
            rows = [task for task in dag.tasks.values()
                if task.metadata.get('operation') == 'non_tensor_parallel_gradient_all_reduce']
            self.assertEqual(len(rows), 4)
            for task in rows:
                elements = 2 * (16 + (128 if sequence_parallel else 0))
                if sequence_parallel and min(task.metadata['group_ranks']) >= 4:
                    elements += 64
                self.assertEqual(task.metadata['payload_bytes'], elements * 4)

    def test_finalize_collectives_are_once_per_step(self):
        model = self.model(share=True)
        first = self.build(model, self.request(global_batch=8))
        second = self.build(model, self.request(global_batch=16))
        for operation in ('word_embedding_gradient_all_reduce',
                          'non_tensor_parallel_gradient_all_reduce'):
            self.assertEqual(sum(task.metadata.get('operation') == operation
                for task in first.tasks.values()), sum(task.metadata.get('operation') == operation
                for task in second.tasks.values()))

    def test_position_embedding_sync_requires_multi_rank_group(self):
        model = self.model(position_elements=8192)
        default = self.build(model, self.request())
        self.assertFalse(any(task.metadata.get('operation') == 'position_embedding_gradient_all_reduce'
            for task in default.tasks.values()))
        enabled = self.build(model, self.request(position_embedding_group_policy='first_last'))
        rows = [task for task in enabled.tasks.values()
            if task.metadata.get('operation') == 'position_embedding_gradient_all_reduce']
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(task.metadata['payload_bytes'] == 16384 for task in rows))

    def test_fsdp_v1_uses_local_gradient_shard_for_embedding_sync(self):
        request = self.request(execution_backend='megatron_fsdp', megatron_fsdp_version=1,
            fsdp_sharding_strategy='optim_grads_params', dense_sharding_strategy='optim_grads_params',
            expert_sharding_strategy='no_shard', dp_inner_size=2, dp_outer_size=1,
            expert_dp_inner_size=2, expert_dp_outer_size=1,
            outer_sharding_strategy='no_shard', expert_outer_sharding_strategy='no_shard',
            fsdp_unit='transformer_layer', fsdp_prefetch=False, fsdp_prefetch_distance=1,
            fsdp_reshard_after_forward=True, fsdp_double_buffer=False,
            fsdp_bucket_elements=1000, checkpoint_format='torch_dist', checkpoint_reshard=True,
            optimizer_checkpoint_mode='disabled', precision_aware_optimizer=False,
            fsdp_multi_chunk_supported=True, fsdp_hybrid_outer_supported=False,
            fsdp_optimizer_checkpoint_supported=False,
            fsdp_separate_expert_outer_mesh_supported=False,
            fsdp_supported_sharding_strategies=['optim_grads_params'],
            fsdp_supported_expert_sharding_strategies=['no_shard'],
            fsdp_supported_units=['transformer_layer'])
        dag = self.build(self.model(mtp=1), request)
        rows = [task for task in dag.tasks.values()
            if task.metadata.get('operation') == 'word_embedding_gradient_all_reduce']
        self.assertTrue(rows)
        self.assertTrue(all(task.metadata['payload_bytes'] == 8192 for task in rows))
        self.assertTrue(all(task.metadata['gradient_layout'] ==
            'megatron_fsdp_v1_local_gradient_shard' for task in rows))
        topological_order(dag.tasks)


if __name__ == '__main__':
    unittest.main()
