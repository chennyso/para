import json
import unittest
from pathlib import Path

from core_data_models import HardwareTopology, LayerType, ModelSpec, Task, TaskType
from cost.runtime_calibration import configuration
from predictor.dag_projection import DAGProjector
from predictor.state_sharding import _widths, parameter_state_ownership, phased_memory_bounds, state_sharding_errors
from region_partition.region_partitioner import RegionPartitioner
from search.seed_planner import SeedPlanner
from scheduling.schedule_simulator import ScheduleCostEvaluator, topological_order
from scheduling.task_graph_builder import TaskGraphBuilder


class StateShardingTests(unittest.TestCase):
    def request(self, strategy='no_shard', dp=4, tp=2, pp=1, vpp=1):
        world = tp * pp * dp
        return dict(run_id='state-test', tp=tp, cp=1, pp=pp, dp=dp, ep=1, etp=tp,
            expert_dp=dp, vpp=vpp, group=max(1, pp), sequence=128, micro_batch=1,
            global_batch=max(8, dp * 2), offload=False, recompute='full',
            decoder_chunks=[4 // (pp * vpp)] * (pp * vpp), node_count=1,
            gpus_per_node=world, precision='bfloat16', execution_backend='megatron_fsdp',
            main_grads_dtype='float32', megatron_fsdp_main_params_dtype='float32',
            exp_avg_dtype='float32', exp_avg_sq_dtype='float32',
            megatron_fsdp_version=1, optimizer_sharding='distributed',
            fsdp_sharding_strategy=strategy, dense_sharding_strategy=strategy,
            expert_sharding_strategy='no_shard',
            dp_inner_size=dp, dp_outer_size=1, expert_dp_inner_size=dp,
            expert_dp_outer_size=1, outer_sharding_strategy='no_shard',
            expert_outer_sharding_strategy='no_shard', checkpoint_format='fsdp_dtensor',
            optimizer_checkpoint_mode='disabled', fsdp_multi_chunk_supported=True,
            fsdp_hybrid_outer_supported=False,
            fsdp_supported_sharding_strategies=['optim_grads_params'],
            fsdp_supported_expert_sharding_strategies=['no_shard'],
            fsdp_unit='transformer_layer',
            fsdp_prefetch=False, fsdp_prefetch_distance=1,
            fsdp_reshard_after_forward=True, fsdp_double_buffer=False,
            fsdp_bucket_elements=1000)

    def test_v1_dense_zero_three_ownership(self):
        request = self.request('optim_grads_params')
        config = configuration(request)
        counts = dict(dense={rank: 1000 for rank in range(config.world_size)},
            expert={rank: 0 for rank in range(config.world_size)})
        dense = parameter_state_ownership(counts, config, request)[0]['components']['dense']
        self.assertEqual((dense['persistent_parameter_bytes'],
            dense['persistent_gradient_bytes'], dense['gpu_optimizer_bytes'],
            dense['unshard_parameter_bytes']), (500, 1000, 3000, 2000))

    def test_dp_one_has_no_cross_rank_sharding_gain(self):
        request = self.request('optim_grads_params', dp=1, tp=8)
        config = configuration(request)
        counts = dict(dense={rank: 1000 for rank in range(config.world_size)},
            expert={rank: 0 for rank in range(config.world_size)})
        dense = parameter_state_ownership(counts, config, request)[0]['components']['dense']
        self.assertEqual(dense['persistent_parameter_bytes'], 2000)
        self.assertEqual(dense['persistent_gradient_bytes'], 4000)
        self.assertEqual(dense['gpu_optimizer_bytes'], 12000)
        self.assertFalse(dense['sharding_effective'])

    def test_gradient_residency_uses_main_dtype_independent_of_communication(self):
        request = self.request('optim_grads_params')
        request.update(main_grads_dtype='float32', grad_comm_dtype='bfloat16')
        self.assertEqual(_widths(request)['gradient'], 4)
        request.update(main_grads_dtype='torch.bfloat16', grad_comm_dtype='float32')
        self.assertEqual(_widths(request)['gradient'], 2)

    def test_precision_aware_gradient_dtype_requires_explicit_configuration(self):
        request = self.request('optim_grads_params')
        request['precision_aware_optimizer'] = True
        request.pop('main_grads_dtype')
        with self.assertRaisesRegex(ValueError, 'main_grads_dtype'):
            _widths(request)

    def test_outer_sharding_is_rejected(self):
        request = self.request('optim_grads_params')
        request.update(dp_inner_size=2, dp_outer_size=2,
            expert_dp_inner_size=2, expert_dp_outer_size=2,
            outer_sharding_strategy='optim')
        self.assertTrue(any('Outer Sharding' in row
            for row in state_sharding_errors(request, configuration(request))))

    def test_fsdp_version_is_fixed_to_v1(self):
        request = self.request('optim_grads_params')
        request['megatron_fsdp_version'] = 2
        self.assertTrue(any('FSDP v1' in row
            for row in state_sharding_errors(request, configuration(request))))

    def test_cpu_optimizer_offload_requires_ddp_distributed_precision_aware(self):
        request = self.request('optim_grads_params')
        request.update(offload=True, optimizer_offload_fraction=1.0,
            optimizer_transfer_overlap=True)
        errors = state_sharding_errors(request, configuration(request))
        self.assertTrue(any('Megatron DDP' in row for row in errors))
        request.update(execution_backend='megatron_ddp', megatron_fsdp_version=0,
            fsdp_sharding_strategy='no_shard', dense_sharding_strategy='no_shard',
            expert_sharding_strategy='no_shard', checkpoint_format='torch_dist',
            precision_aware_optimizer=True)
        self.assertFalse(state_sharding_errors(request, configuration(request)))

    def test_precision_aware_optimizer_rejects_replicated_ddp(self):
        request = self.request('no_shard')
        request.update(execution_backend='megatron_ddp', megatron_fsdp_version=0,
            optimizer_sharding='replicated', checkpoint_format='torch_dist',
            precision_aware_optimizer=True)
        errors = state_sharding_errors(request, configuration(request))
        self.assertTrue(any('Precision-Aware Optimizer' in row for row in errors))

    def test_seed_state_families_disable_precision_aware_for_replicated_ddp(self):
        request = self.request('no_shard')
        request.update(allowed_execution_backends=['megatron_ddp'],
            allowed_optimizer_sharding=['distributed', 'replicated'],
            precision_aware_optimizer=True)
        families = SeedPlanner.state_families(request, dense_dp=4, expert_dp=4, is_moe=False)
        distributed = next(row for row in families if row['optimizer_sharding'] == 'distributed')
        replicated = next(row for row in families if row['optimizer_sharding'] == 'replicated')
        self.assertTrue(distributed['precision_aware_optimizer'])
        self.assertFalse(replicated['precision_aware_optimizer'])

    def test_cpu_optimizer_offload_moves_optimizer_state_to_host(self):
        request = self.request('optim_grads_params')
        request.update(execution_backend='megatron_ddp', megatron_fsdp_version=0,
            optimizer_sharding='distributed', fsdp_sharding_strategy='no_shard',
            dense_sharding_strategy='no_shard', expert_sharding_strategy='no_shard',
            checkpoint_format='torch_dist', precision_aware_optimizer=True,
            main_params_dtype='bfloat16', exp_avg_dtype='bfloat16',
            exp_avg_sq_dtype='bfloat16', main_grads_dtype='bfloat16',
            offload=True, optimizer_offload_fraction=0.5,
            optimizer_transfer_overlap=True)
        config = configuration(request)
        counts = dict(dense={rank: 1000 for rank in range(config.world_size)},
            expert={rank: 0 for rank in range(config.world_size)})
        row = parameter_state_ownership(counts, config, request)[0]
        self.assertGreater(row['host_optimizer_bytes'], 0)
        self.assertGreater(row['host_pinned_buffer_bytes'], 0)
        self.assertGreater(row['gpu_persistent_bytes'], 0)

    def test_loading_and_training_peaks_are_separate(self):
        request = self.request('optim_grads_params')
        config = configuration(request)
        counts = dict(dense={rank: 1000 for rank in range(config.world_size)},
            expert={rank: 0 for rank in range(config.world_size)})
        phases = phased_memory_bounds(counts, config, request,
            {rank: 10000 for rank in range(config.world_size)}, 1000, 2000,
            checkpoint_buffer_bytes=8000)
        self.assertNotEqual(phases[0]['gpu_loading_peak_bytes'],
            phases[0]['gpu_training_peak_bytes'])
        self.assertIn(phases[0]['most_dangerous_phase'], phases[0]['gpu_phases'])

    def test_v1_accepts_existing_torch_dist_checkpoint(self):
        request = self.request('optim_grads_params')
        request['checkpoint_format'] = 'torch_dist'
        self.assertFalse(state_sharding_errors(request, configuration(request)))

    def test_v1_accepts_pipeline_and_virtual_pipeline_parallelism(self):
        request = self.request('optim_grads_params', dp=2, tp=1, pp=2, vpp=2)
        self.assertFalse(state_sharding_errors(request, configuration(request)))


class FSDPDAGTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelSpec('fsdp-dag', 4, 64, 8,
            [LayerType.MLP] * 4, list(range(4)), ffn_hidden_size=128)

    def build(self, prefetch=False):
        request = StateShardingTests().request('optim_grads_params', dp=2, tp=1, pp=1)
        request.update(fsdp_prefetch=prefetch)
        config = configuration(request)
        regions = RegionPartitioner(self.model, config).from_counts([4])
        return TaskGraphBuilder().build(self.model, config, regions, config.num_microbatches)

    def test_fsdp_dag_has_parameter_and_gradient_lifecycle(self):
        dag = self.build()
        kinds = {task.task_type for task in dag.tasks.values()}
        self.assertIn(TaskType.FSDP_UNSHARD, kinds)
        self.assertIn(TaskType.FSDP_RESHARD, kinds)
        self.assertIn(TaskType.FSDP_GRAD_REDUCE_SCATTER, kinds)
        self.assertTrue(all(task.metadata.get('compute_task') in dag.tasks
            for task in dag.tasks.values()
            if task.task_type == TaskType.FSDP_GRAD_REDUCE_SCATTER))
        self.assertNotIn('CPUOPT:0', dag.tasks)
        topological_order(dag.tasks)
        self.assertTrue(all(dag.tasks[f'DP:{rank}'].task_type == TaskType.CUDA_SYNC
            for rank in range(2)))

    def test_prefetch_is_real_collective_with_wait_node(self):
        dag = self.build(prefetch=True)
        prefetch = next(task for task in dag.tasks.values()
            if task.task_type == TaskType.FSDP_PREFETCH)
        wait = dag.tasks[prefetch.task_id.replace('FSP:', 'FSU:')]
        self.assertEqual(prefetch.metadata['cost_role'], 'prefetched_all_gather')
        self.assertEqual(wait.metadata['cost_role'], 'prefetch_wait')
        self.assertIn(prefetch.task_id, wait.dependencies)
        self.assertEqual(prefetch.metadata['communication_payload_definition'], 'local_input_shard_bytes')
        self.assertEqual(prefetch.metadata['communication_dtype'], 'bfloat16')
        self.assertEqual(prefetch.metadata['communication_full_payloads_bytes'],
            [2 * value for value in prefetch.metadata['communication_payloads']])
        self.assertEqual(prefetch.metadata['payload_bytes'],
            max(prefetch.metadata['communication_full_payloads_bytes']))
        topological_order(dag.tasks)

    def test_parameter_and_gradient_collectives_keep_full_buffers(self):
        dag = self.build()
        for task in dag.tasks.values():
            if task.task_type not in (TaskType.FSDP_UNSHARD, TaskType.FSDP_GRAD_REDUCE_SCATTER):
                continue
            metadata = task.metadata
            self.assertEqual(metadata['communication_full_payloads_bytes'],
                [2 * value for value in metadata['communication_payloads']])
            self.assertEqual(metadata['payload_bytes'], max(metadata['communication_full_payloads_bytes']))
            self.assertEqual(metadata['communication_dtype'], 'bfloat16'
                if task.task_type == TaskType.FSDP_UNSHARD else 'float32')

    def test_single_rank_group_generates_no_sharding_collectives(self):
        request = StateShardingTests().request('optim_grads_params', dp=1, tp=1, pp=1)
        config = configuration(request)
        regions = RegionPartitioner(self.model, config).from_counts([4])
        dag = TaskGraphBuilder().build(self.model, config, regions, config.num_microbatches)
        self.assertFalse(any(task.task_type in (TaskType.FSDP_UNSHARD,
            TaskType.FSDP_GRAD_REDUCE_SCATTER) for task in dag.tasks.values()))

    def test_explicit_gradient_communication_dtype_is_propagated(self):
        request = StateShardingTests().request('optim_grads_params', dp=2, tp=1, pp=1)
        request.update(main_grads_dtype='float32', grad_comm_dtype='bfloat16')
        config = configuration(request)
        regions = RegionPartitioner(self.model, config).from_counts([4])
        dag = TaskGraphBuilder().build(self.model, config, regions, config.num_microbatches)
        reduced = [task for task in dag.tasks.values()
            if task.task_type == TaskType.FSDP_GRAD_REDUCE_SCATTER]
        self.assertTrue(reduced)
        self.assertTrue(all(task.metadata['communication_dtype'] == 'bfloat16' for task in reduced))

    def test_dense_zero_three_with_replicated_experts_keeps_expert_finalize(self):
        model = ModelSpec('fsdp-moe', 4, 64, 8, [LayerType.MOE] * 4,
            list(range(4)), num_experts=8, ffn_hidden_size=128)
        request = StateShardingTests().request('optim_grads_params', dp=2, tp=1, pp=1)
        request.update(expert_sharding_strategy='no_shard',
            fsdp_supported_sharding_strategies=['optim_grads_params'],
            fsdp_supported_expert_sharding_strategies=['no_shard'])
        config = configuration(request)
        regions = RegionPartitioner(model, config).from_counts([4])
        dag = TaskGraphBuilder().build(model, config, regions, config.num_microbatches)
        reduced_families = {task.metadata['state_family'] for task in dag.tasks.values()
            if task.task_type == TaskType.FSDP_GRAD_REDUCE_SCATTER}
        self.assertEqual(reduced_families, {'dense'})
        self.assertTrue(any(task.task_type == TaskType.DP_REDUCE for task in dag.tasks.values()))
        topological_order(dag.tasks)

    def test_fsdp_group_collective_has_one_representative(self):
        dag = self.build()
        DAGProjector._coalesce_fsdp_collectives(dag.tasks)
        instances = {}
        for task in dag.tasks.values():
            instance = task.metadata.get('collective_instance_id')
            if instance is not None:
                instances.setdefault(instance, []).append(task)
        self.assertTrue(instances)
        for rows in instances.values():
            self.assertEqual(sum(task.metadata['collective_representative'] for task in rows), 1)
            representative = next(task for task in rows
                if task.metadata['collective_representative'])
            self.assertTrue(all(task.dependencies == [representative.task_id]
                for task in rows if task is not representative))
        topological_order(dag.tasks)


class ResourceAwareReplayTests(unittest.TestCase):
    def test_zero_duration_terminal_tasks_complete_replay(self):
        tasks = {
            'compute': Task('compute', 0, TaskType.FORWARD_COMPUTE,
                resource_requirements=('gpu:0:compute',)),
            'sync': Task('sync', 0, TaskType.CUDA_SYNC,
                dependencies=['compute'], resource_requirements=()),
        }
        _, predecessors, _ = topological_order(tasks)
        times, _, _ = ScheduleCostEvaluator(None).simulate_resource_aware_events(
            tasks, predecessors, {'compute': 1.0, 'sync': 0.0}, None)
        self.assertEqual(times['sync'], (1.0, 1.0))

    def test_shared_link_uses_dynamic_equal_bandwidth(self):
        tasks = {
            'c0': Task('c0', 0, TaskType.NCCL_GENERIC,
                resource_requirements=('ib:0',), metadata={'group_ranks': [0, 1]}),
            'c1': Task('c1', 1, TaskType.NCCL_GENERIC,
                resource_requirements=('ib:0',), metadata={'group_ranks': [0, 1]}),
        }
        order, predecessors, _ = topological_order(tasks)
        self.assertEqual(len(order), 2)
        times, _, diagnostics = ScheduleCostEvaluator(None).simulate_resource_aware_events(
            tasks, predecessors, {'c0': 10.0, 'c1': 10.0}, None)
        self.assertEqual(times['c0'][1], 20.0)
        self.assertEqual(times['c1'][1], 20.0)
        self.assertTrue(diagnostics['bandwidth_sharing_intervals'])

    def test_calibrated_overlap_gamma_slows_compute_progress(self):
        tasks = {
            'compute': Task('compute', 0, TaskType.BACKWARD_COMPUTE,
                resource_requirements=('gpu:0:compute',), metadata={'operation': 'region'}),
            'comm': Task('comm', 0, TaskType.NCCL_GENERIC,
                resource_requirements=('ib:0',), metadata={'group_ranks': [0, 1],
                    'participant_ranks': [0, 1], 'collective': 'reduce_scatter'}),
        }
        _, predecessors, _ = topological_order(tasks)
        model = {'compute_communication_slowdowns': [{
            'compute_kind': 'backward_compute', 'collective': 'reduce_scatter',
            'gamma': 2.0, 'evidence_ids': ['pair-profile']}]}
        times, _, diagnostics = ScheduleCostEvaluator(None).simulate_resource_aware_events(
            tasks, predecessors, {'compute': 10.0, 'comm': 10.0}, None, model)
        self.assertEqual(times['comm'][1], 10.0)
        self.assertEqual(times['compute'][1], 15.0)
        self.assertTrue(diagnostics['compute_communication_overlap_intervals'])
        self.assertFalse(diagnostics['missing_overlap_profiles'])

    def test_differential_anchor_scope_excludes_global_family_change(self):
        request = StateShardingTests().request('optim_grads_params')
        region = dict(request, decoder_chunks=[2, 2])
        changed = StateShardingTests().request('optim_grads_params', dp=2, tp=4)
        self.assertEqual(DAGProjector.differential_family(request),
            DAGProjector.differential_family(region))
        self.assertNotEqual(DAGProjector.differential_family(request),
            DAGProjector.differential_family(changed))


if __name__ == '__main__':
    unittest.main()
