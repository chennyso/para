import math
import unittest

from core_data_models import ParallelConfig, HardwareTopology, Task, TaskType, ExecutionPlan, ModelSpec, LayerType
from cost.cost_model import ExecutionCostModel, MissingCalibration
from scheduling.schedule_simulator import ScheduleCostEvaluator
from scheduling.task_graph_builder import TaskGraphBuilder
from region_partition.region_partitioner import RegionPartitioner
from validation.parallel_validator import ParallelValidator
from scheduling.critical_path import CriticalPathAnalyzer
from scheduling.microbatch_scheduler import MicrobatchScheduler


class CostModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ParallelConfig(tp=1, pp=1, dp=1, global_batch_size=1)
        self.hardware = HardwareTopology(1, 1, 1, 32.0)
        self.costs = ExecutionCostModel()

    def measured(self, task, milliseconds):
        task.metadata['memory_evidence'] = 'analytic_unit_test:显式固定显存生命周期'
        self.costs.update(dict(task=task, config=self.config, milliseconds=milliseconds,
            source='analytic_unit_test', evidence_id='固定图数学验证'))

    def plan(self, tasks, schedule):
        return ExecutionPlan(self.config, [], {0: schedule}, tasks={task.task_id: task for task in tasks},
                             persistent_bytes={0: 1000})

    def test_no_unmeasured_defaults(self):
        with self.assertRaises(MissingCalibration):
            self.costs.get_task_cost(TaskType.OPTIMIZER_STEP, self.config)

    def test_critical_path_handles_join_and_optimizer(self):
        from core_data_models import ExecutionDAG
        tasks = [Task('F', 0, TaskType.FORWARD_COMPUTE, duration=8),
                 Task('B', 0, TaskType.BACKWARD_COMPUTE, duration=16, dependencies=['F']),
                 Task('COMM', 1, TaskType.DP_REDUCE, duration=4, dependencies=['F']),
                 Task('OPT', 0, TaskType.OPTIMIZER_STEP, duration=30, dependencies=['B', 'COMM'])]
        analyzer = CriticalPathAnalyzer()
        self.assertEqual(analyzer.identify_critical_path(ExecutionDAG(
            {task.task_id: task for task in tasks}, [])), ['F', 'B', 'OPT'])
        self.assertEqual(analyzer.get_critical_path_length(), 54)
        self.assertEqual(analyzer.get_task_slack('COMM'), 12)

    def test_configuration_specific_cost(self):
        task = Task('F', 0, TaskType.FORWARD_COMPUTE)
        self.measured(task, 8)
        self.assertEqual(self.costs.estimate(task, self.config).estimated_ms, 8)
        with self.assertRaises(MissingCalibration):
            self.costs.estimate(task, self.config.with_updates(tp=2))

    def test_full_iteration_includes_optimizer(self):
        tasks = [Task('F', 0, TaskType.FORWARD_COMPUTE),
                 Task('B', 0, TaskType.BACKWARD_COMPUTE, dependencies=['F']),
                 Task('DP', 0, TaskType.DP_REDUCE, dependencies=['B']),
                 Task('OPT', 0, TaskType.OPTIMIZER_STEP, dependencies=['DP'])]
        for task, ms in zip(tasks, (8, 16, 4, 30)):
            self.measured(task, ms)
        result = ScheduleCostEvaluator(self.costs).evaluate(self.plan(tasks, ['F', 'B', 'DP', 'OPT']), self.hardware)
        self.assertEqual(result.predicted_iteration_time_ms, 58)
        self.assertEqual(result.critical_path_tasks, ['F', 'B', 'DP', 'OPT'])

    def test_missing_backward_is_not_implicitly_zero(self):
        task = Task('B', 0, TaskType.BACKWARD_COMPUTE)
        with self.assertRaises(MissingCalibration):
            ScheduleCostEvaluator(self.costs).evaluate(self.plan([task], ['B']), self.hardware)

    def test_pair_uses_measured_contention(self):
        forward = Task('F', 0, TaskType.FORWARD_COMPUTE)
        communication = Task('C', 0, TaskType.PP_P2P_SEND)
        self.measured(forward, 8)
        self.measured(communication, 4)
        self.costs.update_pair(forward, communication, self.config, 8, 4, 10, '配对数学验证')
        plan = self.plan([forward, communication], ['F', 'C'])
        plan.concurrent_pairs = [('F', 'C')]
        result = ScheduleCostEvaluator(self.costs).evaluate(plan, self.hardware)
        self.assertEqual(result.predicted_iteration_time_ms, 10)
        self.assertEqual(result.exposed_communication_ms, 2)
        self.assertEqual(result.measured_pair_savings_ms, 2)
        self.assertEqual(self.costs.estimate_pair(forward, communication, self.config).benefit_ms, 2)

    def test_pair_can_be_slower_than_serial(self):
        forward = Task('F', 0, TaskType.FORWARD_COMPUTE)
        communication = Task('C', 0, TaskType.PP_P2P_SEND)
        self.costs.update_pair(forward, communication, self.config, 8, 4, 15, '配对数学验证')
        self.assertEqual(self.costs.estimate_pair(forward, communication, self.config).benefit_ms, -3)

    def test_scheduler_selects_pair_by_complete_iteration_cost(self):
        from core_data_models import ExecutionDAG
        forward = Task('F', 0, TaskType.FORWARD_COMPUTE)
        communication = Task('C', 0, TaskType.PP_P2P_SEND)
        optimizer = Task('OPT', 0, TaskType.OPTIMIZER_STEP, dependencies=['F', 'C'])
        for task, milliseconds in ((forward, 8), (communication, 4), (optimizer, 30)):
            self.measured(task, milliseconds)
        self.costs.update_pair(forward, communication, self.config, 8, 4, 10, '完整迭代配对数学验证')
        dag = ExecutionDAG({task.task_id: task for task in (forward, communication, optimizer)}, [])
        plan = MicrobatchScheduler(self.costs).schedule(dag, self.config,
            memory_budget_mb=32 * 1024, persistent_bytes={0: 1000},
            pair_candidates=[('F', 'C')], topology=self.hardware)
        self.assertEqual(plan.concurrent_pairs, [('F', 'C')])
        result = ScheduleCostEvaluator(self.costs).evaluate(plan, self.hardware)
        self.assertEqual(result.predicted_iteration_time_ms, 40)

    def test_missing_measurement_report_requires_actual_plan(self):
        task = Task('F', 0, TaskType.FORWARD_COMPUTE)
        self.assertTrue(self.costs.get_missing_measurements()['requirements_not_provided'])
        report = self.costs.get_missing_measurements(self.plan([task], ['F']))
        self.assertEqual(report['unmeasured_tasks'], ['F'])

    def test_scheduler_preserves_tp_collective_order_with_different_rank_costs(self):
        from core_data_models import ExecutionDAG
        self.config = ParallelConfig(tp=2, pp=1, dp=1, global_batch_size=2)
        tasks = {}
        for rank in range(2):
            for mb in range(2):
                task = Task(f'F:{rank}:{mb}', rank, TaskType.FORWARD_COMPUTE,
                            microbatch_id=mb, region_id=0)
                self.measured(task, 100 if mb == rank else 1)
                tasks[task.task_id] = task
        plan = MicrobatchScheduler(self.costs).schedule(ExecutionDAG(tasks, []), self.config,
            memory_budget_mb=1024, persistent_bytes={0: 1000, 1: 1000})
        self.assertEqual([tasks[key].microbatch_id for key in plan.rank_schedules[0]],
                         [tasks[key].microbatch_id for key in plan.rank_schedules[1]])

    def test_peak_tracks_released_activation(self):
        forward = Task('F', 0, TaskType.FORWARD_COMPUTE, activation_delta_bytes=2000)
        backward = Task('B', 0, TaskType.BACKWARD_COMPUTE, dependencies=['F'], activation_delta_bytes=-2000)
        self.measured(forward, 8)
        self.measured(backward, 16)
        plan = self.plan([forward, backward], ['F', 'B'])
        result = ScheduleCostEvaluator(self.costs).evaluate(plan, self.hardware)
        self.assertEqual(result.rank_peak_bytes[0], 3000)

    def test_cycle_and_incomplete_schedule_fail(self):
        task = Task('F', 0, TaskType.FORWARD_COMPUTE, dependencies=['F'])
        with self.assertRaises(ValueError):
            ScheduleCostEvaluator(self.costs).evaluate(self.plan([task], ['F']), self.hardware)
        with self.assertRaises(ValueError):
            ScheduleCostEvaluator(self.costs).evaluate(self.plan([task], []), self.hardware)

    def test_ep_is_not_world_size_multiplier(self):
        config = ParallelConfig(tp=2, pp=4, dp=2, ep=2)
        self.assertEqual(config.world_size, 16)
        self.assertEqual(config.num_microbatches, 8)
        with self.assertRaises(ValueError):
            ParallelConfig(tp=2, pp=4, dp=2, global_batch_size=15)

    def test_vpp_preserves_wrap_boundary_and_complete_update(self):
        model = ModelSpec('实际层数量数学验证', 52, 2688, 32,
            [LayerType.MAMBA] * 52, list(range(52)), num_experts=128)
        config = ParallelConfig(tp=2, pp=8, dp=1, vpp=2, microbatch_group=8)
        regions = RegionPartitioner(model, config).generate_uniform_regions()
        validator = ParallelValidator(model, HardwareTopology(16, 2, 8, 32))
        self.assertTrue(validator.validate_region_layout(config, regions)[0])
        dag = TaskGraphBuilder().build(model, config, regions, 16)
        self.assertIn(('F:0:0:7', 'SF:0:0:8'), dag.edges)
        self.assertIn(('RF:0:0:8', 'F:0:0:8'), dag.edges)
        self.assertIn(('B:0:0:8', 'SB:0:0:7'), dag.edges)
        self.assertEqual(dag.tasks['F:0:0:8'].metadata['input_task'], 'RF:0:0:8')
        self.assertEqual(dag.tasks['B:0:0:8'].metadata['input_task'], 'RB:0:0:8')
        self.assertNotIn('input_task', dag.tasks['F:0:0:0'].metadata)
        self.assertNotIn('input_task', dag.tasks['B:0:0:15'].metadata)
        self.assertEqual(sum(task.task_type == TaskType.OPTIMIZER_STEP for task in dag.tasks.values()), 16)
        self.assertEqual(sum(task.task_type == TaskType.FORWARD_COMPUTE for task in dag.tasks.values()), 512)

    def test_nonuniform_partition_keeps_contiguous_layers(self):
        model = ModelSpec('分区数学验证', 8, 64, 8, [LayerType.MLP] * 8, list(range(8)))
        config = ParallelConfig(tp=1, pp=2, dp=1, global_batch_size=4)
        partitioner = RegionPartitioner(model, config)
        partitioner.set_layer_costs({index: 10 if index == 0 else 1 for index in range(8)})
        regions = partitioner.generate_load_balanced_regions()
        self.assertEqual([region.num_layers for region in regions], [1, 7])
        self.assertEqual(regions[0].layer_end, regions[1].layer_start)


if __name__ == '__main__':
    unittest.main()
