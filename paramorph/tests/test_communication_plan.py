from copy import deepcopy
from graphlib import CycleError
import unittest

from calibration_plan_compiler import CalibrationPlanCompiler
from core_data_models import LayerType, ModelSpec, ParallelConfig
from region_partition.region_partitioner import RegionPartitioner
from scheduling.task_graph_builder import TaskGraphBuilder
from schedule_validator import ScheduleValidator


class CommunicationPlanTests(unittest.TestCase):
    def protocol(self):
        tasks = {}
        for source, destination in ((0, 1), (1, 0)):
            send, receive, consume = [f'{role}{source}' for role in ('S', 'R', 'C')]
            metadata = dict(channel=f'forward:{source}', group_ranks=[source, destination],
                            direction='forward', shape=[8, 8], dtype='bfloat16')
            tasks[send] = dict(kind='pp_p2p_send', rank=source, region_id=destination,
                              microbatch=0, dependencies=[], metadata=dict(metadata))
            tasks[receive] = dict(kind='pp_p2p_recv', rank=destination, region_id=destination,
                                 microbatch=0, dependencies=[send], metadata=dict(metadata))
            tasks[consume] = dict(kind='communication_consume', rank=destination,
                region_id=destination, microbatch=0, dependencies=[receive],
                metadata=dict(input_task=receive, direction='forward'))
        return dict(tasks=tasks, config=dict(tp=1, pp=2), num_regions=2,
                    rank_schedules={'0': ['S0', 'R1', 'C1'], '1': ['S1', 'R0', 'C0']})

    def test_blocking_send_cycle_is_rejected(self):
        plan = self.protocol()
        self.assertTrue(ScheduleValidator().validate(plan)['valid'])
        with self.assertRaises(CycleError):
            ScheduleValidator().validate(plan, blocking_send=True)

    def test_communication_shape_mismatch_is_rejected(self):
        plan = self.protocol()
        plan['tasks']['R0']['metadata']['shape'] = [4, 8]
        with self.assertRaises(ValueError):
            ScheduleValidator().validate(plan)

    def test_consume_before_receive_is_rejected(self):
        plan = self.protocol()
        plan['rank_schedules']['0'] = ['S0', 'C1', 'R1']
        with self.assertRaises(ValueError):
            ScheduleValidator().validate(plan)

    def test_serial_and_bounded_publication_share_validator(self):
        model = ModelSpec('执行计划数学验证', 4, 64, 8, [LayerType.ATTENTION] * 4, list(range(4)))
        config = ParallelConfig(tp=1, pp=2, dp=1, vpp=2, global_batch_size=4, microbatch_group=2)
        regions = RegionPartitioner(model, config).generate_uniform_regions()
        dag = TaskGraphBuilder().build(model, config, regions, 4)
        plan = dict(config=dict(tp=1, pp=2, cp=1, dp=1), vpp=2, num_microbatches=4,
                    num_regions=4, tasks={key: dict(rank=task.rank, kind=task.task_type.value,
                    microbatch=task.microbatch_id, region_id=task.region_id,
                    dependencies=list(task.dependencies), metadata=deepcopy(task.metadata))
                    for key, task in dag.tasks.items()})
        compiler = CalibrationPlanCompiler()
        serial = compiler.compile(plan)
        serial['rank_schedules'] = serial['serial_calibration_rank_schedules']
        asynchronous = compiler.compile_async_publication(serial, receive_window=2)
        self.assertTrue(asynchronous['asynchronous_communication_legality']['valid'])
        for report in asynchronous['distributed_progress_validation']['interleavings']:
            completed = {event['task_id']: event['index'] for event in report['events']
                if event['event'] in ('TASK_COMPLETED', 'COLLECTIVE_COMPLETED')}
            reductions = [event for event in report['events']
                if event['event'] == 'COLLECTIVE_COMPLETED'
                and event['task_id'].startswith('ONORM:AR:')]
            self.assertEqual(len(reductions), 1)
            reduction = reductions[0]
            self.assertEqual(reduction['participants'], [0, 1])
            for rank in reduction['participants']:
                self.assertLess(completed[f'OPREP:{rank}'], completed[f'ONORM:{rank}'])
                self.assertLess(completed[f'ONORM:{rank}'], reduction['index'])
                self.assertLess(reduction['index'], completed[f'ONORM_POST:{rank}'])
                self.assertLess(completed[f'ONORM_POST:{rank}'], completed[f'OCLIP:{rank}'])
                self.assertLess(completed[f'OCLIP:{rank}'], completed[f'OPT:{rank}'])
        self.assertNotEqual(serial['rank_schedules'], asynchronous['rank_schedules'])
        self.assertTrue(ScheduleValidator().validate(serial,
            'serial_calibration_rank_schedules', blocking_send=True)['valid'])
        requested = deepcopy(serial)
        for rank, sequence in requested['rank_schedules'].items():
            receives = [key for key in sequence if requested['tasks'][key]['kind'] == 'pp_p2p_recv']
            remaining = [key for key in sequence if requested['tasks'][key]['kind'] != 'pp_p2p_recv']
            requested['rank_schedules'][rank] = receives + remaining
        bounded = compiler.compile_async_publication(requested, receive_window=2)
        self.assertTrue(bounded['asynchronous_communication_legality']['valid'])
        for rank in serial['rank_schedules']:
            self.assertEqual(
                [key for key in serial['rank_schedules'][rank]
                 if serial['tasks'][key]['kind'] != 'pp_p2p_recv'],
                [key for key in bounded['rank_schedules'][rank]
                 if bounded['tasks'][key]['kind'] != 'pp_p2p_recv'])


if __name__ == '__main__':
    unittest.main()
