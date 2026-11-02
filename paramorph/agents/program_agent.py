from dataclasses import asdict
import hashlib
import json

from calibration_plan_compiler import CalibrationPlanCompiler
from scheduling.native_microbatches import native_schedule_table


class ProgramAgent:
    def __init__(self, decision_service=None):
        self.decision_service = decision_service

    @staticmethod
    def lower_native(program, reference_plan):
        if program['executor'] not in ('megatron_interleaved_table', 'megatron_1f1b'):
            raise ValueError('原生入口不能执行未完成数值认证的自研任务计划')
        config = program['config']
        mapping = config['device_mapping']
        overrides = program.get('native_execution_overrides', {})
        plan = dict(reference_plan)
        if 'backend_request' in program:
            plan.update(program['backend_request'])
        plan.update(tp=config['tp'], cp=config['cp'], pp=config['pp'], ep=config['ep'], etp=config['etp'],
            vpp=config['vpp'], group=config['microbatch_group'],
            decoder_chunks=program['decoder_chunks'],
            micro_batch=config['micro_batch_size'], global_batch=config['global_batch_size'],
            sequence=config['sequence_length'])
        plan.update({key: value for key, value in overrides.items() if value is not None})
        plan['dp'] = config['dp']
        plan['expert_dp'] = config['tp'] * config['cp'] * config['dp'] // (config['etp'] * config['ep'])
        if mapping:
            plan['rank_mapping'] = list(mapping)
        if config['vpp'] == 1 and 'overlap_p2p' not in overrides:
            plan['overlap_p2p'] = False
        world = plan.get('world', plan.get('node_count', 1) * plan.get('gpus_per_node', config['tp'] * config['cp'] * config['pp'] * config['dp']))
        if world != config['tp'] * config['cp'] * config['pp'] * config['dp']:
            raise ValueError('原生程序GPU数量不一致')
        return plan

    def compile(self, candidate, execution_plan=None):
        if not candidate.is_valid or candidate.region_layout is None:
            raise ValueError('ProgramAgent只能编译合法完整候选')
        config = candidate.parallel_config
        if execution_plan is None:
            execution_plan = candidate.execution_plan
        chunks = [region.num_layers for region in candidate.region_layout]
        group = config.microbatch_group
        table = []
        if config.vpp > 1:
            table = native_schedule_table(config.num_microbatches, config.pp, config.vpp, group)
        program = dict(candidate_id=candidate.candidate_id, config=asdict(config),
            decoder_chunks=chunks, schedule_table=table,
            executor='megatron_interleaved_table' if config.vpp > 1 else 'megatron_1f1b',
            scope='region_layout_and_microbatch_chunk_table',
            unsupported_actions=['arbitrary_forward_backward_order', 'independent_cp_ep_kernel_reordering'])
        if candidate.transformation_record:
            program['transformation_record'] = candidate.transformation_record
            program['memory_admission'] = 'requires_actual_measurement'
            program['native_execution_overrides'] = dict(config.execution_implementation)
        if execution_plan is not None:
            if execution_plan.parallel_config != config or execution_plan.regions != candidate.region_layout:
                raise ValueError('自研执行计划的配置或者Region与候选不同')
            if not execution_plan.task_times or not execution_plan.cost_model_version:
                raise ValueError('自研执行计划必须完成完整成本评价')
            tasks = {}
            for key, task in execution_plan.tasks.items():
                tasks[key] = dict(rank=task.rank, kind=task.task_type.value,
                    microbatch=task.microbatch_id, region_id=task.region_id,
                    dependencies=task.dependencies, metadata=task.metadata)
            program.update(executor='paramorph_task_dag', scope='complete_microbatch_communication_execution_plan',
                unsupported_actions=['independent_cp_ep_kernel_reordering'],
                execution_plan=dict(executor='paramorph_task_dag',
                    config=asdict(config), num_microbatches=config.num_microbatches,
                    vpp=config.vpp, num_regions=config.total_vp_stages,
                    regions=[asdict(region) for region in candidate.region_layout],
                    tasks=tasks, rank_schedules={str(rank): sequence for rank, sequence
                        in execution_plan.rank_schedules.items()},
                    concurrent_pairs=execution_plan.concurrent_pairs,
                    predicted_task_times=execution_plan.task_times,
                    persistent_bytes=execution_plan.persistent_bytes,
                    cost_model_version=execution_plan.cost_model_version))
            compiler = CalibrationPlanCompiler()
            program['execution_plan'] = compiler.compile_async_publication(program['execution_plan'], receive_window=2)
            program['execution_plan'] = compiler.compile(program['execution_plan'])
        program['program_sha256'] = hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
        if execution_plan is not None:
            candidate.execution_plan_sha256 = hashlib.sha256(json.dumps(
                program['execution_plan'], sort_keys=True).encode()).hexdigest()
        if self.decision_service is not None:
            summary = dict(candidate_id=program['candidate_id'], config=program['config'],
                decoder_chunks=program['decoder_chunks'], executor=program['executor'],
                scope=program['scope'], unsupported_actions=program['unsupported_actions'],
                program_sha256=program['program_sha256'])
            if execution_plan is not None:
                summary.update(task_count=len(execution_plan.tasks),
                    concurrent_pairs=execution_plan.concurrent_pairs,
                    rank_schedule_prefixes={str(rank): sequence[:12]
                        for rank, sequence in execution_plan.rank_schedules.items()},
                    cost_model_version=execution_plan.cost_model_version,
                    predicted_makespan_ms=max(end for start, end in execution_plan.task_times.values()))
            program['agent_decision'] = self.decision_service('program', dict(
                execution_program=summary, candidates=[candidate.to_dict()]))
        return program
