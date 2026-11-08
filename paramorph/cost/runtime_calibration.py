from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import statistics

from core_data_models import ParallelConfig, MemoryPolicy, Task, TaskType, ExecutionPlan
from cost.cost_model import MissingCalibration
from scheduling.native_schedule_lowerer import verify_native_execution_events


def configuration(request):
    precision = request.get('precision', 'bfloat16')
    if precision not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('训练精度需要使用bfloat16、float16或者float32')
    request = dict(request, precision=precision)
    if request['recompute'] not in ('none', 'full', 'selective-mlp', 'selective'):
        raise ValueError('重计算策略没有对应的原生执行路径')
    if 'bucket_elements' in request and (type(request['bucket_elements']) is not int
                                       or request['bucket_elements'] <= 0):
        raise ValueError('梯度Bucket容量必须为正整数')
    if 'overlap_p2p' in request and type(request['overlap_p2p']) is not bool:
        raise ValueError('P2P重叠设置必须为布尔值')
    memory = request.get('memory_policy')
    if request['recompute'] == 'selective' and (memory is None
            or memory.get('recompute_granularity') != 'selective'
            or not memory.get('recompute_modules')):
        raise ValueError('原生选择性重计算必须显式提供selective粒度及具体模块')
    policy = MemoryPolicy(**memory) if memory is not None else MemoryPolicy(
        optimizer_offload_fraction=1.0 if request['offload'] else 0.0,
        recompute_granularity=('full' if request['recompute'] == 'full' else
                              'selective' if request['recompute'] == 'selective-mlp' else None),
        recompute_modules=('mlp',) if request['recompute'] == 'selective-mlp' else ())
    return ParallelConfig(tp=request['tp'], pp=request['pp'], dp=request['dp'],
        cp=request['cp'], ep=request['ep'], vpp=request['vpp'],
        etp=request.get('etp', request['tp']),
        device_mapping=tuple(request.get('rank_mapping', ())),
        global_batch_size=request['global_batch'], micro_batch_size=request['micro_batch'],
        sequence_length=request['sequence'], microbatch_group=request['group'],
        use_sequence_parallel=request.get('sequence_parallel', request['tp'] > 1),
        optimizer_offload=request['offload'],
        memory_policy=policy,
        use_activation_recomputation=request['recompute'] == 'full',
        execution_implementation={name: request.get(name) for name in
            ('model', 'recompute', 'mlp_recompute_every', 'fused_cross_entropy',
             'attention_backend', 'gradient_path', 'precision', 'cross_entropy_impl',
             'cuda_device_max_connections', 'communication_executor_protocol',
             'overlap_p2p', 'batch_p2p_comm', 'overlap_p2p_comm_warmup_flush',
             'pipeline_schedule', 'optimizer_sharding', 'bucket_elements',
             'precision_aware_optimizer', 'execution_backend', 'fsdp_sharding_strategy',
             'dense_sharding_strategy', 'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size',
             'expert_dp_inner_size', 'expert_dp_outer_size', 'outer_sharding_strategy',
             'expert_outer_sharding_strategy', 'fsdp_hybrid_outer_supported',
             'fsdp_multi_chunk_supported', 'fsdp_optimizer_checkpoint_supported',
             'fsdp_precision_aware_optimizer_supported',
             'fsdp_separate_expert_outer_mesh_supported', 'fsdp_supported_sharding_strategies',
             'fsdp_supported_expert_sharding_strategies', 'fsdp_supported_units',
             'megatron_fsdp_version', 'megatron_revision',
             'megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype',
             'fsdp_unit', 'fsdp_prefetch', 'fsdp_prefetch_distance', 'fsdp_reshard_after_forward',
             'fsdp_double_buffer', 'fsdp_bucket_elements', 'checkpoint_format', 'checkpoint_reshard',
             'optimizer_checkpoint_mode', 'main_grads_dtype', 'grad_comm_dtype',
             'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype',
             'overlap_grad_reduce', 'pad_buckets_for_high_nccl_busbw',
             'optimizer_offload_fraction', 'optimizer_transfer_overlap',
             'clip_grad', 'num_distributed_optimizer_instances',
             'position_embedding_group_policy')
             if name not in ('megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype')
             or request.get(name) is not None})


class RuntimeCalibration:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.paths = sorted(self.directory.glob('rank*.paramorph-cost.json'))
        if not self.paths:
            raise MissingCalibration('缺少实际GPU运行产生的成本文件')
        self.payloads = [json.loads(path.read_text()) for path in self.paths]
        self.config = configuration(self.payloads[0]['request'])
        if sorted(item['rank'] for item in self.payloads) != list(range(self.config.world_size)):
            raise ValueError('成本文件必须覆盖全部训练rank一次')
        if any(configuration(item['request']) != self.config for item in self.payloads):
            raise ValueError('成本文件来自不同并行配置')
        if any(not item['forward_backward_coverage_verified'] for item in self.payloads):
            raise ValueError('实际执行未覆盖完整Forward和Backward')
        request = self.payloads[0]['request']
        self.native_execution_events = (verify_native_execution_events(request, self.payloads)
            if request.get('record_native_schedule') and not request.get('paramorph_execution_plan')
            and not request.get('paramorph_schedule_table') else None)
        self.steps = {}
        self.tasks = {}

    def ingest(self, costs, held_out_steps=1):
        if held_out_steps < 1:
            raise ValueError('必须保留独立迭代用于检验预测误差')
        for path, payload in zip(self.paths, self.payloads):
            rank = payload['rank']
            request = payload['request']
            evidence = hashlib.sha256(path.read_bytes()).hexdigest()
            events = defaultdict(list)
            for event in payload['events']:
                events[event['step']].append(event)
            windows = [window for window in payload['windows']
                       if window['step'] >= request['warmup']]
            # Profiler迭代仅保留分析用途，不参与成本校准和误差检验。
            if request.get('paramorph_trace'):
                windows = windows[:-1]
            if len(windows) <= held_out_steps:
                raise MissingCalibration('排除预热和Profiler后，校准与独立检验迭代数量不足')
            rank_steps = []
            for window in windows:
                phase_events = [event for event in events[window['step']]
                    if event['kind'] in ('pipeline', 'optimizer_step')]
                if [event['kind'] for event in phase_events] != ['pipeline', 'optimizer_step']:
                    raise ValueError('完整迭代必须包含一次流水线更新和一次Optimizer更新')
                duration = (window['host_end_ns'] - window['host_start_ns']) / 1e6
                phase_ms = [(event['host_end_ns'] - event['host_start_ns']) / 1e6
                            for event in phase_events]
                residual = duration - sum(phase_ms)
                if residual < 0:
                    raise ValueError('迭代阶段的计时范围相互重叠')
                rank_steps.append(dict(step=window['step'], iteration_ms=duration,
                    pipeline_ms=phase_ms[0], optimizer_ms=phase_ms[1], overhead_ms=residual,
                    persistent_bytes=window['persistent_bytes'],
                    peak_allocated_bytes=window.get('peak_allocated_bytes')))
            self.steps[rank] = rank_steps
            tasks = []
            for task_type, operation, field in (
                (TaskType.PIPELINE_UPDATE, 'pipeline_update', 'pipeline_ms'),
                (TaskType.OPTIMIZER_STEP, 'optimizer_step', 'optimizer_ms'),
                (TaskType.ITERATION_OVERHEAD, 'iteration_overhead', 'overhead_ms')):
                task = Task(f'{operation}:{rank}', rank, task_type,
                    metadata=dict(operation=operation, rank_class=rank,
                        layer_ids=request['decoder_chunks'],
                        includes=['forward', 'backward', 'tp', 'cp', 'ep', 'pp', 'gradient_finalize']
                        if task_type == TaskType.PIPELINE_UPDATE else [operation]))
                tasks.append(task)
                for row in rank_steps[:-held_out_steps]:
                    costs.update(dict(task=task, config=self.config, milliseconds=row[field],
                        source='gpu_training_host_wall_phase',
                        evidence_id=f'{evidence}:rank{rank}:step{row["step"]}'))
            self.tasks[rank] = tasks
            for event in payload['events']:
                if event['step'] not in {row['step'] for row in rank_steps[:-held_out_steps]}:
                    continue
                if event['kind'] not in ('forward_compute', 'backward_compute'):
                    continue
                if event['cuda_ms'] is None:
                    raise MissingCalibration('Region成本缺少CUDA事件计时')
                vp = event['identity']['vp_stage']
                index = vp * self.config.pp + payload['pp_rank']
                counts = request['decoder_chunks']
                first = sum(counts[:index])
                task = Task(f'region:{rank}:{vp}:{event["kind"]}', rank,
                    TaskType(event['kind']), metadata=dict(operation='region', rank_class=rank,
                    layer_ids=list(range(first, first + counts[index])), includes=['tp', 'cp', 'ep']))
                costs.update(dict(task=task, config=self.config, milliseconds=event['cuda_ms'],
                    source='gpu_training_cuda_event', evidence_id=f'{evidence}:{event["step"]}:{task.task_id}'))
            for event in payload['events']:
                if event['step'] not in {row['step'] for row in rank_steps[:-held_out_steps]}:
                    continue
                if event['kind'] not in ('pp_p2p_exchange', 'dp_reduce'):
                    continue
                identity = event['identity']
                if event['kind'] == 'pp_p2p_exchange' and not identity['completion_waited']:
                    continue
                if event['cuda_ms'] is None:
                    raise MissingCalibration('通信完成成本缺少CUDA事件计时')
                task = Task(f'{event["operation"]}:{rank}', rank, TaskType(event['kind']),
                    metadata=dict(identity, operation=event['operation'], rank_class=rank,
                                  includes=['peer_readiness_wait']))
                costs.update(dict(task=task, config=self.config, milliseconds=event['cuda_ms'],
                    source='gpu_training_communication_completion_including_peer_wait',
                    evidence_id=f'{evidence}:{event["step"]}:{event["host_start_ns"]}'))
            for event in payload['events']:
                if event['step'] not in {row['step'] for row in rank_steps[:-held_out_steps]}:
                    continue
                if event['kind'] != 'layer_compute':
                    continue
                if event['cuda_ms'] is None:
                    raise MissingCalibration('Layer成本缺少CUDA事件计时')
                task = Task(f'layer:{rank}:{event["identity"]["layer_id"]}', rank,
                    TaskType.FORWARD_COMPUTE, metadata=dict(operation=event['operation'],
                    layer_ids=[event['identity']['layer_id']], rank_class=rank, includes=['tp', 'cp', 'ep']))
                costs.update(dict(task=task, config=self.config, milliseconds=event['cuda_ms'],
                    source='gpu_training_layer_cuda_event',
                    evidence_id=f'{evidence}:{event["step"]}:{event["host_start_ns"]}'))
        predictions = {rank: sum(costs.estimate(task, self.config).estimated_ms
            for task in tasks) for rank, tasks in self.tasks.items()}
        expected = max(predictions.values())
        held_out = [max(self.steps[rank][-offset]['iteration_ms']
            for rank in self.steps) for offset in range(1, held_out_steps + 1)]
        actual = statistics.median(held_out)
        return dict(configuration=asdict(self.config), cost_model_version=costs.version,
            predicted_iteration_time_ms=expected, held_out_iteration_time_ms=actual,
            relative_error=(expected - actual) / actual, rank_predictions_ms=predictions,
            held_out_steps=held_out_steps,
            granularity='complete_iteration_measured_phase',
            data_gaps=['独立PP链路通信成本', '真实计算通信并发配对成本', '跨rank完整关键路径'],
            rank_steps=self.steps, native_execution_events=self.native_execution_events)

    def execution_plan(self, costs, held_out_steps=1):
        if not self.tasks:
            raise ValueError('需要先导入真实成本校准')
        tasks, schedules, persistent = {}, {}, {}
        for rank, phases in self.tasks.items():
            rows = self.steps[rank][:-held_out_steps]
            if any(row['peak_allocated_bytes'] is None for row in rows):
                raise MissingCalibration('完整执行计划缺少每次迭代峰值显存测量')
            persistent[rank] = max(row['persistent_bytes'] for row in rows)
            extra = max(row['peak_allocated_bytes'] - row['persistent_bytes'] for row in rows)
            for index, task in enumerate(phases):
                task.dependencies = [phases[index - 1].task_id] if index else []
                task.workspace_bytes = max(0, extra)
                task.metadata['memory_scope'] = 'measured_complete_iteration_envelope'
                task.metadata['memory_evidence'] = [dict(step=row['step'],
                    persistent_bytes=row['persistent_bytes'], peak_bytes=row['peak_allocated_bytes'])
                    for row in rows]
                tasks[task.task_id] = task
            schedules[rank] = [task.task_id for task in phases]
        return ExecutionPlan(self.config, [], schedules, tasks=tasks,
            persistent_bytes=persistent, cost_model_version=costs.version)
