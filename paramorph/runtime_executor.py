from contextlib import ExitStack
import faulthandler
import hashlib
import json
import math
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import schedules
from megatron.core.utils import get_model_config
from communication_executor import CommunicationExecutor
from schedule_validator import ScheduleValidator
from progress_validator import DistributedProgressValidator


class PlanExecutor:
    def __init__(self, evidence, plan):
        self.evidence = evidence
        self.plan = plan
        self.communication = None
        self.last_task_snapshot = 0.0
        self.digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        self.validate()

    def validate(self):
        tasks = self.plan['tasks']
        if self.plan['executor'] != 'paramorph_task_dag':
            raise ValueError('自研执行器需要完整Task DAG计划')
        for task in tasks.values():
            if task['kind'] not in ('forward_compute', 'backward_compute',
                    'pp_p2p_send', 'pp_p2p_recv', 'dp_reduce', 'optimizer_step', 'cuda_sync'):
                raise ValueError('ExecutionPlan包含未实现的执行任务')
        ScheduleValidator().validate(self.plan)
        if not DistributedProgressValidator().validate(self.plan)['valid']:
            raise ValueError('实际训练计划未通过分布式资源进展验证')
        if 'serial_calibration_rank_schedules' in self.plan:
            ScheduleValidator().validate(self.plan, 'serial_calibration_rank_schedules', blocking_send=True)

    def __call__(self, *, forward_step_func, data_iterator, model, num_microbatches,
                 seq_length, micro_batch_size, decoder_seq_length=None, forward_only=False,
                 collect_non_loss_data=False, first_val_step=None, adjust_tensor_shapes_fn=None,
                 p2p_communicator=None, pg_collection=None, force_all_reduce=False):
        if forward_only or collect_non_loss_data or adjust_tensor_shapes_fn is not None:
            raise ValueError('当前完整训练计划要求Forward与Backward共同执行')
        if num_microbatches != self.plan['num_microbatches'] or len(model) != self.plan['vpp']:
            raise ValueError('实际训练的Microbatch或者VPP与计划不同')
        if pg_collection is None:
            pg_collection = schedules._build_default_pg_collection()
        config = get_model_config(model[0])
        if self.communication is None:
            maximum = max(math.prod(task['metadata']['shape'])
                          for task in self.plan['tasks'].values() if task['kind'] == 'pp_p2p_send')
            self.evidence.communication_initialization_payload = next(parameter
                for chunk in model for parameter in chunk.parameters() if parameter.numel() >= maximum)
            self.communication = CommunicationExecutor(self.evidence, self.plan, config.pipeline_dtype)
            self.communication.initialize()
        if config.defer_embedding_wgrad_compute or getattr(config, 'overlap_moe_expert_parallel_comm', False):
            raise ValueError('当前Task执行器要求完整Backward执行路径')
        tasks = self.plan['tasks']
        serial = self.evidence.request.get('paramorph_calibration', False) and (
            self.evidence.step == 0 or self.evidence.request['warmup'] <= self.evidence.step
            <= self.evidence.request.get('paramorph_serial_calibration_steps', 0))
        schedule_key = 'serial_calibration_rank_schedules' if serial else 'rank_schedules'
        sequence = self.plan[schedule_key][str(dist.get_rank())]
        self.communication.begin_iteration(serial, schedule_key)
        if self.evidence.request.get('paramorph_calibration', False):
            self.thread_log = (self.evidence.output / f'rank{dist.get_rank()}.paramorph-threads.log').open('a')
            faulthandler.dump_traceback_later(45, repeat=True, file=self.thread_log)
        buffers, activations = {}, {}
        forward_data_store = []
        total_num_tokens = torch.zeros([], dtype=torch.int, device='cuda')
        self.evidence.external_activation_storages = set()
        sync_functions = config.param_sync_func
        if sync_functions is not None:
            functions = sync_functions if isinstance(sync_functions, list) else [sync_functions] * len(model)
            for chunk, function in zip(model, functions):
                function(chunk.parameters())
        actual = []
        with ExitStack() as contexts:
            no_sync = config.no_sync_func
            if no_sync is not None:
                functions = no_sync if isinstance(no_sync, list) else [no_sync]
                for function in functions:
                    contexts.enter_context(function())
            for key in sequence:
                task = tasks[key]
                kind, metadata = task['kind'], task['metadata']
                if kind == 'optimizer_step':
                    break
                identity = dict(task_id=key, plan_sha256=self.digest,
                    microbatch=task.get('microbatch'), region_id=task.get('region_id'))
                self.evidence.active_plan_task = key
                if (self.evidence.request.get('paramorph_calibration', False)
                    and time.monotonic() - self.last_task_snapshot >= 1):
                    (self.evidence.output / f'rank{dist.get_rank()}.paramorph-active-task.json').write_text(
                        json.dumps(dict(step=self.evidence.step, task_id=key, kind=kind,
                                        schedule_key=schedule_key), indent=2) + '\n')
                    self.last_task_snapshot = time.monotonic()
                vp = metadata.get('vp_stage')
                if vp is not None:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(vp)
                if kind == 'pp_p2p_recv':
                    self.communication.publish_receive(key)
                elif kind == 'forward_compute':
                    mb = task['microbatch']
                    received = metadata.get('input_task')
                    input_tensor = self.communication.consume_receive(received, key) if received is not None else None
                    self.evidence.external_activation_storages = (
                        {input_tensor.untyped_storage().data_ptr()} if input_tensor is not None else set())
                    output, tokens = schedules.forward_step(forward_step_func, data_iterator[vp],
                        model[vp], num_microbatches, input_tensor, forward_data_store, config,
                        cp_group_size=pg_collection.cp.size(), collect_non_loss_data=False,
                        is_first_microbatch=mb == 0, current_microbatch=mb, vp_stage=vp,
                        is_last_stage=task['region_id'] == self.plan['num_regions'] - 1)
                    activations[(vp, mb)] = (input_tensor, output, received)
                    buffers[key] = output
                    total_num_tokens += tokens
                elif kind == 'backward_compute':
                    mb = task['microbatch']
                    input_tensor, output, forward_receive = activations.pop((vp, mb))
                    self.evidence.external_activation_storages = set()
                    received = metadata.get('input_task')
                    gradient = self.communication.consume_receive(received, key) if received is not None else None
                    buffers[key] = schedules.backward_step(input_tensor, output, gradient, config)
                    self.communication.release_receive(received)
                    self.communication.release_receive(forward_receive)
                elif kind == 'pp_p2p_send':
                    self.communication.publish_send(key, buffers.pop(metadata['producer_task']))
                    producer = tasks[metadata['producer_task']]
                    if metadata['direction'] == 'forward':
                        output = activations[(producer['metadata']['vp_stage'], producer['microbatch'])][1]
                        schedules.deallocate_output_tensor(output, config.deallocate_pipeline_outputs)
                elif kind == 'dp_reduce':
                    self.communication.finish_iteration()
                    contexts.close()
                    grad_sync = config.grad_sync_func
                    if grad_sync is not None:
                        functions = grad_sync if isinstance(grad_sync, list) else [grad_sync] * len(model)
                        for chunk, function in zip(model, functions):
                            function(chunk.parameters())
                    config.finalize_model_grads_func(model,
                        total_num_tokens if config.calculate_per_token_loss else None,
                        pg_collection=pg_collection, force_all_reduce=force_all_reduce)
                elif kind == 'cuda_sync':
                    torch.cuda.synchronize()
                actual.append(key)
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        self.evidence.active_plan_task = None
        if activations:
            raise ValueError('ExecutionPlan结束后仍存在未完成的计算或者接收任务')
        self.pending_iteration = dict(step=self.evidence.step, plan_sha256=self.digest,
            task_order=actual, planned_order=sequence)
        if self.evidence.request.get('paramorph_calibration', False):
            faulthandler.cancel_dump_traceback_later()
            self.thread_log.close()
        return forward_data_store

    def finish_optimizer(self):
        actual = self.pending_iteration['task_order']
        sequence = self.pending_iteration['planned_order']
        for key in sequence[len(actual):]:
            kind = self.plan['tasks'][key]['kind']
            if kind == 'cuda_sync':
                self.evidence.active_plan_task = key
                self.evidence.record(kind, 'post_optimizer_sync', torch.cuda.synchronize, (), {}, dict(task_id=key))
            elif kind != 'optimizer_step':
                raise ValueError('Optimizer之后包含未执行的训练任务')
            actual.append(key)
        self.pending_iteration['matches_plan'] = actual == sequence
        self.evidence.executed_plans.append(self.pending_iteration)
        self.evidence.active_plan_task = None
        self.pending_iteration = None
