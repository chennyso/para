import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from megatron.core.datasets.indexed_dataset import IndexedDataset

from calibration_plan_compiler import CalibrationPlanCompiler
from communication_executor import CommunicationExecutor
from schedule_validator import ScheduleValidator


class Evidence:
    def __init__(self, output):
        self.output = output
        self.request = dict(paramorph_calibration=True)
        self.step = 0
        self.active_plan_task = None
        self.events = []
        self.cuda_origins = {}
        self.communication_iterations = []
        self.communication_initialization_records = []

    def record(self, kind, operation, function, args, kwargs, identity):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.events.append(dict(kind=kind, operation=operation, identity=identity,
                                cuda_start=start, cuda_end=end))
        return result


def build_plan(microbatches, shape, pp=2):
    tasks = {}
    for mb in range(microbatches):
        previous = None
        for direction, prefix, regions in (
            ('forward', 'F', range(1, pp * 2)), ('backward', 'B', reversed(range(pp * 2 - 1)))):
            for region in regions:
                source = (region - 1 if prefix == 'F' else region + 1) % pp
                destination = region % pp
                send, receive, consume = [f'{role}{prefix}:0:{mb}:{region}' for role in ('S', 'R', 'C')]
                metadata = dict(channel=f'{prefix}:{region}', group_ranks=[source, destination],
                    shape=shape, dtype='bfloat16', direction=direction, producer_task=previous)
                tasks[send] = dict(kind='pp_p2p_send', rank=source, microbatch=mb,
                    region_id=region, dependencies=[] if previous is None else [previous], metadata=dict(metadata))
                tasks[receive] = dict(kind='pp_p2p_recv', rank=destination, microbatch=mb,
                    region_id=region, dependencies=[send], metadata=dict(metadata))
                tasks[consume] = dict(kind='communication_consume', rank=destination, microbatch=mb,
                    region_id=region, dependencies=[receive], metadata=dict(input_task=receive, direction=direction))
                previous = consume
    sequences = {str(rank): [] for rank in range(pp)}
    for prefix, regions in (('F', range(1, pp * 2)), ('B', reversed(range(pp * 2 - 1)))):
        for region in regions:
            for mb in range(microbatches):
                for role in ('S', 'R', 'C'):
                    key = f'{role}{prefix}:0:{mb}:{region}'
                    sequences[str(tasks[key]['rank'])].append(key)
    return CalibrationPlanCompiler().compile_communication(dict(tasks=tasks, rank_schedules=sequences,
        config=dict(tp=1, cp=1, dp=1, pp=pp), vpp=2, num_regions=pp * 2, num_microbatches=microbatches))


def calibrate_pair(evidence, payload):
    evidence.request['paramorph_progress_snapshots'] = False
    shape = list(payload.shape)
    metadata = dict(channel='calibration:forward:1', group_ranks=[0, 1], shape=shape,
                    dtype='bfloat16', direction='forward')
    tasks = {
        'S': dict(kind='pp_p2p_send', rank=0, microbatch=0, region_id=1,
                  dependencies=[], metadata=dict(metadata)),
        'R': dict(kind='pp_p2p_recv', rank=1, microbatch=0, region_id=1,
                  dependencies=['S'], metadata=dict(metadata)),
        'C': dict(kind='communication_consume', rank=1, microbatch=0, region_id=1,
                  dependencies=['R'], metadata=dict(input_task='R', direction='forward'))}
    sequences = {str(rank): [] for rank in range(dist.get_world_size())}
    sequences.update({'0': ['S'], '1': ['R', 'C']})
    plan = dict(tasks=tasks, rank_schedules=sequences,
                config=dict(tp=1, cp=1, dp=1, pp=2), vpp=1, num_regions=2, num_microbatches=1)
    executor = CommunicationExecutor(evidence, plan, torch.bfloat16)
    executor.initialize()
    matrix = payload.detach() / 128
    output = torch.empty_like(matrix)
    rows = []
    for mode in ('compute', 'communication', 'concurrent'):
        for repetition in range(13):
            evidence.step += 1
            dist.barrier()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            evidence.cuda_origins[evidence.step] = start
            if mode != 'compute':
                executor.begin_iteration(False, 'rank_schedules')
                if dist.get_rank() == 0:
                    executor.publish_send('S', payload)
                elif dist.get_rank() == 1:
                    executor.publish_receive('R')
            if mode != 'communication':
                for _ in range(20):
                    torch.mm(matrix, matrix, out=output)
            if mode != 'compute':
                if dist.get_rank() == 0:
                    executor.wait_send('S', 'pair_measurement_complete')
                elif dist.get_rank() == 1:
                    received = executor.consume_receive('R', 'C')
                    executor.release_receive('R')
            end.record()
            end.synchronize()
            if mode != 'compute':
                executor.finish_iteration()
                if dist.get_rank() == 1 and not torch.eq(received, payload).all().item():
                    raise ValueError('并发校准的实际通信载荷发生变化')
            if mode != 'communication' and not torch.isfinite(output).all().item():
                raise ValueError('真实GEMM校准输出存在非有限数值')
            if repetition >= 3:
                rows.append(dict(mode=mode, repetition=repetition - 3,
                    milliseconds=start.elapsed_time(end), rank=dist.get_rank(),
                    communication_bytes=payload.numel() * payload.element_size(),
                    compute_kernel='torch.mm_bfloat16_1024x1024_repeated20',
                    evidence_id=f'{evidence.output.name}:rank{dist.get_rank()}:step{evidence.step}'))
    means = {mode: sum(row['milliseconds'] for row in rows if row['mode'] == mode) / 10
             for mode in ('compute', 'communication', 'concurrent')}
    return dict(rows=rows, means_ms=means,
        measured_savings_ms=means['compute'] + means['communication'] - means['concurrent'],
        source='actual_cuda_events_and_nccl', kernel_scope='specified_gemm_microbenchmark',
        model_region_cost=False, communication_shape=shape, dtype='bfloat16',
        communication_participant=dist.get_rank() in (0, 1),
        concurrent_compute_ranks=dist.get_world_size())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', dest='run', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads((args.run / 'request.json').read_text())
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=120), device_id=torch.device('cuda', torch.cuda.current_device()))
    if dist.get_world_size() != request['pp'] or dist.get_world_size() not in (2, 4, 8):
        raise ValueError('通信功能测试要求配置对应的2、4或者8个实际GPU Rank')
    data = IndexedDataset(request['data'])
    samples = [torch.as_tensor(data[index].copy(), device='cuda').to(torch.bfloat16) for index in range(4)]
    shape = [1024, 1024]
    payloads = [sample.repeat((1048576 + sample.numel() - 1) // sample.numel())[:1048576].reshape(shape)
                for sample in samples]
    plan = build_plan(4, shape, pp=request['pp'])
    evidence = Evidence(args.run)
    evidence.communication_initialization_payload = payloads[0]
    executor = CommunicationExecutor(evidence, plan, torch.bfloat16)
    executor.initialize()
    verification = []
    for step, serial in enumerate((True, False)):
        evidence.step = step
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        evidence.cuda_origins[step] = origin
        key = 'serial_calibration_rank_schedules' if serial else 'rank_schedules'
        executor.begin_iteration(serial, key)
        relays = {}
        for task_id in plan[key][str(dist.get_rank())]:
            task = plan['tasks'][task_id]
            if task['kind'] == 'pp_p2p_recv':
                executor.publish_receive(task_id)
            elif task['kind'] == 'pp_p2p_send':
                producer = task['metadata']['producer_task']
                tensor = payloads[task['microbatch']] if producer is None else relays.pop(producer)
                executor.publish_send(task_id, tensor)
            else:
                received = task['metadata']['input_task']
                tensor = executor.consume_receive(received, task_id)
                equal = torch.eq(tensor, payloads[task['microbatch']]).all()
                relays[task_id] = tensor.detach().clone()
                verification.append(equal)
                executor.release_receive(received)
        executor.finish_iteration()
        if not torch.stack(verification).all().item():
            raise ValueError('实际C4载荷在PP/VPP通信后发生数值变化')
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, executor.iterations)
    progress = ScheduleValidator().validate_progress([row for rank in rows for row in rank])
    pair_costs = calibrate_pair(evidence, payloads[0])
    result = dict(status='PASS', rank=dist.get_rank(), source='actual_nccl',
        validation_scope='pp_vpp_communication_with_real_c4_payload',
        verified_tensors=len(verification), progress=progress,
        serial_legality=plan['serial_calibration_legality'],
        concurrent_legality=ScheduleValidator().validate(plan))
    result['independent_and_concurrent_costs'] = pair_costs
    (args.run / f'rank{dist.get_rank()}.protocol-result.json').write_text(json.dumps(result, indent=2) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
