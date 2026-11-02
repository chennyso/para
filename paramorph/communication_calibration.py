import json

import torch
import torch.distributed as dist


def allocate_pipeline_payload(shape, dtype, device):
    if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError('Pipeline通信Shape必须包含三个正整数维度')
    if not isinstance(dtype, torch.dtype):
        raise ValueError('Pipeline通信必须使用实际后端配置的Tensor精度')
    payload = torch.zeros(shape, dtype=dtype, device=device)
    return payload, torch.empty_like(payload)


def calibrate_pipeline_links(output, request, models):
    from megatron.core import parallel_state

    group = parallel_state.get_pipeline_model_parallel_group()
    ranks = dist.get_process_group_ranks(group)
    physical = group.rank()
    config = models[0].config
    if len(ranks) == 1:
        (output / f'rank{dist.get_rank()}.paramorph-links.json').write_text(json.dumps(dict(
            rank=dist.get_rank(), request=request, rows=[], source='single_stage_has_no_pipeline_transport'),
            indent=2, allow_nan=False) + '\n')
        return
    if request['sequence'] % request['cp']:
        raise ValueError('Pipeline通信序列长度不能被CP完整分片')
    sequence = request['sequence'] // request['cp']
    if config.sequence_parallel:
        if sequence % request['tp']:
            raise ValueError('Pipeline通信序列长度不能被Sequence Parallel完整分片')
        sequence //= request['tp']
    shape = [sequence, request['micro_batch'], config.hidden_size]
    payload, receive = allocate_pipeline_payload(shape, config.pipeline_dtype,
        torch.device('cuda', torch.cuda.current_device()))
    rows = []
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
    for edge in range(len(ranks)):
        for direction in ('forward', 'backward'):
            source, destination = edge, (edge + 1) % len(ranks)
            if direction == 'backward':
                source, destination = destination, source
            for repetition in range(6):
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                if physical in (source, destination):
                    start.record()
                    if physical == source:
                        work = dist.isend(payload, dst=ranks[destination], group=group)
                    else:
                        work = dist.irecv(receive, src=ranks[source], group=group)
                    work.wait()
                    end.record()
                    end.synchronize()
                    wait_start, wait_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wait_start.record()
                    work.wait()
                    wait_end.record()
                    wait_end.synchronize()
                    if repetition:
                        rows.append(dict(direction=direction, source=ranks[source], destination=ranks[destination],
                            role='send' if physical == source else 'recv', shape=shape,
                            dtype=str(config.pipeline_dtype), milliseconds=start.elapsed_time(end),
                            completed_wait_milliseconds=wait_start.elapsed_time(wait_end),
                            bytes=payload.numel() * payload.element_size(), repetition=repetition,
                            evidence_id=f'{request["run_id"]}:rank{dist.get_rank()}:{edge}:{direction}:{repetition}'))
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
    (output / f'rank{dist.get_rank()}.paramorph-links.json').write_text(json.dumps(dict(
        rank=dist.get_rank(), request=request, rows=rows,
        source='actual_nccl_p2p_cuda_event_calibration',
        payload_source='independent_cuda_tensor_with_pipeline_shape_and_dtype', synchronized_peer_readiness=True),
        indent=2, allow_nan=False) + '\n')
