import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def tensor_elements(payload_bytes, dtype):
    return max(1, payload_bytes // torch.tensor([], dtype=dtype).element_size())


def communication_buffers(collective, payload_bytes, group_size, dtype, device):
    elements = tensor_elements(payload_bytes, dtype)
    if collective == 'all_gather':
        return torch.ones(elements, dtype=dtype, device=device), torch.empty(
            elements * group_size, dtype=dtype, device=device)
    if collective == 'reduce_scatter':
        return torch.ones(elements * group_size, dtype=dtype, device=device), torch.empty(
            elements, dtype=dtype, device=device)
    if collective == 'all_reduce':
        tensor = torch.ones(elements, dtype=dtype, device=device)
        return tensor, tensor
    raise ValueError('并发测量仅支持all_gather、reduce_scatter和all_reduce')


def launch_collective(collective, source, destination, group):
    if collective == 'all_gather':
        return dist.all_gather_into_tensor(destination, source, group=group, async_op=True)
    if collective == 'reduce_scatter':
        return dist.reduce_scatter_tensor(destination, source, group=group, async_op=True)
    return dist.all_reduce(source, group=group, async_op=True)


def measured_region_backward(left, right):
    output = left @ right
    output.sum().backward()
    left.grad = None
    right.grad = None


def event_duration(operation):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group-ranks', required=True)
    parser.add_argument('--collective', required=True)
    parser.add_argument('--payload-bytes', type=int, required=True)
    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--warmup', type=int, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    args = parser.parse_args()
    ranks = [int(value) for value in args.group_ranks.split(',')]
    if len(ranks) < 2 or len(set(ranks)) != len(ranks):
        raise ValueError('并发测量进程组需要至少两个不同Rank')
    if min(args.payload_bytes, args.m, args.n, args.k, args.warmup, args.iterations) < 1:
        raise ValueError('并发测量的消息、Shape和迭代数量必须为正整数')
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    group = dist.new_group(ranks=ranks, backend='nccl')
    participant = rank in ranks
    row = None
    if participant:
        left = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device, requires_grad=True)
        right = torch.randn(args.k, args.n, dtype=torch.bfloat16, device=device, requires_grad=True)
        source, destination = communication_buffers(
            args.collective, args.payload_bytes, len(ranks), torch.bfloat16, device)
        communication_stream = torch.cuda.Stream(device=device)
        compute_samples = []
        communication_samples = []
        overlap_compute_samples = []
        overlap_communication_samples = []
        overlap_pair_samples = []
        for index in range(args.warmup + args.iterations):
            dist.barrier(group=group)
            compute_ms = event_duration(lambda: measured_region_backward(left, right))
            dist.barrier(group=group)

            def communication_only():
                operation = launch_collective(args.collective, source, destination, group)
                operation.wait()

            communication_ms = event_duration(communication_only)
            dist.barrier(group=group)
            origin = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            communication_end = torch.cuda.Event(enable_timing=True)
            origin.record()
            with torch.cuda.stream(communication_stream):
                communication_stream.wait_event(origin)
                operation = launch_collective(args.collective, source, destination, group)
                operation.wait()
                communication_end.record(communication_stream)
            measured_region_backward(left, right)
            compute_end.record()
            torch.cuda.synchronize()
            if index >= args.warmup:
                compute_samples.append(compute_ms)
                communication_samples.append(communication_ms)
                overlap_compute_samples.append(origin.elapsed_time(compute_end))
                overlap_communication_samples.append(origin.elapsed_time(communication_end))
                overlap_pair_samples.append(max(overlap_compute_samples[-1],
                    overlap_communication_samples[-1]))
        gamma = statistics.median(overlap_compute_samples) / statistics.median(compute_samples)
        row = dict(rank=rank, compute_alone_ms=compute_samples,
            communication_alone_ms=communication_samples,
            overlapped_compute_ms=overlap_compute_samples,
            overlapped_communication_ms=overlap_communication_samples,
            overlapped_pair_ms=overlap_pair_samples, gamma=gamma)
    gathered = [None for _ in range(dist.get_world_size())] if rank == 0 else None
    dist.gather_object(row, gathered, dst=0)
    if rank == 0:
        rows = [item for item in gathered if item is not None]
        if sorted(item['rank'] for item in rows) != sorted(ranks):
            raise ValueError('并发测量结果没有覆盖完整通信进程组')
        measured_gamma = statistics.median(item['gamma'] for item in rows)
        result = dict(source='real_gpu_compute_communication_microbenchmark',
            hardware_scope='actual_rank_group_and_current_torch_cuda_nccl_environment',
            collective=args.collective, payload_bytes=args.payload_bytes,
            group_ranks=ranks, compute_shape=dict(M=args.m, N=args.n, K=args.k),
            dtype='bfloat16', samples=rows,
            gamma=max(1.0, measured_gamma), measured_gamma=measured_gamma,
            gamma_dispersion=statistics.pstdev(item['gamma'] for item in rows),
            compute_kind='backward_compute')
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
