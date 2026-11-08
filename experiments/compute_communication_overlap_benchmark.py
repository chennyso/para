import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def tensor_elements(payload_bytes, dtype):
    count, remainder = divmod(payload_bytes, torch.tensor([], dtype=dtype).element_size())
    if count <= 0 or remainder:
        raise ValueError('通信字节数必须是通信精度字节数的正整数倍')
    return count


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
    if collective == 'p2p' and group_size == 2:
        return torch.ones(elements, dtype=dtype, device=device), torch.empty(elements, dtype=dtype, device=device)
    raise ValueError('不支持当前并发通信类型或进程组')


def launch_collective(collective, source, destination, group, ranks, rank, mode):
    if collective == 'p2p':
        send = rank == ranks[0]
        peer = ranks[1] if send else ranks[0]
        operation = dist.isend if send else dist.irecv
        tensor = source if send else destination
        if mode == 'batched_p2p':
            return dist.batch_isend_irecv([dist.P2POp(operation, tensor, peer, group)])
        return [operation(tensor, peer, group=group)]
    if collective == 'all_gather':
        return [dist.all_gather_into_tensor(destination, source, group=group, async_op=True)]
    if collective == 'reduce_scatter':
        return [dist.reduce_scatter_tensor(destination, source, group=group, async_op=True)]
    return [dist.all_reduce(source, group=group, async_op=True)]


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
    parser.add_argument('--condition-json', required=True)
    args = parser.parse_args()
    condition = json.loads(args.condition_json)
    ranks = [int(value) for value in args.group_ranks.split(',')]
    if len(ranks) < 2 or len(set(ranks)) != len(ranks):
        raise ValueError('并发测量进程组需要至少两个不同Rank')
    if args.collective == 'p2p' and (len(ranks) != 2 or condition['concurrency_mode'] not in (
            'batched_p2p', 'unbatched_p2p')):
        raise ValueError('P2P 测量需要两个有方向的 Rank 和明确的启动方式')
    if min(args.payload_bytes, args.m, args.n, args.k, args.warmup, args.iterations) < 1:
        raise ValueError('并发测量的消息、Shape和迭代数量必须为正整数')
    if (condition['collective'] != args.collective or condition['payload_bytes'] != args.payload_bytes
            or condition['physical_group_ranks'] != ranks
            or condition['compute_shape'] != dict(M=args.m, N=args.n, K=args.k)):
        raise ValueError('并发测量参数与已登记的成本条件不一致')
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    group = dist.new_group(ranks=ranks, backend='nccl')
    participant = rank in ranks
    row = None
    if participant:
        dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[condition['compute_dtype']]
        comm_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[condition['communication_dtype']]
        if condition['compute_kind'] not in ('forward_compute', 'backward_compute'):
            raise ValueError('未登记的并发计算类型')
        if condition['kernel_measurement'] != 'single_gemm':
            raise ValueError('复合计算必须使用真实 Region 事件采集')
        left = torch.randn(args.m, args.k, dtype=dtype, device=device)
        right = torch.randn(args.k, args.n, dtype=dtype, device=device)
        output_tensor = torch.empty((args.m, args.n), dtype=dtype, device=device)
        def compute():
            torch.mm(left, right, out=output_tensor)

        def communicate():
            for work in launch_collective(args.collective, source, destination, group, ranks, rank,
                    condition['concurrency_mode']):
                work.wait()

        source, destination = communication_buffers(
            args.collective, args.payload_bytes, len(ranks), comm_dtype, device)
        communication_stream = torch.cuda.Stream(device=device)
        compute_samples = []
        communication_samples = []
        overlap_compute_samples = []
        overlap_communication_samples = []
        overlap_pair_samples = []
        for index in range(args.warmup + args.iterations):
            dist.barrier(group=group)
            compute_ms = event_duration(compute)
            dist.barrier(group=group)

            communication_ms = event_duration(communicate)
            dist.barrier(group=group)
            origin = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            communication_end = torch.cuda.Event(enable_timing=True)
            origin.record()
            with torch.cuda.stream(communication_stream):
                communication_stream.wait_event(origin)
                communicate()
                communication_end.record(communication_stream)
            compute()
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
            dtype=condition['compute_dtype'], samples=rows, condition=condition,
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version())),
            gamma=max(1.0, measured_gamma), measured_gamma=measured_gamma,
            gamma_dispersion=statistics.pstdev(item['gamma'] for item in rows),
            compute_kind=condition['compute_kind'])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
