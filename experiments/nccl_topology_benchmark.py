from __future__ import annotations

import argparse
import json
import hashlib
import os
import statistics
from datetime import datetime, timezone

import torch
import torch.distributed as dist


COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
DTYPES = {
    'bfloat16': torch.bfloat16, 'float16': torch.float16,
    'float32': torch.float32, 'float64': torch.float64,
    'int8': torch.int8, 'int32': torch.int32, 'int64': torch.int64,
}
PAYLOAD_DEFINITIONS = {
    'p2p_send_recv': '发送端传输到接收端的一份Tensor字节数',
    'all_reduce': '每个Rank完整输入及输出Tensor的字节数',
    'all_gather': '每个Rank本地输入Tensor的字节数；完整输出为参与者数量乘以该值',
    'reduce_scatter': '每个Rank输出分片的字节数；完整输入为参与者数量乘以该值',
    'all_to_all': '每个源Rank发送到每个目的Rank的均匀分片字节数',
}


def _payloads(value: str) -> tuple[int, ...]:
    result = tuple(int(item) * 1024 * 1024 for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise ValueError("payloads must be positive MiB values")
    return result


def _groups(world_size: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    if group_size < 2 or world_size % group_size:
        raise ValueError(f"world size {world_size} is not divisible by group size {group_size}")
    return tuple(tuple(range(start, start + group_size)) for start in range(0, world_size, group_size))


def _timed(operation, warmup: int, iterations: int, prepare=None) -> list[float]:
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        if prepare is not None:
            prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def _summary(values: list[float], payload_bytes: int, participants: int) -> dict:
    ordered = sorted(values)
    p10 = ordered[max(0, int(len(ordered) * 0.10) - 1)]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))]
    median_us = statistics.median(ordered)
    gbps = payload_bytes / (median_us / 1_000_000.0) / 1_000_000_000.0
    return {
        "median_us": median_us,
        "p10_us": p10,
        "p90_us": p90,
        "payload_gbytes_per_second": gbps,
        "bandwidth_definition": "payload_bytes除以操作时间；字节定义按payload_definition解释",
        "wire_bytes": None,
        "payload_bytes": payload_bytes,
        "participants": participants,
        "samples": values,
        "sample_unit": "microseconds",
    }


def _collective_tensor(payload_bytes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    element_bytes = torch.empty((), dtype=dtype).element_size()
    if type(payload_bytes) is not int or payload_bytes <= 0 or payload_bytes % element_bytes:
        raise ValueError('消息字节数必须为当前dtype的element_size的正整数倍')
    return torch.ones(payload_bytes // element_bytes, dtype=dtype, device=device)


def _run_collective(name: str, payload_bytes: int, group, group_size: int, warmup: int, iterations: int, device: torch.device, dtype: torch.dtype) -> list[float]:
    tensor = _collective_tensor(payload_bytes, device, dtype)
    prepare = None
    if name == "all_reduce":
        operation = lambda: dist.all_reduce(tensor, group=group)
        prepare = lambda: tensor.fill_(1)
    elif name == "all_gather":
        output = [torch.empty_like(tensor) for _ in range(group_size)]
        operation = lambda: dist.all_gather(output, tensor, group=group)
    elif name == "reduce_scatter":
        inputs = [torch.ones_like(tensor) for _ in range(group_size)]
        output = torch.empty_like(tensor)
        operation = lambda: dist.reduce_scatter(output, inputs, group=group)
    elif name == "all_to_all":
        output = [torch.empty_like(tensor) for _ in range(group_size)]
        inputs = [torch.ones_like(tensor) for _ in range(group_size)]
        operation = lambda: dist.all_to_all(output, inputs, group=group)
    else:
        raise ValueError(name)
    return _timed(operation, warmup, iterations, prepare)


def _run_p2p(payload_bytes: int, group, pair: tuple[int, int], warmup: int, iterations: int, device: torch.device, rank: int, dtype: torch.dtype) -> list[float]:
    tensor = _collective_tensor(payload_bytes, device, dtype)
    received = torch.empty_like(tensor)

    def operation() -> None:
        if rank == pair[0]:
            requests = dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, pair[1], group)])
        else:
            requests = dist.batch_isend_irecv([dist.P2POp(dist.irecv, received, pair[0], group)])
        for request in requests:
            request.wait()

    return _timed(operation, warmup, iterations)


def _group_samples(values, group, ranks):
    gathered = [None for _ in ranks]
    dist.all_gather_object(gathered, values, group=group)
    return [max(samples[index] for samples in gathered) for index in range(len(values))], {
        str(rank): samples for rank, samples in zip(ranks, gathered)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-sizes", default="2,4,8")
    parser.add_argument("--payloads-mib", default="1,4,16,64,256")
    parser.add_argument('--payloads-bytes')
    parser.add_argument('--dtype', choices=tuple(DTYPES), default='bfloat16')
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument('--rank-groups-json')
    args = parser.parse_args()
    if args.backend != 'nccl' or args.warmup < 0 or args.iterations <= 0:
        raise ValueError('NCCL测量需要nccl后端、非负预热次数与正整数测量次数')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NCCL topology benchmark")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=args.backend)
    rank_sum = torch.tensor(rank + 1, device=device, dtype=torch.int64)
    dist.all_reduce(rank_sum)
    if rank_sum.item() != world_size * (world_size + 1) // 2:
        raise RuntimeError('NCCL全局Rank验证失败')
    group_sizes = tuple(int(item) for item in args.group_sizes.split(","))
    payloads = tuple(int(item) for item in args.payloads_bytes.split(',')) if args.payloads_bytes else _payloads(args.payloads_mib)
    local_records = []
    group_rows = [ranks for size in group_sizes for ranks in _groups(world_size, size)]
    selected_cases = None
    concurrent_batches = []
    if args.rank_groups_json:
        with open(args.rank_groups_json, encoding='utf-8') as handle:
            document = json.load(handle)
            groups = document['nccl_rank_groups'] if isinstance(document, dict) else document
            group_rows = [tuple(row) for row in groups]
            if isinstance(document, dict) and document.get('nccl_payloads_bytes') is not None:
                payloads = tuple(document['nccl_payloads_bytes'])
            if isinstance(document, dict) and document.get('nccl_cases') is not None:
                selected_cases = {(row['collective'], tuple(row['ranks']), row['payload_bytes'])
                                  for row in document['nccl_cases']}
                if any(operation not in (*COLLECTIVES, 'p2p_send_recv')
                        or (tuple(sorted(ranks)) if operation == 'p2p_send_recv' else ranks) not in group_rows
                        or type(size) is not int or size not in payloads
                        for operation, ranks, size in selected_cases):
                    raise ValueError('NCCL测量条件必须引用实际进程组、消息字节数和支持的操作')
            if isinstance(document, dict):
                concurrent_batches = document.get('nccl_concurrent_group_batches', [])
                for batch in concurrent_batches:
                    batch_groups = [tuple(row) for row in batch['groups']]
                    if (batch['collective'] != 'all_reduce'
                            or batch['payload_bytes'] not in payloads
                            or not batch.get('concurrency_key')
                            or len({rank for row in batch_groups for rank in row}) !=
                               sum(len(row) for row in batch_groups)
                            or any(row not in group_rows for row in batch_groups)):
                        raise ValueError('并发NCCL Batch需要互不重叠的实际进程组、已登记消息和唯一并发身份')
                    batch['groups'] = batch_groups
        if any(len(row) < 2 or len(set(row)) != len(row) or
               any(type(rank) is not int or not 0 <= rank < world_size for rank in row) for row in group_rows):
            raise ValueError('NCCL实测Rank Group必须完整包含合法且不同的参与者')
    if not payloads or any(type(value) is not int or value <= 0 for value in payloads):
        raise ValueError('NCCL消息字节数需要非空正整数列表')
    dtype = DTYPES[args.dtype]
    created_groups = {ranks: dist.new_group(ranks=list(ranks), backend=args.backend)
                      for ranks in group_rows}
    for ranks in group_rows:
        group_size = len(ranks)
        if ranks:
            group = created_groups[ranks]
            is_member = rank in ranks
            leader = ranks[0]
            if is_member:
                actual_ranks = tuple(dist.get_process_group_ranks(group))
                leader = actual_ranks[0]
                for payload_bytes in payloads:
                    for collective in COLLECTIVES:
                        if selected_cases is not None and (collective, actual_ranks, payload_bytes) not in selected_cases:
                            continue
                        dist.barrier(group=group)
                        values = _run_collective(
                            collective, payload_bytes, group, group_size, args.warmup, args.iterations, device, dtype
                        )
                        values, rank_samples = _group_samples(values, group, actual_ranks)
                        if rank == leader:
                            local_records.append({
                                "kind": "collective",
                                "group_size": group_size,
                                "ranks": list(actual_ranks),
                                "leader_rank": leader,
                                "collective": collective,
                                'dtype': args.dtype,
                                'payload_definition': PAYLOAD_DEFINITIONS[collective],
                                'input_bytes_per_rank': payload_bytes * group_size if collective in ('reduce_scatter', 'all_to_all') else payload_bytes,
                                'output_bytes_per_rank': payload_bytes * group_size if collective in ('all_gather', 'all_to_all') else payload_bytes,
                                'timing_scope': 'maximum_participant_duration_per_iteration',
                                'rank_samples_microseconds': rank_samples,
                                **_summary(values, payload_bytes, group_size),
                            })
                    pairs = []
                    if group_size == 2:
                        pairs = [actual_ranks] if selected_cases is None else [direction for direction in
                            (actual_ranks, tuple(reversed(actual_ranks)))
                            if ('p2p_send_recv', direction, payload_bytes) in selected_cases]
                    for pair in pairs:
                        dist.barrier(group=group)
                        values = _run_p2p(payload_bytes, group, pair, args.warmup, args.iterations, device, rank, dtype)
                        values, rank_samples = _group_samples(values, group, actual_ranks)
                        if rank == leader:
                            local_records.append({
                                "kind": "p2p_send_recv",
                                "group_size": group_size,
                                "ranks": list(pair),
                                "leader_rank": leader,
                                'dtype': args.dtype,
                                'payload_definition': PAYLOAD_DEFINITIONS['p2p_send_recv'],
                                'sender_rank': pair[0], 'receiver_rank': pair[1],
                                'timing_scope': 'maximum_sender_receiver_duration_per_iteration',
                                'rank_samples_microseconds': rank_samples,
                                **_summary(values, payload_bytes, group_size),
                            })
            dist.barrier()
    for batch in concurrent_batches:
        batch_groups = batch['groups']
        member_group = next((ranks for ranks in batch_groups if rank in ranks), None)
        tensor = (_collective_tensor(batch['payload_bytes'], device, dtype)
                  if member_group is not None else None)
        local_values = []
        for iteration in range(args.warmup + args.iterations):
            if tensor is not None:
                tensor.fill_(1)
            dist.barrier()
            if member_group is not None:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(tensor, group=created_groups[member_group])
                end.record()
                end.synchronize()
                if iteration >= args.warmup:
                    local_values.append(float(start.elapsed_time(end)) * 1000.0)
            dist.barrier()
        gathered_values = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_values, local_values)
        if rank == 0:
            for ranks in batch_groups:
                values = [max(gathered_values[participant][index] for participant in ranks)
                          for index in range(args.iterations)]
                rank_samples = {str(participant): gathered_values[participant]
                                for participant in ranks}
                local_records.append({
                    'kind': 'collective', 'group_size': len(ranks), 'ranks': list(ranks),
                    'leader_rank': ranks[0], 'collective': batch['collective'],
                    'dtype': args.dtype, 'payload_definition': PAYLOAD_DEFINITIONS[batch['collective']],
                    'input_bytes_per_rank': batch['payload_bytes'],
                    'output_bytes_per_rank': batch['payload_bytes'],
                    'timing_scope': 'maximum_participant_duration_per_iteration',
                    'rank_samples_microseconds': rank_samples,
                    'concurrent_operations': [batch['concurrency_key']],
                    'concurrency_context': dict(
                        scheduled_operations=[dict(collective=batch['collective'], groups=[list(row)
                            for row in batch_groups], payload_bytes=batch['payload_bytes'])],
                        external_contention='unobserved', cuda_stream='current_stream'),
                    **_summary(values, batch['payload_bytes'], len(ranks)),
                })
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records)
    devices = [None] * world_size
    dist.all_gather_object(devices, dict(rank=rank, local_rank=local_rank,
        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid), gpu_name=torch.cuda.get_device_name(device),
        node=os.environ.get('PARAMELD_PHYSICAL_NODE', os.uname().nodename), pod=os.uname().nodename))
    conditions = [None] * world_size
    dist.all_gather_object(conditions, dict(
        environment={key: value for key, value in os.environ.items() if key.startswith(('NCCL_', 'CUDA_', 'UCX_'))},
        software=dict(torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version())))
    if rank == 0:
        records = [record for records_for_rank in gathered for record in records_for_rank]
        result = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "world_size": world_size,
            "verified_rank_sum": int(rank_sum.item()),
            "backend": args.backend,
            "cuda_device": torch.cuda.get_device_name(device),
            "environment": {key: value for key, value in os.environ.items() if key.startswith(("NCCL_", "CUDA_", "UCX_"))},
            "records": records,
            'source': 'actual_nccl_cuda_event_microbenchmark',
            'software': dict(torch=torch.__version__, cuda=torch.version.cuda,
                             nccl=torch.cuda.nccl.version()),
            'devices': devices,
            'dtype': args.dtype,
            'sample_unit': 'microseconds',
            'payload_definitions': PAYLOAD_DEFINITIONS,
            'concurrent_operations': [],
            'concurrency_context': dict(scheduled_operations='sequential_collectives_and_process_groups',
                                       external_contention='unobserved', cuda_stream='current_stream'),
            'paths': None,
            'environment_per_rank': {str(index): row['environment'] for index, row in enumerate(conditions)},
            'software_per_rank': {str(index): row['software'] for index, row in enumerate(conditions)},
        }
        result['hardware_fingerprint'] = hashlib.sha256(json.dumps(dict(devices=devices,
            software=result['software'], environment={key: value for key, value in result['environment'].items()
                if key != 'NCCL_DEBUG_FILE'},
            environment_per_rank={rank: {key: value for key, value in environment.items() if key != 'NCCL_DEBUG_FILE'}
                                  for rank, environment in result['environment_per_rank'].items()},
            software_per_rank=result['software_per_rank'], paths=result['paths']), sort_keys=True, separators=(',', ':'),
                allow_nan=False).encode()).hexdigest()
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
