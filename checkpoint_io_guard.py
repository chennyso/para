from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from filelock import FileLock
import psutil
import torch
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.planner import LoadItemType


def install_checkpoint_io_guard(output, request):
    output = Path(output)
    original = FileSystemReader.read_data
    lock_path = Path('/dev/shm') / ('parameld-checkpoint-' + hashlib.sha256(str(output).encode()).hexdigest()[:16] + '.lock')
    split_workers = request.get('gpus_per_worker', request.get('gpus_per_node')) != request.get('gpus_per_node')
    readers = request.get('checkpoint_io_readers_per_node', 1)
    if readers not in (1, 2):
        raise ValueError('每个节点仅支持一个或两个并发检查点读取进程')
    if split_workers:
        physical_node = os.environ['POD_NODE_NAME']
        if physical_node not in ('g5', 'g6'):
            raise ValueError('分Pod检查点读取需要已确认的物理节点')
        lock_path = output / ('checkpoint-io-' + physical_node + '.lock')
    timeout = float(request.get('max_allocation_seconds', 600))

    def read_data(reader, plan, planner):
        rank = torch.distributed.get_rank()
        physical_rank = request['rank_mapping'][rank] if 'rank_mapping' in request else rank
        lane = physical_rank % readers
        active_lock = lock_path.with_name(f'{lock_path.stem}-{lane}{lock_path.suffix}') if readers > 1 else lock_path
        process = psutil.Process()
        current_path = Path('/sys/fs/cgroup/memory.current')
        metadata = reader.read_metadata()
        ranges, optimizer_keys, samples = defaultdict(list), [], []
        requested_bytes = 0
        for item in plan.items:
            storage = reader.storage_data[item.storage_index]
            ranges[str(storage.relative_path)].append((int(storage.offset), int(storage.length)))
            name = item.storage_index.fqn
            if name.startswith(('optimizer', 'optim.', 'lr_scheduler')):
                optimizer_keys.append(name)
            if item.type == LoadItemType.TENSOR:
                tensor_metadata = metadata.state_dict_metadata[name]
                element_bytes = torch.empty(0, dtype=tensor_metadata.properties.dtype).element_size()
                local_elements = math.prod(item.lengths)
                source_elements = math.prod(tensor_metadata.size)
                requested_bytes += local_elements * element_bytes
                if len(samples) < 12:
                    samples.append(dict(name=name, requested_elements=local_elements,
                        source_tensor_elements=source_elements, source_slice_bytes=int(storage.length),
                        requested_shape=list(item.lengths), source_offsets=list(item.storage_offsets)))
        if request.get('optimizer_checkpoint_mode', 'fresh') in ('fresh', 'disabled') and optimizer_keys:
            raise ValueError('新优化器试验的检查点读取计划包含旧优化器状态')
        record = dict(rank=rank, checkpoint=str(reader.path), source='actual_torch_dcp_read_plan',
            io_guard_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            read_items=len(plan.items), source_files=len(ranges), requested_tensor_bytes=requested_bytes,
            serialized_slice_bytes=sum(length for rows in ranges.values() for _, length in rows),
            optimizer_state_items=len(optimizer_keys), tensor_samples=samples,
            rss_before=process.memory_info().rss, cgroup_before=int(current_path.read_text()),
            advised_bytes=0, status='STARTED', started_at=time.time(), io_readers_per_node=readers,
            reader_lane=lane)
        record_path = output / f'rank{rank}.checkpoint-read.jsonl'
        with record_path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        original_commit = planner.commit_tensor
        descriptors = {}

        def commit(item, tensor):
            result = original_commit(item, tensor)
            current = int(current_path.read_text())
            available = psutil.virtual_memory().available
            if current > 192 * 1024**3 or available < 64 * 1024**3:
                storage = reader.storage_data[item.storage_index]
                path = Path(reader.path) / storage.relative_path
                if path not in descriptors:
                    descriptors[path] = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                os.posix_fadvise(descriptors[path], int(storage.offset), int(storage.length), os.POSIX_FADV_DONTNEED)
                record['advised_bytes'] += int(storage.length)
            return result

        with FileLock(str(active_lock), timeout=timeout):
            record['lock_acquired_at'] = time.time()
            record['status'] = 'ACQUIRED'
            with record_path.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            planner.commit_tensor = commit
            try:
                future = original(reader, plan, planner)
                future.wait()
                record['status'] = 'COMPLETED'
                return future
            finally:
                planner.commit_tensor = original_commit
                for descriptor in descriptors.values():
                    os.close(descriptor)
                if record['status'] != 'COMPLETED':
                    record['status'] = 'INTERRUPTED'
                record.update(finished_at=time.time(), rss_after=process.memory_info().rss,
                    cgroup_after=int(current_path.read_text()),
                    node_available_bytes=psutil.virtual_memory().available)
                with record_path.open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    FileSystemReader.read_data = read_data
