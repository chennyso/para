from collections import defaultdict
from pathlib import Path

import psutil


GIB = 2 ** 30


def host_memory_snapshot():
    base = Path('/sys/fs/cgroup')
    current = int((base / 'memory.current').read_text())
    limit = (base / 'memory.max').read_text().strip()
    stat = {name: int(value) for name, value in (line.split() for line in (base / 'memory.stat').read_text().splitlines())}
    events = {name: int(value) for name, value in (line.split() for line in (base / 'memory.events').read_text().splitlines())}
    cache = max(0, min(current, stat['inactive_file'], stat['file'] - stat['shmem']))
    return dict(current_bytes=current, working_set_bytes=current - cache, reclaimable_file_bytes=cache,
        limit_bytes=None if limit == 'max' else int(limit), node_available_bytes=psutil.virtual_memory().available,
        events=events)


def cgroup_working_set_limit(limit):
    return None if limit is None else limit - max(16 * GIB, limit // 20)


def check_host_admission(preflight, nodes, ranks_per_worker, planned):
    physical = defaultdict(list)
    violations = []
    for node, row in zip(nodes, preflight, strict=True):
        if 'host_memory' not in row:
            continue
        index = node['index']
        memory = row['host_memory']
        lower_bound = sum(planned.get(str(rank), 0) for rank in range(index * ranks_per_worker,
            (index + 1) * ranks_per_worker))
        limit = cgroup_working_set_limit(memory['limit_bytes'])
        physical[node['node']].append((lower_bound, memory['node_available_bytes']))
        if limit is not None and memory['working_set_bytes'] + lower_bound >= limit:
            violations.append(dict(worker=index, cause='container_working_set',
                lower_bound_bytes=lower_bound, available_bytes=max(0, limit - memory['working_set_bytes'])))
    for node, workers in physical.items():
        lower_bound = sum(value for value, _ in workers)
        available = min(available for _, available in workers) - 32 * GIB
        if lower_bound > available:
            violations.append(dict(node=node, cause='physical_node_headroom',
                lower_bound_bytes=lower_bound, available_bytes=max(0, available)))
    observed = sum(len(workers) for workers in physical.values())
    return dict(status='REJECTED' if violations else 'LOWER_BOUND_CHECKED' if observed == len(nodes) else 'UNKNOWN',
        predicted_optimizer_memory_complete=bool(planned), violations=violations)
