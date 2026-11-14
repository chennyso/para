from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import median
import subprocess


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection_duration(first, second):
    left = right = total = 0
    while left < len(first) and right < len(second):
        a, b = first[left], second[right]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            left += 1
        else:
            right += 1
    return total


def summarize_kernels(rows):
    per_stream = {}
    for start, end, device, stream, name, *_ in rows:
        if end > start:
            per_stream.setdefault((device, stream, 'nccl' in name.lower()), []).append((start, end))
    kernel_ms = {'compute': 0.0, 'nccl': 0.0}
    for flag, label in ((False, 'compute'), (True, 'nccl')):
        kernel_ms[label] = sum(end - start for (device, stream, is_nccl), times in per_stream.items()
            if is_nccl == flag for start, end in merge_intervals(times)) / 1e6
    overlaps, communication_by_compute_stream = [], {}
    for (device, stream, is_nccl), times in per_stream.items():
        if is_nccl:
            continue
        other = merge_intervals((start, end) for (peer, link, flag), spans in per_stream.items()
            if flag and peer == device and link != stream for start, end in spans)
        communication_by_compute_stream[device, stream] = other
        overlaps.append(intersection_duration(merge_intervals(times), other))
    populations = defaultdict(lambda: [[], []])
    for start, end, device, stream, name, *launch_shape in rows:
        if not launch_shape or end <= start or 'nccl' in name.lower():
            continue
        overlap = intersection_duration([(start, end)], communication_by_compute_stream[device, stream])
        if not overlap or overlap * 2 >= end - start:
            populations[device, stream, name, tuple(launch_shape)][bool(overlap)].append((end - start) / 1e6)
    matched = []
    for (device, stream, name, launch_shape), (isolated, overlapping) in populations.items():
        if min(len(isolated), len(overlapping)) >= 3:
            matched.append(dict(device=device, stream=stream, kernel=name, launch_shape=launch_shape,
                isolated_count=len(isolated), overlapping_count=len(overlapping),
                isolated_median_ms=median(isolated), overlapping_median_ms=median(overlapping)))
    matched.sort(key=lambda row: -min(row['isolated_count'], row['overlapping_count']))
    return dict(kernels=len(rows),
        kernel_stream_ms=kernel_ms, observed_overlap_compute_stream_ms=sum(overlaps) / 1e6,
        matched_kernel_family_count=len(matched), observed_duration_pairs=matched[:12])


def nsys_kernels(path):
    exported = path.with_suffix('.overlap.sqlite')
    if not exported.is_file():
        subprocess.run(['/usr/local/cuda/bin/nsys', 'export', '--type=sqlite',
            '--output=' + str(exported), str(path)], check=True, capture_output=True)
    with sqlite3.connect(exported.as_uri() + '?mode=ro', uri=True) as db:
        columns = {row[1] for row in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        if not columns:
            return []
        geometry = ('gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'dynamicSharedMemory')
        signature = ',k.' + ',k.'.join(geometry) if set(geometry) <= columns else ''
        return db.execute('SELECT k.start,k.end,k.deviceId,k.streamId,s.value' + signature + ' '
            'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()


def kineto_kernels(path):
    with path.open() as handle:
        events = json.load(handle)['traceEvents']
    rows = []
    for event in events:
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        args = event['args']
        start = round(event['ts'] * 1000)
        duration = round(event['dur'] * 1000)
        if duration <= 0:
            continue
        grid, block = args.get('grid'), args.get('block')
        shape = (*grid, *block) if isinstance(grid, list) and isinstance(block, list) else ()
        rows.append((start, start + duration, args['device'], args['stream'], event['name'], *shape))
    return rows


def observe_native_overlap(result, request):
    kineto = request.get('native_overlap_trace_backend') == 'kineto'
    source = 'kineto' if kineto else 'nsys'
    if kineto:
        index = Path(result['raw_artifacts']) / 'paramorph-trace-index.json'
        reports = json.loads(index.read_text())['reports'] if index.is_file() else []
        mapping = request['rank_mapping']
        expected = {f'rank{mapping.index(rank)}.paramorph-trace.json' for rank in request['profile_ranks']}
    else:
        reports = result['nsys_index']['reports']
        expected = {f'rank{rank}.profile.nsys-rep' for rank in request['profile_ranks']}
    available = {report['name'] for report in reports}
    observations = []
    for report in reports:
        if report['name'] not in expected:
            continue
        path = Path(report['remote_path'])
        if kineto:
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if path.stat().st_size != report['bytes'] or digest != report['sha256']:
                raise ValueError(f'GPU 采样文件与索引不一致：{path.name}')
        rows = kineto_kernels(path) if kineto else nsys_kernels(path)
        observations.append(dict(report=report['name'], **summarize_kernels(rows)))
    if expected != available:
        status = 'UNEXPECTED_RANK_REPORTS'
    elif not all(row['kernels'] for row in observations):
        status = 'NO_GPU_KERNEL_EVENTS'
    else:
        status = 'PASS'
    return dict(status=status, source=f'actual_native_{source}_cuda_kernel_intervals',
                missing_reports=sorted(expected - available), extra_reports=sorted(available - expected),
                observations=observations,
                scope='同设备不同 CUDA stream 的时间交集；未估计因果减速系数',
                slowdown_measured=False, ranking_eligible=False)
