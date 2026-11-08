import argparse
import hashlib
import json
import math
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--gpu-seconds', type=float, required=True)
args = parser.parse_args()
root = args.run
request = json.loads((root / 'request.json').read_text())
sources = json.loads((root / 'source_hashes.json').read_text())
for name, digest in sources.items():
    if hashlib.sha256((root / 'code' / name).read_bytes()).hexdigest() != digest:
        raise ValueError('实际运行源码与登记SHA256不同')
world = request.get('node_count', 2) * request.get('gpus_per_node', 8)
nodes = world // request.get('gpus_per_worker', request.get('gpus_per_node', 8))
exits = [json.loads((root / f'node{node}.exit.json').read_text()) for node in range(nodes)]
codes = [row['returncode'] for row in exits]
files = {}
for pattern in ('rank*.manifest.json', 'rank*.measurements.json', 'node*.log',
        'node*.cuda-preflight.json', 'node*.exit.json', 'node*.host-memory.json'):
    for path in root.glob(pattern):
        files[path.name] = path.read_text()
record = dict(request, source_hashes=sources, returncodes=codes,
    gpu_allocation_seconds=args.gpu_seconds, performance_eligible=False,
    raw_artifacts=str(root), raw_artifacts_storage='PVC',
    execution_deadline_reached=any(row.get('execution_deadline_reached', False) for row in exits),
    host_memory_failures=[dict(node=index, **row['host_memory']) for index, row in enumerate(exits)
        if row.get('host_memory') is not None])
if any(code != 0 for code in codes):
    logs = ''.join(files.get(f'node{node}.log', '') for node in range(nodes))
    cuda_oom = 'CUDA out of memory' in logs or 'Failed to CUDA calloc async' in logs
    record['status'] = ('CUDA_OOM' if cuda_oom else 'HOST_MEMORY_BUDGET_EXCEEDED'
        if record['host_memory_failures'] else 'FAILED')
else:
    rows = [json.loads(files[f'rank{rank}.measurements.json']) for rank in range(world)]
    manifests = [json.loads(files[f'rank{rank}.manifest.json']) for rank in range(world)]
    if request.get('recompute') == 'selective':
        modules = request['memory_policy']['recompute_modules']
        for module in modules:
            if sum(manifest['native_selective_recomputation']['enabled_modules'][module]
                    for manifest in manifests) == 0:
                raise ValueError(f'原生重计算模块没有实际启用：{module}')
        if any(manifest['native_selective_recomputation']['requested_modules'] != modules
                for manifest in manifests):
            raise ValueError('原生重计算模块与完整策略不同')
        record['native_selective_recompute_verified'] = True
    for rank, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        if row['rank'] != rank or row['status'] != 'complete' or manifest['world'] != world:
            raise ValueError('逐Rank执行身份或完成状态不同')
        if row['updates'] != request['warmup'] + request['measured']:
            raise ValueError('实际更新次数不符合预热和测量约定')
        if row['warmup'] != request['warmup'] or row['measured'] != request['measured']:
            raise ValueError('实际测量窗口不符合请求')
        if row['consumed_samples'] != row['updates'] * request['global_batch']:
            raise ValueError('实际消耗样本数量不符合固定负载')
        for step in row['steps']:
            if not step['optimizer_update_successful']:
                raise ValueError('Optimizer更新未成功')
            if step['grad_norm'] is not None and not math.isfinite(step['grad_norm']):
                raise ValueError('梯度范数出现非有限值')
            if any(not math.isfinite(value) for value in step['loss'].values()):
                raise ValueError('损失出现非有限值')
        for key in ('tp', 'cp', 'pp', 'ep', 'vpp', 'sequence', 'micro_batch', 'global_batch'):
            if manifest[key] != request[key]:
                raise ValueError('实际并行配置与请求不同：' + key)
        for key, expected in (('dp', request['dp']), ('etp', request.get('etp', request['tp'])),
            ('expert_dp', world // (request.get('etp', request['tp']) * request['ep'] * request['pp'])),
            ('optimizer_sharding', request.get('optimizer_sharding', 'distributed'))):
            if key in manifest and manifest[key] != expected:
                raise ValueError('实际进程组或状态分片与请求不同：' + key)
        if ('optimizer_pin_cpu_params' in request
                and manifest['optimizer_pin_cpu_params'] != request['optimizer_pin_cpu_params']):
            raise ValueError('实际CPU参数锁页与请求不同')
        if ('optimizer_cpu_copy_mode' in request
                and manifest['optimizer_cpu_copy_mode'] != request['optimizer_cpu_copy_mode']):
            raise ValueError('实际CPU参数复制方式与请求不同')
    window = max(row['window_seconds'] for row in rows)
    record.update(status='PASS', performance_eligible=not bool(request.get('validation_mode')),
        step_seconds=window / request['measured'],
        tokens_per_second=request['measured'] * request['global_batch'] * request['sequence'] / window,
        peak_allocated_gib=max(row['allocated_gib'] for row in rows),
        peak_reserved_gib=max(row['reserved_gib'] for row in rows),
        numerical_equivalence_certified=False, validation_scope=rows[0]['validation_scope'])
(root / 'result.json').write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
print(json.dumps(dict(result=record, files=files), ensure_ascii=False, allow_nan=False))
