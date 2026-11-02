import argparse
import ast
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def source_contracts(root):
    entry_path = root / 'hybrid_training_entry.py'
    adapter_path = root / 'paramorph/adapters/megatron_adapter.py'
    runtime_path = root / 'paramorph/runtime.py'
    trees = {}
    hashes = {}
    for path in (entry_path, adapter_path, runtime_path):
        source = path.read_text()
        trees[path.name] = ast.parse(source, filename=str(path))
        compile(trees[path.name], str(path), 'exec')
        hashes[str(path.relative_to(root))] = hashlib.sha256(source.encode()).hexdigest()
    pretrain = function(trees['hybrid_training_entry.py'], 'pretrain')
    explicit = [node for node in ast.walk(pretrain) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "'clip_grad' in REQUEST"]
    if len(explicit) != 1:
        raise ValueError('训练入口没有唯一的显式clip_grad分支')
    updates = [node for node in ast.walk(explicit[0]) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == 'cfg.optimizer.clip_grad' for target in node.targets)]
    if len(updates) != 1 or ast.unparse(updates[0].value) != 'float(clip_grad)':
        raise ValueError('训练入口没有将显式clip_grad应用到实际Optimizer配置')
    all_updates = [node for node in ast.walk(pretrain) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == 'cfg.optimizer.clip_grad' for target in node.targets)]
    if all_updates != updates:
        raise ValueError('未指定clip_grad时存在额外Optimizer默认值改写')
    ownership = function(trees['hybrid_training_entry.py'], 'record_ownership')
    manifest_updates = [node for node in ast.walk(ownership) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "manifest['clip_grad']" for target in node.targets)]
    if len(manifest_updates) != 1 or ast.unparse(manifest_updates[0].value) != 'context.state.cfg.optimizer.clip_grad':
        raise ValueError('Manifest没有记录实际运行的clip_grad')
    checks = [node for node in ast.walk(ownership) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "'clip_grad' in REQUEST and manifest['clip_grad'] != REQUEST['clip_grad']"]
    if len(checks) != 1 or not any(isinstance(node, ast.Raise) for node in ast.walk(checks[0])):
        raise ValueError('Manifest没有检查显式请求与实际clip_grad一致性')
    execute = function(trees['megatron_adapter.py'], 'execute')
    profiles = [node for node in ast.walk(execute) if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'profile']
    if not any(isinstance(node, ast.keyword) and node.arg == 'paramorph_optimizer_calibration'
            and isinstance(node.value, ast.Constant) and node.value.value is True
            for branch in profiles for node in ast.walk(branch)):
        raise ValueError('正常Profiling没有启用Optimizer组件测量')
    calibration = function(trees['runtime.py'], 'install_optimizer_cost_calibration')
    counts = [node for node in ast.walk(calibration) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'parameter_elements' for target in node.targets)]
    if len(counts) != 1 or 'to_local_if_dtensor(parameter).numel()' not in ast.unparse(counts[0].value):
        raise ValueError('Optimizer拥有的参数数量没有使用实际本地DTensor分片')
    prepares = [node for node in ast.walk(calibration) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'install'
        and any(isinstance(argument, ast.Constant) and argument.value == '_copy_model_grads_to_main_grads'
            for argument in node.args)]
    if len(prepares) != 1:
        raise ValueError('Optimizer准备函数没有注册实际调用计时')
    return dict(status='PASS', verification_scope='actual_python_source_wiring_and_syntax', source_hashes=hashes)


def dtensor_worker(rank, directory):
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory / 'rendezvous').as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    mesh = DeviceMesh('cpu', torch.arange(2))
    local = torch.arange(rank * 8, (rank + 1) * 8, dtype=torch.float32)
    parameter = torch.nn.Parameter(DTensor.from_local(local, mesh, [Shard(0)], run_check=True))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    parameter.grad = DTensor.from_local(torch.ones_like(local), mesh, [Shard(0)], run_check=True)
    optimizer.step()
    global_elements = sum(value.numel() for group in optimizer.param_groups for value in group['params'])
    local_elements = sum(value.to_local().numel() for group in optimizer.param_groups for value in group['params'])
    total = torch.tensor(local_elements, dtype=torch.int64)
    dist.all_reduce(total)
    if global_elements != 16 or local_elements != 8 or total.item() != global_elements:
        raise ValueError('实际CPU DTensor的全局参数与本地分片元素数不一致')
    result = dict(rank=rank, global_elements=global_elements, local_elements=local_elements,
        summed_local_elements=total.item(), optimizer='torch.optim.Adam', update_completed=True)
    (directory / f'rank{rank}.json').write_text(json.dumps(result) + '\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    source = source_contracts(args.root)
    if args.source_only:
        result = dict(status='SOURCE_CONTRACT_PASS', source_contract=source,
            dtensor_verification=dict(status='NOT_RUN'),
            gpu_measurement_performed=False, cuda_event_measurement_verified=False,
            fsdp_v1_runtime_verified=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False))
        return
    directory = args.output.parent / ('dtensor-' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    multiprocessing.spawn(dtensor_worker, args=(str(directory),), nprocs=2, join=True)
    ranks = [json.loads((directory / f'rank{rank}.json').read_text()) for rank in range(2)]
    result = dict(status='PASS', source_contract=source,
        dtensor_verification=dict(backend='gloo', device='cpu', world_size=2, ranks=ranks),
        gpu_measurement_performed=False, cuda_event_measurement_verified=False,
        fsdp_v1_runtime_verified=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
