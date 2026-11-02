import argparse
import ast
import json
from pathlib import Path
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from memory_audit import tensor_storage_inventory


def verify_rank(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    mesh = DeviceMesh('cpu', torch.arange(2))
    parameter = distribute_tensor(torch.arange(16, dtype=torch.float32).reshape(1, -1),
        mesh, placements=[Shard(0)])
    pool = torch.arange(32, dtype=torch.float32)
    average, squared = torch.ones(8), torch.ones(8)
    bindings = [('model', 'parameters', pool[:8]),
        ('model_alias', 'parameters', pool[:8]),
        ('gradient', 'gradients', pool[8:16]),
        ('main_parameter', 'optimizer', pool[16:24]),
        ('exp_avg', 'optimizer', average), ('exp_avg_sq', 'optimizer', squared),
        ('distributed_parameter', 'parameters', parameter)]
    inventory = tensor_storage_inventory(bindings, device_type='cpu')
    local_elements = parameter.to_local().numel()
    expected = 128 + 32 + 32 + local_elements * 4
    assert inventory['unique_storage_bytes'] == expected
    assert inventory['logical_tensor_bytes'] == dict(
        parameters=32 + local_elements * 4, gradients=32, optimizer=96)
    pool_row = next(row for row in inventory['storages']
        if row['storage_pointer'] == pool.untyped_storage().data_ptr())
    assert set(pool_row['categories']) == {'parameters', 'gradients', 'optimizer'}
    assert set(pool_row['tensor_keys']) == {'model', 'model_alias', 'gradient', 'main_parameter'}
    Path(output, f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,
        local_shard_elements=local_elements, expected_unique_storage_bytes=expected,
        inventory=inventory), indent=2) + '\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    runtime = ast.parse((project / 'runtime.py').read_text())
    calls = [node for node in ast.walk(runtime) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == 'self.memory_audit.inventory_fsdp']
    assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == 'context.model'
    audit_source = (project / 'memory_audit.py').read_text()
    audit = ast.parse(audit_source)
    inventory = next(node for node in ast.walk(audit)
        if isinstance(node, ast.FunctionDef) and node.name == 'inventory_fsdp')
    assert any(isinstance(node, ast.Call)
        and ast.unparse(node.func) == 'self.evidence.fsdp_model_bindings'
        for node in ast.walk(inventory))
    assert not any(isinstance(node, ast.Attribute) and node.attr == 'gbuf_ranges'
        for node in ast.walk(inventory))
    compile(audit_source, str(project / 'memory_audit.py'), 'exec')
    rendezvous = (output / ('gloo-' + uuid.uuid4().hex)).as_uri()
    multiprocessing.spawn(verify_rank, args=(rendezvous, str(output)), nprocs=2, join=True)
    ranks = [json.loads((output / f'rank{rank}.json').read_text()) for rank in range(2)]
    report = dict(status='PASS', actual_processes=2, allocation_device='cpu',
        shared_storage_deduplicated=True, categories_preserved=True,
        empty_dtensor_local_shard_covered=any(row['local_shard_elements'] == 0 for row in ranks),
        fsdp_wrapper_entrypoint='context.model', resolver='RuntimeEvidence.fsdp_model_bindings',
        wrapper_entrypoint_validation='actual_source_ast', ranks=ranks)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', report=str(output / 'report.json'))))


if __name__ == '__main__':
    main()
