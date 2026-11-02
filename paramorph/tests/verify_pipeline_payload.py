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

from communication_calibration import allocate_pipeline_payload


def verify_rank(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    mesh = DeviceMesh('cpu', torch.arange(2))
    source = distribute_tensor(torch.arange(1048576, dtype=torch.float32).reshape(1, -1),
        mesh, placements=[Shard(0)])
    local = source.to_local()
    assert source.numel() == 1048576
    assert local.numel() == (1048576 if rank == 0 else 0)
    measurements = []
    for dtype in (torch.bfloat16, torch.float32):
        shape = [1024, 1, 1024]
        payload, receive = allocate_pipeline_payload(shape, dtype, torch.device('cpu'))
        assert type(payload) is torch.Tensor
        assert list(payload.shape) == shape
        assert payload.dtype == dtype and receive.dtype == dtype
        assert payload.device == torch.device('cpu')
        assert payload.data_ptr() != receive.data_ptr()
        assert payload.data_ptr() != local.data_ptr()
        assert payload.is_contiguous() and receive.is_contiguous()
        receive.fill_(1)
        assert torch.count_nonzero(payload).item() == 0
        assert source.to_local().numel() == local.numel()
        measurements.append(dict(dtype=str(dtype), shape=shape,
            bytes=payload.numel() * payload.element_size(), independent_storage=True))
    Path(output, f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,
        source_type=type(source).__name__, local_shard_elements=local.numel(),
        payloads=measurements), indent=2) + '\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = Path(__file__).resolve().parents[1] / 'communication_calibration.py'
    tree = ast.parse(module.read_text())
    calibration = next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'calibrate_pipeline_links')
    assert not any(isinstance(node, ast.Attribute) and node.attr == 'parameters'
        for node in ast.walk(calibration))
    rendezvous = (output / ('gloo-' + uuid.uuid4().hex)).as_uri()
    multiprocessing.spawn(verify_rank, args=(rendezvous, str(output)), nprocs=2, join=True)
    ranks = [json.loads((output / f'rank{rank}.json').read_text()) for rank in range(2)]
    report = dict(status='PASS', actual_processes=2, transport='gloo',
        allocation_device='cpu', cuda_nccl_timing_measured=False,
        empty_local_shard_covered=any(row['local_shard_elements'] == 0 for row in ranks),
        model_parameter_access_removed=True, ranks=ranks)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', report=str(output / 'report.json'))))


if __name__ == '__main__':
    main()
