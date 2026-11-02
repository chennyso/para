import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapters.megatron_adapter import MegatronAdapter
from contracts import SystemConfig
from predictor.profile_database import ProfileDatabase


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pods', nargs='+', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    document = json.loads(args.config.read_text())
    document.update(output_directory=str(args.output.resolve()), pods=args.pods,
                    gpu_lock_path=str(args.output.resolve() / 'gpu.lock'))
    document['request']['node_count'] = len(args.pods)
    document['request']['dp'] = len(args.pods) * document['request']['gpus_per_node'] // (
        document['request']['tp'] * document['request']['cp'] * document['request']['pp'])
    config = SystemConfig.model_validate(document)
    request = dict(config.request)
    world = request['node_count'] * request['gpus_per_node']
    groups = [[left, left + request['gpus_per_node']] for left in range(request['gpus_per_node'])]
    groups = [group for group in groups if max(group) < world]
    groups.extend([list(range(start, start + size)) for size in (2, 4, world)
                   if world % size == 0 for start in range(0, world, size)])
    request.update(run_id=args.run_id, nccl_profile=True, nccl_group_sizes=[2, 4, world],
        nccl_rank_groups=groups, nccl_payloads_mib=[1, 4, 16, 64],
        warmup=3, measured=10, port=35123, max_allocation_seconds=600)
    result, directory = MegatronAdapter(config).execute(request)
    if result['status'] != 'PASS':
        raise ValueError(f'真实NCCL成本测试未通过：{directory}')
    ProfileDatabase(config.profile_database).add_communication(args.run_id, result['nccl_benchmark'])
    report = dict(status='PASS', run_id=args.run_id, world_size=world,
        gpu_hours=result['gpu_allocation_seconds'] * world / 3600,
        records=len(result['nccl_benchmark']['records']), raw_directory=str(directory))
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
