import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts import SystemConfig
from adapters.megatron_adapter import MegatronAdapter
from predictor.profile_database import plan_signature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numerical', action='store_true')
    parser.add_argument('--run-suffix', default='')
    args = parser.parse_args()
    config = json.loads((args.search / 'system-config.json').read_text())
    report = json.loads((args.search / 'search-result.json').read_text())
    best = report['best_measured']
    if best['measurement']['status'] != 'PASS' or not best['measurement']['performance_eligible']:
        raise ValueError('最佳计划需要通过真实正式性能测量')
    config['output_directory'] = str(args.output.resolve())
    config = SystemConfig.model_validate(config)
    request = dict(best['plan']['backend_config'])
    request['run_id'] += '-numerical' if args.numerical else '-confirmation'
    request['run_id'] += args.run_suffix
    request['port'] = 34991 if args.numerical else 34992
    request.update(warmup=1 if args.numerical else 10, measured=2 if args.numerical else 30)
    result, directory = MegatronAdapter(config).execute(request, numerical=args.numerical)
    if result['status'] != 'PASS':
        raise ValueError(f'最佳计划重新执行失败：{directory}')
    if args.numerical and not result.get('numerical_equivalence_certified', False):
        raise ValueError('完整训练数值验证没有通过')
    if plan_signature(result) != plan_signature(best['plan']['backend_config']):
        raise ValueError('验收结果与最佳策略的完整执行身份不同')
    args.output.mkdir(parents=True, exist_ok=True)
    performance = best['measurement']
    gpu_count = request['node_count'] * request['gpus_per_node']
    document = dict(best_plan_hash=best['plan_hash'], source_search=str(args.search.resolve()),
        verification='native_schedule_repeatability' if args.numerical else 'fresh_performance_confirmation',
        cross_partition_numerical_equivalence_certified=False, result=result,
        acceptance=dict(performance_run_id=performance['run_id'],
            measured_tokens_per_second=performance['tokens_per_second'],
            performance_gpu_hours=gpu_count * performance['gpu_allocation_seconds'] / 3600,
            verification_gpu_hours=gpu_count * result['gpu_allocation_seconds'] / 3600,
            numerical_run_id=result['run_id'] if args.numerical else None,
            numerical_verified=args.numerical and result['numerical_equivalence_certified']))
    (args.output / f'{result["run_id"]}.verification.json').write_text(json.dumps(document, ensure_ascii=False,
        indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('run_id', 'status', 'step_seconds',
        'tokens_per_second', 'peak_allocated_gib', 'numerical_equivalence_certified')}))
