import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapters.megatron_adapter import MegatronAdapter
from contracts import SystemConfig
from cost.runtime_calibration import configuration
from graph.model_graph import read_model
from predictor.parameter_metadata import checkpoint_parameter_counts
from region_partition.region_partitioner import RegionPartitioner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--historical-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--optimizer-offload-fraction', type=float, required=True)
    parser.add_argument('--parallelism', nargs=5, type=int, metavar=('TP', 'PP', 'EP', 'ETP', 'VPP'))
    parser.add_argument('--sequence', type=int)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--measured', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.measured < 1:
        raise ValueError('重复次数和正式测量步数必须为正整数，预热步数不能为负数')
    document = json.loads(args.config.read_text())
    document['output_directory'] = str(args.output.resolve())
    config = SystemConfig.model_validate(document)
    previous = json.loads((args.historical_run / 'artifacts/request.json').read_text())
    request = dict(config.request, **previous)
    request.update(run_id=args.run_id, paramorph_parameter_ledger=True,
        precision_aware_optimizer=True,
        optimizer_offload_fraction=args.optimizer_offload_fraction,
        offload=args.optimizer_offload_fraction > 0,
        optimizer_transfer_overlap=False, warmup=args.warmup, measured=args.measured,
        max_allocation_seconds=1200, profile=False, port=37321)
    for field in ('precision', 'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
        request[field] = config.request[field]
    if args.parallelism is not None:
        from search.seed_planner import SeedPlanner
        tp, pp, ep, etp, vpp = args.parallelism
        world = request['node_count'] * request['gpus_per_node']
        if world % (tp * pp * request['cp']) or world % (etp * ep * pp):
            raise ValueError('跨家族校验的并行度无法完整覆盖十六张GPU')
        dp, expert_dp = world // (tp * pp * request['cp']), world // (etp * ep * pp)
        request.update(tp=tp, pp=pp, ep=ep, etp=etp, vpp=vpp, dp=dp,
            expert_dp=expert_dp, dp_inner_size=dp * request['cp'], expert_dp_inner_size=expert_dp,
            group=pp if vpp > 1 else 1, overlap_p2p=vpp > 1,
            pipeline_schedule='interleaved_1f1b' if vpp > 1 else '1f1b',
            decoder_chunks=SeedPlanner.balanced_partition(sum(previous['decoder_chunks']), pp * vpp))
    if args.sequence is not None:
        request['sequence'] = args.sequence
    args.output.mkdir(parents=True, exist_ok=True)
    model, graph = read_model(config.model_config_path, config.model_parameter_metadata_path)
    candidate = configuration(request)
    regions = RegionPartitioner(model, candidate).from_counts(request['decoder_chunks'])
    counts = checkpoint_parameter_counts(graph['parameter_metadata'], candidate, regions, request)
    adapter = MegatronAdapter(config)
    repetitions = []
    for index in range(args.repeats):
        run_request = dict(request, run_id=(args.run_id if args.repeats == 1
            else f'{args.run_id}-repeat{index + 1}'))
        print(json.dumps(dict(event='START', run_id=run_request['run_id'],
            repeat=index + 1, repeats=args.repeats)), flush=True)
        result, directory = adapter.execute(run_request)
        rows = []
        for rank in range(candidate.world_size):
            path = directory / 'artifacts' / f'rank{rank}.parameter-ledger.json'
            runtime = json.loads(path.read_text())
            observed = sum(row['local_elements'] for row in runtime['parameters'])
            expected = counts['dense'][rank] + counts['expert'][rank]
            if observed != expected:
                raise ValueError(f'真实模型参数数量与Ledger不同：rank={rank}')
            rows.append(dict(rank=rank, parameter_elements=observed, difference=observed - expected,
                native_buffer_layouts=runtime['native_buffer_layouts']))
        summary = dict(status=result['status'], runtime_ledger_ranks=len(rows),
            training_result=str(directory / 'result.json'), ranks=rows)
        (directory / 'runtime-parameter-validation.json').write_text(
            json.dumps(summary, indent=2, allow_nan=False) + '\n')
        if args.repeats == 1:
            (args.output / 'runtime-parameter-validation.json').write_text(
                json.dumps(summary, indent=2, allow_nan=False) + '\n')
        repetition = dict(status=result['status'], run_id=run_request['run_id'],
            runtime_ledger_ranks=len(rows), tokens_per_second=result.get('tokens_per_second'),
            step_seconds=result.get('step_seconds'), peak_allocated_gib=result.get('peak_allocated_gib'),
            training_result=str(directory / 'result.json'))
        repetitions.append(repetition)
        (args.output / 'repeated-training-results.json').write_text(
            json.dumps(dict(requested_repeats=args.repeats, completed_repeats=len(repetitions),
                runs=repetitions), indent=2, allow_nan=False) + '\n')
        print(json.dumps(repetition), flush=True)
        if result['status'] != 'PASS':
            raise ValueError(f'参数记录采集后的真实训练失败：{directory}')
    throughput = [row['tokens_per_second'] for row in repetitions]
    summary = dict(requested_repeats=args.repeats, completed_repeats=len(repetitions),
        runs=repetitions, mean_tokens_per_second=statistics.mean(throughput),
        median_tokens_per_second=statistics.median(throughput),
        stdev_tokens_per_second=statistics.stdev(throughput) if len(throughput) > 1 else None)
    (args.output / 'repeated-training-results.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
