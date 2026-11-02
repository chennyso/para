import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import SystemConfig
from predictor import FastPredictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=4000)
    args = parser.parse_args()
    if args.candidates < 1:
        raise ValueError('候选数量需要为正整数')
    config = SystemConfig.model_validate_json(args.config.read_text())
    predictor = FastPredictor(config.profile_database)
    baseline = config.request['decoder_chunks']
    if len(baseline) < 2:
        raise ValueError('Region评价需要至少两个Region')
    records = []
    started = time.perf_counter()
    for offsets in itertools.product(range(-2, 3), repeat=len(baseline) - 1):
        counts = [count + offset for count, offset in zip(baseline[:-1], offsets)]
        counts.append(baseline[-1] - sum(offsets))
        if min(counts) < 1 or counts == baseline:
            continue
        request = dict(config.request, decoder_chunks=counts, run_id=f'benchmark-{len(records)}')
        result = predictor.evaluate(request, config.profile_id, persist=False, blind=True)
        if result['validation']['status'] not in ('PROJECTABLE', 'REJECT_OOM'):
            raise ValueError(f'候选缺少可评价的成本：{counts}')
        records.append(dict(decoder_chunks=counts, prediction_id=result['prediction_id'],
            seconds=result['evaluation_seconds'], step_time_ms=result['performance']['step_time_ms'],
            memory_admission=result['memory']['admission']))
        predictor.predictions.clear()
        if len(records) == args.candidates:
            break
    elapsed = time.perf_counter() - started
    if len(records) != args.candidates:
        raise ValueError('实际合法候选数量不足')
    timings = sorted(row['seconds'] for row in records)
    report = dict(model=config.request['model'], profile_id=config.profile_id,
        distinct_candidates=len(records), elapsed_seconds=elapsed,
        evaluations_per_second=len(records) / elapsed,
        p50_seconds=statistics.median(timings), p95_seconds=timings[int(0.95 * (len(timings) - 1))],
        target_4000_in_600_seconds=len(records) >= 4000 and elapsed <= 600,
        source='real_gpu_profile_with_full_candidate_simulation', candidates=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'candidates'}))
