import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean


def evaluate(rows):
    accepted, unknown = [], []
    for row in rows:
        prediction, measurement = row['prediction'], row['measurement']
        if not measurement.get('performance_eligible') or measurement['status'] != 'PASS':
            continue
        if prediction['plan_signature'] != row['measured_plan_signature']:
            raise ValueError('预测与测量的完整计划身份不同')
        evidence = prediction['evidence']
        if (evidence.get('candidate_plan_excluded_from_calibration') is not True
                or evidence.get('compatible_anchor_id') == measurement['run_id']):
            raise ValueError('预测必须来自排除本候选测量的盲测')
        step = prediction['performance'].get('step_time_ms')
        if step is None or prediction['performance'].get('ranking_eligible') is False:
            unknown.append(measurement['run_id'])
            continue
        actual = measurement['step_seconds'] * 1000
        if not all(math.isfinite(value) and value > 0 for value in (step, actual)):
            raise ValueError('预测和实测步时必须是有限正数')
        peaks = prediction.get('memory', {}).get('peak_per_rank_bytes') or {}
        measured_peak = measurement['peak_allocated_gib'] * 2**30
        accepted.append(dict(run_id=measurement['run_id'], workload=row['workload_fingerprint'],
            cost_revision=prediction['cost_database_revision'], predicted_step_ms=step, measured_step_ms=actual,
            relative_step_error=abs(step - actual) / actual,
            relative_peak_error=(abs(max(peaks.values()) - measured_peak) / measured_peak
                if peaks and measured_peak > 0 else None),
            evaluation_seconds=prediction.get('evaluation_seconds')))
    pairs = []
    for a, b in combinations(accepted, 2):
        if (a['workload'], a['cost_revision']) != (b['workload'], b['cost_revision']):
            continue
        predicted = b['predicted_step_ms'] - a['predicted_step_ms']
        actual = b['measured_step_ms'] - a['measured_step_ms']
        pairs.append(dict(runs=[a['run_id'], b['run_id']], predicted_delta_ms=predicted,
            measured_delta_ms=actual, delta_error_ms=abs(predicted - actual),
            ordering_matches=(predicted > 0) == (actual > 0) if predicted and actual else None))
    order = [row['ordering_matches'] for row in pairs if row['ordering_matches'] is not None]
    return dict(predictions=accepted, unknown_runs=unknown, pairs=pairs,
        mean_relative_step_error=mean(row['relative_step_error'] for row in accepted) if accepted else None,
        pairwise_ranking_accuracy=mean(order) if order else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate(records), indent=2, ensure_ascii=False, allow_nan=False))
