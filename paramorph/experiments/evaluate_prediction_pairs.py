import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


def repeat_difference(left, right):
    blocks = sorted(left.keys() & right.keys())
    if len(blocks) < 3:
        return dict(status='INSUFFICIENT_INDEPENDENT_REPETITIONS', repeat_blocks=blocks,
            confidence_interval_ms=None, difference_std_ms=None)
    differences = [right[block] - left[block] for block in blocks]
    average, dispersion = mean(differences), stdev(differences)
    width = t.ppf(0.975, len(blocks) - 1) * dispersion / math.sqrt(len(blocks))
    interval = [average - width, average + width]
    return dict(status='INDISTINGUISHABLE' if interval[0] <= 0 <= interval[1] else 'DISTINGUISHABLE',
        repeat_blocks=blocks, confidence_interval_ms=interval, difference_std_ms=dispersion,
        mean_difference_ms=average)


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
            plan_signature=prediction['plan_signature'], repeat_block=row.get('independent_repeat_block'),
            cost_revision=prediction['cost_database_revision'], predicted_step_ms=step, measured_step_ms=actual,
            relative_step_error=abs(step - actual) / actual,
            relative_peak_error=(abs(max(peaks.values()) - measured_peak) / measured_peak
                if peaks and measured_peak > 0 else None),
            evaluation_seconds=prediction.get('evaluation_seconds')))
    groups = {}
    for row in accepted:
        key = (row['workload'], row['cost_revision'], row['plan_signature'])
        groups.setdefault(key, []).append(row)
    pairs = []
    for left, right in combinations(groups.values(), 2):
        a, b = left[0], right[0]
        if (a['workload'], a['cost_revision']) != (b['workload'], b['cost_revision']):
            continue
        blocks = []
        for group in (left, right):
            indexed = {row['repeat_block']: row['measured_step_ms'] for row in group if row['repeat_block'] is not None}
            if len(indexed) != sum(row['repeat_block'] is not None for row in group):
                raise ValueError('同一策略在一个独立重复配对中出现多次')
            blocks.append(indexed)
        comparison = repeat_difference(*blocks)
        predicted = mean(row['predicted_step_ms'] for row in right) - mean(row['predicted_step_ms'] for row in left)
        actual = comparison.get('mean_difference_ms',
            mean(row['measured_step_ms'] for row in right) - mean(row['measured_step_ms'] for row in left))
        pairs.append(dict(runs=[a['run_id'], b['run_id']], predicted_delta_ms=predicted,
            measured_delta_ms=actual, delta_error_ms=abs(predicted - actual),
            measured_difference=comparison,
            descriptive_order_matches=(predicted > 0) == (actual > 0) if predicted and actual else None,
            ordering_matches=(predicted > 0) == (actual > 0)
                if predicted and comparison['status'] == 'DISTINGUISHABLE' else None))
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
