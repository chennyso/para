import argparse
import json
import math
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve(strict=True)
    config = json.loads((output / 'system-config.json').read_text())
    limit = config['budget']['max_total_gpu_hours']
    if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0:
        raise ValueError('真实实验配置中的GPU总额度非法')

    database = (output / 'experiments.sqlite').resolve(strict=True)
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as store:
        charges = dict(store.execute(
            'SELECT resource, SUM(amount) FROM charges GROUP BY resource'))
        reservations = [dict(
            identity=identity, run_id=run_id, category=category,
            hours=hours, active=active)
            for identity, run_id, category, hours, active in store.execute(
                'SELECT id, run_id, category, hours, active '
                'FROM gpu_execution_reservations ORDER BY id')]
        receipts = [json.loads(encoded) for (encoded,) in store.execute(
            'SELECT document FROM search_state_repository '
            "WHERE kind='GPU_EXECUTION_RECEIPT' ORDER BY sequence")]

    actual = charges.get('gpu_hours', 0.0)
    actual_profiles = charges.get('profiling_gpu_hours', 0.0)
    if (not math.isfinite(actual) or not math.isfinite(actual_profiles)
            or actual < 0 or actual_profiles < 0
            or actual_profiles > actual + 1e-8):
        raise ValueError('真实账本中的GPU费用或校准费用不一致')

    active = [row for row in reservations if row['active']]
    reserved = sum(row['hours'] for row in active)
    if (any(row['category'] not in ('profiling', 'validation', 'benchmark')
            or not math.isfinite(row['hours']) or row['hours'] <= 0
            for row in active) or actual + reserved > limit + 1e-8):
        raise ValueError('实际费用与活动GPU预留超出统一实验总额度')

    print(json.dumps(dict(
        actual_gpu_hours=actual,
        actual_calibration_and_validation_gpu_hours=actual_profiles,
        unified_gpu_hours_limit=limit,
        current_reserved_gpu_hours=reserved,
        available_after_reservations_gpu_hours=limit - actual - reserved,
        active_reservations=active,
        gpu_execution_receipt_count=len(receipts),
        scope='read_only_actual_sqlite_ledger_and_completed_gpu_receipts',
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
