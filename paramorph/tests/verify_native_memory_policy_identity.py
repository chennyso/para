import argparse
import json
from pathlib import Path
import sqlite3

from predictor.profile_database import ProfileDatabase, normalized_execution, plan_signature, same_training_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = args.workspace/'parameld-cp-fsdp/qwen25-s8192'
    identifiers = ('candidate-b34a0f1cf9b3c93b42dbf59e', 'candidate-0f193170d54f4e394c14046c',
        'candidate-7996c613799f12f0f039b6e7', 'candidate-a52da043a9b61edb2b0848f1',
        'candidate-7dcfd52633e0878feb18a217')
    with sqlite3.connect((case/'experiments.sqlite').as_uri()+'?mode=ro', uri=True) as connection:
        requests = [json.loads(connection.execute('SELECT document FROM trials WHERE id=?',
            (identifier,)).fetchone()[0])['plan']['backend_config'] for identifier in identifiers]
    first, second = requests[:2]
    assert first.get('memory_policy') is None and second['memory_policy']['recompute_granularity'] == 'full'
    assert same_training_plan(first, second)
    assert plan_signature(first) == plan_signature(second)
    assert normalized_execution(second)['memory_policy'] is None
    args.output.mkdir(parents=True, exist_ok=False)
    clone = args.output/'costs.sqlite'
    with sqlite3.connect((args.workspace/'shared-costs.sqlite').as_uri()+'?mode=ro', uri=True) as source:
        with sqlite3.connect(clone) as destination:
            source.backup(destination)
    database = ProfileDatabase(clone)
    matched = database.select(first)
    assert matched is not None and database.select(second) == matched
    assert database.matches(matched, second)
    previous = {row['profile_id'] for row in database.reusable_compute_profiles(first)}
    current = {row['profile_id'] for row in database.reusable_compute_profiles(second)}
    assert previous and previous == current
    for changed in requests[2:]:
        assert not same_training_plan(first, changed)
        assert plan_signature(first) != plan_signature(changed)
    database.connection.close()
    report = dict(status='PASS', real_trial_ids=identifiers, matched_profile_id=matched,
        reusable_compute_profiles=len(current), distinct_real_trials=identifiers[2:],
        actual_gpu_launches=0, source_database_modified=False)
    (args.output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
