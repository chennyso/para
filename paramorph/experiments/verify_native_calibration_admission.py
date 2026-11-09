import argparse
import gzip
import json
from pathlib import Path
import sqlite3

from predictor.profile_database import (
    backend_reuse_evidence, profile_execution_request, same_training_plan,
)
from search.native_calibration_admission import admit_exact_native_calibration


def open_read_only(path):
    resolved = Path(path).resolve(strict=True)
    return sqlite3.connect(resolved.as_uri() + '?mode=ro', uri=True)


def full_graph_prediction(prediction):
    report = dict(prediction)
    graph = prediction.get('execution_graph')
    if graph is None or 'tasks' in graph:
        return report
    path = Path(graph['storage_path']).resolve(strict=True)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        archive = json.load(stream)
    report['execution_graph'] = archive['execution_graph']
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-prefix', required=True)
    args = parser.parse_args()

    output = args.output.resolve(strict=True)
    configuration = json.loads((output / 'system-config.json').read_text())
    with open_read_only(output / 'experiments.sqlite') as store:
        candidates = [json.loads(row[0]) for row in store.execute(
            'SELECT document FROM trials')]
    candidates = [trial for trial in candidates if any(
        str(value).startswith(args.candidate_prefix)
        for value in (trial['trial_id'], trial['plan_hash'],
                      trial['plan']['backend_config']['run_id']))]
    if len(candidates) != 1:
        raise ValueError('真实实验数据库需要唯一匹配的完整候选')
    trial = candidates[0]
    request = trial['plan']['backend_config']
    original = full_graph_prediction(trial['prediction'])

    with open_read_only(configuration['profile_database']) as database:
        profiles = [dict(id=identifier, **json.loads(encoded),
            registered_source_directory=directory)
            for identifier, encoded, directory in database.execute(
                'SELECT profiles.id, profiles.document, profile_sources.directory '
                'FROM profiles LEFT JOIN profile_sources '
                'ON profiles.id=profile_sources.profile_id')]
    matching = []
    for profile in profiles:
        profile['request'] = profile_execution_request(
            profile, profile['registered_source_directory'])
        if not same_training_plan(profile['request'], request):
            continue
        if backend_reuse_evidence(request, profile)['status'] == 'MISMATCHED':
            continue
        matching.append(profile)
    if not matching:
        raise ValueError('候选缺少真实、相同完整训练配置的原生成本校准')

    accepted = []
    for profile in matching:
        result = admit_exact_native_calibration(
            original, request, profile['id'], profile)
        if result is not original:
            accepted.append((result, profile))
    if not accepted:
        raise ValueError('真实原生校准没有形成足够的干净时间窗口和显存记录')
    accepted.sort(key=lambda row: len(row[0]['evidence'][
        'native_calibration_admission']['stable_step_ids']), reverse=True)
    report, profile = accepted[0]
    if report['prediction_status'] != 'PREDICTED_CALIBRATED':
        raise ValueError('成本校准候选未获得独立标注的有效预测状态')
    if report['memory']['admission'] != 'WITHIN_ESTIMATED_CAPACITY':
        raise ValueError('实际相同配置的训练峰值仍未获得容量准入')

    evidence = report['evidence']['native_calibration_admission']
    print(json.dumps(dict(
        trial_id=trial['trial_id'],
        source_profile_id=profile['id'],
        calibration_step_ms=evidence['measured_step_time_ms'],
        calibration_tokens_per_second=report['performance']['tokens_per_second'],
        original_raw_dag_step_ms=report['performance'][
            'raw_resource_aware_step_time_ms'],
        physical_peak_bytes=report['memory']['peak_per_rank_bytes'],
        memory_admission=report['memory']['admission'],
        clean_step_count=len(evidence['stable_step_ids']),
        independent_prediction_error_eligible=False,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
