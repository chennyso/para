import argparse
import json
from pathlib import Path
import sqlite3

from contracts import TrainingPlan
from predictor.ddp_buckets import bucket_layout_signature
from predictor.profile_database import effective_execution_config, plan_signature
from predictor.runtime_receipt import validate_training_result
from search.failure_registry import classify_failure
from search.system_search import EvidenceGuidedSearch
from storage.experiment_store import ExperimentStore


def read(path):
    return json.loads(path.read_text())


def verify(original, short_run):
    failed, = (original / 'training/runs').glob('candidate-4ade*/result.json')
    initial = read(failed)
    requested = read(failed.with_name('request.json'))
    manifests = [read(path) for path in sorted((failed.parent / 'artifacts').glob('rank*.manifest.json'))]
    assert initial['status'] == 'VALIDATION_FAILED' and len(manifests) == 16
    checked = dict(initial)
    validate_training_result(checked, requested, initial['source_hashes'], manifests)
    assert checked['runtime_validation_kind'] == 'EXECUTION_CONFIG_MISMATCH'
    mismatches = checked['runtime_config_mismatch']
    assert {item['field'] for item in mismatches} == {'overlap_grad_reduce', 'overlap_param_gather'}
    assert all(item['requested'] is True and len(item['actual_by_rank']) == 16
        and set(item['actual_by_rank'].values()) == {False} for item in mismatches)
    actual = dict(requested, **{item['field']: False for item in mismatches})
    assert plan_signature(requested) != plan_signature(actual)
    assert bucket_layout_signature(requested) != bucket_layout_signature(actual)
    assert effective_execution_config(actual)['overlap_param_gather'] is False
    with sqlite3.connect((original / 'experiments.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as database:
        stored_hash, encoded = database.execute("SELECT plan_hash,document FROM trials WHERE id LIKE 'candidate-4ade%'").fetchone()
        round_data = json.loads(database.execute('SELECT document FROM search_rounds WHERE id=1').fetchone()[0])
        global_ids = round_data['global_family_candidate_ids']
        global_rows = [json.loads(database.execute('SELECT document FROM trials WHERE id=?', (identifier,)).fetchone()[0])
            for identifier in global_ids]
    plan = TrainingPlan.model_validate(json.loads(encoded)['plan'])
    assert plan.plan_hash == stored_hash
    new_plan = plan.model_copy(update=dict(identity_revision='v5'))
    assert new_plan.plan_hash != new_plan.model_copy(update=dict(backend_config=actual)).plan_hash
    failure = classify_failure(checked, plan.model_dump(mode='json'), 'benchmark')
    assert failure['failure_stage'] == 'runtime_configuration' and failure['retry_eligibility'] == 'requires_changed_execution'
    assert failure['backend_source_hashes'] == initial['source_hashes']
    assert len(global_rows) == 2 and all(row['plan']['backend_config']['pp'] == 1
        and EvidenceGuidedSearch.available_for_joint_measurement(row) for row in global_rows)
    short = read(short_run / 'result.json')
    assert short['status'] == 'PASS' and all(short['effective_config'][item['field']] for item in mismatches)
    request_group = [('native-check', short['run_id'], 'validation', short['gpu_allocation_seconds'] * 16 / 3600),
        ('native-benchmark', initial['run_id'], 'benchmark', initial['gpu_allocation_seconds'] * 16 / 3600)]
    store = ExperimentStore(':memory:')
    total = sum(row[3] for row in request_group)
    assert not store.reserve_gpu_execution_group(request_group, total - 0.01, total)
    assert store.connection.execute('SELECT count(*) FROM gpu_execution_reservations').fetchone()[0] == 0
    assert store.reserve_gpu_execution_group(request_group, total, total)
    assert store.usage().get('full_training_trials', 0) == 0
    assert store.reserve_gpu_execution_group(request_group, total, total)
    print(json.dumps(dict(native_config_mismatch_ranks=len(manifests), ddp_fields=len(mismatches),
        historical_plan_hash_preserved=True, eligible_global_pp1_candidates=len(global_rows),
        atomic_validation_benchmark_reservation_verified=True), ensure_ascii=False))


def verify_resume(original, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ('experiments.sqlite', 'costs.sqlite'):
        with sqlite3.connect((original / name).resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(output / name) as destination:
                source.backup(destination)
    config = read(original / 'system-config.json')
    config.update(output_directory=str(output), profile_database=str(output / 'costs.sqlite'),
        gpu_execution_enabled=False, agent_execution_enabled=False)

    class RecoverySearch(EvidenceGuidedSearch):
        def baseline_hardware_matches(self, measurement, request, directory):
            return super().baseline_hardware_matches(dict(measurement, raw_artifacts=str(directory / 'artifacts')),
                request, directory)

    search = RecoverySearch(config)
    parent = search.initialize()
    assert parent['measurement']['performance_eligible']
    assert '-as-run-' in parent['trial_id']
    assert all(parent['plan']['backend_config'][name] is False for name in ('overlap_grad_reduce', 'overlap_param_gather'))
    assert search.store.best_measured()['trial_id'] == parent['trial_id']
    print(json.dumps(dict(verified_historical_baseline=parent['measurement']['run_id'],
        effective_ddp_overlap=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--short-run', type=Path, required=True)
    parser.add_argument('--resume-copy', type=Path)
    args = parser.parse_args()
    verify(args.original, args.short_run)
    if args.resume_copy:
        verify_resume(args.original, args.resume_copy)
