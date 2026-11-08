import argparse
import json
import math
from pathlib import Path

from scheduling.native_microbatches import native_schedule_settings, schedule_delta, verify_native_schedule
from search.experiment_value import experiment_value
from search.system_search import EvidenceGuidedSearch


def load(path):
    return json.loads(path.read_text())


def archived_run(root, campaign, arm, prefix):
    run, = (root / campaign / arm / 'training' / 'runs').glob(prefix + '*')
    result = load(run / 'result.json')
    assert result['run_id'] == run.name
    return load(run / 'request.json'), result


def validate(archive, native_run):
    mimo, mimo_result = archived_run(archive, 'mimo-paired-agent', 'full', 'candidate-035ba')
    mimo_wide, mimo_wide_result = archived_run(
        archive, 'mimo-rendezvous-retry-paired-agent', 'without_dag_evidence', 'candidate-4fcd22')
    assert mimo_result['performance_eligible'] and mimo_wide_result['performance_eligible']
    grouped = schedule_delta(mimo, mimo_wide)
    assert grouped['before']['group'] == 2 and grouped['after']['group'] == 4
    assert grouped['schedule_order_changed'] and not grouped['overlap_permission_changed']
    assert grouped['first_changed_positions'][0] == dict(index=2, before=[0, 1], after=[2, 0])

    falcon_overlap, falcon_overlap_result = archived_run(
        archive, 'falcon-pvc-recovery-paired-agent', 'full', 'candidate-24733015')
    falcon_batch, falcon_batch_result = archived_run(
        archive, 'falcon-pvc-recovery-paired-agent', 'without_dag_evidence', 'candidate-661b4b')
    assert falcon_overlap_result['performance_eligible'] and falcon_batch_result['performance_eligible']
    communication = schedule_delta(falcon_batch, falcon_overlap)
    assert communication['overlap_permission_changed'] and not communication['schedule_order_changed']
    assert communication['before']['batch_p2p_comm'] and not communication['after']['batch_p2p_comm']
    assert communication['observed_overlap_ms'] is None
    try:
        native_schedule_settings(dict(falcon_overlap, batch_p2p_comm=True))
    except ValueError as error:
        assert '互斥' in str(error)
    else:
        raise AssertionError('Megatron P2P 互斥配置未被拒绝')

    failed_request, failed_result = archived_run(archive, 'mimo-paired-agent', 'full', 'candidate-9f83')
    assert failed_request['execution_backend'] == 'megatron_fsdp'
    assert not EvidenceGuidedSearch.available_for_joint_measurement(
        dict(validation=dict(load_validation=failed_result)))

    history = [dict(plan=dict(backend_config=config), measurement=result) for config, result in
               ((mimo, mimo_result), (mimo_wide, mimo_wide_result))]
    estimate = experiment_value(mimo, None, history, mimo_result['tokens_per_second'], 240)
    assert estimate['uncertainty'] == 'UNKNOWN' and estimate['expected_improvement_per_gpu_hour'] is None
    seconds = (mimo_result['gpu_allocation_seconds'] + mimo_wide_result['gpu_allocation_seconds']) / 2
    assert estimate['cost_observations'] == 2 and math.isclose(
        estimate['expected_gpu_hours'], seconds * mimo['node_count'] * mimo['gpus_per_node'] / 3600)

    request, result = load(native_run / 'request.json'), load(native_run / 'result.json')
    manifests = [load(path) for path in (native_run / 'artifacts').glob('rank*.manifest.json')]
    assert len(manifests) == request['node_count'] * request['gpus_per_node']
    assert result['performance_eligible'] and 0 < result['gpu_allocation_seconds']
    assert verify_native_schedule(request, manifests) == result['native_pipeline_schedule']
    print(json.dumps(dict(mimo_runs=[mimo_result['run_id'], mimo_wide_result['run_id']],
        falcon_runs=[falcon_batch_result['run_id'], falcon_overlap_result['run_id']],
        blocked_fsdp_run=failed_result['run_id'], native_run=result['run_id'],
        verified_rank_manifests=len(manifests), uncertainty=estimate['uncertainty']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--native-run', required=True, type=Path)
    args = parser.parse_args()
    validate(args.archive, args.native_run)
