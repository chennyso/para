import argparse
import json
from pathlib import Path
import runpy

from contracts import fingerprint
from evaluation.prediction_metrics import held_out_ranking, prediction_provenance
from search.benchmark_admission import formal_rejection
from storage.experiment_store import ExperimentStore


def read(path):
    return json.loads(path.read_text())


def verify(root):
    campaign = root / 'parameld-megatron-gpt26-v1'
    round_dir = campaign / 'round-03'
    chosen = read(round_dir / 'selection.json')
    decision = read(round_dir / 'independent-admission.json')
    first = read(root / 'runs/parameld-gpt26-megatron-formal-01/receipt.json')
    leader = read(root / 'runs/parameld-gpt26-megatron-formal-03/receipt.json')
    oom_selection = read(campaign / 'round-02/selection.json')
    oom = read(root / 'runs' / oom_selection['run_id'] / 'receipt.json')
    rejected = next(row for row in chosen['proposals'] if row['id'] != chosen['validation']['id'])
    assert oom['failure_type'] == 'CUDA_OOM' and oom['formal_starts'] == 0
    assert first['formal_starts'] == leader['formal_starts'] == 1
    assert decision['prediction_kind'] == 'INDEPENDENT_RESOURCE_AWARE_DAG'
    real = dict(validated=chosen['validation']['id'] == chosen['id'] and bool(chosen['validation_trace']),
        complete_graph=bool(decision['graph_prediction_id']),
        predicted_rate=decision['admission_throughput_floor'], leader=first['tokens_per_second'],
        memory_admitted=decision['status'] == 'ADMIT', peaks=dict(enumerate(decision['predicted_peak_bytes_by_rank'])),
        capacities={rank: decision['capacity_bytes'] for rank in range(8)})
    assert formal_rejection(**real) is None
    unauthorized = formal_rejection(**(real | dict(validated=rejected['id'] == chosen['validation']['id'],
        complete_graph=False, predicted_rate=None)))
    stale = formal_rejection(**(real | dict(leader=leader['tokens_per_second'])))
    unproven_memory = formal_rejection(**(real | dict(memory_admitted=oom['status'] == 'PASS',
        peaks=dict(enumerate(oom.get('peak_by_rank_bytes', []))))))
    assert unauthorized == 'VALIDATION_REQUIRED'
    assert stale == 'PREDICTED_NOT_FASTER_THAN_MEASURED_LEADER'
    assert unproven_memory == 'MEMORY_NOT_ADMITTED'
    return dict(accepted_run=leader['run_id'], unvalidated_candidate=rejected['id'], validation_gate=unauthorized,
        stale_leader_gate=stale, memory_evidence_run=oom['run_id'], rank_memory_gate=unproven_memory)


def verify_native_feedback(root, setup):
    functions = runpy.run_path(str(setup / 'plan-gpt26-megatron-calibration.py'))
    campaign = root / 'parameld-megatron-gpt26-v1'
    trials, history = [], []
    for number in ('03', '04'):
        folder = campaign / f'round-{number}'
        chosen = read(folder / 'selection.json')
        strategy = chosen['strategy']
        feedback = functions['native_formal_feedback'](folder, strategy)
        receipt = read(root / 'runs' / feedback['run_id'] / 'receipt.json')
        request = read(root / 'runs' / feedback['run_id'] / 'request.json')
        calibration = read(root / 'runs' / chosen['run_id'] / 'receipt.json')
        admission = request['admission']
        assert receipt['tokens_per_second'] == feedback['observed_tps']
        assert feedback['observed_tps'] != calibration['tokens_per_second']
        assert not feedback['cross_configuration_rank_evidence']
        assert admission['uncertainty'] == 'EXACT_SCOPE_NATIVE_CALIBRATION_REPLAY_NO_CROSS_STRATEGY_TRANSFER'
        trial = dict(plan_hash=fingerprint(strategy), measurement=dict(receipt,
            performance_eligible=True, step_seconds=receipt['measurement_seconds'] / request['workload']['measured_steps']),
            prediction=dict(performance=dict(step_time_ms=admission['resource_replay']['step_ms']),
                evidence=dict(profile_ids=[chosen['run_id']], calibration_scope='SAME_CONFIGURATION')))
        assert prediction_provenance(trial)['status'] == 'SAME_CONFIGURATION_CALIBRATION'
        trials.append(trial)
        history.append(dict(strategy=strategy, calibration_run=chosen['run_id'],
            calibration_outcome=calibration, formal_result=feedback))
    failed = read(campaign / 'round-05/selection.json')
    receipt = read(root / 'runs' / failed['run_id'] / 'receipt.json')
    assert receipt['failure_type'] == 'CUDA_OOM' and receipt['formal_starts'] == 0
    history.append(dict(strategy=failed['strategy'], calibration_run=failed['run_id'], calibration_outcome=receipt))
    observations = functions['memory_observations'](failed['strategy'], history)
    assert [(row['micro_batch'], row['failure_type']) for row in observations] == [(8, 'CUDA_OOM'), (4, None)]
    assert observations[1]['peak_allocated_gib'] == history[1]['formal_result']['observed_peak_gib']
    assert held_out_ranking(trials)['status'] == 'INSUFFICIENT_EVIDENCE'
    return dict(memory_observations=observations, held_out_ranking=held_out_ranking(trials)['status'])


def verify_checkpoint(root, output):
    folder = root / 'parameld-megatron-gpt26-v1/round-04'
    choice = read(folder / 'selection.json')
    admission = read(folder / 'independent-admission.json')
    store = ExperimentStore(output)
    approved = dict(round_index=4, validation_evidence_id=fingerprint(choice['validation']),
        candidates=[dict(candidate_id=choice['id'], plan_hash=fingerprint(choice['strategy']))])
    store.add_search_state('PLANNING_VALIDATED_CANDIDATES', approved)
    store.add_search_state('PLANNING_COMPONENT_PRIORITIES', dict(calibration=admission['calibration_run']))
    store.save_round(4, stage='MEASUREMENTS', controller='four_agent_investigation',
        candidate_ids=[choice['id']], validation_evidence_id=approved['validation_evidence_id'])
    store.connection.close()
    reopened = ExperimentStore(output)
    assert reopened.get_round(4)['stage'] == 'MEASUREMENTS'
    assert reopened.search_states('PLANNING_VALIDATED_CANDIDATES')[-1]['document'] == approved
    assert reopened.search_states('PLANNING_COMPONENT_PRIORITIES')[-1]['document']['calibration'] == admission['calibration_run']
    reopened.connection.close()
    return dict(checkpoint='MEASUREMENTS', retained_candidate=choice['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--setup', type=Path)
    parser.add_argument('--checkpoint-output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False))
    if args.setup:
        print(json.dumps(verify_native_feedback(args.root, args.setup), ensure_ascii=False))
    if args.checkpoint_output:
        print(json.dumps(verify_checkpoint(args.root, args.checkpoint_output), ensure_ascii=False))
