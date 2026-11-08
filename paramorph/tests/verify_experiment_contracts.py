import argparse
import json
import math
from pathlib import Path
import sqlite3

from agents.evidence_policy import AblationEvidencePolicy
from contracts import TrainingPlan
from core_data_models import HardwareTopology
from graph.capability_registry import CapabilityRegistry
from graph.complete_rewrite import CompleteAndRewrite
from graph.model_graph import read_model
from predictor.profile_database import plan_signature
from predictor.runtime_receipt import effective_runtime_receipt, validate_training_result
from predictor.stage_model import aggregate_stage_cost, project_stage_cost
from scheduling.native_schedule_lowerer import verify_native_execution_events
from search.failure_registry import classify_failure
from search.system_search import EvidenceGuidedSearch
from storage.experiment_store import ExperimentStore


def load(path):
    return json.loads(path.read_text())


def read_rows(path, statement):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as database:
        return [json.loads(row[0]) for row in database.execute(statement)]


def verify_resume(campaign, directory):
    source = campaign / 'mistral/full'
    directory.mkdir(parents=True, exist_ok=False)
    for filename in ('experiments.sqlite', 'costs.sqlite'):
        with sqlite3.connect((source / filename).resolve().as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(directory / filename) as isolated:
                original.backup(isolated)
    config = load(source / 'effective-system-config.json')
    config.update(output_directory=str(directory), profile_database=str(directory / 'costs.sqlite'),
        hardware_topology_paths=[str(source / 'hardware-topology.json')], gpu_execution_enabled=False,
        agent_execution_enabled=False)
    search = EvidenceGuidedSearch(config)
    parent = search.initialize()
    assert parent['measurement']['performance_eligible']
    current = search.shared_context(parent)['current_training_plan']
    assert current['evidence_id'] != current['plan_hash']
    assert search.shared_context(parent)['current_training_plan']['evidence_id'] == current['evidence_id']
    base = parent['plan']['backend_config']
    double_dp = dict(dp=2, expert_dp=2, dp_inner_size=2, expert_dp_inner_size=2)
    variants = dict(tp=dict(tp=2, etp=2, **double_dp),
        pp=dict(pp=2, decoder_chunks=[20, 20], **double_dp),
        vpp=dict(vpp=2, group=4, decoder_chunks=[5] * 8), micro_batch=dict(micro_batch=2),
        partition=dict(decoder_chunks=[9, 10, 10, 11]))
    stage_predictions = {}
    for axis, changes in variants.items():
        request = dict(base, **changes)
        search.rewriter.candidate(request)
        estimate, costs = project_stage_cost(request, search.predictor.projector)
        if estimate is None:
            reasons = sorted({row['reason'] for row in costs if row.get('milliseconds') is None and row.get('reason')})
            assert reasons, axis
            stage_predictions[axis] = dict(status='NEEDS_PROFILE', reasons=reasons)
        else:
            assert estimate['step_time_ms'] > 0, axis
            stage_predictions[axis] = estimate['step_time_ms']
    return dict(training_plan_evidence=current['evidence_id'], stage_model_variants_ms=stage_predictions)


def verify(campaign, archive, output):
    config = load(campaign / 'mimo/full/effective-system-config.json')
    model, structure = read_model(config['model_config_path'],
        parameter_metadata_path=config.get('model_parameter_metadata_path'))
    rank_zero, = (campaign / 'mimo/full/training/runs').glob('candidate-b6a640*/result.json')
    result = load(rank_zero)
    request = load(rank_zero.with_name('request.json'))
    manifests = [load(path) for path in sorted((rank_zero.parent / 'artifacts').glob('rank*.manifest.json'))]
    receipt = effective_runtime_receipt(request, result['source_hashes'], manifests)
    assert receipt['runtime_rank_count'] == 16 and receipt['effective_config']['overlap_p2p'] is True
    assert receipt['effective_config']['group'] == 2
    for proposed, actual in ((dict(request, group=4), manifests), (request, manifests[1:])):
        invalid = dict(result)
        validate_training_result(invalid, proposed, result['source_hashes'], actual)
        assert invalid['status'] == 'VALIDATION_FAILED' and not invalid['performance_eligible']

    topo = HardwareTopology(16, 2, 8, 31.84)
    registry = CapabilityRegistry(structure['module_structure'])
    rewriter = CompleteAndRewrite(model, topo, registry, registry.parallel_domains(model, topo, request))
    original = rewriter.candidate(request)
    grouped = rewriter.candidate(dict(request, group=4))
    blocking = rewriter.candidate(dict(request, overlap_p2p=False, batch_p2p_comm=True))
    grouping = rewriter.compiler.graph_delta(original, grouped)['changed_schedule_edges']
    communication = rewriter.compiler.graph_delta(blocking, original)['changed_schedule_edges']
    assert grouping['added'] and grouping['removed']
    assert communication['removed']
    flushed = rewriter.candidate(dict(request, overlap_p2p_comm_warmup_flush=True))
    flush_delta = rewriter.compiler.graph_delta(original, flushed)['changed_schedule_edges']
    assert flush_delta['removed']
    assert plan_signature(request) != plan_signature(dict(request, group=4))
    initial, = (campaign / 'mimo/common/training/runs').glob('*/request.json')
    initial = load(initial)
    assert initial['vpp'] == 1 and plan_signature(initial) == plan_signature(dict(initial, group=4))

    profiles = read_rows(campaign / 'mistral/without_dag_evidence/costs.sqlite', 'SELECT document FROM profiles')
    estimate = aggregate_stage_cost(profiles[0]['request'], profiles[0])
    assert estimate is not None and estimate['step_time_ms'] > 0
    assert AblationEvidencePolicy('without_dag_model').observation(dict(graph_delta=grouping)) == {}

    failed, = (archive / 'mimo-paired-agent/full/training/runs').glob('candidate-9f83*/result.json')
    failed_result = load(failed)
    records = read_rows(archive / 'mimo-paired-agent/full/experiments.sqlite',
        "SELECT document FROM trials WHERE id LIKE 'candidate-9f83%'")
    plan, = [row['plan'] for row in records]
    failure = classify_failure(failed_result, plan, 'load_validation')
    assert failure['failure_type'] == 'NUMERICAL_CORRECTNESS'
    assert failure['failure_stage'] == 'optimizer_update'
    assert failure['retry_eligibility'] == 'requires_changed_execution'

    store = ExperimentStore(output)
    identity = failure['effective_plan_hash'] + ':validation'
    before = store.usage().get('gpu_hours', 0)
    declared = failed_result['max_allocation_seconds'] * 16 / 3600
    assert store.reserve_gpu_execution(identity, failed_result['run_id'], 'validation',
        declared, before + declared, before + declared)
    benchmark_hours = result['max_allocation_seconds'] * 16 / 3600
    benchmark_row, = read_rows(campaign / 'mimo/full/experiments.sqlite',
        "SELECT document FROM trials WHERE id LIKE 'candidate-b6a640%'")
    trial, _ = store.put_trial(TrainingPlan.model_validate(benchmark_row['plan']))
    assert not store.reserve_gpu_execution(trial['plan_hash'], result['run_id'], 'benchmark',
        benchmark_hours, before + declared, before + declared)
    assert store.usage().get('full_training_trials', 0) == 0 and store.trials()[0]['state'] == 'PENDING'
    first = store.record_gpu_receipt(identity, failed_result['run_id'], 'validation', 16, failed_result)
    assert store.record_gpu_receipt(identity, failed_result['run_id'], 'validation', 16, failed_result) == first
    assert math.isclose(store.usage()['gpu_hours'] - before, first)
    assert store.reserve_gpu_execution(trial['plan_hash'], result['run_id'], 'benchmark',
        benchmark_hours, before + first + benchmark_hours, before + declared)
    for _ in range(2):
        store.record_gpu_launch(trial['plan_hash'], 'benchmark', trial['trial_id'])
    assert store.usage()['full_training_trials'] == 1 and store.trials()[0]['state'] == 'RUNNING'
    store.record_gpu_receipt(trial['plan_hash'], result['run_id'], 'benchmark', 16, result)
    native_events = {}
    for model_name, pattern in (('mistral', '*'), ('mimo', 'profile-4ef0281*')):
        profile, = (campaign / f'{model_name}/full/profiling/runs').glob(f'{pattern}/artifacts/rank0.paramorph-cost.json')
        profile_request = load(profile.parent.parent / 'request.json')
        native_events[model_name] = verify_native_execution_events(profile_request,
            (load(path) for path in profile.parent.glob('rank*.paramorph-cost.json')))
    print(json.dumps(dict(native_rank_manifests=len(manifests), grouping_edges=len(grouping['added']),
        p2p_blocking_edges_removed=len(communication['removed']), stage_model_step_ms=estimate['step_time_ms'],
        failure_type=failure['failure_type'], charged_gpu_hours=first, native_execution_events=native_events), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-workspace', type=Path)
    args = parser.parse_args()
    verify(args.campaign, args.archive, args.output)
    if args.resume_workspace:
        print(json.dumps(dict(resumed_training_plan_evidence=verify_resume(args.campaign, args.resume_workspace))))
