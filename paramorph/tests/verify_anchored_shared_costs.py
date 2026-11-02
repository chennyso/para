import argparse
import json
import math
from pathlib import Path
import sqlite3

from contracts import SystemConfig
from graph.bottleneck_analysis import BottleneckGraph, collective_cost_bindings, replay_shared_cost_scenarios
from search.joint_planning import JointPlanning
from storage.experiment_store import ExperimentStore
from verify_projection_cost_bindings import configure, read_trials


def measured_raw_scenario(report, cost_key, value, resource_model):
    graph = BottleneckGraph(report['execution_graph'])
    bindings = collective_cost_bindings(report, cost_key)
    overrides = {task_id: graph.records[task_id]['duration_ms'] + sum(
        (value - component['milliseconds']) * component['occurrences'] for component in components)
        for task_id, components in bindings.items()}
    return graph.replay(overrides, resource_model=resource_model)['step_time_ms']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    directory = args.output.parent / 'anchored-cost-predictions'
    directory.mkdir(parents=True, exist_ok=True)
    predictor, requests = configure(args.workspace, 'mimo', directory)
    source = args.workspace / 'outputs/mimo-full-search'
    trials = {trial['trial_id']: trial for trial in read_trials(source / 'experiments.sqlite')}
    baseline_id = 'paramorph-mimo-16gpu-baseline-20261011-cfd9dffb3f97'
    baseline = trials[baseline_id]
    measured_ms = baseline['measurement']['step_seconds'] * 1000
    parent = predictor.evaluate_projection(requests[baseline_id], None, None, persist=False, blind=True)
    parent_context = dict(requests[baseline_id], _measured_step_time_ms=measured_ms,
        _measurement_run_id=baseline['measurement']['run_id'], _prediction_id=parent['prediction_id'],
        _predicted_raw_step_time_ms=parent['performance']['raw_resource_aware_step_time_ms'])
    identifiers = ['rewrite-be7b8951724c8ac5-cfd9dffb3f97',
        'rewrite-c7142217ca6f72d4-cfd9dffb3f97', 'mimo-tail-stage3-5-stage0-11-16gpu-20261011']
    reports = []
    for identity in identifiers:
        report = predictor.evaluate_projection(requests[identity], None, parent_context, persist=False, blind=True)
        if not report['performance']['graph_delta_anchored']:
            raise ValueError('实际同家族候选没有使用真实父计划差分校准')
        reports.append(report)
        (directory / f'{identity}.json').write_text(json.dumps(report, ensure_ascii=False) + '\n')
    anchors = {parent['prediction_id']: parent}
    (directory / 'parent.json').write_text(json.dumps(parent, ensure_ascii=False) + '\n')
    shared = {component['cost_key']: component['milliseconds']
        for task in parent['execution_graph']['tasks'].values()
        if task['kind'] in ('forward_compute', 'backward_compute')
        for component in task['metadata']['cost_components']}
    for report in reports:
        keys = {component['cost_key'] for task in report['execution_graph']['tasks'].values()
                for component in task['metadata']['cost_components']}
        shared = {key: value for key, value in shared.items() if key in keys}
    key = max(shared, key=lambda identity: (shared[identity], identity))
    value = shared[key]
    resource_model = predictor.projector.parameters['resource_replay']
    models = {report['prediction_id']: resource_model for report in [parent, *reports]}
    result = replay_shared_cost_scenarios(reports, key, value * 0.9, value * 1.1,
        resource_models=models, anchor_reports=anchors, top_k=3)
    if result['status'] != 'AVAILABLE' or not result['affected_anchor_task_ids']:
        raise ValueError('共享成本情景没有实际重放父图')
    checks = []
    for label, scenario in result['scenarios'].items():
        anchor_raw = measured_raw_scenario(parent, key, scenario['cost_value'], resource_model)
        for report in reports:
            candidate_raw = measured_raw_scenario(report, key, scenario['cost_value'], resource_model)
            expected = measured_ms + candidate_raw - anchor_raw
            actual = scenario['step_time_ms'][report['candidate_id']]
            if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('差分校准情景未同时扰动候选及父图')
            checks.append(dict(scenario=label, candidate_id=report['candidate_id'],
                measured_anchor_ms=measured_ms, anchor_raw_ms=anchor_raw,
                candidate_raw_ms=candidate_raw, calibrated_ms=actual))
    unavailable = replay_shared_cost_scenarios(reports, key, value * 0.9, value * 1.1,
        resource_models=models, top_k=3)
    if unavailable['status'] != 'UNAVAILABLE' or 'scenarios' in unavailable:
        raise ValueError('父图缺失时不应生成排序情景')
    ledger_path = directory / 'experiments.sqlite'
    with sqlite3.connect((source / 'experiments.sqlite').as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(ledger_path) as target:
            original.backup(target)
    planner = JointPlanning()
    planner.predictor = predictor
    planner.config = SystemConfig.model_validate(json.loads((source / 'system-config.json').read_text()))
    planner.store = ExperimentStore(ledger_path)
    planner.cost_revision = predictor.database.cost_database_revision()
    planner.store.update_trial(baseline_id, prediction=parent)
    chosen = [planner.store.update_trial(report['candidate_id'], prediction=report) for report in reports]
    probes = planner.shared_ranking_probes(chosen)
    if not probes or any(probe['status'] != 'AVAILABLE' for probe in probes):
        raise ValueError('实际uncertainty_ms未进入共享成本排序调查')
    charges_before = planner.store.usage()
    cached = planner.shared_ranking_probes(chosen)
    if probes != cached or planner.store.usage() != charges_before:
        raise ValueError('共享成本缓存重复执行或者重复收费')
    historical = json.loads((args.workspace / 'outputs/representative-predictor-validation/mimo' /
        f'{baseline_id}.prediction.json').read_text())
    historical_trial = planner.store.update_trial(baseline_id, prediction=historical)
    legacy_rejection = planner.shared_ranking_probes([historical_trial, *chosen])
    if len(legacy_rejection) != 1 or legacy_rejection[0]['status'] != 'UNAVAILABLE':
        raise ValueError('真实历史候选缺少成本绑定时不能进行子集全局排序')
    output = dict(measured_parent_run_id=baseline_id, measured_parent_step_time_ms=measured_ms,
        cost_revision=planner.cost_revision, anchored_candidate_count=len(reports),
        compute_shared_scenario=result, independent_formula_checks=checks,
        missing_parent_result=unavailable, production_probes=probes,
        incomplete_legacy_candidate_result=legacy_rejection,
        idempotent_cache_verified=True, uncertainty_ms_probe_count=len(probes))
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), anchored_candidates=len(reports),
        formula_checks=len(checks), scenario_replays=result['evaluation_count'],
        production_uncertainty_probes=len(probes), measured_anchor_ms=measured_ms), ensure_ascii=False))


if __name__ == '__main__':
    main()
