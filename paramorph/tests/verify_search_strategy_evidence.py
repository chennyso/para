import argparse
from copy import deepcopy
import gzip
import json
from math import isfinite
from pathlib import Path
import sqlite3
import time

from ortools.sat.python import cp_model

from contracts import SystemConfig, fingerprint
from core_data_models import HardwareTopology
from cost.runtime_calibration import configuration
from graph.bottleneck_analysis import BottleneckGraph
from graph.capability_registry import CapabilityRegistry
from graph.complete_rewrite import CompleteAndRewrite
from graph.model_graph import read_model
from graph.strategy_impact import GraphStrategyImpact
from predictor.fast_predictor import FastPredictor
from predictor.profile_database import plan_signature
from search.complete_partition import CompletePartitionSearch
from search.execution_path_model import ExecutionPathModel
from search.graph_action_compiler import GraphActionCompiler, GraphGuidedIntent
from search.investigation_control import InvestigationBudget, ParallelInvestigationRanges
from search.graph_candidate_planner import GraphGuidedCandidatePlanner
from search.planning_portfolio import PortfolioControl
from search.system_search import EvidenceGuidedSearch
from search.tp_measurements import compare_tp_increase, tp_selection_exclusion
from storage.experiment_store import ExperimentStore


def documents(path, table):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as database:
        return [json.loads(row[0]) for row in database.execute(f'SELECT document FROM {table}')]


def load(path):
    return json.loads(path.read_text())


def independent_trial(root, round_number):
    identifier = f'{round_number:02d}'
    directory = root / 'parameld-independent' / f'round-{identifier}'
    run = root / 'runs' / f'parameld-gpt26-independent-{identifier}'
    selection = load(directory / 'selection.json')
    catalog = load(directory / 'catalog.json')
    request = catalog['candidates'][selection['chosen']['candidate_id']]['request']
    workload = load(root / 'matched-contract.json')
    architecture = load(directory / 'model.json')
    request = dict(request, model='GPT-3 2.6B')
    receipt = load(run / 'receipt.json')
    assert receipt['formal_starts'] == 1 and receipt['status'] == 'PASS'
    losses = [set() for _ in range(receipt['steps'])]
    for rank in range(request['node_count'] * request['gpus_per_node']):
        steps = [json.loads(line) for line in (run / f'rank{rank}.steps.jsonl').read_text().splitlines()]
        assert [row['step'] for row in steps] == list(range(1, receipt['steps'] + 1))
        assert all(row['rank'] == rank and not row['skipped'] and isfinite(row['gradient_norm']) for row in steps)
        for index, row in enumerate(steps):
            losses[index].update(row['loss'].values())
    assert all(values and all(isfinite(value) for value in values) for values in losses)
    receipt['performance_eligible'] = True
    return dict(trial_id=run.name, plan=dict(backend_config=request,
        model_fingerprint=fingerprint(architecture), workload_fingerprint=fingerprint(workload),
        hardware_fingerprint=fingerprint((root / 'parameld-independent/gpus.csv').read_text()),
        backend_fingerprint=fingerprint(workload)), measurement=receipt)


def verify_recorded_tp_regression(root, output):
    baseline, slower = (independent_trial(root, number) for number in (2, 3))
    record = compare_tp_increase(baseline, slower)
    assert record is not None and record['throughput_change_fraction'] < -0.5
    assert {'tp', 'pp', 'dp'} <= record['changed_fields'].keys()
    assert record['interpretation'] == 'JOINT_STRATEGY'
    store = ExperimentStore(output / 'search.sqlite')
    first = store.add_search_state('MEASURED_TP_REGRESSION', record)
    assert first == store.add_search_state('MEASURED_TP_REGRESSION', record)
    logged, = store.search_states('MEASURED_TP_REGRESSION')
    assert tp_selection_exclusion(baseline, slower, [logged])['maximum_tp'] == 1
    assert tp_selection_exclusion(baseline, baseline, [logged]) is None
    return dict(regression_fraction=record['throughput_change_fraction'], excluded_tp=slower['plan']['backend_config']['tp'])


def verify_real_replay(directory):
    trials = documents(directory / 'experiments.sqlite', 'trials')
    profiles = documents(directory / 'costs.sqlite', 'profiles')
    profile, = profiles
    trial = next(row for row in trials if row['trial_id'] == 'schedule-common-mistral')
    request, prediction = trial['plan']['backend_config'], trial['prediction']
    path = directory / 'prediction-graphs' / Path(prediction['execution_graph']['storage_path']).name
    with gzip.open(path, 'rt') as stream:
        graph = json.load(stream)['execution_graph']
    guide = ExecutionPathModel(request, profile, graph)
    assert abs(guide.score(request['decoder_chunks']) - guide.raw_step_ms) < 0.001
    program = cp_model.CpModel()
    fixed = [program.new_constant(cost) for cost in guide.costs(request['decoder_chunks'])]
    objective = guide.attach(program, fixed)
    program.minimize(objective)
    solver = cp_model.CpSolver()
    assert solver.solve(program) == cp_model.OPTIMAL
    assert abs(solver.value(objective) / 1000 - guide.raw_step_ms) < 0.002
    native = configuration(request)
    replay = BottleneckGraph(graph).replay(config=native)
    masked = deepcopy(graph)
    task = next(row for row in masked['tasks'].values() if row['duration_ms'] > 0
        and len(row['metadata'].get('cost_components', [])) == 1)
    component, = task['metadata'].pop('cost_components')
    key, cost = component['cost_key'], component['milliseconds']
    assert abs(task['duration_ms'] - cost * component['occurrences']) < 1e-6
    task['duration_ms'] = None
    task.pop('start_ms')
    task.pop('end_ms')
    task['metadata'].update(missing_cost_keys=[key], missing_cost_occurrences={key: component['occurrences']})
    bounded = BottleneckGraph(masked).replay_bounded_costs({key: [cost, cost]}, config=native)
    assert bounded['status'] == 'MODELED_SCENARIOS' and not bounded['proved_lower_bound']
    assert all(abs(value - replay['step_time_ms']) < 1e-6 for value in bounded['step_time_scenario_ms'])
    changed = GraphStrategyImpact.compare_candidates(request, prediction, graph, request, prediction, masked)
    assert changed['changed_task_count'] == 1 and changed['changes']['step_delta_ms'] is None, changed
    strategy_impact = GraphStrategyImpact(graph, replay, prediction['memory'])
    impact = strategy_impact.describe()
    probes = strategy_impact.quantitative_probes(impact, config=native)
    assert probes and all(row['target_task_ids'] and isfinite(row['full_iteration_impact_ms']) for row in probes)
    target = next(row for row in impact['tasks'] if type(row['region']) is int and row['region'] >= 0)
    boundary = max(1, min(target['region'], len(request['decoder_chunks']) - 1))
    intent = GraphGuidedIntent(target_task_ids=[target['task_id']], affected_regions=[target['region']],
        target_resources=target['resources'], movable_boundaries=[boundary],
        evidence_refs=[impact['impact_id']], expected_effects='核对真实任务的模块边界可调整范围')
    compiler = GraphActionCompiler(impact)
    action = compiler.compile(intent, request, request, {}, {})
    assert action['preferred_boundaries'] == [boundary] and action['timing'] == 'LOCAL_REFERENCE_PATH'
    other = next(row['plan']['backend_config'] for row in trials
        if row['plan']['backend_config']['pp'] != request['pp'])
    rebuilt = compiler.compile(intent, request, other, {'pp': [other['pp']]}, {})
    assert rebuilt['timing'] == 'REBUILD_AND_REPLAY_TARGET_DAG' and rebuilt['preferred_boundaries'] == []
    return guide.summary() | dict(affected_tasks=impact['affected_task_count'], action_modes=[action['timing'], rebuilt['timing']])


def verify_unknown_partition(root, output):
    directory = root / 'parameld-independent/round-05'
    rows = load(directory / 'catalog.json')['candidates']
    request = rows['tp1-pp4-dp2-mb1-full']['request']
    model, structure = read_model(directory / 'model.json')
    topology = HardwareTopology(8, 1, 8, 32)
    registry = CapabilityRegistry(structure['module_structure'])
    rewriter = CompleteAndRewrite(model, topology, registry, registry.parallel_domains(model, topology, request))
    predictor = FastPredictor(output / 'unknown-costs.sqlite')
    checked = []

    def evaluate(candidate):
        graph = rewriter.compiler.graph(rewriter.candidate(candidate))
        checked.append((plan_signature(candidate), len(graph.tasks)))
        return predictor.needs_profile(candidate, time.perf_counter(), ['目标执行条件缺少逐模块成本'],
            partition_specific=True, persist=False)

    search = CompletePartitionSearch(model.num_layers, request, None, (), evaluate,
        time.perf_counter() + 20, [request], {}, engine='path')
    result = search.run(3, 3)
    assert result['evaluations'] >= 2 and all(tasks > 0 for _, tasks in checked)
    assert len({identifier for identifier, _ in checked}) == len(checked)
    assert all(row['step_time_ms'] is None and row['generation_objective_ms'] is None
        and row['prediction_status'] == 'NEEDS_PROFILE' for row in result['candidates'])
    return dict(layouts=[row['decoder_chunks'] for row in result['candidates']], graph_task_counts=[n for _, n in checked])


def verify_fixed_layout(root, output):
    directory = root / 'parameld-independent/round-05'
    base = load(directory / 'catalog.json')['candidates']['tp1-pp4-dp2-mb1-full']['request']
    model, structure = read_model(directory / 'model.json')
    topology = HardwareTopology(8, 1, 8, 31.8427734375)
    registry = CapabilityRegistry(structure['module_structure'])
    request = dict(base, pp=1, dp=8, expert_dp=8, decoder_chunks=[model.num_layers],
        optimizer_sharding='distributed', allowed_optimizer_sharding=['distributed', 'replicated'],
        main_grads_dtype='float16', main_params_dtype='float32', exp_avg_dtype='float32', exp_avg_sq_dtype='float32',
        recompute='none')
    rewriter = CompleteAndRewrite(model, topology, registry, registry.parallel_domains(model, topology, request))
    predictor = FastPredictor(output / 'fixed-layout-costs.sqlite')
    predictor.configure_projection(model, structure, topology, {}, cost_mode='measured_only')
    checked = []

    def evaluate(candidate):
        graph = rewriter.compiler.graph(rewriter.candidate(candidate))
        checked.append(graph)
        return predictor.evaluate(candidate, persist=False)

    result = CompletePartitionSearch(model.num_layers, request, None, (), evaluate,
        time.perf_counter() + 60, (), {}, engine='path').run(3, 3)
    candidate, = result['candidates']
    graph, = checked
    assert result['solver_seconds'] == result['expansions'] == 0
    assert result['evaluations'] == 1 and result['statuses'] == ['FIXED_LAYOUT']
    assert candidate['decoder_chunks'] == [model.num_layers] and candidate['step_time_ms'] is None
    assert candidate['prediction_status'] == 'NEEDS_PROFILE'
    assert any(task.task_type.value == 'dp_reduce' for task in graph.tasks.values())
    distributed = predictor.evaluate_memory_bound(request)['memory']
    replicated_request = dict(request, optimizer_sharding='replicated')
    replicated = predictor.evaluate_memory_bound(replicated_request)['memory']
    assert max(distributed['persistent_per_rank_bytes'].values()) < max(replicated['persistent_per_rank_bytes'].values())
    return dict(status=candidate['prediction_status'], tasks=len(graph.tasks),
        distributed=distributed['admission'], replicated=replicated['admission'])


def backup(source, destination):
    with (sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as original,
            sqlite3.connect(destination) as copy):
        original.backup(copy)


def verify_planning_investigation(directory, output):
    run = output / 'planning'
    run.mkdir()
    for filename in ('experiments.sqlite', 'costs.sqlite'):
        backup(directory / filename, run / filename)
    config = load(directory / 'system-config.json')
    config.update(output_directory=str(run), profile_database=str(run / 'costs.sqlite'),
        hardware_topology_paths=[str(directory / 'hardware-topology.json')],
        gpu_execution_enabled=False, agent_execution_enabled=False, planning_control_enabled=True,
        planning_full_strategy_enabled=True, planning_partition_engine='complete')
    planner = EvidenceGuidedSearch(SystemConfig.model_validate(config))
    gpu_hours_before = planner.store.usage().get('gpu_hours', 0)
    parent = next(row for row in planner.store.trials() if row['trial_id'] == config['request']['run_id'])
    families, _ = planner.planning_families(parent)
    visible, unique = PortfolioControl.planning_visible_families(families)
    assert unique >= len(visible) and visible
    assert any(not row['rankable'] and row['request']['optimizer_sharding'] == 'distributed' for row in visible)
    available = next(row for row in families if row['cost_status'] == 'UNKNOWN' and row['request']['pp'] > 1
        and row['request']['tp'] < parent['plan']['backend_config']['tp'] and row['request']['vpp'] == 1)
    shape = available['request']
    planner._planning_projection_reference = planner.projection_parent_context(parent['plan']['backend_config'])
    prepared, evidence_id = GraphGuidedCandidatePlanner(planner).prepare(parent, [available])
    assert evidence_id and prepared and any(row['complete_strategy']['tp'] == shape['tp'] for row in prepared)
    assert all(row['memory_admission'] != 'REJECT_OOM' for row in prepared)
    assert all(row['predicted_step_ms'] is None for row in prepared)
    prepared_ids = {row['candidate_id'] for row in prepared}
    component_evidence = planner.planning_component_evidence(
        [row for row in planner.store.trials() if row['trial_id'] in prepared_ids], [])
    assert component_evidence['missing_components']
    assert all(item['range_ms'] is None for item in component_evidence['missing_components'].values())
    action = planner.deterministic_investigation(parent, 2).model_copy(update=dict(
        parallel_ranges=ParallelInvestigationRanges(**{key: [shape[key]]
            for key in ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp')}),
        budget=InvestigationBudget(expansions=8, states_per_boundary=2, full_evaluations=3, candidates=3, seconds=60)))
    outcome = planner.execute_investigation(action)
    assert outcome['candidate_ids']
    retained = [row for row in planner.store.trials() if row['trial_id'] in outcome['candidate_ids']]
    assert any(row['prediction']['performance']['step_time_ms'] is None
        and row['plan']['backend_config']['tp'] == shape['tp'] for row in retained)
    gpu_hours_delta = planner.store.usage().get('gpu_hours', 0) - gpu_hours_before
    assert gpu_hours_delta == 0
    return dict(tp=shape['tp'], candidates=outcome['candidate_ids'],
        evaluations=outcome['actual_predictions'], additional_gpu_hours=gpu_hours_delta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--independent-root', type=Path, required=True)
    parser.add_argument('--replay-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    print(json.dumps(dict(tp=verify_recorded_tp_regression(args.independent_root, args.output),
        replay=verify_real_replay(args.replay_directory),
        unknown=verify_unknown_partition(args.independent_root, args.output),
        fixed=verify_fixed_layout(args.independent_root, args.output),
        planning=verify_planning_investigation(args.replay_directory, args.output)), ensure_ascii=False))
