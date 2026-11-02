import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

from core_data_models import HardwareTopology
from graph.bottleneck_analysis import replay_shared_cost_scenarios
from graph.model_graph import read_model
from predictor.fast_predictor import FastPredictor


def read_trials(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        return [json.loads(row[0]) for row in connection.execute('SELECT document FROM trials')]


def configure(workspace, name, output):
    validation = workspace / 'outputs/representative-predictor-validation' / name
    source = workspace / 'outputs' / ('qwen3-14b-full-search' if name == 'qwen14' else 'mimo-full-search')
    settings = json.loads((source / 'system-config.json').read_text())
    database = validation / ('profiles.sqlite' if name == 'falcon' else 'costs.sqlite')
    copied = output / f'{name}-costs.sqlite'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(copied) as target:
            original.backup(target)
    predictor = FastPredictor(copied)
    config = validation / 'model-config.json' if name == 'falcon' else settings['model_config_path']
    model, graph = read_model(config)
    hardware = json.loads((source / 'hardware-topology.json').read_text())
    devices = [device | dict(node=node.get('node', node['hostname']), node_index=index,
        physical_rank=index * 8 + device['index']) for index, node in enumerate(hardware)
        for device in node['devices']]
    network = [adapter | dict(node_index=index) for index, node in enumerate(hardware)
               for adapter in node['adapters']]
    topology = HardwareTopology(16, 2, 8,
        min(device['capacity_bytes'] for device in devices) / 2**30,
        devices=devices, network_paths=network, software=hardware[0]['software'],
        environment_per_node={str(index): node['launch_environment'] for index, node in enumerate(hardware)})
    parameters = deepcopy(settings['predictor_hardware_parameters'])
    parameters.setdefault('network', {})['environment_per_node'] = topology.environment_per_node
    predictor.configure_projection(model, graph, topology, parameters)
    if name == 'falcon':
        profile_id = 'paramorph-falcon-module-profile-16gpu-20261011-r3:raw'
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            profile = json.loads(connection.execute('SELECT document FROM profiles WHERE id=?',
                (profile_id,)).fetchone()[0])
        predictor.set_preferred_profile(profile_id)
        request = dict(profile['request'], run_id='falconh1-pp8-s1024-vpp1-r2-134954', warmup=20, measured=100)
        requests = {request['run_id']: request}
    else:
        trials = read_trials(source / 'experiments.sqlite')
        if name == 'mimo':
            trials += read_trials(workspace / 'outputs/mimo-ablation-deterministic-v4/experiments.sqlite')
        requests = {}
        for trial in trials:
            measurement = trial.get('measurement')
            if measurement and measurement['status'] == 'PASS' and measurement['performance_eligible']:
                request = dict(trial['plan']['backend_config'])
                if 'embedding_gradient_allreduce' in settings['request']:
                    request.setdefault('embedding_gradient_allreduce', settings['request']['embedding_gradient_allreduce'])
                requests[request['run_id']] = request
    return predictor, requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    directory = args.output.parent / 'cost-bindings-predictions'
    directory.mkdir(parents=True, exist_ok=True)
    validation = args.workspace / 'outputs/representative-predictor-validation'
    outcomes = []
    mimo_reports = []
    resource_models = {}
    for name in ('mimo', 'qwen14', 'falcon'):
        predictor, requests = configure(args.workspace, name, directory)
        for old_path in sorted((validation / name).glob('*.prediction.json')):
            previous = json.loads(old_path.read_text())
            request = requests[previous['candidate_id']]
            report = predictor.evaluate_projection(request, None, None, persist=False, blind=name != 'falcon')
            tasks = report['execution_graph']['tasks']
            maximum = 0.0
            components = 0
            for task_id, task in tasks.items():
                values = task['metadata']['cost_components']
                total = sum(row['milliseconds'] * row['occurrences'] for row in values)
                residual = abs(total - task['duration_ms'])
                maximum = max(maximum, residual)
                components += len(values)
                if not math.isclose(total, task['duration_ms'], rel_tol=1e-9, abs_tol=1e-7):
                    raise ValueError(f'{name}/{task_id}成本组成不完整')
            old_ms = previous['performance']['raw_resource_aware_step_time_ms']
            new_ms = report['performance']['raw_resource_aware_step_time_ms']
            previous_revision = previous['evidence']['predictor_revision']
            evidence_only_revision = previous_revision == 'finalize_model_grads_projection_v8_20261011'
            destination = directory / f'{request["run_id"]}.json'
            destination.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
            if evidence_only_revision and not math.isclose(old_ms, new_ms, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f'{name}/{request["run_id"]}的v8与v9性能不一致：{old_ms} -> {new_ms}')
            outcome = dict(model=name, candidate_id=request['run_id'], tasks=len(tasks),
                cost_components=components, maximum_component_sum_error_ms=maximum,
                previous_step_time_ms=old_ms, current_step_time_ms=new_ms,
                previous_prediction_revision=previous_revision,
                same_execution_semantics_revision=evidence_only_revision,
                prediction_revision=report['evidence']['predictor_revision'], prediction_path=str(destination))
            outcomes.append(outcome)
            print(json.dumps(outcome, ensure_ascii=False), flush=True)
            if name == 'mimo':
                mimo_reports.append(report)
                resource_models[report['candidate_id']] = predictor.projector.parameters['resource_replay']
    candidates = defaultdict(set)
    costs = {}
    for report in mimo_reports:
        for task in report['execution_graph']['tasks'].values():
            if task['kind'] not in ('forward_compute', 'backward_compute'):
                continue
            for component in task['metadata']['cost_components']:
                candidates[component['cost_key']].add(report['candidate_id'])
                costs[component['cost_key']] = component['milliseconds']
    shared = [key for key, values in candidates.items() if len(values) > 1]
    if not shared:
        raise ValueError('新预测记录缺少跨候选共享的复合计算成本')
    key = max(shared, key=lambda item: (len(candidates[item]), costs[item], item))
    ranking = replay_shared_cost_scenarios(mimo_reports, key, costs[key] * 0.9, costs[key] * 1.1,
        resource_models=resource_models, top_k=3)
    if ranking['reuse_count'] != len(candidates[key]):
        raise ValueError('共享计算成本情景没有覆盖全部关联候选')
    if not all(ranking['affected_task_ids'][candidate] for candidate in candidates[key]):
        raise ValueError('共享计算成本情景缺少实际任务绑定')
    result = dict(verified_predictions=len(outcomes), predictions=outcomes,
        shared_compute_cost_sensitivity=ranking,
        perturbation=dict(cost_key=key, baseline_ms=costs[key], low_multiplier=0.9, high_multiplier=1.1,
            scope='explicit_cost_sensitivity_scenarios', statistical_confidence_interval=False))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), verified_predictions=len(outcomes),
        shared_compute_candidates=ranking['reuse_count'], actual_replays=ranking['evaluation_count']),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
