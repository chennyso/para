import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from core_data_models import HardwareTopology
from graph.model_graph import read_model
from predictor.fast_predictor import FastPredictor


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, document, compact=False):
    options = dict(separators=(',', ':')) if compact else dict(indent=2)
    path.write_text(json.dumps(document, ensure_ascii=False, **options) + '\n')


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_documents(path, table):
    if table not in {'trials', 'profiles', 'communication_profiles'}:
        raise ValueError('不允许读取未列出的数据表')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        return [json.loads(row[0]) for row in connection.execute(f'SELECT document FROM {table}')]


def database_snapshot(source, destination):
    if destination.exists():
        raise FileExistsError(f'验收数据库已经存在：{destination}')
    before = file_digest(source)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(destination) as copied:
            original.backup(copied)
    return dict(source_path=str(source), source_sha256=before,
                snapshot_path=str(destination), snapshot_sha256=file_digest(destination))


def configure(workspace, name, output, hardware_path=None):
    validation = workspace / 'outputs/representative-predictor-validation' / name
    source = workspace / 'outputs' / ('qwen3-14b-full-search' if name == 'qwen14' else 'mimo-full-search')
    settings = read_json(source / 'system-config.json')
    database = validation / ('profiles.sqlite' if name == 'falcon' else 'costs.sqlite')
    copied = output / f'{name}-costs.sqlite'
    snapshot = database_snapshot(database, copied)
    predictor = FastPredictor(copied)
    model_path = validation / 'model-config.json' if name == 'falcon' else Path(settings['model_config_path'])
    model, graph = read_model(model_path)
    hardware_path = source / 'hardware-topology.json' if hardware_path is None else Path(hardware_path)
    hardware = read_json(hardware_path)
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
    comparison = read_json(validation / 'comparison.json')
    comparisons = {row['run_id']: row for row in (comparison if isinstance(comparison, list) else [comparison])}
    if name == 'falcon':
        profile_id = comparison['profile_id']
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            profile = json.loads(connection.execute('SELECT document FROM profiles WHERE id=?',
                (profile_id,)).fetchone()[0])
        predictor.set_preferred_profile(profile_id)
        request = dict(profile['request'], run_id=comparison['run_id'], warmup=20, measured=100)
        requests = {request['run_id']: request}
    else:
        trials = read_documents(source / 'experiments.sqlite', 'trials')
        if name == 'mimo':
            trials += read_documents(workspace / 'outputs/mimo-ablation-deterministic-v4/experiments.sqlite', 'trials')
        requests = {}
        for trial in trials:
            measurement = trial.get('measurement')
            if measurement and measurement['status'] == 'PASS' and measurement['performance_eligible']:
                request = dict(trial['plan']['backend_config'])
                if 'embedding_gradient_allreduce' in settings['request']:
                    request.setdefault('embedding_gradient_allreduce', settings['request']['embedding_gradient_allreduce'])
                requests[request['run_id']] = request
    snapshot.update(model_config_path=str(model_path), model_config_sha256=file_digest(model_path),
                    hardware_path=str(hardware_path), hardware_sha256=file_digest(hardware_path))
    save_json(output / f'{name}-frozen-inputs.json', dict(requests=requests, hardware=hardware,
              model_config=read_json(model_path), predictor_parameters=parameters, provenance=snapshot))
    return predictor, requests, comparisons, snapshot


def cost_summary(report):
    costs = report['evidence'].get('costs', [])
    summary = defaultdict(lambda: dict(records=0, milliseconds_sum=0.0, missing_records=0))
    for cost in costs:
        key = (cost['status'], cost.get('source', cost.get('time_source', 'unspecified')))
        row = summary[key]
        row['records'] += 1
        value = cost.get('milliseconds')
        if value is None:
            row['missing_records'] += 1
        else:
            row['milliseconds_sum'] += value
    return [dict(status=key[0], source=key[1], **value) for key, value in sorted(summary.items())]


def graph_summary(report):
    graph = report.get('execution_graph')
    if graph is None:
        return dict(available=False, reason=report['validation'].get('reason'))
    tasks = graph['tasks']
    for task_id, task in tasks.items():
        missing = set(task['dependencies']) - set(tasks)
        if missing:
            raise ValueError(f'{task_id}存在缺失依赖：{sorted(missing)}')
        components = task['metadata']['cost_components']
        total = sum(row['milliseconds'] * row['occurrences'] for row in components)
        if not math.isclose(total, task['duration_ms'], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f'{task_id}成本分量与任务耗时不一致')
    critical_ids = report['diagnosis']['critical_tasks']
    critical = [tasks[key] for key in critical_ids]
    boundaries = sorted({float(task[field]) for task in critical for field in ('start_ms', 'end_ms')})
    occupancy = defaultdict(float)
    for left, right in zip(boundaries, boundaries[1:]):
        kinds = sorted({task['kind'] for task in critical if task['start_ms'] < right and task['end_ms'] > left})
        if kinds:
            occupancy[' + '.join(kinds)] += right - left
    fsdp = []
    for task_id, task in tasks.items():
        metadata = task['metadata']
        if task['kind'].startswith('fsdp_'):
            fsdp.append(dict(task_id=task_id, kind=task['kind'], rank=task['rank'],
                duration_ms=task['duration_ms'], critical=task_id in critical_ids,
                metadata=metadata))
    return dict(available=True, task_count=len(tasks), task_kind_counts=dict(Counter(task['kind'] for task in tasks.values())),
        critical_task_count=len(critical), critical_interval_union_ms=sum(occupancy.values()),
        critical_disjoint_interval_ms=dict(occupancy),
        interval_accounting='关键任务区间按互斥时间片计算；同时活跃的类型保留组合标签',
        exposed_communication_ms=report['diagnosis'].get('modeled_exposed_communication_ms'),
        resource_replay=report['diagnosis'].get('resource_aware_replay'), fsdp_tasks=fsdp)


def benchmark_inventory(benchmark):
    grouped = defaultdict(list)
    for row in benchmark['records']:
        key = (row.get('collective', row.get('kind')), row.get('dtype', benchmark.get('dtype', 'unspecified')),
               tuple(row['ranks']))
        grouped[key].append(row)
    return dict(hardware_fingerprint=benchmark.get('hardware_fingerprint'),
        groups=[dict(collective=key[0], dtype=key[1], ranks=list(key[2]),
            payload_bytes=sorted({row['payload_bytes'] for row in rows}),
            sample_count=sum(len(row['samples']) for row in rows))
            for key, rows in sorted(grouped.items())])


def communication_inventory(workspace):
    paths = sorted((workspace / 'outputs').glob('*/training/runs/*/artifacts/nccl-costs.json'))
    entries = [dict(path=str(path), sha256=file_digest(path), **benchmark_inventory(read_json(path))) for path in paths]
    return entries


def error_percent(predicted, measured):
    return None if predicted is None else 100 * (predicted - measured) / measured


def write_tables(output, outcomes):
    fields = ['model', 'candidate_id', 'before_raw_ms', 'after_raw_ms', 'measured_ms',
              'before_error_percent', 'after_error_percent', 'after_prediction_status',
              'memory_admission', 'same_cost_database_revision']
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: 'None' if row[key] is None else row[key] for key in fields} for row in outcomes)
    completed = sum(row['graph_summary']['available'] for row in outcomes)
    missing = sum(row['after_prediction_status'] == 'NEEDS_PROFILE' for row in outcomes)
    lines = ['本报告使用冻结训练请求与已有成本数据库进行独立评价；实测数据来自已有普通训练。', '',
             f'共评价 {len(outcomes)} 份候选，其中 {completed} 份完成完整 DAG Replay，{missing} 份返回 NEEDS_PROFILE。',
             '', 'NEEDS_PROFILE 表示无法给出完整预测。None 保留为缺失值，不计为零，不计为成功预测。', '',
             'FSDP 数值来自代码修正与原有成本数据库。FP32 ReduceScatter 仍包含基于 BF16 曲线的显式 dtype-transfer 解析推估。', '',
             '| 模型 | 候选 | 修复前 ms | 修复后 ms | 实测 ms | 修复后误差 % | 当前状态 |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in outcomes:
        values = [row['model'], row['candidate_id']]
        values += ['None' if row[key] is None else f'{row[key]:.3f}' for key in
                   ('before_raw_ms', 'after_raw_ms', 'measured_ms', 'after_error_percent')]
        values.append(row['after_prediction_status'])
        lines.append('| ' + ' | '.join(values) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--models', nargs='+', choices=['mimo', 'qwen14', 'falcon'], default=['mimo', 'qwen14', 'falcon'])
    parser.add_argument('--candidate-id', action='append')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = communication_inventory(workspace)
    save_json(output / 'communication-inventory.json', inventory)
    database_inventory = {}
    for name in ('mimo', 'qwen14', 'falcon'):
        path = workspace / 'outputs/representative-predictor-validation' / name / (
            'profiles.sqlite' if name == 'falcon' else 'costs.sqlite')
        database_inventory[name] = dict(path=str(path), sha256=file_digest(path),
            benchmarks=[benchmark_inventory(benchmark) for benchmark in read_documents(path, 'communication_profiles')])
    save_json(output / 'database-communication-inventory.json', database_inventory)
    if args.inventory_only:
        print(json.dumps(dict(communication_artifacts=len(inventory), output=str(output / 'communication-inventory.json'))))
        return
    source_root = Path(__file__).resolve().parents[2]
    source_files = sorted((source_root / 'paramorph/predictor').glob('*.py'))
    source_files += sorted((source_root / 'paramorph/graph').glob('*.py'))
    source_files += [source_root / 'paramorph/scheduling/task_graph_builder.py',
                     source_root / 'paramorph/scheduling/schedule_simulator.py',
                     source_root / 'paramorph/core_data_models.py']
    algorithm_source = {str(path.relative_to(source_root)): file_digest(path) for path in source_files}
    save_json(output / 'algorithm-source.json', algorithm_source)
    baseline = read_json(workspace / 'outputs/joint-synthesis-verification/cost-bindings.json')
    outcomes = []
    snapshots = []
    for name in args.models:
        predictor, requests, comparisons, snapshot = configure(workspace, name, output)
        snapshots.append(snapshot)
        for item in baseline['predictions']:
            if item['model'] != name:
                continue
            candidate = item['candidate_id']
            if args.candidate_id is not None and candidate not in args.candidate_id:
                continue
            request = requests[candidate]
            comparison = comparisons[candidate]
            before_path = Path(item['prediction_path'])
            before = read_json(before_path)
            report = predictor.evaluate_projection(request, None, None, persist=False, blind=name != 'falcon')
            if report['plan_signature'] != before['plan_signature']:
                raise ValueError(f'{candidate}的训练计划与原始预测不一致')
            destination = output / f'{candidate}.prediction.json'
            save_json(destination, report, compact=True)
            measured = comparison['measured_ms']
            before_ms = before['performance'].get('raw_resource_aware_step_time_ms')
            after_ms = report['performance'].get('raw_resource_aware_step_time_ms')
            outcome = dict(model=name, candidate_id=candidate, frozen_request=request,
                measured_ms=measured, measurement_comparison_path=str(workspace / 'outputs/representative-predictor-validation' / name / 'comparison.json'),
                measurement_artifacts=comparison.get('raw_artifacts'), measured_peak_gib=comparison.get('measured_peak_gib', comparison.get('measured_peak_allocated_gib')),
                before_raw_ms=before_ms, after_raw_ms=after_ms, after_step_time_ms=report['performance'].get('step_time_ms'),
                before_error_percent=error_percent(before_ms, measured), after_error_percent=error_percent(after_ms, measured),
                before_prediction_status=before['prediction_status'], after_prediction_status=report['prediction_status'],
                before_revision=before['evidence']['predictor_revision'], after_revision=report['evidence']['predictor_revision'],
                before_cost_database_revision=before.get('cost_database_revision'),
                after_cost_database_revision=report.get('cost_database_revision'),
                same_cost_database_revision=before.get('cost_database_revision') == report.get('cost_database_revision'),
                before_path=str(before_path), before_sha256=file_digest(before_path), after_path=str(destination),
                cost_coverage=report['evidence'].get('cost_coverage'), cost_sources=cost_summary(report),
                missing_profiles=report['evidence'].get('missing_profiles', []),
                profiling_requests=report['evidence'].get('profiling_requests', []),
                memory_admission=report['memory']['admission'], memory=report['memory'],
                graph_summary=graph_summary(report),
                evaluation_scope='使用现存成本数据库独立评价冻结请求；没有新增GPU实测，未使用父计划差分锚点',
                candidate_plan_excluded_from_calibration=report['evidence'].get('candidate_plan_excluded_from_calibration'))
            outcomes.append(outcome)
            save_json(output / 'comparison.json', outcomes)
            print(json.dumps({key: outcome[key] for key in ('model', 'candidate_id', 'before_raw_ms', 'after_raw_ms',
                  'measured_ms', 'before_error_percent', 'after_error_percent', 'after_prediction_status')}, ensure_ascii=False), flush=True)
        if file_digest(Path(snapshot['source_path'])) != snapshot['source_sha256']:
            raise ValueError('原始成本数据库在验收期间发生变化')
    if {str(path.relative_to(source_root)): file_digest(path) for path in source_files} != algorithm_source:
        raise ValueError('预测器源码在验收期间发生变化，需要使用固定源码重新验收')
    save_json(output / 'provenance.json', dict(source_databases=snapshots, outcomes=len(outcomes),
        algorithm_source=algorithm_source, gpu_experiments=0, external_llm_requests=0,
        prediction_status_counts=dict(Counter(row['after_prediction_status'] for row in outcomes))))
    write_tables(output, outcomes)


if __name__ == '__main__':
    main()
