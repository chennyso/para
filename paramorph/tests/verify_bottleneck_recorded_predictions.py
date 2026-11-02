import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from agents.strategy_transformations import StrategyTransformationCompiler
from graph.bottleneck_analysis import (
    BottleneckGraph, collective_cost_bindings, replay_shared_cost_scenarios,
)


def verify_report(path, epsilon):
    report = json.loads(path.read_text())
    graph = BottleneckGraph(report['execution_graph'], [report['prediction_id']])
    seeds = [key for key in report['diagnosis']['critical_tasks']
             if len(graph._participants(key)) > 1]
    if not seeds:
        raise ValueError(f'{path}没有可用于跨Rank验收的实际通信任务')
    seed = max(seeds, key=lambda key: graph.records[key]['duration_ms'])
    closure = graph.closure([seed], dependency_depth=2)
    offset = 0
    recovered = []
    page_count = 0
    max_page_chars = 0
    while offset is not None:
        page = graph.page(closure, offset=offset, limit=7, max_chars=6000)
        recovered.extend(row['task_id'] for row in page['tasks'])
        size = len(json.dumps(page, ensure_ascii=False))
        if size > 6000:
            raise ValueError('真实子图查询超过文本预算')
        max_page_chars = max(max_page_chars, size)
        page_count += 1
        offset = page['next_offset']
    if recovered != closure['task_ids'] or len(recovered) != len(set(recovered)):
        raise ValueError('子图分页没有无重复地恢复全部闭包任务')
    dependencies = []
    offset = 0
    while offset is not None:
        page = graph.task_dependencies(seed, offset=offset, limit=3, max_chars=6000)
        dependencies.extend(row['task_id'] for row in page['tasks'])
        offset = page['next_offset']
    if dependencies != sorted(graph.predecessors[seed]):
        raise ValueError('依赖分页结果与完整执行图不一致')
    result = graph.epsilon_impact(closure['task_ids'], epsilon)
    expected = report['performance']['raw_resource_aware_step_time_ms']
    if not math.isclose(result['baseline_step_time_ms'], expected, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f'{path}的原始Replay与保存预测不一致，需要提供实际资源模型')
    ranks = sorted({graph.records[key]['rank'] for key in closure['task_ids']})
    if len(ranks) < 2:
        raise ValueError('闭包没有覆盖实际跨Rank依赖')
    observation = dict(training_graph=report['execution_graph'],
        evidence=[dict(evidence_id=report['prediction_id'])])
    queried = StrategyTransformationCompiler.query(observation,
        dict(operation='inspect_bottleneck_subgraph', task_ids=[seed], dependency_depth=2,
            offset=0, limit=7, max_chars=6000))
    if queried['closure_id'] != closure['closure_id']:
        raise ValueError('Agent子图查询与后端原始图闭包不一致')
    regions = sorted({graph.records[key]['region'] for key in closure['task_ids']
                      if graph.records[key].get('region') is not None})
    return report, dict(source=str(path), candidate_id=report['candidate_id'],
        complete_task_count=len(graph.records), seed_task_id=seed,
        closure_task_count=len(closure['task_ids']), ranks=ranks, regions=regions,
        collective_participant_ranks=graph._participants(seed), page_count=page_count,
        max_page_chars=max_page_chars, dependency_count=len(dependencies),
        baseline_step_time_ms=result['baseline_step_time_ms'],
        counterfactual_step_time_ms=result['counterfactual_step_time_ms'],
        epsilon=epsilon, impact_ms=result['impact_ms'],
        affected_task_count=len(closure['task_ids']),
        evidence_type=result['evidence_type'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epsilon', type=float, default=0.1)
    args = parser.parse_args()
    reports = []
    results = []
    for path in sorted(args.predictions.glob('*/*.prediction.json')):
        report, result = verify_report(path, args.epsilon)
        reports.append(report)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    shared = defaultdict(dict)
    source_costs = {}
    mimo = [report for report, result in zip(reports, results) if '/mimo/' in result['source']]
    for report in mimo:
        unique_costs = {row['cost_key']: row for row in report['evidence']['costs'] if row.get('cost_key')}
        for row in unique_costs.values():
            key = row.get('cost_key')
            sigma = row.get('uncertainty_ms')
            if key and sigma is not None and sigma > 0 and collective_cost_bindings(report, key):
                shared[key][report['candidate_id']] = row
                source_costs[key] = row
    keys = [key for key, values in shared.items() if len(values) > 1]
    if not keys:
        raise ValueError('实际成本记录没有跨候选共享且具有实测离散程度的独立Collective')
    key = max(keys, key=lambda value: (len(shared[value]), source_costs[value]['milliseconds'], value))
    cost = source_costs[key]
    low = max(0, cost['milliseconds'] - cost['uncertainty_ms'])
    high = cost['milliseconds'] + cost['uncertainty_ms']
    ranking = replay_shared_cost_scenarios(mimo, key, low, high, top_k=3)
    if ranking['reuse_count'] != len(shared[key]):
        raise ValueError('共享成本情景没有覆盖全部实际关联候选')
    result = dict(recorded_prediction_count=len(results), records=results,
        shared_cost_sensitivity=ranking,
        scenario_source=dict(evidence_id=cost.get('evidence_id'), milliseconds=cost['milliseconds'],
            sample_dispersion_ms=cost['uncertainty_ms'], statistical_confidence_interval=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), verified_predictions=len(results),
        shared_cost_reuse_count=ranking['reuse_count'], scenario_evaluations=ranking['evaluation_count']),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
