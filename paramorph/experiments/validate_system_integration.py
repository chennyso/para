import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts import RewriteIntent, SystemConfig
from core_data_models import HardwareTopology
from graph.capability_registry import CapabilityRegistry
from graph.complete_rewrite import CompleteAndRewrite
from graph.model_graph import read_model
from predictor import FastPredictor
from search.system_search import AgentRewriteIntent, InvestigationToolRequest, bounded_observation
from trace.trace_mapper import TraceMapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = SystemConfig.model_validate_json(args.config.read_text())
    predictor = FastPredictor(config.profile_database)
    profile = predictor.database.get(config.profile_id, config.request)
    model, graph = read_model(config.model_config_path)
    registry = CapabilityRegistry(graph['module_structure'])
    topology = HardwareTopology(config.request['node_count'] * config.request['gpus_per_node'],
        config.request['node_count'], config.request['gpus_per_node'],
        min(row['capacity'] for row in profile['memory'].values()) / 2**30)
    rewriter = CompleteAndRewrite(model, topology, registry, config.capability_domains)
    counts = list(config.request['decoder_chunks'])
    counts[0] -= 1
    counts[-1] += 1
    intent = AgentRewriteIntent.model_validate(dict(operation='RepartitionRegion',
        values=dict(decoder_chunks=counts, tp=config.request['tp']), adjustable=[],
        preserve=['tp', 'pp', 'cp', 'dp', 'ep', 'vpp', 'micro_batch', 'group'],
        hypothesis='检查真实模型连续Region边界变换', evidence_refs=[profile['id']]))
    rows = rewriter.complete_and_rewrite(config.request, intent.executable(), model, topology,
        registry.describe(), 1)
    if len(rows) != 1 or rows[0]['request']['decoder_chunks'] != counts:
        raise ValueError('图变换没有产生指定候选')
    if rows[0]['constraint_result']['status'] != 'PASS' or not rows[0]['execution_graph']['tasks']:
        raise ValueError('图变换缺少实际执行结构')
    predictor.evaluate(config.request, config.profile_id, persist=False)
    incremental = predictor.evaluate(rows[0]['request'], config.profile_id, persist=False, blind=True)
    full = FastPredictor(config.profile_database).evaluate(rows[0]['request'], config.profile_id,
        persist=False, blind=True)
    for key in ('performance', 'memory', 'execution_graph'):
        if incremental[key] != full[key]:
            raise ValueError(f'完整与增量评价不一致：{key}')
    compact = bounded_observation(incremental['diagnosis'])
    if len(compact['critical_tasks']['items']) > 16:
        raise ValueError('Agent观测没有遵守节点数量限制')
    trace = TraceMapper().map_runtime(args.profile_run)
    mapped = 0
    for rank, observations in trace['ranks'].items():
        for event in observations['tasks']:
            if not event.get('mapped', True):
                continue
            task = incremental['execution_graph']['tasks'][event['task_id']]
            if event['region_id'] != task['region']:
                raise ValueError('真实Trace Region与执行图映射不同')
            mapped += 1
    tools = InvestigationToolRequest.model_json_schema()['properties']['tool']['enum']
    if len(tools) < 10 or mapped == 0:
        raise ValueError('调查工具或者真实Trace覆盖不足')
    report = dict(status='PASS', model=graph['model_fingerprint'], profile_id=profile['id'],
        full_delta_equivalence=True, explicit_preserved_values=True,
        graph_tasks=len(rows[0]['execution_graph']['tasks']), mapped_trace_events=mapped,
        investigation_tools=tools, bounded_critical_tasks=compact['critical_tasks']['total_count'],
        source='actual_model_config_gpu_profile_and_trace')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report))
