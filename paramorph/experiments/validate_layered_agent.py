import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts import SystemConfig
from search.system_search import BottleneckDecision, EvidenceGuidedSearch, IntentBatch
from search.system_search import InvestigationToolRequest
from search.system_search import LAYERED_CONTEXT_VERSION


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.config.read_text())
    document['gpu_execution_enabled'] = False
    search = EvidenceGuidedSearch(SystemConfig.model_validate(document))
    baseline = search.reference_trial()
    if baseline is None or baseline['measurement'] is None:
        raise ValueError('Agent分层访问验证需要已登记的真实Baseline')
    checkpoint = search.store.get_round(0)
    if checkpoint is not None and checkpoint.get('context_version') == LAYERED_CONTEXT_VERSION:
        performance_context = checkpoint['observation']
    else:
        performance_context = search.performance_agent_context(baseline)
        checkpoint = search.store.replace_round(0, dict(round_index=0, parent_trial_id=baseline['trial_id'],
            context_version=LAYERED_CONTEXT_VERSION, observation=performance_context, stage='ANALYSIS'))
    if 'diagnosis' in checkpoint:
        diagnosis = BottleneckDecision.model_validate(checkpoint['diagnosis'])
        diagnosis_id = checkpoint['diagnosis_evidence_id']
        query_results = []
        for evidence_id in diagnosis.evidence_refs:
            stored = search.store.connection.execute(
                'SELECT document FROM evidence WHERE id=?', (evidence_id,)).fetchone()
            if stored is None:
                raise ValueError('Agent诊断引用了不存在的证据')
            document = json.loads(stored[0])
            if document.get('tool') is not None:
                query_results.append(dict(evidence_id=evidence_id, tool=document['tool'],
                    result=document['result'], result_tokens=document['result_tokens'],
                    token_encoding=document['token_encoding']))
        performance_rounds = None
    else:
        diagnosis, diagnosis_id = search.invoke_with_tools('PerformanceAnalysisAgent',
            search.performance_instructions(), performance_context, BottleneckDecision)
        performance_session_id = search.agent_session_identity('PerformanceAnalysisAgent',
            search.performance_instructions(), performance_context, BottleneckDecision)
        performance_session = search.store.get_agent_session(performance_session_id)
        query_results = list(performance_session['context'].get('tool_history', []))
        receipts = performance_session['context'].get('query_receipts', [])
        known_result_ids = {row['evidence_id'] for row in query_results}
        for receipt in receipts:
            if receipt['evidence_id'] in known_result_ids:
                continue
            stored = search.store.connection.execute(
                'SELECT document FROM evidence WHERE id=?', (receipt['evidence_id'],)).fetchone()
            if stored is None:
                raise ValueError('Agent查询回执引用了不存在的证据')
            document = json.loads(stored[0])
            query_results.append(dict(evidence_id=receipt['evidence_id'],
                tool=document['tool'], result=document['result'],
                result_tokens=document['result_tokens'], token_encoding=document['token_encoding']))
        performance_rounds = performance_session['rounds']
    tool_names = [row['tool']['tool'] for row in query_results]
    required = {'inspect_dag_subgraph', 'inspect_task_dependencies', 'query_affected_decisions'}
    if not required <= set(tool_names):
        raise ValueError(f'Performance Analysis Agent没有完成必需的分层DAG查询：{sorted(set(tool_names))}')
    graph = search.backend_dag(baseline)
    cross_rank_edges = []
    for row in query_results:
        if row['tool']['tool'] != 'inspect_task_dependencies':
            continue
        row = search.investigate(InvestigationToolRequest.model_validate(row['tool']),
            search.store.trials())
        for task_id, dependencies in row['result']['queried_dependencies'].items():
            for adjacent_id in dependencies['predecessors'] + dependencies['successors']:
                if graph['tasks'][task_id]['rank'] != graph['tasks'][adjacent_id]['rank']:
                    cross_rank_edges.append((task_id, adjacent_id))
        for source_id, destination_id in (row['result'].get('internal_edges', [])
                                          + row['result'].get('external_edge_refs', [])
                                          + row['result'].get('expanded_cross_rank_edges', [])):
            if (source_id in graph['tasks'] and destination_id in graph['tasks']
                    and graph['tasks'][source_id]['rank'] != graph['tasks'][destination_id]['rank']):
                cross_rank_edges.append((source_id, destination_id))
    if not cross_rank_edges:
        raise ValueError('Performance Analysis Agent的任务依赖查询没有定位跨Rank边')
    search.store.save_round(0, diagnosis=diagnosis.model_dump(mode='json'),
        diagnosis_evidence_id=diagnosis_id, stage='STRATEGY')
    checkpoint = search.store.get_round(0)
    if 'strategy_context' in checkpoint:
        strategy_context = checkpoint['strategy_context']
    else:
        strategy_context = search.strategy_agent_context(baseline, diagnosis, diagnosis_id,
            search.global_family_catalog(baseline))
        search.store.save_round(0, strategy_context=strategy_context)
    strategy, strategy_id = search.invoke_with_tools('JointStrategyAgent',
        search.strategy_instructions(), strategy_context, IntentBatch)
    strategy_validation = search.validate_strategy_batch(strategy, baseline)
    legal = []
    rejected = []
    for intent in strategy.intents:
        try:
            rewrites = search.rewriter.complete_and_rewrite(
                baseline['plan']['backend_config'], intent.executable(), limit=1)
        except ValueError as error:
            rejected.append(dict(operation=intent.operation, error=error.args[0]))
            continue
        for rewrite in rewrites:
            if rewrite['candidate']['config']['world'] != 16:
                raise ValueError('Agent联合策略的确定性补全结果没有使用十六张GPU')
            legal.append(dict(operation=intent.operation, run_id=rewrite['request']['run_id'],
                config={key: rewrite['request'][key] for key in
                    ('tp', 'pp', 'cp', 'ep', 'etp', 'expert_dp', 'dp', 'vpp', 'micro_batch', 'group')},
                decoder_chunks=rewrite['request']['decoder_chunks'],
                target_impact=rewrite['graph_delta']['target_impact']))
    if not legal:
        raise ValueError('Joint Strategy Agent没有生成可由CompleteAndRewrite补全的十六卡合法策略：'
            + json.dumps(rejected, ensure_ascii=False))
    search.store.save_round(0, strategy=strategy.model_dump(mode='json'),
        strategy_evidence_id=strategy_id, stage='CANDIDATES')
    metrics = [row for row in search.store.agent_context_metrics()
        if row.get('context_version') == LAYERED_CONTEXT_VERSION
        and row.get('role') in ('PerformanceAnalysisAgent', 'JointStrategyAgent')]
    result = dict(status='PASS', baseline_trial_id=baseline['trial_id'],
        diagnosis_evidence_id=diagnosis_id, strategy_evidence_id=strategy_id,
        performance_tool_queries=tool_names, performance_tool_rounds=performance_rounds,
        cross_rank_dependency_edges=cross_rank_edges,
        strategy_validation=strategy_validation,
        legal_joint_strategy_count=len(legal), legal_joint_strategies=legal,
        rejected_intents=rejected, agent_context_metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
