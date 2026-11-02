import argparse
import json
from pathlib import Path
import sys

from openai.lib._pydantic import to_strict_json_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts import SystemConfig
from search.system_search import AgentRewriteIntent, BottleneckDecision, BottleneckTarget
from search.system_search import EvidenceGuidedSearch, IntentBatch, InvestigationSelection
from search.system_search import InvestigationToolRequest, RewriteValues, token_count


def prompt_size(search, role, instructions, observation, schema):
    names = dict(PerformanceAnalysisAgent='performance_analysis_agent.md',
        JointStrategyAgent='joint_strategy_agent.md',
        ExperimentInvestigationAgent='experiment_investigation_agent.md')
    directory = Path(__file__).resolve().parents[1] / 'agents' / 'prompts'
    complete_instructions = ((directory / names[role]).read_text() + '\n'
        + (directory / 'training_optimization_knowledge.md').read_text() + '\n' + instructions)
    prompt = ('Use only the supplied observation. Do not call tools, access files, edit source code, '
        'or invent measurements. Return one object matching the output schema.\n'
        + complete_instructions + '\n' + json.dumps(observation, ensure_ascii=False, allow_nan=False))
    return dict(prompt_tokens=token_count(prompt), observation_tokens=token_count(observation),
        strict_schema_tokens=token_count(to_strict_json_schema(schema)))


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
        raise ValueError('分层上下文验证需要已登记的真实十六卡Baseline')
    graph = search.backend_dag(baseline)
    if graph['task_count'] != 1456:
        raise ValueError(f'当前MiMo十六卡执行DAG任务数量异常：{graph["task_count"]}')
    cross_rank_edge = next((dependency, task_id) for task_id, task in graph['tasks'].items()
        for dependency in task['dependencies']
        if graph['tasks'][dependency]['rank'] != task['rank'])
    source_id, destination_id = cross_rank_edge
    source = graph['tasks'][source_id]
    destination = graph['tasks'][destination_id]
    subgraph_query = InvestigationToolRequest(tool='inspect_dag_subgraph',
        task_ids=[source_id], dependency_direction='successors', dependency_depth=2, limit=12)
    subgraph = search.investigate(subgraph_query, search.store.trials())
    dependency_query = InvestigationToolRequest(tool='inspect_task_dependencies',
        task_ids=[source_id, destination_id], dependency_direction='both', dependency_depth=2, limit=16)
    dependencies = search.investigate(dependency_query, search.store.trials())
    impact_query = InvestigationToolRequest(tool='query_affected_decisions',
        task_ids=[source_id, destination_id], limit=16)
    impact = search.investigate(impact_query, search.store.trials())
    baseline_evidence = baseline['measurement']['run_id']
    intent = AgentRewriteIntent(operation='JointTransform', values=RewriteValues(pp=2),
        adjustable=['dp', 'vpp', 'decoder_chunks', 'group'], preserve=['tp', 'cp', 'ep', 'micro_batch'],
        hypothesis='使用跨Stage、跨Rank依赖证据检查PP、VPP与Region联合重写',
        evidence_refs=[dependencies['evidence_id'], impact['evidence_id']],
        target_task_ids=[source_id, destination_id],
        target_flow_ids=[value for value in (source.get('flow_id'), destination.get('flow_id')) if value],
        parent_plan_id=baseline['trial_id'], target_bottleneck=BottleneckTarget(source='MEASURED',
            task_ids=[source_id, destination_id],
            flow_ids=[value for value in (source.get('flow_id'), destination.get('flow_id')) if value],
            evidence_refs=[baseline_evidence], description='实际Pipeline跨Rank依赖'))
    preview_query = InvestigationToolRequest(tool='preview_transformation',
        task_ids=[source_id, destination_id], transformation_intent=intent, limit=16)
    preview = search.investigate(preview_query, search.store.trials())
    region_intent = AgentRewriteIntent(operation='SetCommunicationPolicy',
        values=RewriteValues(overlap_p2p=False),
        adjustable=[], preserve=['tp', 'pp', 'cp', 'ep', 'dp', 'vpp', 'micro_batch', 'decoder_chunks'],
        hypothesis='验证P2P通信策略反事实预测', evidence_refs=[baseline_evidence],
        target_task_ids=[source_id, destination_id], parent_plan_id=baseline['trial_id'])
    group_intent = AgentRewriteIntent(operation='SetMicrobatch', values=RewriteValues(group=8),
        adjustable=[], preserve=['tp', 'pp', 'cp', 'ep', 'dp', 'vpp', 'micro_batch'],
        hypothesis='验证Microbatch分组反事实预测', evidence_refs=[baseline_evidence],
        target_task_ids=[source_id, destination_id], parent_plan_id=baseline['trial_id'])
    counterfactual = search.counterfactual_interactions(baseline, [region_intent, group_intent])
    pair = counterfactual['pairs'][0]
    if pair['interaction_status'] == 'PREDICTED' and pair['predicted_interaction_ms'] is None:
        raise ValueError('可比反事实候选缺少联合效应')
    if pair['interaction_status'] == 'INDEPENDENT_PLAN_UNAVAILABLE' and pair['predicted_interaction_ms'] is not None:
        raise ValueError('不可比反事实候选生成了联合效应')
    rewrites = search.rewriter.complete_and_rewrite(
        baseline['plan']['backend_config'], intent.executable(), limit=1)
    rewrite = rewrites[0]
    candidate_plan = search.plan(rewrite['request'], parent=baseline['trial_id'],
        transformations=[intent.executable().model_dump(mode='json')])
    prediction = search.predict_request(rewrite['request'], rewrite['execution_graph'],
        baseline['plan']['backend_config'])
    candidate_trial = dict(trial_id=candidate_plan.plan_id, parent_trial_id=baseline['trial_id'],
        plan=candidate_plan.model_dump(mode='json'), plan_hash=candidate_plan.plan_hash,
        hypothesis=intent.model_dump(mode='json'), graph_delta=rewrite['graph_delta'],
        validation=prediction['validation'], prediction=prediction, measurement=None,
        rewrite=rewrite, state='PROJECTED')
    diagnosis = BottleneckDecision(bottleneck_type='MEASURED',
        affected_regions=sorted({source['region_id'], destination['region_id']}),
        affected_tasks=[source_id, destination_id], related_variables=['pp', 'vpp', 'decoder_chunks'],
        hypotheses=['MEASURED Pipeline跨Rank依赖需要联合调查PP、VPP和Region分区'],
        evidence_refs=[dependencies['evidence_id'], impact['evidence_id']],
        target_regions=sorted({source['region_id'], destination['region_id']}), requested_metrics=[])
    strategy = IntentBatch(intents=[intent], reason='使用完整DAG查询证据生成联合变换',
        evidence_refs=[preview['evidence_id']])
    interactions = search.rewriter.compiler.transformation_interactions(
        search.rewriter.candidate(baseline['plan']['backend_config']), [intent.executable()],
        diagnosis.affected_tasks)
    interaction_id = 'validation:' + preview['evidence_id']
    performance_context = search.performance_agent_context(baseline)
    if 'tasks' in json.dumps(performance_context, ensure_ascii=False):
        raise ValueError('初始Performance Analysis Agent上下文包含完整任务字典')
    strategy_context = search.strategy_agent_context(baseline, diagnosis, dependencies['evidence_id'])
    investigation_context = search.investigation_agent_context(
        baseline, [candidate_trial], strategy, interactions, interaction_id)
    prompt_metrics = dict(
        performance=prompt_size(search, 'PerformanceAnalysisAgent',
            '依据真实测量、图依赖与已有预测提出瓶颈假设。',
            performance_context, BottleneckDecision),
        strategy=prompt_size(search, 'JointStrategyAgent',
            '按照intent_count提出完整联合优化意图。', strategy_context, IntentBatch),
        investigation=prompt_size(search, 'ExperimentInvestigationAgent',
            '比较候选的GraphDelta、显存、成本可信度和预算。',
            investigation_context, InvestigationSelection))
    performance_section_tokens = {key: token_count(value)
        for key, value in performance_context.items()}
    if (prompt_metrics['performance']['prompt_tokens']
            + prompt_metrics['performance']['strict_schema_tokens'] > 10000):
        raise ValueError('初始Performance Analysis Agent Prompt与严格Schema合计超过10000 tokens：'
            + json.dumps(performance_section_tokens, ensure_ascii=False))
    if rewrite['candidate']['config']['world'] != 16:
        raise ValueError('联合变换生成的计划没有使用全部十六张GPU')
    result = dict(status='PASS', baseline_trial_id=baseline['trial_id'],
        graph=dict(graph_sha256=graph['graph_sha256'], task_count=graph['task_count'],
            edge_count=graph['edge_count'], flow_count=graph['flow_count'],
            cost_evidence_refs=graph['cost_evidence_refs'],
            full_dag_serialized_in_initial_prompt=False),
        cross_stage_rank_dependency=dict(source_task_id=source_id, source_rank=source['rank'],
            source_region=source['region_id'], destination_task_id=destination_id,
            destination_rank=destination['rank'], destination_region=destination['region_id']),
        query_chain=[dict(tool=row['tool']['tool'], evidence_id=row['evidence_id'],
            result_tokens=row['result_tokens']) for row in (subgraph, dependencies, impact, preview)],
        legal_joint_rewrite=dict(plan_id=candidate_plan.plan_id,
            parallelism=candidate_plan.parallelism, region_partition=candidate_plan.region_partition,
            world_size=rewrite['candidate']['config']['world'],
            graph_task_count=len(rewrite['execution_graph']['tasks']),
            validation=prediction['validation'], prediction_status=prediction['prediction_status']),
        counterfactual_interaction=counterfactual,
        prompt_metrics=prompt_metrics, performance_section_tokens=performance_section_tokens,
        strict_tool_request_schema_tokens=token_count(to_strict_json_schema(InvestigationToolRequest)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
