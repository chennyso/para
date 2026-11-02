from pathlib import Path
import json
import hashlib
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from megatron.core.frontier.agents.provider import CodexJSONProvider
from agents.strategy_transformations import GraphQuery, GraphReasoningReference, StrategyProposals, StrategyTransformationCompiler


class EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    candidate_ids: list[str]
    evidence_ids: list[str]
    reason: str
    missing_measurements: list[str]
    queries: list[GraphQuery] = Field(default_factory=list, max_length=4)
    reasoning_evidence: list[GraphReasoningReference] = Field(default_factory=list)
    subject_candidate_id: str | None = None
    attribution: Literal['known', 'uncertain', 'unknown', 'task_trace_supported'] = 'unknown'


class DecisionService:
    def __init__(self, output, max_query_rounds=12, model=None, reasoning_effort='medium'):
        self.output = Path(output)
        self.provider = CodexJSONProvider(self.output, model=model, reasoning_effort=reasoning_effort)
        history = self.output / 'agent-interactions.json'
        self.history = json.loads(history.read_text()) if history.exists() else []
        if type(max_query_rounds) is not int or max_query_rounds < 1:
            raise ValueError('图查询资源预算必须为正整数')
        self.max_query_rounds = max_query_rounds
        tool_path = self.output / 'graph-tool-calls.json'
        self.tool_calls = json.loads(tool_path.read_text()) if tool_path.exists() else []

    def execute_queries(self, role, queries, handler, observation):
        records = []
        for query in queries:
            if query.operation not in observation['available_graph_tools']:
                raise ValueError(f'当前Agent阶段没有注册图工具：{query.operation}')
            query_payload = query.model_dump(mode='json')
            pending = dict(agent=role, tool=query.operation, parameters=query_payload,
                candidate_id=query.candidate_id or observation.get('current_candidate_id') or
                    observation.get('parent_candidate', {}).get('candidate_id'), state='submitted')
            self.tool_calls.append(pending)
            (self.output / 'graph-tool-calls.json').write_text(json.dumps(
                self.tool_calls, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            started = perf_counter()
            result = handler(query)
            tool_seconds = perf_counter() - started
            payload = dict(agent=role, tool=query.operation, parameters=query_payload,
                candidate_id=query.candidate_id or observation.get('current_candidate_id') or
                    observation.get('parent_candidate', {}).get('candidate_id'), result=result,
                evidence_ids=query.evidence_ids,
                graph_sha256=result.get('graph_sha256', observation.get('training_graph', {}).get('graph_sha256')))
            payload.update(tool_seconds=tool_seconds, evidence_type='aggregate_metric' if query.operation == 'compare_plans'
                else 'task_trace' if query.operation == 'inspect_task_trace' else 'graph_structure')
            identifier = hashlib.sha256(json.dumps({key: value for key, value in payload.items()
                if key != 'tool_seconds'}, sort_keys=True).encode()).hexdigest()
            payload['query_result_id'] = identifier
            payload['state'] = 'completed'
            records.append(payload)
            pending.update(payload)
        (self.output / 'graph-tool-calls.json').write_text(json.dumps(
            self.tool_calls, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return records

    def reflect_measurement(self, baseline_observation, candidate_observation, evidence, feedback):
        candidate_id = candidate_observation['parent_candidate']['candidate_id']
        snapshot = candidate_observation | dict(evidence=evidence)
        baseline_id = baseline_observation['parent_candidate']['candidate_id']
        query_observation = snapshot | dict(evidence=evidence,
            candidate_observations={candidate_id: snapshot, baseline_id: baseline_observation})
        bounded_evidence = StrategyTransformationCompiler.agent_context(snapshot)['evidence']
        diff = feedback['graph_diff']
        bounded_diff = {key: (value[:16] if key in ('added_task_ids', 'removed_task_ids',
            'modified_task_ids', 'affected_task_ids', 'added_edges', 'removed_edges') else value)
            for key, value in diff.items()}
        bounded_diff['full_list_counts'] = {key: len(value) for key, value in diff.items() if isinstance(value, list)}
        if 'impact' in bounded_diff:
            bounded_diff['impact'] = {key: value[:16] if key in ('compute_task_ids', 'communication_task_ids') else value
                for key, value in bounded_diff['impact'].items()}
        bounded_feedback = feedback | dict(graph_diff=bounded_diff, measurement=bounded_evidence[-1],
            logs={key: value | dict(tail=value['tail'][-16:]) for key, value in feedback['logs'].items()})
        decision = self('reflection', dict(current_candidate_id=candidate_id, baseline_candidate_id=baseline_id,
            report=dict(bottlenecks=[], evidence=bounded_evidence), candidates=[], feedback=bounded_feedback,
            instruction='查询真实测量并对照原始假设、Graph Diff与审查判断；检查成功、性能退化或运行失败，缺少Trace时不能完成区间耗时归因'),
            query_handler=lambda query: StrategyTransformationCompiler.query(query_observation, query))
        path = self.output / 'execution-feedback.json'
        history = json.loads(path.read_text()) if path.exists() else []
        history.append(dict(candidate_id=candidate_id, feedback=feedback, reflection=decision))
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return decision

    @staticmethod
    def query_availability(observation, candidate_diff=False):
        evidence = observation.get('evidence', observation.get('report', {}).get('evidence', []))
        ranks = [record for item in evidence for record in item.get('actual_execution', {}).get('ranks', {}).values()]
        tools = ['inspect_graph', 'inspect_parallelism', 'inspect_strategy_dependencies', 'analyze_critical_path']
        if any(item.get('task_trace_available') or 'task_trace' in item for item in evidence):
            tools.append('inspect_task_trace')
        if any('module_boundary' in record for record in ranks):
            tools.append('inspect_module_boundary')
        if any('semantic_graph_path' in record for record in ranks):
            tools.append('inspect_semantic_regions')
        if candidate_diff:
            tools.append('inspect_candidate_diff')
        if len(evidence) >= 2:
            tools.append('compare_plans')
        return tools

    @staticmethod
    def reasoning_conflicts(decision, records, allow_legacy=False):
        conflicts = []
        if not decision.reasoning_evidence:
            return [dict(issue='missing_reasoning_evidence')]
        known = {record['query_result_id']: record for record in records}
        if getattr(decision, 'attribution', 'unknown') == 'task_trace_supported' and not any(
            record.get('evidence_type') == 'task_trace' for record in records):
            conflicts.append(dict(issue='attribution_requires_task_trace'))
        for index, reference in enumerate(decision.reasoning_evidence):
            if not set(reference.query_result_ids) <= set(known):
                conflicts.append(dict(reference=index, issue='unknown_query_result_ids',
                    values=sorted(set(reference.query_result_ids) - set(known))))
                continue
            selected = [known[key] for key in reference.query_result_ids]
            kinds = {record.get('evidence_type', 'aggregate_metric' if record['tool'] == 'compare_plans'
                else 'graph_structure') for record in selected}
            if reference.evidence_type is None and not allow_legacy:
                conflicts.append(dict(reference=index, issue='missing_evidence_type'))
            if reference.evidence_type == 'task_trace' and 'task_trace' not in kinds:
                conflicts.append(dict(reference=index, issue='task_trace_not_available'))
            if (kinds == {'aggregate_metric'} or reference.evidence_type == 'aggregate_metric') and reference.task_ids:
                conflicts.append(dict(reference=index, issue='aggregate_metrics_do_not_measure_task_intervals',
                    values=reference.task_ids))
            tasks = set().union(*(set(record['result'].get('tasks', {})) |
                set(record['result'].get('selected_tasks', {})) |
                set(record['result'].get('impact', {}).get('compute_task_ids', [])) |
                set(record['result'].get('impact', {}).get('communication_task_ids', [])) |
                set(record['parameters']['task_ids']) for record in selected))
            allow_structural_task_reference = (
                getattr(decision, 'attribution', 'unknown') in ('unknown', 'uncertain')
                and reference.claim_scope in ('structural_fact', 'uncertainty'))
            if not set(reference.task_ids) <= tasks and not allow_structural_task_reference:
                conflicts.append(dict(reference=index, issue='task_ids_not_in_cited_query',
                    values=sorted(set(reference.task_ids) - tasks)))
            if reference.claim_scope == 'measured_fact' and not any(
                record['evidence_ids'] and record['result']['status'] == 'available' and record['tool'] in
                ('inspect_parallelism', 'inspect_module_boundary', 'inspect_semantic_regions', 'compare_plans', 'inspect_task_trace')
                for record in selected):
                conflicts.append(dict(reference=index, issue='measured_claim_without_measurement_query'))
        return conflicts

    @staticmethod
    def validate_reasoning(decision, records, allow_legacy=False):
        conflicts = DecisionService.reasoning_conflicts(decision, records, allow_legacy)
        if conflicts:
            raise ValueError(f'图推理证据校验未通过：{conflicts}')

    def end_failed_stage(self, role, entry, conflicts, corrections):
        entry.update(stage_status='validation_failed', validation_errors=conflicts,
            automatic_corrections=corrections)
        (self.output / f'{role}-failure.json').write_text(json.dumps(entry,
            ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        (self.output / 'agent-interactions.json').write_text(json.dumps(self.history,
            ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        raise ValueError(f'{role}在一次自动修正后仍未通过校验，当前阶段已结束')

    @staticmethod
    def selection_conflicts(decision, observation):
        allowed = {candidate['candidate_id'] for candidate in observation.get('candidates', [])}
        source = observation.get('report', observation)
        evidence = {item['evidence_id'] for item in source.get('evidence', [])}
        evidence.update(identifier for item in source.get('bottlenecks', []) for identifier in item['evidence_ids'])
        conflicts = []
        if not set(decision.candidate_ids) <= allowed:
            conflicts.append(dict(issue='candidate_ids_are_selection_only',
                allowed_candidate_ids=sorted(allowed), current_candidate_id=observation.get('current_candidate_id')))
        if not set(decision.evidence_ids) <= evidence:
            conflicts.append(dict(issue='unknown_measured_evidence_ids', allowed_evidence_ids=sorted(evidence)))
        subject = observation.get('current_candidate_id') or observation.get('parent_candidate', {}).get('candidate_id')
        if decision.subject_candidate_id != subject:
            conflicts.append(dict(issue='subject_candidate_id_mismatch', expected_subject_candidate_id=subject))
        return conflicts

    @staticmethod
    def verified_proposals(directory, parent):
        directory = Path(directory)
        history = json.loads((directory / 'agent-interactions.json').read_text())
        verified = []
        for entry in history:
            if entry['role'] not in ('parallelism', 'vpp', 'execution', 'strategy'):
                continue
            if entry['input']['parent_candidate'] != json.loads(json.dumps(parent.to_dict())):
                raise ValueError('已有Agent提案对应不同的原始训练程序')
            provenance = entry['provenance']
            matches = [path.parent for path in directory.glob('*/provenance.json')
                if json.loads(path.read_text())['response_sha256'] == provenance['response_sha256']]
            if len(matches) != 1:
                raise ValueError('Agent提案缺少唯一的真实响应来源')
            source = matches[0]
            if (hashlib.sha256((source / 'prompt.txt').read_bytes()).hexdigest() != provenance['prompt_sha256'] or
                    hashlib.sha256((source / 'response.json').read_bytes()).hexdigest() != provenance['response_sha256'] or
                    json.loads((source / 'response.json').read_text()) != entry['output']):
                raise ValueError('Agent响应与保存的输入输出记录不一致')
            events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines() if line.strip()]
            complete = [event for event in events if event['type'] == 'turn.completed']
            if (len(complete) != 1 or complete[0]['usage'] != provenance['usage'] or
                    any(event.get('item', {}).get('type') in ('command_execution', 'mcp_tool_call', 'file_change')
                        for event in events)):
                raise ValueError('Agent来源没有有效的单次完成记录')
            for proposal in StrategyProposals.model_validate(entry['output']).proposals:
                verified.append((proposal, provenance))
        return verified

    def propose_transformations(self, role, observation, query_handler=None):
        if role not in ('parallelism', 'vpp', 'execution', 'strategy'):
            raise ValueError('未知策略生成Agent角色')
        responsibilities = {
            'strategy': '依据诊断、训练图与实测历史选择主变换；联合考虑混合并行、Region和已支持的通信调度，说明可证伪的收益假设。',
            'parallelism': '分析并行组及其对计算通信任务的影响；提出需要真实测量的合法联合变换。',
            'vpp': '分析虚拟阶段计算组成、跨阶段依赖及显存生命周期；同时比较通信边界数量和计算粒度。',
            'execution': '分析原生Microbatch分组、P2P重叠和梯度Bucket；判断哪些就绪任务和通信关系能被已支持的操作改变。'}
        tools = self.query_availability(observation, bool(observation.get('shared_graphs')))
        observation = observation | {'prompt_version': 'agent_directed_graph_queries_v5',
            'available_graph_tools': tools}
        instructions = responsibilities[role] + (
            ' Generate up to two complete coupled transformation proposals using only capabilities '
            'in the observation. Every edit must name existing target_task_ids and measured '
            'evidence_ids. Preserve world size, GBS, sequence, precision and optimizer policy. '
            'MBS changes must follow the supplied constraint-completion rules and preserve GBS. '
            'Do not invent internal collective timing, memory predictions or speedups. '
            'State hypotheses separately from measured facts. Do not repeat measured failed or '
            'slower programs. Do not change parallelism merely to examine ranking reversals. '
            'This is budgeted experimental proposal generation, not a proof of speedup. '
            'Use parent_candidate_id to refine a supplied shared candidate; its shared_graphs '
            'show the updated dependencies and parallel groups. It remains unmeasured. '
            'Use null to transform the original measured program. Query candidate_id selects '
            'a supplied compiled candidate, null selects the measured reference graph. '
            'Missing per-region timing or candidate memory measurements do not alone justify '
            'returning no proposals. Use graph structure and measured successes/failures to form '
            'a falsifiable hypothesis, explicitly listing missing measurements. Do not claim '
            'memory admission or correctness certification when only structural checks exist. '
            'For each proposal explain the exact graph change, why it may reduce full iteration '
            'time, and which observed result would falsify it. A negative outcome is valid. '
            'Use inspect_strategy_dependencies to inspect constraints for a parallelism or MBS change. '
            'Choose the complete rewrite parameters yourself after reading the returned constraints. '
            'The host derives DP and validates groups, layouts and schedules; it does not choose '
            'MBS, VPP, Region boundaries or experimental priority for you. '
            'A proposal must jointly satisfy all backend constraints; use all Region layer counts '
            'when needed. parameters is a fixed-schema object: only fields listed for the chosen '
            'operation may be non-null, all others must be null. Region layer counts belong solely '
            'in SetRegionLayout, not in SetParallelism. Resolve supplied compiler_conflicts exactly. '
            'Return no proposals only if supported actions cannot produce a new legal program, '
            'or all proposed programs duplicate measured failures/regressions; explain the '
            'specific exhausted capability or contradiction. '
            'Use queries to request inspect_graph, inspect_parallelism, inspect_module_boundary, inspect_semantic_regions, compare_plans, '
            'inspect_candidate_diff, inspect_strategy_dependencies, analyze_critical_path or evaluate_plan. '
            'Your first strategy response must request graph queries before proposing a program. '
            'Use only operations listed in available_graph_tools; the measured reference has no rewrite diff. '
            'Queries are structured requests executed '
            'by the host; do not invoke external tools. A response may contain queries or '
            'proposals, not both. Choose additional queries when needed within the resource budget; '
            'there is no required query count. Every final conclusion must include reasoning_evidence '
            'with actual query_result_ids and inspected task_ids. Return empty queries when ready. '
            'Label every reasoning reference evidence_type as aggregate_metric, graph_structure or task_trace. '
            'At most one automatic correction of a rejected final output is allowed. '
            'measurement_required is a truthful tool result and does not forbid '
            'an explicitly uncertain experimental hypothesis. '
            'Output must match StrategyProposals exactly: proposals is always an array (use [] when querying), '
            'reason is required, queries is an array of at most 4 GraphQuery objects, and each query '
            'logical_layers list has at most 2 integers. Each reasoning_evidence item requires '
            'conclusion, query_result_ids, claim_scope and evidence_type; use only schema field names. '
            'Each proposal must use exactly name, edits, missing_measurements and parent_candidate_id. '
            'Each edit must use exactly operation, target_task_ids, evidence_ids, parameters and hypothesis. '
            'Never use proposal_id, operations, falsifier, measurement_required or proposal-level reasoning_evidence. '
            'Return JSON conforming exactly to the supplied StrategyProposals output schema; extra keys are invalid. '
            'For compare_plans, provide exactly two evidence_ids belonging to distinct measured plans; '
            'otherwise choose inspect_graph, inspect_parallelism or inspect_strategy_dependencies. '
            'Cite tokens_per_second, step_seconds and peak_allocated_gib only from inspect_parallelism '
            'with measured evidence_ids or compare_plans, using the returned query_result_id. '
            'Graph structure queries cannot support measured_fact claims. '
            'Explain hypotheses and missing measurements in Chinese.')
        records = []
        corrections = 0
        for query_round in range(self.max_query_rounds + 1):
            decision, provenance = self.provider.generate('paramorph_transform_' + role,
                instructions, observation | {'query_round': query_round,
                    'remaining_query_budget': self.max_query_rounds - query_round}, StrategyProposals)
            result = decision.model_dump(mode='json')
            entry = dict(role=role, input=observation, output=result, provenance=provenance)
            self.history.append(entry)
            if decision.queries:
                if decision.proposals or query_handler is None or query_round == self.max_query_rounds:
                    raise ValueError('Agent图查询违反互斥、工具能力或者查询预算')
                entry['tool_results'] = self.execute_queries(role, decision.queries, query_handler, observation)
                records.extend(entry['tool_results'])
                observation = observation | dict(tool_results=[{key: value for key, value in record.items()
                    if key != 'tool_seconds'} for record in records])
            (self.output / 'agent-interactions.json').write_text(json.dumps(
                self.history, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            if not decision.queries:
                conflicts = self.reasoning_conflicts(decision, records)
                if conflicts:
                    entry['evidence_validation_errors'] = conflicts
                    observation = observation | dict(evidence_validation_errors=conflicts,
                        rejected_decision=result, tool_results=list(records))
                    if corrections >= 1:
                        self.end_failed_stage(role, entry, conflicts, corrections)
                    corrections += 1
                    continue
                self.validate_reasoning(decision, records)
                entry.update(stage_status='passed', automatic_corrections=corrections)
                (self.output / 'agent-interactions.json').write_text(json.dumps(
                    self.history, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
                return decision
        raise ValueError('Agent未在查询预算内返回策略')

    def __call__(self, role, observation, query_handler=None):
        observation = observation | {'response_contract': EvidenceDecision.model_json_schema()}
        if query_handler is not None:
            tools = self.query_availability(observation, role in ('review', 'reflection'))
            observation = observation | dict(available_graph_tools=tools)
            observation['role_goal'] = {
                'diagnosis': '查询当前图结构及并行组，提出有证据的结构判断；没有Trace时以明确的不确定性结束，不需要枚举所有节点。',
                'review': '查询候选完整图及Graph Diff，检查假设、代价和风险，给出性能推荐或探索性测量判断。',
                'reflection': '查询真实对照结果并检验原始假设；缺少Trace时记录无法归因的时间差并结束。',
            }.get(role, '依据现有图证据完成当前决策')
            records = []
            corrections = 0
            for query_round in range(self.max_query_rounds + 1):
                decision, provenance = self.provider.generate('paramorph_' + role,
                    'Return candidate_ids, evidence_ids, reason, missing_measurements, queries and reasoning_evidence. '
                    'First request inspect_graph, analyze_critical_path or inspect_candidate_diff. '
                    'Use only registered available_graph_tools. A measured reference has no rewrite diff; '
                    'Diagnosis should inspect its tasks and dependencies. '
                    'For review, query the supplied compiled candidate_id and inspect its workload_summary. '
                    'Do not select a candidate before actual graph-query results have been returned. '
                    'Queries and candidate_ids are mutually exclusive. Explore further dependencies '
                    'If queries is nonempty, candidate_ids must be []; if candidate_ids is nonempty, '
                    'queries must be []. After Graph Diff and legality are inspected, a legal exploratory '
                    'candidate can be selected with queries=[] and attribution=uncertain. '
                    'as needed within the resource budget, with no required query count. '
                    'Link every final conclusion to actual query_result_ids and inspected task_ids '
                    'using reasoning_evidence. Critical-path data gaps prohibit claiming '
                    'a measured bottleneck or predicted speedup; exploratory decisions must state risk. '
                    'Compare microbatch counts, F/B task counts, logical P2P frequency and bytes, '
                    'compute shapes, Region placement and missing collective/optimizer costs. '
                    'Do not repeat queries for a confirmed unavailable measurement. A declared '
                    'Trace gap is a valid final uncertainty conclusion; it cannot be solved by '
                    'querying more static nodes. Complete the decision when evidence supports '
                    'structural conclusions and explicitly report unavailable performance attribution. '
                    'candidate_ids selects only entries in the supplied candidates list. '
                    'For Diagnosis and Reflection that list is empty, so return empty candidate_ids; '
                    'current_candidate_id identifies the subject and must not be copied into the selection. '
                    'Set subject_candidate_id to current_candidate_id. Each reasoning reference must '
                    'label evidence_type as aggregate_metric, graph_structure or task_trace. '
                    'Aggregate metric comparisons support only global performance, never a task interval. '
                    'Do not cite inspect_graph or analyze_critical_path for tokens_per_second, step_seconds '
                    'or peak_allocated_gib. To include measured facts, request inspect_parallelism with '
                    'the measured evidence_id and cite that returned query_result_id. Otherwise omit '
                    'aggregate measurements from reasoning_evidence and keep structural conclusions. '
                    'Set attribution to unknown unless a mapped task_trace supports attribution. '
                    'For review: if no task_trace is available, structural evidence alone is sufficient '
                    'to select a candidate for experimental measurement — the real GPU run is how the '
                    'missing performance data is obtained. Do not return empty candidate_ids solely '
                    'because performance attribution is unavailable; instead set attribution to '
                    'uncertain and select the structurally sound candidate for measurement. '
                    'Use only supplied evidence_ids. Explain the result in Chinese.',
                    observation | dict(query_round=query_round,
                        remaining_query_budget=self.max_query_rounds - query_round), EvidenceDecision)
                result = decision.model_dump(mode='json')
                entry = dict(role=role, input=observation, output=result, provenance=provenance)
                self.history.append(entry)
                if decision.queries:
                    if decision.candidate_ids or query_round == self.max_query_rounds:
                        raise ValueError('图决策违反查询互斥或预算约束')
                    entry['tool_results'] = self.execute_queries(role, decision.queries, query_handler, observation)
                    records.extend(entry['tool_results'])
                    observation = observation | dict(tool_results=[{key: value for key, value in record.items()
                        if key != 'tool_seconds'} for record in records])
                else:
                    conflicts = self.reasoning_conflicts(decision, records) + self.selection_conflicts(decision, observation)
                    if conflicts:
                        entry['evidence_validation_errors'] = conflicts
                        observation = observation | dict(evidence_validation_errors=conflicts,
                            rejected_decision=result, tool_results=list(records))
                        (self.output / 'agent-interactions.json').write_text(json.dumps(
                            self.history, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
                        if corrections >= 1:
                            self.end_failed_stage(role, entry, conflicts, corrections)
                        corrections += 1
                        continue
                    self.validate_reasoning(decision, records)
                    entry.update(stage_status='passed', automatic_corrections=corrections)
                (self.output / 'agent-interactions.json').write_text(json.dumps(
                    self.history, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
                if not decision.queries:
                    break
        else:
            decision, provenance = self.provider.generate('paramorph_' + role,
            'Return candidate_ids, evidence_ids, reason, missing_measurements, queries and reasoning_evidence. '
            'queries must be empty because this decision has no graph-query handler. '
            'The supplied response_contract specifies every required field. '
            'Choose candidate identifiers only from the supplied legal candidates. '
            'Use only supplied measured evidence identifiers. Performance claims require actual '
            'measurements. List unavailable compute, communication, overlap or memory measurements. '
            'For a diagnostic or compilation observation without selectable candidates, return '
            'an empty candidate_ids list. When decision_mode is experimental_priority, rank '
            'the supplied structurally valid pending candidates for the next measurement. '
            'Pending candidates are expected to lack measurements: do not reject all of them '
            'solely for that reason. Prefer explicit falsifiable graph mechanisms and useful '
            'evidence over unsupported claims; list uncertainty without asserting a speedup. '
            'Explain the decision in Chinese.',
            observation, EvidenceDecision)
        candidates = observation.get('candidates', [])
        allowed = {item['candidate_id'] for item in candidates}
        if set(decision.candidate_ids) - allowed:
            raise ValueError('Agent选择了观测中不存在的候选')
        evidence_source = observation.get('report', observation)
        allowed_evidence = {identifier for item in evidence_source.get('bottlenecks', [])
                            for identifier in item['evidence_ids']}
        allowed_evidence.update(item['evidence_id'] for item in evidence_source.get('evidence', []))
        if set(decision.evidence_ids) - allowed_evidence:
            raise ValueError('Agent引用了观测中不存在的实测证据')
        result = decision.model_dump(mode='json')
        if not any(item['provenance']['prompt_sha256'] == provenance['prompt_sha256']
                   and item['provenance']['response_sha256'] == provenance['response_sha256']
                   for item in self.history):
            self.history.append(dict(role=role, input=observation, output=result, provenance=provenance))
        (self.output / 'agent-interactions.json').write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return result
