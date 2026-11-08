from collections import defaultdict, deque
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
import math

from core_data_models import Task, TaskType
from scheduling.schedule_simulator import ScheduleCostEvaluator, topological_order


def _identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name}必须为有限非负数值')
    return float(value)


class BottleneckGraph:
    def __init__(self, graph, evidence_refs=()):
        self.graph = deepcopy(graph)
        self.records = self.graph['tasks']
        if not self.records:
            raise ValueError('瓶颈查询需要完整的非空执行图')
        declared = self.graph.get('full_task_count', self.graph.get('task_count', len(self.records)))
        if declared != len(self.records):
            raise ValueError('瓶颈查询必须使用后端完整执行图')
        self.evidence_refs = tuple(sorted(set(evidence_refs)))
        self.graph_id = _identity(self.graph)
        self.rank_mapping = {row.get('logical_rank', row.get('metadata', {}).get('logical_rank', row['rank'])):
            row.get('metadata', {}).get('physical_rank', row['rank']) for row in self.records.values()}
        self.predecessors = {key: set(row['dependencies']) for key, row in self.records.items()}
        self.successors = defaultdict(set)
        self.resource_tasks = defaultdict(set)
        self.collective_tasks = defaultdict(set)
        self.collective_keys = {}
        for key, row in self.records.items():
            if not self.predecessors[key] <= self.records.keys() or key in self.predecessors[key]:
                raise ValueError(f'任务依赖引用无效：{key}')
            for parent in self.predecessors[key]:
                self.successors[parent].add(key)
            for resource in row.get('resources', row.get('resource_requirements', ())):
                self.resource_tasks[resource].add(key)
            metadata = row.get('metadata', {})
            instance = metadata.get('collective_instance_id')
            if instance is not None:
                identity = ('collective', instance)
            elif 'channel' in metadata:
                identity = ('pipeline', metadata['channel'], row.get('microbatch'))
            elif metadata.get('group_ranks') and row.get('kind', row.get('operation', '')).startswith('fsdp_'):
                compute = self.records.get(metadata.get('compute_task'), {})
                identity = ('fsdp', row.get('kind', row.get('operation')),
                    metadata.get('state_family'), tuple(metadata['group_ranks']),
                    row.get('microbatch'), row.get('region', row.get('region_id')),
                    compute.get('kind', compute.get('operation')),
                    tuple(metadata.get('communication_payloads', [])))
            else:
                identity = None
            if identity is not None:
                self.collective_keys[key] = identity
                self.collective_tasks[identity].add(key)
        indegree = {key: len(value) for key, value in self.predecessors.items()}
        ready = deque(key for key, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            key = ready.popleft()
            visited += 1
            for successor in self.successors[key]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited != len(self.records):
            raise ValueError('瓶颈查询的完整执行图存在循环依赖')

    def _validate_task_ids(self, task_ids):
        result = set(task_ids)
        if not result or not result <= self.records.keys():
            raise ValueError('瓶颈查询必须引用完整执行图中存在的任务')
        return result

    def _participants(self, key):
        row = self.records[key]
        metadata = row.get('metadata', {})
        ranks = metadata.get('physical_group_ranks')
        if ranks is None:
            logical = metadata.get('participant_ranks', metadata.get('group_ranks', []))
            if not set(logical) <= self.rank_mapping.keys():
                raise ValueError(f'通信参与Rank缺少完整执行图记录：{key}')
            ranks = [self.rank_mapping[rank] for rank in logical]
        return sorted(set(ranks or [row['rank']]))

    def _resource_neighbours(self, key):
        row = self.records[key]
        candidates = set().union(*(self.resource_tasks[resource] for resource in
            row.get('resources', row.get('resource_requirements', ()))))
        if 'start_ms' not in row or 'end_ms' not in row:
            return candidates
        return {other for other in candidates if
            'start_ms' not in self.records[other] or 'end_ms' not in self.records[other] or
            max(row['start_ms'], self.records[other]['start_ms']) <
            min(row['end_ms'], self.records[other]['end_ms'])}

    def closure(self, seed_task_ids, dependency_depth=1, include_resources=True,
                include_predecessors=True, include_successors=True):
        seeds = self._validate_task_ids(seed_task_ids)
        if dependency_depth is not None and (type(dependency_depth) is not int or dependency_depth < 0):
            raise ValueError('依赖展开深度必须为非负整数或None')
        selected = set(seeds)
        reasons = {key: {'seed'} for key in seeds}
        frontier = set(seeds)
        depth = 0
        while frontier and (dependency_depth is None or depth <= dependency_depth):
            # 同一个通信实例和同一时段竞争资源的任务必须共同保留。
            pending = deque(sorted(frontier))
            expanded = set()
            while pending:
                key = pending.popleft()
                if key in expanded:
                    continue
                expanded.add(key)
                related = {'collective_participant': self.collective_tasks.get(
                    self.collective_keys.get(key), set())}
                if include_resources:
                    related['shared_resource'] = self._resource_neighbours(key)
                for reason, identities in related.items():
                    for identity in identities:
                        reasons.setdefault(identity, set()).add(reason)
                        if identity not in selected:
                            selected.add(identity)
                            pending.append(identity)
                            frontier.add(identity)
            if dependency_depth is not None and depth == dependency_depth:
                break
            following = set()
            for key in frontier:
                dependencies = []
                if include_predecessors:
                    dependencies.append(('predecessor', self.predecessors[key]))
                if include_successors:
                    dependencies.append(('successor', self.successors[key]))
                for reason, identities in dependencies:
                    for identity in identities:
                        reasons.setdefault(identity, set()).add(reason)
                        if identity not in selected:
                            selected.add(identity)
                            following.add(identity)
            frontier = following
            depth += 1
        boundary = sorted({parent for key in selected for parent in self.predecessors[key]
                           if parent not in selected})
        return dict(graph_id=self.graph_id, seed_task_ids=sorted(seeds),
            task_ids=sorted(selected), inclusion_reasons={key: sorted(reasons[key]) for key in selected},
            dependency_depth=dependency_depth, include_resources=include_resources,
            include_predecessors=include_predecessors, include_successors=include_successors,
            boundary_dependency_ids=boundary,
            closure_id=_identity(dict(graph=self.graph_id, seeds=sorted(seeds),
                task_ids=sorted(selected), dependency_depth=dependency_depth,
                include_resources=include_resources, include_predecessors=include_predecessors,
                include_successors=include_successors)))

    def _summary(self, key, reasons):
        row = self.records[key]
        metadata = row.get('metadata', {})
        evidence = metadata.get('duration_evidence')
        return dict(task_id=key, rank=row['rank'], logical_rank=row.get('logical_rank', row['rank']),
            region_id=row.get('region', row.get('region_id')),
            virtual_stage=row.get('virtual_stage', metadata.get('vp_stage')),
            microbatch=row.get('microbatch'), operation=row.get('kind', row.get('operation')),
            duration_ms=row.get('duration_ms'), start_ms=row.get('start_ms'), end_ms=row.get('end_ms'),
            dependency_count=len(self.predecessors[key]), successor_count=len(self.successors[key]),
            dependencies=sorted(self.predecessors[key])[:16],
            dependency_next_offset=16 if len(self.predecessors[key]) > 16 else None,
            collective_instance_id=metadata.get('collective_instance_id'),
            flow_id=(f'flow:{metadata["channel"]}:mb:{row.get("microbatch")}'
                     if 'channel' in metadata else metadata.get('collective_instance_id')),
            participant_ranks=self._participants(key),
            resources=row.get('resources', row.get('resource_requirements', [])),
            evidence_ref=evidence, inclusion_reasons=reasons)

    def causal_closure(self, seed_task_ids, replay, dependency_depth=1):
        if type(dependency_depth) is not int or dependency_depth < 0:
            raise ValueError('阻塞关系展开深度必须为非负整数')
        seeds = self._validate_task_ids(seed_task_ids)
        diagnostics = replay['resource_diagnostics']
        waits = diagnostics['task_wait_causes']
        related = defaultdict(lambda: defaultdict(set))
        for key, row in waits.items():
            related[key]['dependency_ready'].update(row['dependency_task_ids'])
            related[key]['exclusive_resource_wait'].update(item['task_id'] for item in row['resource_blockers'])
        for row in diagnostics['bandwidth_sharing_intervals']:
            for key in row['task_ids']:
                related[key]['bandwidth_sharing'].update(set(row['task_ids']) - {key})
        for row in diagnostics['compute_communication_overlap_intervals']:
            if row['gamma'] > 1:
                related[row['compute_task_id']]['compute_communication_slowdown'].update(row['communication_task_ids'])
        selected = set(seeds)
        reasons = {key: {'seed'} for key in seeds}
        frontier = set(seeds)
        for depth in range(dependency_depth + 1):
            following = set()
            for key in frontier:
                peers = self.collective_tasks.get(self.collective_keys.get(key), set())
                connections = dict(collective_participant=peers)
                if depth < dependency_depth:
                    connections.update(related[key])
                for reason, identities in connections.items():
                    for identity in identities:
                        reasons.setdefault(identity, set()).add(reason)
                        if identity not in selected:
                            selected.add(identity)
                            following.add(identity)
            frontier = following
        boundary = sorted({parent for key in selected for parent in self.predecessors[key] if parent not in selected})
        return dict(graph_id=self.graph_id, seed_task_ids=sorted(seeds), task_ids=sorted(selected),
            inclusion_reasons={key: sorted(reasons[key]) for key in selected},
            boundary_dependency_ids=boundary, scope='replayed_blocking_relations',
            closure_id=_identity(dict(graph=self.graph_id, seeds=sorted(seeds),
                selected=sorted(selected), depth=dependency_depth, scope='replayed_blocking_relations')))

    @staticmethod
    def _bounded_page(header, rows, offset, limit, max_chars, row_name):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError('子图查询分页参数无效')
        if type(max_chars) is not int or max_chars < 2048:
            raise ValueError('子图查询文本预算必须至少为2048字符')
        if offset > len(rows):
            raise ValueError('子图查询分页偏移超过结果数量')
        result = header | {row_name: [], 'offset': offset, 'total_count': len(rows),
                           'next_offset': offset if offset < len(rows) else None}
        for row in rows[offset:offset + limit]:
            candidate = deepcopy(result)
            candidate[row_name].append(row)
            end = offset + len(candidate[row_name])
            candidate['next_offset'] = end if end < len(rows) else None
            if len(json.dumps(candidate, ensure_ascii=False)) > max_chars:
                break
            result = candidate
        if offset < len(rows) and not result[row_name]:
            raise ValueError('单个真实任务记录超过文本预算，需要增加max_chars')
        return result

    def page(self, closure, offset=0, limit=24, max_chars=12000):
        if closure['graph_id'] != self.graph_id:
            raise ValueError('子图分页引用了其他执行图')
        self._validate_task_ids(closure['task_ids'])
        header = dict(status='available', graph_id=self.graph_id, closure_id=closure['closure_id'],
            complete_graph_task_count=len(self.records),
            selected_task_count=len(closure['task_ids']),
            boundary_dependency_count=len(closure['boundary_dependency_ids']),
            source_scope='original_complete_executable_dag', evidence_refs=list(self.evidence_refs),
            detail_access='task_dependencies及task_ids继续查询完整后端图')
        rows = [self._summary(key, closure['inclusion_reasons'][key]) for key in closure['task_ids']]
        return self._bounded_page(header, rows, offset, limit, max_chars, 'tasks')

    def task_dependencies(self, task_id, direction='predecessors', offset=0, limit=32,
                          max_chars=12000):
        self._validate_task_ids([task_id])
        if direction not in ('predecessors', 'successors'):
            raise ValueError('依赖查询方向无效')
        index = self.predecessors if direction == 'predecessors' else self.successors
        rows = [self._summary(key, [direction]) for key in sorted(index[task_id])]
        return self._bounded_page(dict(graph_id=self.graph_id, task_id=task_id,
            direction=direction, scope='original_complete_executable_dag'),
            rows, offset, limit, max_chars, 'tasks')

    def replay(self, duration_overrides=None, config=None, resource_model=None):
        overrides = duration_overrides or {}
        if not set(overrides) <= self.records.keys():
            raise ValueError('反事实成本引用了不存在的任务')
        tasks = {}
        for key, row in self.records.items():
            if 'resources' not in row or 'metadata' not in row or 'duration_ms' not in row:
                raise ValueError('反事实Replay需要完整资源、任务元数据和成本记录')
            duration = _finite_nonnegative(overrides.get(key, row['duration_ms']), f'{key}耗时')
            metadata = deepcopy(row['metadata'])
            if not metadata.get('duration_evidence'):
                raise ValueError(f'反事实Replay的任务缺少成本来源：{key}')
            tasks[key] = Task(task_id=key, rank=row.get('logical_rank', row['rank']),
                task_type=TaskType(row.get('kind', row.get('operation'))),
                microbatch_id=row.get('microbatch'), region_id=row.get('region', row.get('region_id')),
                duration=duration, dependencies=list(row['dependencies']), metadata=metadata,
                resource_requirements=tuple(row['resources']))
        _, predecessors, _ = topological_order(tasks)
        times, critical_parent, diagnostics = ScheduleCostEvaluator(None).simulate_resource_aware_events(
            tasks, predecessors, {key: task.duration for key, task in tasks.items()}, config, resource_model)
        end = max(times, key=lambda key: times[key][1])
        step = times[end][1]
        critical = []
        while end is not None:
            critical.append(end)
            end = critical_parent[end]
        return dict(graph_id=self.graph_id, resource_model_id=_identity(resource_model or {}),
            step_time_ms=step, critical_task_ids=list(reversed(critical)),
            task_times=times, resource_diagnostics=diagnostics)

    def epsilon_impact(self, task_ids, epsilon, config=None, resource_model=None, baseline_replay=None):
        selected = self._validate_task_ids(task_ids)
        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError('反事实epsilon必须处于(0,1]')
        before = baseline_replay or self.replay(config=config, resource_model=resource_model)
        if before['graph_id'] != self.graph_id or before['resource_model_id'] != _identity(resource_model or {}):
            raise ValueError('目标任务敏感性需要相同执行图与资源模型的参照回放')
        overrides = {key: (1 - epsilon) * self.records[key]['duration_ms'] for key in selected}
        after = self.replay(overrides, config=config, resource_model=resource_model)
        return dict(graph_id=self.graph_id, selected_task_ids=sorted(selected), epsilon=epsilon,
            baseline_step_time_ms=before['step_time_ms'], counterfactual_step_time_ms=after['step_time_ms'],
            impact_ms=before['step_time_ms'] - after['step_time_ms'],
            baseline_critical_task_ids=before['critical_task_ids'],
            counterfactual_critical_task_ids=after['critical_task_ids'],
            replay_count=1 if baseline_replay is not None else 2,
            target_wait_causes={key: before['resource_diagnostics']['task_wait_causes'][key] for key in selected},
            evidence_refs=list(self.evidence_refs),
            resource_model_id=_identity(resource_model or {}),
            evidence_type='predicted_counterfactual', measured_strategy_gain=False,
            scope='selected_task_duration_scaling_with_unchanged_full_dag_resources_and_dependencies')


def ranking_changes(before, after, top_k):
    if len(set(before)) != len(before) or set(before) != set(after) or len(after) != len(before):
        raise ValueError('排序比较必须覆盖相同且唯一的候选')
    if type(top_k) is not int or top_k < 1 or not before:
        raise ValueError('排序比较需要非空候选和正整数Top-K')
    first, second = before[:top_k], after[:top_k]
    common = set(first) & set(second)
    positions = {key: index for index, key in enumerate(second)}
    reversals = [[left, right] for left, right in combinations(first, 2)
                 if left in common and right in common and positions[left] > positions[right]]
    return dict(best_changed=before[0] != after[0], topk_order_reversals=reversals,
        topk_membership_changed=set(first) != set(second),
        entered_topk=sorted(set(second) - set(first)), exited_topk=sorted(set(first) - set(second)))


def shared_cost_ranking_scenarios(baseline_ms, cost_key, low, high, affected_candidate_ids,
                                  predict_with_override, top_k=3, gpu_hours=None):
    if not baseline_ms or not isinstance(cost_key, str) or not cost_key:
        raise ValueError('共享成本排序分析需要候选和稳定cost_key')
    baseline = {key: _finite_nonnegative(value, '候选Step Time') for key, value in baseline_ms.items()}
    low = _finite_nonnegative(low, '低成本情景')
    high = _finite_nonnegative(high, '高成本情景')
    if low > high:
        raise ValueError('低成本情景不能超过高成本情景')
    affected = set(affected_candidate_ids)
    if not affected or not affected <= baseline.keys():
        raise ValueError('共享成本必须引用已评价且实际关联的候选')
    if gpu_hours is not None:
        gpu_hours = _finite_nonnegative(gpu_hours, '测量GPUHours')
    order = sorted(baseline, key=lambda key: (baseline[key], key))
    scenarios = {}
    for label, value in (('low', low), ('high', high)):
        times = dict(baseline)
        # 一个cost_key的情景值同时作用于全部关联候选。
        for candidate_id in sorted(affected):
            times[candidate_id] = _finite_nonnegative(
                predict_with_override(candidate_id, {cost_key: value}), '情景Step Time')
        ranking = sorted(times, key=lambda key: (times[key], key))
        scenarios[label] = dict(cost_value=value, step_time_ms=times, ranking=ranking,
            changes_vs_baseline=ranking_changes(order, ranking, top_k))
    between = ranking_changes(scenarios['low']['ranking'], scenarios['high']['ranking'], top_k)
    changes = [scenarios[label]['changes_vs_baseline'] for label in ('low', 'high')] + [between]
    best = any(row['best_changed'] for row in changes)
    internal = any(row['topk_order_reversals'] for row in changes)
    membership = any(row['topk_membership_changed'] for row in changes)
    return dict(cost_key=cost_key, baseline_ranking=order, affected_candidate_ids=sorted(affected),
        reuse_count=len(affected), scenarios=scenarios, changes_between_scenarios=between,
        best_changed=best, topk_internal_order_changed=internal, topk_membership_changed=membership,
        priority_class=0 if best else 1 if internal else 2 if membership else 3,
        profiling_gpu_hours=gpu_hours, evaluation_count=2 * len(affected),
        scenario_scope='shared_cost_sensitivity_scenarios', statistical_confidence_interval=False)


def collective_cost_bindings(report, cost_key):
    records = [row for row in report['evidence']['costs'] if row.get('cost_key') == cost_key]
    if not records:
        return {}
    unique = {_identity(row): row for row in records}
    bindings = {}
    for task_id, task in report['execution_graph']['tasks'].items():
        metadata = task['metadata']
        if 'cost_components' in metadata:
            components = [row for row in metadata['cost_components'] if row['cost_key'] == cost_key]
            if components:
                if any(type(row['occurrences']) is not int or row['occurrences'] < 1 for row in components):
                    raise ValueError('任务成本出现次数必须为正整数')
                known = {row['milliseconds'] for row in records}
                if any(row['milliseconds'] not in known for row in components):
                    raise ValueError('任务成本绑定与原始成本证据不一致')
                bindings[task_id] = deepcopy(components)
            continue
        if not metadata.get('collective') or not metadata.get('communication_phases'):
            continue
        matches = [row for row in unique.values()
            if row.get('collective') == metadata['collective']
            and row.get('ranks') == metadata.get('physical_group_ranks')
            and row.get('source') == metadata.get('duration_evidence')
            and row.get('communication_phases') == metadata['communication_phases']
            and row.get('milliseconds') == task['duration_ms']]
        if matches:
            signatures = {(row['milliseconds'], row.get('dtype'), tuple(row['ranks'])) for row in matches}
            if len(signatures) != 1:
                raise ValueError('独立Collective具有冲突的成本证据')
            bindings[task_id] = [dict(cost_key=cost_key, milliseconds=matches[0]['milliseconds'],
                occurrences=1, evidence_binding='exact_collective_participants_service_phases_and_source')]
    return bindings


def missing_anchor_predictions(reports, anchor_reports):
    missing = []
    available = {report['prediction_id']: report for report in reports} | anchor_reports
    for report in reports:
        if not report['performance'].get('graph_delta_anchored', False):
            continue
        anchor = report['evidence'].get('graph_delta_calibration')
        if not isinstance(anchor, dict) or not anchor.get('prediction_id'):
            raise ValueError('差分校准报告缺少实际父预测身份')
        if anchor['prediction_id'] not in available:
            missing.append(dict(candidate_id=report['candidate_id'],
                anchor_prediction_id=anchor['prediction_id'], reason='missing_actual_anchor_prediction'))
            continue
        parent = available[anchor['prediction_id']]
        current_revision = report.get('cost_database_revision')
        parent_revision = parent.get('cost_database_revision')
        if current_revision is None or parent_revision is None or current_revision != parent_revision:
            missing.append(dict(candidate_id=report['candidate_id'],
                anchor_prediction_id=anchor['prediction_id'], reason='incompatible_anchor_cost_revision',
                candidate_cost_revision=current_revision, anchor_cost_revision=parent_revision))
            continue
        tasks = parent.get('execution_graph', {}).get('tasks', {})
        if not tasks or any(row.get('duration_ms', 0) > 0 and
                'cost_components' not in row.get('metadata', {}) for row in tasks.values()):
            missing.append(dict(candidate_id=report['candidate_id'],
                anchor_prediction_id=anchor['prediction_id'], reason='incomplete_anchor_cost_bindings'))
    return missing


def replay_shared_cost_scenarios(reports, cost_key, low, high, resource_models=None,
                                 top_k=3, gpu_hours=None, anchor_reports=None):
    if len({report['candidate_id'] for report in reports}) != len(reports):
        raise ValueError('共享成本Replay的候选身份重复')
    anchors = anchor_reports or {}
    missing = missing_anchor_predictions(reports, anchors)
    if missing:
        return dict(status='UNAVAILABLE', reason='anchor_prediction_not_evaluable', anchor_issues=missing,
            cost_key=cost_key, candidate_ids=sorted({row['candidate_id'] for row in missing}),
            missing_anchor_prediction_ids=sorted({row['anchor_prediction_id'] for row in missing
                if row['reason'] == 'missing_actual_anchor_prediction'}))
    sources = {report['prediction_id']: report for report in reports}
    referenced_anchors = {report['evidence']['graph_delta_calibration']['prediction_id']
        for report in reports if report['performance'].get('graph_delta_anchored', False)}
    for identity in referenced_anchors - sources.keys():
        if anchors[identity]['prediction_id'] != identity:
            raise ValueError('差分校准父图身份与实际预测记录不一致')
        sources[identity] = anchors[identity]
    graphs = {identity: BottleneckGraph(report['execution_graph'],
        evidence_refs=[identity]) for identity, report in sources.items()}
    bindings = {identity: collective_cost_bindings(report, cost_key) for identity, report in sources.items()}
    models = resource_models or {}
    source_models = {identity: models.get(identity, models.get(report['candidate_id']))
                     for identity, report in sources.items()}
    raw = {}
    for identity, report in sources.items():
        actual = graphs[identity].replay(resource_model=source_models[identity])['step_time_ms']
        expected = report['performance']['raw_resource_aware_step_time_ms']
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f'{identity}的原始Replay不一致，需要完整原始资源模型')
        raw[identity] = actual
    baseline = {}
    candidate_reports = {report['candidate_id']: report for report in reports}
    affected = []
    for report in reports:
        candidate_id = report['candidate_id']
        identity = report['prediction_id']
        value = raw[identity]
        has_binding = bool(bindings[identity])
        if report['performance'].get('graph_delta_anchored', False):
            calibration = report['evidence']['graph_delta_calibration']
            parent_id = calibration['prediction_id']
            measured = _finite_nonnegative(calibration['measured_step_time_ms'], '父计划实测Step Time')
            if not math.isclose(raw[parent_id], calibration['raw_parent_step_time_ms'], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('父图Replay与差分校准记录的父成本不一致')
            if not math.isclose(value, calibration['raw_candidate_step_time_ms'], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('候选图Replay与差分校准记录的候选成本不一致')
            value = measured + value - raw[parent_id]
            has_binding = has_binding or bool(bindings[parent_id])
        if not math.isclose(value, report['performance']['step_time_ms'], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('共享成本基准与候选实际使用的预测Step Time不一致')
        baseline[candidate_id] = value
        if has_binding:
            affected.append(candidate_id)

    scenario_cache = {}

    def replay_source(identity, value):
        if not bindings[identity]:
            return raw[identity]
        cache_key = identity, value
        if cache_key in scenario_cache:
            return scenario_cache[cache_key]
        graph = graphs[identity]
        overrides = {task_id: graph.records[task_id]['duration_ms'] + sum(
            (value - row['milliseconds']) * row['occurrences'] for row in components)
            for task_id, components in bindings[identity].items()}
        actual = graph.replay(overrides, resource_model=source_models[identity])['step_time_ms']
        scenario_cache[cache_key] = actual
        return actual

    def predict(candidate_id, override):
        report = candidate_reports[candidate_id]
        value = override[cost_key]
        current = replay_source(report['prediction_id'], value)
        if report['performance'].get('graph_delta_anchored', False):
            calibration = report['evidence']['graph_delta_calibration']
            return (calibration['measured_step_time_ms'] + current
                    - replay_source(calibration['prediction_id'], value))
        return current

    result = shared_cost_ranking_scenarios(baseline, cost_key, low, high,
        affected, predict, top_k, gpu_hours)
    result.update(status='AVAILABLE', affected_task_ids={report['candidate_id']: sorted(bindings[report['prediction_id']])
            for report in reports if bindings[report['prediction_id']]},
        affected_anchor_task_ids={key: sorted(bindings[key]) for key in referenced_anchors if bindings[key]},
        anchor_prediction_ids=sorted(referenced_anchors),
        baseline_step_time_ms=baseline, baseline_replay_count=len(sources),
        scenario_candidate_evaluation_count=result['evaluation_count'], evaluation_count=len(scenario_cache),
        replay_engine='ScheduleCostEvaluator.simulate_resource_aware_events',
        step_time_scope=('measured_anchor_plus_candidate_raw_minus_anchor_raw'
                        if referenced_anchors else 'raw_resource_aware_dag_replay'))
    return result
