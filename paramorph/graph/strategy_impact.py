from collections import defaultdict
from math import isfinite

import networkx as nx

from contracts import fingerprint
from graph.bottleneck_analysis import BottleneckGraph


class GraphStrategyImpact:
    @staticmethod
    def compare_candidates(left, before, before_graph, right, after, after_graph):
        from search.execution_comparison import compare_execution_reports

        reference_tasks = (before_graph or {}).get('tasks', {})
        candidate_tasks = (after_graph or {}).get('tasks', {})
        changed = reference_tasks.keys() ^ candidate_tasks.keys()
        fields = ('kind', 'rank', 'dependencies', 'resources', 'duration_ms', 'start_ms', 'end_ms')
        changed.update(key for key in reference_tasks.keys() & candidate_tasks.keys()
            if any(reference_tasks[key].get(field) != candidate_tasks[key].get(field) for field in fields))

        def affected(tasks, prediction):
            critical = set(prediction.get('diagnosis', {}).get('critical_tasks', []))
            return sorted(changed & tasks.keys(), key=lambda key: (key not in critical,
                tasks[key].get('duration_ms') is not None, key))[:16]

        comparison = compare_execution_reports(left,
            before | dict(execution_graph=before_graph), right, after | dict(execution_graph=after_graph))
        delta = comparison.get('predicted_step_delta_ms')
        return dict(status=comparison['status'],
            task_ids=dict(reference=affected(reference_tasks, before), candidate=affected(candidate_tasks, after)),
            changed_task_count=len(changed),
            changes=dict(step_delta_ms=delta, predicted_gain_ms=None if delta is None else -delta,
                communication_delta_ms=comparison.get('exposed_communication_delta_ms'),
                peak_memory_delta_bytes=comparison.get('peak_memory_delta_bytes'),
                operations=comparison.get('operation_changes', [])))

    def __init__(self, graph, replay, memory):
        self.graph = BottleneckGraph(graph)
        self.replay = replay
        self.memory = memory
        if replay['graph_id'] != self.graph.graph_id or set(replay['task_times']) != self.graph.records.keys():
            raise ValueError('任务影响分析需要同一完整执行图的回放')
        self.times = replay['task_times']
        self.span = replay['step_time_ms']
        self.paths = self.path_lengths()

    def quantitative_probes(self, impact, *, config=None, resource_model=None, limit=3):
        buckets = defaultdict(list)
        for row in impact['tasks']:
            if row['duration_ms'] > 0:
                buckets[row['operation'], row['region']].append(row)
        ordered = sorted(buckets.items(), key=lambda item: (
            -sum(row['replay_critical'] for row in item[1]),
            -sum(row['resource_wait_ms'] for row in item[1]), str(item[0])))
        reports = []
        for (operation, region), rows in ordered[:limit]:
            task_ids = [row['task_id'] for row in rows]
            scenario = self.graph.epsilon_impact(task_ids, 0.10, config=config,
                resource_model=resource_model, baseline_replay=self.replay)
            record = dict(operation=operation, region_id=region, target_task_ids=task_ids,
                target_wait_causes=scenario['target_wait_causes'],
                full_iteration_impact_ms=scenario['impact_ms'], task_duration_reduction_fraction=0.10,
                baseline_step_time_ms=scenario['baseline_step_time_ms'],
                scenario_step_time_ms=scenario['counterfactual_step_time_ms'],
                scope=scenario['scope'], source_impact_id=impact['impact_id'])
            reports.append(record | dict(probe_id=fingerprint(record)))
        return reports

    def path_lengths(self):
        ordering = nx.DiGraph()
        ordering.add_nodes_from(self.graph.records)
        for task_id, predecessors in self.graph.predecessors.items():
            ordering.add_edges_from((previous, task_id) for previous in predecessors)
        serial = defaultdict(list)
        for task_id, task in self.graph.records.items():
            start, end = self.times[task_id]
            if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
                raise ValueError('任务执行时间无效')
            if start < end:
                for resource in task['resources']:
                    if resource.startswith('rank:') or resource.endswith(':compute'):
                        serial[resource].append(task_id)
        for resource, tasks in serial.items():
            ordered = sorted(tasks, key=lambda key: (*self.times[key], key))
            for left, right in zip(ordered, ordered[1:]):
                if self.times[left][1] <= self.times[right][0] + 1e-6:
                    ordering.add_edge(left, right)
                else:
                    raise ValueError(f'排他资源的回放任务发生时间重叠：{resource}')
        order = list(nx.topological_sort(ordering))
        durations = {task: end - start for task, (start, end) in self.times.items()}
        head, tail = {}, {}
        for task in order:
            head[task] = durations[task] + max((head[key] for key in ordering.predecessors(task)), default=0)
        for task in reversed(order):
            tail[task] = durations[task] + max((tail[key] for key in ordering.successors(task)), default=0)
        return {key: head[key] + tail[key] - durations[key] for key in order}

    def describe(self, task_ids=None, *, limit=24, slack_fraction=0.03):
        if limit < 1 or not 0 <= slack_fraction <= 1:
            raise ValueError('任务影响查询需要合法的返回范围')
        slack = max(0.05, self.span * slack_fraction)
        critical = set(self.replay['critical_task_ids'])
        seeds = (list(dict.fromkeys(task_ids)) if task_ids is not None else
            sorted(self.paths, key=lambda key: (key not in critical, -self.paths[key], key))[:limit])
        closure = self.graph.causal_closure(seeds, self.replay)
        waits = self.replay['resource_diagnostics']['task_wait_causes']
        records = []
        for key in closure['task_ids']:
            task = self.graph.records[key]
            start, end = self.times[key]
            wait = waits.get(key, {})
            gap = max(0.0, self.span - self.paths[key])
            records.append(dict(task_id=key, region=task.get('region', task.get('region_id')),
                operation=task.get('kind', task.get('operation')), rank=task['rank'], resources=task['resources'],
                duration_ms=end - start, path_gap_ms=gap, near_critical=gap <= slack, replay_critical=key in critical,
                cost_source=task['metadata'].get('duration_evidence'),
                resource_wait_ms=wait.get('resource_wait_ms', 0),
                relations=closure['inclusion_reasons'][key]))
        ranked = sorted(records, key=lambda row: ('seed' not in row['relations'], not row['replay_critical'],
            row['path_gap_ms'], -row['resource_wait_ms'], row['task_id']))
        regions = sorted({row['region'] for row in records if type(row['region']) is int and row['region'] >= 0})
        document = dict(kind='graph_strategy_impact', graph_id=self.graph.graph_id, replay_ms=self.span,
            slack_ms=slack, seed_task_ids=closure['seed_task_ids'], affected_task_count=len(records),
            affected_regions=regions, tasks=ranked[:max(limit, len(seeds))],
            memory={key: self.memory[key] for key in ('admission', 'peak_per_rank_bytes',
                'gpu_training_peak_per_rank_bytes', 'persistent_per_rank_bytes',
                'memory_headroom_per_rank_bytes', 'peak_live_objects')
                if key in self.memory}, path_scope='local_reference_schedule')
        document['memory']['largest_buffer_lifetimes'] = sorted(self.memory.get('buffer_lifetimes', []),
            key=lambda row: (-row['bytes'], row['buffer_id']))[:8]
        return document | dict(impact_id=fingerprint(document))
