from collections import defaultdict
from itertools import accumulate

import networkx as nx


class ExecutionPathModel:
    def __init__(self, request, profile, graph):
        self.slots = len(request['decoder_chunks'])
        self.prefix = {phase: list(accumulate((int(round(1000 * profile['layers'][str(index)][phase + '_ms']))
            for index in range(sum(request['decoder_chunks']))), initial=0))
            for phase in ('forward', 'backward')}
        self.cuts, self.graph_tasks, self.resource_edges = self.extract(request, graph)
        self.raw_step_ms = max(task['end_ms'] for task in graph['tasks'].values())
        if abs(self.score(request['decoder_chunks']) - self.raw_step_ms) > 0.001:
            raise ValueError('执行路径代理未保持参照布局的完整时间')

    def costs(self, counts):
        if (len(counts) != self.slots or min(counts) < 1
                or sum(counts) != len(self.prefix['forward']) - 1):
            raise ValueError('路径代理需要覆盖全部模块的相同区间数量')
        boundaries = list(accumulate(counts, initial=0))
        return [self.prefix[phase][right] - self.prefix[phase][left]
            for phase in ('forward', 'backward') for left, right in zip(boundaries, boundaries[1:])]

    def extract(self, request, graph):
        tasks = graph['tasks']
        dag = nx.DiGraph()
        dag.add_nodes_from(tasks)
        for key, task in tasks.items():
            for parent in task['dependencies']:
                if parent not in tasks:
                    raise ValueError('执行路径需要完整任务依赖')
                dag.add_edge(parent, key)
        resources = defaultdict(list)
        for key, task in tasks.items():
            if task['end_ms'] <= task['start_ms']:
                continue
            for resource in task['resources']:
                if resource.endswith(':compute') or resource.startswith('rank:'):
                    resources[resource].append(key)
        resource_edges = 0
        for keys in resources.values():
            ordered = sorted(keys, key=lambda key: (tasks[key]['start_ms'], tasks[key]['end_ms'], key))
            for before, after in zip(ordered, ordered[1:]):
                if tasks[before]['end_ms'] > tasks[after]['start_ms'] + 1e-5:
                    raise ValueError('参照执行中的排他资源出现时间重叠')
                dag.add_edge(before, after)
                resource_edges += 1
        order = list(nx.topological_sort(dag))
        heads, tails, forward_counts, backward_counts = {}, {}, {}, {}
        duration = {key: task['end_ms'] - task['start_ms'] for key, task in tasks.items()}
        edges = list(accumulate(request['decoder_chunks'], initial=0))
        units = {}
        for key, task in tasks.items():
            coefficients = [0] * (2 * self.slots)
            if task['kind'] in ('forward_compute', 'backward_compute'):
                slot = task['region']
                if (type(slot) is not int or not 0 <= slot < self.slots
                        or task['metadata']['layer_ids'] != list(range(edges[slot], edges[slot + 1]))):
                    raise ValueError('执行路径任务与完整模块区间不一致')
                coefficients[slot + (self.slots if task['kind'] == 'backward_compute' else 0)] = 1
            units[key] = coefficients
        for key in order:
            parent = max(dag.predecessors(key), key=lambda value: heads[value], default=None)
            heads[key] = duration[key] + (0 if parent is None else heads[parent])
            forward_counts[key] = list(units[key]) if parent is None else [a + b
                for a, b in zip(units[key], forward_counts[parent])]
        for key in reversed(order):
            child = max(dag.successors(key), key=lambda value: tails[value], default=None)
            tails[key] = duration[key] + (0 if child is None else tails[child])
            backward_counts[key] = list(units[key]) if child is None else [a + b
                for a, b in zip(units[key], backward_counts[child])]
        raw = max(task['end_ms'] for task in tasks.values())
        if abs(max(heads.values()) - raw) > 1e-4:
            raise ValueError('资源依赖没有重建完整参照执行时间')
        baseline_costs = self.costs(request['decoder_chunks'])
        cuts = {}
        for middle in order:
            coefficients = [a + b - c for a, b, c in zip(forward_counts[middle], backward_counts[middle], units[middle])]
            path_time = heads[middle] + tails[middle] - duration[middle]
            constant = int(round(1000 * path_time)) - sum(
                coefficient * cost for coefficient, cost in zip(coefficients, baseline_costs))
            identity = tuple(coefficients)
            if identity not in cuts or cuts[identity] < constant:
                cuts[identity] = constant
        return [(list(coefficients), constant) for coefficients, constant in cuts.items()], len(tasks), resource_edges

    def score(self, counts):
        costs = self.costs(counts)
        return max(constant + sum(a * b for a, b in zip(coefficients, costs))
            for coefficients, constant in self.cuts) / 1000

    def attach(self, model, variables):
        if len(variables) != self.slots * 2:
            raise ValueError('路径代理缺少完整区间的前向和反向变量')
        bounds = [self.prefix[phase][-1] for phase in ('forward', 'backward') for _ in range(self.slots)]
        upper = max(max(0, constant) + sum(a * b for a, b in zip(coefficients, bounds))
            for coefficients, constant in self.cuts)
        objective = model.new_int_var(0, upper, 'path_span')
        for coefficients, constant in self.cuts:
            model.add(objective >= constant + sum(coefficient * variable
                for coefficient, variable in zip(coefficients, variables)))
        return objective

    def summary(self):
        return dict(path_count=len(self.cuts), graph_tasks=self.graph_tasks,
            resource_edges=self.resource_edges, reference_raw_ms=self.raw_step_ms,
            objective_scope='observed_schedule_path_surrogate_full_DAG_selects_candidates')

    def influence(self, request):
        counts = request['decoder_chunks']
        edges = list(accumulate(counts, initial=0))
        base = self.score(counts)
        effects = []
        for boundary in range(1, self.slots):
            for shift in (-1, 1):
                changed = list(counts)
                changed[boundary - 1] += shift
                changed[boundary] -= shift
                if min(changed) < 1:
                    continue
                effects.append([boundary, shift, edges[boundary] if shift > 0 else edges[boundary] - 1,
                    boundary if shift > 0 else boundary - 1, boundary - 1 if shift > 0 else boundary,
                    self.score(changed) - base])
        costs = self.costs(counts)
        ordered = sorted(self.cuts, key=lambda cut: cut[1] + sum(a * b for a, b in zip(cut[0], costs)), reverse=True)
        return dict(self.summary(), boundary_columns=['boundary', 'shift', 'module_id', 'source_virtual',
            'target_virtual', 'predicted_path_change_ms'], boundary_effects=effects,
            longest_observed_paths=[dict(forward_counts=coefficients[:self.slots],
                backward_counts=coefficients[self.slots:],
                baseline_path_ms=(constant + sum(a * b for a, b in zip(coefficients, costs))) / 1000)
                for coefficients, constant in ordered[:4]])
