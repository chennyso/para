from collections import defaultdict
from itertools import accumulate

import networkx as nx

from graph.process_groups import process_groups


def partition_cost_prefixes(profile, layers):
    phases = {phase: list(accumulate((int(round(1000 * profile['layers'][str(index)][phase + '_ms']))
        for index in range(layers)), initial=0)) for phase in ('forward', 'backward')}
    parameters = profile.get('module_parameters') or {}
    if set(parameters) == {str(index) for index in range(layers)}:
        for family in ('dense', 'expert'):
            phases[family + '_parameters'] = list(accumulate(
                (round(parameters[str(index)][family] / 1024) for index in range(layers)), initial=0))
    return phases


class ExecutionPathModel:
    def __init__(self, request, profile, graph):
        self.coefficient_scale = 10000
        self.slots = len(request['decoder_chunks'])
        self.prefix = partition_cost_prefixes(profile, sum(request['decoder_chunks']))
        self.stages = {rank: stage for group in process_groups(request)['logical']['pp']
            for stage, rank in enumerate(group)} if 'dense_parameters' in self.prefix else {}
        self.raw_step_ms = max(task['end_ms'] for task in graph['tasks'].values())
        self.near_critical_slack_ms = max(0.05, self.raw_step_ms * 0.03)
        self.cuts, self.graph_tasks, self.resource_edges = self.extract(request, graph)
        if abs(self.score(request['decoder_chunks']) - self.raw_step_ms) > 0.001:
            raise ValueError('执行路径代理未保持参照布局的完整时间')

    def costs(self, counts):
        if (len(counts) != self.slots or min(counts) < 1
                or sum(counts) != len(self.prefix['forward']) - 1):
            raise ValueError('路径代理需要覆盖全部模块的相同区间数量')
        boundaries = list(accumulate(counts, initial=0))
        return [self.prefix[phase][right] - self.prefix[phase][left]
            for phase in self.prefix for left, right in zip(boundaries, boundaries[1:])]

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
        heads, tails = {}, {}
        duration = {key: task['end_ms'] - task['start_ms'] for key, task in tasks.items()}
        waiting = {}
        for key in order:
            ready = max((tasks[parent]['end_ms'] for parent in dag.predecessors(key)),default=0.0)
            delay = tasks[key]['start_ms'] - ready
            if delay < -1e-5:
                raise ValueError('参照执行时间早于已登记依赖的完成时间')
            waiting[key] = max(0.0,delay)
        self.observed_wait_ms = waiting
        elapsed = {key:duration[key]+waiting[key] for key in tasks}
        edges = list(accumulate(request['decoder_chunks'], initial=0))
        baseline_costs = self.costs(request['decoder_chunks'])
        units = {}
        for key, task in tasks.items():
            coefficients = [0] * (len(self.prefix) * self.slots)
            if task['kind'] in ('forward_compute', 'backward_compute'):
                slot = task['region']
                if (type(slot) is not int or not 0 <= slot < self.slots
                        or task['metadata']['layer_ids'] != list(range(edges[slot], edges[slot + 1]))):
                    raise ValueError('执行路径任务与完整模块区间不一致')
                if duration[key] > 0:
                    coefficients[slot + (self.slots if task['kind'] == 'backward_compute' else 0)] = self.coefficient_scale
            elif (task['kind'] == 'cuda_kernel' and task['metadata'].get('logical_layer') is not None
                    and task['metadata'].get('operation') in ('forward_compute', 'backward_compute')):
                slot = task['region']
                layer = task['metadata']['logical_layer']
                if type(slot) is not int or not 0 <= slot < self.slots or not edges[slot] <= layer < edges[slot + 1]:
                    raise ValueError('层级执行路径与原始模块区间不一致')
                index = slot + (self.slots if task['metadata']['operation'] == 'backward_compute' else 0)
                if baseline_costs[index] <= 0:
                    raise ValueError('层级执行路径需要正数的区域成本')
                coefficients[index] = int(round(self.coefficient_scale * duration[key] * 1000 / baseline_costs[index]))
            basis = task['metadata'].get('partition_cost_basis')
            if basis is not None and self.stages and duration[key] > 0:
                logical_rank = task.get('logical_rank', task['metadata'].get('logical_rank', task['rank']))
                stage = self.stages[logical_rank]
                slots = ([task['region']] if basis['scope'] == 'region'
                    else list(range(stage, self.slots, request['pp'])))
                denominator = basis['reference_elements']
                if denominator > 0:
                    for component, scale in basis['elements_per_parameter'].items():
                        offset = list(self.prefix).index(component + '_parameters') * self.slots
                        for slot in slots:
                            coefficients[offset + slot] = int(round(
                                duration[key] * 1000 * self.coefficient_scale * 1024 * scale / denominator))
            units[key] = tuple(coefficients)

        def collect(sequence, adjacent, paths):
            for key in sequence:
                options = {}
                for other in adjacent(key):
                    for length, previous in paths[other]:
                        coefficients = tuple(a + b for a, b in zip(units[key], previous))
                        options[coefficients] = max(options.get(coefficients, 0), length + elapsed[key])
                if not options:
                    paths[key] = [(elapsed[key], units[key])]
                    continue
                longest = max(options.values())
                paths[key] = sorted(((length, coefficients) for coefficients, length in options.items()
                    if length >= longest - self.near_critical_slack_ms), reverse=True)[:4]

        collect(order, dag.predecessors, heads)
        collect(reversed(order), dag.successors, tails)
        if abs(max(paths[0][0] for paths in heads.values()) - self.raw_step_ms) > 1e-4:
            raise ValueError('资源依赖没有重建完整参照执行时间')
        cuts = {}
        for middle in order:
            longest = heads[middle][0][0] + tails[middle][0][0] - elapsed[middle]
            if longest < self.raw_step_ms - self.near_critical_slack_ms:
                continue
            for head_time, head in heads[middle]:
                for tail_time, tail in tails[middle]:
                    path_time = head_time + tail_time - elapsed[middle]
                    if path_time < self.raw_step_ms - self.near_critical_slack_ms:
                        continue
                    coefficients = tuple(a + b - c for a, b, c in zip(head, tail, units[middle]))
                    constant = int(round(1000 * self.coefficient_scale * path_time)) - sum(
                        coefficient * cost for coefficient, cost in zip(coefficients, baseline_costs))
                    if coefficients not in cuts or cuts[coefficients] < constant:
                        cuts[coefficients] = constant
        return [(list(coefficients), constant) for coefficients, constant in cuts.items()], len(tasks), resource_edges

    def score(self, counts):
        costs = self.costs(counts)
        return max(constant + sum(a * b for a, b in zip(coefficients, costs))
            for coefficients, constant in self.cuts) / (1000 * self.coefficient_scale)

    def attach(self, model, variables):
        if len(variables) != self.slots * len(self.prefix):
            raise ValueError('路径代理缺少完整区间的前向和反向变量')
        bounds = [self.prefix[phase][-1] for phase in self.prefix for _ in range(self.slots)]
        upper = max(max(0, constant) + sum(a * b for a, b in zip(coefficients, bounds))
            for coefficients, constant in self.cuts)
        objective = model.new_int_var(0, (upper + self.coefficient_scale - 1) // self.coefficient_scale,
            'path_span')
        for coefficients, constant in self.cuts:
            model.add(objective * self.coefficient_scale >= constant + sum(coefficient * variable
                for coefficient, variable in zip(coefficients, variables)))
        return objective

    def summary(self):
        return dict(path_count=len(self.cuts), graph_tasks=self.graph_tasks,
            resource_edges=self.resource_edges, reference_raw_ms=self.raw_step_ms,
            near_critical_slack_ms=self.near_critical_slack_ms,
            variable_cost_dimensions=list(self.prefix),
            coefficient_scale=self.coefficient_scale,
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
            longest_observed_paths=[dict(forward_weights=[value / self.coefficient_scale for value in coefficients[:self.slots]],
                backward_weights=[value / self.coefficient_scale for value in coefficients[self.slots:2*self.slots]],
                parameter_cost_weights=[value / self.coefficient_scale for value in coefficients[2*self.slots:]],
                baseline_path_ms=(constant + sum(a * b for a, b in zip(coefficients, costs))) / (1000 * self.coefficient_scale))
                for coefficients, constant in ordered[:4]])
