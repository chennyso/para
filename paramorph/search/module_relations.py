from itertools import accumulate

from contracts import ModuleRelation
from ortools.sat.python import cp_model


def validate_relations(relations, layers, slots):
    rows = tuple(ModuleRelation.model_validate(row) for row in relations)
    for row in rows:
        if max(row.module_ids) >= layers or any(slot >= slots for slot in row.virtual_slots):
            raise ValueError('模块关系超出当前模型或虚拟区间数量')
    return rows


def owner_patterns(relations, layers, pp, vpp, limit, time_limit_s=5.0):
    if not relations:
        return [()], True
    modules = sorted({module for row in relations for module in row.module_ids})
    slots = pp * vpp
    model = cp_model.CpModel()
    owners = {module: model.new_int_var(max(0, slots - layers + module),
        min(slots - 1, module), f'owner_{module}') for module in modules}
    physical = {module: model.new_int_var(0, pp - 1, f'physical_{module}') for module in modules}
    for module in modules:
        model.add_modulo_equality(physical[module], owners[module], pp)
    for left, right in zip(modules, modules[1:]):
        model.add(owners[left] <= owners[right])
        model.add(owners[right] - owners[left] <= right - left)
    for row in relations:
        if row.relation == 'allowed_virtual':
            for module in row.module_ids:
                model.add_allowed_assignments([owners[module]], [(slot,) for slot in row.virtual_slots])
        else:
            left, right = row.module_ids
            if row.relation == 'same_virtual':
                model.add(owners[left] == owners[right])
            elif row.relation == 'same_physical':
                model.add(physical[left] == physical[right])
            else:
                model.add(physical[left] != physical[right])

    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.patterns = []

        def on_solution_callback(self):
            self.patterns.append(tuple((module, self.value(owners[module])) for module in modules))
            if len(self.patterns) >= limit:
                self.stop_search()

    collector = Collector()
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    if time_limit_s <= 0:
        raise ValueError('模块归属求解时间必须为正数')
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(model, collector)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.INFEASIBLE, cp_model.UNKNOWN):
        raise ValueError('模块归属组合求解没有取得可用的完成状态')
    return sorted(set(collector.patterns)), status in (cp_model.OPTIMAL, cp_model.INFEASIBLE)


def prefix_support(counts, layers, pp, vpp, relations):
    slots = pp * vpp
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError('分区前缀必须包含正整数模块数量')
    if len(counts) > slots or sum(counts) + slots - len(counts) > layers:
        return False
    boundaries = tuple(accumulate((0, *counts)))
    end = boundaries[-1]
    relevant = sorted({module for row in relations for module in row.module_ids})
    domains = {}
    for module in relevant:
        if module < end:
            domains[module] = {next(slot for slot in range(len(counts))
                                    if boundaries[slot] <= module < boundaries[slot + 1])}
        else:
            lower = max(len(counts), slots - (layers - module))
            upper = min(slots - 1, len(counts) + module - end)
            domains[module] = set(range(lower, upper + 1))
        if not domains[module]:
            return False
    changed = True
    while changed:
        changed = False
        pairs = []
        for left, right in zip(relevant, relevant[1:]):
            pairs.append((left, right, 'ordered'))
        for row in relations:
            if row.relation == 'allowed_virtual':
                for module in row.module_ids:
                    reduced = domains[module].intersection(row.virtual_slots)
                    changed |= reduced != domains[module]
                    domains[module] = reduced
            else:
                pairs.append((*row.module_ids, row.relation))
        for left, right, relation in pairs:
            def compatible(a, b):
                if relation == 'ordered':
                    return a <= b
                if relation == 'same_virtual':
                    return a == b
                if relation == 'same_physical':
                    return a % pp == b % pp
                return a % pp != b % pp
            support = {(a, b) for a in domains[left] for b in domains[right] if compatible(a, b)}
            first = {a for a, _ in support}
            second = {b for _, b in support}
            changed |= first != domains[left] or second != domains[right]
            domains[left], domains[right] = first, second
        if any(not values for values in domains.values()):
            return False
    return True
