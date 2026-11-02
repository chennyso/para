from itertools import accumulate
import time

from ortools.sat.python import cp_model

from predictor.profile_database import plan_signature, signature
from search.module_relations import prefix_support
from search.execution_path_model import ExecutionPathModel


class CompletePartitionSearch:
    def __init__(self, layers, request, profile, relations, evaluate, deadline, donors, scores,
                 *, reports=None, graph_loader=None, engine='complete'):
        self.layers, self.request, self.profile = layers, request, profile
        self.relations, self.evaluate, self.deadline = relations, evaluate, deadline
        self.donors, self.scores = donors, scores
        self.reports, self.graph_loader, self.engine = reports or {}, graph_loader, engine

    def run(self, limit, expansion_budget):
        pp, slots = self.request['pp'], self.request['pp'] * self.request['vpp']
        report = dict(engine=self.engine, expansions=0, evaluations=0, solver_seconds=0.0,
            candidates=[], statuses=[], center_updates=[], path_templates=[],
            memory_scope='full_DAG_separate_from_generation_proxy')
        if self.profile is None:
            report['statuses'].append('NEEDS_MODULE_PROFILE')
            return report
        model = cp_model.CpModel()
        boundaries = [model.new_int_var(index, self.layers - slots + index, f'b{index}')
                      for index in range(slots + 1)]
        model.add(boundaries[0] == 0)
        model.add(boundaries[-1] == self.layers)
        for left, right in zip(boundaries, boundaries[1:]):
            model.add(right > left)
        for relation in self.relations:
            if relation.relation != 'allowed_virtual' or len(relation.virtual_slots) != 1:
                raise ValueError('完整布局构造需要明确的固定模块归属')
            slot = relation.virtual_slots[0]
            for module in relation.module_ids:
                model.add(boundaries[slot] <= module)
                model.add(boundaries[slot + 1] > module)
        prefixes = {phase: list(accumulate((int(round(1000 * self.profile['layers'][str(index)][phase + '_ms']))
            for index in range(self.layers)), initial=0)) for phase in ('forward', 'backward')}
        load_terms = [[] for _ in range(pp)]
        region_variables = []
        for phase, prefix in prefixes.items():
            if any(right < left for left, right in zip(prefix, prefix[1:])):
                raise ValueError('完整布局需要非负的真实模块成本')
            for slot in range(slots):
                first = model.new_int_var(0, prefix[-1], f'{phase}_first{slot}')
                last = model.new_int_var(0, prefix[-1], f'{phase}_last{slot}')
                model.add_element(boundaries[slot], prefix, first)
                model.add_element(boundaries[slot + 1], prefix, last)
                work = model.new_int_var(0, prefix[-1], f'{phase}_work{slot}')
                model.add(work == last - first)
                region_variables.append(work)
                load_terms[slot % pp].append(work)
        maximum = model.new_int_var(0, sum(prefix[-1] for prefix in prefixes.values()), 'maximum_work')
        for terms in load_terms:
            model.add(maximum >= sum(terms))
        compatible = [donor for donor in self.donors if len(donor['decoder_chunks']) == slots
            and prefix_support(donor['decoder_chunks'], self.layers, pp, self.request['vpp'], self.relations)]
        reference = min(compatible, key=lambda row: self.scores.get(plan_signature(row), float('inf')))
        if signature(reference) != signature(self.request):
            raise ValueError('完整布局中心与当前调查的执行条件不同')
        reference_boundaries = list(accumulate(reference['decoder_chunks'], initial=0))
        for donor in compatible:
            model.add_forbidden_assignments(boundaries, [list(accumulate(donor['decoder_chunks'], initial=0))])
        best_score = self.scores.get(plan_signature(reference), float('inf'))
        center = reference_boundaries
        guide = None
        if self.engine == 'path':
            prediction = self.reports.get(plan_signature(reference))
            if prediction is None or prediction['prediction_status'] == 'NEEDS_PROFILE':
                report['statuses'].append('NEEDS_PATH_REFERENCE')
                return report
            guide = ExecutionPathModel(reference, self.profile, self.graph_loader(prediction))
            report['path_templates'].append(dict(guide.summary(), decoder_chunks=reference['decoder_chunks']))
        for ordinal in range(min(limit, expansion_budget)):
            remaining = self.deadline - time.perf_counter()
            if remaining <= 0:
                break
            trial_model = model.clone()
            radius = self.layers * slots if ordinal == 0 else min(self.layers * slots, 2 ** ordinal)
            displacement = []
            for index in range(1, slots):
                change = trial_model.new_int_var(0, self.layers, f'current_edit{ordinal}_{index}')
                trial_model.add_abs_equality(change, boundaries[index] - center[index])
                displacement.append(change)
            trial_model.add(sum(displacement) <= radius)
            for index in range(1, slots):
                trial_model.add_hint(boundaries[index], center[index])
            objective = maximum if guide is None else guide.attach(trial_model, region_variables)
            trial_model.minimize(objective * (self.layers * slots + 1) + sum(displacement))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_time_in_seconds = min(1.5, remaining)
            started = time.perf_counter()
            status = solver.solve(trial_model)
            report['solver_seconds'] += time.perf_counter() - started
            report['statuses'].append(solver.status_name(status))
            if status == cp_model.MODEL_INVALID:
                raise ValueError(trial_model.validate())
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                if status == cp_model.INFEASIBLE:
                    continue
                break
            values = [solver.value(value) for value in boundaries]
            counts = [right - left for left, right in zip(values, values[1:])]
            if not prefix_support(counts, self.layers, pp, self.request['vpp'], self.relations):
                raise ValueError('完整布局违反固定模块归属')
            model.add_forbidden_assignments(boundaries, [values])
            report['expansions'] += 1
            if time.perf_counter() >= self.deadline:
                break
            candidate = dict(self.request, decoder_chunks=counts)
            result = self.evaluate(candidate)
            if result is None:
                break
            report['evaluations'] += 1
            report['candidates'].append(dict(decoder_chunks=counts,
                boundary_displacement_budget=radius,
                proxy_work_microseconds=solver.value(maximum),
                generation_objective_ms=solver.value(objective) / 1000,
                step_time_ms=result['performance'].get('step_time_ms'),
                prediction_status=result['prediction_status']))
            score = result['performance'].get('step_time_ms')
            if (score is not None and result['prediction_status'] != 'NEEDS_PROFILE'
                    and result.get('memory', {}).get('admission') != 'REJECT_OOM' and score < best_score):
                report['center_updates'].append(dict(previous_ms=best_score if best_score != float('inf') else None,
                    new_ms=score, decoder_chunks=counts))
                best_score, center = score, values
                if self.engine == 'path':
                    guide = ExecutionPathModel(candidate, self.profile, self.graph_loader(result))
                    report['path_templates'].append(dict(guide.summary(), decoder_chunks=counts))
        return report
