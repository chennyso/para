from itertools import accumulate
import math
import time

from ortools.sat.python import cp_model

from graph.process_groups import process_groups
from predictor.profile_database import plan_signature, signature
from search.module_relations import prefix_support
from search.execution_path_model import ExecutionPathModel, partition_cost_prefixes


class CompletePartitionSearch:
    def __init__(self, layers, request, profile, relations, evaluate, deadline, donors, scores,
                 *, reports=None, graph_loader=None, engine='complete', gpu_capacity_per_physical_rank=None,
                 preferred_boundaries=()):
        self.layers, self.request, self.profile = layers, request, profile
        self.relations, self.evaluate, self.deadline = relations, evaluate, deadline
        self.donors, self.scores = donors, scores
        self.reports, self.graph_loader, self.engine = reports or {}, graph_loader, engine
        self.gpu_capacity_per_physical_rank = gpu_capacity_per_physical_rank
        self.preferred_boundaries = set(preferred_boundaries)

    def run(self, limit, expansion_budget):
        pp, slots = self.request['pp'], self.request['pp'] * self.request['vpp']
        report = dict(engine=self.engine, expansions=0, evaluations=0, solver_seconds=0.0,
            candidates=[], statuses=[], center_updates=[], path_templates=[], radius_updates=[],
            memory_scope='full_DAG_separate_from_generation_proxy')
        if slots == 1:
            counts = [self.layers]
            if not prefix_support(counts, self.layers, pp, self.request['vpp'], self.relations):
                raise ValueError('唯一的模块布局违反固定归属')
            report['statuses'].append('FIXED_LAYOUT')
            candidate = dict(self.request, decoder_chunks=counts)
            identity = plan_signature(candidate)
            result = self.reports.get(identity)
            if result is None and min(limit, expansion_budget) > 0 and time.perf_counter() < self.deadline:
                result = self.evaluate(candidate)
                report['evaluations'] = int(result is not None)
            if result is not None:
                report['candidates'].append(dict(decoder_chunks=counts,
                    generation_objective='FIXED_LAYOUT', step_time_ms=result['performance'].get('step_time_ms'),
                    prediction_status=result['prediction_status']))
            return report
        model = cp_model.CpModel()
        if self.profile is None:
            report['statuses'].append('STRUCTURAL_LAYOUT_NEEDS_MODULE_PROFILE')
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
        memory = (self.profile or {}).get('partition_parameter_bytes_by_rank')
        if memory is not None and self.gpu_capacity_per_physical_rank is not None and len(memory) == self.layers:
            groups = process_groups(self.request)
            pp_groups = groups['logical']['pp']
            capacity = self.gpu_capacity_per_physical_rank
            mapping = groups['rank_mapping']
            if set(capacity) != set(mapping) or any(type(value) is not int or value < 1 for value in capacity.values()):
                raise ValueError('分区容量约束需要全部物理Rank的有效容量')
            for stage in range(pp):
                stage_prefixes = {(tuple(accumulate((memory[str(index)][ranks[stage]]
                    for index in range(self.layers)), initial=0)), capacity[mapping[ranks[stage]]]) for ranks in pp_groups}
                for ordinal, (prefix, limit_bytes) in enumerate(sorted(stage_prefixes)):
                    owned = []
                    for slot in range(stage, slots, pp):
                        first = model.new_int_var(0, prefix[-1], f'memory_first{stage}_{ordinal}_{slot}')
                        last = model.new_int_var(0, prefix[-1], f'memory_last{stage}_{ordinal}_{slot}')
                        model.add_element(boundaries[slot], prefix, first)
                        model.add_element(boundaries[slot + 1], prefix, last)
                        owned.append(last - first)
                    model.add(sum(owned) <= limit_bytes)
            report['persistent_parameter_constraint'] = 'DENSE_DDP_PARAMETER_BYTES_PER_LOGICAL_RANK'
        prefixes = {} if self.profile is None else partition_cost_prefixes(self.profile, self.layers)
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
                if phase in ('forward', 'backward'):
                    load_terms[slot % pp].append(work)
        maximum = None
        if prefixes:
            maximum = model.new_int_var(0, sum(prefixes[key][-1] for key in ('forward', 'backward')), 'maximum_work')
            for terms in load_terms:
                model.add(maximum >= sum(terms))
        else:
            largest_chunk = model.new_int_var(1, self.layers, 'largest_module_count')
            for left, right in zip(boundaries, boundaries[1:]):
                model.add(largest_chunk >= right - left)
        compatible = [donor for donor in self.donors if len(donor['decoder_chunks']) == slots
            and prefix_support(donor['decoder_chunks'], self.layers, pp, self.request['vpp'], self.relations)]
        if any(signature(donor) != signature(self.request) for donor in compatible):
            raise ValueError('完整布局中心与当前调查的执行条件不同')
        references = sorted(compatible, key=lambda row: self.scores.get(plan_signature(row), float('inf')))
        anchors, seen = [], set()
        radius_limit = self.layers * slots
        reference_radius = min(radius_limit, max(1, min(self.layers // 16, slots)))

        def load_timed_graph(result):
            if (self.engine == 'path' and self.profile is not None and result is not None
                    and result['prediction_status'] != 'NEEDS_PROFILE'):
                graph = self.graph_loader(result)
                if graph and graph.get('tasks'):
                    return graph
            return None

        def add_anchor(candidate, graph, score, radius):
            values = list(accumulate(candidate['decoder_chunks'], initial=0))
            if any(values == row['boundaries'] for row in anchors):
                return
            guide = None
            features = frozenset()
            if graph is not None:
                guide = ExecutionPathModel(candidate, self.profile, graph)
                path_features = {('path', *(round(8 * value / max(coefficients)) for value in coefficients))
                    for coefficients, _ in guide.cuts if max(coefficients) > 0}
                tasks = graph['tasks']
                waits = {}
                for task in tasks.values():
                    ready = max((tasks[key]['end_ms'] for key in task['dependencies']), default=0)
                    delay = task['start_ms'] - ready
                    if delay > 0:
                        key = (task['kind'], task.get('region'), tuple(sorted(task['resources'])))
                        waits[key] = waits.get(key, 0) + delay
                ordered_waits = sorted(waits, key=lambda key: (-waits[key], repr(key)))[:6]
                features = frozenset(path_features | {('ready_wait', *key) for key in ordered_waits})
            point = dict(boundaries=values, guide=guide, score=score, radius=radius, features=features)
            pool = [*anchors, point]
            by_score = sorted(pool, key=lambda row: (row['score'] is None,
                float('inf') if row['score'] is None else row['score'], row['boundaries']))
            selected = by_score[:1]
            while len(selected) < min(3, len(pool)):
                def novelty(row):
                    return min((len(row['features'] ^ other['features']) /
                        max(1, len(row['features'] | other['features'])) for other in selected), default=0.0)

                remaining = [row for row in by_score if row not in selected]
                selected.append(max(remaining, key=lambda row: (
                    row['guide'] is not None, novelty(row), -by_score.index(row))))
            if point not in selected:
                return
            anchors[:] = selected
            if guide is not None:
                report['path_templates'].append(dict(guide.summary(), decoder_chunks=candidate['decoder_chunks'],
                    distinct_path_features=len(features), selection='best_replayed_or_distinct_path_features'))

        for index, donor in enumerate(references):
            donor_id = plan_signature(donor)
            values = tuple(accumulate(donor['decoder_chunks'], initial=0))
            if values not in seen and donor_id in self.reports:
                model.add_forbidden_assignments(boundaries, [values])
                seen.add(values)
            if index < max(3, min(8, limit + 2)):
                graph = load_timed_graph(self.reports.get(donor_id))
                add_anchor(donor, graph, self.scores.get(donor_id),
                    reference_radius if graph is not None else radius_limit)
        if not anchors:
            add_anchor(self.request, None, None, radius_limit)
        for ordinal in range(min(limit, expansion_budget)):
            remaining = self.deadline - time.perf_counter()
            if remaining <= 0:
                break
            trial_model = model.clone()
            anchor = anchors[ordinal % len(anchors)]
            center, guide, radius = (anchor[key] for key in ('boundaries', 'guide', 'radius'))
            displacement = []
            preferred_displacement = []
            for index in range(1, slots):
                change = trial_model.new_int_var(0, self.layers, f'current_edit{ordinal}_{index}')
                trial_model.add_abs_equality(change, boundaries[index] - center[index])
                displacement.append(change)
                if index in self.preferred_boundaries:
                    preferred_displacement.append(change)
            trial_model.add(sum(displacement) <= radius)
            if preferred_displacement and ordinal % 2 == 0:
                trial_model.add(sum(preferred_displacement) >= 1)
            if slots > 2 and ordinal % 2 and tuple(center) in seen:
                trial_model.add(sum(displacement) >= 2)
            for index in range(1, slots):
                trial_model.add_hint(boundaries[index], center[index])
            if guide is not None:
                path_objective = guide.attach(trial_model, region_variables)
                microbatches = self.request['global_batch'] // (self.request['dp'] * self.request['micro_batch'])
                objective = trial_model.new_int_var(0, 10**12, f'combined_path_span_{ordinal}')
                trial_model.add(objective >= path_objective)
                stage_overhead = (self.profile or {}).get('boundary_ms', ())
                for stage, work in enumerate(load_terms):
                    overhead = round(1000 * sum(stage_overhead[stage].values())) if stage_overhead else 0
                    trial_model.add(objective >= (sum(work) + overhead) * microbatches)
            else:
                objective = maximum if maximum is not None else largest_chunk
            trial_model.minimize(objective * (self.layers * slots + 1) + sum(displacement))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            if math.isfinite(remaining):
                solver.parameters.max_time_in_seconds = min(1.5, remaining)
            started = time.perf_counter()
            status = solver.solve(trial_model)
            report['solver_seconds'] += time.perf_counter() - started
            report['statuses'].append(solver.status_name(status))
            if status == cp_model.MODEL_INVALID:
                raise ValueError(trial_model.validate())
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                if status == cp_model.INFEASIBLE:
                    anchor['radius'] = min(radius_limit, max(reference_radius, radius * 2))
                    continue
                break
            values = [solver.value(value) for value in boundaries]
            counts = [right - left for left, right in zip(values, values[1:])]
            if not prefix_support(counts, self.layers, pp, self.request['vpp'], self.relations):
                raise ValueError('完整布局违反固定模块归属')
            model.add_forbidden_assignments(boundaries, [values])
            seen.add(tuple(values))
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
                preferred_boundaries=sorted(self.preferred_boundaries),
                proxy_work_microseconds=None if maximum is None else solver.value(maximum),
                generation_objective_ms=None if maximum is None else solver.value(objective) / 1000,
                generation_objective=('RESOURCE_PATH' if guide is not None
                    else 'MODULE_WORK' if maximum is not None else 'STRUCTURAL_MODULE_COUNT'),
                step_time_ms=result['performance'].get('step_time_ms'),
                prediction_status=result['prediction_status']))
            score = result['performance'].get('step_time_ms')
            if (score is not None and result['prediction_status'] != 'NEEDS_PROFILE'
                    and result.get('memory', {}).get('admission') != 'REJECT_OOM'):
                previous = anchor['score']
                graph = load_timed_graph(result)
                if guide is not None and graph is not None:
                    expected_delta = guide.score(counts) - guide.raw_step_ms
                    actual_delta = max(task['end_ms'] for task in graph['tasks'].values()) - guide.raw_step_ms
                    consistent = (expected_delta * actual_delta > 0
                        and abs(expected_delta - actual_delta) <= max(0.05, abs(actual_delta) * 0.5))
                    anchor['radius'] = (min(radius_limit, radius * 2) if consistent and actual_delta < 0
                        else max(1, radius // 2))
                    report['radius_updates'].append(dict(previous=radius, current=anchor['radius'],
                        proxy_delta_ms=expected_delta, replay_delta_ms=actual_delta))
                before = [row['boundaries'] for row in anchors]
                add_anchor(candidate, graph, score, anchor['radius'])
                if before != [row['boundaries'] for row in anchors]:
                    report['center_updates'].append(dict(previous_ms=previous, new_ms=score, decoder_chunks=counts))
        report['candidates'].sort(key=lambda row: (row['step_time_ms'] is None,
            float('inf') if row['step_time_ms'] is None else row['step_time_ms'], row['decoder_chunks']))
        return report
