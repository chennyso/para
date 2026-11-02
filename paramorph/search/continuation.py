from copy import deepcopy
from itertools import accumulate
import math
import random
import time

from predictor.profile_database import plan_signature, signature


class ContinuationGuide:
    def __init__(self, requests, evaluate, max_evaluations, admission_evaluator=None, *,
                 prefix_guidance=True, donor_coverage=True, completion_filter=None, deadline=None,
                 selection_policy='round_robin', donor_limit=None, selection_seed=0, donor_scores=None):
        if type(max_evaluations) is not int or max_evaluations < 0:
            raise ValueError('完整续接的评价预算必须为非负整数')
        if requests and evaluate is None:
            raise ValueError('完整续接需要实际执行评价器')
        self.requests = []
        seen = set()
        for request in requests:
            counts = request['decoder_chunks']
            if (len(counts) != request['pp'] * request['vpp']
                    or any(type(value) is not int or value < 1 for value in counts)):
                raise ValueError('完整续接方案需要非空原生模块区间')
            identity = plan_signature(request)
            if identity not in seen:
                seen.add(identity)
                self.requests.append(deepcopy(request))
        self.evaluate = evaluate
        if completion_filter is not None and not callable(completion_filter):
            raise ValueError('接续范围检查必须为可调用函数')
        self.completion_filter = completion_filter
        self.deadline = deadline
        if selection_policy not in ('round_robin', 'coverage', 'anchored_coverage', 'direct_time', 'random'):
            raise ValueError('完整接续来源选择策略不在支持范围内')
        if donor_limit is not None and (type(donor_limit) is not int or donor_limit < 1):
            raise ValueError('完整接续来源数量上限必须为正整数')
        self.selection_policy = selection_policy
        self.donor_limit = donor_limit
        self.selection_seed = selection_seed
        self.donor_scores = {} if donor_scores is None else dict(donor_scores)
        identifiers = {plan_signature(row) for row in self.requests}
        if not set(self.donor_scores) <= identifiers or any(not math.isfinite(value) or value <= 0
                for value in self.donor_scores.values()):
            raise ValueError('接续来源的直接成本必须对应已登记来源及有限正数')
        self.donor_selection_rounds = []
        self.filtered_completions = set()
        if type(prefix_guidance) is not bool or type(donor_coverage) is not bool:
            raise ValueError('接续的前缀引导与来源覆盖开关必须为布尔值')
        self.prefix_guidance = prefix_guidance
        self.donor_coverage = donor_coverage
        self.admission_evaluator = admission_evaluator
        self.admissions = {}
        self.admission_seconds = 0.0
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.evaluation_calls = 0
        self.cache_hits = 0
        self.cache = {}
        self.evaluated_requests = {}
        self.decisions = []
        self.frontiers = []
        self.competitions = []
        self.depth = None
        self.depth_limit = max_evaluations
        self.first_evaluation_depth = None
        self.retention_snapshots = []
        self.reference_seeds = []

    def begin_depth(self, depth, slots):
        self.depth = depth
        remaining = self.max_evaluations - self.evaluations
        self.depth_limit = self.evaluations + remaining // (slots - depth + 1)

    def compatible(self, request):
        identity = signature(request)
        return [row for row in self.requests if signature(row) == identity
                and sum(row['decoder_chunks']) == sum(request['decoder_chunks'])]

    def order_ends(self, request, counts, ends):
        targets = []
        for row in self.compatible(request):
            target = sum(row['decoder_chunks'][:len(counts) + 1])
            if target in ends and target not in targets:
                targets.append(target)
        return targets + [end for end in ends if end not in targets]

    def prefix_distance(self, request, counts):
        prefix = tuple(accumulate(counts))
        return min((sum(abs(left - right) for left, right in zip(
            prefix, accumulate(row['decoder_chunks']))) for row in self.compatible(request)), default=0)

    @staticmethod
    def complete(request, counts, continuation):
        slots = request['pp'] * request['vpp']
        if not counts or len(counts) > slots:
            raise ValueError('部分分区的区间数量无效')
        ends = list(accumulate(continuation['decoder_chunks'], initial=0))
        prefix = list(accumulate(counts, initial=0))
        boundaries = prefix + ends[len(counts) + 1:]
        if len(boundaries) != slots + 1 or boundaries[-1] != sum(request['decoder_chunks']):
            return None
        completed = [right - left for left, right in zip(boundaries, boundaries[1:])]
        if any(value <= 0 for value in completed):
            return None
        result = dict(request, decoder_chunks=completed)
        result['run_id'] = 'continuation-' + plan_signature(result)[:16]
        return result

    def prepare_frontier(self, states, request, local_score):
        continuations = self.select_donors(states, request)
        if not states or not continuations:
            return
        def priority(state):
            return self.prefix_distance(request, state.counts), local_score(state)

        ordered = sorted(states, key=priority)
        if self.depth is not None and any(len(state.counts) != self.depth for state in ordered):
            raise ValueError('完整续接的评价前沿必须属于当前分区深度')
        before = self.evaluations
        evaluated = []
        # 每轮先调查不同前缀；轮换后缀起点，避免首项布局独占有限配额。
        for round_index in range(len(continuations)):
            for state_index, state in enumerate(ordered):
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    self.max_evaluations = self.evaluations
                    self.depth_limit = self.evaluations
                    return
                continuation = continuations[(state_index + round_index) % len(continuations)]
                completed = self.complete(request, state.counts, continuation)
                if completed is None:
                    continue
                if not self.allows(completed):
                    continue
                identity = plan_signature(completed)
                if identity in self.cache:
                    continue
                if self.evaluations >= self.depth_limit:
                    break
                record = self.evaluate_completion(completed)
                if record is None:
                    self.max_evaluations = self.evaluations
                    self.depth_limit = self.evaluations
                    break
                evaluated.append(dict(prefix=list(state.counts), plan_signature=identity,
                    continuation_id=plan_signature(continuation)))
            if self.evaluations >= self.depth_limit:
                break
        self.frontiers.append(dict(depth=self.depth, state_count=len(ordered),
            new_evaluations=self.evaluations - before, evaluated=evaluated))
        if self.evaluations > before and self.first_evaluation_depth is None:
            self.first_evaluation_depth = self.depth

    def evaluate_completion(self, completed):
        identity = plan_signature(completed)
        if identity in self.cache:
            return self.cache[identity]
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            return None
        if self.admission_evaluator is not None and identity not in self.admissions:
            started = time.perf_counter()
            admission = self.admission_evaluator(completed)
            self.admission_seconds += time.perf_counter() - started
            if not isinstance(admission, dict) or admission.get('plan_signature', identity) != identity:
                raise ValueError('接续容量检查需要对应当前完整候选')
            self.admissions[identity] = deepcopy(admission)
        admission = self.admissions.get(identity, {})
        memory = admission.get('memory', admission)
        if memory.get('admission') == 'REJECT_OOM' and memory.get('status') == 'REJECTED_MEMORY_LOWER_BOUND':
            record = dict(plan_signature=identity, decoder_chunks=completed['decoder_chunks'],
                prediction_id=admission.get('prediction_id'), prediction_status='REJECTED_MEMORY_LOWER_BOUND',
                step_time_ms=None, memory_admission='REJECT_OOM', profiling_requests=[],
                admission_evidence=deepcopy(admission))
            self.cache[identity] = record
            return record
        report = self.evaluate(completed)
        if report is None:
            return None
        self.evaluation_calls += 1
        newly_evaluated = report.get('planning_prediction_new', True)
        if type(newly_evaluated) is not bool:
            raise ValueError('接续评价需要明确新增预测状态')
        self.evaluations += int(newly_evaluated)
        self.cache_hits += int(not newly_evaluated)
        return self.record_completed_report(completed, report)

    def record_completed_report(self, completed, report):
        identity = plan_signature(completed)
        self.evaluated_requests[identity] = completed
        if report.get('plan_signature', identity) != identity:
            raise ValueError('完整接续评价返回了不同配置的结果')
        value = report.get('performance', {}).get('step_time_ms')
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('完整接续评价需要有限正数迭代时间')
        admission = report.get('memory', {}).get('admission', 'UNKNOWN')
        usable = value is not None and admission != 'REJECT_OOM' and report.get('prediction_status') != 'NEEDS_PROFILE'
        record = dict(plan_signature=identity, decoder_chunks=completed['decoder_chunks'],
            prediction_id=report.get('prediction_id'), prediction_status=report.get('prediction_status', 'UNKNOWN'),
            step_time_ms=value if usable else None, memory_admission=admission,
            profiling_requests=deepcopy(report.get('evidence', {}).get('profiling_requests', [])))
        self.cache[identity] = record
        return record

    def prime_reference(self, request, report):
        if not self.allows(request):
            raise ValueError('完整参照违反当前调查的固定模块归属')
        record = self.record_completed_report(request, report)
        self.reference_seeds.append(dict(plan_signature=plan_signature(request),
            prediction_id=record['prediction_id'], step_time_ms=record['step_time_ms']))

    def select_donors(self, states, request):
        donors = self.compatible(request)
        if self.selection_policy == 'round_robin':
            return donors if self.donor_limit is None else donors[:self.donor_limit]
        started = time.perf_counter()
        coverage = {}
        supports = {}
        for donor in donors:
            identifier = plan_signature(donor)
            coverage[identifier] = set()
            supports[identifier] = set()
            for state in states:
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    return []
                completed = self.complete(request, state.counts, donor)
                if completed is None or not self.allows(completed):
                    continue
                key = plan_signature(completed)
                if key not in self.cache:
                    coverage[identifier].add(key)
                    supports[identifier].add(state.counts)
        remaining = {plan_signature(row): row for row in donors}
        limit = len(donors) if self.donor_limit is None else min(self.donor_limit, len(donors))
        selected, covered, supported = [], set(), set()
        own_time = lambda key: (self.cache.get(key, {}).get('step_time_ms')
            or self.donor_scores.get(key, float('inf')))
        if self.selection_policy == 'anchored_coverage' and remaining and limit:
            anchor = min(remaining, key=lambda key: (own_time(key), key))
            if math.isfinite(own_time(anchor)):
                selected.append(remaining.pop(anchor))
                covered.update(coverage[anchor])
                supported.update(supports[anchor])
        if self.selection_policy == 'random':
            order = sorted(remaining)
            random.Random(self.selection_seed + (self.depth or 0)).shuffle(order)
        while remaining and len(selected) < limit:
            if self.selection_policy in ('coverage', 'anchored_coverage'):
                key = min(remaining, key=lambda item: (-len(coverage[item] - covered),
                    -len(supports[item] - supported), own_time(item), item))
                if not coverage[key] - covered:
                    break
            elif self.selection_policy == 'direct_time':
                key = min(remaining, key=lambda item: (own_time(item), item))
            else:
                key = next(item for item in order if item in remaining)
            selected.append(remaining.pop(key))
            covered.update(coverage[key])
            supported.update(supports[key])
        all_prefixes = {state.counts for state in states}
        equivalent = {}
        for key, value in coverage.items():
            equivalent.setdefault(frozenset(value), []).append(key)
        self.donor_selection_rounds.append(dict(depth=self.depth, policy=self.selection_policy,
            candidate_donors=len(donors), selected_donors=[plan_signature(row) for row in selected],
            selected_layouts=[row['decoder_chunks'] for row in selected], prefix_count=len(all_prefixes),
            unique_new_completions=len(covered), supported_prefix_count=len(supported),
            uncovered_prefix_count=len(all_prefixes - supported),
            uncovered_prefixes=[list(row) for row in sorted(all_prefixes - supported)[:8]],
            per_donor_new_completions={key: len(value) for key, value in coverage.items()},
            structurally_distinct_donor_sets=len(equivalent),
            redundant_donor_count=len(donors) - len(equivalent),
            equivalent_donor_groups=[sorted(value) for value in equivalent.values() if len(value) > 1][:8],
            selection_seconds=time.perf_counter() - started,
            objective='unique_structurally_compatible_unqueried_completions'))
        return selected

    def prepare_competitions(self, buckets, request, width, local_score):
        if type(width) is not int or width < 1:
            raise ValueError('完整接续的保留宽度必须为正整数')
        competing = [bucket for bucket in buckets if len(bucket) > width]
        before = self.evaluations
        self.prepare_frontier([state for bucket in competing for state in bucket], request, local_score)
        self.competitions.append(dict(depth=self.depth, boundary_groups=len(buckets),
            competing_groups=len(competing), competing_states=sum(len(bucket) for bucket in competing),
            skipped_noncompetitive_states=sum(len(bucket) for bucket in buckets if len(bucket) <= width),
            retention_width=width, new_evaluations=self.evaluations - before))

    def witnesses(self, request, counts):
        rows = []
        for continuation in self.compatible(request):
            completed = self.complete(request, counts, continuation)
            if completed is None:
                continue
            if not self.allows(completed):
                continue
            identity = plan_signature(completed)
            row = self.cache.get(identity)
            if row is None:
                continue
            if row['step_time_ms'] is not None:
                rows.append(dict(row, continuation_id=plan_signature(continuation)))
        return sorted(rows, key=lambda row: (row['step_time_ms'], row['continuation_id']))

    def witness(self, request, counts):
        rows = self.witnesses(request, counts)
        return rows[0] if rows else None

    def score(self, state, request, local_score):
        witness = self.witness(request, state.counts) if state.counts else None
        return ((0, witness['step_time_ms'], 0, local_score(state)) if witness is not None
                else (1, 0.0, self.prefix_distance(request, state.counts)
                    if self.prefix_guidance else 0, local_score(state)))

    def retain(self, states, request, width, local_score):
        local = sorted(states, key=local_score)
        evaluated = sorted(local, key=lambda state: self.score(state, request, local_score))
        winners = {}
        for state in local if self.donor_coverage else ():
            for row in self.witnesses(request, state.counts):
                donor = row['continuation_id']
                priority = (row['step_time_ms'], local_score(state))
                if donor not in winners or priority < winners[donor][0]:
                    winners[donor] = (priority, state)
        represented = {state.counts: state for priority, state in sorted(winners.values(), key=lambda row: row[0])}
        selected = list(represented.values())[:width]
        selected.extend(state for state in evaluated if state not in selected)
        selected = selected[:width]
        # 未取得完整续接成本的部分状态仍保留一个局部搜索位置。
        unknown = [state for state in local if self.witness(request, state.counts) is None]
        if width > 1 and unknown and unknown[0] not in selected:
            selected = selected[:width - 1] + [unknown[0]]
        if [state.counts for state in selected] != [state.counts for state in local[:width]]:
            self.decisions.append(dict(prefix_length=len(local[0].counts),
                local_counts=[list(state.counts) for state in local[:width]],
                retained_counts=[list(state.counts) for state in selected],
                donor_winners={donor: list(row[1].counts) for donor, row in winners.items()},
                local_witnesses=[self.witness(request, state.counts) for state in local[:width]],
                witnesses=[self.witness(request, state.counts) for state in selected]))
        self.retention_snapshots.append(dict(depth=self.depth, end=local[0].end,
            retained=[list(state.counts) for state in selected], discarded_count=max(0, len(local) - width),
            discarded_before_first_new_DAG=(max(0, len(local) - width)
                if self.first_evaluation_depth is None else 0)))
        return selected

    def report(self):
        return dict(evaluations=self.evaluations, max_evaluations=self.max_evaluations,
            evaluation_calls=self.evaluation_calls, cache_hits=self.cache_hits,
            first_evaluation_depth=self.first_evaluation_depth, retention_snapshots=self.retention_snapshots,
            reference_seeds=self.reference_seeds,
            continuation_ids=[plan_signature(row) for row in self.requests],
            completed_candidates=list(self.cache.values()),
            evaluated_frontiers=self.frontiers, retention_changes=self.decisions,
            competition_frontiers=self.competitions,
            prefix_guidance=self.prefix_guidance, donor_coverage=self.donor_coverage,
            admission_checks=len(self.admissions), admission_seconds=self.admission_seconds,
            memory_rejections=[row for row in self.cache.values()
                if row['prediction_status'] == 'REJECTED_MEMORY_LOWER_BOUND'],
            evaluation_policy='shared_complete_prefix_priority_and_donor_coverage', ranking_calls_evaluator=False,
            lower_bound_claimed=False, scoped_completion_filter=self.completion_filter is not None,
            filtered_completion_count=len(self.filtered_completions),
            donor_selection_rounds=self.donor_selection_rounds)

    def allows(self, request):
        allowed = True if self.completion_filter is None else self.completion_filter(request)
        if type(allowed) is not bool:
            raise ValueError('接续范围检查需要返回明确布尔值')
        if not allowed:
            self.filtered_completions.add(plan_signature(request))
        return allowed

    def completed(self, request):
        identity = signature(request)
        rows = [row for key, row in self.evaluated_requests.items()
                if signature(row) == identity and self.cache[key]['step_time_ms'] is not None
                and self.allows(row)]
        return sorted(rows, key=lambda row: self.cache[plan_signature(row)]['step_time_ms'])
