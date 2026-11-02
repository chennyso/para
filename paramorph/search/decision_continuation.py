from itertools import accumulate
import math
import time

from predictor.profile_database import plan_signature, signature
from search.continuation import ContinuationGuide


class DecisionContinuationGuide(ContinuationGuide):
    def __init__(self, *args, timing='budgeted_paired', early_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if timing not in ('fixed_depth', 'early_independent', 'paired', 'budgeted_paired', 'terminal_paired', 'revisit_paired'):
            raise ValueError('前瞻评价时机不在支持范围内')
        if not math.isfinite(early_fraction) or not 0 <= early_fraction <= 1:
            raise ValueError('前瞻评价比例必须位于零与一之间')
        self.timing = timing
        self.early_budget = (self.max_evaluations if timing == 'paired' else
            min(self.max_evaluations, max(0, int(self.max_evaluations * early_fraction) // 2 * 2)))
        self.started = time.perf_counter()
        self.pairs = []
        self.retention_snapshots = []
        self.pair_preferences = {}
        self.first_evaluation_depth = None

    def begin_depth(self, depth, slots):
        self.depth = depth
        self.slots = slots
        self.depth_limit = self.early_budget if depth < slots else self.max_evaluations
        if self.timing == 'fixed_depth' and depth != max(2, slots // 4) and depth < slots:
            self.depth_limit = self.evaluations

    def prepare_competitions(self, buckets, request, width, local_score):
        if self.depth < self.slots and (self.timing == 'terminal_paired'
                or self.timing == 'revisit_paired' and self.depth <= request['pp']):
            self.depth_limit = self.evaluations
        competing = [sorted(bucket, key=local_score) for bucket in buckets if len(bucket) > width]
        before = self.evaluations
        if self.timing == 'early_independent':
            self.prepare_frontier([state for bucket in competing for state in bucket], request, local_score)
        else:
            # 优先调查保留边界两侧局部代价接近的前缀。
            competing.sort(key=lambda bucket: (abs(local_score(bucket[width])[0] - local_score(bucket[width - 1])[0]),
                self.prefix_distance(request, bucket[width].counts), bucket[width].counts))
            donors = self.compatible(request)
            for bucket in competing:
                keep, drop = bucket[width - 1], bucket[width]
                for donor in donors:
                    if self.evaluations + 2 > self.depth_limit:
                        break
                    completed = [self.complete(request, state.counts, donor) for state in (keep, drop)]
                    if any(row is None or not self.allows(row) for row in completed):
                        continue
                    if any(signature(row) != signature(request) for row in completed):
                        raise ValueError('配对前瞻的完整执行条件发生变化')
                    ids = [plan_signature(row) for row in completed]
                    key = (self.depth, keep.counts, drop.counts, plan_signature(donor))
                    if any(row['key'] == key for row in self.pairs):
                        continue
                    reports = [self.evaluate_completion(row) for row in completed]
                    usable = all(row is not None and row['step_time_ms'] is not None for row in reports)
                    reversal = usable and reports[1]['step_time_ms'] < reports[0]['step_time_ms']
                    record = dict(key=key, depth=self.depth, elapsed_seconds=time.perf_counter() - self.started,
                        execution_signature=signature(request), donor_id=plan_signature(donor),
                        donor_boundaries=list(accumulate(donor['decoder_chunks'], initial=0)),
                        keep_prefix=list(keep.counts), drop_prefix=list(drop.counts),
                        completed_ids=ids, reports=reports, status='COMPARABLE' if usable else 'UNRESOLVED',
                        rank_reversal=bool(reversal), retention_changed=False)
                    self.pairs.append(record)
                    if usable:
                        self.pair_preferences[self.depth, keep.counts, drop.counts] = record
                    break
        if self.evaluations > before and self.first_evaluation_depth is None:
            self.first_evaluation_depth = self.depth
        self.competitions.append(dict(depth=self.depth, boundary_groups=len(buckets),
            competing_groups=len(competing), competing_states=sum(len(bucket) for bucket in competing),
            retention_width=width, new_evaluations=self.evaluations - before))

    def retain(self, states, request, width, local_score):
        local = sorted(states, key=local_score)
        if self.timing == 'early_independent':
            selected = super().retain(states, request, width, local_score)
        else:
            selected = list(local[:width])
            if len(local) > width:
                keep, drop = local[width - 1], local[width]
                record = self.pair_preferences.get((self.depth, keep.counts, drop.counts))
                if record is not None and record['rank_reversal']:
                    selected[-1] = drop
                    record['retention_changed'] = True
                    self.decisions.append(dict(prefix_length=self.depth,
                        local_counts=[list(state.counts) for state in local[:width]],
                        retained_counts=[list(state.counts) for state in selected],
                        paired_evidence=record))
        if self.timing != 'early_independent':
            self.retention_snapshots.append(dict(depth=self.depth, end=local[0].end,
                retained=[list(state.counts) for state in selected], discarded_count=max(0, len(local) - width),
                discarded_before_first_new_DAG=(max(0, len(local) - width)
                    if self.first_evaluation_depth is None else 0)))
        return selected

    def report(self):
        return dict(super().report(), evaluation_policy=self.timing, early_budget=self.early_budget,
            first_evaluation_depth=self.first_evaluation_depth, paired_comparisons=self.pairs,
            retention_snapshots=self.retention_snapshots,
            comparable_pairs=sum(row['status'] == 'COMPARABLE' for row in self.pairs),
            rank_reversals=sum(row['rank_reversal'] for row in self.pairs),
            changed_pair_decisions=sum(row['retention_changed'] for row in self.pairs),
            pruned_before_first_new_DAG=sum(row['discarded_before_first_new_DAG'] for row in self.retention_snapshots))
