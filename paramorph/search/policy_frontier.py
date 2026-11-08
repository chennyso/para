from copy import deepcopy
from pydantic import TypeAdapter

from contracts import RewriteIntent, fingerprint
from core_data_models import MemoryPolicy
from graph.native_recomputation import selective_modules
from predictor.profile_database import plan_signature
from search.complete_strategy import compile_complete_strategy, complete_values


LAYOUT_ID_FIELDS = ('model', 'checkpoint', 'data', 'global_batch', 'sequence', 'precision',
    'optimizer', 'tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping',
    'vpp', 'micro_batch', 'group', 'decoder_chunks', 'pipeline_schedule', 'overlap_p2p')


class PolicyFrontier:
    def __init__(self, rewriter):
        self.rewriter = rewriter

    @staticmethod
    def retain(evaluations):
        rejected = {row['plan_signature'] for row in evaluations
            if (row.get('memory') or {}).get('admission') == 'REJECT_OOM'}
        ranked = [row for row in evaluations if row['status'].startswith('PREDICTED')
            and row['step_time_ms'] is not None
            and not row['missing_profiles'] and row.get('memory') is not None
            and row['memory'].get('admission') == 'WITHIN_ESTIMATED_CAPACITY'
            and row['memory'].get('peak_per_rank_bytes')
            and row['comparison_conditions'].get('revision') is not None
            and row['comparison_conditions'].get('predictor') is not None]
        dominated = {}
        for row in ranked:
            for other in ranked:
                if (row['plan_signature'] == other['plan_signature'] or row['layout_id'] != other['layout_id']
                        or row['comparison_conditions'] != other['comparison_conditions']):
                    continue
                own = (row['step_time_ms'], max(row['memory']['peak_per_rank_bytes'].values()))
                alternative = (other['step_time_ms'], max(other['memory']['peak_per_rank_bytes'].values()))
                if all(a <= b for a, b in zip(alternative, own, strict=True)) and alternative != own:
                    dominated[row['plan_signature']] = other['plan_signature']
                    break
        return dict(dominated=dominated,
            rejected_oom=sorted(rejected),
            retained=[row['plan_signature'] for row in evaluations if row['plan_signature'] not in dominated
                and row['plan_signature'] not in rejected],
            unresolved=[row['plan_signature'] for row in evaluations if row not in ranked
                and row['plan_signature'] not in rejected])

    def expand(self, parent, rows, scope, limit, *, measured_signatures=()):
        if limit < 1:
            raise ValueError('策略比较预算必须为正整数')
        groups = {}
        for row in rows:
            key = fingerprint({name: row['request'].get(name) for name in LAYOUT_ID_FIELDS})
            groups.setdefault(key, []).append(row)
        output, comparisons, seen = [], [], set()
        measured_signatures = set(measured_signatures)
        for layout_id, seeds in groups.items():
            proposals = []
            for seed in seeds:
                request = seed['request']
                policies = [(request['recompute'], (request.get('memory_policy') or {}).get('recompute_modules', []))]
                if 'recompute' in scope.adjustable and 'memory_policy' in scope.adjustable:
                    policies = [('none', []), *policies, ('full', []),
                        *(('selective', [name]) for name in selective_modules(self.rewriter.model))]
                for mode, modules in policies:
                    if mode not in scope.allowed_ranges.get('recompute', [mode]):
                        continue
                    candidate = deepcopy(request)
                    candidate['recompute'] = mode
                    candidate['memory_policy'] = dict(candidate.get('memory_policy') or {},
                        recompute_granularity=None if mode == 'none' else 'selective' if mode.startswith('selective') else mode,
                        recompute_modules=modules)
                    if 'memory_policy' in scope.allowed_ranges:
                        adapter = TypeAdapter(MemoryPolicy)
                        actual = adapter.dump_python(adapter.validate_python(candidate['memory_policy']))
                        permitted = [adapter.dump_python(adapter.validate_python(value))
                            for value in scope.allowed_ranges['memory_policy']]
                        if actual not in permitted:
                            continue
                    key = plan_signature(candidate)
                    if key not in {item[0] for item in proposals}:
                        proposals.append((key, candidate, seed))
            evaluated, pending, rejected, references = [], [], [], []
            for key, candidate, seed in proposals:
                if key in seen:
                    continue
                if key == plan_signature(parent) or key in measured_signatures:
                    seen.add(key)
                    references.append(key)
                    continue
                if len(output) >= limit:
                    pending.append(key)
                    continue
                intent = RewriteIntent(operation='JointTransform', values=complete_values(candidate),
                    adjustable=[], preserve=[], hypothesis='在相同执行布局下比较后端支持的存储策略',
                    parent_plan_id=scope.parent_plan_id, evidence_refs=list(scope.evidence_refs))
                try:
                    compiled = compile_complete_strategy(self.rewriter, parent, intent)
                except ValueError as error:
                    if not isinstance(error.args[0], dict) or error.args[0].get('status') != 'CONSTRAINT_CONFLICT':
                        raise
                    rejected.append(dict(plan_signature=key, reason=error.args[0]))
                    continue
                seen.add(key)
                compiled['intent'] = intent.model_dump(mode='json')
                compiled['partition_evidence'] = seed['partition_evidence']
                compiled['policy_layout_id'] = layout_id
                compiled['proposal_kind'] = 'conditional_intent'
                output.append(compiled)
                evaluated.append(key)
            comparisons.append(dict(layout_id=layout_id, compiled_policies=evaluated,
                reference_policies=references,
                pending_policies=pending, rejected_policies=rejected,
                status='PARTIAL' if pending else 'COMPLETE',
                termination='POLICY_COMPARISON_BUDGET' if pending else 'EXHAUSTED_DECLARED_POLICIES'))
        return dict(candidates=output, comparisons=comparisons)
