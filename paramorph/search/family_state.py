from datetime import datetime, timezone
import json
import math

from contracts import fingerprint
from predictor.profile_database import normalized_execution, plan_signature


GLOBAL_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping')


def global_values(request):
    normalized = normalized_execution(request)
    return {name: normalized[name] for name in GLOBAL_FIELDS}


def global_identity(request):
    return 'global-family-' + fingerprint(global_values(request))[:16]


class FamilyLedger:
    def __init__(self, store):
        self.store = store
        self.connection = store.connection
        self.connection.execute('CREATE TABLE IF NOT EXISTS parallel_families '
            '(id TEXT PRIMARY KEY, document TEXT NOT NULL)')
        self.connection.commit()

    def records(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM parallel_families ORDER BY id')]

    def get(self, request):
        identifier = global_identity(request)
        row = self.connection.execute('SELECT document FROM parallel_families WHERE id=?',
            (identifier,)).fetchone()
        if row is not None:
            return json.loads(row[0])
        return dict(family_id=identifier, global_values=global_values(request),
            status='UNKNOWN', candidate_ids=[], candidates={}, best_predicted=None,
            best_measured=None, lower_bound_ms=None, lower_bound_evidence=None,
            cost_revision=None, solve_records=[], pending_cost_keys=[],
            prediction_scope='cost_model', search_complete=False)

    def save(self, document):
        document = dict(document, updated_at=datetime.now(timezone.utc).isoformat())
        self.connection.execute('INSERT INTO parallel_families VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET document=excluded.document',
            (document['family_id'], json.dumps(document, sort_keys=True, allow_nan=False)))
        self.connection.commit()
        return document

    def record_solver(self, request, result, cost_revision):
        document = self.get(request)
        if result['status'] not in ('FEASIBLE', 'INFEASIBLE', 'UNKNOWN'):
            raise ValueError('家族求解状态未注册')
        record = {key: result[key] for key in ('status', 'search_complete', 'termination',
            'counters', 'proofs', 'missing_cost_requests', 'lower_bound_ms',
            'lower_bound_evidence', 'lower_bound_status', 'layer_load_reference_ms',
            'layer_load_reference_evidence', 'conditional_status') if key in result}
        record['cost_revision'] = cost_revision
        record['scope'] = result.get('scope', 'conditional_search_domain')
        record['fixed_values'] = result.get('fixed_values', {})
        record['allowed_ranges'] = result.get('allowed_ranges', {})
        record['record_id'] = fingerprint(record)
        if record['record_id'] not in {row['record_id'] for row in document['solve_records']}:
            document['solve_records'].append(record)
        pending = {row['cost_key'] for row in result.get('missing_cost_requests', [])
            if 'cost_key' in row}
        if record['scope'] == 'all_family_completions':
            document['pending_cost_keys'] = sorted(pending)
            document['search_complete'] = result['search_complete']
        else:
            document['pending_cost_keys'] = sorted(set(document['pending_cost_keys']) | pending)
        if result['status'] == 'FEASIBLE':
            document['status'] = 'FEASIBLE'
        # 局部意图的不可行证明不能排除整个 H 家族。
        if (result['status'] == 'INFEASIBLE' and result['search_complete']
                and result.get('scope') == 'all_family_completions'
                and not result.get('fixed_values') and not result.get('allowed_ranges')
                and result.get('proofs')):
            if (document['best_measured'] is not None or
                    document['status'] == 'FEASIBLE' and document['cost_revision'] == cost_revision):
                raise ValueError('家族不可行证明与已有真实计划或同版本可行证据冲突')
            document['status'] = 'INFEASIBLE'
            document['search_complete'] = True
        bound = result.get('lower_bound_ms')
        if bound is not None:
            if not math.isfinite(bound) or bound < 0 or not result.get('lower_bound_evidence'):
                raise ValueError('性能下界需要有限数值和证明来源')
            evidence = result['lower_bound_evidence']
            global_bound = (record['scope'] == 'all_family_completions'
                and not record['fixed_values'] and not record['allowed_ranges']
                and evidence.get('scope') == 'all_family_completions'
                and evidence.get('admissible_replay_lower_bound') is True
                and evidence.get('prediction_target') == 'step_time_ms'
                and bool(evidence.get('proof_refs')))
            if global_bound:
                document['lower_bound_ms'] = bound
                document['lower_bound_evidence'] = evidence
        document['cost_revision'] = cost_revision
        return self.save(document)

    def record_trial(self, trial, cost_revision):
        request = trial['plan']['backend_config']
        document = self.get(request)
        prediction, measurement = trial.get('prediction'), trial.get('measurement')
        performance = (prediction or {}).get('performance', {})
        memory = (prediction or {}).get('memory', {})
        value = performance.get('step_time_ms')
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('候选预测耗时必须为有限正数')
        passed = bool(measurement and measurement.get('status') == 'PASS'
            and measurement.get('performance_eligible', False))
        entry = dict(trial_id=trial['trial_id'], plan_hash=plan_signature(request),
            state=trial['state'], prediction_id=(prediction or {}).get('prediction_id'),
            predicted_step_ms=value, memory_admission=memory.get('admission'),
            cost_revision=(prediction or {}).get('cost_database_revision'), measured_pass=passed,
            measured_step_ms=measurement['step_seconds'] * 1000 if passed else None,
            measured_tokens_per_s=measurement['tokens_per_second'] if passed else None,
            layout={key: request.get(key) for key in ('vpp', 'micro_batch', 'decoder_chunks',
                'recompute', 'pipeline_schedule', 'overlap_p2p')},
            state_sharding={key: normalized_execution(request)[key] for key in (
                'execution_backend', 'optimizer_sharding', 'dense_sharding_strategy',
                'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size',
                'expert_dp_inner_size', 'expert_dp_outer_size')})
        document['candidates'][trial['trial_id']] = entry
        document['candidate_ids'] = sorted(document['candidates'])
        current = [row for row in document['candidates'].values()
            if row['cost_revision'] == cost_revision and row['predicted_step_ms'] is not None
            and row['memory_admission'] in ('ADMIT', 'ADMIT_ANALYTICAL', 'ADMIT_CALIBRATED',
                'WITHIN_ESTIMATED_CAPACITY')]
        measured = [row for row in document['candidates'].values() if row['measured_pass']]
        document['best_predicted'] = (min(current, key=lambda row: (
            row['predicted_step_ms'], row['plan_hash'])) if current else None)
        document['best_measured'] = (max(measured, key=lambda row: (
            row['measured_tokens_per_s'], row['plan_hash'])) if measured else None)
        if current or measured:
            document['status'] = 'FEASIBLE'
        document['cost_revision'] = cost_revision
        return self.save(document)

    def invalidate_costs(self, cost_revision):
        changed = []
        for document in self.records():
            if document['cost_revision'] == cost_revision:
                continue
            document.update(cost_revision=cost_revision, best_predicted=None,
                lower_bound_ms=None, lower_bound_evidence=None, search_complete=False,
                status='FEASIBLE' if document['best_measured'] else 'UNKNOWN')
            changed.append(self.save(document)['family_id'])
        return changed
