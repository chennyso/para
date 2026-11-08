from contracts import fingerprint
from search.graph_exploration import execution_shape
from predictor.profile_database import plan_signature


PENDING_STATES = frozenset({'PENDING', 'APPROVED', 'DEFERRED'})


def candidate_cost_needs(trial):
    prediction = trial.get('prediction') or {}
    return {
        item.get('cost_key') or fingerprint(item.get('identity', item))
        for item in prediction.get('evidence', {}).get('profiling_requests', [])}


def ranking_sensitive_candidates(ranking):
    by_cost = {}
    for row in ranking['ranking_sensitive_costs']:
        if row['best_candidate_changed'] or row['top_k_order_inversions'] or row['top_k_set_changed']:
            by_cost.setdefault(row['cost_key'], set()).update(row['candidates'])
    return by_cost


def profile_decisions(requests, candidates, parent, sensitive_candidates, attempted_plans):
    attempted = set(attempted_plans) | {plan_signature(parent['plan']['backend_config'])}
    eligible = {}
    for trial in candidates:
        prediction = trial.get('prediction') or {}
        if (not prediction or trial.get('measurement') is not None or trial.get('failure')
                or prediction.get('memory', {}).get('admission') == 'REJECT_OOM'):
            continue
        identifier = trial['trial_id']
        shape = execution_shape(trial['plan']['backend_config'])
        unresolved = (prediction.get('prediction_status') == 'NEEDS_PROFILE'
            or prediction.get('performance', {}).get('step_time_ms') is None)
        eligible[identifier] = (shape, plan_signature(trial['plan']['backend_config']),
            unresolved, candidate_cost_needs(trial))

    decisions = []
    for request in requests:
        if request['state'] not in PENDING_STATES:
            continue
        keys = set(request['cost_keys'])
        covered = {}
        reasons = set()
        for identifier in sorted(set(request['trial_ids']) & eligible.keys()):
            shape, plan_id, unresolved, required = eligible[identifier]
            applicable = keys & required
            rank_keys = {key for key in keys if identifier in sensitive_candidates.get(key, ())}
            if rank_keys:
                covered[identifier] = (shape, rank_keys)
                reasons.add('RANKING_SENSITIVE')
            if unresolved and plan_id not in attempted and applicable:
                previous = covered.get(identifier, (shape, set()))[1]
                covered[identifier] = (shape, previous | applicable)
                reasons.add('UNRANKED_COMPLETE_STRATEGY')
        if not covered:
            continue
        decisions.append(dict(request_id=request['request_id'], reasons=sorted(reasons),
            candidate_ids=sorted(covered),
            shape_count=len({shape for shape, _ in covered.values()}),
            matching_cost_keys=sorted(set().union(*(keys for _, keys in covered.values()))),
            estimated_gpu_hours=request['estimated_gpu_hours']))
    return sorted(decisions, key=lambda row: (
        'RANKING_SENSITIVE' not in row['reasons'], -row['shape_count'],
        row['estimated_gpu_hours'], row['request_id']))
