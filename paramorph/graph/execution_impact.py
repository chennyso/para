from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionImpact:
    parent_plan_id: str
    candidate_plan_id: str
    parent_prediction_id: str | None
    candidate_prediction_id: str | None
    cost_revision: str
    numeric_comparison_eligible: bool
    step_time_delta_ms: float | None
    peak_delta_by_rank_bytes: dict
    persistent_delta_by_rank_bytes: dict
    affected_modules: tuple
    critical_before: tuple
    critical_after: tuple
    memory_admission: str | None
    missing_profiles: tuple
    evidence_ids: tuple


def execution_impact(parent, candidate, before, after, comparison, cost_revision):
    before, after = before or {}, after or {}
    return ExecutionImpact(parent['plan']['plan_id'], candidate['plan']['plan_id'],
        before.get('prediction_id'), after.get('prediction_id'), cost_revision,
        comparison.get('numeric_comparison_eligible', False), comparison.get('predicted_step_delta_ms'),
        {str(row['rank']): row['delta']['peak_bytes'] for row in comparison.get('rank_changes', [])
            if 'peak_bytes' in row['delta']},
        {str(row['rank']): row['delta']['persistent_bytes'] for row in comparison.get('rank_changes', [])
            if 'persistent_bytes' in row['delta']},
        tuple(row['module_id'] for row in comparison.get('module_ownership_changes', [])),
        tuple(before.get('diagnosis', {}).get('critical_tasks', [])),
        tuple(after.get('diagnosis', {}).get('critical_tasks', [])),
        after.get('memory', {}).get('admission'),
        tuple(after.get('evidence', {}).get('missing_profiles', [])),
        tuple(value for value in (before.get('prediction_id'), after.get('prediction_id')) if value))
