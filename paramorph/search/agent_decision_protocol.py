from typing import Literal

from pydantic import Field

from contracts import Contract
from search.dag_agent_control import DAGCandidateReview
from search.system_search import InvestigationSelection


class CandidateVerification(Contract):
    trial_id: str = Field(min_length=1)
    prediction_evidence_id: str = Field(min_length=1)
    prediction_status: str = Field(min_length=1)
    memory_admission: str = Field(min_length=1)
    recommended_action: Literal['PROFILE', 'LOAD_VALIDATE', 'BENCHMARK', 'DEFER']
    reason: str = Field(min_length=1)


class EvidencedDAGReview(DAGCandidateReview):
    candidate_verifications: list[CandidateVerification] = Field(
        min_length=1, max_length=8)


class EvidencedExperimentDecision(InvestigationSelection):
    stop_reason: Literal[
        'CONTINUE', 'NO_GPU_BUDGET', 'NO_EXECUTABLE_MEASUREMENT',
        'NO_IMPROVEMENT_WORTH_TESTING',
    ] = 'CONTINUE'


def review_conflicts(review, observation, trials):
    by_id = {row['trial_id']: row for row in trials}
    required = set(observation['candidate_ids'])
    sources = observation['candidate_prediction_evidence']
    declared = [row.trial_id for row in review.candidate_verifications]
    problems = []
    if set(declared) != required or len(declared) != len(set(declared)):
        problems.append(dict(
            field='candidate_verifications',
            expected_trial_ids=sorted(required),
            submitted_trial_ids=declared,
        ))
    for row in review.candidate_verifications:
        if row.trial_id not in required:
            continue
        prediction = by_id[row.trial_id]['prediction']
        expected = dict(
            prediction_evidence_id=sources[row.trial_id],
            prediction_status=prediction['prediction_status'],
            memory_admission=prediction['memory']['admission'],
        )
        observed = dict(
            prediction_evidence_id=row.prediction_evidence_id,
            prediction_status=row.prediction_status,
            memory_admission=row.memory_admission,
        )
        if observed != expected:
            problems.append(dict(
                trial_id=row.trial_id, expected=expected, submitted=observed))
        if row.recommended_action == 'BENCHMARK' and (
                prediction['prediction_status'] == 'NEEDS_PROFILE'
                or prediction['memory']['admission'] != 'WITHIN_ESTIMATED_CAPACITY'):
            problems.append(dict(
                trial_id=row.trial_id,
                reason='当前成本或容量信息仍有缺项，应先补测或验证；'
                    '需要直接探索训练时由Evaluation Agent明确使用探索权限',
            ))
    return problems


def experiment_conflicts(decision, observation, trials, *, gpu_enabled,
                         remaining_gpu_hours):
    if not isinstance(decision, EvidencedExperimentDecision):
        raise TypeError('DAG-Agent评价必须提交带停止原因的结构化决策')
    visible = observation.get('candidates', {})
    columns = visible.get('columns', [])
    rows = visible.get('rows', [])
    active_ids = set()
    if 'trial_id' in columns:
        active_ids = {row[columns.index('trial_id')] for row in rows}
    available = {row['request_id']: row
        for row in observation.get('profiling_requests', [])}
    chosen = set(decision.accepted_candidate_ids)
    load = set(decision.load_validation_candidate_ids)
    profiles = set(decision.accepted_profiling_request_ids)
    problems = []
    if not chosen <= active_ids:
        problems.append(dict(
            field='accepted_candidate_ids',
            invalid_ids=sorted(chosen - active_ids)))
    if not load <= active_ids:
        problems.append(dict(
            field='load_validation_candidate_ids',
            invalid_ids=sorted(load - active_ids)))
    if not profiles <= available.keys():
        problems.append(dict(
            field='accepted_profiling_request_ids',
            invalid_ids=sorted(profiles - available.keys())))
    if (len(chosen) != len(decision.accepted_candidate_ids)
            or len(load) != len(decision.load_validation_candidate_ids)
            or len(profiles) != len(decision.accepted_profiling_request_ids)):
        problems.append(dict(reason='同一执行操作包含重复的候选或测量请求'))
    if chosen or load or profiles:
        if decision.stop_reason != 'CONTINUE':
            problems.append(dict(
                field='stop_reason',
                reason='已经选择实际工作时必须继续执行'))
        for identifier in profiles & available.keys():
            if not active_ids.intersection(available[identifier]['trial_ids']):
                problems.append(dict(
                    request_id=identifier,
                    reason='成本采集与本次候选集合没有关联'))
        return problems

    pending = [
        row for row in available.values()
        if row['state'] in ('PENDING', 'APPROVED', 'DEFERRED', 'OUT_OF_BUDGET')
        and active_ids.intersection(row['trial_ids'])
    ]
    by_id = {row['trial_id']: row for row in trials}
    unresolved = [identifier for identifier in active_ids
        if by_id[identifier]['prediction']['prediction_status'] == 'NEEDS_PROFILE']
    if (gpu_enabled and remaining_gpu_hours > 0 and pending and unresolved):
        problems.append(dict(
            reason='存在合法候选、缺失成本以及可执行Profile，'
                '必须选择实际成本测量或提供客观无法执行的证据',
            pending_profile_request_ids=[row['request_id'] for row in pending],
            unresolved_candidate_ids=sorted(unresolved)))
    if decision.stop_reason == 'CONTINUE':
        problems.append(dict(
            field='stop_reason',
            reason='没有选择执行操作时必须明确停止依据'))
    if decision.stop_reason == 'NO_GPU_BUDGET' and (
            not gpu_enabled or remaining_gpu_hours > 0):
        problems.append(dict(
            field='stop_reason',
            reason='实际GPU预算状态与停止原因不一致'))
    if decision.stop_reason == 'NO_EXECUTABLE_MEASUREMENT' and pending:
        problems.append(dict(
            field='stop_reason',
            reason='仍有待执行的成本测量请求'))
    if decision.stop_reason == 'NO_IMPROVEMENT_WORTH_TESTING' and unresolved:
        problems.append(dict(
            field='stop_reason',
            reason='仍有缺少完整性能成本的合法候选，无法证明没有收益'))
    return problems
