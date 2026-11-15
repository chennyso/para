from dataclasses import dataclass
import heapq
from itertools import count as serial_numbers
import math
from typing import Literal

from pydantic import Field, model_validator

from contracts import Contract, fingerprint
from search.linear_pricing_bound import LinearPricingBound


class PricingPolicy(Contract):
    preferred_stages: list[int] = Field(default_factory=list)
    interval_expansion_rule: Literal['reduced_cost', 'seed_distance'] = 'reduced_cost'
    memory_priority: float = Field(default=0, ge=0, allow_inf_nan=False)
    communication_priority: float = Field(default=0, ge=0, allow_inf_nan=False)
    max_frontier_expansions: int = Field(default=4096, ge=1)
    refresh_after_stagnant_rounds: int = Field(default=2,ge=2)
    minimum_improvement_fraction: float = Field(default=0.001,ge=0,lt=1,allow_inf_nan=False)
    reason: str = Field(min_length=1,max_length=600)
    evidence_refs: list[str] = Field(min_length=1,max_length=64)

    @model_validator(mode='after')
    def distinct_stages(self):
        if len(self.preferred_stages) != len(set(self.preferred_stages)):
            raise ValueError('优先阶段不能重复')
        return self


@dataclass(frozen=True)
class FrontierState:
    stage: int
    assigned_intervals: tuple
    endpoint_domains: tuple
    resource_signature: str
    pricing_bound: float | None = None


class FrontierPricer:
    def __init__(self, policy=None, generate=None,policy_model=None):
        if policy is None and generate is None:
            raise ValueError('增量定价需要明确策略或真实Agent接口')
        self.policy = policy
        self.generate = generate
        self.family_id = None
        self.pending = []
        self.proposed = set()
        self.policy_model = policy_model
        self.learned_program = None
        self.last_policy_history_length = 0

    def _state(self, master, oracle, stage, intervals):
        witness = master.domain.partial_witness(stage, intervals)
        fixed = {index:value for vp, pair in enumerate(intervals)
            for index,value in ((vp * master.domain.pp + stage, pair[0]),
                (vp * master.domain.pp + stage + 1, pair[1]))}
        domains = tuple((fixed[index],) if index in fixed else values
            for index,values in enumerate(master.domain.boundaries))
        identity = fingerprint(dict(family=self.family_id, stage=stage,
            assigned_intervals=intervals, endpoint_domains=domains))
        return FrontierState(stage, intervals, domains, identity), witness

    def _priority(self, state, witness, master, oracle, lp):
        intervals = master.domain.intervals(witness, state.stage)
        distance = sum(abs(a - b) for a,b in zip(witness, master.domain.seed))
        preferred = self.policy.preferred_stages
        stage_order = preferred.index(state.stage) if state.stage in preferred else len(preferred)
        # 排序模型的下界不能作为真实完整Projection的剪枝证明。
        score = self.pricing[state.stage].bound(state.assigned_intervals)
        if self.policy.memory_priority:
            memory = oracle.fragment_state_bytes(intervals)
            capacity = next(column.evidence['capacity_bytes'] for column in master.columns.values()
                if column.stage == state.stage)
            score += self.policy.memory_priority * memory / capacity
        if self.policy.communication_priority:
            estimated = oracle.estimated_column(master,state.stage,intervals)
            communication = sum(value for key,value in estimated.durations.items()
                if oracle.tasks[key]['kind'] not in ('forward_compute','backward_compute','optimizer_step'))
            score += self.policy.communication_priority * communication
        if not math.isfinite(score):
            raise ValueError('实际LP价格产生非法排序数值')
        return ((score, distance) if self.policy.interval_expansion_rule == 'reduced_cost'
            else (distance, score)) + (stage_order, -len(state.assigned_intervals))

    def propose(self, master, oracle, lp, count, observation):
        if type(count) is not int or count < 1:
            raise ValueError('增量定价候选数量必须为正整数')
        family = fingerprint(dict(domain=master.domain.__dict__, request=oracle.request,
            revision=oracle.revision, template=oracle.template))
        provider = dict(provider_calls=0)
        initial = self.family_id is None
        history = observation.get('history', [])
        refresh = (not initial and self.generate is not None
            and len(history)-self.last_policy_history_length >= self.policy.refresh_after_stagnant_rounds
            and all(row['lp_after_ms'] >= row['lp_before_ms']*(1-self.policy.minimum_improvement_fraction)
                for row in history[-self.policy.refresh_after_stagnant_rounds:]))
        if not initial and self.family_id != family:
            raise ValueError('执行结构或成本版本改变后需要建立新的定价器')
        if initial or refresh:
            self.family_id = family
            known = {oracle.revision, oracle.profile_id} | {
                row['prediction_id'] for row in observation['evaluations']}
            if observation.get('pricing_evidence_id') is not None:
                known.add(observation['pricing_evidence_id'])
            known.update(row['task_id'] for row in observation.get('prices', {}).get('event_prices', []))
            known.update(row['id'] for row in observation.get('prices', {}).get('generated_columns', []))
            if self.policy_model is not None:
                from search.pricing_program_learning import LearnedPricingProgram
                if self.learned_program is None:
                    self.learned_program = LearnedPricingProgram(self.policy_model)
                prior = self.learned_program.propose(oracle)
                known.add(prior['evidence_id'])
                observation = dict(observation,learned_policy_prior=prior)
            if self.policy is None or refresh:
                self.policy, provenance = self.generate('PhysicalStagePricingPolicyAgent',
                    '根据完整执行证据生成可重复执行的候选扩展策略。CPU定价器逐个扩展虚拟区间，'
                    '每次LP更新后重新排序。仅使用所给证据身份；排序权重不构成性能下界。'
                    'max_frontier_expansions控制一次CPU搜索的状态展开数量。',
                    dict(observation,evidence_catalog=sorted(value for value in known if value is not None)),
                    PricingPolicy)
                provider = dict(provider_calls=provenance.get('actual_provider_calls',1),provenance=provenance)
                self.last_policy_history_length = len(history)
            if not set(self.policy.evidence_refs) <= known:
                raise ValueError('定价策略引用了未提供的证据')
            if any(type(stage) is not int or not 0 <= stage < master.domain.pp
                    for stage in self.policy.preferred_stages):
                raise ValueError('定价策略引用了非法物理阶段')
            if initial:
                self.pending = [self._state(master, oracle, stage, ())[0]
                    for stage in range(master.domain.pp)]
        self.pricing = {stage:LinearPricingBound(master,oracle,lp,stage)
            for stage in range(master.domain.pp)}
        serial = serial_numbers()
        heap = []
        for state in self.pending:
            witness = master.domain.partial_witness(state.stage, state.assigned_intervals)
            heapq.heappush(heap, (self._priority(state,witness,master,oracle,lp), next(serial),state))
        rows, expanded, infeasible = [], 0, 0
        while heap and len(rows) < count and expanded < self.policy.max_frontier_expansions:
            _,_,state = heapq.heappop(heap)
            intervals,stage = state.assigned_intervals,state.stage
            if len(intervals) == master.domain.vpp:
                key = (stage, intervals)
                if key not in self.proposed:
                    self.proposed.add(key)
                    rows.append(dict(stage=stage, intervals=intervals, reason=self.policy.reason,
                        evidence_refs=self.policy.evidence_refs,
                        estimated_reduced_cost=master.reduced_cost(
                            oracle.estimated_column(master,stage,intervals),lp['prices'])))
                continue
            expanded += 1
            slot = stage + len(intervals) * master.domain.pp
            fixed = {index:value for vp,pair in enumerate(intervals)
                for index,value in ((vp * master.domain.pp + stage,pair[0]),
                    (vp * master.domain.pp + stage + 1,pair[1]))}
            for left in master.domain.boundaries[slot]:
                for right in master.domain.boundaries[slot + 1]:
                    values = dict(fixed)
                    if (left >= right or (slot in fixed and fixed[slot] != left)
                            or (slot + 1 in fixed and fixed[slot + 1] != right)):
                        infeasible += 1
                        continue
                    values.update({slot:left,slot+1:right})
                    witness = master.domain.complete_fixed(values)
                    if witness is None:
                        infeasible += 1
                        continue
                    child,_ = self._state(master,oracle,stage,intervals+((left,right),))
                    heapq.heappush(heap,(self._priority(child,witness,master,oracle,lp),next(serial),child))
        self.pending = [item[2] for item in heap]
        return rows, dict(provider, frontier_states=expanded, infeasible_pruned=infeasible,
            policy_refreshed=refresh,learned_policy_evidence_id=None if self.learned_program is None
                else self.learned_program.manifest['evidence_id'],
            dominance_pruned=0, pending_frontier_states=len(heap),
            pricing_certified=False, enumeration_complete=not heap,
            pricing_bound_scope='independent_interval_relaxation_for_linear_pricing_surrogate_only',
            stop='EXHAUSTED_DOMAIN' if not heap else 'OUTPUT_BATCH' if len(rows) == count
                else 'FRONTIER_EXPANSIONS', policy_id=fingerprint(self.policy.model_dump(mode='json')),
            policy=self.policy.model_dump(mode='json'))
