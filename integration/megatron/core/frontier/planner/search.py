from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

from megatron.core.frontier.ir.memory import MemoryExecutionPolicy
from megatron.core.frontier.ir.boundary import PendingOpportunity


@dataclass(frozen=True)
class SearchState:
    next_region: int
    domains: tuple
    pp_stage: int
    vp_chunk: int
    active_obligations: tuple
    stage_load_us: tuple[float, ...]
    peak_memory_bytes: int
    partial_score: float
    memory_policy: MemoryExecutionPolicy = MemoryExecutionPolicy()
    topology_mapping: object | None = None
    pending_opportunities: tuple[PendingOpportunity, ...] = ()

    def __post_init__(self):
        if not self.stage_load_us or any(not math.isfinite(value) or value < 0 for value in self.stage_load_us):
            raise ValueError('search state requires finite nonnegative physical stage loads')
        if self.peak_memory_bytes < 0 or not math.isfinite(self.partial_score) or self.partial_score < 0:
            raise ValueError('search state requires valid memory and execution cost')


class ParallelProgramSearch:
    def __init__(self, model_graph, domain_candidates, boundary_candidates, simulator, feasibility, beam_width: int = 32, memory_policies: Sequence[MemoryExecutionPolicy] | None = None, topology_mappings: Sequence[object] | None = None) -> None:
        self.model_graph = model_graph
        self.domain_candidates = domain_candidates
        self.boundary_candidates = boundary_candidates
        self.simulator = simulator
        self.feasibility = feasibility
        self.beam_width = beam_width
        self.memory_policies = tuple(memory_policies) if memory_policies is not None else (
            MemoryExecutionPolicy("replicated", "none", "none", "none", "disabled"),
            MemoryExecutionPolicy("distributed_optimizer", "selective", "none", "none", "disabled"),
            MemoryExecutionPolicy("fsdp", "prefix", "cpu_overlap", "none", "disabled"),
        )
        self.topology_mappings = tuple(topology_mappings) if topology_mappings is not None else (None,)

    def search(self, neighborhood=None):
        beam = [self._initial_state()]
        total_regions = self.model_graph.num_regions
        while beam and any(state.next_region < total_regions for state in beam):
            expanded = []
            for state in beam:
                if state.next_region >= total_regions:
                    expanded.append(state)
                    continue
                expanded.extend(self._expand_state(state, neighborhood))
            beam = self._select_frontier(expanded)
        programs = [self._compile_search_state(state) for state in beam if state.next_region >= total_regions]
        if not programs:
            raise RuntimeError("parallel program search produced no complete program")
        return min(programs, key=lambda program: self.simulator.simulate(program).step_time_us)

    def _initial_state(self):
        return SearchState(0, (), 0, 0, (), (0.0,), 0, 0.0, self.memory_policies[0], self.topology_mappings[0])

    def _expand_state(self, state, neighborhood):
        candidates = self.domain_candidates.get(state.next_region, ())
        expanded = []
        for candidate in candidates:
            for policy in self.memory_policies:
                for topology_mapping in self.topology_mappings:
                    threshold = policy.memory_pressure_threshold_bytes
                    if policy.optimizer_offload == "cpu" and (threshold is None or candidate.memory_lower_bound <= threshold):
                        continue
                    expanded.append(replace(
                        state,
                        next_region=candidate.end_region + 1,
                        domains=state.domains + (candidate,),
                        peak_memory_bytes=max(state.peak_memory_bytes, candidate.memory_lower_bound),
                        partial_score=state.partial_score + candidate.lower_bound_us,
                        memory_policy=policy,
                        topology_mapping=topology_mapping,
                    ))
        return expanded

    def _semantic_key(self, state):
        obligations = tuple(sorted((item.boundary_id, item.latest_region, item.state_signature) for item in state.active_obligations))
        policy = state.memory_policy
        opportunities = tuple(sorted({item.signature() for item in state.pending_opportunities}, key=repr))
        return state.next_region, policy, state.pp_stage, state.vp_chunk, obligations, opportunities, state.topology_mapping

    def _select_frontier(self, states):
        if self.beam_width < 1:
            raise ValueError('opportunity beam requires a positive capacity per semantic signature')
        buckets = {}
        for state in self._obligation_preserving_prune(states):
            buckets.setdefault(self._semantic_key(state), []).append(state)
        selected = []
        # 每个机会签名单独分配容量，避免一个机会集合占据全部候选位置。
        for bucket in buckets.values():
            bucket.sort(key=lambda state: (state.partial_score, state.peak_memory_bytes, state.stage_load_us))
            selected.extend(bucket[:self.beam_width])
        return selected

    def _obligation_preserving_prune(self, states):
        buckets = {}
        for state in states:
            buckets.setdefault(self._semantic_key(state), []).append(state)
        kept = []
        for bucket in buckets.values():
            frontier = []
            for state in bucket:
                if any(self._dominates(other, state) for other in frontier):
                    continue
                frontier = [other for other in frontier if not self._dominates(state, other)]
                frontier.append(state)
            kept.extend(frontier)
        return kept

    def _dominates(self, left, right):
        if self._semantic_key(left) != self._semantic_key(right) or len(left.stage_load_us) != len(right.stage_load_us):
            return False
        return (all(a <= b for a, b in zip(left.stage_load_us, right.stage_load_us, strict=True))
                and left.peak_memory_bytes <= right.peak_memory_bytes and left.partial_score <= right.partial_score)

    def _compile_search_state(self, state):
        compiler = getattr(self.simulator, "compile_program", None)
        if compiler is None:
            return state
        return compiler(state)
