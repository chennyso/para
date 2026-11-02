"""
BeamPlanner — main beam search loop over model regions.

Expands one layer at a time.  Every `simulator_interval` layers calls the
full simulator for accurate cost; otherwise uses profile-based lower bounds.

The search variables per step are:
  ChooseContract    — which RegionContract to use for this layer
  ResolveTransition — resolve a pending transition now (vs defer)
  CarryTransition   — defer a pending transition to a later layer
  PlacePPCut        — start a new PP stage after this layer
  StartVPChunk      — start a new VPP chunk after this layer
  ChooseLowering    — which TransitionLowering to apply when resolving
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Sequence, TYPE_CHECKING

from megatron.core.frontier.config import FrontierConfig
from megatron.core.frontier.contracts import RegionContract, REGISTRY
from megatron.core.frontier.mobility import compute_all_windows
from megatron.core.frontier.planner.candidate import SegmentCandidate
from megatron.core.frontier.planner.dominance import select_top_k
from megatron.core.frontier.planner.segment import build_segment_candidates
from megatron.core.frontier.state import DistributedState
from megatron.core.frontier.transition import PendingTransition, TransitionSpec
from megatron.core.frontier.ir.memory import MemoryExecutionPolicy

if TYPE_CHECKING:
    from megatron.core.frontier.profiling.database import ProfileDatabase
    from megatron.core.frontier.simulator.pipeline import PipelineSimulator


@dataclass
class PlannerState:
    """One node in the beam — a partial plan up to layer `region_index`."""
    region_index: int
    candidate: SegmentCandidate
    contract_sequence: list[str] = field(default_factory=list)
    transition_decisions: dict[int, object] = field(default_factory=dict)
    pp_cuts: list[int] = field(default_factory=list)
    vp_chunks: list[list[int]] = field(default_factory=list)
    current_pp_stage: int = 0
    current_vp_chunk: int = 0
    memory_policy: MemoryExecutionPolicy = field(default_factory=MemoryExecutionPolicy)
    topology_mapping: object | None = None


class BeamPlanner:
    """
    Beam search over (contract × transition_placement × PP/VPP) joint space.

    The search is deterministic given a fixed profile database and beam width.
    """

    def __init__(
        self,
        config: FrontierConfig,
        db: "ProfileDatabase | None" = None,
        simulator: "PipelineSimulator | None" = None,
        memory_policies: Sequence[MemoryExecutionPolicy] | None = None,
        topology_mappings: Sequence[object] | None = None,
    ):
        self.config = config
        self.db = db
        self.simulator = simulator
        self.memory_policies = tuple(memory_policies) if memory_policies is not None else self._default_memory_policies()
        self.topology_mappings = tuple(topology_mappings) if topology_mappings is not None else (None,)

    def _default_memory_policies(self) -> tuple[MemoryExecutionPolicy, ...]:
        policies = []
        for sharding in ("replicated", "distributed_optimizer", "fsdp"):
            for recompute in ("none", "selective", "prefix", "full"):
                for offload in ("none", "cpu_sync", "cpu_overlap", "cpu_prefetch"):
                    policies.append(MemoryExecutionPolicy(sharding, recompute, offload, "none", "disabled"))
        if self.config.memory_budget_bytes is not None:
            for sharding in ("distributed_optimizer", "fsdp"):
                policies.append(MemoryExecutionPolicy(
                    sharding,
                    "selective",
                    "none",
                    "cpu",
                    "disabled",
                    memory_pressure_threshold_bytes=self.config.memory_budget_bytes,
                ))
        return tuple(policies)

    def plan(
        self,
        num_layers: int,
        num_pp_stages: int,
        vpp: int,
        input_state: DistributedState,
        all_contracts: Sequence[RegionContract] | None = None,
        memory_policies: Sequence[MemoryExecutionPolicy] | None = None,
        topology_mappings: Sequence[object] | None = None,
    ) -> list[PlannerState]:
        """
        Run beam search and return the top-k final plans.

        `all_contracts[i]` is the list of legal contracts for layer i.
        If None, uses REGISTRY for every layer.
        """
        start_time = time.monotonic()
        deadline = (
            start_time + self.config.planner_time_budget_seconds
            if self.config.planner_time_budget_seconds
            else None
        )

        if all_contracts is None:
            all_contracts_by_layer = [
                REGISTRY.contracts_for_layer(i) for i in range(num_layers)
            ]
        else:
            all_contracts_by_layer = list(all_contracts)

        # Initialize beam with one empty state.
        policy_candidates = tuple(memory_policies) if memory_policies is not None else self.memory_policies
        topology_candidates = tuple(topology_mappings) if topology_mappings is not None else self.topology_mappings
        beam: list[PlannerState] = []
        for policy in policy_candidates:
            for topology_mapping in topology_candidates:
                beam.append(PlannerState(
                    region_index=-1,
                    candidate=SegmentCandidate(
                        start_region=0,
                        end_region=-1,
                        contract_ids=(),
                        internal_transitions=(),
                        pending_transitions=(),
                        output_state=input_state,
                        resource_signature=(policy.state_sharding, policy.recompute, policy.activation_offload, policy.optimizer_offload, policy.precision_aware_optimizer, repr(topology_mapping)),
                        memory_policy=policy,
                    ),
                    memory_policy=policy,
                    topology_mapping=topology_mapping,
                ))

        for layer_idx in range(num_layers):
            if deadline and time.monotonic() > deadline:
                break

            expanded: list[PlannerState] = []

            for state in beam:
                current_out = state.candidate.output_state
                pending = state.candidate.pending_transitions

                # Build segment candidates for this layer.
                layer_contracts = all_contracts_by_layer[layer_idx]
                new_candidates = build_segment_candidates(
                    layer_index=layer_idx,
                    input_state=current_out,
                    pending=pending,
                    pp_stage=state.current_pp_stage,
                    vp_chunk=state.current_vp_chunk,
                    contracts=layer_contracts,
                    all_contracts=all_contracts_by_layer,
                    db=self.db,
                )

                for cand in new_candidates:
                    for policy in policy_candidates:
                        if policy.optimizer_offload == "cpu" and not self._memory_pressure_requires_offload(cand, policy):
                            continue
                        resource_signature = (
                            policy.state_sharding,
                            policy.recompute,
                            policy.activation_offload,
                            policy.optimizer_offload,
                            policy.precision_aware_optimizer,
                            repr(state.topology_mapping),
                        )
                        cand_with_policy = replace(cand, resource_signature=resource_signature, memory_policy=policy)
                        new_state = PlannerState(
                            region_index=layer_idx,
                            candidate=cand_with_policy,
                            contract_sequence=state.contract_sequence + [cand_with_policy.contract_ids[-1]],
                            transition_decisions={
                                **state.transition_decisions,
                                **{td.transition_id: td for td in cand.internal_transitions},
                            },
                            pp_cuts=list(state.pp_cuts),
                            vp_chunks=list(state.vp_chunks),
                            current_pp_stage=state.current_pp_stage,
                            current_vp_chunk=state.current_vp_chunk,
                            memory_policy=policy,
                            topology_mapping=state.topology_mapping,
                        )
                        expanded.append(new_state)

                        # PP cut option: start a new stage after this layer.
                        if (
                            len(state.pp_cuts) < num_pp_stages - 1
                            and (layer_idx + 1) % max(1, num_layers // num_pp_stages) == 0
                            and (not state.pp_cuts or state.pp_cuts[-1] != layer_idx)
                        ):
                            pp_state = PlannerState(
                                region_index=layer_idx,
                                candidate=cand_with_policy,
                                contract_sequence=new_state.contract_sequence,
                                transition_decisions=dict(new_state.transition_decisions),
                                pp_cuts=state.pp_cuts + [layer_idx],
                                vp_chunks=list(state.vp_chunks),
                                current_pp_stage=state.current_pp_stage + 1,
                                current_vp_chunk=state.current_vp_chunk,
                                memory_policy=policy,
                                topology_mapping=state.topology_mapping,
                            )
                            expanded.append(pp_state)

            # Run simulator periodically for accurate cost.
            if (
                self.simulator is not None
                and layer_idx % self.config.simulator_interval == 0
                and expanded
            ):
                for s in expanded:
                    result = self.simulator.estimate_step_time(
                        contract_sequence=s.contract_sequence,
                        pp_cuts=s.pp_cuts,
                        vpp=vpp,
                        pending_transitions=list(s.candidate.pending_transitions),
                        memory_policy=s.memory_policy,
                        topology_mapping=s.topology_mapping,
                    )
                    s.candidate.max_stage_load_us = result.max_stage_load_us
                    s.candidate.peak_memory_bytes = result.peak_memory_bytes
                    s.candidate.stage_ready_times = result.stage_ready_times
                    s.candidate.max_ready_time_us = max(result.stage_ready_times)

            # Dominance pruning + beam selection.
            beam_candidates = [s.candidate for s in expanded]
            kept = select_top_k(beam_candidates, self.config.beam_width)
            kept_set = {id(c) for c in kept}
            beam = [s for s in expanded if id(s.candidate) in kept_set]

        return beam

    def _memory_pressure_requires_offload(self, candidate: SegmentCandidate, policy: MemoryExecutionPolicy) -> bool:
        threshold = policy.memory_pressure_threshold_bytes
        if threshold is None:
            threshold = self.config.memory_budget_bytes
        if threshold is None:
            return False
        return candidate.peak_memory_bytes > threshold
