"""
FrontierPlan and compile_plan — the final output of the planner.

compile_plan takes the best PlannerState from the beam search and assembles
a FrontierPlan, which is the object consumed by the Megatron schedule builder.

The plan is purely declarative: it says what to do, not how to wire PyTorch
autograd.  The Megatron integration hooks (fine_grained_callables.py and
model_chunk_schedule_plan.py patches) read the plan and insert ResolveNodes
at the correct positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from megatron.core.frontier.transition import TransitionDecision
from megatron.core.frontier.ir.memory import MemoryExecutionPolicy

if TYPE_CHECKING:
    from megatron.core.frontier.planner.beam import PlannerState
    from megatron.core.frontier.runtime.boundary import BoundarySchema


@dataclass(frozen=True)
class FrontierPlan:
    """
    Compiled training plan produced by the beam planner.

    All fields are immutable after construction.  The plan is distributed to
    all ranks before training begins; each rank uses it to initialize its
    ResolveScheduleNodes and buffer pool.
    """
    # Contract name for each layer (length = num_layers).
    contract_ids: tuple[str, ...]

    # Keyed by transition_id.
    transition_decisions: dict[int, TransitionDecision]

    # PP cut positions: layer indices *after* which a new stage starts.
    # Length = num_pp_stages - 1.
    pp_cuts: tuple[int, ...]

    # VPP chunk assignments: pp_cuts divides layers into stages; vp_chunks
    # further divides each stage into VPP virtual chunks.
    # vp_chunks[stage][chunk] = list of layer indices in that chunk.
    vp_chunks: tuple[tuple[tuple[int, ...], ...], ...]

    # Simulator estimates.
    estimated_step_ms: float = 0.0
    estimated_peak_memory_bytes: int = 0

    # BoundarySchemas, keyed by transition_id.  Populated by compile_plan.
    boundary_schemas: dict[int, "BoundarySchema"] = field(default_factory=dict)
    memory_policy: MemoryExecutionPolicy = field(default_factory=MemoryExecutionPolicy)
    topology_mapping: object | None = None

    def num_pp_stages(self) -> int:
        return len(self.pp_cuts) + 1

    def layers_for_stage(self, stage: int) -> tuple[int, ...]:
        """Return the layer indices assigned to PP stage `stage`."""
        cuts = ((-1,) + self.pp_cuts + (len(self.contract_ids) - 1,))
        start = cuts[stage] + 1
        end = cuts[stage + 1] + 1
        return tuple(range(start, end))

    def contract_for_layer(self, layer: int) -> str:
        return self.contract_ids[layer]

    def resolve_boundary_for_transition(self, tid: int) -> int | None:
        d = self.transition_decisions.get(tid)
        return d.resolve_boundary if d else None


def compile_plan(best_state: "PlannerState", num_layers: int) -> FrontierPlan:
    """
    Build a FrontierPlan from the best beam search result.
    """
    # Pad contract sequence to num_layers if beam ended early.
    contracts = list(best_state.contract_sequence)
    while len(contracts) < num_layers:
        contracts.append(contracts[-1] if contracts else "")

    pp_cuts = tuple(sorted(best_state.pp_cuts))
    num_stages = len(pp_cuts) + 1

    # Build vp_chunks as a list of lists.
    stage_layers: list[list[int]] = [[] for _ in range(num_stages)]
    cuts_ext = list(pp_cuts) + [num_layers]
    stage_idx = 0
    for l in range(num_layers):
        stage_layers[stage_idx].append(l)
        if cuts_ext and l == cuts_ext[stage_idx]:
            stage_idx += 1

    vp_chunks_raw: list[tuple[tuple[int, ...], ...]] = []
    for stage in stage_layers:
        vp_chunks_raw.append((tuple(stage),))

    return FrontierPlan(
        contract_ids=tuple(contracts),
        transition_decisions=dict(best_state.transition_decisions),
        pp_cuts=pp_cuts,
        vp_chunks=tuple(vp_chunks_raw),
        estimated_step_ms=best_state.candidate.total_us() / 1000.0,
        estimated_peak_memory_bytes=best_state.candidate.peak_memory_bytes,
        memory_policy=best_state.memory_policy,
        topology_mapping=best_state.topology_mapping,
    )
