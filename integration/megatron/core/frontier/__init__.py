from __future__ import annotations
"""
Frontier: Parallelism Transition Planning for Megatron-LM.

Frontier adds a first-class optimization layer between region-level parallel
policy selection and the Megatron PP/VPP pipeline schedule.  The key idea is
that when adjacent model regions carry different parallel layouts, the required
layout conversion (a *Parallelism Transition*) can often be deferred, moved,
or fused—and the best placement depends jointly on the PP cut, the VPP chunk
assignment, and the communication overlap window available at each boundary.

This package is structured so that all planning happens before execution:

    profiling  → build a profile database
    contracts  → register region implementations
    mobility   → compute legal transition windows
    planner    → select PP/VPP + transition placements
    compiler   → emit a FrontierPlan consumed by Megatron schedules
    runtime    → execute pre-compiled resolve nodes on the hot path

The training hot path never calls the planner, agent, or simulator.
"""

from megatron.core.frontier.config import FrontierConfig
from megatron.core.frontier.state import (
    Placement,
    PlacementKind,
    DistributedState,
    TransitionObligation,
    ObligationKind,
)
from megatron.core.frontier.contracts import (
    RegionKind,
    ParallelPolicy,
    StatePattern,
    RegionContract,
    REGISTRY,
    register_contract,
    lookup_contracts,
)
from megatron.core.frontier.transition import (
    TransitionSpec,
    TransitionDecision,
    TransitionAnalysis,
)
from megatron.core.frontier.mobility import (
    StateTransferRule,
    compute_mobility_window,
    propagate_state,
)
from megatron.core.frontier.planner.compiler import (
    FrontierPlan,
    compile_plan,
)
from megatron.core.frontier.ir import (
    HardwareMapping,
    MemoryExecutionPolicy,
    ValueContract,
    PendingObligation,
    ParallelProgram,
    BoundaryProgram,
)
from megatron.core.frontier.capability import CapabilityCatalog, BoundaryImplementation
from megatron.core.frontier.boundary import BoundarySynthesizer, BoundaryVerifier, build_canonical_boundary
from megatron.core.frontier.hardware import (
    GPUDevice,
    HardwareTopology,
    LinkClass,
    LinkMeasurement,
    MappingTemplate,
    NICDevice,
    NodeTopology,
)
from megatron.core.frontier.memory_admission import (
    AdmissionLedger,
    AdmissionRecord,
    AdmissionStatus,
    MemoryAdmission,
    MemoryBreakdown,
    MemoryPrediction,
    StageMemoryOwnership,
    StageMemoryPrediction,
)

__all__ = [
    "FrontierConfig",
    "Placement",
    "PlacementKind",
    "DistributedState",
    "TransitionObligation",
    "ObligationKind",
    "RegionKind",
    "ParallelPolicy",
    "StatePattern",
    "RegionContract",
    "REGISTRY",
    "register_contract",
    "lookup_contracts",
    "TransitionSpec",
    "TransitionDecision",
    "TransitionAnalysis",
    "StateTransferRule",
    "compute_mobility_window",
    "propagate_state",
    "FrontierPlan",
    "compile_plan",
    "ValueContract",
    "MemoryExecutionPolicy",
    "HardwareMapping",
    "PendingObligation",
    "ParallelProgram",
    "BoundaryProgram",
    "CapabilityCatalog",
    "BoundaryImplementation",
    "BoundarySynthesizer",
    "BoundaryVerifier",
    "build_canonical_boundary",
    "GPUDevice",
    "HardwareTopology",
    "LinkClass",
    "LinkMeasurement",
    "MappingTemplate",
    "NICDevice",
    "NodeTopology",
    "AdmissionLedger",
    "AdmissionRecord",
    "AdmissionStatus",
    "MemoryAdmission",
    "MemoryBreakdown",
    "MemoryPrediction",
    "StageMemoryOwnership",
    "StageMemoryPrediction",
]
