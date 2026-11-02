from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from megatron.core.frontier.ir.boundary import BoundaryProgram
from megatron.core.frontier.ir.memory import MemoryExecutionPolicy
from megatron.core.frontier.ir.region import ParallelDomain, RegionKind
from megatron.core.frontier.ir.contract import ProcessGroupContract
from megatron.core.frontier.global_search.policy import GlobalParallelPolicy


@dataclass(frozen=True)
class RegionParallelMapping:
    region_id: str
    policy: GlobalParallelPolicy
    process_groups: tuple[tuple[str, tuple[ProcessGroupContract, ...]], ...]
    implementation_key: str
    region_class: RegionKind = RegionKind.OTHER_REGISTERED

    def __post_init__(self):
        axes = dict(self.process_groups)
        expected = {'tp', 'cp', 'pp', 'dp', 'dp-cp'}
        if self.policy.expert is not None:
            expected |= {'etp', 'ep', 'expert_dp'}
        if (not self.region_id or not self.implementation_key or set(axes) != expected
                or len(axes) != len(self.process_groups)):
            raise ValueError('Region mappings require the complete native dense and expert process groups')
        ranks = set(range(self.policy.dense.world_size))
        sizes = {'tp': self.policy.dense.tp, 'cp': self.policy.dense.cp,
            'pp': self.policy.dense.pp, 'dp': self.policy.dense.dp,
            'dp-cp': self.policy.dense.dp * self.policy.dense.cp}
        if self.policy.expert is not None:
            sizes.update(etp=self.policy.expert.etp, ep=self.policy.expert.ep,
                expert_dp=self.policy.expert.expert_dp)
        for axis, groups in self.process_groups:
            members = [rank for group in groups for rank in group.ranks]
            if set(members) != ranks or len(members) != len(ranks):
                raise ValueError(f'Region {axis} groups must partition the actual GPU allocation')
            if any(len(group.ranks) != sizes[axis] for group in groups):
                raise ValueError(f'Actual {axis} group size differs from the region parallel decomposition')
        if any(len(group.ranks) != self.policy.dense.pp for group in axes['pp']):
            raise ValueError('Actual pipeline process groups differ from the declared PP degree')


@dataclass(frozen=True)
class HybridStructure:
    pp_degree: int
    dp: int
    micro_batch_size: int
    num_microbatches: int
    global_batch_size: int
    physical_stage_ranks: tuple[tuple[int, ...], ...]
    region_mappings: tuple[RegionParallelMapping, ...]

    def __post_init__(self):
        if min(self.pp_degree, self.dp, self.micro_batch_size, self.num_microbatches, self.global_batch_size) < 1:
            raise ValueError('Hybrid training dimensions must be positive')
        if self.global_batch_size != self.dp * self.micro_batch_size * self.num_microbatches:
            raise ValueError('Hybrid structures must preserve the complete global training batch')
        if (len(self.physical_stage_ranks) != self.pp_degree
                or any(not stage for stage in self.physical_stage_ranks)):
            raise ValueError('Every physical pipeline stage requires actual allocated ranks')
        members = [rank for stage in self.physical_stage_ranks for rank in stage]
        if len(set(members)) != len(members) or set(members) != set(range(len(members))):
            raise ValueError('Physical pipeline stages must partition the actual GPU allocation')
        if not self.region_mappings or len({mapping.region_id for mapping in self.region_mappings}) != len(self.region_mappings):
            raise ValueError('Hybrid structures require distinct region parallel mappings')
        for mapping in self.region_mappings:
            if mapping.policy.dense.pp != self.pp_degree or mapping.policy.dense.world_size != len(members):
                raise ValueError('Every region mapping must cover the transformed physical allocation')
            groups = dict(mapping.process_groups)['pp']
            for stage, ranks in enumerate(self.physical_stage_ranks):
                if {group.ranks[stage] for group in groups} != set(ranks):
                    raise ValueError('Region pipeline group ordering differs from its physical stage ownership')


@dataclass(frozen=True)
class EffectSignature:
    affected_contracts: frozenset[str]
    affected_process_groups: frozenset[str]
    affected_tensor_lifetimes: frozenset[str]
    affected_readiness_events: frozenset[str]
    affected_compute_resources: frozenset[str]
    affected_communication_resources: frozenset[str]
    affected_memory_intervals: frozenset[str]
    affected_profiling_keys: frozenset[str]


@dataclass(frozen=True)
class PipelinePlacement:
    pp_cuts: tuple[int, ...]
    vp_assignment: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionPolicy:
    microbatches: int
    recompute_regions: frozenset[int]

    def __post_init__(self) -> None:
        if self.microbatches < 1:
            raise ValueError("microbatches must be positive")


@dataclass(frozen=True)
class HardwareMapping:
    """逻辑设备到物理设备的固定映射。"""

    assignments: tuple[tuple[str, str], ...] = ()
    topology: str = "unspecified"

    def __post_init__(self) -> None:
        keys = [key for key, _ in self.assignments]
        if len(keys) != len(set(keys)):
            raise ValueError("hardware mapping contains duplicate logical devices")

    @classmethod
    def from_mapping(cls, assignments: Mapping[str, str], topology: str = "unspecified") -> "HardwareMapping":
        return cls(tuple(sorted((str(key), str(value)) for key, value in assignments.items())), topology)


@dataclass(frozen=True)
class ParallelProgram:
    domains: tuple[ParallelDomain, ...]
    boundaries: tuple[BoundaryProgram, ...]
    pipeline: PipelinePlacement
    execution: ExecutionPolicy
    estimated_step_us: float | None = None
    estimated_peak_memory: int | None = None
    memory_policy: MemoryExecutionPolicy = MemoryExecutionPolicy()
    hardware_mapping: HardwareMapping = HardwareMapping()
    region_modes: tuple = ()

    @property
    def domains_and_modes(self) -> tuple[tuple[ParallelDomain, ...], tuple]:
        return self.domains, self.region_modes
