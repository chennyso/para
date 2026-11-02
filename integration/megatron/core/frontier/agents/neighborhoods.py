from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RegionIdentifier = Annotated[str, Field(pattern=r'^(decoder:[0-9]+|layer:[0-9]+:(ATTENTION|MLP|PRE_DISPATCH|ROUTER|DISPATCH|EXPERT|COMBINE|HYBRID))$')]
ParallelDegree = Annotated[int, Field(ge=1, le=16)]


class CutWindow(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    begin_region: RegionIdentifier
    end_region: RegionIdentifier


class Neighborhood(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    evidence_ids: list[str]
    rationale: str


class GlobalParallelNeighborhood(Neighborhood):
    tp: list[ParallelDegree]
    cp: list[ParallelDegree]
    ep: list[ParallelDegree]
    etp: list[ParallelDegree]
    pp: list[ParallelDegree]
    state_sharding: list[Literal['replicated', 'distributed_optimizer', 'fsdp']]
    mapping_templates: list[str]
    domain_split_regions: list[RegionIdentifier]

    @model_validator(mode='after')
    def check_degrees(self):
        if not self.tp or not self.cp or not self.pp or not self.state_sharding or not self.mapping_templates:
            raise ValueError('global parallel neighborhood must contain complete policy dimensions')
        return self


class PipelineNeighborhood(Neighborhood):
    cut_windows: list[CutWindow]
    vpp: list[Annotated[int, Field(ge=1, le=3)]]
    chunk_templates: list[Literal['UNIFORM', 'COMPUTE_BALANCED', 'BOUNDARY_AWARE']]
    boundary_alternatives: list[str]
    recompute: list[Literal['none', 'selective', 'prefix', 'full']]
    activation_offload: list[Literal['none', 'cpu_sync', 'cpu_overlap', 'cpu_prefetch']]


class CriticEvidence(Neighborhood):
    event_class: str
    residual_us: float
    hypotheses: list[str]
    proposed_constraints: list[str]
    confidence: float = Field(ge=0, le=1)
    evidence_level: Literal['HYPOTHESIS', 'OBSERVED', 'REPEATED', 'CONFIRMED', 'REFUTED']
