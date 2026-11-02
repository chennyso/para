import math
from dataclasses import dataclass

from megatron.core.frontier.ir.memory import MemoryExecutionPolicy
from megatron.core.frontier.ir.program import HardwareMapping


@dataclass(frozen=True)
class VirtualChunk:
    vp_stage: int
    pp_rank: int
    region_ids: tuple[str, ...]
    boundary_ids: tuple[str, ...] = ()
    forward_us: float = 0.0
    backward_us: float = 0.0

    def __post_init__(self):
        if self.vp_stage < 0 or self.pp_rank < 0 or not self.region_ids:
            raise ValueError('virtual chunks require a valid coordinate and nonempty regions')
        if len(set(self.region_ids)) != len(self.region_ids):
            raise ValueError('virtual chunk contains duplicate regions')
        if any(not math.isfinite(value) or value < 0 for value in (self.forward_us, self.backward_us)):
            raise ValueError('virtual chunk profiles must be finite and nonnegative')


@dataclass(frozen=True)
class PipelineProgram:
    pp_size: int
    vp_size: int
    chunks: tuple[VirtualChunk, ...]
    cuts: tuple[str, ...]
    boundary_alternatives: tuple
    memory_policy: MemoryExecutionPolicy
    physical_mapping: HardwareMapping

    def __post_init__(self):
        if min(self.pp_size, self.vp_size) < 1:
            raise ValueError('PP and VPP sizes must be positive')
        expected = {(vp, pp) for vp in range(self.vp_size) for pp in range(self.pp_size)}
        actual = {(chunk.vp_stage, chunk.pp_rank) for chunk in self.chunks}
        if actual != expected or len(self.chunks) != len(expected):
            raise ValueError('one virtual chunk is required for every PP/VPP coordinate')
        regions = tuple(region for chunk in self.chunks for region in chunk.region_ids)
        if len(set(regions)) != len(regions):
            raise ValueError('each region must be owned by exactly one virtual chunk')
        if len(self.cuts) != len(self.chunks) - 1:
            raise ValueError('pipeline cuts must describe every virtual chunk boundary')

    def verify_coverage(self, ordered_regions):
        ordered = sorted(self.chunks, key=lambda chunk: (chunk.vp_stage, chunk.pp_rank))
        actual = tuple(region for chunk in ordered for region in chunk.region_ids)
        if actual != tuple(ordered_regions):
            raise ValueError('pipeline program does not preserve full model region order')

    def decoder_layout(self):
        ordered = sorted(self.chunks, key=lambda chunk: (chunk.vp_stage, chunk.pp_rank))
        if any(not region.startswith('decoder:') for chunk in ordered for region in chunk.region_ids):
            raise ValueError('this native lowering requires complete registered decoder regions')
        layout = [['decoder'] * len(chunk.region_ids) for chunk in ordered]
        layout[0].insert(0, 'embedding')
        layout[-1].append('loss')
        return layout
