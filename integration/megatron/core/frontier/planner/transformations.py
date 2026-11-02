from dataclasses import dataclass, replace

from megatron.core.frontier.ir.program import HybridStructure, RegionParallelMapping
from megatron.core.frontier.ir.region import RegionKind


@dataclass(frozen=True)
class RegionRepartition:
    region_class: RegionKind
    old_parallel_mapping: tuple[RegionParallelMapping, ...]
    new_parallel_mapping: tuple[RegionParallelMapping, ...]

    def __post_init__(self):
        old = {mapping.region_id: mapping for mapping in self.old_parallel_mapping}
        new = {mapping.region_id: mapping for mapping in self.new_parallel_mapping}
        if (not old or set(old) != set(new) or len(old) != len(self.old_parallel_mapping)
                or len(new) != len(self.new_parallel_mapping)):
            raise ValueError('RegionRepartition must replace the same distinct region identities atomically')
        for identifier, original in old.items():
            replacement = new[identifier]
            if (original.region_class is not self.region_class or replacement.region_class is not self.region_class
                    or original.policy.dense.pp != replacement.policy.dense.pp):
                raise ValueError('RegionRepartition preserves physical PP and targets one declared semantic class')


@dataclass(frozen=True)
class PipelineRescale:
    old_pp: int
    new_pp: int
    compatible_region_factorizations: tuple[RegionParallelMapping, ...]
    physical_stage_ranks: tuple[tuple[int, ...], ...]
    dp: int
    micro_batch_size: int
    num_microbatches: int

    def __post_init__(self):
        if min(self.old_pp, self.new_pp, self.dp, self.micro_batch_size, self.num_microbatches) < 1:
            raise ValueError('PipelineRescale requires positive complete atomic training dimensions')
        if self.old_pp == self.new_pp:
            raise ValueError('PipelineRescale requires a changed physical PP degree')


class TransformLibrary:
    def apply(self, structure, action):
        if isinstance(action, RegionRepartition):
            current = {mapping.region_id: mapping for mapping in structure.region_mappings}
            for mapping in action.old_parallel_mapping:
                if current.get(mapping.region_id) != mapping:
                    raise ValueError('RegionRepartition was generated for a different current structure')
            replacements = {mapping.region_id: mapping for mapping in action.new_parallel_mapping}
            result = replace(structure, region_mappings=tuple(replacements.get(mapping.region_id, mapping)
                for mapping in structure.region_mappings))
        elif isinstance(action, PipelineRescale):
            if action.old_pp != structure.pp_degree:
                raise ValueError('PipelineRescale was generated for a different current PP degree')
            if {mapping.region_id for mapping in action.compatible_region_factorizations} != {
                    mapping.region_id for mapping in structure.region_mappings}:
                raise ValueError('PipelineRescale must regenerate every existing region mapping')
            result = HybridStructure(action.new_pp, action.dp, action.micro_batch_size,
                action.num_microbatches, structure.global_batch_size, action.physical_stage_ranks,
                action.compatible_region_factorizations)
        else:
            raise TypeError('Only RegionRepartition and PipelineRescale are registered outer structure edits')
        original_ranks = {rank for stage in structure.physical_stage_ranks for rank in stage}
        final_ranks = {rank for stage in result.physical_stage_ranks for rank in stage}
        if final_ranks != original_ranks or result.global_batch_size != structure.global_batch_size:
            raise ValueError('Structural transformations must preserve the GPU allocation and training workload')
        return result
