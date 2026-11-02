import math
import random

from core_data_models import Region
from cost.cost_model import MissingCalibration


class RegionPartitioner:
    def __init__(self, model_spec, config):
        self.model_spec = model_spec
        self.config = config
        self.layer_costs = {}

    def set_layer_costs(self, layer_costs):
        if set(layer_costs) != set(range(self.model_spec.num_layers)):
            raise MissingCalibration('Region优化需要全部真实Layer成本')
        if any(not math.isfinite(value) or value < 0 for value in layer_costs.values()):
            raise ValueError('Layer成本必须为有限非负数值')
        self.layer_costs = dict(layer_costs)

    def from_counts(self, counts):
        if len(counts) != self.config.total_vp_stages or min(counts) < 1 or sum(counts) != self.model_spec.num_layers:
            raise ValueError('Region分配必须完整覆盖全部模型层')
        offset, regions = 0, []
        for index, count in enumerate(counts):
            regions.append(Region(index, index % self.config.pp, index // self.config.pp,
                                  offset, offset + count))
            offset += count
        return regions

    def generate_uniform_regions(self):
        slots = self.config.total_vp_stages
        return self.from_counts([self.model_spec.num_layers // slots +
            (index < self.model_spec.num_layers % slots) for index in range(slots)])

    def generate_diverse_regions(self, limit=32, seed=0):
        if type(limit) is not int or limit < 1:
            raise ValueError('分区候选数量必须为正整数')
        layers, slots = self.model_spec.num_layers, self.config.total_vp_stages
        if layers < slots:
            raise ValueError('非空Region数量超过模型层数')
        uniform = tuple(region.num_layers for region in self.generate_uniform_regions())
        counts = [uniform]
        seen = {uniform}
        generator = random.Random(seed)
        for weights in ([index + 1 for index in range(slots)],
                        [slots - index for index in range(slots)]):
            excess = layers - slots
            allocation = [1 + int(excess * weight / sum(weights)) for weight in weights]
            for index in sorted(range(slots), key=lambda index: (-weights[index], index))[:layers - sum(allocation)]:
                allocation[index] += 1
            row = tuple(allocation)
            if row not in seen:
                seen.add(row)
                counts.append(row)
        if self.layer_costs:
            prefix = [0.0]
            for layer in range(layers):
                prefix.append(prefix[-1] + self.layer_costs[layer])
            boundaries = [0]
            for slot in range(1, slots):
                target = prefix[-1] * slot / slots
                end = min(range(boundaries[-1] + 1, layers - (slots - slot) + 1),
                          key=lambda index: abs(prefix[index] - target))
                boundaries.append(end)
            boundaries.append(layers)
            row = tuple(right - left for left, right in zip(boundaries, boundaries[1:]))
            if row not in seen:
                seen.add(row)
                counts.append(row)
        attempts = 0
        while len(counts) < limit and attempts < limit * 32:
            attempts += 1
            boundaries = [0, *sorted(generator.sample(range(1, layers), slots - 1)), layers]
            row = tuple(right - left for left, right in zip(boundaries, boundaries[1:]))
            if row not in seen:
                seen.add(row)
                counts.append(row)
        return [self.from_counts(row) for row in counts[:limit]]

    def beam_search(self, scorer, beam_width=32, memory_admission=None):
        if not self.layer_costs:
            raise MissingCalibration('缺少Region分区所需的真实Layer成本')
        if beam_width < 1:
            raise ValueError('beam_width必须为正整数')
        layers, slots = self.model_spec.num_layers, self.config.total_vp_stages
        prefix = [0.0]
        for layer in range(layers):
            prefix.append(prefix[-1] + self.layer_costs[layer])
        states = [(0, (), (0.0,) * self.config.pp)]
        for slot in range(slots):
            generated = []
            for offset, counts, costs in states:
                ends = [layers] if slot == slots - 1 else range(offset + 1, layers - (slots - slot - 1) + 1)
                for end in ends:
                    if memory_admission is not None and not memory_admission(slot, offset, end, counts):
                        continue
                    next_costs = list(costs)
                    next_costs[slot % self.config.pp] += prefix[end] - prefix[offset]
                    next_counts = counts + (end - offset,)
                    partial = (end, next_counts, tuple(next_costs))
                    score = scorer(self.from_counts(next_counts)) if end == layers and slot == slots - 1 else max(next_costs)
                    generated.append((score, partial))
            if not generated:
                raise ValueError('没有满足显存与模型顺序要求的分区')
            generated.sort(key=lambda item: (item[0], item[1][1]))
            states = [item[1] for item in generated[:beam_width]]
        return [self.from_counts(counts) for _, counts, _ in states]

    def generate_load_balanced_regions(self, layer_costs=None):
        if layer_costs is not None:
            self.set_layer_costs(layer_costs)
        def score(regions):
            totals = [0.0] * self.config.pp
            for region in regions:
                totals[region.pp_rank] += sum(self.layer_costs[layer]
                    for layer in range(region.layer_start, region.layer_end))
            return max(totals)
        return self.beam_search(score)[0]

    def move_boundary(self, regions, boundary_id, offset):
        counts = [region.num_layers for region in regions]
        counts[boundary_id] += offset
        counts[boundary_id + 1] -= offset
        return self.from_counts(counts)

    def split_region(self, regions, region_id, cut):
        if self.config.total_vp_stages != len(regions) + 1:
            raise ValueError('分割Region必须先更新PP/VPP配置')
        counts = [region.num_layers for region in regions]
        original = counts.pop(region_id)
        counts[region_id:region_id] = [cut, original - cut]
        return self.from_counts(counts)

    def merge_regions(self, regions, left, right):
        if right != left + 1 or self.config.total_vp_stages != len(regions) - 1:
            raise ValueError('合并Region必须相邻且匹配更新后的PP/VPP')
        counts = [region.num_layers for region in regions]
        counts[left:right + 1] = [counts[left] + counts[right]]
        return self.from_counts(counts)

    def estimate_region_costs(self, regions):
        if not self.layer_costs:
            raise MissingCalibration('缺少真实Layer成本')
        return {region.region_id: sum(self.layer_costs[layer] for layer in
            range(region.layer_start, region.layer_end)) for region in regions}


def create_regions_for_candidate(model_spec, config, strategy='uniform', layer_costs=None):
    partitioner = RegionPartitioner(model_spec, config)
    if strategy == 'uniform':
        return partitioner.generate_uniform_regions()
    if strategy == 'load_balanced':
        return partitioner.generate_load_balanced_regions(layer_costs)
    raise ValueError('分区策略需要提供对应的成本优化实现')
