from dataclasses import dataclass
from itertools import product

from megatron.core.frontier.global_search.policy import DenseParallelGrid, ExpertParallelGrid, GlobalParallelPolicy


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    attention_heads: int
    kv_heads: int
    sequence_length: int
    layers: int
    experts: int | None

    def __post_init__(self):
        if min(self.hidden_size, self.attention_heads, self.kv_heads, self.sequence_length, self.layers) < 1:
            raise ValueError('model shape must contain positive actual dimensions')
        if self.experts is not None and self.experts < 1:
            raise ValueError('expert count must be positive')


def enumerate_policies(shape, world_size, rank_generator, allowed_cp, allowed_tp=(1, 2, 4),
                       allowed_pp=(2, 4, 8), allowed_ep=(1, 2, 4, 8), allowed_etp=(1, 2),
                       state_sharding=('distributed_optimizer',)):
    if world_size not in (8, 16):
        raise ValueError('this campaign supports the reserved eight or sixteen GPUs')
    if not allowed_cp or any(value < 1 for value in (*allowed_cp, *allowed_tp, *allowed_pp, *allowed_ep, *allowed_etp)):
        raise ValueError('policy neighborhoods must contain positive supported degrees')
    candidates = []
    expert_degrees = tuple(product(allowed_ep, allowed_etp)) if shape.experts is not None else ((1, 1),)
    for tp, cp, pp, sharding in product(allowed_tp, allowed_cp, allowed_pp, state_sharding):
        if world_size % (tp * cp * pp) or shape.layers < pp:
            continue
        if shape.hidden_size % tp or shape.attention_heads % tp or shape.kv_heads % tp:
            continue
        if shape.sequence_length % (2 * cp):
            continue
        dense = DenseParallelGrid(tp, cp, pp, world_size // (tp * cp * pp))
        for ep, etp in expert_degrees:
            expert = None
            if shape.experts is not None:
                if world_size % (etp * ep * pp) or shape.experts % ep or shape.hidden_size % etp:
                    continue
                expert = ExpertParallelGrid(etp, ep, world_size // (etp * ep * pp))
            policy = GlobalParallelPolicy(dense, expert, tp > 1, sharding, 'actual_topology_required')
            witness = policy.process_group_witness(rank_generator)
            candidates.append((policy, witness))
    return tuple(candidates)
