from dataclasses import dataclass


@dataclass(frozen=True)
class DenseParallelGrid:
    tp: int
    cp: int
    pp: int
    dp: int
    gtp: int = 1

    def __post_init__(self):
        if min(self.tp, self.cp, self.pp, self.dp, self.gtp) < 1:
            raise ValueError('dense parallel sizes must be positive')
        if self.gtp != 1:
            raise ValueError('GTP requires a separately verified backend capability')

    @property
    def world_size(self):
        return self.tp * self.cp * self.pp * self.dp


@dataclass(frozen=True)
class ExpertParallelGrid:
    etp: int
    ep: int
    expert_dp: int
    expert_gtp: int = 1

    def __post_init__(self):
        if min(self.etp, self.ep, self.expert_dp, self.expert_gtp) < 1:
            raise ValueError('expert parallel sizes must be positive')
        if self.expert_gtp != 1:
            raise ValueError('expert GTP requires a separately verified backend capability')


@dataclass(frozen=True)
class GlobalParallelPolicy:
    dense: DenseParallelGrid
    expert: ExpertParallelGrid | None
    sequence_parallel: bool
    state_sharding: str
    mapping_template: str

    def __post_init__(self):
        if self.state_sharding not in ('replicated', 'distributed_optimizer', 'fsdp'):
            raise ValueError('unsupported state sharding policy')
        if self.sequence_parallel and self.dense.tp == 1:
            raise ValueError('sequence parallelism requires TP greater than one')
        if not self.mapping_template:
            raise ValueError('a physical mapping template is required')
        if self.expert and self.expert.etp * self.expert.ep * self.expert.expert_dp * self.dense.pp != self.dense.world_size:
            raise ValueError('expert grid must cover the same world as the dense grid')

    def process_group_witness(self, rank_generator, order='tp-cp-ep-dp-pp'):
        dense = rank_generator(tp=self.dense.tp, cp=self.dense.cp, ep=1,
                               dp=self.dense.dp, pp=self.dense.pp, order=order)
        groups = {axis: tuple(tuple(row) for row in dense.get_ranks(axis))
                  for axis in ('tp', 'cp', 'pp', 'dp', 'dp-cp')}
        if self.expert:
            expert = rank_generator(tp=self.expert.etp, ep=self.expert.ep, cp=1,
                                    dp=self.expert.expert_dp, pp=self.dense.pp, order=order)
            expert_pp = tuple(tuple(row) for row in expert.get_ranks('pp'))
            if expert_pp != groups['pp']:
                raise ValueError('Megatron dense and expert pipeline groups differ')
            for axis, token in (('etp', 'tp'), ('ep', 'ep'), ('expert_dp', 'dp')):
                groups[axis] = tuple(tuple(row) for row in expert.get_ranks(token))
        expected = set(range(self.dense.world_size))
        for axis, rows in groups.items():
            flat = tuple(rank for row in rows for rank in row)
            if set(flat) != expected or len(flat) != len(expected):
                raise ValueError(f'{axis} process groups do not partition the actual world')
        return groups
