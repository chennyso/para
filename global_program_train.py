import json
from pathlib import Path

import global_megatron_train as training

SPECIFICATION = json.loads(Path(__file__).with_name('global_program.json').read_text())
original_configuration = training.configuration
original_start = training.TrainingEvidence.on_train_start


def configuration():
    cfg = original_configuration()
    cfg.model.vocab_size = SPECIFICATION['vocab_size']
    cfg.model.should_pad_vocab = False
    cfg.dataset.num_workers = SPECIFICATION.get('data_workers', cfg.dataset.num_workers)
    if 'recompute_modules' in SPECIFICATION:
        if training.PLAN.recompute != 'selective' or not set(SPECIFICATION['recompute_modules']) <= {'mlp', 'core_attn'}:
            raise ValueError('The global program requires supported selective recomputation modules')
        cfg.model.recompute_modules = SPECIFICATION['recompute_modules']
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    if not cfg.optimizer.use_distributed_optimizer or not cfg.ddp.use_distributed_optimizer:
        raise ValueError('global program requires native distributed optimizer state sharding')
    if cfg.model.vocab_size % (cfg.model.make_vocab_size_divisible_by * training.PLAN.tp):
        raise ValueError('pinned checkpoint vocabulary is incompatible with the selected tensor parallel degree')
    return cfg


def start(self, context):
    original_start(self, context)
    cfg = context.state.cfg
    training.write_json(training.OUTPUT / f'rank{self.rank}.global-program.json',
        {'plan': training.PLAN.model_dump(mode='json'), 'dp': training.PLAN.dp,
         'distributed_optimizer': cfg.optimizer.use_distributed_optimizer,
         'optimizer_sharding_group_size': training.dist.get_world_size(self.groups['dp_cp']),
         'optimizer_offload_fraction': training.REQUEST.effective_optimizer_offload_fraction,
         'overlap_grad_reduce': cfg.ddp.overlap_grad_reduce,
         'overlap_param_gather': cfg.ddp.overlap_param_gather,
         'tp_comm_overlap': cfg.model.tp_comm_overlap,
         'cuda_graph_impl': cfg.model.cuda_graph_impl,
         'recompute_granularity': cfg.model.recompute_granularity,
         'recompute_modules': cfg.model.recompute_modules,
         'vocab_size': cfg.model.vocab_size, 'precision': str(cfg.model.params_dtype),
         'actual_layer_ids': [[layer.layer_number - 1 for layer in model.decoder.layers] for model in self.models],
         'fine_grained_cut_lowering': 'not_requested', 'boundary_lowering': 'native_canonical'})


training.configuration = configuration
training.TrainingEvidence.on_train_start = start
training.main()
