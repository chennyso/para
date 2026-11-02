import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import megatron.core

SOURCE_ROOT = Path('/workspace/frontier/region-code-20261007-v2/megatron/core')
megatron.core.__path__.append(str(SOURCE_ROOT))

import global_megatron_train as training
from megatron.core.frontier.megatron.dense_partition import DenseRegionCut, install_dense_region_partition, region_ids
from megatron.core.frontier.runtime.typed_p2p import TypedBoundaryBinding, install_typed_dense_p2p


plan = training.PLAN
if plan.model not in ('Qwen3-8B', 'Qwen3-14B') or plan.cp != 1 or plan.vpp != 1 or plan.pp < 2:
    raise ValueError('registered dense region execution requires a dense CP1 VPP1 pipeline')
cut = DenseRegionCut(plan.decoder_chunks[0] - 1, 0)
hidden_size = training.WORKLOADS[plan.model]['hidden']
binding = TypedBoundaryBinding(cut.logical_layer, cut.source_pp,
                               (plan.sequence // plan.tp, plan.micro_batch, hidden_size), plan.microbatches,
                               tp_size=plan.tp)
install_dense_region_partition(cut)
instances = install_typed_dense_p2p(binding)
original_sources = training.loaded_sources
original_start = training.TrainingEvidence.on_train_start
original_step_end = training.TrainingEvidence.on_train_step_end
original_end = training.TrainingEvidence.on_train_end


def loaded_sources():
    result = original_sources()
    for path in (SOURCE_ROOT / 'frontier').rglob('*.py'):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_ownership(models, layout, pp_rank):
    for vp_stage, model in enumerate(models):
        expected_layers = layout.get_layer_id_list(vp_stage=vp_stage, pp_rank=pp_rank)
        expected = cut.expected_regions(expected_layers, pp_rank, vp_stage)
        actual = region_ids(model.decoder)
        if actual != expected or actual != model.decoder.frontier_region_ids:
            raise ValueError('actual fine-grained model ownership differs from its compiled region program')
        for layer in model.decoder.layers:
            kind = getattr(layer, 'frontier_region_kind', None)
            keys = tuple(name for name, _ in layer.named_parameters())
            if kind == 'ATTENTION' and any(key.startswith(('mlp.', 'pre_mlp_layernorm.')) for key in keys):
                raise ValueError('source attention region still owns consumer MLP parameters')
            if kind == 'MLP' and any(not key.startswith(('mlp.', 'pre_mlp_layernorm.')) for key in keys):
                raise ValueError('consumer MLP region owns parameters outside its semantic region')


def train_start(self, context):
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    original_start(self, context)
    rows = []
    for model in self.models:
        checkpoint = [{'key': value.key, 'global_offset': value.global_offset,
                       'local_shape': value.local_shape, 'global_shape': value.global_shape}
                      for value in nested_values(model.sharded_state_dict()) if isinstance(value, ShardedTensor)]
        rows.append({'region_ids': region_ids(model.decoder), 'checkpoint_ownership': checkpoint,
                     'parameters': [{'logical_layer': layer.layer_number - 1,
                                     'region_kind': getattr(layer, 'frontier_region_kind', 'ATTENTION_AND_MLP'),
                                     'keys': [name for name, _ in layer.named_parameters()],
                                     'elements': sum(parameter.numel() for parameter in layer.parameters())}
                                    for layer in model.decoder.layers]})
    training.write_json(training.OUTPUT / f'rank{self.rank}.region-program.json',
                        {'cut': asdict(cut), 'binding': asdict(binding), 'actual_ownership': rows,
                         'boundary_lowering': 'typed_region_NCCL', 'fine_grained_cut_lowering': 'constructed'})


def step_end(self, context):
    original_step_end(self, context)
    for instance in instances:
        instance.finish_update()


def train_end(self, context):
    for instance in instances:
        if instance.completed_updates != training.REQUEST.warmup + training.REQUEST.measured:
            raise ValueError('typed boundary did not execute every complete training update')
    training.write_json(training.OUTPUT / f'rank{self.rank}.typed-boundary.json',
                        {'cut': asdict(cut), 'instances': [instance.evidence() for instance in instances]})
    local_ids = [instance.program.executable_witness_key for instance in instances]
    all_ids = [None] * training.dist.get_world_size()
    training.dist.all_gather_object(all_ids, local_ids)
    identifiers = sorted({identity for row in all_ids for identity in row})
    if not identifiers:
        raise ValueError('real model execution did not produce a BoundaryProgram witness')
    path = training.OUTPUT / f'rank{self.rank}.region-program.json'
    metadata = json.loads(path.read_text())
    metadata.update(fine_grained_cut_lowering='executed', boundary_program_ids=identifiers,
                    completed_optimizer_updates=training.REQUEST.warmup + training.REQUEST.measured)
    training.write_json(path, metadata)
    original_end(self, context)


training.verify_decoder_ownership = verify_ownership
training.loaded_sources = loaded_sources
training.TrainingEvidence.on_train_start = train_start
training.TrainingEvidence.on_train_step_end = step_end
training.TrainingEvidence.on_train_end = train_end
training.main()
