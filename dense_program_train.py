import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import megatron.core

SPECIFICATION = json.loads(Path(__file__).with_name('dense_program.json').read_text())
SOURCE_ROOT = Path(SPECIFICATION['source_root']) / 'megatron/core'
megatron.core.__path__.append(str(SOURCE_ROOT))

import global_megatron_train as training
from megatron.core.frontier.megatron.dense_partition import DenseRegionCut, install_dense_region_partition, region_ids
from megatron.core.frontier.runtime.typed_p2p import TypedBoundaryBinding, install_typed_dense_p2p


plan = training.PLAN
if (plan.model not in ('Qwen3-8B', 'Qwen3-14B') or (plan.tp, plan.cp, plan.pp, plan.ep, plan.dp) != (4, 1, 4, 1, 1)
        or plan.vpp != 2 or plan.recompute != 'none' or SPECIFICATION['variant'] not in ('canonical', 'frontier')):
    raise ValueError('registered dense program requires actual Qwen TP4 CP1 PP4 DP1 VPP2 eager training')
cut = DenseRegionCut(SPECIFICATION['logical_layer'], SPECIFICATION['source_pp'], SPECIFICATION['source_vp'])
index = cut.source_vp * plan.pp + cut.source_pp
if (cut.source_pp >= plan.pp - 1 or cut.source_vp >= plan.vpp
        or sum(plan.decoder_chunks[:index + 1]) - 1 != cut.logical_layer):
    raise ValueError('registered dense boundary differs from the actual virtual chunk layout')
binding = TypedBoundaryBinding(cut.logical_layer, cut.source_pp,
    (plan.sequence // plan.tp, plan.micro_batch, training.WORKLOADS[plan.model]['hidden']), plan.microbatches,
    tp_size=plan.tp, virtual_pipeline_size=plan.vpp, source_vp=cut.source_vp)
instances = []
if SPECIFICATION['variant'] == 'frontier':
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
        expected = cut.expected_regions(layout.get_layer_id_list(vp_stage=vp_stage, pp_rank=pp_rank), pp_rank, vp_stage)
        if region_ids(model.decoder) != expected or model.decoder.frontier_region_ids != expected:
            raise ValueError('actual dense region ownership differs from its compiled virtual program')


def train_start(self, context):
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories
    from megatron.core.dist_checkpointing.utils import extract_sharded_tensors_and_factories

    original_start(self, context)
    rows = []
    for model in self.models:
        checkpoint, _ = extract_sharded_tensors_and_factories(model.sharded_state_dict())
        apply_factories(checkpoint)
        rows.append({'region_ids': region_ids(model.decoder), 'checkpoint_ownership': [
            {'key': value.key, 'global_offset': value.global_offset,
             'local_shape': value.local_shape, 'global_shape': value.global_shape}
            for value in nested_values(checkpoint) if isinstance(value, ShardedTensor)]})
    training.write_json(training.OUTPUT / f'rank{self.rank}.region-program.json',
        {'variant': SPECIFICATION['variant'], 'cut': asdict(cut), 'binding': asdict(binding), 'actual_ownership': rows,
         'decoder_chunks': plan.decoder_chunks, 'virtual_pipeline_size': plan.vpp,
         'boundary_lowering': 'typed_region_NCCL' if instances or SPECIFICATION['variant'] == 'frontier' else 'native_canonical',
         'fine_grained_cut_lowering': 'constructed' if SPECIFICATION['variant'] == 'frontier' else 'not_requested'})


def step_end(self, context):
    original_step_end(self, context)
    for instance in instances:
        instance.finish_update()


def train_end(self, context):
    updates = training.REQUEST.warmup + training.REQUEST.measured
    if any(instance.completed_updates != updates for instance in instances):
        raise ValueError('actual dense virtual transport omitted a complete optimizer update')
    local_ids = [instance.program.executable_witness_key for instance in instances]
    all_ids = [None] * training.dist.get_world_size()
    training.dist.all_gather_object(all_ids, local_ids)
    identifiers = sorted({identifier for row in all_ids for identifier in row})
    if SPECIFICATION['variant'] == 'frontier' and len(identifiers) != plan.tp:
        raise ValueError('actual dense execution omitted a compiled tensor-parallel boundary endpoint')
    training.write_json(training.OUTPUT / f'rank{self.rank}.typed-boundary.json',
        {'cut': asdict(cut), 'instances': [instance.evidence() for instance in instances]})
    path = training.OUTPUT / f'rank{self.rank}.region-program.json'
    metadata = json.loads(path.read_text())
    metadata.update(boundary_program_ids=identifiers, completed_optimizer_updates=updates,
        fine_grained_cut_lowering='executed' if SPECIFICATION['variant'] == 'frontier' else 'not_requested')
    training.write_json(path, metadata)
    original_end(self, context)


if SPECIFICATION['variant'] == 'frontier':
    training.verify_decoder_ownership = verify_ownership
training.loaded_sources = loaded_sources
training.TrainingEvidence.on_train_start = train_start
training.TrainingEvidence.on_train_step_end = step_end
training.TrainingEvidence.on_train_end = train_end
training.main()
