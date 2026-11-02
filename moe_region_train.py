import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import megatron.core

SPECIFICATION = json.loads(Path(__file__).with_name('moe_program.json').read_text())
SOURCE_ROOT = Path(SPECIFICATION['source_root']) / 'megatron/core'
megatron.core.__path__.append(str(SOURCE_ROOT))

import global_megatron_train as training
import global_moe_validation as native_routing_validation
from megatron.core.frontier.global_search.policy import DenseParallelGrid, ExpertParallelGrid, GlobalParallelPolicy
from megatron.core.frontier.megatron.moe_partition import MoERouterCut, install_moe_router_partition, moe_region_ids
from megatron.core.frontier.megatron.moe_validation import install_region_routing_validation
from megatron.core.frontier.runtime.moe_payload import MoERouterPayloadBinding
from megatron.core.frontier.runtime.moe_compose_payload import MoEComposeGeometry
from megatron.core.frontier.runtime.moe_compose_p2p import install_composed_moe_p2p
from megatron.core.frontier.runtime.typed_p2p import install_static_payload_p2p
from megatron.core.parallel_state import RankGenerator


plan = training.PLAN
if (plan.model, plan.tp, plan.cp, plan.pp, plan.ep, plan.dp, plan.rank_order) != (
        'Qwen3-30B-A3B', 1, 2, 4, 4, 2, 'tp-cp-ep-dp-pp') or plan.vpp not in (1, 2, 3):
    raise ValueError('registered MoE boundary requires actual Qwen30 TP1/CP2/PP4/EP4/DP2 eager training')
policy = GlobalParallelPolicy(DenseParallelGrid(plan.tp, plan.cp, plan.pp, plan.dp),
                              ExpertParallelGrid(1, plan.ep, 1), False, 'distributed_optimizer', 'actual_device_order')
groups = policy.process_group_witness(RankGenerator, plan.rank_order)
source_pp = SPECIFICATION.get('source_pp', 0)
source_vp = SPECIFICATION.get('source_vp', 0)
if not 0 <= source_pp < plan.pp - 1 or not 0 <= source_vp < plan.vpp:
    raise ValueError('registered MoE boundary must join consecutive physical stages in the same virtual chunk')
source_experts, destination_experts = groups['ep'][source_pp:source_pp + 2]
context_rank = {rank: coordinate for group in groups['cp'] for coordinate, rank in enumerate(group)}
chunk_index = source_vp * plan.pp + source_pp
cut = MoERouterCut(sum(plan.decoder_chunks[:chunk_index + 1]) - 1, source_pp, SPECIFICATION['variant'], source_vp)
if cut.logical_layer != SPECIFICATION['logical_layer']:
    raise ValueError('actual decoder chunks differ from the registered logical boundary')
binding = MoERouterPayloadBinding(cut.logical_layer, cut.source_pp,
                                  (plan.sequence // plan.cp, plan.micro_batch, training.WORKLOADS[plan.model]['hidden']),
                                  plan.microbatches, 128, 8, plan.cp, source_experts, destination_experts,
                                  tuple(context_rank[rank] for rank in source_experts), cut.variant,
                                  compose_receiver_token_capacity=SPECIFICATION.get('receiver_token_capacity'),
                                  virtual_pipeline_size=plan.vpp, source_vp=source_vp,
                                  compose_control_encoding=SPECIFICATION.get('control_encoding', 'int64_carrier'))
install_region_routing_validation(native_routing_validation)
group_state = {}
if cut.variant == 'compose':
    geometry = MoEComposeGeometry(binding.native_shape, binding.num_experts, binding.topk, len(source_experts),
                                  binding.compose_receiver_token_capacity, binding.probability_dtype, binding.compose_control_encoding)
    instances = install_composed_moe_p2p(binding, geometry, group_state)
    install_moe_router_partition(cut, binding, instances)
else:
    install_moe_router_partition(cut, binding)
    instances = install_static_payload_p2p(binding)
original_start = training.TrainingEvidence.on_train_start
original_step_end = training.TrainingEvidence.on_train_step_end
original_end = training.TrainingEvidence.on_train_end
original_sources = training.loaded_sources


def loaded_sources():
    result = original_sources()
    for path in (SOURCE_ROOT / 'frontier').rglob('*.py'):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_ownership(models, layout, pp_rank):
    for vp_stage, model in enumerate(models):
        expected = cut.expected_regions(layout.get_layer_id_list(vp_stage=vp_stage, pp_rank=pp_rank), pp_rank, vp_stage)
        if moe_region_ids(model.decoder) != expected or model.decoder.frontier_region_ids != expected:
            raise ValueError('actual MoE parameter regions differ from compiled ownership')


def train_start(self, context):
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories
    from megatron.core.dist_checkpointing.utils import extract_sharded_tensors_and_factories

    original_start(self, context)
    training.dist.barrier(group=self.groups['pp'])
    if cut.variant == 'compose':
        group_state['handoff'] = training.dist.new_group(ranks=list(source_experts + destination_experts), backend='nccl')
        if self.rank in source_experts + destination_experts:
            training.dist.barrier(group=group_state['handoff'])
            if len(instances) != 1:
                raise ValueError('actual source/expert endpoint omitted its shared COMPOSE communicator')
    actual_experts = tuple(training.dist.get_process_group_ranks(training.parallel_state.get_expert_model_parallel_group()))
    expected_experts = next(group for group in groups['ep'] if self.rank in group)
    if actual_experts != expected_experts or training.parallel_state.get_context_parallel_rank() != context_rank[self.rank]:
        raise ValueError('actual expert/context groups differ from typed MoE token ownership')
    ownership = []
    for model in self.models:
        checkpoint, _ = extract_sharded_tensors_and_factories(model.sharded_state_dict())
        apply_factories(checkpoint)
        metadata = [{'key': value.key, 'global_offset': value.global_offset, 'local_shape': value.local_shape,
                     'global_shape': value.global_shape} for value in nested_values(checkpoint) if isinstance(value, ShardedTensor)]
        ownership.append({'regions': moe_region_ids(model.decoder), 'checkpoint_ownership': metadata,
                          'parameters': [{'logical_layer': layer.layer_number - 1,
                                          'region_kind': getattr(layer, 'frontier_region_kind', 'MOE_FULL_LAYER'),
                                          'keys': [name for name, _ in layer.named_parameters()],
                                          'elements': sum(parameter.numel() for parameter in layer.parameters())}
                                         for layer in model.decoder.layers]})
    training.write_json(training.OUTPUT / f'rank{self.rank}.moe-region-program.json',
                        {'cut': asdict(cut), 'binding': asdict(binding), 'actual_ownership': ownership,
                         'fine_grained_cut_lowering': 'constructed', 'global_policy': asdict(policy), 'process_groups': groups})


def step_end(self, context):
    original_step_end(self, context)
    for instance in instances:
        instance.finish_update()


def train_end(self, context):
    evidence = [instance.evidence() for instance in instances]
    for row in evidence:
        if row['completed_updates'] != training.REQUEST.warmup + training.REQUEST.measured:
            raise ValueError('actual MoE transport omitted an optimizer update or its adjoint')
    local_ids = [instance.program.executable_witness_key for instance in instances]
    all_ids = [None] * training.dist.get_world_size()
    training.dist.all_gather_object(all_ids, local_ids)
    identifiers = sorted({identifier for rank_ids in all_ids for identifier in rank_ids})
    if len(identifiers) != len(source_experts):
        raise ValueError('actual MoE execution omitted a compiled expert/context PP boundary')
    training.write_json(training.OUTPUT / f'rank{self.rank}.moe-boundary.json',
                        {'variant': cut.variant, 'fine_grained_cut_lowering': 'executed',
                         'boundary_lowering': {'canonical': 'native_moe_attention_PP_cut',
                                               'move': 'native_moe_router_MOVE_typed_PP',
                                               'compose': 'native_moe_COMPOSE_PP_EP_direct_routing'}[cut.variant],
                         'boundary_program_ids': identifiers, 'endpoints': evidence})
    original_end(self, context)


training.loaded_sources = loaded_sources
training.verify_decoder_ownership = verify_ownership
training.TrainingEvidence.on_train_start = train_start
training.TrainingEvidence.on_train_step_end = step_end
training.TrainingEvidence.on_train_end = train_end
training.main()
