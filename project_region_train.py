import hashlib
import json
import os
import socket
from dataclasses import asdict
from pathlib import Path

import megatron.core

SPECIFICATION = json.loads(Path(__file__).with_name('project_program.json').read_text())
SOURCE_ROOT = Path(SPECIFICATION['source_root']) / 'megatron/core'
megatron.core.__path__.append(str(SOURCE_ROOT))

import global_megatron_train as training
from megatron.core.frontier.global_search.policy import DenseParallelGrid, GlobalParallelPolicy
from megatron.core.frontier.megatron.project_partition import activate_project_rows, install_project_partition
from megatron.core.frontier.runtime.projected_payload import ProjectedPayloadPlan
from megatron.core.frontier.runtime.projected_p2p import install_projected_p2p
from megatron.core.parallel_state import RankGenerator


plan = training.PLAN
if plan.model != 'Qwen3-8B' or plan.cp != 1 or plan.vpp != 1 or plan.tp <= 1 or plan.dp != 1:
    raise ValueError('registered model transition requires actual dense TP>1 CP1 DP1 VPP1')
variant = SPECIFICATION['variant']
layer = plan.decoder_chunks[0] - 1
shard_shape = (plan.sequence // plan.tp, plan.micro_batch, training.WORKLOADS[plan.model]['hidden'])
policy = GlobalParallelPolicy(DenseParallelGrid(plan.tp, plan.cp, plan.pp, plan.dp), None,
                              True, 'distributed_optimizer', 'actual_device_order')
groups = policy.process_group_witness(RankGenerator, plan.rank_order)
payload_plan = ProjectedPayloadPlan(groups['tp'][0], groups['tp'][1], shard_shape,
                                    min(plan.pp, plan.microbatches), plan.microbatches, layer, variant,
                                    reduction_ring=tuple(SPECIFICATION['reduction_ring']))
if variant == 'project' and os.environ.get('NCCL_ALGO') != 'Ring':
    raise ValueError('registered BF16 PROJECT capability requires its certified NCCL Ring arithmetic')
os.environ['TRITON_CACHE_DIR'] = str(training.OUTPUT / 'triton-cache' / ('rank' + os.environ['RANK']))
group_state = {'handoff': None}
install_project_partition(layer, 0, variant, shard_shape)
instances = install_projected_p2p(payload_plan, group_state)
original_start = training.TrainingEvidence.on_train_start
original_step_start = training.TrainingEvidence.on_train_step_start
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
        expected = list(layout.get_layer_id_list(vp_stage=vp_stage, pp_rank=pp_rank))
        if pp_rank == 1:
            expected.insert(0, layer)
            consumer = model.decoder.layers[0]
            if consumer.frontier_region_kind != 'MLP_RESOLVE_BDA' or tuple(consumer.parameters()):
                raise ValueError('consumer must execute the actual parameter-free BDA region')
        actual = [item.layer_number - 1 for item in model.decoder.layers]
        if actual != expected:
            raise ValueError('actual row/BDA model partition differs from its registered program')


def train_start(self, context):
    original_start(self, context)
    training.dist.barrier(group=self.groups['tp'])
    training.dist.barrier(group=self.groups['pp'])
    group_state['handoff'] = training.dist.new_group(ranks=list(payload_plan.source_ranks + payload_plan.destination_ranks),
                                                    backend='nccl')
    if self.rank in payload_plan.source_ranks + payload_plan.destination_ranks:
        actual_tp = tuple(training.dist.get_process_group_ranks(self.groups['tp']))
        expected_tp = payload_plan.source_ranks if self.rank in payload_plan.source_ranks else payload_plan.destination_ranks
        if actual_tp != expected_tp:
            raise ValueError('actual source/destination TP groups differ from the compiled transition')
        training.dist.barrier(group=group_state['handoff'])
        if variant == 'project' and self.rank == payload_plan.source_ranks[0]:
            path = Path(os.environ['NCCL_DEBUG_FILE'].replace('%h', socket.gethostname()).replace('%p', str(os.getpid())))
            content = path.read_text()
            if not all(f'Channel {channel:02}/04 : 0 1 2 3' in content for channel in range(4)):
                raise ValueError('current source NCCL channels differ from the registered BF16 reduction ring')
    diagnostic_directory = training.OUTPUT if variant == 'project' and training.REQUEST.mode == 'validate' else None
    group_state['diagnostic_directory'] = diagnostic_directory
    activate_project_rows(self.models, diagnostic_directory)
    if training.REQUEST.mode == 'profile':
        group_state['profiling_origin'] = training.torch.cuda.Event(enable_timing=True)
    training.write_json(training.OUTPUT / f'rank{self.rank}.project-program.json',
                        {'variant': variant, 'logical_layer': layer, 'global_policy': asdict(policy),
                         'process_groups': groups, 'typed_payload': asdict(payload_plan),
                         'fine_grained_cut_lowering': 'constructed',
                         'checkpoint_ownership': 'native global layer keys, parameter-free consumer BDA'})


def step_start(self, context):
    original_step_start(self, context)
    if training.REQUEST.mode == 'profile':
        group_state['profiling_origin'].record(training.torch.cuda.current_stream())


def step_end(self, context):
    original_step_end(self, context)
    for runtime in instances.values():
        runtime.finish_update()


def train_end(self, context):
    evidence = [runtime.evidence() for runtime in instances.values()]
    for row in evidence:
        if row['completed_updates'] != training.REQUEST.warmup + training.REQUEST.measured:
            raise ValueError('model transition did not execute every complete optimizer update')
    all_rows = [None] * training.dist.get_world_size()
    training.dist.all_gather_object(all_rows, evidence)
    ids = sorted({row['boundary_program_id'] for rank_rows in all_rows for row in rank_rows})
    if len(ids) != 1:
        raise ValueError('all endpoints must execute the same complete model boundary witness')
    training.write_json(training.OUTPUT / f'rank{self.rank}.project-boundary.json',
                        {'variant': variant, 'fine_grained_cut_lowering': 'executed',
                         'boundary_lowering': 'resolve_on_handoff_model_bf16' if variant == 'project' else 'canonical_row_boundary',
                         'boundary_program_id': ids[0], 'endpoints': evidence})
    original_end(self, context)


training.verify_decoder_ownership = verify_ownership
training.loaded_sources = loaded_sources
training.TrainingEvidence.on_train_start = train_start
training.TrainingEvidence.on_train_step_start = step_start
training.TrainingEvidence.on_train_step_end = step_end
training.TrainingEvidence.on_train_end = train_end
training.main()
