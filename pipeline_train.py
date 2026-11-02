import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import megatron.core

megatron.core.__path__.append('/workspace/frontier/validation-code/megatron/core')

import global_megatron_train as training
from megatron.core.frontier.global_search.policy import DenseParallelGrid, ExpertParallelGrid, GlobalParallelPolicy
from megatron.core.frontier.ir.memory import MemoryExecutionPolicy
from megatron.core.frontier.ir.pipeline_program import PipelineProgram, VirtualChunk
from megatron.core.frontier.ir.program import HardwareMapping
from megatron.core.parallel_state import RankGenerator


original_start = training.TrainingEvidence.on_train_start
original_sources = training.loaded_sources


def loaded_sources():
    result = original_sources()
    for path in Path('/workspace/frontier/validation-code/megatron/core/frontier').rglob('*.py'):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def train_start(self, context):
    plan = training.PLAN
    policy = GlobalParallelPolicy(DenseParallelGrid(plan.tp, plan.cp, plan.pp, plan.dp),
                                  None if plan.ep == 1 and training.WORKLOADS[plan.model]['experts'] is None else
                                  ExpertParallelGrid(plan.tp, plan.ep, plan.world // (plan.tp * plan.ep * plan.pp)),
                                  plan.tp > 1, 'distributed_optimizer', 'actual_device_order')
    groups = policy.process_group_witness(RankGenerator, plan.rank_order)
    chunks = []
    offset = 0
    for index, count in enumerate(plan.decoder_chunks):
        regions = tuple(f'decoder:{layer}' for layer in range(offset, offset + count))
        chunks.append(VirtualChunk(index // plan.pp, index % plan.pp, regions))
        offset += count
    memory = MemoryExecutionPolicy('distributed_optimizer', plan.recompute, 'none',
                                  'cpu' if training.REQUEST.effective_optimizer_offload_fraction else 'none', 'disabled')
    program = PipelineProgram(plan.pp, plan.vpp, tuple(chunks),
                              tuple(chunk.region_ids[-1] for chunk in chunks[:-1]), (), memory,
                              HardwareMapping(tuple((f'rank:{node * 8 + local}', f'g{node + 5}:gpu:{gpu}')
                                                    for node, order in enumerate(plan.device_order)
                                                    for local, gpu in enumerate(order)), 'g5-g6'))
    program.verify_coverage(tuple(f'decoder:{layer}' for layer in range(plan.num_layers)))
    assert program.decoder_layout() == plan.layout()
    # 同时核验当前实例使用的原生 PP/VPP layout。
    actual_layout = context.state.cfg.model.pipeline_model_parallel_layout
    for chunk in chunks:
        expected_layers = [int(region.removeprefix('decoder:')) for region in chunk.region_ids]
        assert actual_layout.get_layer_id_list(vp_stage=chunk.vp_stage, pp_rank=chunk.pp_rank) == expected_layers
    assert context.state.cfg.ddp.use_distributed_optimizer
    training.write_json(training.OUTPUT / f'rank{training.dist.get_rank()}.pipeline-program.json',
                        {'global_policy': asdict(policy), 'process_group_witness': groups,
                         'pipeline_program': asdict(program), 'boundary_lowering': 'native_canonical',
                         'fine_grained_cut_lowering': 'requires_transport_adapter'})
    original_start(self, context)


training.TrainingEvidence.on_train_start = train_start
training.loaded_sources = loaded_sources
training.main()
