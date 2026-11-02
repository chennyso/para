from dataclasses import dataclass

from megatron.core.frontier.ir.pipeline_program import PipelineProgram, VirtualChunk
from megatron.core.frontier.ir.program import HardwareMapping
from megatron.core.frontier.pipeline.simulator import simulate_native_program


@dataclass(frozen=True)
class SimResult:
    total_step_us: float
    max_stage_load_us: float
    peak_memory_bytes: int
    bubble_fraction: float
    stage_ready_times: tuple[float, ...]
    events: tuple
    native_prediction: object
    profile_id: str
    prediction_scope: str = 'measured_decoder_regions_and_PP'


class PipelineSimulator:
    def __init__(self, num_stages, num_microbatches, *, profiles=None, pp_comm_us=None,
                 profile_id=None, profile_memory_policy=None, static_bytes_by_stage=None):
        if num_stages < 1 or num_microbatches < 1:
            raise ValueError('simulation requires positive stage and microbatch counts')
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.profiles = profiles
        self.pp_comm_us = pp_comm_us
        self.profile_id = profile_id
        self.profile_memory_policy = profile_memory_policy
        self.static_bytes_by_stage = static_bytes_by_stage

    def _require_profiles(self):
        if (self.profiles is None or self.pp_comm_us is None or not self.profile_id
                or self.profile_memory_policy is None or self.static_bytes_by_stage is None):
            raise ValueError('simulation requires measured profiles, PP timing, policy and stage ownership')
        if len(self.static_bytes_by_stage) != self.num_stages or min(self.static_bytes_by_stage) < 0:
            raise ValueError('static memory ownership must cover every physical stage')

    def simulate(self, program, schedule_table):
        self._require_profiles()
        if program.pp_size != self.num_stages:
            raise ValueError('pipeline coordinates differ from the simulator stage ownership')
        if program.memory_policy != self.profile_memory_policy:
            raise ValueError('memory policy changes require matching measured region and transfer profiles')
        prediction = simulate_native_program(program, self.num_microbatches, schedule_table,
                                             self.profiles, self.pp_comm_us)
        stage_ready = []
        stage_busy = []
        for stage in range(self.num_stages):
            nodes = tuple(node for node in prediction.dag.nodes.values()
                          if node.resource.device == f'pp:{stage}' and 'GPU_COMPUTE' in node.resource.resources)
            stage_ready.append(max(prediction.intervals[node.node_id][1] for node in nodes))
            stage_busy.append(sum(node.duration_us for node in nodes))
        peak = max(static + activation for static, activation in
                   zip(self.static_bytes_by_stage, prediction.peak_activation_bytes, strict=True))
        capacity = prediction.step_us * self.num_stages
        bubble = (capacity - sum(stage_busy)) / capacity if capacity else 0.0
        if not 0 <= bubble <= 1:
            raise ValueError('scheduled compute occupancy exceeds physical stage capacity')
        return SimResult(prediction.step_us, max(stage_busy) / self.num_microbatches, peak,
                         bubble, tuple(stage_ready), tuple(prediction.dag.nodes.values()),
                         prediction, self.profile_id)

    def estimate_step_time(self, contract_sequence, pp_cuts, vpp=1, pending_transitions=(),
                           memory_policy=None, topology_mapping=None, *, pipeline_program=None,
                           schedule_table=None):
        self._require_profiles()
        if pending_transitions:
            raise ValueError('pending boundaries require an explicit executable completion DAG')
        regions = tuple(contract_sequence)
        if not regions or len(set(regions)) != len(regions):
            raise ValueError('simulation requires distinct measured region identities')
        if pipeline_program is None:
            if vpp != 1:
                raise ValueError('VPP simulation requires a complete VirtualChunk composition')
            cuts = tuple(pp_cuts)
            if len(cuts) + 1 != self.num_stages or tuple(sorted(set(cuts))) != cuts:
                raise ValueError('pipeline cuts must define every physical stage exactly once')
            offsets = (0,) + tuple(cut + 1 for cut in cuts) + (len(regions),)
            if any(right <= left for left, right in zip(offsets, offsets[1:])):
                raise ValueError('pipeline cuts must create nonempty consecutive regions')
            chunks = tuple(VirtualChunk(0, stage, regions[offsets[stage]:offsets[stage + 1]])
                           for stage in range(self.num_stages))
            policy = memory_policy or self.profile_memory_policy
            mapping = topology_mapping or HardwareMapping()
            pipeline_program = PipelineProgram(self.num_stages, 1, chunks,
                                               tuple(chunk.region_ids[-1] for chunk in chunks[:-1]),
                                               (), policy, mapping)
        if pipeline_program.vp_size != vpp:
            raise ValueError('VPP degree differs from the complete pipeline program')
        pipeline_program.verify_coverage(regions)
        if schedule_table is None:
            if vpp > 1:
                raise ValueError('interleaving requires the registered native schedule table')
            schedule_table = tuple((microbatch, 0) for microbatch in range(self.num_microbatches))
        return self.simulate(pipeline_program, schedule_table)
