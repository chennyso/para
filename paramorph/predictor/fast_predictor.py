from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from core_data_models import ExecutionPlan, HardwareTopology, Task, TaskType
from cost.cost_model import ExecutionCostModel, MissingCalibration
from cost.runtime_calibration import configuration
from scheduling.schedule_simulator import ScheduleCostEvaluator
from scheduling.native_schedule_lowerer import native_stage_sequence
from .profile_database import ProfileDatabase, normalized_execution, signature, plan_signature
from .graph_delta import graph_delta, profile_key
from .compute_cost import KernelCostSurface
from .communication_cost import TopologyAwareNCCLCostSurface
from .measured_modules import (annotate_measurement_semantics, collect_optimizer_local_costs,
    collect_optimizer_component_costs)


def median(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise MissingCalibration('预测需要完整的有限非负实测成本')
    return statistics.median(values)


def ranges(counts):
    offset = 0
    result = []
    for count in counts:
        result.append(list(range(offset, offset + count)))
        offset += count
    return result


class FastPredictor:
    def __init__(self, database):
        self.database = ProfileDatabase(database)
        self.compute_surface = KernelCostSurface(self.database)
        self.communication_surface = TopologyAwareNCCLCostSurface(self.database)
        self.profiles = {}
        self.baselines = {}
        self.templates = {}
        self.region_costs = {}
        self.predictions = {}
        self.replay_count = 0
        self.projector = None
        self.cost_mode = 'hybrid'
        self.context_topology = None

    def configure_projection(self, model, model_graph, topology, hardware_parameters, cost_mode='hybrid'):
        from .dag_projection import DAGProjector
        if cost_mode not in ('hybrid', 'measured_only', 'analytical_only'):
            raise ValueError('预测成本模式未注册')
        self.cost_mode = cost_mode
        self.context_topology = topology
        self.model_graph = deepcopy(model_graph)
        hardware_parameters = deepcopy(hardware_parameters)
        resource_replay = dict(hardware_parameters.get('resource_replay', {}))
        measured = [dict(compute_kind=row['benchmark']['compute_kind'],
            collective=row['benchmark']['collective'],
            physical_group_ranks=row['benchmark']['group_ranks'],
            payload_bytes=row['benchmark']['payload_bytes'], gamma=row['benchmark']['gamma'],
            evidence_ids=[row['id']], source=row['benchmark']['source'])
            for row in self.database.overlap_profiles()]
        resource_replay['compute_communication_slowdowns'] = [
            *resource_replay.get('compute_communication_slowdowns', []), *measured]
        resource_replay.pop('unmeasured_overlap_gamma', None)
        hardware_parameters['resource_replay'] = resource_replay
        self.projector = DAGProjector(model, model_graph, topology, self.compute_surface,
            self.communication_surface, hardware_parameters, self._native_stage_sequence)

    @staticmethod
    def _special_component(module_name):
        if module_name.startswith('mtp.'):
            return 'mtp'
        if module_name == 'output_layer' or module_name.endswith('.output_layer'):
            return 'output'
        if 'final_layernorm' in module_name:
            return 'final_layernorm'
        return None

    def _special_module_breakdown(self, profile):
        if profile.get('special_module_costs') is not None:
            return profile
        profile = deepcopy(profile)
        samples = defaultdict(float)
        source = Path(profile.get('source_directory', '')) / 'artifacts'
        request = profile['request']
        config = configuration(request)
        if source.is_dir():
            for rank in range(config.world_size):
                payload = json.loads((source / f'rank{rank}.paramorph-cost.json').read_text())
                for event in payload['events']:
                    if event['step'] < request['warmup']:
                        continue
                    if event['kind'] == 'special_compute' and event['operation'] == 'loss':
                        identity = event['identity']
                        key = (rank, identity['vp_stage'], 'forward', 'loss',
                               event['step'], identity['microbatch'])
                        samples[key] += event['cuda_ms']
                        continue
                    if event['kind'] != 'operator_compute':
                        continue
                    identity = event['identity']
                    component = self._special_component(identity['module_name'])
                    if component is None:
                        continue
                    phase = 'forward' if identity['phase'] == 'forward' else 'backward'
                    key = (rank, identity['vp_stage'], phase, component,
                           event['step'], identity['microbatch'])
                    samples[key] += event['cuda_ms']
                operator_path = source / f'rank{rank}.paramorph-operators.json'
                if operator_path.is_file():
                    for event in json.loads(operator_path.read_text())['backward']:
                        if event['step'] < request['warmup']:
                            continue
                        identity = event['identity']
                        component = self._special_component(identity['module_name'])
                        if component is None:
                            continue
                        key = (rank, identity['vp_stage'], 'backward', component,
                               event['step'], identity['microbatch'])
                        samples[key] += event['milliseconds']
        grouped = defaultdict(list)
        for (rank, vp, phase, component, _, _), milliseconds in samples.items():
            grouped[(rank, vp, phase, component)].append(milliseconds)
        special = {f'{rank}:{vp}:{phase}:{component}': median(values)
                   for (rank, vp, phase, component), values in grouped.items()}
        regions = ranges(request['decoder_chunks'])
        lanes = config.world_size // config.pp
        fixed = {}
        for key, row in profile['regions'].items():
            rank_text, vp_text, kind = key.split(':')
            rank, vp = int(rank_text), int(vp_text)
            phase = 'forward' if kind == 'forward_compute' else 'backward'
            region = vp * config.pp + rank // lanes
            components = {name: special.get(f'{rank}:{vp}:{phase}:{name}', 0.0)
                          for name in ('final_layernorm', 'output', 'loss', 'mtp')}
            attributed = sum(components.values())
            remainder = max(0.0, row['overhead_ms'] - attributed)
            if region == len(regions) - 1:
                components['unattributed_runtime_remainder'] = remainder
            else:
                components['region_runtime'] = remainder
            fixed[f'{rank}:{vp}:{phase}'] = dict(total_fixed_ms=row['overhead_ms'],
                sampling_stddev_ms=row['residual_ms'], components=components,
                evidence_scope='special_operator_events_plus_region_minus_layer_composite')
        profile['special_module_costs'] = special
        profile['fixed_cost_breakdown'] = fixed
        return profile

    def query_compute_cost(self, identity, analytical_spec=None, mode=None):
        return self.compute_surface.query(identity, analytical_spec=analytical_spec, mode=mode or self.cost_mode)

    def query_communication_cost(self, collective, payload_bytes, ranks, topology_identity, concurrency=(),
                                 analytical_spec=None, mode=None, dtype='bfloat16'):
        return self.communication_surface.query(collective, payload_bytes, ranks, topology_identity, concurrency,
            analytical_spec=analytical_spec, mode=mode or self.cost_mode, dtype=dtype)

    def register_overlap_profile(self, identifier, benchmark):
        self.database.add_overlap(identifier, benchmark)
        if self.projector is None:
            return
        resource_replay = self.projector.parameters.setdefault('resource_replay', {})
        rows = resource_replay.setdefault('compute_communication_slowdowns', [])
        record = dict(compute_kind=benchmark['compute_kind'], collective=benchmark['collective'],
            physical_group_ranks=benchmark['group_ranks'], payload_bytes=benchmark['payload_bytes'],
            gamma=benchmark['gamma'], evidence_ids=[identifier], source=benchmark['source'])
        if record not in rows:
            rows.append(record)

    def set_preferred_profile(self, profile_id):
        if self.projector is not None:
            self.projector.preferred_profile_id = profile_id

    def evaluate_memory_bound(self, request):
        if self.projector is None:
            raise ValueError('解析显存上界需要模型图和实际硬件配置')
        started = time.perf_counter()
        memory = self.projector.memory_bound(request)
        identity = hashlib.sha256(json.dumps(dict(kind='bootstrap_memory_bound',
            request=normalized_execution(request), decoder_chunks=request['decoder_chunks'],
            memory=memory, hardware_parameters=self.projector.parameters),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return dict(prediction_id=identity, candidate_id=request['run_id'],
            plan_signature=plan_signature(request), prediction_status=('NEEDS_PROFILE'
                if memory['status'] == 'UNAVAILABLE_PARAMETER_OWNERSHIP' else 'ANALYTICAL_MEMORY_BOUND'),
            memory=memory, evaluation_seconds=time.perf_counter() - started,
            evidence=dict(source='model_parameter_ownership_batch_schedule_and_hardware_memory_parameters'))

    def project_dag(self, request, candidate_graph, parent_plan, persist, cost_mode, started, blind=False):
        if self.projector is None:
            return self.needs_profile(request, started, ['模型图和硬件解析参数没有登记'], persist=persist)
        parent_request = parent_plan.backend_config if hasattr(parent_plan, 'backend_config') else parent_plan
        report = self.projector.project(request, candidate_graph=candidate_graph,
            parent_request=parent_request, cost_mode=cost_mode, blind=blind)
        if report['prediction_status'] == 'NEEDS_PROFILE':
            report['memory'] = self.projector.memory_bound(request, blind=blind)
            report['evidence']['memory_evaluation_scope'] = (
                'parameter_ownership_and_activation_bound_independent_of_missing_task_times')
        report['evaluation_seconds'] = time.perf_counter() - started
        report['cost_database_revision'] = self.database.cost_database_revision()
        report['prediction_id'] = hashlib.sha256(json.dumps(dict(
            projection_id=report['prediction_id'],
            parent_execution_context=parent_request,
            preferred_profile_id=self.projector.preferred_profile_id,
            cost_database_revision=report['cost_database_revision']),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if persist:
            self.database.save_prediction(report)
        return report

    def evaluate_projection(self, request, candidate_graph, parent_plan, persist=True, blind=False,
                            cost_mode=None):
        from .dag_projection import PROJECTION_REVISION
        started = time.perf_counter()
        selected_mode = cost_mode or self.cost_mode
        parent_request = parent_plan.backend_config if hasattr(parent_plan, 'backend_config') else parent_plan
        cache_identity = hashlib.sha256(json.dumps(dict(revision=PROJECTION_REVISION,
            blind=blind, preferred_profile_id=self.projector.preferred_profile_id,
            model_graph=self.model_graph, topology=self.projector.topology.devices,
            plan=plan_signature(request),
            parent_plan=plan_signature(parent_request) if parent_request is not None else None,
            parent_measurement=None if parent_request is None else {
                key: parent_request.get(key) for key in ('_measured_step_time_ms',
                    '_measurement_run_id', '_predicted_raw_step_time_ms', '_prediction_id')},
            candidate_graph=candidate_graph, cost_mode=selected_mode,
            hardware_parameters=self.projector.parameters,
            cost_database_revision=self.database.cost_database_revision()),
            sort_keys=True).encode()).hexdigest()
        cached = self.database.cached_dag_projection(cache_identity)
        if cached is not None:
            report = deepcopy(cached)
            report['evaluation_seconds'] = time.perf_counter() - started
            report['diagnosis']['prediction_cache_hit'] = True
            return report
        report = self.project_dag(request, candidate_graph, parent_plan, False,
            selected_mode, started, blind)
        report['prediction_cache_kind'] = 'complete_dag_projection'
        report['cost_database_revision'] = self.database.cost_database_revision()
        report['projection_cache_identity'] = cache_identity
        report['diagnosis']['prediction_cache_hit'] = False
        if persist:
            self.database.save_prediction(report)
        return report

    def _region_cost(self, profile, rank, vp, kind, identifiers):
        key = profile_key(profile, kind, rank, vp, identifiers)
        if key in self.region_costs:
            return self.region_costs[key]
        config = configuration(profile['request'])
        lanes = config.world_size // config.pp
        rates = [row['activation_bytes'] / row['layer_count'] for name, row in profile['regions'].items()
            if name.endswith(':forward_compute') and
            (int(name.split(':')[0]) // lanes, int(name.split(':')[1])) not in
            ((0, 0), (config.pp - 1, config.vpp - 1))]
        measured = profile['regions'][f'{rank}:{vp}:{kind}']
        forward = profile['regions'][f'{rank}:{vp}:forward_compute']
        field = 'forward_ms' if kind == 'forward_compute' else 'backward_ms'
        original = ranges(profile['request']['decoder_chunks'])[vp * config.pp + rank // lanes]
        if not rates:
            if identifiers != original:
                raise MissingCalibration('当前特殊Stage缺少可分离的逐层激活成本，需要实际候选Profiling')
            activation = forward['activation_bytes']
        else:
            rate = median(rates)
            fixed = max(0, forward['activation_bytes'] - rate * forward['layer_count'])
            activation = max(0, rate * len(identifiers) + fixed)
        workspace = measured['workspace_bytes']
        if kind == 'forward_compute':
            workspace = max(0, workspace - forward['activation_bytes'])
        phase = 'forward' if kind == 'forward_compute' else 'backward'
        cost = dict(duration=sum(profile['layers'][str(layer)][field] for layer in identifiers)
            + measured['overhead_ms'], activation=int(activation), workspace=int(workspace),
            uncertainty_ms=measured['residual_ms'], profile_key=key,
            uncertainty_key=f'{profile["id"]}:{rank}:{vp}:{kind}',
            fixed_cost_breakdown=profile.get('fixed_cost_breakdown', {}).get(f'{rank}:{vp}:{phase}'))
        self.region_costs[key] = cost
        return cost

    def _persistent(self, request, profile):
        config = configuration(request)
        current, previous = ranges(request['decoder_chunks']), ranges(profile['request']['decoder_chunks'])
        result = {}
        for rank, sequence in profile['sequences'].items():
            physical = sequence['pp_rank']
            old = [layer for vp in range(config.vpp) for layer in previous[vp * config.pp + physical]]
            new = [layer for vp in range(config.vpp) for layer in current[vp * config.pp + physical]]
            result[int(rank)] = profile['memory'][rank]['persistent'] + sum(
                profile['layers'][str(layer)]['state_bytes'] for layer in new) - sum(
                profile['layers'][str(layer)]['state_bytes'] for layer in old)
        return result

    def memory_admission(self, request, profile):
        config = configuration(request)
        counts = request['decoder_chunks']
        if len(counts) != config.pp * config.vpp or min(counts) < 1 or sum(counts) != len(profile['layers']):
            raise ValueError('显存过滤需要合法且完整的Region分区')
        persistent = self._persistent(request, profile)
        regions = ranges(counts)
        estimated, resident_counts, peak_events, peak_regions, breakdown = {}, {}, {}, {}, {}
        for rank_text, sequence in profile['sequences'].items():
            rank, physical = int(rank_text), sequence['pp_rank']
            live, peak, inflight = 0, 0, 0
            residents = {}
            maximum_inflight = 0
            for kind, vp, mb in sequence['events']:
                ids = regions[vp * config.pp + physical]
                cost = self._region_cost(profile, rank, vp, kind, ids)
                if kind == 'forward_compute':
                    live += cost['activation']
                    residents[(vp, mb)] = vp * config.pp + physical
                    inflight += 1
                    maximum_inflight = max(maximum_inflight, inflight)
                    current = live + cost['workspace']
                else:
                    current = live + cost['workspace']
                if current > peak:
                    peak = current
                    breakdown[rank] = dict(measured_persistent_and_runtime_bytes=persistent[rank],
                        saved_activations_bytes=live, kernel_workspace_bytes=cost['workspace'],
                        communication_buffers_bytes=None, allocator_reserved_bytes=None,
                        external_device_allocations_bytes=None)
                    peak_events[rank] = f'{kind}:{rank}:{vp}:{mb}'
                    peak_regions[rank] = [dict(region=region, virtual_stage=identity[0], microbatch=identity[1])
                        for identity, region in residents.items()]
                if kind == 'backward_compute':
                    live -= cost['activation']
                    del residents[(vp, mb)]
                    inflight -= 1
                if live < 0 or inflight < 0:
                    raise ValueError('显存过滤发现Activation释放顺序非法')
            if live != 0 or inflight != 0:
                raise ValueError('显存过滤发现未释放Activation')
            estimated[rank] = persistent[rank] + peak
            resident_counts[rank] = maximum_inflight
        rejected = [rank for rank, value in persistent.items()
            if value >= profile['memory'][str(rank)]['capacity']]
        return dict(admission='REJECT_OOM' if rejected else 'REQUIRES_TIMELINE_CHECK',
            persistent_per_rank_bytes=persistent, peak_per_rank_bytes=estimated,
            maximum_resident_microbatches=resident_counts, rejected_ranks=rejected,
            peak_event_per_rank=peak_events, peak_live_regions_per_rank=peak_regions,
            breakdown_per_rank=breakdown,
            memory_budget_per_rank_bytes={int(rank): row['capacity'] for rank, row in profile['memory'].items()},
            memory_headroom_per_rank_bytes={rank: profile['memory'][str(rank)]['capacity'] - peak for rank, peak in estimated.items()},
            rejection_basis='measured_local_model_states_and_runtime', full_replays=0)

    def ingest(self, profile_directory, anchor_directory=None):
        profile_directory = Path(profile_directory)
        request = json.loads((profile_directory / 'request.json').read_text())
        result = json.loads((profile_directory / 'result.json').read_text())
        if result['status'] != 'PASS':
            raise ValueError('成本采集必须完成真实训练验证')
        anchor = None
        if anchor_directory is not None:
            anchor_directory = Path(anchor_directory)
            anchor_request = json.loads((anchor_directory / 'request.json').read_text())
            anchor = json.loads((anchor_directory / 'result.json').read_text())
            if anchor['status'] != 'PASS' or not anchor['performance_eligible']:
                raise ValueError('Anchor必须完成正式真实训练测量')
            if signature(request) != signature(anchor_request) or request['decoder_chunks'] != anchor_request['decoder_chunks']:
                raise ValueError('成本采集与Anchor执行条件不兼容')
            for name in ('native_hybrid_train.py', 'hybrid_training_entry.py', 'frontier_hybrid_provider.py'):
                if result['source_hashes'][name] != anchor['source_hashes'][name]:
                    raise ValueError('成本采集与Anchor使用不同后端实现')
        config = configuration(request)
        directory = profile_directory / 'artifacts'
        payloads = [json.loads((directory / f'rank{rank}.paramorph-cost.json').read_text())
                    for rank in range(config.world_size)]
        statics = [json.loads((directory / f'rank{rank}.paramorph-static.json').read_text())
                   for rank in range(config.world_size)]
        sample_sets = defaultdict(list)
        region_sets = defaultdict(list)
        sequences, memory, state, links = {}, {}, defaultdict(list), defaultdict(list)
        for rank, payload in enumerate(payloads):
            if payload['rank'] != rank or not payload['forward_backward_coverage_verified']:
                raise ValueError('实际任务记录未覆盖全部训练Rank')
            windows = [row for row in payload['windows'] if row['step'] >= request['warmup']]
            if len(windows) < 3:
                raise MissingCalibration('成本采集至少需要三个稳定迭代')
            first = windows[0]['step']
            compute = sorted((event for event in payload['events'] if event['step'] == first
                and event['kind'] in ('forward_compute', 'backward_compute')),
                key=lambda event: event['host_start_ns'])
            sequence = [(event['kind'], event['identity']['vp_stage'], event['identity']['microbatch'])
                        for event in compute]
            expected = {(kind, vp, mb) for kind in ('forward_compute', 'backward_compute')
                for vp in range(config.vpp) for mb in range(config.num_microbatches)}
            if set(sequence) != expected or len(sequence) != len(expected):
                raise ValueError('实际Megatron调度没有完整覆盖每个Region和microbatch')
            sequences[str(rank)] = dict(pp_rank=payload['pp_rank'], events=sequence)
            for event in payload['events']:
                if event['step'] < request['warmup']:
                    continue
                identity = event['identity']
                if event['kind'] == 'layer_compute':
                    if event['operation'] == 'layer_backward' and not identity.get('backward_input_gradient_observed'):
                        continue
                    sample_sets[(identity['layer_id'], event['operation'])].append(event['cuda_ms'])
                elif event['kind'] in ('forward_compute', 'backward_compute'):
                    region_sets[(rank, identity['vp_stage'], event['kind'])].append(event)
                elif event['kind'] in ('optimizer_step', 'dp_reduce'):
                    sample_sets[(rank, event['kind'])].append(event['cuda_ms'])
            static = statics[rank]
            for layer in static['layers']:
                if 'state_bytes' not in layer:
                    raise MissingCalibration('需要实际逐层Optimizer状态显存')
                state[layer['layer_id']].append(layer['state_bytes'])
            memory[str(rank)] = dict(persistent=median([row['persistent_bytes'] for row in windows]),
                peak=max(row['peak_allocated_bytes'] for row in windows),
                capacity=min(static['device_capacity_bytes'] - static.get('non_allocator_bytes', 0), static['allocator_limit_bytes']),
                non_allocator_bytes=static.get('non_allocator_bytes', 0))
            link_payload = json.loads((directory / f'rank{rank}.paramorph-links.json').read_text())
            for row in link_payload['rows']:
                if row['role'] == 'send':
                    links[(row['source'], row['destination'], row['direction'])].append(row['milliseconds'])
        layer_rows = {}
        for layer in range(sum(request['decoder_chunks'])):
            layer_rows[str(layer)] = dict(
                forward_ms=median(sample_sets[(layer, 'layer_forward')]),
                backward_ms=median(sample_sets[(layer, 'layer_backward')]) +
                    (median(sample_sets[(layer, 'layer_recompute_forward')])
                     if sample_sets[(layer, 'layer_recompute_forward')] else 0.0),
                state_bytes=median(state[layer]))
        baseline_regions = ranges(request['decoder_chunks'])
        region_rows = {}
        for (rank, vp, kind), events in region_sets.items():
            pp_rank = sequences[str(rank)]['pp_rank']
            ids = baseline_regions[vp * config.pp + pp_rank]
            field = 'forward_ms' if kind == 'forward_compute' else 'backward_ms'
            compute_ms = sum(layer_rows[str(layer)][field] for layer in ids)
            observed = [event['cuda_ms'] for event in events]
            region_rows[f'{rank}:{vp}:{kind}'] = dict(
                overhead_ms=max(0.0, median(observed) - compute_ms),
                residual_ms=statistics.pstdev(observed),
                activation_bytes=median([event['saved_activation_bytes'] for event in events])
                    if kind == 'forward_compute' else 0,
                workspace_bytes=max(event['workspace_upper_bound_bytes'] for event in events),
                layer_count=len(ids))
        document = dict(id=request['run_id'] + (':anchored' if anchor is not None else ':raw'), request=request, anchor=anchor,
            source_hashes=result['source_hashes'], sequences=sequences, layers=layer_rows,
            regions=region_rows, memory=memory,
            links={f'{source}:{destination}:{direction}': median(values)
                   for (source, destination, direction), values in links.items()},
            optimizer={str(rank): median(sample_sets[(rank, 'optimizer_step')]) for rank in range(config.world_size)},
            gradient_finalize={str(rank): median(sample_sets[(rank, 'dp_reduce')]) for rank in range(config.world_size)},
            evidence_files={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.glob('rank*.paramorph-*.json')},
            measured_windows={str(rank): [row for row in payload['windows'] if row['step'] >= request['warmup']]
                              for rank, payload in enumerate(payloads)})
        document['hardware'] = [row['hardware'] for row in statics]
        document['source_directory'] = str(profile_directory.resolve())
        document['cost_granularity'] = 'measured_layer_region_and_pipeline_link_composite_costs'
        optimizer_local = collect_optimizer_local_costs(payloads, request)
        if optimizer_local is not None:
            document['optimizer_local_costs'] = optimizer_local
            document['optimizer_component_costs'] = collect_optimizer_component_costs(payloads, request)
        document = annotate_measurement_semantics(document)
        for rank, payload in enumerate(payloads):
            for index, event in enumerate(payload['events']):
                if event['step'] < request['warmup'] or event['kind'] != 'operator_compute':
                    continue
                identity = event['identity']
                key = dict(model=request['model'], hardware=statics[rank]['hardware'],
                    backend=result['source_hashes'], implementation=identity['implementation'],
                    inputs=identity['inputs'], operation=event['operation'], phase=identity['phase'],
                    parameter_shapes=identity['parameter_shapes'], expert_token_counts=identity.get('expert_token_counts'),
                    precision=request.get('precision', 'bfloat16'),
                    layout=dict(tp=request['tp'], cp=request['cp'], etp=request.get('etp', request['tp']), ep=request['ep']))
                if identity.get('features') is not None:
                    key.update(features=identity['features'], granularity=identity['granularity'])
                self.database.add_operator_cost(dict(identity=key, milliseconds=event['cuda_ms'],
                    evidence_id=f'{request["run_id"]}:{rank}:{index}',
                    source='actual_module_composite_cuda_event',
                    scope=identity['cost_scope'], expert_token_counts=identity.get('expert_token_counts')),
                    commit=False)
            operator_path = directory / f'rank{rank}.paramorph-operators.json'
            if operator_path.is_file():
                backward = json.loads(operator_path.read_text())['backward']
                for index, event in enumerate(backward):
                    if event['step'] < request['warmup']:
                        continue
                    identity = event['identity']
                    key = dict(model=request['model'], hardware=statics[rank]['hardware'],
                        backend=result['source_hashes'], implementation=identity['implementation'],
                        inputs=identity['inputs'], parameter_shapes=identity['parameter_shapes'],
                        expert_token_counts=identity.get('expert_token_counts'),
                        operation=event['operation'], phase='backward',
                        precision=request.get('precision', 'bfloat16'),
                        layout=dict(tp=request['tp'], cp=request['cp'], etp=request.get('etp', request['tp']), ep=request['ep']))
                    if identity.get('features') is not None:
                        key.update(features=identity['features'], granularity=identity['granularity'])
                    self.database.add_operator_cost(dict(identity=key, milliseconds=event['milliseconds'],
                        evidence_id=f'{request["run_id"]}:{rank}:operator_backward:{index}',
                        source=event['source'], scope='actual_module_output_to_input_gradient_boundary'),
                        commit=False)
        self.database.add(document)
        return document['id']

    def _native_stage_sequence(self, pp_rank, pp, vp_size, num_microbatches, group):
        """按 Megatron 交错调度重建某个物理阶段的执行事件序列。

        返回 [kind, vp, mb] 列表，kind 为 forward_compute / backward_compute。
        与 ProgramAgent.compile 使用相同的原生schedule_table。
        再按 simulate_native_program 的 warmup 规则展开为 per-stage 序列。
        """
        return native_stage_sequence(pp_rank, pp, vp_size, num_microbatches, group)

    def _simulate(self, request, profile):
        template_key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        delta = graph_delta(profile['request'], request)
        if not delta['reusable_template']:
            raise MissingCalibration('执行配置变化需要对应的实际调度模板')
        if delta.get('reuse_scope') == 'atomic_layer_and_link_costs':
            raise MissingCalibration('新PP/VPP组织需要相容的原生调度及通信、状态成本')
        if template_key in self.templates:
            plan, topology = deepcopy(self.templates[template_key])
            regions = ranges(request['decoder_chunks'])
            plan.persistent_bytes = self._persistent(request, profile)
            for task in plan.tasks.values():
                if task.region_id in delta['changed_regions']:
                    vp = task.region_id // plan.parallel_config.pp
                    cost = self._region_cost(profile, task.rank, vp, task.task_type.value, regions[task.region_id])
                    task.duration = cost['duration']
                    task.activation_delta_bytes = cost['activation'] * (1 if task.task_type == TaskType.FORWARD_COMPUTE else -1)
                    task.workspace_bytes = cost['workspace']
                    task.metadata.update(layer_ids=regions[task.region_id], **cost)
                if task.task_type == TaskType.OPTIMIZER_STEP:
                    task.duration = profile['optimizer'][str(task.rank)] * plan.persistent_bytes[task.rank] / profile['memory'][str(task.rank)]['persistent']
            self.replay_count += 1
            return ScheduleCostEvaluator(ExecutionCostModel()).evaluate(plan, topology, enforce_memory_limit=False), plan
        config = configuration(request)
        counts = request['decoder_chunks']
        if len(counts) != config.pp * config.vpp or min(counts) < 1 or sum(counts) != len(profile['layers']):
            raise ValueError('Region分区必须完整覆盖当前模型全部层')
        candidate_regions, baseline_regions = ranges(counts), ranges(profile['request']['decoder_chunks'])
        lanes = config.world_size // config.pp
        tasks, schedules, persistent = {}, {}, {}
        slots = {}
        for rank_text, sequence in profile['sequences'].items():
            rank, physical = int(rank_text), sequence['pp_rank']
            slots[(physical, rank % lanes)] = rank
        activation_rates = [row['activation_bytes'] / row['layer_count']
            for key, row in profile['regions'].items() if key.endswith(':forward_compute')
            and int(key.split(':')[0]) // lanes not in (0, config.pp - 1)]
        activation_rate = median(activation_rates)
        for rank_text, sequence in profile['sequences'].items():
            rank, physical = int(rank_text), sequence['pp_rank']
            old_ids = [layer for vp in range(config.vpp) for layer in baseline_regions[vp * config.pp + physical]]
            new_ids = [layer for vp in range(config.vpp) for layer in candidate_regions[vp * config.pp + physical]]
            old_state = sum(profile['layers'][str(layer)]['state_bytes'] for layer in old_ids)
            new_state = sum(profile['layers'][str(layer)]['state_bytes'] for layer in new_ids)
            persistent[rank] = profile['memory'][rank_text]['persistent'] + new_state - old_state
            schedules[rank] = []
            for kind, vp, mb in sequence['events']:
                region = vp * config.pp + physical
                ids = candidate_regions[region]
                measured = profile['regions'][f'{rank}:{vp}:{kind}']
                field = 'forward_ms' if kind == 'forward_compute' else 'backward_ms'
                duration = sum(profile['layers'][str(layer)][field] for layer in ids) + measured['overhead_ms']
                forward_row = profile['regions'][f'{rank}:{vp}:forward_compute']
                fixed_activation = max(0, forward_row['activation_bytes'] - activation_rate * forward_row['layer_count'])
                activation = max(0, activation_rate * len(ids) + fixed_activation)
                key = f'{kind}:{rank}:{vp}:{mb}'
                workspace = measured['workspace_bytes']
                if kind == 'forward_compute':
                    workspace = max(0, workspace - forward_row['activation_bytes'])
                task = Task(key, rank, TaskType(kind), microbatch_id=mb, region_id=region,
                    duration=duration, activation_delta_bytes=int(activation) * (1 if kind == 'forward_compute' else -1),
                    workspace_bytes=int(workspace), metadata=dict(layer_ids=ids,
                        duration_evidence=profile['id'], memory_evidence=profile['id']),
                    resource_requirements=(f'gpu:{rank}:compute',))
                tasks[key] = task
                schedules[rank].append(key)
        for rank_text, sequence in profile['sequences'].items():
            rank, physical = int(rank_text), sequence['pp_rank']
            lane = rank % lanes
            expanded = []
            for kind, vp, mb in sequence['events']:
                region = vp * config.pp + physical
                key = f'{kind}:{rank}:{vp}:{mb}'
                expanded.append(key)
                if kind == 'backward_compute':
                    tasks[key].dependencies.append(f'forward_compute:{rank}:{vp}:{mb}')
                neighbor = region + 1 if kind == 'forward_compute' else region - 1
                if not 0 <= neighbor < len(counts):
                    continue
                destination = slots[(neighbor % config.pp, lane)]
                next_vp = neighbor // config.pp
                direction = 'forward' if kind == 'forward_compute' else 'backward'
                link_key = f'{rank}:{destination}:{direction}'
                if link_key not in profile['links']:
                    raise MissingCalibration('Region通信路径缺少实测成本')
                transfer = f'transfer:{direction}:{region}:{lane}:{mb}'
                tasks[transfer] = Task(transfer, rank, TaskType.PP_P2P_SEND,
                    duration=profile['links'][link_key], dependencies=[key],
                    metadata=dict(duration_evidence=profile['id'], memory_evidence=profile['id']),
                    resource_requirements=(f'link:{min(rank, destination)}:{max(rank, destination)}',))
                expanded.append(transfer)
                consumer = f'{kind}:{destination}:{next_vp}:{mb}'
                tasks[consumer].dependencies.append(transfer)
            schedules[rank] = expanded
            backward_tasks = [key for key in expanded if tasks[key].task_type == TaskType.BACKWARD_COMPUTE]
            finalize = f'finalize:{rank}'
            optimizer = f'optimizer:{rank}'
            tasks[finalize] = Task(finalize, rank, TaskType.DP_REDUCE,
                duration=profile['gradient_finalize'][rank_text], dependencies=backward_tasks,
                metadata=dict(duration_evidence=profile['id'], memory_evidence=profile['id']),
                resource_requirements=(f'gpu:{rank}:compute',))
            state_ratio = persistent[rank] / profile['memory'][rank_text]['persistent']
            tasks[optimizer] = Task(optimizer, rank, TaskType.OPTIMIZER_STEP,
                duration=profile['optimizer'][rank_text] * state_ratio, dependencies=[finalize],
                metadata=dict(duration_evidence=profile['id'], memory_evidence=profile['id']),
                resource_requirements=(f'gpu:{rank}:compute',))
            schedules[rank].extend((finalize, optimizer))
        plan = ExecutionPlan(config, [], schedules, tasks=tasks, persistent_bytes=persistent)
        topology = HardwareTopology(config.world_size, request['node_count'], request['gpus_per_node'],
            min(row['capacity'] for row in profile['memory'].values()) / 2**30)
        if request['decoder_chunks'] == profile['request']['decoder_chunks']:
            self.templates[template_key] = deepcopy((plan, topology))
        self.replay_count += 1
        prediction = ScheduleCostEvaluator(ExecutionCostModel()).evaluate(plan, topology, enforce_memory_limit=False)
        return prediction, plan

    def needs_profile(self, request, started, missing, *, blind=False, partition_specific=False, persist=True):
        reusable = self.database.reusable_compute_profiles(request, blind=blind)
        identity = dict(plan=plan_signature(request), missing=missing,
            cost_database_revision=self.database.cost_database_revision(),
            reusable_profiles=[row['profile_id'] for row in reusable], partition_specific=partition_specific)
        report = dict(prediction_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            cost_database_revision=identity['cost_database_revision'],
            candidate_id=request['run_id'], plan_signature=plan_signature(request), prediction_status='NEEDS_PROFILE',
            validation=dict(status='REQUIRES_PROFILING', backend_supported=True),
            performance=dict(step_time_ms=None, tokens_per_second=None, speedup_vs_parent=None,
                             prediction_interval_ms=None),
            memory=dict(peak_per_rank_bytes={}, admission='REQUIRE_MEMORY_PROFILE', physical_memory_status='UNKNOWN'),
            diagnosis=dict(critical_regions=[], pipeline_idle_ms=None, exposed_communication_ms=None),
            evidence=dict(compatible_anchor_id=None, missing_profiles=missing, partition_specific=partition_specific,
                reusable_compute_profiles=reusable,
                profiling_requests=[dict(kind='native_training_cost_profile', request=request,
                    required_observations=['layer_forward_backward_recompute', 'pipeline_links',
                                           'saved_activations', 'optimizer_states', 'native_schedule'])]),
            evaluation_seconds=time.perf_counter() - started)
        if self.projector is not None:
            report['memory'] = self.projector.memory_bound(request, blind=blind)
            report['evidence']['memory_evaluation_scope'] = (
                'parameter_ownership_and_activation_bound_independent_of_missing_task_times')
            report['prediction_id'] = hashlib.sha256(json.dumps(dict(
                source_prediction=report['prediction_id'], memory=report['memory']),
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            report['evaluation_seconds'] = time.perf_counter() - started
        if persist:
            self.database.save_prediction(report)
        return report

    def evaluate(self, request=None, profile_id=None, persist=True, *, model=None,
                 workload=None, hardware=None, candidate=None, parent_plan=None, blind=False,
                 candidate_graph=None, cost_mode=None):
        started = time.perf_counter()
        if request is None:
            if not all(isinstance(value, dict) for value in (model, workload, hardware, candidate)):
                raise ValueError('预测接口需要完整模型、工作负载、硬件和候选字典')
            request = dict(candidate, **workload)
            request['model'] = model['path']
            request['node_count'] = hardware['node_count']
            request['gpus_per_node'] = hardware['gpus_per_node']
        config = configuration(request)
        parent_request = parent_plan.backend_config if hasattr(parent_plan, 'backend_config') else parent_plan
        if config.world_size != request['node_count'] * request['gpus_per_node']:
            raise ValueError('候选并行配置与实际GPU数量不一致')
        selected_mode = cost_mode or self.cost_mode
        if self.projector is not None:
            return self.evaluate_projection(request, candidate_graph, parent_plan,
                persist=persist, blind=blind, cost_mode=selected_mode)
        if candidate_graph is not None:
            raise ValueError('完整候选执行图预测需要先配置模型图与硬件拓扑')
        if selected_mode == 'analytical_only':
            return self.project_dag(request, candidate_graph, parent_plan, persist, selected_mode, started, blind)
        if profile_id is None:
            profile_id = self.database.select(request, blind=blind)
        elif not self.database.matches(profile_id, request):
            profile_id = self.database.select(request, blind=blind)
        if profile_id is None:
            if self.projector is not None:
                return self.project_dag(request, candidate_graph, parent_plan, persist, selected_mode, started, blind)
            return self.needs_profile(request, started, ['当前执行条件的计算、通信和显存成本'],
                blind=blind, persist=persist)
        profile_document = self._special_module_breakdown(
            self.database.get(profile_id, request, blind=blind))
        if self.context_topology is not None and self.context_topology.devices:
            devices = {row['physical_rank']: row for row in self.context_topology.devices}
            mapping = request.get('rank_mapping', list(range(config.world_size)))
            actual_ids = [devices[physical].get('uuid', devices[physical].get('gpu_uuid')) for physical in mapping]
            profile_ids = [row.get('gpu_uuid') for row in profile_document['hardware']]
            actual_ids = [str(value).removeprefix('GPU-').lower() if value is not None else None for value in actual_ids]
            profile_ids = [str(value).removeprefix('GPU-').lower() if value is not None else None for value in profile_ids]
            if any(identifier is None for identifier in actual_ids) or actual_ids != profile_ids:
                return self.project_dag(request, candidate_graph, parent_plan, persist, selected_mode, started, blind)
        if hardware is not None and hardware.get('devices') != profile_document['hardware']:
            raise ValueError('调用方硬件和软件环境与成本记录不一致')
        evidence_version = hashlib.sha256(json.dumps(profile_document, sort_keys=True).encode()).hexdigest()
        residuals = self.database.validation_residuals(request)
        parent_measurement = None if parent_request is None else {
            key: parent_request.get(key) for key in ('_measured_step_time_ms',
                '_measurement_run_id', '_predicted_raw_step_time_ms', '_prediction_id')}
        prediction_key = hashlib.sha256(json.dumps(dict(predictor_revision='parametric_execution_events_20261011_v2',
            request=request, evidence=evidence_version, cost_mode=selected_mode,
            cost_database_revision=self.database.cost_database_revision(),
            residuals=residuals, blind=blind, parent_measurement=parent_measurement),
            sort_keys=True).encode()).hexdigest()
        if prediction_key in self.predictions:
            report = deepcopy(self.predictions[prediction_key])
            report['evaluation_seconds'] = time.perf_counter() - started
            report['diagnosis']['full_replays'] = 0
            report['diagnosis']['prediction_cache_hit'] = True
            if persist:
                self.database.save_prediction(report)
            return report
        cache_key = (profile_id, signature(request), evidence_version)
        if cache_key not in self.profiles:
            self.profiles[cache_key] = profile_document
        profile = self.profiles[cache_key]
        if hardware is not None and hardware.get('devices') != profile['hardware']:
            raise ValueError('调用方硬件和软件环境与成本记录不一致')
        try:
            admission_check = self.memory_admission(request, profile)
        except MissingCalibration as error:
            if self.projector is not None:
                return self.project_dag(request, candidate_graph, parent_plan, persist, selected_mode, started, blind)
            return self.needs_profile(request, started, [str(error)], blind=blind,
                partition_specific=True, persist=persist)
        replay_start = self.replay_count
        if admission_check['admission'] == 'REJECT_OOM':
            report = dict(prediction_id=prediction_key, candidate_id=request['run_id'],
                cost_database_revision=self.database.cost_database_revision(),
                plan_signature=plan_signature(request), validation=dict(status='REJECT_OOM', backend_supported=True),
                performance=dict(step_time_ms=None, tokens_per_second=None, speedup_vs_parent=None, prediction_interval_ms=None),
                memory=dict(admission_check), diagnosis=dict(full_replays=0),
                evidence=dict(profile_id=profile_id, static_memory_filter=True),
                evaluation_seconds=time.perf_counter() - started)
            self.predictions[prediction_key] = deepcopy(report)
            if persist:
                self.database.save_prediction(report)
            return report
        try:
            if cache_key not in self.baselines:
                self.baselines[cache_key] = self._simulate(profile['request'], profile)[0]
            baseline = self.baselines[cache_key]
            candidate, plan = self._simulate(request, profile)
        except MissingCalibration as error:
            if self.projector is not None:
                return self.project_dag(request, candidate_graph, parent_plan, persist, selected_mode, started, blind)
            return self.needs_profile(request, started, [str(error)], blind=blind,
                partition_specific=True, persist=persist)
        anchor = profile['anchor']
        anchor_simulated = baseline.predicted_iteration_time_ms
        if (parent_request is not None and parent_request.get('_measured_step_time_ms') is not None
                and signature(parent_request) == signature(request)):
            parent_candidate, _ = self._simulate(parent_request, profile)
            anchor = dict(run_id=parent_request.get('_measurement_run_id'),
                step_seconds=parent_request['_measured_step_time_ms'] / 1000,
                prediction_id=parent_request.get('_prediction_id'), source='latest_measured_parent_plan')
            anchor_simulated = parent_candidate.predicted_iteration_time_ms
        step_ms = (anchor['step_seconds'] * 1000 + candidate.predicted_iteration_time_ms - anchor_simulated
                   if anchor is not None else candidate.predicted_iteration_time_ms)
        if step_ms <= 0:
            raise ValueError('校准后预测耗时非法，需要补充成本测量')
        ranks = {}
        for rank, raw_peak in candidate.rank_peak_bytes.items():
            observed = profile['memory'][str(rank)]
            correction = observed['peak'] - baseline.rank_peak_bytes[rank]
            ranks[rank] = max(plan.persistent_bytes[rank], raw_peak + correction)
        raw_residual = abs(anchor['step_seconds'] * 1000 - anchor_simulated) if anchor is not None else 0
        empirical = max(statistics.pstdev([(row['host_end_ns'] - row['host_start_ns']) / 1e6
            for row in windows]) for windows in profile['measured_windows'].values())
        uncertainty = raw_residual + 2 * empirical
        validation_residuals = self.database.validation_residuals(request)
        if validation_residuals:
            uncertainty = max(uncertainty, max(abs(row['time_residual_ms']) for row in validation_residuals))
        memory_residual = max(abs(profile['memory'][str(rank)]['peak'] - baseline.rank_peak_bytes[rank])
                              for rank in ranks)
        if validation_residuals:
            memory_residual = max(memory_residual, max(abs(row['memory_residual_bytes']) for row in validation_residuals))
        lower = {rank: max(plan.persistent_bytes[rank], peak - memory_residual) for rank, peak in ranks.items()}
        upper = {rank: peak + memory_residual for rank, peak in ranks.items()}
        if any(lower[rank] >= profile['memory'][str(rank)]['capacity'] for rank in ranks):
            admission = 'REJECT_OOM'
        elif any(upper[rank] >= profile['memory'][str(rank)]['capacity'] for rank in ranks):
            admission = 'REQUIRE_MEMORY_PROFILE'
        else:
            admission = 'WITHIN_ESTIMATED_CAPACITY'
        identity = prediction_key
        report = dict(prediction_id=identity, candidate_id=request['run_id'], plan_signature=plan_signature(request),
            cost_database_revision=self.database.cost_database_revision(),
            validation=dict(status='PROJECTABLE', backend_supported=True),
            performance=dict(step_time_ms=step_ms, tokens_per_second=request['global_batch'] * request['sequence'] / (step_ms / 1000),
                speedup_vs_parent=(anchor['step_seconds'] / (step_ms / 1000) - 1 if anchor is not None else None),
                raw_resource_aware_step_time_ms=candidate.predicted_iteration_time_ms,
                anchor_raw_step_time_ms=anchor_simulated if anchor is not None else None,
                prediction_interval_ms=[max(0, step_ms - uncertainty), step_ms + uncertainty],
                interval_scope='observed_variability_and_baseline_simulation_residual_not_statistical_coverage'),
            memory=dict(peak_per_rank_bytes=ranks, lower_per_rank_bytes=lower,
                upper_per_rank_bytes=upper, admission=admission,
                physical_memory_status='MEMORY_UNCERTAIN',
                missing_profiles=['communication_storage_lifetimes', 'allocator_reserved_peak',
                    'external_device_allocations'],
                peak_event_per_rank=admission_check['peak_event_per_rank'],
                peak_live_regions_per_rank=admission_check['peak_live_regions_per_rank'],
                peak_event_scope='measured_native_compute_order_logical_memory',
                breakdown_per_rank=admission_check['breakdown_per_rank'],
                memory_budget_per_rank_bytes=admission_check['memory_budget_per_rank_bytes'],
                memory_headroom_per_rank_bytes={rank: profile['memory'][str(rank)]['capacity'] - upper[rank]
                    for rank in ranks},
                oom_risk='predicted_over_capacity' if admission == 'REJECT_OOM' else
                    ('uncertain' if admission == 'REQUIRE_MEMORY_PROFILE' else 'within_measured_model_bounds'),
                scope='measured_layer_states_and_simulated_activation_lifetimes_with_baseline_residual'),
            diagnosis=dict(critical_tasks=candidate.critical_path_tasks,
                simulated_step_time_ms=candidate.predicted_iteration_time_ms,
                baseline_simulated_step_time_ms=baseline.predicted_iteration_time_ms,
                pipeline_idle_ms=candidate.idle_time_ms,
                modeled_exposed_communication_ms=candidate.exposed_communication_ms),
            evidence=dict(compatible_anchor_id=anchor['run_id'] if anchor is not None else None,
                anchor_source=anchor.get('source', 'profile_anchor') if anchor is not None else None,
                profile_id=profile_id,
                fixed_cost_breakdown=profile.get('fixed_cost_breakdown', {}),
                missing_profiles=['计算与通信资源竞争的独立并发测量', '跨分区Workspace变化', '独立候选预测残差'],
                profiling_requests=[dict(kind='compute_communication_pair',
                    region_tasks=candidate.critical_path_tasks, execution_signature=signature(request)),
                    dict(kind='region_workspace', decoder_chunks=request['decoder_chunks'])],
                schedule_source='actual_megatron_native_forward_backward_order',
                calibration_excludes_candidate_measurement=plan_signature(profile['request']) != plan_signature(request),
                blind=blind, candidate_plan_excluded_from_calibration=blind,
                source_hashes=profile['source_hashes']),
            evaluation_seconds=time.perf_counter() - started)
        report['execution_graph'] = dict(task_count=len(plan.tasks),
            tasks={key: dict(rank=task.rank, kind=task.task_type.value, dependencies=task.dependencies,
                duration_ms=task.duration, region=task.region_id, microbatch=task.microbatch_id,
                start_ms=candidate.task_times[key][0], end_ms=candidate.task_times[key][1],
                activation_delta_bytes=task.activation_delta_bytes, workspace_bytes=task.workspace_bytes)
                for key, task in plan.tasks.items()}, rank_schedules=plan.rank_schedules)
        report['diagnosis']['critical_regions'] = sorted({plan.tasks[key].region_id
            for key in candidate.critical_path_tasks if plan.tasks[key].region_id is not None})
        report['diagnosis']['changed_operations'] = [dict(region=index,
            previous_layers=old, candidate_layers=new) for index, (old, new) in enumerate(zip(
                ranges(profile['request']['decoder_chunks']), ranges(request['decoder_chunks']))) if old != new]
        report['evidence']['profiling_selection'] = 'ranking_probes_requires_candidate_pair'
        same_layout = request['decoder_chunks'] == profile['request']['decoder_chunks']
        report['prediction_status'] = 'PREDICTED_CALIBRATED' if same_layout and anchor is not None else 'PREDICTED_ANALYTICAL'
        report['evidence']['calibration_coverage'] = dict(task_costs='measured_layer_composite',
            schedule='measured_native_order', candidate_layout='measured' if same_layout else 'projected',
            communication_contention='unmeasured', candidate_ranking='requires_independent_measurements')
        report['evidence']['cost_granularity'] = profile.get('cost_granularity', 'measured_layer_region_composite')
        report['evidence']['prior_validation_count'] = len(validation_residuals)
        report['evidence']['overlap_status'] = 'overlap_unknown'
        report['memory']['static_filter'] = admission_check
        report['diagnosis']['graph_delta'] = graph_delta(profile['request'], request)
        changed_regions = report['diagnosis']['graph_delta']['changed_regions']
        affected_ranks = {task.rank for task in plan.tasks.values() if task.region_id in changed_regions}
        report['diagnosis']['graph_delta'].update(
            changed_tasks=[key for key, task in plan.tasks.items() if task.region_id in changed_regions
                or (task.rank in affected_ranks and task.task_type == TaskType.OPTIMIZER_STEP)],
            changed_dependencies=[], changed_communications=[],
            changed_buffer_lifetimes=[key for key, task in plan.tasks.items() if task.region_id in changed_regions],
            state_reuse='costs_and_graph_template', event_state_reuse=False)
        report['diagnosis']['full_replays'] = self.replay_count - replay_start
        report['diagnosis']['prediction_cache_hit'] = False
        self.predictions[prediction_key] = deepcopy(report)
        if persist:
            self.database.save_prediction(report)
        return report

    def ranking_probes(self, left, right, profile_id=None):
        if signature(left) != signature(right):
            raise ValueError('候选排序敏感度需要相同执行条件的实测成本')
        profile_id = profile_id or self.database.select(left)
        if profile_id is None:
            raise MissingCalibration('候选排序敏感度缺少实测成本')
        profile = self.database.get(profile_id, left)
        for request in (left, right):
            if self.memory_admission(request, profile)['admission'] == 'REJECT_OOM':
                raise ValueError('候选排序敏感度仅接受通过显存过滤的候选')
        self._simulate(profile['request'], profile)
        left_result, left_plan = self._simulate(left, profile)
        right_result, right_plan = self._simulate(right, profile)
        gap = left_result.predicted_iteration_time_ms - right_result.predicted_iteration_time_ms
        config = configuration(left)
        topology = HardwareTopology(config.world_size, left['node_count'], left['gpus_per_node'],
            min(row['capacity'] for row in profile['memory'].values()) / 2**30)
        probes = []
        for source, row in profile['regions'].items():
            sigma = row['residual_ms']
            if sigma <= 0:
                continue
            rank, vp, kind = source.split(':')
            rank, vp = int(rank), int(vp)
            differences = []
            scenario_times = []
            for sign in (-1, 1):
                times = []
                for original in (left_plan, right_plan):
                    plan = deepcopy(original)
                    for task in plan.tasks.values():
                        if task.rank == rank and task.region_id is not None and task.region_id // config.pp == vp and task.task_type.value == kind:
                            task.duration = max(0, task.duration + sign * sigma)
                    result = ScheduleCostEvaluator(ExecutionCostModel()).evaluate(plan, topology, enforce_memory_limit=False)
                    self.replay_count += 1
                    times.append(result.predicted_iteration_time_ms)
                differences.append(times[0] - times[1])
                scenario_times.append(times)
            impact = max(abs(value - gap) for value in differences)
            if min(differences + [gap]) <= 0 <= max(differences + [gap]) and impact > 0:
                gpu_hours = row.get('benchmark_gpu_hours')
                if gpu_hours is not None and (not math.isfinite(gpu_hours) or gpu_hours <= 0):
                    raise ValueError('选择测量需要真实且有效的GPU时间成本')
                probes.append(dict(kind='region_compute_and_workspace', rank=rank, vp=vp,
                    operation=kind, measured_uncertainty_ms=sigma, ranking_impact_ms=impact,
                    benchmark_gpu_hours=gpu_hours, priority=impact / gpu_hours if gpu_hours else impact,
                    candidates=[left['run_id'], right['run_id']],
                    scenario_step_time_ms=dict(low=dict(left=scenario_times[0][0], right=scenario_times[0][1]),
                        high=dict(left=scenario_times[1][0], right=scenario_times[1][1])),
                    scenario_gap_ms=dict(low=differences[0], high=differences[1])))
        return dict(candidate_gap_ms=gap, profiling_requests=sorted(probes,
            key=lambda row: row['priority'], reverse=True), overlap_status='overlap_unknown',
            scope='correlated_single_source_measured_variability_ranking_sensitivity',
            joint_uncertainty_assessed=False)
