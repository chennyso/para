from collections import Counter, defaultdict
from copy import deepcopy
from bisect import bisect_left
import math
import json
import statistics
import time

from contracts import fingerprint
from core_data_models import LayerType, Task, TaskType
from cost.runtime_calibration import configuration
from graph.process_groups import process_groups
from graph.recompute_effect import resolve_recompute_effect
from predictor.profile_database import normalized_execution, plan_signature, same_training_plan
from predictor.communication_cost import bind_candidate_execution_conditions
from predictor.ddp_buckets import compile_bucket_tasks
from predictor.measured_modules import (enrich_module_profile, boundary_costs, compute_profile_quality,
    optimizer_local_cost_evidence, optimizer_component_cost_evidence)
from predictor.parameter_metadata import checkpoint_parameter_counts
from predictor.physical_paths import communication_resources
from predictor.state_sharding import parameter_state_ownership, phased_memory_bounds, state_sharding_config
from region_partition.region_partitioner import RegionPartitioner
from scheduling.native_schedule_lowerer import lower_native_schedule
from scheduling.native_microbatches import native_schedule_settings
from scheduling.schedule_simulator import ScheduleCostEvaluator, topological_order, union_length
from scheduling.task_graph_builder import TaskGraphBuilder


DTYPE_BYTES = {'bfloat16': 2, 'float16': 2, 'float32': 4}

PROJECTION_REVISION = 'native_event_graph_v44_phased_p2p_and_overlap_conditions'

OPTIMIZER_PROFILE_FIELDS = (
    'model', 'checkpoint', 'optimizer', 'optimizer_config', 'optimizer_sharding',
    'execution_backend', 'precision', 'precision_aware_optimizer',
    'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype',
    'offload', 'optimizer_offload_fraction', 'optimizer_transfer_overlap', 'clip_grad',
)


def optimizer_profile_key(request):
    execution = normalized_execution(request)
    return tuple((field, execution.get(field)) for field in OPTIMIZER_PROFILE_FIELDS)


def exact_profile_anchor(document, request):
    anchor = document.get('anchor')
    if not isinstance(anchor, dict) or not same_training_plan(document['request'], request):
        return None
    step_seconds = anchor.get('step_seconds')
    if (anchor.get('status') != 'PASS' or anchor.get('performance_eligible') is not True
            or not isinstance(step_seconds, (int, float)) or not math.isfinite(step_seconds)
            or step_seconds <= 0 or not isinstance(anchor.get('run_id'), str)):
        return None
    return dict(run_id=anchor['run_id'], measured_step_time_ms=step_seconds * 1000,
        scope='exact_training_plan_whole_step_measurement')


class DAGProjector:
    def __init__(self, model, model_graph, topology, compute_surface,
                 communication_surface, hardware_parameters, native_sequence):
        self.model = model
        self.model_graph = model_graph
        self.topology = topology
        self.compute_surface = compute_surface
        self.communication_surface = communication_surface
        self.parameters = deepcopy(hardware_parameters)
        self.native_sequence = native_sequence
        self.preferred_profile_id = None
        self.module_profiles = {}
        self.optimizer_profiles = {}
        self.reusable_profile = None
        self.optimizer_profile = None
        self.parameter_counts_cache = {}
        self.mode = 'hybrid'
        self.blind = False
        self.active_communication_topology = None

    @staticmethod
    def differential_family(request):
        config = configuration(request)
        sharding = state_sharding_config(request, config)
        return dict(tp=config.tp, cp=config.cp, pp=config.pp, dp=config.dp,
            ep=config.ep, etp=config.etp, vpp=config.vpp,
            micro_batch=config.micro_batch_size, group=config.microbatch_group,
            recompute=request['recompute'], precision=request.get('precision'),
            pipeline_schedule=request.get('pipeline_schedule',
                'interleaved_1f1b' if config.vpp > 1 else '1f1b'),
            rank_mapping=request.get('rank_mapping', list(range(config.world_size))),
            state_sharding=sharding)

    def _hardware(self, physical_rank, dtype):
        compute = self.parameters.get('compute', {})
        selected = compute.get('per_rank', {}).get(str(physical_rank), compute)
        device = next((row for row in self.topology.devices
                       if row.get('physical_rank', row.get('rank')) == physical_rank), None)
        identity = selected.get('identity', device)
        if identity is None:
            identity = dict(physical_rank=physical_rank,
                            topology_fingerprint=fingerprint(self.topology.devices))
        return dict(selected, identity=identity, dtype=dtype)

    def _compute(self, operation, features, phase, rank, config, request, costs,
                 components=None, layer_id=None, analytical_operation=None):
        physical_rank = self.groups['rank_mapping'][rank]
        hardware = self._hardware(physical_rank, self.dtype)
        binding = self.parameters.get('operator_bindings', {}).get(
            f'layer:{layer_id}:{operation}', self.parameters.get('operator_bindings', {}).get(operation))
        if binding is not None and self.blind and binding.get('source_plan_signature') in (None, plan_signature(request)):
            binding = None
        if self.blind and hardware.get('throughput_kind') == 'calibrated_effective':
            sources = hardware.get('calibration_source_plan_signatures')
            if not sources or plan_signature(request) in sources:
                result = dict(status='NEEDS_PROFILE', milliseconds=None,
                    reason='独立预测缺少可排除候选自身测量的硬件标定来源',
                    identity=dict(operation=operation, features=features, physical_rank=physical_rank),
                    profiling_request=dict(kind='operator_cost',
                        identity=dict(operation=operation, features=features, physical_rank=physical_rank),
                        cost_key=fingerprint(dict(operation=operation, features=features, physical_rank=physical_rank))))
                costs.append(result)
                return None
        identity = dict(gpu=hardware['identity'], dtype=self.dtype,
            implementation=f'backend_semantic:{operation}', granularity='semantic_projection',
            operation=operation, phase=phase,
            features={key: value for key, value in features.items() if type(value) in (int, float)},
            semantic_shape=features,
            layout=dict(tp=config.tp, cp=config.cp, etp=config.etp, ep=config.ep,
                        sequence_parallel=config.use_sequence_parallel),
            model=fingerprint(self.model_graph['module_structure']))
        if binding is not None:
            if not binding.get('source') or not binding.get('identity'):
                raise ValueError('算子执行绑定需要实际记录来源和完整执行身份')
            identity = deepcopy(binding['identity'])
            # Shape必须由绑定提供明确的映射规则，复合模块的输入不能冒充内部Kernel输入。
            if binding.get('shape_kind') == 'gemm_mnk' and set(features) == {'M', 'N', 'K'}:
                identity['features'] = features
                identity['phase'] = phase
            elif binding.get('semantic_shape') != features or identity.get('phase') != phase:
                identity = dict(gpu=hardware['identity'], dtype=self.dtype,
                    implementation=f'backend_semantic:{operation}', granularity='semantic_projection',
                    operation=operation, phase=phase,
                    features={key: value for key, value in features.items() if type(value) in (int, float)},
                    semantic_shape=features,
                    model=fingerprint(self.model_graph['module_structure']))
        if binding is not None and identity.get('implementation') == binding['identity'].get('implementation'):
            hardware = dict(hardware, identity=identity.get('gpu', identity.get('hardware')),
                            dtype=identity.get('dtype', identity.get('precision', self.dtype)))
        specification = dict(operation=analytical_operation or operation, features=features,
                             phase=phase, hardware=hardware)
        if components is not None:
            specification = dict(operation='composite', components=components, hardware=hardware)
        identity_key = fingerprint(dict(identity=identity, specification=specification, mode=self.mode))
        if identity_key not in self.cost_cache:
            self.cost_cache[identity_key] = self.compute_surface.query(
                identity, analytical_spec=specification, mode=self.mode,
                calibration_run_ids=self.calibration_run_ids)
        result = deepcopy(self.cost_cache[identity_key])
        result.update(projection_operation=operation, logical_layer=layer_id,
                      physical_rank=physical_rank, cost_key=identity_key,
                      single_gemm=components is None and analytical_operation == 'gemm')
        costs.append(result)
        return result['milliseconds']

    def _communication(self, collective, payload_bytes, ranks, costs, matrix=None, dtype=None,
                       concurrency=()):
        if len(ranks) == 1:
            return 0.0, []
        physical = [self.groups['rank_mapping'][rank] for rank in ranks]
        network = self.parameters.get('network', {})
        topology_identity = self.active_communication_topology
        if topology_identity is None:
            raise ValueError('通信成本查询缺少当前候选的执行环境')
        spec = None
        if network.get('source'):
            spec = dict(source=network['source'], assumptions=network.get('assumptions', []),
                algorithm='direct' if collective == 'p2p' else 'pairwise' if collective == 'all_to_all' else 'ring',
                physical_ranks=physical, topology_identity=topology_identity,
                links=[row for row in network.get('links', [])
                       if row['source_rank'] in physical and row['destination_rank'] in physical],
                resources=network.get('resources', {}))
            if matrix is not None:
                spec['send_bytes_matrix'] = matrix
        communication_dtype = dtype or self.dtype
        identity_key = fingerprint(dict(collective=collective, payload_bytes=payload_bytes,
            ranks=physical, topology=topology_identity, specification=spec, mode=self.mode,
            dtype=communication_dtype, concurrency=list(concurrency)))
        if identity_key not in self.cost_cache:
            self.cost_cache[identity_key] = self.communication_surface.query(
                collective, max(1, int(payload_bytes)), physical, topology_identity,
                concurrency=concurrency, analytical_spec=spec, mode=self.mode,
                dtype=communication_dtype)
        result = deepcopy(self.cost_cache[identity_key])
        result.update(cost_key=identity_key, projection_operation=collective)
        requests = [result['profiling_request']] if result.get('profiling_request') else []
        requests.extend(result.get('profiling_targets', []))
        for request in requests:
            identity = request.get('identity', {})
            if (request.get('kind') == 'nccl' and identity.get('collective') == collective
                    and identity.get('payload_bytes') == payload_bytes):
                identity.update(dtype=communication_dtype, physical_ranks=physical,
                    topology_identity=topology_identity, concurrent_operations=list(concurrency))
                request['cost_key'] = identity_key
        if result['milliseconds'] is not None and isinstance(topology_identity, dict):
            binding = communication_resources(topology_identity, physical)
            result['resource_binding'] = binding
            result['resource_ids'] = binding['resource_ids']
            for phase in result.get('communication_phases', []):
                phase['resource_ids'] = binding['resource_ids']
        costs.append(result)
        return result['milliseconds'], result.get('resource_ids', [])

    def _group(self, kind, rank):
        return next(row for row in self.groups['logical'][kind] if rank in row)

    @staticmethod
    def _total(values):
        return None if any(value is None for value in values) else sum(values)

    @staticmethod
    def _cost_components(records):
        components = []
        for row in records:
            milliseconds = row.get('milliseconds')
            if milliseconds is None or milliseconds == 0:
                continue
            if 'cost_key' not in row:
                raise ValueError('执行任务的成本组成缺少稳定身份')
            evidence_ids = list(row.get('evidence_ids', []))
            if row.get('evidence_id'):
                evidence_ids.append(row['evidence_id'])
            components.append(dict(cost_key=row['cost_key'], milliseconds=milliseconds,
                occurrences=1, evidence_ids=sorted(set(evidence_ids))))
        return components

    @staticmethod
    def _coalesce_fsdp_collectives(tasks):
        collective_types = {TaskType.FSDP_UNSHARD, TaskType.FSDP_PREFETCH,
            TaskType.FSDP_GRAD_REDUCE_SCATTER, TaskType.FSDP_OUTER_GRAD_REDUCE,
            TaskType.FSDP_OUTER_PARAM_SYNC}
        grouped = defaultdict(list)
        for key, task in tasks.items():
            if task.task_type not in collective_types or task.metadata.get('cost_role') == 'prefetch_wait':
                continue
            compute = tasks.get(task.metadata.get('compute_task'))
            phase = None if compute is None else compute.task_type.value
            identity = (task.task_type.value, task.metadata.get('state_family'),
                tuple(task.metadata.get('group_ranks', [])), task.microbatch_id,
                task.region_id, phase, tuple(task.metadata.get('communication_payloads', [])),
                task.metadata.get('communication_dtype'),
                task.metadata.get('communication_payload_definition'))
            grouped[identity].append(key)
        for identity, keys in grouped.items():
            group_ranks = identity[2]
            if len(keys) == 1:
                tasks[keys[0]].metadata['collective_instance_id'] = fingerprint(identity)
                tasks[keys[0]].metadata['collective_representative'] = True
                continue
            if len(keys) != len(group_ranks):
                raise ValueError('FSDP Collective实例没有覆盖完整通信进程组')
            representative = min(keys)
            dependencies = sorted({dependency for key in keys
                for dependency in tasks[key].dependencies if dependency not in keys})
            tasks[representative].dependencies = dependencies
            instance_id = fingerprint(identity)
            tasks[representative].metadata.update(collective_instance_id=instance_id,
                collective_representative=True, collective_alias_task_ids=sorted(
                    key for key in keys if key != representative))
            for key in keys:
                if key == representative:
                    continue
                tasks[key].dependencies = [representative]
                tasks[key].metadata.update(collective_instance_id=instance_id,
                    collective_representative=False, collective_representative_task_id=representative)

    def _gemm(self, name, m, n, k, phase, rank, config, request, costs, layer):
        if phase == 'backward':
            components = [dict(operation='gemm', phase='dgrad', features=dict(M=m, N=n, K=k)),
                          dict(operation='gemm', phase='wgrad', features=dict(M=m, N=n, K=k))]
            return self._compute(name, dict(M=m, N=n, K=k), phase, rank, config,
                                 request, costs, components=components, layer_id=layer)
        return self._compute(name, dict(M=m, N=n, K=k), phase, rank, config,
                             request, costs, layer_id=layer, analytical_operation='gemm')

    def _elementwise(self, name, count, traffic, phase, rank, config, request, costs, layer):
        if phase == 'backward':
            count *= 2
            traffic *= 2
        features = dict(operation_count=count, traffic_bytes=traffic,
            operation_count_source=f'model_semantics:{name}:{phase}',
            traffic_source=f'tensor_shape:{name}:{phase}')
        value = self._compute(name, features, phase, rank, config,
                              request, costs, layer_id=layer, analytical_operation='elementwise')
        costs[-1]['semantic_operation'] = name
        return value

    def _expert_tokens(self, rank, layer, config):
        routing = self.parameters.get('moe_token_distribution', {})
        row = routing.get('per_layer_rank', {}).get(f'{layer}:{rank}')
        if row is None:
            return None
        if self.blind and row.get('source_plan_signature') in (None, self.current_plan_signature):
            return None
        counts = row.get('expert_token_counts')
        if (not row.get('source') or not isinstance(counts, list) or
                len(counts) != self.model.num_experts // config.ep or
                any(type(value) is not int or value < 0 for value in counts)):
            raise ValueError('MoE成本需要具有来源的全部本地Expert Token数量')
        if row.get('micro_batch') != config.micro_batch_size or row.get('sequence') != config.sequence_length:
            return None
        if row.get('parallelism') != dict(tp=config.tp, cp=config.cp, ep=config.ep, etp=config.etp):
            return None
        return row

    def _layer(self, layer, phase, rank, config, request, costs):
        if self.reusable_profile is not None and phase in ('forward', 'backward'):
            effect = resolve_recompute_effect(request, layer, self.reusable_profile)
            value = effect.forward_ms if phase == 'forward' else effect.backward_ms
            costs.append(dict(status='MEASURED_REUSED', milliseconds=value,
                cost_key=fingerprint(dict(profile=self.reusable_profile['id'],
                    operation='layer_composite', layer=layer, phase=phase)),
                projection_operation='layer_composite', logical_layer=layer, phase=phase,
                module_type=self.model.layer_types[layer].value,
                module_kind=(self.model.module_kinds[layer] if self.model.module_kinds else 'transformer_block'),
                source='actual_layer_composite_cost_with_unchanged_compute_conditions',
                cost_scope='actual_layer_composite_including_internal_collectives',
                measurement_scope=self.reusable_profile['measurement_semantics']['layer'],
                evidence_ids=[self.reusable_profile['id']],
                assumptions=['复用同一GPU集合及相同TP、CP、EP、ETP、MBS、Sequence和重计算条件的逐层复合测量',
                    'PP、VPP、DP与Region变化重新构造执行组织；层内成本保持原始复合测量粒度']))
            return value
        if self.model.module_kinds is not None:
            identity = dict(module_kind=self.model.module_kinds[layer], layer=layer,
                tp=config.tp, cp=config.cp, ep=config.ep, etp=config.etp,
                sequence=config.sequence_length, micro_batch=config.micro_batch_size,
                phase=phase, precision=self.dtype, recompute=request['recompute'])
            costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                reason='当前混合模块执行条件缺少代表模块实测成本',
                profiling_request=dict(kind='representative_module', identity=identity,
                    cost_key=fingerprint(identity))))
            return None
        model = self.model
        tokens = config.micro_batch_size * config.sequence_length // config.cp
        hidden = model.hidden_size
        dimension = model.head_dimension or hidden // model.num_attention_heads
        heads = model.num_attention_heads // config.tp
        kv = max(1, (model.num_query_groups or model.num_attention_heads) // config.tp)
        qkv = (heads + 2 * kv) * dimension
        element_bytes = DTYPE_BYTES[self.dtype]
        local_tokens = tokens // config.tp if config.use_sequence_parallel else tokens
        values = [self._elementwise('attention_norm', local_tokens * hidden * 8,
            local_tokens * hidden * element_bytes * 3, phase, rank, config, request, costs, layer)]
        if config.use_sequence_parallel and config.tp > 1:
            communication, _ = self._communication('all_gather', local_tokens * hidden * element_bytes,
                                                    self._group('tp', rank), costs)
            values.append(communication)
        values.append(self._gemm('qkv_projection', tokens, qkv, hidden, phase,
                                 rank, config, request, costs, layer))
        attention_options = self.parameters.get('attention', {})
        attention = dict(batch=config.micro_batch_size, query_heads=heads, kv_heads=kv,
            query_length=config.sequence_length // config.cp, key_length=config.sequence_length,
            head_dimension=dimension, value_dimension=dimension,
            causal=self.model_graph['module_structure'].get('is_causal', True),
            materializes_attention_scores=attention_options.get('materializes_attention_scores', False))
        for name in ('softmax_operations_per_element', 'softmax_backward_operations_per_element',
                     'softmax_operation_source'):
            if name in attention_options:
                attention[name] = attention_options[name]
        values.append(self._compute('attention', attention, phase, rank, config,
                                    request, costs, layer_id=layer))
        if config.cp > 1:
            ranks = self._group('cp', rank)
            size = tokens * kv * dimension * element_bytes * 2
            comm, _ = self._communication('all_gather', size, ranks, costs)
            values.append(comm)
            if phase == 'backward':
                comm, _ = self._communication('reduce_scatter', size, ranks, costs)
                values.append(comm)
            costs[-1].setdefault('assumptions', []).append(
                'CP通信按完整KV交换估计；内部Attention通信算法与重叠需要后端记录标定')
        values.append(self._gemm('attention_output', tokens, hidden, heads * dimension,
                                 phase, rank, config, request, costs, layer))
        output_collective = 'reduce_scatter' if config.use_sequence_parallel else 'all_reduce'
        payload = local_tokens * hidden * element_bytes if config.use_sequence_parallel else tokens * hidden * element_bytes
        values.append(self._communication(output_collective, payload, self._group('tp', rank), costs)[0])
        values.append(self._elementwise('mlp_norm_and_residual', local_tokens * hidden * 10,
            local_tokens * hidden * element_bytes * 5, phase, rank, config, request, costs, layer))
        if model.layer_types[layer] == LayerType.MOE:
            distribution = self._expert_tokens(rank, layer, config)
            if distribution is None:
                identity = dict(layer_id=layer, rank=rank, micro_batch=config.micro_batch_size,
                                sequence=config.sequence_length, tp=config.tp, cp=config.cp,
                                ep=config.ep, etp=config.etp)
                costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                    reason='当前MoE条件缺少实际Expert Token分布和发送字节矩阵',
                    profiling_request=dict(kind='operator_cost', identity=identity,
                                           cost_key=fingerprint(identity))))
                return None
            values.append(self._gemm('router', local_tokens, model.num_experts, hidden,
                                     phase, rank, config, request, costs, layer))
            counts = distribution['expert_token_counts']
            ep_group = self._group('ep', rank)
            matrix = distribution.get('send_bytes_matrix')
            if config.ep > 1 and matrix is None:
                costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                    reason='MoE通信缺少当前EP组实际Token发送字节矩阵',
                    profiling_request=dict(kind='nccl', identity=distribution,
                                           cost_key=fingerprint(distribution))))
                return None
            if config.ep > 1:
                values.append(self._communication('all_to_all', max(1, local_tokens * hidden * element_bytes),
                                                  ep_group, costs, matrix)[0])
            ffn = model.ffn_hidden_size // config.etp
            for count in counts:
                if count:
                    values.append(self._gemm('expert_gate_up', count, 2 * ffn, hidden,
                                             phase, rank, config, request, costs, layer))
                    values.append(self._gemm('expert_down', count, hidden, ffn,
                                             phase, rank, config, request, costs, layer))
            values.append(self._elementwise('expert_activation', sum(counts) * ffn * 8,
                sum(counts) * ffn * element_bytes * 3, phase, rank, config, request, costs, layer))
            if config.ep > 1:
                reverse = [list(row) for row in zip(*matrix)]
                values.append(self._communication('all_to_all', max(1, local_tokens * hidden * element_bytes),
                                                  ep_group, costs, reverse)[0])
            if model.shared_expert_size:
                shared = model.shared_expert_size // config.tp
                values.extend([self._gemm('shared_gate_up', tokens, 2 * shared, hidden,
                    phase, rank, config, request, costs, layer),
                    self._gemm('shared_down', tokens, hidden, shared,
                    phase, rank, config, request, costs, layer)])
        else:
            ffn = model.ffn_hidden_size // config.tp
            if config.use_sequence_parallel and config.tp > 1:
                values.append(self._communication('all_gather', payload, self._group('tp', rank), costs)[0])
            values.extend([self._gemm('gate_up', tokens, 2 * ffn, hidden, phase,
                                     rank, config, request, costs, layer),
                           self._elementwise('swiglu', tokens * ffn * 8,
                             tokens * ffn * element_bytes * 3, phase, rank, config, request, costs, layer),
                           self._gemm('down', tokens, hidden, ffn, phase,
                                      rank, config, request, costs, layer)])
            values.append(self._communication(output_collective, payload, self._group('tp', rank), costs)[0])
        return self._total(values)

    def _selective_recompute(self, layer, rank, config, request, costs):
        modules = set(resolve_recompute_effect(request, layer).modules)
        unsupported = modules - {'core_attn', 'mlp'}
        if (unsupported or ('mlp' in modules and self.model.layer_types[layer] == LayerType.MOE)
                or ('core_attn' in modules and config.cp > 1)):
            identity = dict(layer=layer, modules=sorted(modules), tp=config.tp, cp=config.cp,
                micro_batch=config.micro_batch_size, sequence=config.sequence_length)
            costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                reason='选择性重计算的原生复合执行需要相容模块测量',
                profiling_request=dict(kind='representative_module', identity=identity, cost_key=fingerprint(identity))))
            return None
        tokens = config.micro_batch_size * config.sequence_length // config.cp
        hidden, width = self.model.hidden_size, DTYPE_BYTES[self.dtype]
        local = tokens // config.tp if config.use_sequence_parallel else tokens
        values = []
        if 'core_attn' in modules:
            dimension = self.model.head_dimension or hidden // self.model.num_attention_heads
            features = dict(batch=config.micro_batch_size, query_heads=self.model.num_attention_heads // config.tp,
                kv_heads=max(1, (self.model.num_query_groups or self.model.num_attention_heads) // config.tp),
                query_length=config.sequence_length, key_length=config.sequence_length,
                head_dimension=dimension, value_dimension=dimension,
                causal=self.model_graph['module_structure'].get('is_causal', True),
                materializes_attention_scores=self.parameters.get('attention', {}).get('materializes_attention_scores', False))
            values.append(self._compute('attention', features, 'recomputation', rank, config, request, costs, layer_id=layer))
        if 'mlp' in modules:
            ffn = self.model.ffn_hidden_size // config.tp
            if config.use_sequence_parallel and config.tp > 1:
                values.append(self._communication('all_gather', local * hidden * width, self._group('tp', rank), costs)[0])
            values.extend((self._gemm('gate_up_recompute', tokens, 2 * ffn, hidden,
                'recomputation', rank, config, request, costs, layer),
                self._elementwise('swiglu_recompute', tokens * ffn * 8, tokens * ffn * width * 3,
                    'recomputation', rank, config, request, costs, layer),
                self._gemm('down_recompute', tokens, hidden, ffn,
                    'recomputation', rank, config, request, costs, layer)))
            operation = 'reduce_scatter' if config.use_sequence_parallel else 'all_reduce'
            values.append(self._communication(operation, local * hidden * width,
                self._group('tp', rank), costs)[0])
        return self._total(values)

    def _parameter_counts(self, config, regions):
        model, structure = self.model, self.model_graph['module_structure']
        metadata = self.model_graph.get('parameter_metadata')
        if metadata is not None:
            identity = (config.tp, config.etp, config.ep, config.pp, config.vpp, config.cp, config.dp,
                fingerprint({name: config.execution_implementation.get(name) for name in
                    ('execution_backend', 'optimizer_sharding', 'precision', 'main_grads_dtype',
                     'bucket_elements', 'overlap_grad_reduce', 'pad_buckets_for_high_nccl_busbw',
                     'num_distributed_optimizer_instances')}),
                tuple((region.layer_start, region.layer_end, region.pp_rank) for region in regions))
            if identity in self.parameter_counts_cache:
                return deepcopy(self.parameter_counts_cache[identity])
            execution = dict({name: value for name, value in config.execution_implementation.items()
                if value is not None},
                precision=self.dtype, main_grads_dtype=config.execution_implementation.get('main_grads_dtype'),
                tp=config.tp, pp=config.pp, cp=config.cp, dp=config.dp, ep=config.ep,
                etp=config.etp, expert_dp=config.world_size // (config.pp * config.ep * config.etp),
                node_count=1, gpus_per_node=config.world_size,
                rank_mapping=list(config.device_mapping) or list(range(config.world_size)))
            counts = checkpoint_parameter_counts(metadata, config, regions, execution)
            counts['padded_vocab_size'] = structure.get('padded_vocab_size', structure['vocab_size'])
            self.parameter_counts_cache[identity] = deepcopy(counts)
            return counts
        vocab = structure.get('padded_vocab_size', structure.get('vocab_size'))
        if type(vocab) is not int or vocab <= 0:
            return None
        if vocab % config.tp:
            vocab = math.ceil(vocab / config.tp) * config.tp
        heads = model.num_attention_heads // config.tp
        kv = max(1, (model.num_query_groups or model.num_attention_heads) // config.tp)
        dimension = model.head_dimension or model.hidden_size // model.num_attention_heads
        dense, expert = defaultdict(int), defaultdict(int)
        lanes = config.world_size // config.pp
        for rank in range(config.world_size):
            physical_stage = rank // lanes
            ids = [layer for region in regions if region.pp_rank == physical_stage
                   for layer in range(region.layer_start, region.layer_end)]
            for layer in ids:
                if model.module_kinds is not None:
                    measured = None if self.reusable_profile is None else self.reusable_profile.get('module_parameters')
                    if measured is None:
                        return None
                    if model.num_experts and not self.reusable_profile['module_expert_ownership_labeled']:
                        return None
                    counts = measured[str(layer)]
                    dense[rank] += counts['dense']
                    expert[rank] += counts['expert']
                    continue
                dense[rank] += model.hidden_size * ((heads + 2 * kv) * dimension + heads * dimension) + 2 * model.hidden_size
                if (model.finalize_grad_semantics or {}).get('qk_layernorm', False):
                    dense[rank] += 2 * dimension
                if model.layer_types[layer] == LayerType.MOE:
                    dense[rank] += model.hidden_size * model.num_experts
                    expert[rank] += 3 * model.hidden_size * model.ffn_hidden_size * model.num_experts // (config.etp * config.ep)
                    if model.shared_expert_size:
                        dense[rank] += 3 * model.hidden_size * model.shared_expert_size // config.tp
                else:
                    dense[rank] += 3 * model.hidden_size * model.ffn_hidden_size // config.tp
                if structure.get('attention_bias', structure.get('add_bias_linear', False)):
                    dense[rank] += (heads + 2 * kv) * dimension + model.hidden_size
                if structure.get('mlp_bias', structure.get('add_bias_linear', False)):
                    dense[rank] += 2 * model.ffn_hidden_size // config.tp + model.hidden_size
            boundary = {} if self.reusable_profile is None else self.reusable_profile.get('boundary_parameter_counts', {})
            category = 'both' if config.pp == 1 else 'first' if physical_stage == 0 else 'last' if physical_stage == config.pp - 1 else 'internal'
            if not model.num_experts and category in boundary:
                dense[rank] += boundary[category]
                continue
            if not model.num_experts and category == 'both' and all(key in boundary for key in ('first', 'last', 'internal')):
                dense[rank] += boundary['first'] + boundary['last'] - boundary['internal']
                if structure.get('tie_word_embeddings', False):
                    dense[rank] -= vocab * model.hidden_size // config.tp
                continue
            if physical_stage == 0:
                dense[rank] += vocab * model.hidden_size // config.tp
            if physical_stage == config.pp - 1:
                if config.pp > 1 or not structure.get('tie_word_embeddings', False):
                    dense[rank] += vocab * model.hidden_size // config.tp
                dense[rank] += model.hidden_size
        return dict(dense=dense, expert=expert, padded_vocab_size=vocab)

    def _state_memory(self, counts, config, request):
        ownership = parameter_state_ownership(counts, config, request)
        output = {}
        for rank, row in ownership.items():
            parameters = sum(item['persistent_parameter_bytes'] for item in row['components'].values())
            gradients = sum(item['persistent_gradient_bytes'] for item in row['components'].values())
            optimizer = sum(item['gpu_optimizer_bytes'] for item in row['components'].values())
            output[rank] = dict(parameter_bytes=parameters, gradient_bytes=gradients,
                optimizer_bytes=optimizer, persistent_bytes=row['gpu_persistent_bytes'],
                model_buffer_bytes=row.get('model_buffer_bytes', 0),
                host_optimizer_bytes=row['host_optimizer_bytes'],
                host_pinned_buffer_bytes=row['host_pinned_buffer_bytes'],
                fsdp_unshard_bytes=row['fsdp_unshard_bytes'],
                fsdp_padding_bytes=row['fsdp_padding_bytes'],
                fsdp_communication_unit_bytes=row['fsdp_communication_unit_bytes'],
                parameter_element_bytes=row['widths']['parameter'],
                gradient_element_bytes=row['widths']['gradient'],
                optimizer_element_bytes=(row['widths']['master'] + row['widths']['first_moment']
                    + row['widths']['second_moment']),
                ownership_components=row['components'], source=row['source'],
                assumptions=['Dense与Expert状态按实际进程组独立计算',
                    'Distributed Optimizer与Megatron-FSDP状态分片互斥'])
        return output

    def _activation_bytes(self, layers, rank, config, request):
        tokens = config.micro_batch_size * config.sequence_length // config.cp
        local = tokens // config.tp if config.use_sequence_parallel else tokens
        hidden, width = self.model.hidden_size, DTYPE_BYTES[self.dtype]
        if request['recompute'] == 'full':
            return (len(layers) + 1) * local * hidden * width
        if self.model.module_kinds is not None:
            self.memory_missing.add('hybrid_module_saved_activation_lifetimes')
            return (len(layers) + 1) * local * hidden * width
        total = (len(layers) + 1) * local * hidden * width
        dimension = self.model.head_dimension or hidden // self.model.num_attention_heads
        heads = self.model.num_attention_heads // config.tp
        kv = max(1, (self.model.num_query_groups or self.model.num_attention_heads) // config.tp)
        effect = resolve_recompute_effect(request, layers[0] if layers else -1, self.reusable_profile)
        recomputed = set(effect.modules)
        if request['recompute'] == 'selective-mlp' and request.get('mlp_recompute_every', 1) != 1:
            recomputed = set()
            self.memory_missing.add('partial_mlp_recomputation_activation_lifetimes')
        if recomputed:
            self.memory_missing.add('selective_saved_tensor_lifetimes:' + ','.join(sorted(recomputed)))
        for layer in layers:
            total += tokens * ((heads + 2 * kv) * dimension + heads * dimension) * width
            total += local * hidden * width * 2
            is_moe = self.model.layer_types[layer] == LayerType.MOE
            retains_ffn = ('moe' not in recomputed if is_moe else 'mlp' not in recomputed)
            if retains_ffn:
                if is_moe:
                    if 'moe_act' in recomputed:
                        self.memory_missing.add('partially_recomputed_moe_activation_lifetimes')
                    distribution = self._expert_tokens(rank, layer, config)
                    if distribution is None:
                        self.memory_missing.add('expert_activation_token_distribution')
                    else:
                        total += sum(distribution['expert_token_counts']) * self.model.ffn_hidden_size // config.etp * width * 3
                else:
                    total += tokens * self.model.ffn_hidden_size // config.tp * width * 3
            if (self.parameters.get('attention', {}).get('materializes_attention_scores', False)
                    and 'core_attn' not in recomputed):
                total += config.micro_batch_size * heads * config.sequence_length // config.cp * config.sequence_length * width
        return total

    def _native_tasks(self, graph, request, config, regions):
        if graph is None:
            graph = TaskGraphBuilder().build(self.model, config, regions, config.num_microbatches)
            source = dict(tasks={key: dict(rank=task.rank, operation=task.task_type.value,
                microbatch=task.microbatch_id, region_id=task.region_id,
                dependencies=task.dependencies, metadata=task.metadata) for key, task in graph.tasks.items()},
                representation='region_training_task_dag', source='same_TaskGraphBuilder_used_by_CompleteAndRewrite')
        else:
            source = deepcopy(graph)
        tasks = {}
        for key, row in source['tasks'].items():
            operation = row.get('operation', row.get('kind'))
            if operation not in {value.value for value in TaskType}:
                return source, None, f'执行图包含未支持任务类型：{operation}'
            tasks[key] = Task(key, row['rank'], TaskType(operation),
                microbatch_id=row.get('microbatch', row.get('microbatch_id')),
                region_id=row.get('region_id', row.get('region')),
                dependencies=list(row.get('dependencies', [])), metadata=deepcopy(row.get('metadata', {})))
        topological_order(tasks)
        lower_native_schedule(tasks, config, native_schedule_settings(request),
            lambda parent, child: tasks[child].dependencies.append(parent))
        return source, tasks, None

    def _select_reusable_profile(self, request):
        if self.mode == 'analytical_only':
            return None
        actual_uuids = {str(row.get('uuid', row.get('gpu_uuid'))).removeprefix('GPU-').lower()
                        for row in self.topology.devices if row.get('uuid', row.get('gpu_uuid')) is not None}
        if not actual_uuids or None in actual_uuids:
            return None
        candidates = []
        for row in self.compute_surface.database.reusable_compute_profiles(request, blind=self.blind):
            if {str(device.get('gpu_uuid')).removeprefix('GPU-').lower() for device in row['hardware']} != actual_uuids:
                continue
            encoded = self.compute_surface.database.connection.execute(
                'SELECT document FROM profiles WHERE id=?', (row['profile_id'],)).fetchone()
            document = json.loads(encoded[0])
            document['measurement_semantics'] = row['measurement_semantics']
            document['backend_identity_evidence'] = row['backend_identity_evidence']
            previous = document['request']
            if self.blind and same_training_plan(previous, request):
                continue
            if previous.get('rank_mapping', list(range(len(actual_uuids)))) != request.get(
                    'rank_mapping', list(range(len(actual_uuids)))):
                continue
            if previous['node_count'] != request['node_count'] or previous['gpus_per_node'] != request['gpus_per_node']:
                continue
            quality, perturbation = compute_profile_quality(document)
            candidates.append((same_training_plan(previous, request),
                quality, perturbation,
                document['request']['decoder_chunks'] == request['decoder_chunks'],
                sum(len(windows) for windows in document.get('measured_windows', {}).values()),
                row['profile_id'] == self.preferred_profile_id,
                row['profile_id'], document))
        selected = max(candidates, default=(None, None, None, None, None, None, '', None))[-1]
        if selected is None:
            return None
        if selected['id'] not in self.module_profiles:
            self.module_profiles[selected['id']] = enrich_module_profile(selected)
        return self.module_profiles[selected['id']]

    def _select_optimizer_profile(self, request):
        if self.mode == 'analytical_only':
            return None
        actual_uuids = {str(row.get('uuid', row.get('gpu_uuid'))).removeprefix('GPU-').lower()
                        for row in self.topology.devices if row.get('uuid', row.get('gpu_uuid')) is not None}
        target_key = optimizer_profile_key(request)
        candidates = []
        for identifier, encoded in self.compute_surface.database.connection.execute(
                'SELECT id, document FROM profiles ORDER BY id'):
            document = json.loads(encoded)
            if self.blind and same_training_plan(document['request'], request):
                continue
            if document.get('optimizer_component_costs') is None:
                continue
            if {str(device.get('gpu_uuid')).removeprefix('GPU-').lower()
                    for device in document['hardware']} != actual_uuids:
                continue
            if optimizer_profile_key(document['request']) != target_key:
                continue
            candidates.append((identifier == self.preferred_profile_id,
                document.get('anchor') is not None,
                sum(len(windows) for windows in document.get('measured_windows', {}).values()),
                identifier, document))
        selected = max(candidates, default=(None, None, None, '', None))[-1]
        if selected is None:
            return None
        if selected['id'] not in self.module_profiles:
            self.module_profiles[selected['id']] = enrich_module_profile(selected)
        return self.module_profiles[selected['id']]

    def _native_optimizer_cost(self, rank, counts, config, request, costs):
        physical = self.groups['rank_mapping'][rank]
        observation = self.parameters.get('optimizer_offload', {}).get(str(physical))
        if self.blind and observation is not None and observation.get('source_plan_signature') in (
                None, self.current_plan_signature):
            observation = None
        owned = counts['dense'][rank] + counts['expert'][rank]
        sharded = request.get('optimizer_sharding', 'distributed') == 'distributed'
        if sharded:
            owned = counts['dense'][rank] / len(self._group('dp_cp', rank))
            owned += counts['expert'][rank] / len(self._group('expert_dp', rank))
        if (observation is not None and observation.get('source')
                and observation.get('cost_scope') == 'local_optimizer_update_and_copy'
                and observation.get('parameter_count') == owned):
            value = observation['milliseconds']
            costs.append(dict(status='MEASURED_EXACT', milliseconds=value,
                source=observation['source'], evidence_ids=observation.get('evidence_ids', [])))
            return value
        profile = self.optimizer_profile
        if profile is not None:
            if profile['id'] not in self.optimizer_profiles:
                self.optimizer_profiles[profile['id']] = optimizer_local_cost_evidence(profile)
            local_evidence = self.optimizer_profiles[profile['id']]
            if not local_evidence['verified']:
                identity = dict(operation='local_optimizer_update_and_copy',
                    physical_rank=physical, parameter_count=owned,
                    optimizer=request.get('optimizer', 'adam'),
                    optimizer_sharding=request.get('optimizer_sharding', 'distributed'))
                costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                    cost_key=fingerprint(identity),
                    reason=local_evidence['reason'], evidence_ids=[profile['id']],
                    rejected_cost_scope=profile['measurement_semantics']['optimizer']['scope'],
                    profiling_request=dict(kind='operator_cost', identity=identity,
                        excluded_operations=['gradient_norm_collectives', 'overflow_collectives',
                            'parameter_gather', 'upstream_stream_waits'],
                        cost_key=fingerprint(identity))))
                return None
        if profile is not None and bool(profile['request'].get('offload')) == bool(request.get('offload')):
            old_config = configuration(profile['request'])
            old_regions = RegionPartitioner(self.model, old_config).from_counts(profile['request']['decoder_chunks'])
            old_counts = self._parameter_counts(old_config, old_regions)
            old_groups = process_groups(profile['request'])['logical']
            samples = defaultdict(list)
            for previous_rank in range(old_config.world_size):
                previous_owned = old_counts['dense'][previous_rank] + old_counts['expert'][previous_rank]
                if sharded:
                    previous_owned = old_counts['dense'][previous_rank] / len(next(
                        group for group in old_groups['dp_cp'] if previous_rank in group))
                    previous_owned += old_counts['expert'][previous_rank] / len(next(
                        group for group in old_groups['expert_dp'] if previous_rank in group))
                samples[previous_owned].append(local_evidence['costs_by_rank'][str(previous_rank)])
            shapes = sorted(samples)
            if shapes[0] <= owned <= shapes[-1]:
                if owned in samples:
                    value = statistics.median(samples[owned])
                else:
                    right = bisect_left(shapes, owned)
                    low, high = shapes[right - 1], shapes[right]
                    fraction = (owned - low) / (high - low)
                    value = statistics.median(samples[low]) * (1 - fraction) + statistics.median(samples[high]) * fraction
                costs.append(dict(status='INTERPOLATED', milliseconds=value,
                    source='actual_native_optimizer_cost_within_measured_parameter_ownership_range',
                    evidence_ids=local_evidence['evidence_ids'], parameter_count=owned,
                    calibration=dict(shape_range=[shapes[0], shapes[-1]], measured_shapes=shapes),
                    cost_scope='local_optimizer_update_and_copy',
                    assumptions=['相同实际GPU集合对应的原生Optimizer执行条件',
                        '按本地Optimizer拥有的参数数量在实际测量范围内插值；共享资源竞争保持原测量条件']))
                return value
            nearest = min(shapes, key=lambda shape: abs(math.log2(owned / shape)))
            distance = abs(math.log2(owned / nearest))
            if distance <= 2:
                measured = statistics.median(samples[nearest])
                value = measured * owned / nearest
                costs.append(dict(status='RATIO_CALIBRATED', milliseconds=value,
                    source='actual_native_optimizer_parameter_ownership_ratio',
                    evidence_ids=local_evidence['evidence_ids'], parameter_count=owned,
                    calibration=dict(measured_parameter_count=nearest,
                        target_parameter_count=owned, maximum_log2_distance=2),
                    cost_scope='native_optimizer_step_parameter_proportional_projection',
                    assumptions=['相同Optimizer、精度、分片实现和实际GPU集合',
                        'Optimizer更新成本按本地拥有参数数量进行实测比例投影'],
                    profiling_request=dict(kind='operator_cost',
                        identity=dict(physical_rank=physical, parameter_count=owned,
                            optimizer_sharding=request.get('optimizer_sharding', 'distributed')),
                        cost_key=fingerprint(dict(kind='optimizer_step', physical_rank=physical,
                            parameter_count=owned, optimizer_sharding=request.get(
                                'optimizer_sharding', 'distributed'))))))
                return value
        identity = dict(physical_rank=physical, parameter_count=owned,
                        optimizer_sharding=request.get('optimizer_sharding', 'distributed'))
        costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
            reason='原生Optimizer缺少当前参数所有权范围内的更新及传输成本',
            profiling_request=dict(kind='operator_cost', identity=identity, cost_key=fingerprint(identity))))
        return None

    def _optimizer_component_cost(self, operation, rank, elements, request, costs):
        identity = dict(operation=operation, physical_rank=self.groups['rank_mapping'][rank],
            parameter_count=elements, optimizer=request.get('optimizer', 'adam'),
            optimizer_sharding=request.get('optimizer_sharding', 'distributed'))
        cost_key = fingerprint(identity)
        profile = self.optimizer_profile
        evidence = None
        if profile is not None:
            cache = self.__dict__.setdefault('optimizer_component_cache', {})
            key = (profile['id'], operation)
            if key not in cache:
                cache[key] = optimizer_component_cost_evidence(profile, operation)
            evidence = cache[key]
        reason = '当前执行条件缺少独立Optimizer组件成本'
        if evidence is not None and evidence['verified']:
            samples = defaultdict(list)
            for previous_rank, value in evidence['costs_by_rank'].items():
                samples[evidence['elements_by_rank'][previous_rank]].append(value)
            shapes = sorted(samples)
            if elements in samples:
                value, status = statistics.median(samples[elements]), 'MEASURED_EXACT'
            elif shapes[0] < elements < shapes[-1]:
                position = bisect_left(shapes, elements)
                low, high = shapes[position - 1], shapes[position]
                fraction = (elements - low) / (high - low)
                value = statistics.median(samples[low]) * (1 - fraction) + statistics.median(samples[high]) * fraction
                status = 'INTERPOLATED'
            else:
                positive = [shape for shape in shapes if shape > 0]
                nearest = min(positive, key=lambda shape: abs(math.log2(elements / shape))) if elements > 0 and positive else None
                if nearest is None or not 0.5 <= elements / nearest <= 2:
                    value = None
                    reason = 'Optimizer组件参数数量超过实测范围两倍，需要补充对应范围测量'
                else:
                    value = statistics.median(samples[nearest]) * elements / nearest
                    status = 'RATIO_CALIBRATED'
            if value is not None:
                costs.append(dict(status=status, milliseconds=value, cost_key=cost_key,
                    identity=identity, source='actual_independent_optimizer_component',
                    cost_scope=operation, evidence_ids=evidence['evidence_ids'],
                    measured_parameter_range=[shapes[0], shapes[-1]],
                    calibration_scope='actual_component_parameter_ownership_interpolation_or_bounded_ratio'))
                return value
        elif evidence is not None:
            reason = evidence['reason']
        costs.append(dict(status='NEEDS_PROFILE', milliseconds=None, cost_key=cost_key,
            identity=identity, reason=reason,
            profiling_request=dict(kind='optimizer_component', cost_key=cost_key, identity=identity)))
        return None

    def _region_overhead(self, region_id, phase, config, costs):
        if self.reusable_profile is None:
            return None
        value, category = boundary_costs(self.reusable_profile, region_id, config, phase)
        if value is None:
            return None
        costs.append(dict(status='MEASURED_REUSED', milliseconds=value, phase=phase,
            cost_key=fingerprint(dict(profile=self.reusable_profile['id'],
                operation='region_composite_overhead', category=category, phase=phase)),
            boundary_category=category,
            projection_operation='region_composite_overhead', source='actual_region_cost_minus_actual_layer_costs',
            cost_scope='measured_special_module_and_region_execution_overhead',
            evidence_ids=[self.reusable_profile['id']],
            assumptions=['按首个、最后与内部Region分别复用相同计算形状下的复合执行开销']))
        return value

    def _memory(self, tasks, times, states, config, request):
        memory = self.parameters.get('memory', {})
        reserve = memory.get('reserve_bytes')
        workspace = memory.get('workspace_bytes')
        if reserve is None or not memory.get('reserve_source'):
            self.memory_missing.add('allocator_and_external_reserve')
            reserve = 0
        if workspace is None or not memory.get('workspace_source'):
            self.memory_missing.add('kernel_workspace')
            workspace = 0
        if type(reserve) is not int or reserve < 0 or type(workspace) is not int or workspace < 0:
            raise ValueError('Workspace和安全余量必须使用非负整数字节数')
        events, lifetimes = defaultdict(list), []
        backward = {(task.rank, task.region_id, task.microbatch_id): key for key, task in tasks.items()
                    if task.task_type == TaskType.BACKWARD_COMPUTE}
        receives = {parent: key for key, task in tasks.items() if task.task_type == TaskType.PP_P2P_RECV
                    for parent in task.dependencies if tasks[parent].task_type == TaskType.PP_P2P_SEND}
        receive_ids = set(receives.values())
        consumers = {task.metadata['input_task']: key for key, task in tasks.items()
                     if task.metadata.get('input_task') in receive_ids}
        communication_workspace = memory.get('communication_workspace_bytes')
        if communication_workspace is None or not memory.get('communication_workspace_source'):
            self.memory_missing.add('nccl_internal_workspace')
            communication_workspace = 0
        if type(communication_workspace) is not int or communication_workspace < 0:
            raise ValueError('通信Workspace必须使用非负整数字节数')

        def buffer(identity, ranks, size, start, end, kind, aliases=None):
            if size < 0 or end < start:
                raise ValueError('动态Buffer大小或者生命周期非法')
            if size == 0 or end == start:
                return
            for rank in ranks:
                events[rank].extend(((start, 1, size, identity), (end, 0, -size, identity)))
            lifetimes.append(dict(buffer_id=identity, logical_ranks=ranks,
                physical_ranks=[self.groups['rank_mapping'][rank] for rank in ranks], bytes=size,
                start_ms=start, end_ms=end, kind=kind, aliases=aliases or [], source='analytical_dag_tensor_lifetime'))

        for key, task in tasks.items():
            start, end = times[key]
            if task.task_type in (TaskType.FSDP_UNSHARD, TaskType.FSDP_PREFETCH) and task.metadata.get(
                    'cost_role') != 'prefetch_wait':
                compute = task.metadata['compute_task']
                release = next((identity for identity, candidate in tasks.items()
                    if candidate.task_type == TaskType.FSDP_RESHARD
                    and candidate.metadata.get('releases') == key), None)
                if release is None:
                    release = next((identity for identity, candidate in tasks.items()
                        if candidate.task_type == TaskType.FSDP_GRAD_REDUCE_SCATTER
                        and candidate.metadata.get('state_family') == task.metadata.get('state_family')
                        and candidate.metadata.get('compute_task') == compute), compute)
                buffer(f'fsdp_unshard:{key}', [task.rank], task.metadata['payload_bytes'],
                    end, times[release][1], 'fsdp_unsharded_parameter',
                    aliases=['persistent_parameter_shard'])
            if end <= start:
                continue
            if task.task_type == TaskType.FORWARD_COMPUTE:
                release = backward[(task.rank, task.region_id, task.microbatch_id)]
                buffer(f'activation:{key}', [task.rank], task.metadata['activation_bytes'],
                       start, times[release][1], 'saved_activation',
                       aliases=['pipeline_receive_input_is_saved_region_input'])
            if task.task_type in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE, TaskType.OPTIMIZER_STEP):
                buffer(f'workspace:{key}', [task.rank], workspace, start, end, 'kernel_workspace')
            if task.task_type == TaskType.PP_P2P_SEND:
                receive = receives[key]
                consumer = consumers[receive]
                destination = tasks[consumer].rank
                size = task.metadata['communication_buffer_bytes']
                if task.metadata['direction'] == 'forward':
                    buffer(f'pipeline_receive:{key}', [destination], size, start, times[consumer][0],
                           'pipeline_input_before_consumer', aliases=[f'activation:{consumer}'])
                else:
                    producer = task.metadata['producer_task']
                    buffer(f'pipeline_gradient_send:{key}', [task.rank], size,
                           times[producer][1], end, 'pipeline_output_gradient')
                    buffer(f'pipeline_gradient_receive:{key}', [destination], size,
                           start, times[consumer][1], 'pipeline_input_gradient')
            if task.is_communication:
                buffer(f'nccl_workspace:{key}', task.metadata.get('participant_ranks', [task.rank]),
                       communication_workspace, start, end, 'communication_workspace',
                       aliases=task.metadata.get('buffer_aliases'))
            if task.task_type == TaskType.FSDP_GRAD_REDUCE_SCATTER:
                buffer(f'fsdp_gradient_bucket:{key}', task.metadata['group_ranks'],
                    task.metadata['payload_bytes'], start, end, 'fsdp_gradient_bucket')
        peaks, allocated_peaks, breakdown, headroom = {}, {}, {}, {}
        for rank in range(config.world_size):
            live, peak, peak_time, active, peak_buffers = 0, 0, 0.0, set(), []
            for stamp, order, delta, identity in sorted(events[rank]):
                live += delta
                if order:
                    active.add(identity)
                else:
                    active.remove(identity)
                if live < 0:
                    raise ValueError('动态Buffer生命周期出现提前释放')
                if live > peak:
                    peak, peak_time, peak_buffers = live, stamp, sorted(active)
            if live != 0:
                raise ValueError('完整Step结束后仍有未释放动态Buffer')
            physical = self.groups['rank_mapping'][rank]
            value = states[rank]['persistent_bytes'] + peak + reserve
            allocated_peaks[physical] = states[rank]['persistent_bytes'] + peak
            peaks[physical] = value
            breakdown[physical] = dict(states[rank], dynamic_peak_bytes=peak,
                reserve_bytes=reserve, peak_time_ms=peak_time, peak_buffer_ids=peak_buffers)
            headroom[physical] = int(self.topology.memory_per_gpu_gb * 2**30) - value
        capacity = int(self.topology.memory_per_gpu_gb * 2**30)
        rejected = (any(row['parameter_bytes'] + row['gradient_bytes'] > capacity for row in states.values())
                    or (not self.memory_missing and any(value < 0 for value in headroom.values())))
        admission = 'REJECT_OOM' if rejected else 'REQUIRE_MEMORY_PROFILE' if self.memory_missing else 'WITHIN_ESTIMATED_CAPACITY'
        return dict(peak_per_rank_bytes=peaks, breakdown_per_rank=breakdown,
            peak_allocated_per_rank_bytes=allocated_peaks,
            memory_headroom_per_rank_bytes=headroom, admission=admission,
            status=('REJECTED_MEMORY_LOWER_BOUND' if rejected else
                'NEEDS_MEMORY_VALIDATION' if self.memory_missing else 'PREDICTED_SAFE'),
            physical_memory_status='MEMORY_UNCERTAIN' if self.memory_missing else 'ANALYTICAL',
            missing_profiles=sorted(self.memory_missing), buffer_lifetimes=lifetimes,
            scope='analytical_parameter_ownership_and_projection_dag_buffer_lifetimes',
            confidence_interval_bytes=None,
            host_cpu_memory_peak_per_rank_bytes={self.groups['rank_mapping'][rank]:
                states[rank].get('host_optimizer_bytes', 0) + states[rank].get('host_pinned_buffer_bytes', 0)
                for rank in range(config.world_size)})

    def memory_bound(self, request, *, blind=False):
        self.memory_missing = set()
        self.blind = blind
        self.current_plan_signature = plan_signature(request)
        self.dtype = request.get('precision', 'bfloat16').replace('torch.', '')
        if self.dtype not in DTYPE_BYTES:
            raise ValueError('解析训练状态需要明确的bfloat16、float16或float32精度')
        config = configuration(request)
        self.groups = process_groups(request)
        self.reusable_profile = self._select_reusable_profile(request)
        self.optimizer_profile = self._select_optimizer_profile(request)
        self.pipeline_dtype = request.get('pipeline_dtype',
            self.reusable_profile.get('pipeline_dtype', self.dtype) if self.reusable_profile else self.dtype)
        regions = RegionPartitioner(self.model, config).from_counts(request['decoder_chunks'])
        counts = self._parameter_counts(config, regions)
        if counts is None:
            reason = ('异构模型显存准入需要Checkpoint参数元数据或实测module_parameters'
                if self.model.module_kinds is not None else '解析显存上界需要模型Vocabulary与FFN维度')
            return dict(peak_per_rank_bytes={}, admission='REQUIRE_MEMORY_PROFILE',
                status='UNAVAILABLE_PARAMETER_OWNERSHIP', physical_memory_status='UNKNOWN',
                missing_profiles=[reason], scope='parameter_ownership_not_available_for_capacity_bound',
                confidence_interval_bytes=None)
        states = self._state_memory(counts, config, request)
        memory = self.parameters.get('memory', {})
        reserve = memory.get('reserve_bytes')
        workspace = memory.get('workspace_bytes')
        communication_workspace = memory.get('communication_workspace_bytes')
        missing = []
        if reserve is None or not memory.get('reserve_source'):
            missing.append('allocator_reserve')
            reserve = 0
        if workspace is None or not memory.get('workspace_source'):
            missing.append('kernel_workspace')
            workspace = 0
        if communication_workspace is None or not memory.get('communication_workspace_source'):
            missing.append('nccl_internal_workspace')
            communication_workspace = 0
        if any(type(value) is not int or value < 0
                for value in (reserve, workspace, communication_workspace)):
            raise ValueError('显存Workspace与安全余量必须使用非负整数字节数')
        lanes = config.world_size // config.pp
        payload = config.sequence_length // config.cp * config.micro_batch_size * self.model.hidden_size * DTYPE_BYTES[self.pipeline_dtype]
        capacity = int(self.topology.memory_per_gpu_gb * 2**30)
        host_capacity = memory.get('host_capacity_bytes_per_node')
        request = dict(request, gpu_capacity_bytes=capacity, host_capacity_bytes=host_capacity)
        checkpoint_buffer = memory.get('checkpoint_buffer_bytes')
        checkpoint_buffer_source = memory.get('checkpoint_buffer_source')
        if checkpoint_buffer is None or not checkpoint_buffer_source:
            checkpoint_buffer = 0
            if request.get('checkpoint'):
                missing.append('checkpoint_load_buffer')
        activation_by_rank = {}
        for rank in range(config.world_size):
            stage = rank // lanes
            local_regions = [region for region in regions if region.pp_rank == stage]
            activation = sum(self._activation_bytes(
                list(range(region.layer_start, region.layer_end)), rank, config, request)
                for region in local_regions) * config.num_microbatches
            activation_by_rank[rank] = activation + payload * 4 + communication_workspace
        phases = phased_memory_bounds(counts, config, request, activation_by_rank,
            workspace, reserve, checkpoint_buffer_bytes=checkpoint_buffer)
        peaks, breakdown, headroom = {}, {}, {}
        for rank, row in phases.items():
            physical = self.groups['rank_mapping'][rank]
            peak = row['most_dangerous_gpu_bytes']
            peaks[physical] = peak
            headroom[physical] = capacity - peak
            breakdown[physical] = dict(states[rank], **row,
                activation_upper_bound_bytes=activation_by_rank[rank],
                pipeline_payload_upper_bound_bytes=payload * 4,
                communication_workspace_bytes=communication_workspace,
                kernel_workspace_bytes=workspace, reserve_bytes=reserve)
        host_by_node = defaultdict(int)
        for rank, row in phases.items():
            physical = self.groups['rank_mapping'][rank]
            host_by_node[physical // self.topology.gpus_per_node] += row['host_cpu_memory_peak_bytes']
        persistent = {self.groups['rank_mapping'][rank]: states[rank]['persistent_bytes']
            for rank in range(config.world_size)}
        rejected_ranks = sorted(rank for rank, value in persistent.items() if value + reserve > capacity)
        rejected = bool(rejected_ranks)
        predicted_over = any(value < 0 for value in headroom.values())
        near_capacity = min(headroom.values()) < max(reserve, capacity // 10)
        host_over = host_capacity is not None and any(value > host_capacity for value in host_by_node.values())
        needs_validation = bool(missing or predicted_over or near_capacity or host_over)
        admission = ('REJECT_OOM' if rejected else 'REQUIRE_MEMORY_PROFILE'
            if needs_validation else 'WITHIN_ESTIMATED_CAPACITY')
        status = ('REJECTED_MEMORY_LOWER_BOUND' if rejected else
            'NEEDS_MEMORY_VALIDATION' if needs_validation else 'PREDICTED_SAFE')
        return dict(peak_per_rank_bytes=peaks, breakdown_per_rank=breakdown,
            persistent_per_rank_bytes=persistent, rejected_ranks=rejected_ranks,
            rejection_basis='persistent_state_plus_configured_reserve' if rejected else None,
            memory_headroom_per_rank_bytes=headroom, admission=admission,
            status=status,
            physical_memory_status='MEMORY_UNCERTAIN' if missing else 'ANALYTICAL_UPPER_BOUND',
            scope=('partial_memory_components_require_measurement' if missing
                else 'analytical_state_activation_pipeline_workspace_upper_bound'),
            confidence_interval_bytes=None,
            gpu_training_peak_per_rank_bytes={self.groups['rank_mapping'][rank]:
                row['gpu_training_peak_bytes'] for rank, row in phases.items()},
            gpu_loading_peak_per_rank_bytes={self.groups['rank_mapping'][rank]:
                row['gpu_loading_peak_bytes'] for rank, row in phases.items()},
            gpu_optimizer_peak_per_rank_bytes={self.groups['rank_mapping'][rank]:
                row['gpu_optimizer_peak_bytes'] for rank, row in phases.items()},
            host_cpu_memory_peak_per_rank_bytes={self.groups['rank_mapping'][rank]:
                row['host_cpu_memory_peak_bytes'] for rank, row in phases.items()},
            host_cpu_memory_peak_per_node_bytes=dict(host_by_node),
            host_cpu_memory_capacity_per_node_bytes=host_capacity,
            missing_profiles=missing,
            most_dangerous_rank=max(peaks, key=peaks.get),
            most_dangerous_phase=breakdown[max(peaks, key=peaks.get)]['most_dangerous_phase'])

    def prepare_cost_surfaces(self, request, cost_mode='hybrid', *, blind=False):
        if cost_mode not in ('hybrid', 'measured_only', 'analytical_only'):
            raise ValueError('预测成本模式需要hybrid、measured_only或者analytical_only')
        self.mode, self.cost_cache, self.memory_missing, self.blind = cost_mode, {}, set(), blind
        self.current_plan_signature = plan_signature(request)
        network = self.parameters.get('network', {})
        default_topology = dict(devices=self.topology.devices,
            software=self.topology.software, environment=network.get('environment', {}),
            environment_per_node=network.get('environment_per_node', self.topology.environment_per_node),
            paths=self.topology.network_paths)
        if self.topology.environment_per_rank:
            default_topology['environment_per_rank'] = self.topology.environment_per_rank
        self.active_communication_topology = bind_candidate_execution_conditions(
            network.get('topology_identity', default_topology), request)
        self.calibration_run_ids = None
        if blind:
            registered = [json.loads(row[0])['request'] for row in
                          self.compute_surface.database.connection.execute('SELECT document FROM profiles')]
            self.calibration_run_ids = tuple(row['run_id'] for row in registered
                if not same_training_plan(row, request))
        self.dtype = request.get('precision', 'bfloat16').replace('torch.', '')
        if self.dtype not in DTYPE_BYTES:
            raise ValueError('解析训练状态需要明确的bfloat16、float16或float32精度')
        config = configuration(request)
        self.groups = process_groups(request)
        self.reusable_profile = self._select_reusable_profile(request)
        self.optimizer_profile = self._select_optimizer_profile(request)
        self.pipeline_dtype = request.get('pipeline_dtype',
            self.reusable_profile.get('pipeline_dtype', self.dtype) if self.reusable_profile else self.dtype)
        if config.world_size != self.topology.num_gpus:
            raise ValueError('预测计划与实际硬件GPU数量不同')
        return config

    def project(self, request, candidate_graph=None, parent_request=None, cost_mode='hybrid', *, blind=False):
        started = time.perf_counter()
        config = self.prepare_cost_surfaces(request, cost_mode, blind=blind)
        regions = RegionPartitioner(self.model, config).from_counts(request['decoder_chunks'])
        costs, tasks = [], {}
        source, tasks, unsupported = self._native_tasks(candidate_graph, request, config, regions)
        if unsupported is None and request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
            self._coalesce_fsdp_collectives(tasks)
        mtp_layers = self.model_graph.get('special_modules', {}).get('mtp_layers', 0)
        if unsupported is not None:
            return self._report(request, source, costs, started, 'UNSUPPORTED', unsupported)
        if self.model.ffn_hidden_size is None:
            return self._report(request, source, costs, started, 'NEEDS_PROFILE',
                                '实际模型配置缺少Vocabulary或FFN维度')
        counts = self._parameter_counts(config, regions)
        if counts is None:
            if self.model.module_kinds is not None:
                identity = dict(model=self.model.name, module_kinds=self.model.module_kinds,
                    tp=config.tp, cp=config.cp, ep=config.ep, etp=config.etp,
                    sequence=config.sequence_length, micro_batch=config.micro_batch_size)
                costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                    reason='PARAM_METADATA_MISSING: 混合模块缺少逐参数Ledger或module_parameters',
                    reason_code='PARAM_METADATA_MISSING',
                    profiling_request=dict(kind='parameter_metadata', identity=identity,
                        cost_key=fingerprint(identity))))
            return self._report(request, source, costs, started, 'NEEDS_PROFILE',
                                'PARAM_METADATA_MISSING: 实际模型缺少适用的参数归属记录')
        states = self._state_memory(counts, config, request)
        layer_cache = {}
        for key, task in tasks.items():
            task_cost_start = len(costs)
            reused_components = []
            physical = self.groups['rank_mapping'][task.rank]
            task.resource_requirements = (f'gpu:{physical}:compute',)
            task.metadata['duration_evidence'] = 'projection_cost_composition'
            if task.metadata.get('collective_representative') is False:
                task.duration = 0.0
                task.resource_requirements = ()
                task.metadata['duration_evidence'] = 'shared_collective_instance_alias'
            elif task.task_type in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE):
                phase = 'forward' if task.task_type == TaskType.FORWARD_COMPUTE else 'backward'
                region = regions[task.region_id]
                layers = list(range(region.layer_start, region.layer_end))
                durations = []
                for layer in layers:
                    cache_key = (layer, phase, task.rank)
                    if cache_key not in layer_cache:
                        layer_cost_start = len(costs)
                        value = self._layer(layer, phase, task.rank, config, request, costs)
                        layer_cache[cache_key] = (value, self._cost_components(costs[layer_cost_start:]))
                    else:
                        reused_components.extend(layer_cache[cache_key][1])
                    durations.append(layer_cache[cache_key][0])
                    if phase == 'backward' and request['recompute'] == 'full' and self.reusable_profile is None:
                        recompute_key = (layer, 'recomputation', task.rank)
                        if recompute_key not in layer_cache:
                            layer_cost_start = len(costs)
                            value = self._layer(layer, 'recomputation', task.rank, config, request, costs)
                            layer_cache[recompute_key] = (value, self._cost_components(costs[layer_cost_start:]))
                        else:
                            reused_components.extend(layer_cache[recompute_key][1])
                        durations.append(layer_cache[recompute_key][0])
                    elif (phase == 'backward' and request['recompute'] in ('selective', 'selective-mlp')
                            and self.reusable_profile is None):
                        recompute_key = (layer, 'selective_recomputation', task.rank)
                        if recompute_key not in layer_cache:
                            layer_cost_start = len(costs)
                            value = self._selective_recompute(layer, task.rank, config, request, costs)
                            layer_cache[recompute_key] = (value, self._cost_components(costs[layer_cost_start:]))
                        else:
                            reused_components.extend(layer_cache[recompute_key][1])
                        durations.append(layer_cache[recompute_key][0])
                task.duration = self._total(durations)
                task.metadata.update(layer_ids=layers, projection_only_cost_composition=True,
                    internal_collective_scope='region_composite_internal_communication_estimate',
                    recompute_policy=request['recompute'],
                    recompute_modules=list(resolve_recompute_effect(request, layers[0] if layers else -1).modules),
                    measured_composite_includes_recomputation=self.reusable_profile is not None,
                    activation_bytes=self._activation_bytes(layers, task.rank, config, request))
                overhead = self._region_overhead(task.region_id, phase, config, costs)
                if overhead is not None:
                    task.duration = self._total((task.duration, overhead))
                if task.region_id == 0 and overhead is None:
                    tokens = config.micro_batch_size * config.sequence_length // config.cp
                    lookup = self._elementwise('embedding_lookup_and_gradient_scatter',
                        tokens * self.model.hidden_size,
                        tokens * self.model.hidden_size * DTYPE_BYTES[self.dtype] * 2,
                        phase, task.rank, config, request, costs, None)
                    collective = ('reduce_scatter' if config.use_sequence_parallel else 'all_reduce')
                    local_tokens = tokens // config.tp if config.use_sequence_parallel else tokens
                    embed_comm = self._communication(collective,
                        local_tokens * self.model.hidden_size * DTYPE_BYTES[self.dtype],
                        self._group('tp', task.rank), costs)[0]
                    task.duration = self._total((task.duration, lookup, embed_comm))
                # 输出投影按最后Stage实际Vocabulary矩阵Shape计算。
                if task.region_id == len(regions) - 1 and overhead is None:
                    tokens = config.micro_batch_size * config.sequence_length // config.cp
                    output = self._gemm('output_logits', tokens, counts['padded_vocab_size'] // config.tp,
                        self.model.hidden_size, phase, task.rank, config, request, costs, None)
                    loss = self._elementwise('cross_entropy', tokens * counts['padded_vocab_size'] // config.tp * 8,
                        tokens * counts['padded_vocab_size'] // config.tp * DTYPE_BYTES[self.dtype] * 3,
                        phase, task.rank, config, request, costs, None)
                    task.duration = self._total((task.duration, output, loss))
                    if phase == 'forward':
                        losses = [self._communication('all_reduce', tokens * 4,
                            self._group('tp', task.rank), costs, dtype='float32')[0]
                            for _ in range(3)]
                    else:
                        local_tokens = tokens // config.tp if config.use_sequence_parallel else tokens
                        collective = 'reduce_scatter' if config.use_sequence_parallel else 'all_reduce'
                        losses = [self._communication(collective,
                            local_tokens * self.model.hidden_size * DTYPE_BYTES[self.dtype],
                            self._group('tp', task.rank), costs)[0]]
                    task.duration = self._total([task.duration, *losses])
                    if mtp_layers:
                        identity = dict(module_kind='mtp', depth=mtp_layers, phase=phase,
                            tp=config.tp, cp=config.cp, sequence=config.sequence_length,
                            micro_batch=config.micro_batch_size)
                        costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                            reason='MTP需要独立实测或包含MTP的末尾Region复合成本',
                            profiling_request=dict(kind='representative_module', identity=identity,
                                cost_key=fingerprint(identity))))
                        task.duration = None
            elif task.task_type == TaskType.NCCL_GENERIC:
                group = task.metadata.get('group_ranks')
                payload = task.metadata.get('payload_bytes')
                collective = task.metadata.get('collective')
                if (not group or type(payload) is not int or payload <= 0
                        or not collective or not task.metadata.get('dtype')):
                    return self._report(request, source, costs, started, 'UNSUPPORTED',
                        f'FinalizeModelGrads通信任务缺少完整语义：{key}')
                concurrency = tuple([task.metadata['concurrent_collective_family']]
                    if task.metadata.get('concurrent_collective_family') else [])
                task.duration, resources = self._communication(collective, payload,
                    group, costs, dtype=task.metadata['dtype'], concurrency=concurrency)
                measured_concurrent = bool(concurrency and costs[-1].get('time_source') == 'MEASURED')
                task.resource_requirements = (tuple(
                    f'gpu:{self.groups["rank_mapping"][rank]}:communication' for rank in group)
                    if measured_concurrent else tuple(resources) or tuple(
                    f'gpu:{self.groups["rank_mapping"][rank]}:communication' for rank in group))
                task.metadata.update(duration_evidence=costs[-1].get('source'),
                    cost_includes_shared_resource_contention=measured_concurrent,
                    communication_phases=costs[-1].get('communication_phases', []))
            elif task.task_type == TaskType.PP_P2P_SEND:
                size = math.prod(task.metadata['shape']) * DTYPE_BYTES[self.pipeline_dtype]
                task.duration, resources = self._communication('p2p', size, task.metadata['group_ranks'], costs,
                    dtype=self.pipeline_dtype)
                task.metadata['pipeline_dtype'] = self.pipeline_dtype
                task.metadata['payload_bytes'] = size
                task.metadata['physical_group_ranks'] = [self.groups['rank_mapping'][rank] for rank in task.metadata['group_ranks']]
                task.resource_requirements = tuple(resources) or tuple(
                    f'gpu:{self.groups["rank_mapping"][rank]}:communication'
                    for rank in task.metadata['group_ranks'])
                if task.metadata['native_launch_policy'] == 'blocking':
                    task.resource_requirements += (f'gpu:{physical}:compute',)
                task.metadata['communication_buffer_bytes'] = size
                task.metadata['participant_ranks'] = task.metadata['group_ranks']
                task.metadata['communication_phases'] = costs[-1].get('communication_phases', [])
            elif task.task_type == TaskType.FSDP_UNSHARD and task.metadata.get('cost_role') == 'prefetch_wait':
                task.duration = 0.0
                task.resource_requirements = ()
            elif task.task_type in (TaskType.FSDP_UNSHARD, TaskType.FSDP_PREFETCH,
                    TaskType.FSDP_GRAD_REDUCE_SCATTER,
                    TaskType.FSDP_OUTER_GRAD_REDUCE, TaskType.FSDP_OUTER_PARAM_SYNC):
                collective = task.metadata['collective']
                communication_dtype = task.metadata['communication_dtype']
                if task.task_type in (TaskType.FSDP_OUTER_GRAD_REDUCE, TaskType.FSDP_OUTER_PARAM_SYNC):
                    component = states[task.rank]['ownership_components'][task.metadata['state_family']]
                    is_gradient = task.task_type == TaskType.FSDP_OUTER_GRAD_REDUCE
                    full_bytes = component['persistent_gradient_bytes' if is_gradient else 'persistent_parameter_bytes']
                    resident_width = states[task.rank][
                        'gradient_element_bytes' if is_gradient else 'parameter_element_bytes']
                    if full_bytes % resident_width:
                        raise ValueError('FSDP状态Buffer字节数与通信Tensor精度不一致')
                    task.metadata.update(TaskGraphBuilder.fsdp_collective_metadata(
                        [full_bytes // resident_width], task.metadata['group_ranks'],
                        communication_dtype, collective))
                size = task.metadata['payload_bytes']
                payloads = task.metadata['communication_payloads']
                cost_start = len(costs)
                estimates = [self._communication(
                    collective, payload, task.metadata['group_ranks'], costs,
                    dtype=communication_dtype) for payload in payloads]
                task.duration = self._total([row[0] for row in estimates])
                resources = sorted({resource for row in estimates for resource in row[1]})
                task.resource_requirements = tuple(resources) or tuple(
                    f'gpu:{self.groups["rank_mapping"][rank]}:communication'
                    for rank in task.metadata['group_ranks'])
                task.metadata['participant_ranks'] = task.metadata['group_ranks']
                task.metadata['communication_buffer_bytes'] = size
                task.metadata['communication_phases'] = [phase for row in costs[cost_start:]
                    for phase in row.get('communication_phases', [])]
            elif task.task_type == TaskType.FSDP_RESHARD:
                task.duration = 0.0
                task.resource_requirements = ()
            elif task.task_type in (TaskType.PP_P2P_RECV, TaskType.DP_REDUCE, TaskType.CUDA_SYNC):
                task.duration = 0.0
            elif task.task_type in (TaskType.OPTIMIZER_PREPARE, TaskType.GRADIENT_NORM,
                                   TaskType.GRADIENT_NORM_FINALIZE, TaskType.GRADIENT_CLIP):
                local = states[task.rank]['optimizer_bytes'] / states[task.rank]['optimizer_element_bytes']
                task.duration = self._optimizer_component_cost(task.metadata['operation'],
                    task.rank, local, request, costs)
            elif task.task_type == TaskType.OPTIMIZER_STEP:
                if self.reusable_profile is not None:
                    task.duration = self._native_optimizer_cost(task.rank, counts, config, request, costs)
                    if task.duration is not None:
                        costs[-1]['cost_key'] = fingerprint(dict(operation='native_optimizer',
                            physical_rank=physical, state=states[task.rank],
                            source=costs[-1].get('evidence_ids', []),
                            optimizer=request.get('optimizer', 'adam')))
                else:
                    if request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
                        local = states[task.rank]['optimizer_bytes'] / states[task.rank]['optimizer_element_bytes']
                    else:
                        shard = request.get('optimizer_sharding', 'distributed') == 'distributed'
                        local = counts['dense'][task.rank] / (len(self._group('dp_cp', task.rank)) if shard else 1)
                        local += counts['expert'][task.rank] / (len(self._group('expert_dp', task.rank)) if shard else 1)
                    task.duration = self._elementwise('adam_update', math.ceil(local) * 20,
                        math.ceil(local) * 28, 'optimizer', task.rank, config, request, costs, None)
            else:
                return self._report(request, source, costs, started, 'UNSUPPORTED',
                                    f'当前DAG投影不支持独立任务：{task.task_type.value}')
            task.metadata['cost_components'] = [*reused_components,
                *self._cost_components(costs[task_cost_start:])]
            kernel_rows = [row for row in costs[task_cost_start:] if row.get('milliseconds')]
            if (task.task_type in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE)
                    and not reused_components and len(kernel_rows) == 1 and kernel_rows[0]['single_gemm']):
                identity = kernel_rows[0].get('identity') or {}
                shape = identity.get('features') or {}
                if set(shape) == {'M', 'N', 'K'} and all(type(value) is int and value > 0 for value in shape.values()):
                    task.metadata['overlap_kernel'] = dict(compute_kind=task.task_type.value,
                        compute_shape=shape, compute_dtype=identity.get('dtype', self.dtype), kernel_measurement='single_gemm',
                        compute_operation=identity.get('operation'), compute_implementation=identity.get('implementation'),
                        backend_environment=fingerprint(dict(backend=request.get('backend_fingerprint'),
                            software=self.topology.software, hardware=request.get('hardware_fingerprint'))))
            if task.duration is not None and not math.isclose(task.duration, sum(
                    row['milliseconds'] * row['occurrences']
                    for row in task.metadata['cost_components']), rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError(f'任务耗时与成本组成不一致：{key}')
        self._gradient_tasks(tasks, counts, config, request, costs)
        self._parameter_gather_tasks(tasks, counts, config, request, costs)
        if any(task.duration is None for task in tasks.values()):
            report = self._report(request, source, costs, started, 'NEEDS_PROFILE', '候选部分任务缺少可靠成本')
            report['partial_execution_graph'] = dict(task_count=len(tasks),
                source_graph_sha256=fingerprint(source), representation='partially_costed_native_execution_graph',
                tasks={key: dict(rank=self.groups['rank_mapping'][task.rank], logical_rank=task.rank,
                    kind=task.task_type.value, region=task.region_id, microbatch=task.microbatch_id,
                    dependencies=task.dependencies, duration_ms=task.duration,
                    resources=list(task.resource_requirements), metadata=task.metadata)
                    for key, task in tasks.items()})
            return report
        _, predecessors, _ = topological_order(tasks)
        durations = {key: task.duration for key, task in tasks.items()}
        evaluator = ScheduleCostEvaluator(None)
        fixed_times, _, _ = evaluator.simulate_events(tasks, predecessors, durations, {}, config)
        times, critical_parent, resource_replay = evaluator.simulate_resource_aware_events(
            tasks, predecessors, durations, config, self.parameters.get('resource_replay', {}))
        for missing in resource_replay['missing_overlap_profiles']:
            request_identity = dict(kind='compute_communication_overlap', **missing)
            costs.append(dict(status='NEEDS_PROFILE', milliseconds=None,
                reason='计算通信重叠缺少当前Kernel、Collective和物理进程组的减速测量',
                profiling_request=dict(request_identity,
                    cost_key=fingerprint(request_identity))))
        end = max(times, key=lambda key: times[key][1])
        step = times[end][1]
        fixed_step = max(value[1] for value in fixed_times.values())
        critical = []
        while end is not None:
            critical.append(end)
            end = critical_parent[end]
        critical.reverse()
        memory = self._memory(tasks, times, states, config, request)
        runtime_missing = set(self.memory_missing)
        phase_memory = self.memory_bound(request, blind=blind)
        self.memory_missing.update(runtime_missing)
        memory['runtime_training_peak_per_rank_bytes'] = dict(memory['peak_per_rank_bytes'])
        memory['phase_upper_bound_peak_per_rank_bytes'] = phase_memory['peak_per_rank_bytes']
        memory['phase_breakdown_per_rank'] = phase_memory['breakdown_per_rank']
        capacity = int(self.topology.memory_per_gpu_gb * 2**30)
        phase_peaks = {}
        for rank in memory['peak_per_rank_bytes']:
            phases = dict(phase_memory['breakdown_per_rank'][rank]['gpu_phases'])
            # 训练阶段采用实际调度的存活Buffer，其他阶段保持独立状态上界。
            phases['forward_backward'] = memory['runtime_training_peak_per_rank_bytes'][rank]
            phase_peaks[rank] = phases
            memory['peak_per_rank_bytes'][rank] = max(phases.values())
            memory['memory_headroom_per_rank_bytes'][rank] = capacity - max(phases.values())
        if phase_memory['admission'] == 'REJECT_OOM':
            memory['admission'] = 'REJECT_OOM'
        elif phase_memory['admission'] == 'REQUIRE_MEMORY_PROFILE' and memory['admission'] != 'REJECT_OOM':
            memory['admission'] = 'REQUIRE_MEMORY_PROFILE'
        memory['status'] = phase_memory['status']
        memory['gpu_training_peak_per_rank_bytes'] = memory['runtime_training_peak_per_rank_bytes']
        memory['gpu_loading_peak_per_rank_bytes'] = phase_memory['gpu_loading_peak_per_rank_bytes']
        memory['gpu_optimizer_peak_per_rank_bytes'] = phase_memory['gpu_optimizer_peak_per_rank_bytes']
        memory['host_cpu_memory_peak_per_rank_bytes'] = phase_memory['host_cpu_memory_peak_per_rank_bytes']
        memory['host_cpu_memory_peak_per_node_bytes'] = phase_memory['host_cpu_memory_peak_per_node_bytes']
        memory['phase_peak_per_rank_bytes'] = phase_peaks
        memory['most_dangerous_rank'] = max(memory['peak_per_rank_bytes'], key=memory['peak_per_rank_bytes'].get)
        memory['most_dangerous_phase'] = max(phase_peaks[memory['most_dangerous_rank']],
            key=phase_peaks[memory['most_dangerous_rank']].get)
        memory['missing_profiles'] = sorted(set(memory['missing_profiles'])
            | set(phase_memory['missing_profiles']))
        report = self._report(request, source, costs, started,
            'PREDICTED_ANALYTICAL' if any(row['status'].startswith('ANALYTICAL') for row in costs)
            else 'PREDICTED_CALIBRATED')
        predicted_step = step
        anchor = None
        profile_anchor = (None if self.reusable_profile is None
            else exact_profile_anchor(self.reusable_profile, request))
        if profile_anchor is not None:
            predicted_step = profile_anchor['measured_step_time_ms']
        elif parent_request is not None:
            compatible = self.differential_family(request) == self.differential_family(parent_request)
            measured_parent = parent_request.get('_measured_step_time_ms')
            predicted_parent = parent_request.get('_predicted_raw_step_time_ms')
            if compatible and measured_parent is not None and predicted_parent is not None:
                predicted_step = measured_parent + step - predicted_parent
                if predicted_step <= 0:
                    raise ValueError('GraphDelta差分校准产生非法Step Time')
                anchor = dict(run_id=parent_request.get('_measurement_run_id'),
                    prediction_id=parent_request.get('_prediction_id'),
                    measured_step_time_ms=measured_parent,
                    raw_parent_step_time_ms=predicted_parent,
                    raw_candidate_step_time_ms=step,
                    graph_delta_step_time_ms=step - predicted_parent,
                    scope='same_global_parallel_state_sharding_schedule_and_workload_family')
        report['performance'].update(step_time_ms=predicted_step,
            raw_resource_aware_step_time_ms=step,
            graph_delta_anchored=anchor is not None or profile_anchor is not None,
            fixed_duration_dag_step_time_ms=fixed_step,
            resource_contention_added_ms=step - fixed_step,
            speedup_vs_parent=(anchor['measured_step_time_ms'] / predicted_step - 1
                if anchor is not None else None),
            tokens_per_second=request['global_batch'] * request['sequence'] * 1000 / predicted_step)
        active_anchor = profile_anchor if profile_anchor is not None else anchor
        report['evidence']['compatible_anchor_id'] = None if active_anchor is None else active_anchor['run_id']
        report['evidence']['exact_profile_anchor'] = profile_anchor
        report['evidence']['graph_delta_calibration'] = anchor
        report['memory'] = memory
        busy = defaultdict(list)
        for key, task in tasks.items():
            if (task.task_type in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE)
                    or task.metadata.get('execution_granularity') == 'measured_layer_composite'
                    or task.metadata.get('operation') == 'region_boundary_composite'):
                busy[task.rank].append(times[key])
        report['diagnosis'].update(critical_tasks=critical,
            critical_regions=sorted({tasks[key].region_id for key in critical if tasks[key].region_id is not None}),
            pipeline_idle_ms=sum(step - union_length(busy[rank]) for rank in range(config.world_size)),
            modeled_exposed_communication_ms=union_length([times[key] for key in critical if tasks[key].is_communication]),
            exposed_communication_scope='explicit_PP_and_gradient_parameter_collectives_outside_region_composites',
            full_replays=2, graph_source_sha256=fingerprint(source),
            resource_aware_replay=resource_replay)
        report['execution_graph'] = dict(source_graph_sha256=fingerprint(source),
            task_count=len(tasks),
            representation='same_c1_region_dag_with_native_rank_order_and_projection_cost_annotations',
            tasks={key: dict(rank=self.groups['rank_mapping'][task.rank], logical_rank=task.rank,
                kind=task.task_type.value, region=task.region_id, microbatch=task.microbatch_id,
                dependencies=task.dependencies, duration_ms=task.duration, start_ms=times[key][0],
                end_ms=times[key][1], resources=list(task.resource_requirements), metadata=task.metadata)
                for key, task in tasks.items()})
        report['evidence']['profiling_requests'] = self._profiling_requests(costs, set(critical))
        report['evidence']['profiling_requests'].extend(dict(kind='memory',
            cost_key=fingerprint(dict(plan=plan_signature(request), observation=name)),
            identity=dict(plan_signature=plan_signature(request), observation=name)) for name in sorted(self.memory_missing))
        report['prediction_id'] = fingerprint(dict(revision=PROJECTION_REVISION, request=request,
            source=source, mode=cost_mode, hardware_parameters=self.parameters,
            costs=costs, memory=memory, blind=blind))
        cost_sources = Counter(row.get('time_source',
            'PROFILE_REQUIRED' if row['status'] == 'NEEDS_PROFILE' else
            'ANALYTICAL' if row['status'].startswith('ANALYTICAL') else
            'CALIBRATED' if row['status'] in ('INTERPOLATED', 'RATIO_CALIBRATED') else 'MEASURED')
            for row in costs)
        sensitivity_values = [request.get('sensitivity_scenario_ms', 0)
            for request in report['evidence']['profiling_requests']
            if request.get('sensitivity_scenario_ms', 0) > 0]
        sensitivity_delta = sum(sensitivity_values)
        report['performance'].update(
            prediction_uncertainty='UNCALIBRATED' if any(
                row.get('uncertainty') == 'UNCALIBRATED' or row['status'] == 'NEEDS_PROFILE'
                for row in costs) else 'CALIBRATED_OBSERVED_VARIABILITY',
            ranking_sensitivity_scenario_ms=(
                [max(0, predicted_step - sensitivity_delta), predicted_step + sensitivity_delta]
                if sensitivity_values else None),
            ranking_sensitivity_scenario_scope=(
                'aggregate_observed_anchor_residual_and_sample_dispersion_not_confidence_interval'
                if sensitivity_values else None))
        report['evidence']['cost_sources'] = dict(cost_sources)
        report['evidence']['profiling_targets'] = report['evidence']['profiling_requests']
        overlap_unknown = bool(resource_replay['missing_overlap_profiles']) and profile_anchor is None
        report['performance']['ranking_eligible'] = not overlap_unknown
        if overlap_unknown:
            report['prediction_status'] = 'NEEDS_PROFILE'
            report['validation'].update(status='REQUIRES_PROFILING',
                reason='当前计算与通信并发条件缺少实测减速费用')
            report['performance'].update(step_time_ms=None, tokens_per_second=None,
                speedup_vs_parent=None, prediction_interval_ms=None)
            report['evidence']['unmeasured_overlap_timeline_scope'] = 'optimistic_scenario_not_for_ranking'
        return report

    def _gradient_tasks(self, tasks, counts, config, request, costs):
        fsdp = request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp'
        if not fsdp and any(len(group) > 1 for kind in ('dp_cp', 'expert_dp')
                for group in self.groups['logical'][kind]):
            compile_bucket_tasks(self, tasks, config, request, costs, 'gradient')
            return
        fsdp_strategies = {
            'dense': request.get('dense_sharding_strategy',
                request.get('fsdp_sharding_strategy', 'no_shard')),
            'expert': request.get('expert_sharding_strategy',
                request.get('fsdp_sharding_strategy', 'no_shard'))}
        original = [(key, task) for key, task in tasks.items() if task.task_type == TaskType.DP_REDUCE]
        by_rank = {task.rank: key for key, task in original}
        for family, group_kind in (('dense', 'dp_cp'), ('expert', 'expert_dp')):
            if fsdp and fsdp_strategies[family] in ('optim_grads', 'optim_grads_params'):
                continue
            for index, ranks in enumerate(self.groups['logical'][group_kind]):
                if len(ranks) == 1 or not any(counts[family][rank] for rank in ranks):
                    continue
                if len({counts[family][rank] for rank in ranks}) != 1:
                    raise ValueError('梯度同步参与者的参数所有权数量不同')
                amount = counts[family][ranks[0]] * 4
                collective = ('reduce_scatter' if not fsdp
                    and request.get('optimizer_sharding', 'distributed') == 'distributed' else 'all_reduce')
                payload = math.ceil(amount / len(ranks)) if collective == 'reduce_scatter' else amount
                duration, resources = self._communication(collective, payload, ranks, costs, dtype='float32')
                key = f'projection:gradient:{family}:{index}'
                predecessors = sorted({parent for rank in ranks for parent in tasks[by_rank[rank]].dependencies})
                tasks[key] = Task(key, ranks[0], TaskType.NCCL_GENERIC, duration=duration,
                    dependencies=predecessors, resource_requirements=tuple(resources) or tuple(
                        f'gpu:{self.groups["rank_mapping"][rank]}:communication' for rank in ranks),
                    metadata=dict(duration_evidence='projected_collective_cost', projection_only=True,
                        cost_components=self._cost_components(costs[-1:]),
                        participant_ranks=ranks, collective=collective,
                        communication_buffer_bytes=payload,
                        communication_phases=costs[-1].get('communication_phases', []),
                        buffer_aliases=['gradient_bucket_input_is_persistent_gradient_storage']))
                for rank in ranks:
                    tasks[by_rank[rank]].dependencies.append(key)

    def _parameter_gather_tasks(self, tasks, counts, config, request, costs):
        if (request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp'
                or request.get('optimizer_sharding', 'distributed') != 'distributed'):
            return
        if any(len(group) > 1 for kind in ('dp_cp', 'expert_dp')
                for group in self.groups['logical'][kind]):
            compile_bucket_tasks(self, tasks, config, request, costs, 'parameter')
            return
        optimizers = {task.rank: key for key, task in tasks.items()
                      if task.task_type == TaskType.OPTIMIZER_STEP}
        tails = {task.rank: key for key, task in tasks.items()
                 if task.task_type == TaskType.CUDA_SYNC and key == f'END:{task.rank}'}
        for family, group_kind in (('dense', 'dp_cp'), ('expert', 'expert_dp')):
            for index, ranks in enumerate(self.groups['logical'][group_kind]):
                if len(ranks) == 1 or not any(counts[family][rank] for rank in ranks):
                    continue
                payload = math.ceil(counts[family][ranks[0]] * DTYPE_BYTES[self.dtype] / len(ranks))
                duration, resources = self._communication('all_gather', payload, ranks, costs)
                key = f'projection:parameter_gather:{family}:{index}'
                tasks[key] = Task(key, ranks[0], TaskType.NCCL_GENERIC, duration=duration,
                    dependencies=[optimizers[rank] for rank in ranks],
                    resource_requirements=tuple(resources) or tuple(
                        f'gpu:{self.groups["rank_mapping"][rank]}:communication' for rank in ranks),
                    metadata=dict(duration_evidence='projected_collective_cost', projection_only=True,
                        cost_components=self._cost_components(costs[-1:]),
                        participant_ranks=ranks, collective='all_gather', communication_buffer_bytes=payload,
                        communication_phases=costs[-1].get('communication_phases', []),
                        buffer_aliases=['parameter_gather_output_is_persistent_parameter_storage']))
                for rank in ranks:
                    tasks[tails[rank]].dependencies.append(key)

    @staticmethod
    def _profiling_requests(costs, critical_tasks=None):
        critical_tasks = critical_tasks or set()
        grouped = {}
        for row in costs:
            request = row.get('profiling_request')
            if request is None:
                continue
            request = deepcopy(request)
            kind = request.get('kind')
            identity = request.get('identity', request)
            if kind == 'nccl' and 'identity' in request:
                compact_identity = {name: identity[name] for name in (
                    'collective', 'payload_bytes', 'physical_ranks', 'dtype',
                    'send_bytes_matrix', 'measured_payload_range', 'topology_class', 'concurrent_operations')
                    if name in identity}
                if 'topology_identity' in identity:
                    compact_identity['topology_identity_hash'] = fingerprint(identity['topology_identity'])
                request['identity'] = compact_identity
                identity = compact_identity
                request['reason'] = request.get('reason') or row.get('reason')
            if kind == 'compute_communication_overlap':
                feature = identity.get('condition')
                if feature is None:
                    continue
                group_key = fingerprint(dict(kind=kind, feature=feature))
                compute_task = identity.get('compute_task_id')
                communication_task = identity.get('communication_task_id')
                if group_key not in grouped:
                    grouped[group_key] = dict(kind=kind, **feature, condition=feature,
                        cost_key=group_key, reason=request.get('reason', row.get('reason')),
                        occurrence_count=0, critical_occurrence_count=0,
                        profile_required_occurrence_count=0, cost_statuses=[],
                        affected_task_count=0, affected_task_ids=[])
                grouped[group_key]['occurrence_count'] += 1
                if row['status'] not in grouped[group_key]['cost_statuses']:
                    grouped[group_key]['cost_statuses'].append(row['status'])
                if row['status'] == 'NEEDS_PROFILE':
                    grouped[group_key]['profile_required_occurrence_count'] += 1
                affected = [task_id for task_id in (compute_task, communication_task)
                    if task_id is not None]
                grouped[group_key]['affected_task_count'] += len(affected)
                grouped[group_key]['affected_task_ids'].extend(task_id for task_id in affected
                    if task_id not in grouped[group_key]['affected_task_ids']
                    and len(grouped[group_key]['affected_task_ids']) < 32)
                if any(task_id in critical_tasks for task_id in affected):
                    grouped[group_key]['critical_occurrence_count'] += 1
                continue
            group_key = request.get('cost_key', fingerprint(request))
            if group_key not in grouped:
                grouped[group_key] = request | dict(occurrence_count=0,
                    critical_occurrence_count=0, profile_required_occurrence_count=0,
                    cost_statuses=[])
            grouped[group_key]['occurrence_count'] += 1
            if row['status'] not in grouped[group_key]['cost_statuses']:
                grouped[group_key]['cost_statuses'].append(row['status'])
            if row['status'] == 'NEEDS_PROFILE':
                grouped[group_key]['profile_required_occurrence_count'] += 1
        return sorted(grouped.values(), key=lambda request: (
            -request.get('profile_required_occurrence_count', 0),
            -request.get('critical_occurrence_count', 0),
            -request.get('sensitivity_scenario_ms', 0),
            -request.get('occurrence_count', 0), request.get('cost_key', '')))

    def _report(self, request, source, costs, started, status, reason=None):
        profiling = {row.get('cost_key', fingerprint(row)): row
            for row in self._profiling_requests(costs)}
        missing = sorted({row.get('reason', '缺少适用成本') for row in costs if row['status'] == 'NEEDS_PROFILE'})
        if reason and status == 'NEEDS_PROFILE':
            missing.append(reason)
        if status == 'NEEDS_PROFILE':
            native_request = dict(kind='native_training_cost_profile', request=request,
                identity=dict(plan_signature=plan_signature(request), observations=missing),
                cost_key=fingerprint(dict(plan=plan_signature(request), observations=missing)))
            profiling[fingerprint(native_request)] = native_request
        return dict(prediction_id=fingerprint(dict(revision=PROJECTION_REVISION, request=request,
                graph=source, mode=self.mode, hardware_parameters=self.parameters, costs=costs, reason=reason, blind=self.blind)),
            candidate_id=request['run_id'], plan_signature=plan_signature(request), prediction_status=status,
            validation=dict(status='UNSUPPORTED' if status == 'UNSUPPORTED' else
                'REQUIRES_PROFILING' if status == 'NEEDS_PROFILE' else 'PROJECTABLE',
                backend_supported=status != 'UNSUPPORTED', reason=reason),
            performance=dict(step_time_ms=None, tokens_per_second=None, speedup_vs_parent=None,
                             prediction_interval_ms=None),
            memory=dict(peak_per_rank_bytes={}, admission='REQUIRE_MEMORY_PROFILE',
                        physical_memory_status='UNKNOWN'), diagnosis={},
            evidence=dict(cost_mode=self.mode, predictor_revision=PROJECTION_REVISION,
                optimizer_profile_id=(None if self.optimizer_profile is None else self.optimizer_profile['id']),
                compatible_anchor_id=None, costs=costs, cost_coverage=dict(Counter(row['status'] for row in costs)),
                blind=self.blind, candidate_plan_excluded_from_calibration=self.blind,
                module_measurement_semantics=(None if self.reusable_profile is None
                    else self.reusable_profile['measurement_semantics']),
                module_profile_selection=(None if self.reusable_profile is None else dict(
                    profile_id=self.reusable_profile['id'],
                    same_execution_plan=same_training_plan(self.reusable_profile['request'], request),
                    nested_operator_instrumentation=self.reusable_profile['request'].get('paramorph_operator_calibration', False),
                    quality_order=compute_profile_quality(self.reusable_profile)[0],
                    source_window_relative_difference=(self.reusable_profile['measurement_semantics'].get(
                        'paired_step_comparison') or {}).get('relative_difference'),
                    selection_scope='execution_identity_then_verified_scope_and_layer_instrumentation')),
                module_backend_identity=(None if self.reusable_profile is None
                    else self.reusable_profile.get('backend_identity_evidence')),
                calibration_excludes_candidate_measurement=self.blind,
                prediction_independence_scope='candidate_plan_cost_records_excluded' if self.blind else 'ordinary_candidate_projection',
                missing_profiles=missing, profiling_requests=list(profiling.values()),
                schedule_source='megatron_native_PP_VPP_forward_backward_sequence',
                cost_granularity='semantic_projection_or_explicit_actual_operator_binding',
                graph_scope='same_c1_region_tasks_and_dependencies',
                calibration_coverage=dict(internal_collective_timeline='region_composite_projection',
                    communication_contention='dynamic_equal_share_on_physical_resources',
                    compute_communication_overlap='calibrated_gamma_or_explicit_unmeasured_gamma_one',
                    prediction_intervals='unavailable_without_measured_residuals'),
                assumptions=['Region内部算子与Collective成本组成保持Projection粒度，尚未作为后端独立Kernel执行节点',
                    'PP通信可按后端overlap_p2p配置与其他Rank计算重叠',
                    'Gradient Finalize按完整Backward结束后的Dense与Expert独立同步组合',
                    'Distributed Optimizer参数Gather计入Step尾部，尚未声明跨Step预取重叠',
                    '逐层非矩阵Backward操作数量和访存按Forward的两倍进行解析估计',
                    '并发通信按照共享物理资源的动态公平带宽推进',
                    '计算通信并发减速仅使用具有证据的gamma，未测组合保留Profiling Request']),
            evaluation_seconds=time.perf_counter() - started)
