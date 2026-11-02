from dataclasses import asdict
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core_data_models import Candidate
from graph.bottleneck_analysis import BottleneckGraph
from region_partition.region_partitioner import RegionPartitioner
from scheduling.task_graph_builder import TaskGraphBuilder
from validation.parallel_validator import validate_candidate
from scheduling.native_microbatches import valid_microbatch_group


class StrategyParameters(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    tp: int | None = None
    cp: int | None = None
    pp: int | None = None
    ep: int | None = None
    vpp: int | None = None
    decoder_chunks: list[int] | None = None
    microbatch_group: int | None = None
    micro_batch_size: int | None = None
    bucket_elements: int | None = None
    enabled: bool | None = None


class StrategyEdit(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    operation: Literal['SetParallelism', 'SetRegionLayout', 'SetMicrobatchGroup', 'SetMicrobatchSize',
                       'ConfigureGradientBucket', 'ConfigureP2POverlap']
    target_task_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    parameters: StrategyParameters
    hypothesis: str = Field(min_length=1)


class StrategyProposal(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    name: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    edits: list[StrategyEdit] = Field(min_length=1, max_length=8)
    missing_measurements: list[str]
    parent_candidate_id: str | None = None


class StrategyProposals(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    proposals: list[StrategyProposal] = Field(max_length=4)
    reason: str
    queries: list['GraphQuery'] = Field(default_factory=list, max_length=4)
    reasoning_evidence: list['GraphReasoningReference'] = Field(default_factory=list)


class GraphQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    operation: Literal['inspect_graph', 'inspect_parallelism', 'compare_plans',
                       'inspect_module_boundary', 'inspect_semantic_regions', 'inspect_candidate_diff',
                       'inspect_strategy_dependencies', 'inspect_task_trace', 'analyze_critical_path', 'evaluate_plan',
                       'inspect_bottleneck_subgraph', 'inspect_task_dependencies']
    task_ids: list[str] = Field(default_factory=list, max_length=16)
    evidence_ids: list[str] = Field(default_factory=list, max_length=2)
    candidate_id: str | None = None
    logical_layers: list[int] = Field(default_factory=list, max_length=2)
    parameters: StrategyParameters | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=32, ge=1, le=128)
    dependency_depth: int | None = Field(default=1, ge=0)
    direction: Literal['predecessors', 'successors'] = 'predecessors'
    max_chars: int = Field(default=12000, ge=2048, le=32000)


class GraphReasoningReference(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    conclusion: str = Field(min_length=1)
    query_result_ids: list[str] = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    claim_scope: Literal['structural_fact', 'measured_fact', 'hypothesis', 'uncertainty']
    evidence_type: Literal['aggregate_metric', 'graph_structure', 'task_trace'] | None = None


StrategyProposals.model_rebuild()


class BackendCapabilityConflict(ValueError):
    def __init__(self, variable, value, allowed):
        self.conflict = dict(variable=variable, requested=value, supported_values=list(allowed))
        super().__init__(f'当前执行入口不支持{variable}={value}，允许值为{allowed}')


class StrategyTransformationCompiler:
    def __init__(self, model, topology, capabilities=None):
        self.model = model
        self.topology = topology
        self.backend_capabilities = dict(capabilities or {})

    @staticmethod
    def task_flow_id(task):
        if 'channel' in task.metadata:
            return f'flow:{task.metadata["channel"]}:mb:{task.microbatch_id}'
        if task.task_type.value == 'dp_reduce':
            return f'flow:gradient_sync:rank:{task.rank}'
        return None

    @staticmethod
    def validate_semantic_graph_record(graph):
        regions = {region['region_id']: region for region in graph['regions']}
        if len(regions) != len(graph['regions']) or graph['scope'] != 'OPERATOR':
            raise ValueError('实际语义图缺少唯一Region身份或验证范围')
        bound = set()
        edges = {tuple(edge) for edge in graph['edges']}
        positions = {identity: position for position, identity in enumerate(regions)}
        if any(source not in positions or destination not in positions
               or positions[source] >= positions[destination] for source, destination in edges):
            raise ValueError('实际语义图存在不合法的计算依赖')
        for region in regions.values():
            for contract in region['input_contracts'] + region['output_contracts']:
                if contract['dtype'] in ('bool', 'int32', 'int64') and contract['gradient_eligible']:
                    raise ValueError('路由整数元数据不能要求Tensor梯度')
        for connection in graph['tensor_connections']:
            source = regions[connection['producer']]['output_contracts'][connection['producer_output']]
            destination = regions[connection['consumer']]['input_contracts'][connection['consumer_input']]
            port = connection['consumer'], connection['consumer_input']
            if (source != destination or port in bound
                    or (connection['producer'], connection['consumer']) not in edges):
                raise ValueError('实际语义图的Tensor连接不完整或者不一致')
            bound.add(port)
        for identity, index in graph['external_inputs']:
            if (identity, index) in bound:
                raise ValueError('实际语义图的输入存在重复来源')
            bound.add((identity, index))
        required = {(identity, index) for identity, region in regions.items()
                    for index in range(len(region['input_contracts']))}
        parameters = [key for region in regions.values() for key in region['parameter_keys']]
        if (bound != required or len(parameters) != len(set(parameters))
                or set(parameters) != set(graph['expected_parameter_keys'])):
            raise ValueError('实际语义图没有覆盖全部Tensor输入与参数归属')
        return dict(regions=len(regions), tensor_connections=len(graph['tensor_connections']),
                    parameters=len(parameters))

    @staticmethod
    def executed_boundary_graph(measurement):
        actual = measurement.get('actual_execution', {})
        if not actual.get('complete_rank_coverage'):
            raise ValueError('模块边界重建需要完整Rank记录')
        regions, transfers, witnesses = {}, {}, []
        for rank_record in actual['ranks'].values():
            boundary = rank_record.get('module_boundary')
            if boundary is None:
                continue
            for endpoint in boundary['endpoints']:
                binding = endpoint['binding']
                if endpoint['stage'] != binding['source_pp']:
                    continue
                if endpoint['completed_updates'] != measurement['warmup'] + measurement['measured']:
                    raise ValueError('模块边界记录没有完成对应训练更新')
                witness = endpoint['boundary_program']['executable_witness']
                source = endpoint['pp_ranks'][binding['source_pp']]
                destination = endpoint['pp_ranks'][binding['source_pp'] + 1]
                layer = binding['logical_layer']
                for semantic_id, owners in witness['parameter_ownership']:
                    for owner in owners:
                        region_id = f'{semantic_id}:rank:{owner}'
                        node = dict(region_id=region_id, semantic_id=semantic_id, rank=owner,
                            logical_layer=layer, virtual_chunk=binding['source_vp'])
                        if region_id in regions and regions[region_id] != node:
                            raise ValueError('模块边界存在冲突参数归属')
                        regions[region_id] = node
                for microbatch in range(binding['microbatches']):
                    key = f'layer:{layer}:vp:{binding["source_vp"]}:mb:{microbatch}:{source}:{destination}'
                    if key in transfers:
                        raise ValueError('模块边界存在重复传输身份')
                    transfers[key] = dict(communication_id=key,
                        forward=dict(source_rank=source, destination_rank=destination),
                        backward=dict(source_rank=destination, destination_rank=source),
                        microbatch=microbatch, logical_layer=layer,
                        virtual_chunk=binding['source_vp'], payload=endpoint['payload'],
                        transferred_bytes=endpoint['boundary_program']['transferred_bytes_per_microbatch'],
                        buffer_specs=witness['buffer_specs'], tensor_maps=witness['tensor_maps'],
                        adjoint=endpoint['adjoint'], transport=endpoint['transport'],
                        actual_duration_ms=None)
                witnesses.append(dict(source_rank=source, destination_rank=destination,
                    forward_path=witness['forward_nodes'], backward_path=witness['backward_nodes'],
                    schedule_requirements=witness['schedule_requirements'],
                    native_virtual_schedule_table=endpoint.get('native_virtual_schedule_table'),
                    boundary_sha256=rank_record['boundary_sha256']))
        return dict(status='available' if witnesses else 'no_executed_module_boundary',
            evidence_id=measurement['evidence_id'], plan=measurement['plan'],
            model_region_nodes=regions, microbatch_tensor_transfers=transfers,
            execution_contracts=witnesses, scope='executed_typed_boundary_subgraph',
            complete_iteration_graph=False, resource_contention_edges=[],
            missing_measurements=['Region内部Residual与算子依赖', '实际Kernel与通信时间',
                'CPU优化器迁移时间线', 'CUDA Stream与Buffer释放事件'])

    @staticmethod
    def query(observation, query):
        query = GraphQuery.model_validate(query)
        if query.candidate_id is not None and query.candidate_id != observation.get('parent_candidate', {}).get('candidate_id'):
            known = observation.get('candidate_observations', {})
            if query.candidate_id not in known:
                raise ValueError('查询引用了未编译的候选程序')
            observation = known[query.candidate_id]
        graph = observation['training_graph']
        if not set(query.task_ids) <= set(graph['tasks']):
            raise ValueError('图查询引用了不存在的任务')
        if query.operation in ('inspect_bottleneck_subgraph', 'inspect_task_dependencies'):
            evidence = [item['evidence_id'] for item in observation['evidence']]
            backend = BottleneckGraph(graph, evidence_refs=evidence)
            if query.operation == 'inspect_task_dependencies':
                if len(query.task_ids) != 1:
                    raise ValueError('任务依赖查询需要一个稳定Task ID')
                return backend.task_dependencies(query.task_ids[0], query.direction,
                    query.offset, query.limit, query.max_chars)
            closure = backend.closure(query.task_ids, dependency_depth=query.dependency_depth)
            return backend.page(closure, query.offset, query.limit, query.max_chars)
        if query.operation == 'inspect_task_trace':
            known = {item['evidence_id']: item for item in observation['evidence']}
            if len(query.evidence_ids) != 1 or query.evidence_ids[0] not in known:
                raise ValueError('任务Trace查询必须提供一个实际证据身份')
            source = known[query.evidence_ids[0]]
            trace = source.get('task_trace')
            if trace is None:
                raise ValueError('提供的证据没有任务Trace')
            if any(source['plan'][key] != graph['config'][key] for key in ('tp', 'cp', 'pp', 'ep', 'vpp')) or (
                source['plan']['world'] // (source['plan']['tp'] * source['plan']['cp'] * source['plan']['pp']) != graph['config']['dp']) or (
                source['plan']['micro_batch'] != graph['config']['micro_batch_size']) or (
                source['plan']['decoder_chunks'] != [region['layer_end'] - region['layer_start']
                    for region in sorted(graph['regions'], key=lambda item: item['region_id'])]):
                raise ValueError('任务Trace与候选策略不一致')
            rows = []
            for rank, record in trace['ranks'].items():
                for event in record['tasks']:
                    identity = event['identity']
                    matches = [key for key, task in graph['tasks'].items() if identity and
                        task['rank'] == int(rank) and task['microbatch'] == identity['microbatch'] and
                        task['virtual_stage'] == identity['virtual_chunk'] and
                        key.startswith(identity['kind'] + ':')]
                    if query.task_ids and not set(matches).intersection(query.task_ids):
                        continue
                    kernels = event['correlated_gpu_events']
                    rows.append(event | dict(rank=int(rank), task_ids=matches,
                        correlated_gpu_events=kernels[:16], correlated_gpu_event_count=len(kernels),
                        correlated_gpu_duration_sum_ms=sum(kernel['duration_ms'] for kernel in kernels),
                        duration_scope='kernel duration sum; concurrent kernels may overlap',
                        full_event_access=trace['profile_directory'],
                        trace_sha256=record['trace_sha256']))
            return dict(status='available', evidence_type='task_trace',
                tasks={key: graph['tasks'][key] for row in rows for key in row['task_ids']},
                events=rows[query.offset:query.offset + query.limit], total_events=len(rows),
                cross_rank_clock_alignment=trace['cross_rank_clock_alignment'],
                missing_measurements=trace['missing_measurements'], performance_eligible=False)
        if query.operation == 'inspect_strategy_dependencies':
            changes = query.parameters.model_dump(exclude_none=True) if query.parameters is not None else {}
            allowed = {'tp', 'cp', 'pp', 'ep', 'vpp', 'micro_batch_size', 'microbatch_group', 'decoder_chunks', 'enabled'}
            if set(changes) - allowed or any(type(value) is not int or value <= 0
                for key, value in changes.items() if key not in ('decoder_chunks', 'enabled')):
                raise ValueError('策略依赖查询参数不合法')
            if 'enabled' in changes and type(changes['enabled']) is not bool:
                raise ValueError('P2P重叠查询必须提供布尔值')
            counts = changes.get('decoder_chunks')
            if counts is not None and (not counts or any(type(value) is not int or value <= 0 for value in counts)):
                raise ValueError('策略依赖查询的Region层数不合法')
            config = graph['config'] | changes
            world = observation['backend_constraints']['world_size']
            divisor = config['tp'] * config['cp'] * config['pp']
            if world % divisor:
                return dict(status='constraint_conflict', reason='TP乘CP乘PP不能整除实际GPU数量')
            config['dp'] = world // divisor
            config['etp'] = config['tp']
            batch = config['global_batch_size']
            denominator = config['dp'] * config['micro_batch_size']
            microbatches = batch // denominator if batch % denominator == 0 else None
            capacity = config['cp'] * config['dp']
            experts = observation['model']['num_experts']
            ep_choices = [value for value in observation['backend_capabilities'].get('ep', range(1, capacity + 1))
                if capacity % value == 0 and (experts is not None and experts % value == 0 or experts is None and value == 1)]
            constraints = []
            if counts is not None and (len(counts) != config['pp'] * config['vpp'] or
                    sum(counts) != observation['model']['num_layers']):
                constraints.append('Region数量或层覆盖与PP乘VPP及完整模型不一致')
            if config['ep'] not in ep_choices:
                constraints.append('需要Agent选择合法EP，工具保留全部受支持的合法选择')
            if microbatches is None:
                constraints.append('GBS无法被DP乘MBS整除')
            elif not valid_microbatch_group(microbatches, config['pp'], config['vpp'], config['microbatch_group']):
                constraints.append('Microbatch数量或分组不满足当前交错调度要求')
            return dict(status='available', requested_changes=changes,
                necessary_values=dict(dp=config['dp'], etp=config['etp']),
                supported_ep_choices=ep_choices, unresolved_constraints=constraints,
                microbatch_change=[graph['config']['global_batch_size'] //
                    (graph['config']['dp'] * graph['config']['micro_batch_size']), microbatches],
                regions_required=config['pp'] * config['vpp'],
                overlap_p2p=changes.get('enabled', graph['config']['execution_implementation'].get('overlap_p2p')),
                local_pipeline_tensor_shape=[config['sequence_length'] // config['cp'] // config['tp'],
                    config['micro_batch_size'], observation['model']['hidden_size']],
                related_decisions=['EP Folding', 'MBS与GBS整除关系', 'VPP与Microbatch分组', 'Region完整覆盖', '实际显存'],
                predicted_step_time_ms=None, scope='图配置与后端约束；性能参数由Agent选择')
        if query.operation == 'inspect_candidate_diff':
            record = observation['parent_candidate'].get('transformation_record', {})
            if query.candidate_id is None or 'graph_delta' not in record:
                raise ValueError('候选差异查询需要已编译的候选身份')
            delta = record.get('reference_graph_delta', record['graph_delta'])
            selected = {key: graph['tasks'][key] for key in query.task_ids}
            if 'workload_summary' not in delta:
                return dict(status='measurement_required', graph_sha256=graph['graph_sha256'],
                    config_changes=delta['config_changes'], selected_tasks=selected,
                    missing_measurements=['旧快照缺少任务与通信统计，可通过完整图查询读取任务'],
                    predicted_step_time_ms=None)
            impact = {key: value[:16] if isinstance(value, list) else value
                      for key, value in delta['impact'].items()}
            return dict(status='available', graph_sha256=graph['graph_sha256'], config_changes=delta['config_changes'],
                workload_summary=delta['workload_summary'], impact=impact,
                selected_tasks=selected,
                added_task_count=len(delta['added_task_ids']),
                removed_task_count=len(delta['removed_task_ids']),
                modified_task_count=len(delta['modified_task_ids']),
                scope=delta['scope'], opaque_changes=delta['opaque_changes'],
                predicted_step_time_ms=None,
                missing_measurements=['实际计算和通信成本', '全局等待和并发资源竞争'])
        if query.operation == 'inspect_graph':
            sends = [task for task in graph['tasks'].values() if task['operation'] == 'pp_p2p_send']
            inventory = dict(task_types=dict(Counter(task['operation'] for task in graph['tasks'].values())),
                logical_pp_transfers=len(sends),
                logical_pp_tensor_bytes=sum(prod(task['metadata']['shape']) * 2 for task in sends),
                regions=graph['regions'], config=graph['config'],
                opaque_operations=graph['opaque_operations'], actual_gpu_times_available=False)
            selected = set(query.task_ids)
            if not selected:
                keys = sorted(graph['tasks'])
                included = keys[query.offset:query.offset + query.limit]
                return dict(status='available', graph_sha256=graph['graph_sha256'],
                    tasks={key: graph['tasks'][key] for key in included},
                    graph_inventory=inventory,
                    full_task_count=len(keys), offset=query.offset,
                    next_offset=query.offset + len(included) if query.offset + len(included) < len(keys) else None,
                    scope='完整候选图分页访问', actual_gpu_timing=None)
            neighbours = {dependency for key in selected for dependency in graph['tasks'][key]['dependencies']}
            neighbours.update(key for key, task in graph['tasks'].items()
                              if selected & set(task['dependencies']))
            included = sorted(selected) + sorted(neighbours - selected)[:64 - len(selected)]
            tasks = {key: graph['tasks'][key] for key in included}
            return dict(status='available', graph_sha256=graph['graph_sha256'], tasks=tasks,
                graph_inventory=inventory,
                omitted_neighbour_ids=sorted((selected | neighbours) - set(tasks)),
                scope='semantic_region_task_dependencies', actual_gpu_timing=None)
        if query.operation == 'inspect_parallelism':
            known = {item['evidence_id']: item for item in observation['evidence']}
            if not set(query.evidence_ids) <= set(known):
                raise ValueError('并行查询引用了不存在的执行证据')
            return dict(status='available', model=observation['model'], config=graph['config'],
                backend_constraints=observation['backend_constraints'],
                regions=graph['regions'], gemm_measurements=None,
                measured_execution={key: dict(plan=known[key]['plan'],
                    actual_execution=known[key].get('actual_execution'),
                    aggregate_metrics={field: known[key].get(field) for field in
                        ('tokens_per_second', 'step_seconds', 'peak_allocated_gib', 'status',
                         'warmup', 'measured', 'performance_eligible')}) for key in query.evidence_ids},
                opaque_operations=graph['opaque_operations'],
                missing_measurements=['实际Kernel形状与耗时', '独立Collective与路由分布'])
        if query.operation == 'inspect_module_boundary':
            known = {item['evidence_id']: item for item in observation['evidence']}
            if len(query.evidence_ids) != 1 or query.evidence_ids[0] not in known:
                raise ValueError('模块边界查询必须提供一个实际执行证据身份')
            return StrategyTransformationCompiler.executed_boundary_graph(known[query.evidence_ids[0]])
        if query.operation == 'inspect_semantic_regions':
            known = {item['evidence_id']: item for item in observation['evidence']}
            if (len(query.evidence_ids) != 1 or query.evidence_ids[0] not in known
                    or not query.logical_layers or min(query.logical_layers) < 0):
                raise ValueError('语义Region查询需要实际证据和逻辑层身份')
            selected = []
            measurement = known[query.evidence_ids[0]]
            for rank, record in measurement.get('actual_execution', {}).get('ranks', {}).items():
                if 'semantic_graph_path' not in record:
                    continue
                data = Path(record['semantic_graph_path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != record['semantic_graph_sha256']:
                    raise ValueError('实际Semantic Region文件发生变化')
                for chunk in json.loads(data):
                    for graph in chunk['graphs']:
                        if graph['regions'][0]['logical_layer_id'] in query.logical_layers:
                            StrategyTransformationCompiler.validate_semantic_graph_record(graph)
                            selected.append(dict(rank=int(rank), virtual_chunk=chunk['virtual_chunk'],
                                graph=graph, process_groups=record['process_groups'],
                                compiled_placement=chunk.get('compiled_placement')))
            return dict(status='available' if selected else 'measurement_required',
                semantic_region_graphs=selected, complete_iteration_graph=False,
                missing_measurements=['实际Kernel时间与CPU迁移事件'])
        if query.operation == 'compare_plans':
            known = {item['evidence_id']: item for item in observation['evidence']}
            if len(query.evidence_ids) != 2 or not set(query.evidence_ids) <= set(known):
                raise ValueError('方案比较必须提供两个实际证据身份')
            before, after = [known[key] for key in query.evidence_ids]
            difference = {key: [before['plan'][key], after['plan'][key]]
                          for key in before['plan'] if before['plan'][key] != after['plan'][key]}
            result = dict(status='available', plan_changes=difference,
                before_status=before['status'], after_status=after['status'],
                cause_attribution='requires_task_trace', throughput_change_fraction=None)
            if before['status'] == after['status'] == 'complete':
                result['throughput_change_fraction'] = after['tokens_per_second'] / before['tokens_per_second'] - 1
            return result
        return dict(status='measurement_required', operation=query.operation,
            graph_sha256=graph['graph_sha256'], predicted_step_time_ms=None,
            missing_measurements=['完整任务成本', '实测并发成本', '显存生命周期', '跨Rank时钟与执行Trace'])

    def buffer_lifetimes(self, candidate):
        dag = self.graph(candidate)
        buffers = {}
        receivers = {}
        for identity, task in dag.tasks.items():
            if task.task_type.value != 'pp_p2p_recv':
                continue
            channel = (task.metadata['channel'], task.microbatch_id)
            if channel in receivers:
                raise ValueError('通信Buffer生命周期需要唯一匹配的接收任务')
            receivers[channel] = identity
        for key, task in dag.tasks.items():
            if task.task_type.value == 'forward_compute':
                backward = key.replace('F:', 'B:', 1)
                buffers['activation:' + key] = dict(rank=task.metadata['physical_rank'],
                    allocation_task=key, release_task=backward, layer_ids=task.metadata['layer_ids'],
                    microbatch=task.microbatch_id, recomputation=candidate.parallel_config.memory_policy.recompute_granularity,
                    size_source='requires_operator_shapes_and_memory_projection')
            elif task.task_type.value == 'pp_p2p_send':
                channel = (task.metadata['channel'], task.microbatch_id)
                if channel not in receivers:
                    raise ValueError('通信Buffer生命周期需要唯一匹配的接收任务')
                buffers['communication:' + key] = dict(rank=task.metadata['physical_rank'],
                    allocation_task=task.metadata['producer_task'], release_task=receivers[channel],
                    shape=task.metadata['shape'], dtype=task.metadata['dtype'],
                    physical_group=task.metadata['physical_group_ranks'])
        return buffers

    def graph_delta(self, parent, candidate):
        before = self.graph(parent)
        after = self.graph(candidate)
        before_keys, after_keys = set(before.tasks), set(after.tasks)
        config_changes = {key: [asdict(parent.parallel_config)[key], asdict(candidate.parallel_config)[key]]
            for key in asdict(parent.parallel_config)
            if asdict(parent.parallel_config)[key] != asdict(candidate.parallel_config)[key]}

        def workload(dag, plan):
            sends = [task for task in dag.tasks.values() if task.task_type.value == 'pp_p2p_send']
            return dict(nodes=len(dag.tasks), edges=len(dag.edges),
                forward_tasks=sum(task.task_type.value == 'forward_compute' for task in dag.tasks.values()),
                backward_tasks=sum(task.task_type.value == 'backward_compute' for task in dag.tasks.values()),
                microbatches=plan.parallel_config.num_microbatches,
                logical_pp_transfers=len(sends),
                logical_pp_tensor_bytes=sum(prod(task.metadata['shape']) * 2 for task in sends),
                pp_tensor_shapes=[list(shape) for shape in sorted({tuple(task.metadata['shape']) for task in sends})],
                virtual_region_layers=[region.num_layers for region in plan.region_layout],
                physical_stage_layers=[sum(region.num_layers for region in plan.region_layout if region.pp_rank == pp)
                    for pp in range(plan.parallel_config.pp)],
                opaque_collectives=['tp', 'cp', 'ep'],
                measured_kernel_costs=None, optimizer_time=None, pipeline_wait_time=None)
        modified = []
        for key in sorted(before_keys & after_keys):
            old, new = before.tasks[key], after.tasks[key]
            if (old.rank, old.region_id, old.dependencies, old.metadata) != (
                    new.rank, new.region_id, new.dependencies, new.metadata):
                modified.append(key)
        affected = set(modified) | (after_keys - before_keys)
        for key, task in after.tasks.items():
            if 'execution_implementation' in config_changes and task.is_communication:
                affected.add(key)
            if set(config_changes) & {'tp', 'cp', 'dp', 'ep', 'etp', 'microbatch_group',
                                     'micro_batch_size', 'sequence_length', 'use_sequence_parallel'}:
                affected.add(key)
        placements = {key: dict(before=before.tasks[key].metadata.get('physical_rank'),
            after=after.tasks[key].metadata.get('physical_rank')) for key in before_keys & after_keys
            if before.tasks[key].metadata.get('physical_rank') != after.tasks[key].metadata.get('physical_rank')}
        layouts = {key: dict(before={name: before.tasks[key].metadata.get(name) for name in ('shape', 'dtype', 'group_ranks')},
            after={name: after.tasks[key].metadata.get(name) for name in ('shape', 'dtype', 'group_ranks')})
            for key in before_keys & after_keys if any(before.tasks[key].metadata.get(name) !=
                after.tasks[key].metadata.get(name) for name in ('shape', 'dtype', 'group_ranks'))}
        return dict(added_task_ids=sorted(after_keys - before_keys),
            removed_task_ids=sorted(before_keys - after_keys), modified_task_ids=modified,
            added_edges=sorted(set(after.edges) - set(before.edges)),
            removed_edges=sorted(set(before.edges) - set(after.edges)),
            config_changes=config_changes, affected_task_ids=sorted(affected),
            changed_compute_tasks=sorted(key for key in affected if after.tasks[key].is_compute),
            changed_communication_tasks=sorted(key for key in affected if after.tasks[key].is_communication),
            changed_placement=placements, changed_tensor_layout=layouts,
            changed_dependencies={key: dict(before=before.tasks[key].dependencies, after=after.tasks[key].dependencies)
                for key in before_keys & after_keys if before.tasks[key].dependencies != after.tasks[key].dependencies},
            changed_communication_paths={key: dict(before=before.tasks[key].metadata.get('physical_group_ranks'),
                after=after.tasks[key].metadata.get('physical_group_ranks')) for key in before_keys & after_keys
                if before.tasks[key].metadata.get('physical_group_ranks') != after.tasks[key].metadata.get('physical_group_ranks')},
            buffer_lifetime_status='requires_instantiated_execution_timeline',
            workload_summary=dict(before=workload(before, parent), after=workload(after, candidate),
                scope='完整层Region图中的逻辑任务和BF16 P2P字节数；实际NCCL调用与Kernel次数需要Trace'),
            impact=dict(
                compute_task_ids=sorted(key for key in affected if after.tasks[key].is_compute),
                communication_task_ids=sorted(key for key in affected if after.tasks[key].is_communication),
                replica_batch=[parent.parallel_config.global_batch_size // parent.parallel_config.dp,
                               candidate.parallel_config.global_batch_size // candidate.parallel_config.dp],
                num_microbatches=[parent.parallel_config.num_microbatches,
                                  candidate.parallel_config.num_microbatches],
                local_sequence=[parent.parallel_config.sequence_length // parent.parallel_config.cp,
                                candidate.parallel_config.sequence_length // candidate.parallel_config.cp],
                opaque_collective_groups_changed=bool(set(config_changes) & {'tp', 'cp', 'dp', 'ep', 'etp'}),
                memory_prediction=None, exact_new_kernel_costs=None),
            scope='region_dag_and_native_execution_policy',
            opaque_changes='Region内部Collective需运行时证据进一步展开')

    def affected_tasks(self, parent, edit):
        dag = self.graph(parent)
        if edit.operation in ('SetParallelism', 'SetMicrobatchGroup', 'SetMicrobatchSize'):
            return set(dag.tasks)
        if edit.operation == 'ConfigureGradientBucket':
            return {key for key, task in dag.tasks.items()
                    if task.task_type.value in ('dp_reduce', 'optimizer_step')}
        if edit.operation == 'ConfigureP2POverlap':
            return {key for key, task in dag.tasks.items()
                    if task.task_type.value in ('pp_p2p_send', 'pp_p2p_recv')}
        counts = edit.parameters.decoder_chunks
        if len(counts) != len(parent.region_layout):
            return set(dag.tasks)
        offset = 0
        changed = set()
        for index, region in enumerate(parent.region_layout):
            end = offset + counts[index]
            if (offset, end) != (region.layer_start, region.layer_end):
                changed.add(index)
            offset = end
        selected = {key for key, task in dag.tasks.items() if task.region_id in changed}
        selected.update(key for key, task in dag.tasks.items()
                        if selected & set(task.dependencies))
        return selected

    def preview_transformation(self, parent, intent):
        dag = self.graph(parent)
        values = intent.values
        decisions = set(values)
        hard = {
            'tp': ['dp', 'etp', 'expert_dp'], 'cp': ['dp', 'expert_dp', 'micro_batch', 'group'],
            'pp': ['dp', 'expert_dp', 'vpp', 'decoder_chunks', 'pipeline_schedule'],
            'ep': ['etp', 'expert_dp'], 'etp': ['ep', 'expert_dp'],
            'dp': ['micro_batch', 'group'], 'vpp': ['decoder_chunks', 'group', 'pipeline_schedule'],
            'micro_batch': ['group'], 'pipeline_schedule': ['pp', 'vpp', 'group'],
            'rank_mapping': ['process_groups'], 'optimizer_sharding': ['optimizer_state_ownership']}
        counts = values.get('decoder_chunks')
        regions = set()
        if counts is not None and len(counts) == len(parent.region_layout):
            offset = 0
            for region, count in zip(parent.region_layout, counts):
                end = offset + count
                if (offset, end) != (region.layer_start, region.layer_end):
                    regions.add(region.region_id)
                offset = end
        else:
            regions = {region.region_id for region in parent.region_layout}
        selected = {key for key, task in dag.tasks.items() if task.region_id in regions} if 'decoder_chunks' in decisions else set()
        expert_layers = {index for index, kind in enumerate(self.model.layer_types) if kind.value == 'moe'}
        for decision in decisions - {'decoder_chunks'}:
            if decision in ('optimizer_sharding', 'bucket_elements'):
                selected.update(key for key, task in dag.tasks.items() if task.task_type.value in
                    ('dp_reduce', 'optimizer_step', 'cuda_sync'))
            elif decision == 'overlap_p2p':
                selected.update(key for key, task in dag.tasks.items() if task.task_type.value in ('pp_p2p_send', 'pp_p2p_recv'))
            elif decision in ('ep', 'etp', 'expert_dp'):
                if expert_layers:
                    selected.update(key for key, task in dag.tasks.items() if
                        expert_layers & set(task.metadata.get('layer_ids', [])) or
                        task.task_type.value in ('dp_reduce', 'optimizer_step'))
            elif decision == 'recompute':
                selected.update(key for key, task in dag.tasks.items() if task.is_compute)
            elif decision == 'rank_mapping' and values.get(decision) is not None:
                before = list(parent.parallel_config.device_mapping) or list(range(parent.parallel_config.world_size))
                after = values[decision]
                if len(after) != len(before) or sorted(after) != list(range(len(before))):
                    raise ValueError('Effect Preview的物理映射没有完整覆盖设备集合')
                changed_ranks = {rank for rank, pair in enumerate(zip(before, after)) if pair[0] != pair[1]}
                selected.update(key for key, task in dag.tasks.items() if task.rank in changed_ranks or
                    changed_ranks & set(task.metadata.get('group_ranks', [])))
            else:
                selected.update(dag.tasks)
        direct = set(selected)
        selected.update(key for key, task in dag.tasks.items() if task.is_communication and
            direct & set(task.dependencies))
        regions = {dag.tasks[key].region_id for key in selected if dag.tasks[key].region_id is not None}
        flows = sorted({self.task_flow_id(dag.tasks[key]) for key in selected
            if self.task_flow_id(dag.tasks[key]) is not None})
        resources = sorted({f'gpu:{dag.tasks[key].metadata["physical_rank"]}' for key in selected})
        return dict(affected_task_ids=sorted(selected), affected_region_ids=sorted(regions),
            affected_flow_ids=flows, resource_ids=resources,
            hard_constraint_dependencies={key: hard.get(key, []) for key in sorted(decisions)},
            performance_impact_decisions=sorted(decisions),
            potential_placement_changes=bool(decisions & {'pp', 'vpp', 'decoder_chunks', 'rank_mapping'}),
            potential_tensor_shape_changes=bool(decisions & {'tp', 'cp', 'etp', 'micro_batch'}),
            potential_process_group_changes=bool(decisions & {'tp', 'cp', 'pp', 'dp', 'ep', 'etp', 'rank_mapping'}),
            potential_buffer_lifetime_changes=bool(decisions & {'vpp', 'group', 'micro_batch', 'recompute', 'overlap_p2p'}),
            unknown_effects=['actual_kernel_duration', 'actual_resource_contention', 'actual_memory_peak'],
            source='current_training_dag_and_backend_constraint_dependencies',
            scope='structural_effect_preview', predicted_speedup=None)

    def decision_impact_index(self, parent):
        from contracts import RewriteIntent
        names = ('tp', 'cp', 'pp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'micro_batch',
                 'decoder_chunks', 'recompute', 'optimizer_sharding', 'pipeline_schedule',
                 'group', 'overlap_p2p', 'bucket_elements', 'rank_mapping')
        return {name: self.preview_transformation(parent, RewriteIntent(
            operation='JointTransform', values={name: None}, adjustable=[], preserve=[],
            hypothesis='查询决策对应的潜在任务影响', evidence_refs=['structural_query']))
                for name in names}

    def transformation_interactions(self, parent, intents, bottleneck_task_ids=()):
        previews = [self.preview_transformation(parent, intent) for intent in intents]
        dag = self.graph(parent)
        if not set(bottleneck_task_ids) <= set(dag.tasks):
            raise ValueError('交互分析的瓶颈集合包含未知任务')
        edges = []
        for left in range(len(intents)):
            for right in range(left + 1, len(intents)):
                first, second = previews[left], previews[right]
                shared = sorted(set(first['affected_task_ids']) & set(second['affected_task_ids']))
                bottlenecks = sorted(set(shared) & set(bottleneck_task_ids))
                resources = sorted(set(first['resource_ids']) & set(second['resource_ids']))
                connected = sorted((source, destination) for source, destination in dag.edges
                    if source in first['affected_task_ids'] and destination in second['affected_task_ids'])
                dependencies = {name for values in first['hard_constraint_dependencies'].values() for name in values}
                reverse = {name for values in second['hard_constraint_dependencies'].values() for name in values}
                coupling = sorted((dependencies & set(intents[right].values)) | (reverse & set(intents[left].values)))
                for kind, evidence in (
                    ('HardCoupling', dict(constraint_variables=coupling)),
                    ('SharedBottleneck', dict(task_ids=bottlenecks)),
                    ('SharedTaskImpact', dict(task_ids=shared)),
                    ('TemporalInteraction', dict(dependencies=connected)),
                    ('ResourceInteraction', dict(resource_ids=resources))):
                    if any(evidence.values()):
                        edges.append(dict(left=left, right=right, kind=kind,
                            evidence=evidence, evidence_scope='structural',
                            performance_synergy='unmeasured'))
        return dict(previews=previews, edges=edges, measured_interactions=[],
            source='actual_dag_dependencies_and_decision_impact_index')

    def interaction_scope(self, parent, first, second):
        dag = self.graph(parent)
        left = set().union(*(self.affected_tasks(parent, edit) for edit in first.edits))
        right = set().union(*(self.affected_tasks(parent, edit) for edit in second.edits))
        if not left <= set(dag.tasks) or not right <= set(dag.tasks):
            raise ValueError('交互分析引用了未知任务')
        adjacent = {key for key, task in dag.tasks.items()
                    if left & set(task.dependencies)}
        adjacent.update(dependency for key in left for dependency in dag.tasks[key].dependencies)
        shared_ranks = sorted({dag.tasks[key].rank for key in left} &
                              {dag.tasks[key].rank for key in right})
        channels_left = {dag.tasks[key].metadata['channel'] for key in left
                         if 'channel' in dag.tasks[key].metadata}
        channels_right = {dag.tasks[key].metadata['channel'] for key in right
                          if 'channel' in dag.tasks[key].metadata}
        reasons = []
        if left & right:
            reasons.append('shared_tasks')
        if adjacent & right:
            reasons.append('adjacent_dependencies')
        if channels_left & channels_right:
            reasons.append('shared_communication_channels')
        return dict(related=bool(reasons), reasons=reasons,
            shared_task_ids=sorted(left & right), shared_gpu_ranks=shared_ranks,
            dependency_connected_task_ids=sorted(adjacent & right),
            shared_communication_channels=sorted(channels_left & channels_right),
            predicted_interaction_ms=None,
            scope='dependency_and_device_overlap_only',
            missing_measurements=['四个完整计划的实测或校准预测成本', '共享网络干扰'])

    def related_programs(self, parent, proposals, evidence, candidate_budget):
        if type(candidate_budget) is not int or candidate_budget <= 0:
            raise ValueError('关联候选预算必须为正整数')
        proposals = [StrategyProposal.model_validate(proposal) for proposal in proposals]
        if any(proposal.parent_candidate_id not in (None, parent.candidate_id)
               for proposal in proposals):
            raise ValueError('关联搜索的基本操作必须属于同一个原始计划')
        singles = [self.compile(parent, proposal, evidence) for proposal in proposals]
        output = []
        seen = {candidate.candidate_id for candidate in singles}
        for index, first in enumerate(proposals):
            first_variables = set(singles[index].transformation_record['changed_variables'])
            for second_index in range(index + 1, len(proposals)):
                second = proposals[second_index]
                second_variables = set(singles[second_index].transformation_record['changed_variables'])
                if first_variables & second_variables:
                    continue
                relation = self.interaction_scope(parent, first, second)
                if not relation['related']:
                    continue
                combined = self.compose(parent, first, second, evidence)
                if combined.candidate_id in seen:
                    continue
                seen.add(combined.candidate_id)
                # 配置变化后的成本需要重新测量，结构关联不产生加速预测。
                combined.transformation_record['search_selection'] = dict(
                    component_candidate_ids=[singles[index].candidate_id,
                                             singles[second_index].candidate_id],
                    candidate_budget=candidate_budget,
                    predicted_step_time_ms=None,
                    ordering='proposal_order_then_dependency_related_pairs',
                    cost_reuse_allowed=not bool(
                        set(combined.transformation_record['changed_variables']) &
                        {'tp', 'cp', 'ep', 'pp', 'vpp'}))
                output.append(combined)
                if len(output) == candidate_budget:
                    return output
        return output

    def complete_parallel_change(self, parent, changes, evidence_ids, evidence, candidate_budget=6):
        if not changes or set(changes) - {'tp', 'cp', 'pp', 'ep'}:
            raise ValueError('主变换必须指定受支持的混合并行维度')
        if type(candidate_budget) is not int or candidate_budget <= 0:
            raise ValueError('策略补全预算必须为正整数')
        original = parent.parallel_config
        dimensions = {key: changes.get(key, getattr(original, key)) for key in ('tp', 'cp', 'pp', 'ep')}
        if any(type(value) is not int or value <= 0 for value in dimensions.values()):
            raise ValueError('主变换的并行度必须为正整数')
        divisor = dimensions['tp'] * dimensions['cp'] * dimensions['pp']
        if self.topology.num_gpus % divisor:
            raise ValueError('主变换无法保持实际GPU数量')
        dp = self.topology.num_gpus // divisor
        if (dimensions['cp'] * dp) % dimensions['ep']:
            if 'ep' in changes:
                raise ValueError('主变换显式指定的EP不能构造合法Expert Folding')
            capacity = dimensions['cp'] * dp
            legal_ep = [value for value in range(1, min(dimensions['ep'], capacity) + 1)
                if capacity % value == 0 and
                (self.model.num_experts is not None and self.model.num_experts % value == 0
                 or self.model.num_experts is None and value == 1) and
                ('ep' not in self.backend_capabilities or value in self.backend_capabilities['ep'])]
            if not legal_ep:
                raise ValueError('主变换没有后端支持的Expert Folding补全')
            dimensions['ep'] = max(legal_ep)
        closed_changes = changes | {key: value for key, value in dimensions.items()
            if value != getattr(original, key)}
        if original.global_batch_size % dp:
            raise ValueError('全局Batch无法被新的DP划分')
        replica_batch = original.global_batch_size // dp
        mb_sizes = [original.micro_batch_size]
        old_shards = original.cp * (original.tp if original.use_sequence_parallel else 1)
        new_shards = dimensions['cp'] * dimensions['tp']
        numerator = original.micro_batch_size * new_shards
        if numerator % old_shards == 0:
            token_preserving_mbs = numerator // old_shards
            if (token_preserving_mbs > 0 and replica_batch % token_preserving_mbs == 0
                    and token_preserving_mbs not in mb_sizes):
                mb_sizes.append(token_preserving_mbs)
        # 从流水线就绪条件推导补偿MBS，保持GBS并保证至少PP个Microbatch。
        compensation = replica_batch // dimensions['pp']
        if compensation > 0 and replica_batch % compensation == 0 and compensation not in mb_sizes:
            mb_sizes.append(compensation)
        vpps = list(dict.fromkeys([original.vpp, 2, 1]))
        known_tasks = self.graph(parent)
        target = next(key for key, task in known_tasks.tasks.items() if task.is_compute)
        candidates, rejected = [], []
        for mbs in mb_sizes:
            for vpp in vpps:
                values = dimensions | dict(dp=dp, mbs=mbs, vpp=vpp)
                reasons = []
                if replica_batch % mbs:
                    reasons.append('GBS不能被DP乘MBS整除')
                microbatches = replica_batch // mbs
                if vpp > 1 and microbatches < dimensions['pp']:
                    reasons.append('Microbatch数量小于PP，当前交错执行器不支持')
                group = dimensions['pp'] if vpp > 1 else 1
                if not valid_microbatch_group(microbatches, dimensions['pp'], vpp, group):
                    reasons.append('Microbatch数量不能完整划分为调度组')
                if dimensions['pp'] * vpp > self.model.num_layers:
                    reasons.append('模型层数不足以形成非空虚拟Region')
                for key, value in (dimensions | dict(vpp=vpp)).items():
                    if key in self.backend_capabilities and value not in self.backend_capabilities[key]:
                        reasons.append(f'当前后端不支持{key}={value}')
                if self.model.num_attention_heads % dimensions['tp']:
                    reasons.append('Attention heads不能被TP整除')
                if self.model.num_experts is not None and self.model.num_experts % dimensions['ep']:
                    reasons.append('Expert数量不能被EP整除')
                if reasons:
                    rejected.append(dict(config=values, reasons=reasons))
                    continue
                config = original.with_updates(**dimensions, dp=dp, etp=dimensions['tp'],
                    micro_batch_size=mbs, vpp=vpp, microbatch_group=group,
                    use_sequence_parallel=dimensions['tp'] > 1)
                if config.total_vp_stages == len(parent.region_layout):
                    counts = [region.num_layers for region in parent.region_layout]
                else:
                    counts = [region.num_layers for region in
                        RegionPartitioner(self.model, config).generate_uniform_regions()]
                edits = [dict(operation='SetParallelism', parameters=closed_changes | dict(vpp=vpp)),
                    dict(operation='SetRegionLayout', parameters=dict(decoder_chunks=counts)),
                    dict(operation='SetMicrobatchSize', parameters=dict(micro_batch_size=mbs)),
                    dict(operation='SetMicrobatchGroup', parameters=dict(microbatch_group=group))]
                if vpp == 1 and original.execution_implementation.get('overlap_p2p', False):
                    edits.append(dict(operation='ConfigureP2POverlap', parameters=dict(enabled=False)))
                proposal = dict(name='constraint_closed', missing_measurements=[
                    '新的Kernel与通信成本', '实际峰值显存', '完整迭代吞吐量'],
                    edits=[edit | dict(target_task_ids=[target], evidence_ids=evidence_ids,
                        hypothesis='并行变换改变Replica Batch与流水线就绪条件，联合补全MBS和VPP')
                        for edit in edits])
                candidate = self.compile(parent, proposal, evidence)
                candidate.transformation_record['constraint_closure'] = dict(
                    primary_change=changes,
                    closed_parallel_change=closed_changes,
                    necessary=dict(dp=dp, ep=dimensions['ep'], etp=dimensions['tp'],
                        expert_data_parallel=dimensions['cp'] * dp // dimensions['ep']),
                    optional=dict(micro_batch_size=mbs, vpp=vpp, microbatch_group=group),
                    before_microbatches=original.num_microbatches,
                    after_microbatches=config.num_microbatches,
                    before_local_sequence=original.sequence_length // original.cp,
                    after_local_sequence=config.sequence_length // config.cp,
                    handoff_token_count=[original.sequence_length * original.micro_batch_size // old_shards,
                        config.sequence_length * config.micro_batch_size // new_shards],
                    region_completion='preserved' if config.total_vp_stages == len(parent.region_layout)
                        else 'legal_uniform_reference_requires_region_optimization',
                    performance_impact=dict(compute_shape_changed=bool(
                        set(changes) & {'tp', 'cp'} or mbs != original.micro_batch_size),
                        collective_groups_changed=True, schedule_changed=True,
                        memory_admission='requires_actual_measurement', predicted_step_time_ms=None))
                candidates.append(candidate)
        candidates.sort(key=lambda candidate: (
            candidate.parallel_config.vpp != original.vpp,
            candidate.parallel_config.micro_batch_size != original.micro_batch_size,
            -candidate.parallel_config.vpp))
        return dict(candidates=candidates[:candidate_budget], rejected=rejected,
            legal_candidate_count=len(candidates), candidate_budget=candidate_budget,
            necessary_rule='world=TP*CP*PP*DP; expert_group=ETP*EP',
            compensation_rule='保持P2P接口Token数量的MBS，以及满足PP交错就绪条件的MBS；逐项检查实际调度约束')

    def compose(self, parent, first, second, evidence):
        relation = self.interaction_scope(parent, first, second)
        if not relation['related']:
            raise ValueError('两个变换没有已建立的依赖或设备关联')
        merged = {}
        for edit in first.edits + second.edits:
            entry = merged.setdefault(edit.operation, dict(parameters={}, task_ids=set(),
                evidence_ids=set(), hypotheses=[]))
            for key, value in edit.parameters.model_dump(exclude_none=True).items():
                if key in entry['parameters'] and entry['parameters'][key] != value:
                    raise ValueError(f'组合变换冲突：{edit.operation}.{key}')
                entry['parameters'][key] = value
            entry['task_ids'].update(edit.target_task_ids)
            entry['evidence_ids'].update(edit.evidence_ids)
            entry['hypotheses'].append(edit.hypothesis)
        proposal = StrategyProposal(name='coupled_program', missing_measurements=sorted(
            set(first.missing_measurements + second.missing_measurements)),
            edits=[StrategyEdit(operation=operation,
                parameters=StrategyParameters.model_validate(entry['parameters']),
                target_task_ids=sorted(entry['task_ids']), evidence_ids=sorted(entry['evidence_ids']),
                hypothesis='；'.join(entry['hypotheses'])) for operation, entry in merged.items()])
        candidate = self.compile(parent, proposal, evidence)
        candidate.transformation_record['interaction_scope'] = relation
        candidate.transformation_record['component_proposals'] = [first.model_dump(mode='json'),
                                                                 second.model_dump(mode='json')]
        return candidate

    def parameter_conflicts(self, proposal):
        allowed = {'SetParallelism': {'tp', 'cp', 'pp', 'ep', 'vpp'},
            'SetRegionLayout': {'decoder_chunks'},
            'SetMicrobatchGroup': {'microbatch_group'},
            'SetMicrobatchSize': {'micro_batch_size'},
            'ConfigureGradientBucket': {'bucket_elements'},
            'ConfigureP2POverlap': {'enabled'}}
        conflicts = []
        for index, edit in enumerate(proposal.edits):
            supplied = set(edit.parameters.model_dump(exclude_none=True))
            unexpected = supplied - allowed[edit.operation]
            if unexpected:
                conflicts.append(dict(edit_index=index, operation=edit.operation,
                    unexpected_fields=sorted(unexpected), allowed_fields=sorted(allowed[edit.operation]),
                    repair='将其他参数移到对应操作；未使用字段设置null'))
            if edit.operation == 'SetParallelism':
                for variable, value in edit.parameters.model_dump(exclude_none=True).items():
                    supported = self.backend_capabilities.get(variable)
                    if supported is not None and value not in supported:
                        conflicts.append(dict(edit_index=index, variable=variable, requested=value,
                            supported_values=supported, repair='使用当前执行入口支持的取值'))
        return conflicts

    def graph(self, candidate):
        valid, errors = validate_candidate(candidate.parallel_config, self.model,
                                          self.topology, candidate.region_layout)
        if not valid:
            raise ValueError(errors)
        dag = TaskGraphBuilder().build(self.model, candidate.parallel_config,
            candidate.region_layout, candidate.parallel_config.num_microbatches)
        if dag.has_cycle():
            raise ValueError('训练任务图存在循环依赖')
        return dag

    def observation(self, candidate, evidence, history):
        dag = self.graph(candidate)
        graph = dict(config=asdict(candidate.parallel_config),
            regions=[asdict(region) for region in candidate.region_layout],
            tasks={key: dict(rank=task.rank, virtual_stage=task.metadata.get('vp_stage'),
                microbatch=task.microbatch_id, operation=task.task_type.value,
                region_id=task.region_id, dependencies=task.dependencies,
                metadata=task.metadata) for key, task in dag.tasks.items()})
        graph_hash = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
        # 内部Collective保持复合计算语义，避免将未提取的操作伪装成独立节点。
        graph.update(graph_sha256=graph_hash, representation='region_training_task_dag',
            opaque_operations=['region_internal_tp_cp_ep_collectives'],
            semantic_sources=['Megatron PipelineParallelLayerLayout',
                'Megatron get_schedule_table', 'Megatron gradient_finalize'],
            data_gaps=['独立通信等待归因', '层内算子依赖', '候选峰值显存预测'])
        return dict(parent_candidate=candidate.to_dict(), training_graph=graph,
            backend_capabilities=self.backend_capabilities,
            model=dict(name=self.model.name, num_layers=self.model.num_layers,
                hidden_size=self.model.hidden_size, num_attention_heads=self.model.num_attention_heads,
                num_experts=self.model.num_experts, layer_types=[kind.value for kind in self.model.layer_types]),
            backend_constraints=dict(world_size=self.topology.num_gpus,
                memory_limit_gib=self.topology.memory_per_gpu_gb,
                equations=['world=TP*CP*PP*DP', 'TP*CP*DP must divide into ETP*EP groups',
                    'GBS%(DP*MBS)=0', 'microbatches=GBS/(DP*MBS)',
                    'PP<=microbatch_group<=microbatches for VPP>1',
                    'microbatches%microbatch_group=0 or remainder>=PP for VPP>1', 'world=ETP*EP*ExpertDP*PP']),
            evidence=evidence, history=history,
            capabilities={
                'SetParallelism': dict(parameters=['tp', 'cp', 'pp', 'ep', 'vpp'],
                    requires='DP按实际GPU数量推导；Agent查询策略依赖并明确选择EP、MBS、VPP及完整Region'),
                'SetRegionLayout': dict(parameters=['decoder_chunks'],
                    requires='全部层覆盖一次，每个虚拟阶段非空'),
                'SetMicrobatchGroup': dict(parameters=['microbatch_group'],
                    requires='Megatron原生VPP分组且全部microbatch完整分组'),
                'SetMicrobatchSize': dict(parameters=['micro_batch_size'],
                    requires='保持GBS，重新检查Microbatch数量、调度组和显存'),
                'ConfigureGradientBucket': dict(parameters=['bucket_elements'],
                    requires='Megatron DDP实际bucket配置'),
                'ConfigureP2POverlap': dict(parameters=['enabled'],
                    requires='原生VPP实际P2P实现')},
            unavailable=['任意F/B顺序的正式数值认证', '独立TP/EP Kernel重排',
                         '不同Rank虚拟阶段数量', '任意算子切分'])

    @staticmethod
    def candidate_context(candidate):
        full = candidate.to_dict()
        record = candidate.transformation_record
        compact = {key: record[key] for key in ('changed_variables', 'rewrite_families', 'validation_scope',
            'memory_admission', 'constraint_closure', 'search_selection') if key in record}
        for name in ('graph_delta', 'reference_graph_delta', 'interaction_scope'):
            if name not in record:
                continue
            delta = {}
            for key, value in record[name].items():
                if isinstance(value, list):
                    delta[key] = value[:16]
                    delta[key + '_count'] = len(value)
                elif key == 'impact':
                    delta[key] = {field: (items[:16] if isinstance(items, list) else items)
                        for field, items in value.items()}
                    delta[key]['compute_task_count'] = len(value['compute_task_ids'])
                    delta[key]['communication_task_count'] = len(value['communication_task_ids'])
                else:
                    delta[key] = value
            compact[name] = delta
        if 'proposal' in record:
            proposal = dict(record['proposal'])
            proposal['edits'] = [edit | dict(target_task_ids=edit['target_task_ids'][:16],
                full_target_task_count=len(edit['target_task_ids'])) for edit in proposal['edits']]
            compact['proposal'] = proposal
        encoded = json.dumps(full, sort_keys=True).encode()
        compact.update(full_record_sha256=hashlib.sha256(encoded).hexdigest(),
            full_record_bytes=len(encoded),
            context_scope='complete_config_region_layout_and_bounded_graph_difference',
            detail_access='完整候选由控制器保存；按candidate_id查询实际任务与依赖')
        return full | dict(transformation_record=compact)

    @staticmethod
    def agent_context(observation):
        graph = observation['training_graph']
        config = graph['config']
        lanes = config['tp'] * config['cp'] * config['dp']
        selected = {key: task for key, task in graph['tasks'].items()
                    if task['rank'] % lanes == 0 and task['microbatch'] in (None, 0)}
        external = sorted({dependency for task in selected.values() for dependency in task['dependencies']
                           if dependency not in selected})
        evidence = []
        for item in observation['evidence']:
            summary = {key: item[key] for key in ('evidence_id', 'directory', 'plan', 'status',
                'tokens_per_second', 'step_seconds', 'peak_allocated_gib', 'warmup', 'measured',
                'numerical_scope', 'performance_eligible', 'cause_attribution') if key in item}
            execution = item.get('actual_execution', {})
            summary['actual_execution'] = dict(scope=execution.get('scope'),
                complete_rank_coverage=execution.get('complete_rank_coverage'),
                ranks={rank: {key: record[key] for key in ('process_groups', 'manifest_sha256',
                    'semantic_graph_sha256', 'boundary_sha256') if key in record}
                    for rank, record in execution.get('ranks', {}).items()})
            summary['detail_access'] = '通过evidence_id查询完整模块边界和语义Region记录'
            if 'task_trace' in item:
                summary['task_trace_available'] = True
                summary['task_trace'] = dict(cross_rank_clock_alignment=item['task_trace']['cross_rank_clock_alignment'],
                    rank_count=len(item['task_trace']['ranks']), detail_access='inspect_task_trace',
                    performance_eligible=False, missing_measurements=item['task_trace']['missing_measurements'])
            evidence.append(summary)
        return observation | {'evidence': evidence, 'training_graph': graph | dict(tasks=selected,
            full_task_count=len(graph['tasks']), external_dependency_ids=external,
            context_scope='lane0_microbatch0_with_cross_region_and_update_interfaces',
            expansion_contract='相同Region任务对全部lane和microbatch展开；完整图由编译器保存并检查')}

    def compile(self, parent, proposal, evidence):
        proposal = StrategyProposal.model_validate(proposal)
        if proposal.parent_candidate_id not in (None, parent.candidate_id):
            raise ValueError('变换的原始程序身份不匹配')
        original = self.graph(parent)
        known_evidence = {item['evidence_id'] for item in evidence}
        config = parent.parallel_config
        changes = {}
        implementation = dict(config.execution_implementation)
        counts = None
        touched = set()
        for edit in proposal.edits:
            if not set(edit.target_task_ids) <= set(original.tasks):
                raise ValueError('变换引用了原始训练图中不存在的任务')
            if not set(edit.evidence_ids) <= known_evidence:
                raise ValueError('变换引用了未提供的实际证据')
            params = edit.parameters.model_dump(exclude_none=True)
            if edit.operation == 'SetParallelism':
                if not params or set(params) - {'tp', 'cp', 'pp', 'ep', 'vpp'}:
                    raise ValueError('并行变换包含未知参数')
                if any(type(value) is not int or value <= 0 for value in params.values()):
                    raise ValueError('并行度必须为正整数')
                keys = set(params)
                changes.update(params)
            elif edit.operation == 'SetRegionLayout':
                if set(params) != {'decoder_chunks'} or not isinstance(params['decoder_chunks'], list):
                    raise ValueError('Region变换必须提供完整层数列表')
                counts = params['decoder_chunks']
                if any(type(value) is not int for value in counts):
                    raise ValueError('Region层数必须为整数')
                keys = {'decoder_chunks'}
            elif edit.operation == 'SetMicrobatchGroup':
                if set(params) != {'microbatch_group'} or type(params['microbatch_group']) is not int:
                    raise ValueError('Microbatch变换必须提供整数分组')
                changes.update(params)
                keys = set(params)
            elif edit.operation == 'SetMicrobatchSize':
                if (set(params) != {'micro_batch_size'} or type(params['micro_batch_size']) is not int
                        or params['micro_batch_size'] <= 0):
                    raise ValueError('Microbatch大小必须为正整数')
                changes.update(params)
                keys = set(params)
            elif edit.operation == 'ConfigureGradientBucket':
                if (set(params) != {'bucket_elements'} or
                        type(params['bucket_elements']) is not int or params['bucket_elements'] <= 0):
                    raise ValueError('梯度Bucket必须提供正整数元素数量')
                implementation.update(params)
                keys = set(params)
            else:
                if set(params) != {'enabled'} or type(params['enabled']) is not bool:
                    raise ValueError('P2P重叠必须提供布尔开关')
                implementation['overlap_p2p'] = params['enabled']
                keys = {'overlap_p2p'}
            if keys & touched:
                raise ValueError(f'联合提案重复修改同一变量：{sorted(keys & touched)}')
            touched.update(keys)
        dimensions = {key: changes.get(key, getattr(config, key)) for key in ('tp', 'cp', 'pp')}
        divisor = dimensions['tp'] * dimensions['cp'] * dimensions['pp']
        if self.topology.num_gpus % divisor:
            raise ValueError('联合并行度不能整除实际GPU数量')
        changes.update(dp=self.topology.num_gpus // divisor, etp=dimensions['tp'],
                       use_sequence_parallel=dimensions['tp'] > 1,
                       execution_implementation=implementation)
        config = config.with_updates(**changes)
        for variable, allowed in self.backend_capabilities.items():
            if getattr(config, variable) not in allowed:
                raise BackendCapabilityConflict(variable, getattr(config, variable), allowed)
        if counts is None:
            if config.total_vp_stages != len(parent.region_layout):
                raise ValueError('PP/VPP变换必须同时提供对应的完整Region分配')
            counts = [region.num_layers for region in parent.region_layout]
        regions = RegionPartitioner(self.model, config).from_counts(counts)
        valid, errors = validate_candidate(config, self.model, self.topology, regions)
        if not valid:
            raise ValueError(errors)
        if config.vpp == 1 and implementation.get('overlap_p2p', False):
            raise ValueError('当前后端的P2P重叠要求VPP大于一')
        payload = dict(config=asdict(config), chunks=counts)
        source = dict(config=asdict(parent.parallel_config),
                      chunks=[region.num_layers for region in parent.region_layout])
        if payload == source:
            raise ValueError('提案没有改变实际执行程序')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        candidate = Candidate(f'{proposal.name}-{digest}', parent.candidate_id, config, regions,
            [edit.operation for edit in proposal.edits])
        candidate.transformation_record = dict(proposal=proposal.model_dump(mode='json'),
            parent_config=asdict(parent.parallel_config), parent_chunks=source['chunks'],
            changed_variables=sorted(touched),
            validation_scope='structural_and_backend_legality',
            memory_admission='requires_actual_measurement')
        self.graph(candidate)
        candidate.transformation_record['graph_delta'] = self.graph_delta(parent, candidate)
        config_variables = set(candidate.transformation_record['graph_delta']['config_changes'])
        families = []
        if config_variables & {'tp', 'cp', 'dp', 'ep', 'etp'}:
            families.append('RebindParallel')
        if touched & {'decoder_chunks', 'pp', 'vpp'}:
            families.append('RepartitionGraph')
        if touched & {'micro_batch_size', 'microbatch_group', 'bucket_elements', 'overlap_p2p'}:
            families.append('RescheduleGraph')
        candidate.transformation_record['rewrite_families'] = families
        return candidate
