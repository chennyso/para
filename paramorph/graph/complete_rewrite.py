from dataclasses import asdict

from ortools.sat.python import cp_model

from contracts import RewriteIntent, fingerprint
from cost.runtime_calibration import configuration
from core_data_models import Candidate
from region_partition.region_partitioner import RegionPartitioner
from validation.parallel_validator import validate_candidate
from agents.program_agent import ProgramAgent
from agents.strategy_transformations import StrategyTransformationCompiler
from graph.process_groups import process_groups
from predictor.physical_paths import communication_resources


class CompleteAndRewrite:
    def __init__(self, model, topology, registry, domains):
        self.model, self.topology, self.registry = model, topology, registry
        maximum_tp = topology.num_gpus
        self.domains = dict(domains, tp=[value for value in domains['tp'] if value <= maximum_tp])
        if not self.domains['tp']:
            raise ValueError('当前搜索域没有可执行的Tensor并行候选')
        self.compiler = StrategyTransformationCompiler(model, topology, self.domains)

    def candidate(self, request, parent_id=None):
        maximum_tp = self.topology.num_gpus
        if type(request['tp']) is not int or not 1 <= request['tp'] <= maximum_tp:
            raise ValueError(f'Tensor并行候选必须满足1≤TP≤{maximum_tp}')
        config = configuration(request)
        regions = RegionPartitioner(self.model, config).from_counts(request['decoder_chunks'])
        valid, errors = validate_candidate(config, self.model, self.topology, regions)
        if not valid:
            raise ValueError(errors)
        return Candidate(request['run_id'], parent_id, config, regions, [])

    def preview_transformation(self, current_plan, intent):
        request = current_plan.backend_config if hasattr(current_plan, 'backend_config') else current_plan
        return self.compiler.preview_transformation(self.candidate(request), RewriteIntent.model_validate(intent))

    @staticmethod
    def flow_tasks(graph):
        flows = {}
        for task_id, task in graph.tasks.items():
            flow_id = StrategyTransformationCompiler.task_flow_id(task)
            if flow_id is not None:
                flows.setdefault(flow_id, set()).add(task_id)
        return flows

    def complete_and_rewrite(self, current_plan, intent, model=None, hardware=None,
                             backend_capabilities=None, limit=6, *, allow_unchanged=False):
        if model is not None and model != self.model:
            raise ValueError('图变换模型与初始化模型不一致')
        if hardware is not None and hardware != self.topology:
            raise ValueError('图变换硬件与初始化硬件不一致')
        if backend_capabilities is not None and backend_capabilities != self.registry.describe():
            raise ValueError('图变换后端能力与当前注册记录不一致')
        request = current_plan.backend_config if hasattr(current_plan, 'backend_config') else current_plan
        intent = RewriteIntent.model_validate(intent)
        if intent.parent_plan_id is not None and intent.parent_plan_id != request['run_id']:
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['意图的父计划与当前计划不同']))
        rows = self.complete(request, intent, limit, allow_unchanged=allow_unchanged)
        physical_topology = dict(devices=self.topology.devices, paths=self.topology.network_paths,
            environment_per_node=self.topology.environment_per_node)
        for row in rows:
            candidate = self.candidate(row['request'])
            row['execution_graph'] = self.compiler.observation(candidate, [], [])['training_graph']
            row['applied_changes'] = {key: dict(before=request.get(key), after=value)
                for key, value in row['request'].items() if request.get(key) != value and key != 'run_id'}
            row['effect_preview'] = self.preview_transformation(request, intent)
            row['active_decisions'] = list(intent.values)
            row['completed_decisions'] = [key for key in row['applied_changes'] if key not in intent.values]
            row['process_groups'] = process_groups(row['request'])
            mapping = row['process_groups']['rank_mapping']
            resource_changes = {}
            for task_id, task in row['execution_graph']['tasks'].items():
                metadata = task.get('metadata', {})
                participants = metadata.get('group_ranks')
                if participants and self.topology.devices:
                    physical_ranks = [mapping[rank] for rank in participants]
                    binding = communication_resources(physical_topology, physical_ranks)
                    metadata['physical_communication_resources'] = binding
                    resource_changes[task_id] = dict(physical_ranks=physical_ranks,
                        resource_ids=binding['resource_ids'], missing_topology=binding['missing_topology'])
            row['graph_delta']['physical_communication_resources'] = resource_changes
            row['execution_graph']['graph_sha256'] = fingerprint({
                key: value for key, value in row['execution_graph'].items() if key != 'graph_sha256'})
            row['graph_delta']['buffer_lifetime_changes'] = {}
            before_buffers = self.compiler.buffer_lifetimes(self.candidate(request))
            after_buffers = self.compiler.buffer_lifetimes(candidate)
            for key in sorted(set(before_buffers) | set(after_buffers)):
                if before_buffers.get(key) != after_buffers.get(key):
                    row['graph_delta']['buffer_lifetime_changes'][key] = dict(
                        before=before_buffers.get(key), after=after_buffers.get(key))
            row['graph_delta']['buffer_lifetime_status'] = 'structural_allocation_and_release_task_dependencies'
            row['graph_delta']['process_group_changes'] = {
                key: dict(before=process_groups(request)['logical'][key], after=row['process_groups']['logical'][key])
                for key in row['process_groups']['logical']
                if process_groups(request)['logical'][key] != row['process_groups']['logical'][key]}
            changed = (set(row['graph_delta']['affected_task_ids']) |
                       set(row['graph_delta']['removed_task_ids']))
            flow_tasks = self.flow_tasks(self.compiler.graph(self.candidate(request)))
            changed_flows = {flow_id for flow_id, task_ids in flow_tasks.items() if task_ids & changed}
            changed_targets = set(intent.target_task_ids) & changed
            changed_targets.update(set(intent.target_flow_ids) & changed_flows)
            targets = set(intent.target_task_ids) | set(intent.target_flow_ids)
            row['graph_delta']['target_impact'] = dict(target_ids=sorted(targets),
                changed_target_ids=sorted(changed_targets),
                status='AFFECTED' if changed_targets else 'UNCHANGED' if targets else 'UNSPECIFIED')
            row['constraint_result'] = dict(status='PASS', scope='native_backend_structural_legality',
                numerical_validation='requires_actual_training')
            row['layout_conversions'] = dict(communication_task_ids=row['graph_delta']['impact']['communication_task_ids'],
                opaque_collectives=row['execution_graph']['opaque_operations'])
        return rows

    def complete(self, parent_request, intent, limit, *, allow_unchanged=False):
        if type(limit) is not int or limit < 1:
            raise ValueError('候选数量上限必须为正整数')
        intent = RewriteIntent.model_validate(intent)
        self.registry.require(intent.operation)
        for action in intent.proposed_actions:
            self.registry.require(action['operation'])
        groups = process_groups(parent_request)
        parent_request = dict(parent_request,
            etp=parent_request.get('etp', parent_request['tp']),
            expert_dp=groups['expert_dp_size'],
            precision_aware_optimizer=parent_request.get('precision_aware_optimizer', False),
            optimizer_sharding=parent_request.get('optimizer_sharding', 'distributed'),
            execution_backend=parent_request.get('execution_backend', 'megatron_ddp'),
            rank_mapping=parent_request.get('rank_mapping', list(range(self.topology.num_gpus))))
        dimensions = ('tp', 'pp', 'cp', 'ep', 'etp', 'expert_dp', 'dp', 'vpp', 'micro_batch', 'group')
        allowed_fields = set(dimensions) | {'decoder_chunks', 'recompute', 'bucket_elements', 'overlap_p2p',
            'rank_mapping', 'pipeline_schedule', 'optimizer_sharding', 'precision_aware_optimizer',
            'offload', 'optimizer_offload_fraction', 'optimizer_transfer_overlap', 'memory_policy',
            'execution_backend', 'megatron_fsdp_version', 'fsdp_sharding_strategy', 'dense_sharding_strategy',
            'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size', 'outer_sharding_strategy',
            'expert_dp_inner_size', 'expert_dp_outer_size', 'expert_outer_sharding_strategy',
            'fsdp_unit', 'fsdp_prefetch', 'fsdp_prefetch_distance', 'fsdp_reshard_after_forward',
            'fsdp_double_buffer', 'fsdp_bucket_elements',
            'megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype'}
        if (set(intent.values) | set(intent.adjustable) | set(intent.preserve)) - allowed_fields:
            raise ValueError('变换意图包含未注册的变量')
        if set(intent.allowed_ranges) - allowed_fields:
            raise ValueError('意图的允许范围包含未注册变量')
        for name, domain in intent.allowed_ranges.items():
            if not domain:
                raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=[f'{name}的意图允许范围为空']))
            if name in intent.values and intent.values[name] not in domain:
                raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=[f'{name}的主动选择超出意图允许范围']))
        if any(intent.values[name] != parent_request.get(name)
               for name in set(intent.values) & set(intent.preserve)):
            conflicts = sorted(name for name in set(intent.values) & set(intent.preserve)
                if intent.values[name] != parent_request.get(name))
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=conflicts))
        adjustable = set(intent.adjustable) - set(intent.preserve)
        parent = self.candidate(parent_request)
        graph = self.compiler.graph(parent)
        if not set(intent.target_task_ids) <= set(graph.tasks):
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['优化意图引用了不存在的目标Task ID']))

        if not set(intent.target_flow_ids) <= set(self.flow_tasks(graph)):
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['优化意图引用了不存在的目标Flow ID']))
        solver_model = cp_model.CpModel()
        variables = {}
        completion_changes = []
        for name in dimensions:
            initial = parent_request.get(name, getattr(parent.parallel_config, name, None))
            if name == 'expert_dp' and initial is None:
                initial = self.topology.num_gpus // (parent.parallel_config.etp * parent.parallel_config.ep * parent.parallel_config.pp)
            if initial is None:
                raise ValueError(f'原始计划缺少{name}')
            if name in intent.values:
                values = [intent.values[name]]
                domain = self.domains.get(name, self.domains.get(
                    {'micro_batch': 'micro_batch_size', 'group': 'microbatch_group'}.get(name, name)))
                if domain is not None and values[0] not in domain:
                    raise ValueError(dict(status='CONSTRAINT_CONFLICT',
                        conflicts=[f'{name}的目标值不属于当前搜索允许范围']))
            elif (name in adjustable or name == 'expert_dp') and name not in intent.preserve:
                values = self.domains.get(name, self.domains.get({'micro_batch': 'micro_batch_size', 'group': 'microbatch_group'}.get(name, name)))
                if name in ('etp', 'expert_dp') and not values:
                    values = [value for value in range(1, self.topology.num_gpus + 1)
                              if self.topology.num_gpus % value == 0]
                if not values:
                    raise ValueError(f'约束补全缺少{name}的后端允许范围')
            else:
                values = [initial]
            if name == 'tp':
                maximum_tp = max(self.domains['tp'])
                values = [value for value in values if value <= maximum_tp]
                if not values:
                    raise ValueError(dict(status='CONSTRAINT_CONFLICT',
                        conflicts=[f'Tensor并行目标必须满足TP≤{maximum_tp}']))
            if name in intent.allowed_ranges:
                values = [value for value in values if value in intent.allowed_ranges[name]]
                if not values:
                    raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=[f'{name}没有同时满足后端和意图范围的值']))
            if any(type(value) is not int or value < 1 for value in values):
                raise ValueError(f'{name}的允许值必须为正整数')
            variables[name] = solver_model.new_int_var_from_domain(cp_model.Domain.from_values(values), name)
            if name not in intent.values and len(values) > 1:
                changed = solver_model.new_bool_var(name + '_completion_changed')
                solver_model.add(variables[name] != initial).only_enforce_if(changed)
                solver_model.add(variables[name] == initial).only_enforce_if(changed.Not())
                completion_changes.append(changed)
        world = self.topology.num_gpus
        if self.model.num_experts is None:
            solver_model.add(variables['etp'] == variables['tp'])
            solver_model.add(variables['ep'] == 1)
        solver_model.add_multiplication_equality(world, [variables[name] for name in ('tp', 'pp', 'cp', 'dp')])
        solver_model.add_multiplication_equality(world,
            [variables[name] for name in ('etp', 'ep', 'expert_dp', 'pp')])
        batch_denominator = solver_model.new_int_var(1, parent_request['global_batch'], 'batch_denominator')
        solver_model.add_multiplication_equality(batch_denominator, [variables['dp'], variables['micro_batch']])
        microbatches = solver_model.new_int_var(1, parent_request['global_batch'], 'microbatches')
        solver_model.add_multiplication_equality(parent_request['global_batch'], [batch_denominator, microbatches])
        solver_model.minimize(sum(completion_changes))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        candidates = []
        conflicts = []
        deferred = []
        examined = 0
        while len(candidates) < limit and examined < 4096:
            status = solver.solve(solver_model)
            if status == cp_model.MODEL_INVALID:
                raise ValueError(solver_model.validate())
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            values = {key: solver.value(variable) for key, variable in variables.items()}
            solver_model.add_forbidden_assignments(list(variables.values()), [list(values.values())])
            examined += 1
            request = dict(parent_request, **values)
            for key in allowed_fields - set(dimensions):
                if key in intent.values:
                    request[key] = intent.values[key]
            range_conflict = False
            for key in set(intent.allowed_ranges) - set(dimensions) - {'decoder_chunks', 'pipeline_schedule'}:
                defaults = dict(rank_mapping=list(range(world)), optimizer_sharding='distributed', overlap_p2p=False,
                    precision_aware_optimizer=False, execution_backend='megatron_ddp',
                    fsdp_sharding_strategy='no_shard', dense_sharding_strategy='no_shard',
                    expert_sharding_strategy='no_shard',
                    dp_inner_size=request['dp'] * request['cp'], dp_outer_size=1,
                    expert_dp_inner_size=request['expert_dp'], expert_dp_outer_size=1,
                    outer_sharding_strategy='no_shard', expert_outer_sharding_strategy='no_shard',
                    fsdp_unit='transformer_layer', fsdp_prefetch=False, fsdp_prefetch_distance=1,
                    fsdp_reshard_after_forward=True, fsdp_double_buffer=False,
                    fsdp_bucket_elements=request.get('bucket_elements', 40000000))
                selected = request.get(key, defaults.get(key))
                domain = intent.allowed_ranges[key]
                if selected not in domain:
                    if key in adjustable and key not in intent.values:
                        request[key] = domain[0]
                    else:
                        conflicts.append(f'{key}的固定选择超出意图允许范围')
                        range_conflict = True
            if range_conflict:
                continue
            allowed_precision = parent_request.get('allowed_precision_aware_optimizer',
                [parent_request.get('precision_aware_optimizer', False)])
            if request.get('precision_aware_optimizer', False) not in allowed_precision:
                conflicts.append('Precision-Aware Optimizer不满足当前训练合同')
                continue
            backend = request.get('execution_backend', 'megatron_ddp')
            if backend != parent_request.get('execution_backend', 'megatron_ddp'):
                if 'optimizer_sharding' not in intent.values and 'optimizer_sharding' not in intent.preserve:
                    request['optimizer_sharding'] = 'distributed'
                request['checkpoint_format'] = parent_request.get('checkpoint_format', 'torch_dist')
            if backend == 'megatron_fsdp':
                request['megatron_fsdp_version'] = 1
                request['fsdp_sharding_strategy'] = request['dense_sharding_strategy']
                if request.get('megatron_fsdp_main_params_dtype') is None:
                    request['megatron_fsdp_main_params_dtype'] = 'float32'
                if request.get('megatron_fsdp_main_grads_dtype') is None:
                    request['megatron_fsdp_main_grads_dtype'] = request.get('main_grads_dtype',
                        'bfloat16' if request.get('precision_aware_optimizer', False) else 'float32')
            offload_fraction = float(request.get('optimizer_offload_fraction', 0.0))
            request['offload'] = offload_fraction > 0
            if offload_fraction == 0:
                request['optimizer_transfer_overlap'] = False
            if 'dp_inner_size' not in intent.values and 'dp_inner_size' not in intent.preserve:
                request['dp_inner_size'] = request['dp'] * request['cp']
                request['dp_outer_size'] = 1
                request['outer_sharding_strategy'] = 'no_shard'
            if 'expert_dp_inner_size' not in intent.values and 'expert_dp_inner_size' not in intent.preserve:
                request['expert_dp_inner_size'] = request['expert_dp']
                request['expert_dp_outer_size'] = 1
                request['expert_outer_sharding_strategy'] = 'no_shard'
            if request['recompute'] != parent_request['recompute'] and request.get('memory_policy') is not None:
                policy = dict(request['memory_policy'])
                policy.update(recompute_granularity=('full' if request['recompute'] == 'full' else
                    'selective' if request['recompute'] in ('selective-mlp', 'selective') else None),
                    recompute_modules=(policy['recompute_modules'] if request['recompute'] == 'selective' else
                        ['mlp'] if request['recompute'] == 'selective-mlp' else []))
                request['memory_policy'] = policy
            if (request['tp'] * request['cp'] * request['dp']) % (request['etp'] * request['ep']):
                conflicts.append('Expert Folding进程组不能整除设备集合')
                continue
            if request['global_batch'] % (request['dp'] * request['micro_batch']):
                conflicts.append('GBS必须被DP乘MBS整除')
                continue
            config = configuration(request)
            expected_schedule = 'interleaved_1f1b' if config.vpp > 1 else '1f1b'
            if ('pipeline_schedule' in intent.values or 'pipeline_schedule' in intent.preserve) and request.get(
                    'pipeline_schedule', expected_schedule) != expected_schedule:
                conflicts.append('Pipeline Schedule与实际PP/VPP后端入口不同')
                continue
            request['pipeline_schedule'] = expected_schedule
            if 'pipeline_schedule' in intent.allowed_ranges and expected_schedule not in intent.allowed_ranges['pipeline_schedule']:
                conflicts.append('原生Pipeline Schedule不在意图允许范围内')
                continue
            if config.vpp == 1 and 'overlap_p2p' not in intent.values and 'overlap_p2p' not in intent.preserve:
                request['overlap_p2p'] = False
            if request.get('optimizer_sharding', 'distributed') not in ('distributed', 'replicated', 'fsdp'):
                conflicts.append('Optimizer Sharding缺少原生执行接口')
                continue
            process_groups(request)
            slots = config.total_vp_stages
            if slots != len(parent_request['decoder_chunks']) and 'decoder_chunks' not in intent.values:
                if 'decoder_chunks' in intent.preserve:
                    conflicts.append('PP/VPP变化需要允许修改Region分区')
                    continue
                request['decoder_chunks'] = [self.model.num_layers // slots + (index < self.model.num_layers % slots) for index in range(slots)]
            if slots > self.model.num_layers:
                conflicts.append('非空Region数量超过模型层数')
                continue
            valid, errors = validate_candidate(config, self.model, self.topology)
            if not valid:
                conflicts.extend(errors)
                continue
            if config.vpp == 1 and request.get('overlap_p2p', False):
                conflicts.append('原生P2P重叠需要VPP大于一')
                continue
            layouts = [request['decoder_chunks']]
            if 'decoder_chunks' in adjustable and 'decoder_chunks' not in intent.values:
                layouts.extend([region.num_layers for region in row] for row in
                    RegionPartitioner(self.model, config).generate_diverse_regions(min(4, limit)))
            if 'decoder_chunks' in intent.allowed_ranges:
                if 'decoder_chunks' in adjustable and 'decoder_chunks' not in intent.values:
                    layouts = [row for row in intent.allowed_ranges['decoder_chunks'] if len(row) == slots
                        and all(type(count) is int and count > 0 for count in row) and sum(row) == self.model.num_layers]
                else:
                    layouts = [row for row in layouts if row in intent.allowed_ranges['decoder_chunks']]
                if not layouts:
                    conflicts.append('Region分区没有同时满足层覆盖及意图允许范围的值')
                    continue
            emitted = False
            for counts in dict.fromkeys(tuple(row) for row in layouts):
                if (len(counts) != slots or any(type(count) is not int or count < 1 for count in counts)
                        or sum(counts) != self.model.num_layers):
                    conflicts.append('Region分区需要使用正整数完整覆盖全部模型层并匹配PP乘VPP数量')
                    continue
                variant = dict(request, decoder_chunks=list(counts))
                if (not allow_unchanged
                        and all(variant.get(key) == parent_request.get(key) for key in allowed_fields)):
                    continue
                identity = fingerprint({key: value for key, value in variant.items()
                    if key not in ('run_id', 'port', 'warmup', 'measured')})[:16]
                variant['run_id'] = 'rewrite-' + identity
                candidate = self.candidate(variant, parent.candidate_id)
                candidate.transformation_record = dict(intent=intent.model_dump(mode='json'),
                    validation_scope='backend_structural_legality')
                program = ProgramAgent().compile(candidate)
                program['backend_request'] = variant
                delta = self.compiler.graph_delta(parent, candidate)
                candidate.transformation_record['graph_delta'] = delta
                program['program_sha256'] = fingerprint({key: value for key, value in program.items()
                    if key != 'program_sha256'})
                row = dict(request=variant, candidate=candidate.to_dict(), program=program, graph_delta=delta)
                if not emitted:
                    candidates.append(row)
                    emitted = True
                else:
                    deferred.append(row)
        seen = {row['request']['run_id'] for row in candidates}
        for row in deferred:
            if len(candidates) == limit:
                break
            if row['request']['run_id'] not in seen:
                seen.add(row['request']['run_id'])
                candidates.append(row)
        if not candidates:
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=sorted(set(conflicts)) or ['完整后端、进程组及Batch约束无解']))
        return candidates
