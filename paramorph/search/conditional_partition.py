from dataclasses import dataclass
from itertools import product, zip_longest
import math
import time

from contracts import ModuleRelation, RewriteIntent, fingerprint
from cost.runtime_calibration import configuration
from graph.process_groups import process_groups
from graph.native_recomputation import selective_modules
from predictor.profile_database import backend_reuse_evidence, normalized_execution, plan_signature, same_training_plan, signature
from predictor.measured_modules import annotate_measurement_semantics, compute_profile_quality, enrich_module_profile
from predictor.state_sharding import parameter_state_ownership, state_sharding_errors
from region_partition.region_partitioner import RegionPartitioner
from search.seed_planner import SeedPlanner
from search.module_relations import owner_patterns, prefix_support, validate_relations
from validation.parallel_validator import validate_candidate
from search.continuation import ContinuationGuide
from scheduling.native_microbatches import valid_microbatch_group


GLOBAL_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping')
LAYOUT_FIELDS = ('vpp', 'micro_batch', 'group', 'decoder_chunks', 'recompute',
    'pipeline_schedule', 'overlap_p2p')
STATE_FIELDS = tuple(name for name in SeedPlanner.family_fields() if name not in GLOBAL_FIELDS)
REWRITE_FIELDS = GLOBAL_FIELDS + LAYOUT_FIELDS + STATE_FIELDS + (
    'fsdp_sharding_strategy', 'fsdp_unit', 'fsdp_prefetch', 'fsdp_prefetch_distance',
    'fsdp_reshard_after_forward', 'fsdp_double_buffer', 'fsdp_bucket_elements', 'bucket_elements',
    'optimizer_pin_cpu_params', 'memory_policy')
COMPUTE_FIELDS = ('model', 'checkpoint', 'tp', 'cp', 'ep', 'etp', 'micro_batch', 'memory_policy',
    'sequence', 'precision', 'pipeline_dtype', 'recompute', 'attention_backend',
    'sequence_parallel', 'backend_fingerprint', 'hardware_fingerprint',
    'execution_backend', 'dense_sharding_strategy', 'expert_sharding_strategy',
    'precision_aware_optimizer', 'optimizer_sharding', 'gradient_path',
    'mlp_recompute_every', 'cuda_device_max_connections')
STATE_DTYPE_FIELDS = ('precision', 'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype',
    'exp_avg_sq_dtype', 'megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype')


@dataclass(frozen=True)
class PartitionState:
    end: int
    counts: tuple
    boundary: tuple
    resource_signature: tuple
    stage_forward_ms: tuple | None
    stage_backward_ms: tuple | None
    stage_state_bytes: tuple | None

    def metrics(self):
        if self.stage_forward_ms is None or self.stage_state_bytes is None:
            return None
        return self.stage_forward_ms + self.stage_backward_ms + self.stage_state_bytes


def strictly_dominates(left, right):
    if left.boundary != right.boundary or left.resource_signature != right.resource_signature:
        return False
    first, second = left.metrics(), right.metrics()
    return (first is not None and second is not None
        and len(first) == len(second)
        and all(a <= b for a, b in zip(first, second))
        and any(a < b for a, b in zip(first, second)))


class ConditionalPartitionSolver:
    def __init__(self, rewriter, profiles=(), memory_evaluator=None, backend_source_hashes=None, *,
                 fair_expansion=True):
        self.rewriter = rewriter
        self.profiles = list(profiles)
        self.memory_evaluator = memory_evaluator
        self.backend_source_hashes = backend_source_hashes
        if type(fair_expansion) is not bool:
            raise ValueError('状态扩展配额开关必须为布尔值')
        self.fair_expansion = fair_expansion
        self.profile_cache = {}

    def _environment_matches(self, profile):
        topology = self.rewriter.topology
        if not topology.devices or not topology.software or not profile.get('hardware'):
            return False
        if self.backend_source_hashes is None or profile.get('source_hashes') != self.backend_source_hashes:
            return False
        actual = {str(device.get('uuid', device.get('gpu_uuid'))).removeprefix('GPU-').lower()
            for device in topology.devices}
        measured = {str(device.get('gpu_uuid')).removeprefix('GPU-').lower()
            for device in profile['hardware']}
        if 'none' in actual or 'none' in measured or actual != measured:
            return False
        names = {device.get('name', device.get('device_name')) for device in topology.devices}
        if len(names) != 1 or None in names:
            return False
        for device in profile['hardware']:
            if device.get('device_name', device.get('name')) not in names:
                return False
            if (device.get('cuda_version') != topology.software.get('cuda')
                    or device.get('torch_version') != topology.software.get('torch')):
                return False
        return True

    def _profile(self, request):
        target = normalized_execution(request)
        target['precision_aware_optimizer'] = bool(target['precision_aware_optimizer'])
        cache_key = fingerprint(dict(execution=plan_signature(request),
            compute={key: target.get(key) for key in COMPUTE_FIELDS},
            state_dtypes={key: request.get(key) for key in STATE_DTYPE_FIELDS}))
        if cache_key in self.profile_cache:
            return self.profile_cache[cache_key]
        def comparable(row):
            previous = normalized_execution(row['request'])
            previous['precision_aware_optimizer'] = bool(previous['precision_aware_optimizer'])
            return all(previous.get(key) == target.get(key) for key in COMPUTE_FIELDS
                if key not in ('backend_fingerprint', 'hardware_fingerprint'))
        matched = [row for row in self.profiles if self._environment_matches(row) and comparable(row)
            and backend_reuse_evidence(request, row)['status'] != 'MISMATCHED']
        matched = [row for row in matched if set(map(int, row['layers']))
            == set(range(self.rewriter.model.num_layers))]
        if not matched:
            self.profile_cache[cache_key] = None
            return None
        matched = [annotate_measurement_semantics(item) for item in matched]
        row = max(matched, key=lambda item: (
            same_training_plan(item['request'], request),
            *compute_profile_quality(item),
            item['request']['decoder_chunks'] == request['decoder_chunks'],
            sum(len(windows) for windows in item.get('measured_windows', {}).values()), item['id']))
        for cost in row['layers'].values():
            for field in ('forward_ms', 'backward_ms', 'state_bytes'):
                if field not in cost or not math.isfinite(cost[field]) or cost[field] < 0:
                    raise ValueError('条件化分区需要有效的逐层实测计算与状态成本')
        row = self._target_state_costs(row, request)
        self.profile_cache[cache_key] = row
        return row

    def _target_state_costs(self, profile, request):
        row = enrich_module_profile(profile)
        required = ['precision', 'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype']
        if request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
            required.extend(('megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype'))
        missing = [name for name in required if not isinstance(request.get(name), str)]
        if not row.get('module_parameters'):
            missing.append('module_parameters')
        if self.rewriter.model.num_experts and not row.get('module_expert_ownership_labeled'):
            missing.append('module_expert_ownership_labels')
        row['partition_state_bytes'] = None
        row['partition_state_evidence'] = dict(status='UNKNOWN' if missing else 'MODELED_FROM_PARAMETER_OWNERSHIP',
            missing=missing, source_profile_id=profile['id'], target_plan_signature=plan_signature(request),
            target_state_dtypes={key: request.get(key) for key in STATE_DTYPE_FIELDS},
            scope='module_parameter_gradient_optimizer_state_excludes_bucket_padding_buffers_boundaries_and_activations')
        if missing:
            return row
        config = configuration(request)
        state_bytes = {}
        for layer, elements in row['module_parameters'].items():
            counts = {family: {rank: elements[family] for rank in range(config.world_size)}
                for family in ('dense', 'expert')}
            states = parameter_state_ownership(counts, config, request)
            # 模块可放入任一物理阶段，保留所有Rank中的最大参数状态量。
            state_bytes[layer] = max(item['gpu_persistent_bytes'] for item in states.values())
        row['partition_state_bytes'] = state_bytes
        return row

    def _layouts(self, parent, global_values, fixed, allowed_ranges):
        base = dict(parent, **global_values)
        base.update({name: value for name, value in fixed.items() if name not in GLOBAL_FIELDS})
        model, domains = self.rewriter.model, self.rewriter.domains
        states = SeedPlanner.state_families(base, base['dp'] * base['cp'],
            base['expert_dp'], model.num_experts is not None)
        precision_aware = base.get('precision_aware_optimizer', False)
        states = [state for state in states
            if state.get('precision_aware_optimizer', False) == precision_aware]
        state_fixed = {key: value for key, value in fixed.items()
            if key in STATE_FIELDS or key == 'fsdp_sharding_strategy'}
        states = [state for state in states if all(
            state.get(key, base.get(key)) == value for key, value in state_fixed.items())]
        layouts, rejections = [], []
        auxiliary = sorted(set(REWRITE_FIELDS) - set(GLOBAL_FIELDS) - set(LAYOUT_FIELDS)
            - set(STATE_FIELDS) - {'fsdp_sharding_strategy'})
        extra_fields = [key for key in auxiliary if key in fixed or key in allowed_ranges]
        extra_variants = [dict(zip(extra_fields, values)) for values in product(*[
            [fixed[key]] if key in fixed else allowed_ranges[key] for key in extra_fields])]
        axes = []
        for name, default in (('vpp', [base['vpp']]),
                ('micro_batch', [base['micro_batch']]),
                ('recompute', [base['recompute'], 'none', 'full', 'selective-mlp'])):
            values = list(domains.get(name, default))
            if name == 'recompute' and 'selective' in allowed_ranges.get(name, []):
                if selective_modules(model) and 'selective' not in values:
                    values.append('selective')
            if name in allowed_ranges:
                values = [value for value in values if value in allowed_ranges[name]]
            if name in fixed:
                if fixed[name] not in values:
                    raise ValueError(f'固定变量{name}超出条件化分区的后端允许范围')
                values = [fixed[name]]
            axes.append(sorted(set(values), key=lambda value: (value != base.get(name), str(value))))
        for (vpp, micro_batch, recompute), state, extra in product(product(*axes), states, extra_variants):
            if base['global_batch'] % (base['dp'] * micro_batch):
                continue
            microbatches = base['global_batch'] // (base['dp'] * micro_batch)
            schedule = 'interleaved_1f1b' if vpp > 1 else '1f1b'
            for group in sorted(set(domains.get('group', [base['pp'] if vpp > 1 else 1]))):
                if not valid_microbatch_group(microbatches, base['pp'], vpp, group):
                    continue
                overlaps = domains.get('overlap_p2p', [False, True]) if vpp > 1 else [False]
                for overlap in overlaps:
                    request = dict(base, **state, vpp=vpp, micro_batch=micro_batch,
                        recompute=recompute, group=group, pipeline_schedule=schedule,
                        overlap_p2p=overlap)
                    request.update(extra)
                    if any(request.get(key) != value for key, value in fixed.items()
                            if key != 'decoder_chunks'):
                        continue
                    if any(request.get(key) not in values for key, values in allowed_ranges.items()
                            if key != 'decoder_chunks'):
                        continue
                    if request.get('memory_policy') is not None:
                        if recompute == 'selective' and not request['memory_policy'].get('recompute_modules'):
                            modules = selective_modules(model)
                            if not modules:
                                rejections.append(dict(values=dict(recompute=recompute, memory_policy=request['memory_policy']),
                                    reasons=['后端没有支持的选择性重计算模块']))
                                continue
                            request['memory_policy'] = dict(request['memory_policy'], recompute_modules=[next(iter(modules))])
                        request['memory_policy'] = dict(request['memory_policy'],
                            recompute_granularity='full' if recompute == 'full' else
                            'selective' if recompute in ('selective-mlp', 'selective') else None,
                            recompute_modules=(request['memory_policy']['recompute_modules']
                                if recompute == 'selective' else ['mlp'] if recompute == 'selective-mlp' else []))
                    candidate_config = configuration(request)
                    valid, errors = validate_candidate(candidate_config, model, self.rewriter.topology)
                    errors = list(errors) + state_sharding_errors(request, candidate_config)
                    valid = valid and not errors
                    if valid:
                        count_options = ([fixed['decoder_chunks']] if 'decoder_chunks' in fixed else
                            allowed_ranges.get('decoder_chunks', [SeedPlanner.balanced_partition(
                                model.num_layers, base['pp'] * vpp)]))
                        for counts in count_options:
                            if (len(counts) == base['pp'] * vpp and all(type(n) is int and n > 0
                                    for n in counts) and sum(counts) == model.num_layers):
                                layouts.append(dict(request, decoder_chunks=list(counts)))
                    else:
                        rejections.append(dict(values={key: request.get(key) for key in REWRITE_FIELDS},
                            reasons=errors))
        # 状态后端轮流进入预算，防止单个后端占满所有执行布局。
        buckets = {}
        for row in layouts:
            identity = fingerprint({name: row.get(name) for name in STATE_FIELDS})
            buckets.setdefault(identity, []).append(row)
        ordered = [row for rows in zip_longest(*buckets.values()) for row in rows if row is not None]
        backends = {}
        for row in ordered:
            backends.setdefault(row['execution_backend'], []).append(row)
        ordered = [row for rows in zip_longest(*backends.values()) for row in rows if row is not None]
        return list({plan_signature(row): row for row in ordered}.values()), rejections

    def _boundary(self, request, slot, end):
        model = self.rewriter.model
        kind = model.layer_types[end - 1].value if end else 'input'
        return (slot, end, slot % request['pp'], slot // request['pp'],
            request['tp'], request['cp'], request['ep'], request['etp'],
            request['sequence'], request['micro_batch'], model.hidden_size,
            normalized_execution(request)['pipeline_dtype'], kind)

    @staticmethod
    def _score(state):
        if state.stage_forward_ms is None:
            return (max(state.counts, default=0) - min(state.counts, default=0), state.counts)
        combined = tuple(a + b for a, b in zip(state.stage_forward_ms, state.stage_backward_ms))
        return (max(combined), state.stage_state_bytes is None,
            0 if state.stage_state_bytes is None else max(state.stage_state_bytes), state.counts)

    def _partition(self, request, profile, width, budget, fixed_counts=None, module_relations=(),
                   continuation=None, deadline=None):
        if (continuation is not None and (fixed_counts is not None or module_relations)
                and continuation.completion_filter is None):
            raise ValueError('完整续接使用开放模块分区，不能同时指定固定布局或模块归属限制')
        if profile is not None:
            state_evidence = profile.get('partition_state_evidence') or {}
            if (state_evidence.get('target_plan_signature') != plan_signature(request)
                    or state_evidence.get('target_state_dtypes') != {
                        key: request.get(key) for key in STATE_DTYPE_FIELDS}):
                profile = self._target_state_costs(profile, request)
        layers, pp, slots = self.rewriter.model.num_layers, request['pp'], request['pp'] * request['vpp']
        zeros = (0.0,) * pp if profile is not None else None
        start = PartitionState(0, (), self._boundary(request, 0, 0), (), zeros, zeros, zeros)
        states = [start]
        expansions, dominated, truncated = 0, 0, False
        relation_rejected = 0
        prefixes = None
        if profile is not None:
            prefixes = {}
            for field in ('forward_ms', 'backward_ms', 'state_bytes'):
                if field == 'state_bytes' and profile.get('partition_state_bytes') is None:
                    continue
                prefix = [0.0]
                for index in range(layers):
                    value = (profile['partition_state_bytes'][str(index)] if field == 'state_bytes'
                        else profile['layers'][str(index)][field])
                    prefix.append(prefix[-1] + value)
                prefixes[field] = prefix
        for slot in range(slots):
            if deadline is not None and time.perf_counter() >= deadline:
                truncated = True
                break
            if continuation is not None:
                continuation.begin_depth(slot + 1, slots)
            generated = {}
            remaining = slots - slot
            allowance = max(1, (budget - expansions) // remaining)
            stage_expansions = 0
            ordered_states = sorted(states, key=(lambda state: continuation.score(
                state, request, self._score)) if continuation is not None else self._score)
            for state_index, state in enumerate(ordered_states):
                guided_expansion = (self.fair_expansion and continuation is not None
                    and continuation.prefix_guidance)
                state_allowance = (max(1, (allowance - stage_expansions)
                    // (len(ordered_states) - state_index)) if guided_expansion else allowance)
                state_expansions = 0
                if fixed_counts is not None:
                    ends = [state.end + fixed_counts[slot]]
                elif slot == slots - 1:
                    ends = [layers]
                else:
                    target = state.end + (layers - state.end) / remaining
                    ends = sorted(range(state.end + 1, layers - remaining + 2),
                        key=lambda end: (abs(end - target), end))
                    if continuation is not None:
                        ends = continuation.order_ends(request, state.counts, ends)
                for end in ends:
                    if deadline is not None and time.perf_counter() >= deadline:
                        truncated = True
                        break
                    if stage_expansions >= allowance or state_expansions >= state_allowance:
                        truncated = True
                        break
                    stage_expansions += 1
                    state_expansions += 1
                    counts = state.counts + (end - state.end,)
                    if not prefix_support(counts, layers, pp, request['vpp'], module_relations):
                        relation_rejected += 1
                        continue
                    costs = [None, None, None]
                    if prefixes is not None:
                        for index, (field, previous) in enumerate(zip(prefixes, (
                                state.stage_forward_ms, state.stage_backward_ms, state.stage_state_bytes))):
                            costs[index] = list(previous)
                            costs[index][slot % pp] += prefixes[field][end] - prefixes[field][state.end]
                            costs[index] = tuple(costs[index])
                    boundary = self._boundary(request, slot + 1, end)
                    # 已分配区间决定未来VPP资源顺序及存活Buffer；只有相同签名允许严格支配。
                    resource = (counts, tuple(request.get('rank_mapping', range(self.rewriter.topology.num_gpus))),
                        tuple((key, str(request.get(key))) for key in STATE_FIELDS),
                        request['group'], request['recompute'], request['overlap_p2p'])
                    candidate = PartitionState(end, counts, boundary, resource, *costs)
                    bucket = generated.setdefault(boundary, [])
                    if any(existing == candidate for existing in bucket):
                        continue
                    if any(strictly_dominates(existing, candidate) for existing in bucket):
                        dominated += 1
                        continue
                    survivors = [existing for existing in bucket if not strictly_dominates(candidate, existing)]
                    dominated += len(bucket) - len(survivors)
                    generated[boundary] = survivors + [candidate]
                if stage_expansions >= allowance:
                    if state_index + 1 < len(ordered_states):
                        truncated = True
                    break
            expansions += stage_expansions
            if continuation is not None:
                continuation.prepare_competitions(list(generated.values()), request, width, self._score)
            states = []
            for bucket in generated.values():
                if deadline is not None and time.perf_counter() >= deadline:
                    truncated = True
                    break
                if len(bucket) > width:
                    truncated = True
                states.extend(continuation.retain(bucket, request, width, self._score)
                    if continuation is not None else sorted(bucket, key=self._score)[:width])
            if not states:
                break
        completed = {state.counts: state for state in states
                     if len(state.counts) == slots and state.end == layers}
        if continuation is not None:
            for row in continuation.completed(request):
                counts = tuple(row['decoder_chunks'])
                costs = [None, None, None]
                if prefixes is not None:
                    costs = [[0.0] * pp if index < len(prefixes) else None for index in range(3)]
                    end = 0
                    for slot, count in enumerate(counts):
                        for index, values in enumerate(prefixes.values()):
                            costs[index][slot % pp] += values[end + count] - values[end]
                        end += count
                    costs = [None if values is None else tuple(values) for values in costs]
                resource = (counts, tuple(request.get('rank_mapping', range(self.rewriter.topology.num_gpus))),
                    tuple((key, str(request.get(key))) for key in STATE_FIELDS),
                    request['group'], request['recompute'], request['overlap_p2p'])
                completed.setdefault(counts, PartitionState(layers, counts,
                    self._boundary(request, slots, layers), resource, *costs))
        return list(completed.values()), dict(
            expansions=expansions, strictly_dominated=dominated, truncated=truncated,
            relation_rejected=relation_rejected)

    @staticmethod
    def _memory_witness(report, world):
        memory = report.get('memory', report)
        peaks = memory.get('upper_per_rank_bytes') or memory.get('peak_per_rank_bytes', {})
        capacities = memory.get('memory_budget_per_rank_bytes', {})
        if not capacities and peaks and memory.get('memory_headroom_per_rank_bytes'):
            headroom = {int(rank): value for rank, value in memory['memory_headroom_per_rank_bytes'].items()}
            if set(headroom) == set(map(int, peaks)):
                capacities = {rank: value + headroom[int(rank)] for rank, value in peaks.items()}
        if set(map(int, peaks)) != set(range(world)) or set(map(int, capacities)) != set(range(world)):
            return False
        peaks = {int(rank): value for rank, value in peaks.items()}
        capacities = {int(rank): value for rank, value in capacities.items()}
        return memory.get('admission') in ('PASS', 'ACCEPT', 'FEASIBLE', 'WITHIN_ESTIMATED_CAPACITY') and all(
            math.isfinite(peaks[rank]) and peaks[rank] >= 0
            and math.isfinite(capacities[rank]) and 0 < capacities[rank]
            and peaks[rank] <= capacities[rank] for rank in peaks)

    def solve(self, parent_request, global_values, evidence_id, *, fixed_values=None,
              limit=8, max_states_per_boundary=16, max_expansions=20000,
              max_layouts=64, compile_graphs=False, allowed_ranges=None, preserve=(), module_relations=(),
              continuation_requests=(), continuation_evaluator=None, max_continuation_evaluations=0):
        for value in (limit, max_states_per_boundary, max_expansions, max_layouts):
            if type(value) is not int or value < 1:
                raise ValueError('条件化分区的候选和计算预算必须为正整数')
        if set(global_values) != set(GLOBAL_FIELDS):
            raise ValueError('条件化分区需要完整且仅包含H的全局并行变量')
        module_relations = tuple(module_relations)
        fixed = dict(fixed_values or {})
        allowed_ranges = dict(allowed_ranges or {})
        if set(allowed_ranges) - set(REWRITE_FIELDS) or set(preserve) - set(REWRITE_FIELDS):
            raise ValueError('条件化分区的允许范围或保留变量包含未知字段')
        if any(not isinstance(values, (list, tuple)) or not values for values in allowed_ranges.values()):
            raise ValueError('条件化分区的允许范围必须为非空列表')
        normalized_parent = normalized_execution(parent_request)
        for name in preserve:
            if name not in parent_request and normalized_parent.get(name) is None:
                raise ValueError(f'父计划没有可保留的明确变量{name}')
            value = parent_request.get(name, normalized_parent.get(name))
            if name in fixed and fixed[name] != value:
                raise ValueError(f'固定意图与保留变量{name}冲突')
            fixed[name] = value
        if module_relations and 'vpp' not in fixed:
            raise ValueError('模块归属关系需要固定VPP，以明确虚拟区间身份')
        if set(fixed) - set(REWRITE_FIELDS) or any(global_values[key] != fixed[key]
                for key in set(fixed) & set(global_values)):
            raise ValueError('固定意图包含未知变量或与固定全局并行结构冲突')
        fixed = {key: value for key, value in fixed.items() if key not in GLOBAL_FIELDS}
        precision_aware = parent_request.get('precision_aware_optimizer', False)
        if ('precision_aware_optimizer' in fixed
                and fixed['precision_aware_optimizer'] != precision_aware):
            raise ValueError('条件化分区必须保持父计划的Optimizer数值精度')
        if any(value != precision_aware for value in allowed_ranges.get('precision_aware_optimizer', [])):
            raise ValueError('Optimizer数值精度不能进入条件化分区的搜索范围')
        if any(value not in allowed_ranges[name] for name, value in
                dict(global_values, **fixed).items() if name in allowed_ranges):
            raise ValueError('固定全局结构或联合意图超出允许范围')
        for name in GLOBAL_FIELDS:
            domain = self.rewriter.domains.get(name)
            if domain is not None and global_values[name] not in domain:
                raise ValueError(f'固定全局变量{name}超出后端允许范围')
        world = self.rewriter.topology.num_gpus
        proofs = []
        if any(type(global_values[key]) is not int or global_values[key] < 1
                for key in GLOBAL_FIELDS if key != 'rank_mapping'):
            raise ValueError('全局并行度必须为正整数')
        if sorted(global_values['rank_mapping']) != list(range(world)):
            raise ValueError('Rank Mapping必须完整覆盖实际GPU且没有重复')
        if (math.prod(global_values[name] for name in ('tp', 'pp', 'cp', 'dp')) != world
                or math.prod(global_values[name] for name in ('etp', 'ep', 'expert_dp', 'pp')) != world
                or parent_request['global_batch'] % global_values['dp']):
            proofs.append(dict(kind='algebraic_constraint', scope='fixed_global_family',
                reason='固定H不满足GPU数量、专家进程组或Global Batch整除约束'))
        result = dict(global_values=global_values, candidates=[], status='UNKNOWN', search_complete=False,
            termination=[], counters=dict(layouts=0, expansions=0, strictly_dominated=0,
                memory_rejected=0, relation_rejected=0),
            missing_cost_requests=[], capacity_rejections=[], proofs=proofs, performance_lower_bound_ms=None,
            lower_bound_ms=None, lower_bound_evidence=None,
            lower_bound_status='NO_PROVEN_BOUND_FOR_COMPLETE_DAG_PREDICTION',
            optimality='not_claimed', fixed_values=fixed, allowed_ranges=allowed_ranges,
            preserved_fields=sorted(set(preserve)),
            module_relations=[row.model_dump(mode='json') if hasattr(row, 'model_dump') else row
                              for row in module_relations],
            dominance_scope='identical_boundary_and_prefix_resource_lifetime_signature',
            memory_feasibility_scope='complete_plan_prediction_requires_real_training_validation')
        if proofs:
            return dict(result, status='INFEASIBLE', search_complete=True,
                termination=['PROVEN_GLOBAL_CONSTRAINT_CONFLICT'])
        all_layouts, rejected = self._layouts(parent_request, global_values, fixed, allowed_ranges)
        result['layout_rejections'] = rejected
        continuation = None
        if continuation_requests:
            if type(max_continuation_evaluations) is not int:
                raise ValueError('完整续接评价预算必须为整数')
            max_continuation_evaluations = max(0, max_continuation_evaluations)
            for request in continuation_requests:
                if any(request.get(key) != value for key, value in global_values.items()):
                    raise ValueError('完整续接方案必须属于当前全局并行组合')
                self.rewriter.candidate(request)
                if any(request.get(name, normalized_execution(request).get(name)) != value
                        for name, value in fixed.items()):
                    continue
                if any(request.get(name, normalized_execution(request).get(name)) not in values
                        for name, values in allowed_ranges.items()):
                    continue
                invalid = [name for name, values in self.rewriter.domains.items()
                    if name in REWRITE_FIELDS and name in request and request[name] not in values]
                if invalid:
                    raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=invalid))
                all_layouts.append(dict(request))
            if not fixed.get('decoder_chunks') and 'decoder_chunks' not in allowed_ranges:
                all_layouts = list({signature(request): request for request in all_layouts}.values())
            else:
                all_layouts = list({plan_signature(request): request for request in all_layouts}.values())
            continuation = ContinuationGuide(continuation_requests,
                continuation_evaluator, max_continuation_evaluations,
                admission_evaluator=self.memory_evaluator)
            all_layouts.sort(key=lambda request: not bool(continuation.compatible(request)))
        if not all_layouts:
            result['proofs'].append(dict(kind='exhaustive_backend_domain_validation',
                scope='fixed_global_family_and_fixed_values',
                reason='声明的L/Z域内全部布局违反代数或后端结构约束'))
            return dict(result, status='UNKNOWN' if fixed or allowed_ranges else 'INFEASIBLE',
                conditional_status='INFEASIBLE', search_complete=True,
                termination=['PROVEN_BACKEND_DOMAIN_CONFLICT'])
        layout_profiles = {plan_signature(request): self._profile(request)
            for request in all_layouts}
        if all(profile is not None for profile in layout_profiles.values()):
            bounds = [max(cost['forward_ms'] + cost['backward_ms']
                for cost in profile['layers'].values())
                for profile in layout_profiles.values()]
            result['layer_load_reference_ms'] = min(bounds)
            result['layer_load_reference_evidence'] = dict(
                kind='measured_single_layer_load_reference',
                scope='enumerated_layout_layer_cost_surrogate',
                profile_ids=sorted({profile['id'] for profile in layout_profiles.values()}),
                value_ms=result['layer_load_reference_ms'], admissible_replay_lower_bound=False)
        from search.policy_frontier import LAYOUT_ID_FIELDS
        structures = {}
        for row in all_layouts:
            identity = fingerprint({name: row.get(name) for name in LAYOUT_ID_FIELDS})
            structures.setdefault(identity, []).append(row)
        selected_structures = list(structures)[:max_layouts]
        layouts = [row for group in zip_longest(*(structures[key] for key in selected_structures))
            for row in group if row is not None]
        result['structural_layout_count'] = len(structures)
        result['selected_structural_layout_count'] = len(selected_structures)
        if len(structures) > len(selected_structures):
            result['termination'].append('LAYOUT_BUDGET')
        buckets = []
        missing = {}
        for index, request in enumerate(layouts):
            slots = request['pp'] * request['vpp']
            relations = validate_relations(module_relations, self.rewriter.model.num_layers, slots)
            available = max_expansions - result['counters']['expansions']
            if available < slots:
                result['termination'].append('EXPANSION_BUDGET')
                break
            budget = max(slots, available // (len(layouts) - index))
            profile = layout_profiles[plan_signature(request)]
            if profile is None:
                shape = {key: normalized_execution(request).get(key) for key in COMPUTE_FIELDS}
                identity = dict(kind='representative_layer_compute', identity=shape)
                cost_key = fingerprint(identity)
                missing[cost_key] = dict(**identity, cost_key=cost_key,
                    status='NEEDS_PROFILE')
            patterns, patterns_complete = owner_patterns(relations, self.rewriter.model.num_layers,
                request['pp'], request['vpp'], max_states_per_boundary)
            if not patterns_complete:
                result['termination'].append('MODULE_PATTERN_BUDGET')
            pattern_buckets = []
            statistics = dict(expansions=0, strictly_dominated=0, relation_rejected=0, truncated=False)
            for pattern_index, pattern in enumerate(patterns):
                remaining_budget = budget - statistics['expansions']
                if remaining_budget < slots:
                    statistics['truncated'] = True
                    break
                conditioned = list(relations)
                evidence_refs = sorted({reference for row in relations for reference in row.evidence_refs})
                conditioned.extend(ModuleRelation(relation='allowed_virtual', module_ids=[module],
                    virtual_slots=[owner], evidence_refs=evidence_refs,
                    rationale='原始模块关系满足的联合归属组合') for module, owner in pattern)
                generated, current = self._partition(request, profile, max_states_per_boundary,
                    max(slots, remaining_budget // (len(patterns) - pattern_index)),
                    request['decoder_chunks'] if 'decoder_chunks' in allowed_ranges
                    else fixed.get('decoder_chunks'), tuple(conditioned), continuation)
                pattern_buckets.append(sorted(generated, key=(lambda state: continuation.score(
                    state, request, self._score)) if continuation is not None else self._score))
                for name in ('expansions', 'strictly_dominated', 'relation_rejected'):
                    statistics[name] += current[name]
                statistics['truncated'] |= current['truncated']
            states = [state for group in zip_longest(*pattern_buckets)
                      for state in group if state is not None]
            states = list({state.counts: state for state in states}.values())
            result['counters']['layouts'] += 1
            for name in ('expansions', 'strictly_dominated', 'relation_rejected'):
                result['counters'][name] += statistics[name]
            if statistics['truncated']:
                result['termination'].append('PARTITION_BUDGET')
            bucket = []
            for state in states:
                candidate_request = dict(request, decoder_chunks=list(state.counts))
                candidate_request['run_id'] = 'conditional-' + plan_signature(candidate_request)[:16]
                regions = RegionPartitioner(self.rewriter.model, configuration(candidate_request)).from_counts(state.counts)
                valid, errors = validate_candidate(configuration(candidate_request), self.rewriter.model,
                    self.rewriter.topology, regions)
                if not valid:
                    raise ValueError(dict(status='CONDITIONAL_PARTITION_INVALID', constraints=errors))
                values = {key: candidate_request[key] for key in REWRITE_FIELDS if key in candidate_request}
                intent = RewriteIntent(operation='JointTransform', values=values, adjustable=[], preserve=[],
                    parent_plan_id=parent_request['run_id'], evidence_refs=[evidence_id],
                    module_relations=list(relations),
                    hypothesis='固定全局并行结构，联合补全执行布局、连续模块分区与后端状态分片')
                evidence = dict(algorithm='conditional_multi_candidate_dynamic_programming',
                    module_relations=[row.model_dump(mode='json') for row in relations],
                    module_owner_patterns=[list(pattern) for pattern in patterns],
                    module_owner_patterns_complete=patterns_complete,
                    boundary_state=list(state.boundary), resource_signature=fingerprint(state.resource_signature),
                    stage_forward_ms=state.stage_forward_ms, stage_backward_ms=state.stage_backward_ms,
                    stage_state_bytes=state.stage_state_bytes, compute_profile_id=profile['id'] if profile else None,
                    parameter_state_evidence=None if profile is None else profile.get('partition_state_evidence'),
                    cost_status='MEASURED_LAYER_SURROGATE' if profile else 'NEEDS_PROFILE',
                    score_scope='layer_load_and_target_parameter_state_surrogate_requires_complete_dag_replay',
                    memory_status='UNKNOWN', execution_validation='native_backend_structural_legality')
                if continuation is not None:
                    evidence['continuation_witness'] = continuation.witness(request, state.counts)
                groups = process_groups(candidate_request)
                evidence['communication_boundary'] = dict(
                    pipeline_payload_shape=[candidate_request['sequence'] // candidate_request['cp'],
                        candidate_request['micro_batch'], self.rewriter.model.hidden_size],
                    pipeline_dtype=normalized_execution(candidate_request)['pipeline_dtype'],
                    rank_mapping=groups['rank_mapping'], logical_groups=groups['logical'],
                    cost_status='REQUIRES_COMPLETE_DAG_COST_QUERY')
                evidence['region_implementations'] = [dict(region_id=region.region_id,
                    layer_start=region.layer_start, layer_end=region.layer_end,
                    pp_rank=region.pp_rank, vp_rank=region.vp_rank,
                    module_kinds=[self.rewriter.model.layer_types[index].value
                        for index in range(region.layer_start, region.layer_end)],
                    forward_ms=None if profile is None else sum(profile['layers'][str(index)]['forward_ms']
                        for index in range(region.layer_start, region.layer_end)),
                    backward_ms=None if profile is None else sum(profile['layers'][str(index)]['backward_ms']
                        for index in range(region.layer_start, region.layer_end)),
                    state_bytes=None if profile is None or profile.get('partition_state_bytes') is None
                        else sum(profile['partition_state_bytes'][str(index)]
                        for index in range(region.layer_start, region.layer_end)),
                    activation_bytes=None, activation_status='REQUIRES_SCHEDULE_LIFETIME_EVALUATION',
                    boundary_state=list(self._boundary(candidate_request, region.region_id + 1, region.layer_end)))
                    for region in regions]
                bucket.append(dict(request=candidate_request, intent=intent.model_dump(mode='json'),
                    partition_evidence=evidence))
            buckets.append(bucket)
        all_candidates = [row for rows in zip_longest(*buckets) for row in rows if row is not None]
        if len(all_candidates) > limit:
            result['termination'].append('OUTPUT_BUDGET')
        for row in all_candidates:
            if len(result['candidates']) >= limit:
                break
            if self.memory_evaluator is not None:
                report = self.memory_evaluator(row['request'])
                row['partition_evidence']['memory_evaluation'] = report
                if self._memory_witness(report, world):
                    row['partition_evidence']['memory_status'] = 'FEASIBLE'
                    result['status'] = 'FEASIBLE'
                elif report.get('memory', report).get('admission') == 'REJECT_OOM':
                    row['partition_evidence']['memory_status'] = 'REJECTED_COMPLETE_PLAN'
                    result['counters']['memory_rejected'] += 1
                    memory = report.get('memory', report)
                    if len(result['capacity_rejections']) < 4:
                        result['capacity_rejections'].append(dict(
                            source='PREDICTED_MEMORY_BOUND', decoder_chunks=row['request']['decoder_chunks'],
                            execution_backend=row['request']['execution_backend'],
                            capacity_per_gpu_bytes=int(self.rewriter.topology.memory_per_gpu_gb * 2**30),
                            peak_bytes=max(memory.get('peak_per_rank_bytes', {}).values(), default=None),
                            persistent_bytes=max(memory.get('persistent_per_rank_bytes', {}).values(), default=None),
                            rejection_basis=memory.get('rejection_basis'),
                            rejected_ranks=memory.get('rejected_ranks', [])[:4]))
                    continue
            if compile_graphs and plan_signature(row['request']) != plan_signature(parent_request):
                compiled = self.rewriter.complete_and_rewrite(parent_request, row['intent'], limit=1)[0]
                row.update(compiled)
            result['candidates'].append(row)
        if continuation is not None:
            for row in continuation.cache.values():
                for item in row['profiling_requests']:
                    missing.setdefault(item.get('cost_key', fingerprint(item)), item)
        result['missing_cost_requests'] = list(missing.values())
        result['termination'] = sorted(set(result['termination']))
        result['search_complete'] = not result['termination']
        result['termination'] = result['termination'] or ['EXHAUSTED_DECLARED_DOMAINS']
        result['candidate_count_before_output_budget'] = len(all_candidates)
        result['legal_layout_count'] = len(all_layouts)
        if continuation is not None:
            result['continuation_search'] = continuation.report()
        return result
