from itertools import permutations, zip_longest
from dataclasses import asdict
import random

from contracts import RewriteIntent, fingerprint
from predictor.profile_database import plan_signature
from region_partition.region_partitioner import RegionPartitioner
from cost.runtime_calibration import configuration
from validation.parallel_validator import validate_candidate
from search.communication_placement import AXES, axis_placement, shared_block_rank_mappings
from scheduling.native_microbatches import valid_microbatch_group
from predictor.state_sharding import state_sharding_errors
from graph.native_recomputation import selective_modules


class SeedPlanner:
    def __init__(self, rewriter, profile=None):
        self.rewriter = rewriter
        self.profile = profile
        self.rejections = []

    def random_initial_plan(self, request, seed):
        if type(seed) is not int:
            raise ValueError('初始配置随机数种子必须为整数')
        families = self.global_families(request)
        candidates = []
        seen = set()
        for family in families:
            candidate = dict(family['representative_request'])
            identity = plan_signature(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate)
        if not candidates:
            raise ValueError('完整合法配置域中没有可用的初始并行配置')
        index = random.Random(seed).randrange(len(candidates))
        candidate = candidates[index]
        candidate['run_id'] = 'initial-random-' + plan_signature(candidate)[:16]
        self.random_initial_selection = dict(random_seed=seed, selected_index=index,
            legal_candidate_count=len(candidates),
            selection_policy='uniform_over_legal_global_parallel_configurations',
            prediction_used=False, measurement_used=False)
        return candidate

    @staticmethod
    def family_fields():
        return ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping',
            'execution_backend', 'megatron_fsdp_version', 'optimizer_sharding',
            'megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype',
            'precision_aware_optimizer', 'dense_sharding_strategy',
            'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size',
            'expert_dp_inner_size', 'expert_dp_outer_size', 'outer_sharding_strategy',
            'expert_outer_sharding_strategy', 'offload', 'optimizer_offload_fraction',
            'optimizer_transfer_overlap')

    @staticmethod
    def balanced_partition(layers, regions):
        return [layers // regions + (index < layers % regions) for index in range(regions)]

    def rank_mappings(self, request):
        world = self.rewriter.topology.num_gpus
        per_node = self.rewriter.topology.gpus_per_node
        mappings = [list(request.get('rank_mapping', range(world))), list(range(world))]
        if self.rewriter.topology.num_nodes > 1:
            mappings.append([node * per_node + device for device in range(per_node)
                for node in range(self.rewriter.topology.num_nodes)])
            mappings.append([node * per_node + device
                for node in reversed(range(self.rewriter.topology.num_nodes))
                for device in range(per_node)])
        else:
            mappings.append(list(reversed(range(world))))
        mappings.extend(axis_placement(request, order) for order in permutations(AXES))
        block_rows, complete = shared_block_rank_mappings(request,
            request.get('candidate_rank_mapping_limit', 32))
        mappings.extend(row['rank_mapping'] for row in block_rows)
        self.mapping_coverage = dict(shared_block_complete=complete,
            shared_block_candidates=len(block_rows),
            unrestricted_native_permutation_domain=True)
        unique = []
        for mapping in mappings:
            if mapping not in unique:
                unique.append(mapping)
        return unique

    @staticmethod
    def state_families(request, dense_dp, expert_dp, is_moe):
        fsdp_version = 1
        fsdp_registered = bool(request.get('megatron_fsdp_adapter_path')
            and request.get('megatron_fsdp_adapter_sha256'))
        allowed_backends = tuple(request.get(
            'allowed_execution_backends', ('megatron_ddp', 'megatron_fsdp')
                if fsdp_registered else ('megatron_ddp',)))
        if 'megatron_fsdp' in allowed_backends and not fsdp_registered:
            raise ValueError('Megatron-FSDP搜索需要登记实际适配器路径和SHA256')
        allowed_optimizer_sharding = tuple(request.get(
            'allowed_optimizer_sharding', ('distributed', 'replicated')))
        allowed_offload_fractions = tuple(request.get('allowed_optimizer_offload_fractions', (0.0,)))
        if not allowed_backends or not set(allowed_backends) <= {'megatron_ddp', 'megatron_fsdp'}:
            raise ValueError('允许的执行后端范围无效')
        if (not allowed_optimizer_sharding
                or not set(allowed_optimizer_sharding) <= {'distributed', 'replicated'}):
            raise ValueError('允许的Optimizer分片范围无效')
        if (not allowed_offload_fractions or any(isinstance(value, bool)
                or not isinstance(value, (int, float)) or not 0 <= value <= 1
                for value in allowed_offload_fractions)):
            raise ValueError('允许的CPU Optimizer Offload比例无效')
        common = dict(fsdp_sharding_strategy='no_shard', dense_sharding_strategy='no_shard',
            expert_sharding_strategy='no_shard', dp_inner_size=dense_dp, dp_outer_size=1,
            expert_dp_inner_size=expert_dp, expert_dp_outer_size=1,
            outer_sharding_strategy='no_shard', expert_outer_sharding_strategy='no_shard',
            precision_aware_optimizer=request.get('precision_aware_optimizer', False),
            fsdp_prefetch=False, fsdp_reshard_after_forward=True,
            fsdp_double_buffer=False, checkpoint_format='torch_dist', megatron_fsdp_version=0,
            optimizer_checkpoint_mode='disabled', offload=False,
            optimizer_offload_fraction=0.0, optimizer_transfer_overlap=False,
            megatron_fsdp_main_params_dtype=None, megatron_fsdp_main_grads_dtype=None)
        variants = [dict(common, execution_backend='megatron_ddp', optimizer_sharding=mode,
            precision_aware_optimizer=(bool(request.get('precision_aware_optimizer', False))
                if mode == 'distributed' else False))
            for mode in ('distributed', 'replicated')
            if 'megatron_ddp' in allowed_backends and mode in allowed_optimizer_sharding]
        if ('megatron_ddp' in allowed_backends and 'distributed' in allowed_optimizer_sharding
                and (request.get('precision_aware_optimizer', False)
                    or True in request.get('allowed_precision_aware_optimizer', []))):
            for fraction in allowed_offload_fractions:
                if fraction > 0:
                    variants.append(dict(common, execution_backend='megatron_ddp',
                        optimizer_sharding='distributed', offload=True,
                        precision_aware_optimizer=True,
                        optimizer_offload_fraction=float(fraction), optimizer_transfer_overlap=True))
        strategies = tuple(request.get('fsdp_supported_sharding_strategies',
            ('optim_grads_params',)))
        expert_strategies = tuple(request.get('fsdp_supported_expert_sharding_strategies', ('no_shard',)))
        if not is_moe and expert_strategies:
            expert_strategies = ('no_shard',) if 'no_shard' in expert_strategies else expert_strategies[:1]
        precision_values = (bool(request.get('precision_aware_optimizer', False)),)
        for dense in strategies if 'megatron_fsdp' in allowed_backends else ():
            for expert in expert_strategies:
                for precision_aware in precision_values:
                    variants.append(dict(common, execution_backend='megatron_fsdp',
                        megatron_fsdp_version=fsdp_version, optimizer_sharding='distributed',
                        precision_aware_optimizer=precision_aware,
                        fsdp_sharding_strategy=dense, dense_sharding_strategy=dense,
                        expert_sharding_strategy=expert,
                        checkpoint_format=request.get('checkpoint_format', 'torch_dist')))
        if request.get('fsdp_hybrid_outer_supported', False):
            for inner in range(2, dense_dp):
                if dense_dp % inner:
                    continue
                outer_size = dense_dp // inner
                if expert_dp % outer_size:
                    continue
                for outer_strategy in ('no_shard', 'optim'):
                    variants.append(dict(common, execution_backend='megatron_fsdp',
                        megatron_fsdp_version=fsdp_version, optimizer_sharding='distributed',
                        fsdp_sharding_strategy='optim_grads_params',
                        dense_sharding_strategy='optim_grads_params',
                        expert_sharding_strategy='optim_grads_params' if is_moe else 'no_shard',
                        dp_inner_size=inner, dp_outer_size=outer_size,
                        expert_dp_inner_size=expert_dp // outer_size,
                        expert_dp_outer_size=outer_size,
                        outer_sharding_strategy=outer_strategy, checkpoint_format='fsdp_dtensor'))
            if is_moe and request.get('fsdp_separate_expert_outer_mesh_supported', False):
                for inner in range(2, expert_dp):
                    if expert_dp % inner:
                        continue
                    for outer_strategy in ('no_shard', 'optim'):
                        variants.append(dict(common, execution_backend='megatron_fsdp',
                            megatron_fsdp_version=fsdp_version, optimizer_sharding='distributed',
                            fsdp_sharding_strategy='optim_grads_params',
                            dense_sharding_strategy='optim_grads_params',
                            expert_sharding_strategy='optim_grads_params',
                            expert_dp_inner_size=inner, expert_dp_outer_size=expert_dp // inner,
                            expert_outer_sharding_strategy=outer_strategy,
                            checkpoint_format='fsdp_dtensor'))
        for variant in variants:
            if variant['execution_backend'] == 'megatron_fsdp':
                variant['megatron_fsdp_main_params_dtype'] = (
                    request.get('megatron_fsdp_main_params_dtype') or request.get('main_params_dtype', 'float32'))
                variant['megatron_fsdp_main_grads_dtype'] = (
                    request.get('megatron_fsdp_main_grads_dtype') or request.get('main_grads_dtype',
                    'bfloat16' if variant['precision_aware_optimizer'] else 'float32'))
        return variants

    def global_families(self, request):
        world = self.rewriter.topology.num_gpus
        model = self.rewriter.model
        expert_tensor_values = self.rewriter.domains.get('etp', self.rewriter.domains['tp'])
        expert_values = self.rewriter.domains.get('ep', [1]) if model.num_experts is not None else [1]
        families = []
        self.family_rejections = []
        for tp in self.rewriter.domains['tp']:
            for pp in self.rewriter.domains['pp']:
                for cp in self.rewriter.domains['cp']:
                    dense_denominator = tp * pp * cp
                    if world % dense_denominator:
                        continue
                    dp = world // dense_denominator
                    if request['global_batch'] % dp:
                        continue
                    for etp in expert_tensor_values:
                        for ep in expert_values:
                            if model.num_experts is None and (etp not in (1, tp) or ep != 1):
                                continue
                            expert_denominator = etp * ep * pp
                            if world % expert_denominator:
                                continue
                            expert_dp = world // expert_denominator
                            mappings = self.rank_mappings(dict(request, tp=tp, pp=pp, cp=cp,
                                dp=dp, ep=ep, etp=etp, expert_dp=expert_dp))
                            for mapping in mappings:
                                for state in self.state_families(request, dp * cp, expert_dp,
                                        model.num_experts is not None):
                                    if model.num_experts is None and etp != (
                                            1 if state['execution_backend'] == 'megatron_fsdp' else tp):
                                        continue
                                    candidate = dict(request, tp=tp, pp=pp, cp=cp, dp=dp,
                                        ep=ep, etp=etp, expert_dp=expert_dp, vpp=1,
                                        micro_batch=1, group=1, pipeline_schedule='1f1b',
                                        overlap_p2p=False, recompute='full',
                                        decoder_chunks=self.balanced_partition(model.num_layers, pp),
                                        rank_mapping=list(mapping), **state)
                                    candidate_config = configuration(candidate)
                                    valid, errors = validate_candidate(
                                        candidate_config, model, self.rewriter.topology)
                                    errors = list(errors) + state_sharding_errors(candidate, candidate_config)
                                    valid = valid and not errors
                                    if not valid:
                                        self.family_rejections.append(dict(request={key: candidate.get(key)
                                            for key in self.family_fields()}, reasons=errors))
                                        continue
                                    values = {name: candidate[name] for name in self.family_fields()}
                                    identity = 'family-' + fingerprint(values)[:16]
                                    parent_distance = sum(values[name] != request.get(name, request['tp'] if name == 'etp'
                                        else world // (request.get('etp', request['tp']) * request['ep'] * request['pp'])
                                        if name == 'expert_dp' else list(range(world)) if name == 'rank_mapping' else None)
                                        for name in self.family_fields())
                                    families.append(dict(family_id=identity, values=values,
                                        parent_distance=parent_distance, validation_errors=errors,
                                        representative_request=candidate))
        return sorted(families, key=lambda row: tuple(
            row['values'][name] if name != 'rank_mapping' else row['values'][name]
            for name in self.family_fields()))

    def select_global_families(self, request, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError('全局并行家族数量上限必须为正整数')
        families = self.global_families(request)
        parent = self.global_family_parent(request)
        remaining = [row for row in families if row['values'] != parent]
        selected = []
        global_fields = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'rank_mapping')
        state_fields = tuple(name for name in self.family_fields() if name not in global_fields)
        groups = {}
        for row in remaining:
            identity = fingerprint({name: row['values'][name] for name in global_fields})
            groups.setdefault(identity, []).append(row)
        while groups and len(selected) < limit:
            references = [row['values'] for row in selected] or [parent]

            def global_diversity(rows):
                values = rows[0]['values']
                return min(sum(values[name] != reference[name] for name in global_fields)
                    for reference in references)

            group_id = max(groups, key=lambda identity: (
                global_diversity(groups[identity]),
                max(row['parent_distance'] for row in groups[identity]), identity))
            rows = groups.pop(group_id)

            def state_diversity(row):
                return min(sum(row['values'][name] != reference[name] for name in state_fields)
                    for reference in references)

            chosen = max(rows, key=lambda row: (
                state_diversity(row), row['family_id']))
            selected.append(chosen)
        while remaining and len(selected) < limit:
            remaining = [row for row in remaining if row not in selected]
            if not remaining:
                break
            references = [row['values'] for row in selected] or [parent]
            chosen = max(remaining, key=lambda row: (
                min(sum(row['values'][name] != reference[name] for name in self.family_fields())
                    for reference in references), row['parent_distance'], row['family_id']))
            selected.append(chosen)
        return selected

    def global_family_parent(self, request):
        world = self.rewriter.topology.num_gpus
        etp = request.get('etp', request['tp'])
        expert_dp = request.get('expert_dp', world // (etp * request['ep'] * request['pp']))
        backend = request.get('execution_backend') or 'megatron_ddp'
        strategy = request.get('fsdp_sharding_strategy') or 'no_shard'
        return (dict(tp=request['tp'], pp=request['pp'], cp=request['cp'], dp=request['dp'],
            ep=request['ep'], etp=etp, expert_dp=expert_dp,
            rank_mapping=request.get('rank_mapping', list(range(world))), execution_backend=backend,
            megatron_fsdp_version=1 if backend == 'megatron_fsdp' else 0,
            megatron_fsdp_main_params_dtype=request.get('megatron_fsdp_main_params_dtype'),
            megatron_fsdp_main_grads_dtype=request.get('megatron_fsdp_main_grads_dtype'),
            optimizer_sharding=request.get('optimizer_sharding') or 'distributed',
            precision_aware_optimizer=request.get('precision_aware_optimizer', False),
            dense_sharding_strategy=request.get('dense_sharding_strategy') or strategy,
            expert_sharding_strategy=request.get('expert_sharding_strategy') or strategy,
            dp_inner_size=request.get('dp_inner_size') or request['dp'] * request['cp'],
            dp_outer_size=request.get('dp_outer_size') or 1,
            expert_dp_inner_size=request.get('expert_dp_inner_size') or expert_dp,
            expert_dp_outer_size=request.get('expert_dp_outer_size') or 1,
            outer_sharding_strategy=request.get('outer_sharding_strategy') or 'no_shard',
            expert_outer_sharding_strategy=request.get('expert_outer_sharding_strategy') or 'no_shard')
        | dict(offload=bool(request.get('offload', False)),
            optimizer_offload_fraction=float(request.get('optimizer_offload_fraction', 0.0)),
            optimizer_transfer_overlap=bool(request.get('optimizer_transfer_overlap', False))))

    def representative_global_plans(self, request, family, evidence_id, limit=4):
        if type(limit) is not int or limit < 1:
            raise ValueError('家族代表计划数量上限必须为正整数')
        base = dict(family['representative_request'])
        plans = [base]
        pp = base['pp']
        microbatches = base['global_batch'] // (base['dp'] * base['micro_batch'])
        multi_chunk_legal = (base.get('execution_backend') != 'megatron_fsdp'
            or base.get('fsdp_multi_chunk_supported', False))
        if multi_chunk_legal and pp > 1 and valid_microbatch_group(microbatches, pp, 2, pp):
            for vpp in self.rewriter.domains.get('vpp', []):
                regions = pp * vpp
                if vpp <= 1 or regions > self.rewriter.model.num_layers:
                    continue
                uniform_counts = self.balanced_partition(self.rewriter.model.num_layers, regions)
                plans.append(dict(base, vpp=vpp, group=pp,
                    pipeline_schedule='interleaved_1f1b', overlap_p2p=True,
                    decoder_chunks=uniform_counts))
                if max(uniform_counts) > 1:
                    counts = list(uniform_counts)
                    source = counts.index(max(counts))
                    destination = (source + pp) % regions
                    counts[source] -= 1
                    counts[destination] += 1
                    plans.append(dict(base, vpp=vpp, group=pp,
                        pipeline_schedule='interleaved_1f1b', overlap_p2p=True,
                        decoder_chunks=counts))
        elif pp > 1 and max(base['decoder_chunks']) > 1:
            counts = list(base['decoder_chunks'])
            source = counts.index(max(counts))
            destination = (source + 1) % len(counts)
            counts[source] -= 1
            counts[destination] += 1
            plans.append(dict(base, decoder_chunks=counts))
        for micro_batch in self.rewriter.domains.get('micro_batch', []):
            if micro_batch == base['micro_batch'] or base['global_batch'] % (base['dp'] * micro_batch):
                continue
            microbatches = base['global_batch'] // (base['dp'] * micro_batch)
            group = min(base['group'], microbatches)
            if not valid_microbatch_group(microbatches, base['pp'], base['vpp'], group):
                continue
            plans.append(dict(base, micro_batch=micro_batch,
                group=group))
            break
        recompute = 'none' if base['recompute'] == 'full' else 'full'
        recompute_plan = dict(base, recompute=recompute)
        if recompute_plan.get('memory_policy') is not None:
            recompute_plan['memory_policy'] = dict(recompute_plan['memory_policy'],
                recompute_granularity='full' if recompute == 'full' else None,
                recompute_modules=[])
        plans.append(recompute_plan)
        output = []
        for plan in plans[:limit]:
            values = {name: plan[name] for name in self.family_fields()}
            values.update(vpp=plan['vpp'], micro_batch=plan['micro_batch'], group=plan['group'],
                decoder_chunks=plan['decoder_chunks'], recompute=plan['recompute'],
                overlap_p2p=plan['overlap_p2p'], pipeline_schedule=plan['pipeline_schedule'],
                optimizer_sharding=plan['optimizer_sharding'])
            intent = RewriteIntent(operation='JointTransform', values=values,
                adjustable=[], preserve=[], hypothesis=(
                    '从完整合法全局并行家族生成代表性执行计划，并使用该家族自己的执行图评价VPP、Region、通信与显存'),
                evidence_refs=[evidence_id], parent_plan_id=request['run_id'])
            rows = self.rewriter.complete_and_rewrite(request, intent, limit=1)
            if rows:
                rows[0]['intent'] = intent.model_dump(mode='json')
                rows[0]['global_family_id'] = family['family_id']
                output.append(rows[0])
        return output

    def generate(self, request, evidence_id, limit, *, dimensions=()):
        if type(limit) is not int or limit < 1:
            raise ValueError('确定性候选上限必须为正整数')
        parent = self.rewriter.candidate(request)
        partitioner = RegionPartitioner(self.rewriter.model, parent.parallel_config)
        if self.profile is not None:
            partitioner.set_layer_costs({int(index): row['forward_ms'] + row['backward_ms']
                                        for index, row in self.profile['layers'].items()})
        buckets = {}

        def add(family, values, hypothesis, adjustable=()):
            if dimensions and not set(values).intersection(dimensions):
                return
            automatic = {'dp', 'expert_dp', 'etp', 'ep', 'vpp', 'group', 'decoder_chunks'}
            automatic.update(adjustable)
            automatic -= set(values)
            intent = RewriteIntent(operation='JointTransform', values=values,
                adjustable=sorted(automatic), preserve=[], hypothesis=hypothesis,
                evidence_refs=[evidence_id], parent_plan_id=request['run_id'])
            buckets.setdefault(family, []).append(intent)

        for dimension in ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'vpp', 'micro_batch', 'group'):
            for value in self.rewriter.domains.get(dimension, []):
                default = request['tp'] if dimension == 'etp' else 1
                if value != request.get(dimension, default):
                    add(dimension, {dimension: value}, '从后端允许范围选择全局变量并补全相关进程组和Batch约束',
                        ['tp', 'pp', 'cp'] if dimension == 'dp' else [])
        partitions = [[region.num_layers for region in regions]
                      for regions in partitioner.generate_diverse_regions(max(4, limit))]
        for counts in partitions:
            if counts != request['decoder_chunks']:
                add('region', dict(decoder_chunks=counts), '根据模型层数生成完整多边界非均匀分区')
        for policy in ('none', 'full', 'selective-mlp'):
            if policy != request['recompute']:
                add('recompute', dict(recompute=policy), '探索原生重计算策略与连续Region执行关系')
                for counts in partitions[1:4]:
                    add('region_recompute', dict(recompute=policy, decoder_chunks=counts),
                        '联合改变多边界分区和原生重计算策略')
        for module in selective_modules(self.rewriter.model):
            memory = asdict(configuration(request).memory_policy)
            memory.update(recompute_granularity='selective', recompute_modules=[module])
            add('native_recompute', dict(recompute='selective', memory_policy=memory),
                '选择当前模型对应的原生重计算模块并比较通信与保存需求')
        for sharding in ('distributed', 'replicated'):
            if sharding != request.get('optimizer_sharding', 'distributed'):
                add('optimizer', dict(optimizer_sharding=sharding), '探索Optimizer状态分片与梯度同步语义')
        world = self.rewriter.topology.num_gpus
        for mapping in self.rank_mappings(request):
            if mapping != request.get('rank_mapping', list(range(world))):
                add('mapping', dict(rank_mapping=mapping), '改变逻辑进程组在实际Node和GPU上的归属')
                for tp in self.rewriter.domains.get('tp', [])[:3]:
                    add('mapping_tp', dict(rank_mapping=mapping, tp=tp), '联合探索TP进程组与物理设备映射')
        add('communication', dict(overlap_p2p=not request.get('overlap_p2p', False)),
            '选择后端提供的P2P通信重叠策略')
        if request.get('bucket_elements') is not None:
            for size in (max(1, request['bucket_elements'] // 2), request['bucket_elements'] * 2):
                add('communication', dict(bucket_elements=size), '改变原生梯度Bucket容量')
        for left, right in (('pp', 'vpp'), ('cp', 'micro_batch'), ('ep', 'etp')):
            first = self.rewriter.domains.get(left, [])
            second = self.rewriter.domains.get(right, [])
            for index, value in enumerate(first):
                if second:
                    add(left + '_' + right, {left: value, right: second[index % len(second)]},
                        '根据强制约束和共同执行资源生成联合变量候选')
        self.rejections = []
        output = []
        seen = {plan_signature(request)}
        identities = set()
        priority = ('region', 'region_recompute', 'pp_vpp', 'communication', 'mapping_tp',
            'mapping', 'tp', 'pp', 'cp', 'dp', 'vpp', 'micro_batch', 'group',
            'recompute', 'optimizer', 'cp_micro_batch', 'ep_etp')
        ordered = [buckets[name] for name in priority if name in buckets]
        ordered.extend(rows for name, rows in buckets.items() if name not in priority)
        # 逐类轮流展开，同时保证Region和联合动作进入固定候选预算。
        for intents in zip_longest(*ordered):
            for intent in intents:
                if intent is None:
                    continue
                identity = fingerprint(intent.model_dump(mode='json'))
                if identity in identities:
                    continue
                identities.add(identity)
                try:
                    rows = self.rewriter.complete_and_rewrite(request, intent, limit=min(3, limit - len(output)))
                except ValueError as error:
                    if isinstance(error.args[0], dict) and error.args[0].get('status') == 'CONSTRAINT_CONFLICT':
                        self.rejections.append(dict(intent=intent.model_dump(mode='json'), constraints=error.args[0]))
                        continue
                    raise
                for row in rows:
                    identity = plan_signature(row['request'])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    row['intent'] = intent.model_dump(mode='json')
                    output.append(row)
                    if len(output) == limit:
                        return output
        return output
