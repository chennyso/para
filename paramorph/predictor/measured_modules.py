from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics


DTYPE_BYTES = {'torch.bfloat16': 2, 'torch.float16': 2, 'torch.float32': 4}


def compute_profile_quality(document):
    semantics = document.get('measurement_semantics') or {}
    layer = semantics.get('layer') or {}
    paired = semantics.get('paired_step_comparison') or {}
    difference = paired.get('relative_difference')
    verified = layer.get('local_service_time_verified') is True
    inclusive_scope = layer.get('scope') == 'inclusive_layer_cuda_event'
    measured_scope = (inclusive_scope and layer.get('step_comparison_available') is True
        and isinstance(difference, (int, float)) and math.isfinite(difference))
    request = document.get('request')
    nested_instrumentation = None if request is None else request.get('paramorph_operator_calibration', False)
    if nested_instrumentation is not None and type(nested_instrumentation) is not bool:
        raise ValueError('算子采集设置必须明确为布尔值')
    quality = (2 if verified else 1 if inclusive_scope else 0, int(nested_instrumentation is False))
    perturbation = -abs(difference) if measured_scope else float('-inf')
    return quality, perturbation

OPTIMIZER_EXCLUDED_OPERATIONS = frozenset((
    'gradient_norm_collectives', 'overflow_collectives',
    'parameter_gather', 'upstream_stream_waits'))
OPTIMIZER_COMPONENT_SCOPES = {
    'optimizer_prepare_grads': 'local_optimizer_prepare_grads',
    'optimizer_local_grad_norm': 'local_optimizer_gradient_norm',
    'optimizer_local_grad_norm_finalize': 'local_optimizer_gradient_norm_finalize',
    'optimizer_local_grad_clip': 'local_optimizer_gradient_clip',
}


def annotate_measurement_semantics(profile):
    document = deepcopy(profile)
    previous = document.get('measurement_semantics')
    if previous is not None:
        if previous.get('version') != 1:
            raise ValueError('成本测量语义版本无法识别')
        return document
    step_samples = defaultdict(list)
    for windows in document.get('measured_windows', {}).values():
        for window in windows:
            value = (window['host_end_ns'] - window['host_start_ns']) / 1e6
            if not math.isfinite(value) or value <= 0:
                raise ValueError('实测训练步骤的时间区间非法')
            step_samples[window['step']].append(value)
    samples = [max(step_samples[step]) for step in sorted(step_samples)]
    comparison = None
    anchor = document.get('anchor')
    if anchor is not None and anchor.get('step_seconds') is not None and samples:
        reference = anchor['step_seconds'] * 1000
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError('普通训练参照耗时必须为正数')
        comparison = dict(profile_step_ms=samples, reference_step_ms=reference,
            profile_median_step_ms=statistics.median(samples),
            relative_difference=statistics.median(samples) / reference - 1,
            all_profile_steps_slower_than_reference=min(samples) > reference,
            scope='observed_whole_step_difference_without_component_attribution')
    document['measurement_semantics'] = dict(version=1,
        layer=dict(scope='inclusive_layer_cuda_event', local_service_time_verified=False,
            perturbation_status='UNASSESSED',
            step_comparison_available=comparison is not None),
        region=dict(scope='inclusive_region_residual', local_service_time_verified=False,
            includes_special_modules_and_execution_overhead=True),
        optimizer=dict(scope='whole_optimizer_step_cuda_event',
            local_update_and_copy_verified=False, excluded_operations=[]),
        anchor=dict(scope='whole_step_only', available=anchor is not None,
            task_costs_rescaled=False), paired_step_comparison=comparison)
    return document


def _local_optimizer_observations(payload, request):
    steps = {row['step'] for row in payload['windows'] if row['step'] >= request['warmup']}
    by_step = defaultdict(dict)
    components = {}
    for event in payload['events']:
        kind = event['kind']
        if event['step'] not in steps or kind not in (
                'optimizer_local_update', 'optimizer_parameter_copy'):
            continue
        identity = event.get('identity', {})
        if (identity.get('cost_scope') != 'local_optimizer_update_and_copy'
                or not OPTIMIZER_EXCLUDED_OPERATIONS <= set(identity.get('excluded_operations', []))
                or identity.get('synchronization_policy') != 'cuda_device_synchronize_before_component'
                or not identity.get('callable_source_sha256') or not identity.get('callable_name')):
            return None, '原始Optimizer组件缺少独立调用与计时范围证据'
        index = identity['component_index']
        key = (index, kind)
        if key in by_step[event['step']]:
            raise ValueError('同一步骤重复记录相同Optimizer组件')
        value = event['cuda_ms']
        if not math.isfinite(value) or value < 0:
            raise ValueError('本地Optimizer组件实测耗时非法')
        by_step[event['step']][key] = value
        descriptor = {field: identity[field] for field in (
            'parameter_elements', 'callable_module', 'callable_name', 'callable_source_sha256')}
        if key in components and components[key] != descriptor:
            raise ValueError('Optimizer组件的实现或参数所有权在测量期间发生变化')
        components[key] = descriptor
    indices = {index for index, _ in components}
    expected = {(index, kind) for index in indices for kind in (
        'optimizer_local_update', 'optimizer_parameter_copy')}
    if not steps or not indices or any(set(by_step[step]) != expected for step in steps):
        return None, '稳定步骤没有完整覆盖本地更新和参数复制'
    for index in indices:
        if components[index, 'optimizer_local_update']['parameter_elements'] != components[
                index, 'optimizer_parameter_copy']['parameter_elements']:
            raise ValueError('本地更新与参数复制的参数所有权不一致')
    return dict(milliseconds=statistics.median(sum(by_step[step].values()) for step in steps),
        owned_parameter_elements=sum(components[index, 'optimizer_local_update']['parameter_elements']
            for index in indices),
        components=[dict(component_index=index, kind=kind, **components[index, kind])
            for index, kind in sorted(components)],
        measured_step_ms={str(step): sum(by_step[step].values()) for step in sorted(steps)}), None


def collect_optimizer_local_costs(payloads, request):
    if not request.get('paramorph_optimizer_calibration', False):
        return None
    world = request['node_count'] * request['gpus_per_node']
    if len(payloads) != world:
        raise ValueError('本地Optimizer采集没有覆盖全部Rank')
    observations = {}
    for rank, payload in enumerate(payloads):
        if payload['rank'] != rank:
            raise ValueError('Optimizer采集事件的Rank身份错误')
        observation, reason = _local_optimizer_observations(payload, request)
        if observation is None:
            raise ValueError(f'Rank {rank} 的Optimizer采集不完整：{reason}')
        observations[str(rank)] = observation
    return dict(scope='local_optimizer_update_and_copy',
        excluded_operations=sorted(OPTIMIZER_EXCLUDED_OPERATIONS),
        milliseconds_by_rank={rank: row['milliseconds'] for rank, row in observations.items()},
        owned_parameter_elements_by_rank={rank: row['owned_parameter_elements']
            for rank, row in observations.items()},
        rank_observations=observations)


def collect_optimizer_component_costs(payloads, request):
    if not request.get('paramorph_optimizer_calibration', False):
        return None
    operations = {operation: dict(milliseconds_by_rank={}, elements_by_rank={},
        rank_observations={}) for operation in OPTIMIZER_COMPONENT_SCOPES}
    for rank, payload in enumerate(payloads):
        if payload['rank'] != rank:
            raise ValueError('Optimizer组件采集事件的Rank身份错误')
        steps = {row['step'] for row in payload['windows'] if row['step'] >= request['warmup']}
        if not steps:
            raise ValueError('Optimizer组件采集没有稳定步骤')
        for operation, scope in OPTIMIZER_COMPONENT_SCOPES.items():
            events = [event for event in payload['events']
                if event['step'] in steps and event['kind'] == operation]
            if not events:
                continue
            by_step, elements, rows = defaultdict(list), defaultdict(dict), []
            for event in events:
                identity = event['identity']
                if (identity.get('cost_scope') != scope or not identity.get('callable_source_sha256')
                        or not OPTIMIZER_EXCLUDED_OPERATIONS <= set(identity.get('excluded_operations', []))):
                    raise ValueError('Optimizer组件的独立调用证据不完整')
                if operation.startswith('optimizer_local_grad_norm') and identity.get(
                        'synchronization_policy') != 'cuda_event_boundaries_around_collectives':
                    raise ValueError('本地Norm计时没有排除Collective区间')
                value = event['cuda_ms']
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Optimizer组件实测耗时非法')
                by_step[event['step']].append(value)
                elements[event['step']][identity['component_index']] = identity.get(
                    'norm_elements', identity.get('owned_elements'))
                rows.append(dict(step=event['step'], milliseconds=value, identity=identity))
            if set(by_step) != steps:
                raise ValueError('Optimizer组件没有覆盖全部稳定步骤')
            sizes = {sum(elements[step].values()) for step in steps}
            if len(sizes) != 1:
                raise ValueError('Optimizer组件参数数量在稳定步骤之间发生变化')
            item = operations[operation]
            item['milliseconds_by_rank'][str(rank)] = statistics.median(
                sum(by_step[step]) for step in sorted(steps))
            item['elements_by_rank'][str(rank)] = sizes.pop()
            item['rank_observations'][str(rank)] = rows
    return dict(scope='independent_optimizer_component_cuda_events', operations=operations,
        norm_collectives_by_rank={str(rank): [dict(step=event['step'],
            milliseconds=event['cuda_ms'], identity=event['identity']) for event in payload['events']
            if event['step'] >= request['warmup'] and event['kind'] == 'optimizer_gradient_norm_all_reduce']
            for rank, payload in enumerate(payloads)})


def optimizer_component_cost_evidence(profile, operation):
    if operation not in OPTIMIZER_COMPONENT_SCOPES:
        raise ValueError('Optimizer组件类型没有登记')
    rejected = dict(verified=False, evidence_ids=[], costs_by_rank={}, elements_by_rank={})
    record = profile.get('optimizer_component_costs')
    if not record or record.get('scope') != 'independent_optimizer_component_cuda_events':
        return dict(rejected, reason='缺少独立Optimizer组件实测记录')
    if not profile['request'].get('paramorph_optimizer_calibration', False):
        return dict(rejected, reason='原始请求没有启用Optimizer组件采集')
    source = profile.get('source_directory')
    if source is None:
        return dict(rejected, reason='Optimizer组件没有实际采集目录')
    world = profile['request']['node_count'] * profile['request']['gpus_per_node']
    payloads, evidence = [], []
    for rank in range(world):
        name = f'rank{rank}.paramorph-cost.json'
        path = Path(source) / 'artifacts' / name
        expected = profile.get('evidence_files', {}).get(name)
        if not expected or not path.is_file():
            return dict(rejected, reason=f'Rank {rank} 缺少原始Optimizer组件证据')
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError('Optimizer组件原始文件与登记的校验值不同')
        payloads.append(json.loads(encoded))
        evidence.append(f'{profile["id"]}:{name}:{expected}')
    observed = collect_optimizer_component_costs(payloads, profile['request'])
    if observed != record:
        raise ValueError('登记的Optimizer组件成本与原始事件不同')
    item = observed['operations'][operation]
    if len(item['milliseconds_by_rank']) != world:
        return dict(rejected, reason='目标Optimizer组件没有全部Rank的实测覆盖')
    return dict(verified=True, reason=None, evidence_ids=evidence,
        costs_by_rank=item['milliseconds_by_rank'], elements_by_rank=item['elements_by_rank'],
        rank_observations=item['rank_observations'],
        norm_collectives_by_rank=observed['norm_collectives_by_rank'])


def optimizer_local_cost_evidence(profile):
    record = profile.get('optimizer_local_costs')
    rejected = dict(verified=False, evidence_ids=[], costs_by_rank={})
    if not record:
        return dict(rejected, reason='仅有完整Optimizer调用区间，缺少独立本地更新及参数复制证据')
    if record.get('scope') != 'local_optimizer_update_and_copy':
        return dict(rejected, reason='Optimizer成本作用范围没有限定为本地更新及参数复制')
    if not OPTIMIZER_EXCLUDED_OPERATIONS <= set(record.get('excluded_operations', [])):
        return dict(rejected, reason='Optimizer成本没有明确排除同步通信和前序等待')
    source = profile.get('source_directory')
    if source is None:
        return dict(rejected, reason='Optimizer成本没有实际采集目录')
    costs, evidence = {}, []
    request = profile['request']
    world = request['node_count'] * request['gpus_per_node']
    for rank in range(world):
        name = f'rank{rank}.paramorph-cost.json'
        expected_hash = profile.get('evidence_files', {}).get(name)
        path = Path(source) / 'artifacts' / name
        if not expected_hash or not path.is_file():
            return dict(rejected, reason=f'Rank {rank} 缺少可检查的原始Optimizer事件')
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != expected_hash:
            raise ValueError('Optimizer原始采集事件与登记的证据校验值不一致')
        payload = json.loads(encoded)
        if payload['rank'] != rank:
            raise ValueError('Optimizer原始证据的Rank身份错误')
        observation, reason = _local_optimizer_observations(payload, request)
        if observation is None:
            return dict(rejected, reason=f'Rank {rank}：{reason}')
        costs[str(rank)] = observation['milliseconds']
        recorded = record.get('owned_parameter_elements_by_rank', {}).get(str(rank))
        if recorded != observation['owned_parameter_elements']:
            raise ValueError('登记的Optimizer参数所有权与原始事件不一致')
        evidence.append(f'{profile["id"]}:{name}:{expected_hash}')
    if record.get('milliseconds_by_rank') != costs:
        raise ValueError('登记的本地Optimizer成本与实际组件事件不一致')
    return dict(verified=True, reason=None, evidence_ids=evidence, costs_by_rank=costs,
        owned_parameter_elements_by_rank=record['owned_parameter_elements_by_rank'])


def enrich_module_profile(profile):
    document = annotate_measurement_semantics(profile)
    if 'module_parameters' in document:
        return document
    source = document.get('source_directory')
    if source is None:
        return document
    artifacts = Path(source) / 'artifacts'
    world = document['request']['node_count'] * document['request']['gpus_per_node']
    parameters = defaultdict(list)
    parameter_shapes = {}
    expert_ownership_labeled = True
    boundaries = defaultdict(list)
    pipeline_dtypes = set()
    for rank in range(world):
        path = artifacts / f'rank{rank}.paramorph-static.json'
        static = json.loads(path.read_text())
        manifest_path = artifacts / f'rank{rank}.manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            pipeline_dtypes.add(manifest['pipeline_dtype'].removeprefix('torch.'))
            decoder_count = sum(math.prod(parameter['shape'])
                for layer in static['layers'] for parameter in layer['parameters'])
            extra = manifest['parameter_elements'] - decoder_count
            if extra < 0:
                raise ValueError('实测模型参数数量小于Decoder模块参数数量')
            pp = document['request']['pp']
            stage = rank // (world // pp)
            category = 'both' if pp == 1 else 'first' if stage == 0 else 'last' if stage == pp - 1 else 'internal'
            boundaries[category].append(extra)
        for layer in static['layers']:
            key = str(layer['layer_id'])
            counts = {'dense': 0, 'expert': 0}
            for parameter in layer['parameters']:
                expert_ownership_labeled = expert_ownership_labeled and 'expert_parallel' in parameter
                count = math.prod(parameter['shape'])
                family = 'expert' if parameter.get('expert_parallel', False) else 'dense'
                counts[family] += count
            parameters[key].append(counts)
            parameter_shapes[key] = [
                {'shape': item['shape'], 'dtype': item['dtype']}
                for item in layer['parameters']]
    document['module_parameters'] = {}
    for layer, records in parameters.items():
        if any(record != records[0] for record in records):
            raise ValueError(f'同一模块的实测参数布局不一致：layer={layer}')
        document['module_parameters'][layer] = records[0]
        document['layers'][layer]['parameter_shapes'] = parameter_shapes[layer]
    document['module_parameter_source'] = 'actual_loaded_parameter_shapes'
    document['module_expert_ownership_labeled'] = expert_ownership_labeled
    document['boundary_parameter_counts'] = {
        category: int(statistics.median(values)) for category, values in boundaries.items()}
    if pipeline_dtypes:
        if len(pipeline_dtypes) != 1:
            raise ValueError('实际Pipeline张量精度在参与Rank之间不一致')
        document['pipeline_dtype'] = next(iter(pipeline_dtypes))
    return document


def boundary_costs(profile, region_id, config, phase):
    previous = profile['request']
    old_pp, old_vpp = previous['pp'], previous['vpp']
    old_world = previous['node_count'] * previous['gpus_per_node']
    old_lanes = old_world // old_pp
    old_total = old_pp * old_vpp
    total = config.pp * config.vpp
    operation = 'forward_compute' if phase == 'forward' else 'backward_compute'
    categories = defaultdict(list)
    for key, row in profile['regions'].items():
        rank, vp, kind = key.split(':')
        if kind != operation:
            continue
        index = int(vp) * old_pp + int(rank) // old_lanes
        category = 'both' if old_total == 1 else 'first' if index == 0 else 'last' if index == old_total - 1 else 'internal'
        categories[category].append(row['overhead_ms'])
    target = 'both' if total == 1 else 'first' if region_id == 0 else 'last' if region_id == total - 1 else 'internal'
    if target in categories:
        return statistics.median(categories[target]), target
    if target == 'both' and all(name in categories for name in ('first', 'last', 'internal')):
        # 首尾合并只计算一次实测公共调度开销。
        value = statistics.median(categories['first']) + statistics.median(categories['last'])
        value -= statistics.median(categories['internal'])
        if value < 0:
            raise ValueError('首尾模块合并成本与实测公共开销不一致')
        return value, 'first_plus_last_minus_internal'
    return None, target
