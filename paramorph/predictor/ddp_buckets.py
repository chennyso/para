from collections import OrderedDict, defaultdict
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path

import ijson

from contracts import fingerprint
from core_data_models import Task, TaskType
from predictor.profile_database import normalized_execution, plan_signature, same_training_plan
from scheduling.schedule_simulator import topological_order


WIDTHS = {'float32': 4, 'bfloat16': 2, 'float16': 2}


def read_projection_bucket_document(path, expected_digest):
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_digest:
            raise ValueError('DDP Bucket原始文件与登记的校验值不同')
        document = {}
        for field in ('schema_version', 'rank', 'request', 'source', 'event_capture_scope', 'groups'):
            stream.seek(0)
            document[field] = next(ijson.items(stream, field, use_float=True))
        active_groups = {group['group_id'] for group in document['groups']
            if len(group['physical_ranks']) > 1}
        stream.seek(0)
        indexed = [(index, event) for index, event in enumerate(ijson.items(stream, 'events.item', use_float=True))
            if event['group_id'] in active_groups and (
                event['operation'] == 'start_grad_sync' and not event['grad_handle_before']
                or event['operation'] == 'start_param_sync' and not event['param_gather_dispatched_before'])]
        starts = defaultdict(list)
        for _, event in indexed:
            if event['operation'] == 'start_param_sync':
                starts[event['group_id'], event['step']].append(event['host_start_ns'])
        for values in starts.values():
            values.sort()
        next_groups = {group['group_id']: group['next_param_gather_group_id']
            for group in document['groups']}
        stream.seek(0)
        for index, event in enumerate(ijson.items(stream, 'events.item', use_float=True)):
            if event['operation'] != 'finish_param_sync' or event['group_id'] not in active_groups:
                continue
            candidates = starts.get((next_groups[event['group_id']], event['step']), [])
            position = bisect_left(candidates, event['host_start_ns'])
            if position < len(candidates) and candidates[position] <= event['host_end_ns']:
                indexed.append((index, event))
        document['events'] = [event for _, event in sorted(indexed)]
        document['projection_event_selection'] = dict(
            source_sha256=expected_digest,
            scope='actual_collective_dispatch_and_native_prefetch_trigger_events_used_by_compiler')
    return document


def bucket_layout_signature(request):
    fields = (
        'model', 'checkpoint', 'tp', 'cp', 'pp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp',
        'decoder_chunks', 'rank_mapping', 'optimizer', 'optimizer_sharding', 'execution_backend',
        'precision', 'precision_aware_optimizer', 'main_grads_dtype', 'main_params_dtype',
        'exp_avg_dtype', 'exp_avg_sq_dtype', 'fsdp_bucket_elements',
        'overlap_grad_reduce', 'overlap_param_gather', 'align_param_gather',
    )
    resolved = normalized_execution(request)
    return fingerprint({field: resolved[field] if field in resolved else request.get(field)
                        for field in fields})


def load_bucket_evidence(projector, request):
    signature = bucket_layout_signature(request)
    actual_uuids = {str(device.get('uuid', device.get('gpu_uuid'))).removeprefix('GPU-').lower()
                    for device in projector.topology.devices}
    candidates = []
    for identifier, profile in projector.compute_surface.database.measured_profiles():
        if projector.blind and same_training_plan(profile['request'], request):
            continue
        measured_uuids = {str(device.get('gpu_uuid')).removeprefix('GPU-').lower()
                          for device in profile['hardware']}
        if measured_uuids != actual_uuids or bucket_layout_signature(profile['request']) != signature:
            continue
        source = projector.compute_surface.database.connection.execute(
            'SELECT directory FROM profile_sources WHERE profile_id=?',
            (identifier,)).fetchone()
        if source is not None:
            profile['source_directory'] = source[0]
        if not profile.get('source_directory'):
            continue
        candidates.append((identifier == projector.preferred_profile_id,
            profile.get('anchor') is not None,
            sum(len(windows) for windows in profile.get('measured_windows', {}).values()),
            identifier, profile))
    for _, _, _, _, profile in reversed(sorted(candidates, key=lambda row: row[:-1])):
        files = []
        for rank in range(projector.topology.num_gpus):
            name = f'rank{rank}.paramorph-buckets.json'
            expected = profile.get('evidence_files', {}).get(name)
            path = Path(profile['source_directory']) / 'artifacts' / name
            if not expected or not path.is_file():
                break
            metadata = path.stat()
            files.append((str(path), expected, metadata.st_size, metadata.st_mtime_ns))
        if len(files) != projector.topology.num_gpus:
            continue
        cache_key = (profile['id'], signature, tuple(sorted(actual_uuids)), tuple(files))
        database = projector.compute_surface.database
        if not hasattr(database, '_bucket_evidence_cache'):
            database._bucket_evidence_cache = OrderedDict()
        cache = database._bucket_evidence_cache
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]
        documents = []
        for rank in range(projector.topology.num_gpus):
            name = f'rank{rank}.paramorph-buckets.json'
            expected = profile.get('evidence_files', {}).get(name)
            path = Path(profile['source_directory']) / 'artifacts' / name
            if not expected or not path.is_file():
                break
            row = read_projection_bucket_document(path, expected)
            if (row['schema_version'] != 1 or row['rank'] != rank
                    or bucket_layout_signature(row['request']) != signature):
                raise ValueError('DDP Bucket的训练计划或Rank身份不一致')
            active_groups = {group['group_id'] for group in row['groups']
                             if len(group['physical_ranks']) > 1}
            row['events'] = [event for event in row['events'] if event['group_id'] in active_groups]
            documents.append(row)
        if len(documents) == projector.topology.num_gpus:
            result = dict(profile_id=profile['id'], documents=documents,
                evidence_scope='bucket_layout_compatible_structural_metadata_without_cost_timing_reuse')
            cache[cache_key] = result
            cache.move_to_end(cache_key)
            while len(cache) > 4:
                cache.popitem(last=False)
            return result
    return None


def split_measured_regions(tasks, costs):
    records = {row['cost_key']: row for row in costs if 'cost_key' in row}
    positions = {}
    for key, task in list(tasks.items()):
        if task.task_type not in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE):
            continue
        layers = task.metadata.get('layer_ids', [])
        if task.duration is None:
            continue
        components, residual = defaultdict(list), []
        for component in task.metadata['cost_components']:
            source = records[component['cost_key']]
            if source.get('projection_operation') == 'layer_composite' and source.get('logical_layer') in layers:
                components[source['logical_layer']].append(component)
            else:
                residual.append(component)
        categories = {records[row['cost_key']].get('boundary_category') for row in residual}
        if set(components) != set(layers) or 'both' in categories or 'first_plus_last_minus_internal' in categories:
            for layer in layers:
                positions[(task.rank, task.task_type.value, task.microbatch_id, layer)] = key
            task.metadata['ddp_bucket_dependency_granularity'] = 'whole_region'
            continue
        order = layers if task.task_type == TaskType.FORWARD_COMPUTE else list(reversed(layers))
        previous = list(task.dependencies)
        boundary_after = ('last' in categories and task.task_type == TaskType.FORWARD_COMPUTE
            or 'first' in categories and task.task_type == TaskType.BACKWARD_COMPUTE)
        if residual:
            residual_id = f'{key}:boundary'
            duration = sum(row['milliseconds'] * row['occurrences'] for row in residual)
            tasks[residual_id] = Task(residual_id, task.rank, TaskType.CUDA_KERNEL,
                duration=duration, microbatch_id=task.microbatch_id, region_id=task.region_id,
                dependencies=previous if not boundary_after else [], resource_requirements=task.resource_requirements,
                metadata=dict(origin_task=key, operation='region_boundary_composite',
                    cost_components=residual, region_execution_index=len(layers) if boundary_after else -1,
                    duration_evidence=sorted({reference for row in residual for reference in row['evidence_ids']}),
                    special_module_dependency_scope='measured_boundary_composite'))
            if 'first' in categories:
                positions[(task.rank, task.task_type.value, task.microbatch_id, 'embedding')] = residual_id
            if 'last' in categories:
                for special in ('output', 'final_norm', 'mtp'):
                    positions[(task.rank, task.task_type.value, task.microbatch_id, special)] = residual_id
            if not boundary_after:
                previous = [residual_id]
        total = sum(row['milliseconds'] * row['occurrences'] for row in residual)
        for execution_index, layer in enumerate(order):
            child = f'{key}:layer:{layer}'
            duration = sum(row['milliseconds'] * row['occurrences'] for row in components[layer])
            total += duration
            tasks[child] = Task(child, task.rank, TaskType.CUDA_KERNEL, duration=duration,
                microbatch_id=task.microbatch_id, region_id=task.region_id,
                dependencies=previous, resource_requirements=task.resource_requirements,
                metadata=dict(origin_task=key, operation=task.task_type.value, logical_layer=layer,
                    region_execution_index=execution_index,
                    duration_evidence=sorted({reference for row in components[layer]
                        for reference in row['evidence_ids']}),
                    cost_components=components[layer], execution_granularity='measured_layer_composite'))
            positions[(task.rank, task.task_type.value, task.microbatch_id, layer)] = child
            previous = [child]
        if residual and boundary_after:
            tasks[residual_id].dependencies = previous
            previous = [residual_id]
        if not math.isclose(total, task.duration, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError('DDP层任务展开没有保持Region总耗时')
        task.dependencies = previous
        task.duration = 0.0
        task.resource_requirements = ()
        task.metadata['cost_components'] = []
        task.metadata['duration_source'] = 'structural_region_completion_marker'
        task.metadata['ddp_bucket_dependency_granularity'] = 'measured_layer'
    return positions


def bucket_groups(evidence):
    result = {}
    for document in evidence['documents']:
        for group in document['groups']:
            if group['distributed_optimizer_instances'] != 1:
                raise ValueError('DDP Bucket当前需要单个Distributed Optimizer实例')
            identity = (tuple(group['physical_ranks']), group['group_id'])
            if identity in result:
                if result[identity]['group'] != group:
                    raise ValueError('同一通信组各Rank的DDP Bucket布局不一致')
            else:
                result[identity] = dict(group=group, ranks={})
            result[identity]['ranks'][document['rank']] = document
    for (ranks, _), row in result.items():
        if set(ranks) != set(row['ranks']):
            raise ValueError('DDP Bucket证据没有包含全部通信参与者')
    return result


def compile_bucket_tasks(projector, tasks, config, request, costs, phase):
    if phase == 'gradient':
        evidence = load_bucket_evidence(projector, request)
        projector.ddp_bucket_evidence = evidence
        if evidence is None:
            identity = dict(bucket_layout_signature=bucket_layout_signature(request),
                operation='actual_ddp_bucket_layout_and_dispatch')
            costs.append(dict(status='NEEDS_PROFILE', milliseconds=None, reason='缺少当前计划真实DDP分桶与重叠调度证据',
                cost_key=fingerprint(identity), profiling_request=dict(kind='ddp_bucket_metadata',
                    identity=identity, cost_key=fingerprint(identity))))
            next(task for task in tasks.values() if task.task_type == TaskType.DP_REDUCE).duration = None
            return
        projector.ddp_layer_positions = split_measured_regions(tasks, costs)
    else:
        evidence = projector.ddp_bucket_evidence
        if evidence is None:
            return
    positions = projector.ddp_layer_positions
    groups = bucket_groups(evidence)
    operations = {task.rank: key for key, task in tasks.items()
        if task.task_type == (TaskType.DP_REDUCE if phase == 'gradient' else TaskType.OPTIMIZER_STEP)}
    region_tasks = {(task.rank, task.task_type.value, task.microbatch_id, task.region_id // config.pp): key
        for key, task in tasks.items() if task.task_type in (TaskType.FORWARD_COMPUTE, TaskType.BACKWARD_COMPUTE)}
    communication_ids, consumers, dispatch_order = {}, {}, defaultdict(list)
    dispatch_events, parameter_prerequisites = {}, {}
    base_dependencies = {key: list(task.dependencies) for key, task in tasks.items()}
    for identity, record in groups.items():
        recorded_ranks, group_id = identity
        group = record['group']
        if len(recorded_ranks) == 1 or (phase == 'parameter' and not group['use_distributed_optimizer']):
            continue
        # schema_version=1由dist.get_process_group_ranks记录重映射后的全局逻辑Rank。
        logical_ranks = list(recorded_ranks)
        physical_ranks = tuple(projector.groups['rank_mapping'][rank] for rank in logical_ranks)
        collective = ('reduce_scatter' if group['use_distributed_optimizer'] else 'all_reduce') if phase == 'gradient' else 'all_gather'
        event_name = 'start_grad_sync' if phase == 'gradient' else 'start_param_sync'
        observed = []
        for rank in logical_ranks:
            document = record['ranks'][rank]
            events = [event for event in document['events'] if event['group_id'] == group_id
                and event['operation'] == event_name and event['step'] >= document['request']['warmup']
                and not (event['grad_handle_before'] if phase == 'gradient'
                         else event['param_gather_dispatched_before'])]
            if not events:
                raise ValueError(f'DDP Bucket缺少稳定步骤的实际通信提交事件：Rank {rank}，{group_id}，{event_name}')
            observed.append(min(events, key=lambda row: (row['step'], row['host_start_ns'])))
        key = f'projection:ddp_{phase}:{fingerprint(identity)[:20]}'
        communication_ids[identity] = key
        dispatch_events[identity] = dict(zip(logical_ranks, observed))
        parameter_prerequisites[identity] = {}
        dispatch_order[physical_ranks].append((observed[0]['host_start_ns'], identity))
        dependencies, first_consumers = set(), []
        for logical, event in zip(logical_ranks, observed):
            vp = group['vp_stage']
            microbatch = config.num_microbatches - 1 if phase == 'gradient' else 0
            kind = 'backward_compute' if phase == 'gradient' else 'forward_compute'
            region = region_tasks[(logical, kind, microbatch, vp)]
            layer_ids = {parameter['layer_id'] for bucket in group['buckets'] for parameter in bucket['parameters']
                if parameter['layer_id'] is not None}
            specials = {parameter['special_module'] for bucket in group['buckets'] for parameter in bucket['parameters']
                if parameter['layer_id'] is None}
            accesses = [positions.get((logical, kind, microbatch, layer), region) for layer in layer_ids]
            accesses.extend(positions.get((logical, kind, microbatch, special), region) for special in specials)
            if phase == 'gradient':
                if group['overlap_grad_reduce']:
                    dependencies.update(accesses)
                else:
                    dependencies.update(tasks[operations[logical]].dependencies)
            elif group['overlap_param_gather']:
                consumer = min(accesses, key=lambda node: tasks[node].metadata.get('region_execution_index', -2))
                if consumer == region and tasks[region].metadata.get('ddp_bucket_dependency_granularity') == 'measured_layer':
                    origin = [node for node, task in tasks.items() if task.metadata.get('origin_task') == region]
                    starts = [node for node in origin if not set(tasks[node].dependencies) & set(origin)]
                    if len(starts) != 1:
                        raise ValueError('DDP Region展开后缺少唯一计算起点')
                    consumer = starts[0]
                dependencies.update(base_dependencies[consumer])
                parameter_prerequisites[identity][logical] = set(base_dependencies[consumer])
                first_consumers.append(consumer)
            else:
                dependencies.add(operations[logical])
        beginning = len(costs)
        values, resources, payloads = [], set(), []
        for bucket in group['buckets']:
            field = 'grad' if phase == 'gradient' else 'param'
            elements = bucket[f'{field}_elements']
            dtype = bucket[f'{field}_dtype'].removeprefix('torch.')
            if (elements is None or dtype not in WIDTHS
                    or collective != 'all_reduce' and elements % len(logical_ranks)):
                raise ValueError('DDP Bucket通信Tensor没有合法的填充与分片信息')
            payload = elements * WIDTHS[dtype] // (len(logical_ranks) if collective != 'all_reduce' else 1)
            value, used = projector._communication(collective, payload, logical_ranks, costs, dtype=dtype)
            values.append(value)
            resources.update(used)
            payloads.append(dict(bucket_id=bucket['bucket_id'], payload_bytes=payload, dtype=dtype))
        components = projector._cost_components(costs[beginning:])
        tasks[key] = Task(key, logical_ranks[0], TaskType.NCCL_GENERIC, duration=projector._total(values),
            dependencies=sorted(dependencies), resource_requirements=tuple(sorted(resources)),
            metadata=dict(operation=f'ddp_bucket_{phase}', collective=collective,
                participant_ranks=logical_ranks, group_id=group_id, physical_ranks=list(physical_ranks),
                communication_payloads=payloads, cost_components=components,
                duration_evidence=sorted({reference for row in components for reference in row['evidence_ids']}),
                evidence_ids=[evidence['profile_id']], buffer_aliases=['persistent_DDP_parameter_and_gradient_buffers'],
                ownership_iteration='previous_optimizer_to_current_forward' if phase == 'parameter' and group['overlap_param_gather'] else 'current_step',
                observed_dispatch_regions=[event['active_region'] for event in observed]))
        if phase == 'gradient':
            for logical in logical_ranks:
                tasks[operations[logical]].dependencies.append(key)
        elif group['overlap_param_gather']:
            consumers[identity] = first_consumers
            for consumer in first_consumers:
                tasks[consumer].dependencies.append(key)
        else:
            for logical in logical_ranks:
                tasks[f'END:{logical}'].dependencies.append(key)
    for physical_ranks, sequence in dispatch_order.items():
        ordered = [identity for _, identity in sorted(sequence)]
        for previous, following in zip(ordered, ordered[1:]):
            tasks[communication_ids[following]].dependencies.append(communication_ids[previous])
    if phase == 'parameter':
        for identity, record in groups.items():
            next_group = record['group']['next_param_gather_group_id']
            next_identity = (identity[0], next_group)
            if identity not in consumers or next_identity not in communication_ids:
                continue
            if record['group']['align_param_gather']:
                continue
            for consumer in consumers[identity]:
                logical = tasks[consumer].rank
                following = dispatch_events[next_identity][logical]
                source = record['ranks'][logical]
                triggers = [event for event in source['events'] if event['group_id'] == identity[1]
                    and event['operation'] == 'finish_param_sync' and event['step'] == following['step']
                    and event['host_start_ns'] <= following['host_start_ns'] <= event['host_end_ns']]
                if not triggers:
                    continue
                launch = f'{communication_ids[identity]}:consumed:{tasks[consumer].rank}'
                tasks[launch] = Task(launch, tasks[consumer].rank, TaskType.CUDA_SYNC, duration=0,
                    dependencies=[*base_dependencies[consumer], communication_ids[identity]],
                    metadata=dict(operation='bucket_prefetch_dispatch', cost_components=[],
                        duration_evidence=[evidence['profile_id']],
                        duration_source='structural_zero_duration_dispatch',
                        dispatch_evidence='recorded_start_param_sync_inside_finish_param_sync',
                        source_step=following['step']))
                parameter_prerequisites[next_identity][logical] = {launch}
        for identity, prerequisites in parameter_prerequisites.items():
            if identity not in consumers:
                continue
            task = tasks[communication_ids[identity]]
            ordered_communications = {key for key in task.dependencies if key in communication_ids.values()}
            task.dependencies = sorted(ordered_communications | set().union(*prerequisites.values()))
        topological_order(tasks)
