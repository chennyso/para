from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from contracts import ProfilingRequest, fingerprint
from predictor.profile_database import effective_execution_config, plan_signature, same_training_plan
from predictor.measured_modules import enrich_module_profile
from predictor.state_sharding import native_overlap_errors


SOURCE = 'paired_native_component_service_time_calibration'
PROTOCOL = 'paired_component_services_v1'
CONTROL = dict(overlap_grad_reduce=False,overlap_param_gather=False,align_param_gather=False,overlap_p2p=False)
FSDP_CONTROL_PROTOCOL = 'native_fsdp_unit_boundary_services_v1'


def isolated_request(request):
    if request.get('execution_backend') == 'megatron_fsdp':
        return dict(request, paramorph_fsdp_component_services=True,
            paramorph_fsdp_component_isolation=True, communication_executor_protocol=FSDP_CONTROL_PROTOCOL)
    return dict(request, **CONTROL)


def component_key(row, rows):
    metadata = row['metadata']
    if metadata.get('execution_granularity') == 'measured_layer_composite':
        phase, kind = metadata['operation'], 'layer'
    elif metadata.get('operation') == 'region_boundary_composite':
        phase, kind = rows[metadata['origin_task']]['kind'], 'boundary'
    elif row['kind'] in ('forward_compute','backward_compute'):
        phase, kind = row['kind'], 'region'
    else:
        return None
    return fingerprint(dict(rank=row['logical_rank'],region=row['region'],phase=phase,kind=kind,
        layer=metadata.get('logical_layer'),microbatch=row.get('microbatch')))


def verified_profile(database, identifier):
    profiles = dict(database.measured_profiles())
    profile = database.get(identifier,profiles[identifier]['request'])
    source = profile['source_directory']
    for rank in range(profile['request']['node_count']*profile['request']['gpus_per_node']):
        for suffix in ('cost','static'):
            name = f'rank{rank}.paramorph-{suffix}.json'
            path = Path(source)/'artifacts'/name
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==profile['evidence_files'][name]
        assert len(profile['measured_windows'][str(rank)])>=3
        if profile['request'].get('paramorph_fsdp_component_services'):
            name = f'rank{rank}.paramorph-fsdp-services.json'
            path = Path(source)/'artifacts'/name
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == profile['evidence_files'][name]
            control = json.loads(path.read_text())
            assert control['protocol'] == FSDP_CONTROL_PROTOCOL
            assert control['isolated'] == profile['request'].get('paramorph_fsdp_component_isolation', False)
            assert control['pending_gradient_calls'] == 0
            assert control['overlap_grad_reduce'] is True and control['overlap_param_gather'] is True
    return enrich_module_profile(profile)


def context_record(projector, request):
    if projector.blind or projector.mode=='analytical_only' or getattr(projector,'forced_component_profile_id',None) is not None:
        return None
    signature = plan_signature(request)
    uuids = sorted(str(row.get('uuid',row.get('gpu_uuid'))).removeprefix('GPU-').lower()
        for row in projector.topology.devices)
    matching = [row for row in projector.compute_surface.database.overlap_profiles()
        if row['benchmark']['source']==SOURCE and row['benchmark']['target_plan_signature']==signature
        and row['benchmark']['hardware_uuids']==uuids]
    return max(matching,key=lambda row:row['benchmark']['sampling_finished_unix'],default=None)


def selected_component_profile(projector, request):
    if projector.blind or projector.mode=='analytical_only':
        return None
    identifier = getattr(projector,'forced_component_profile_id',None)
    if identifier is None and projector.native_component_context is not None:
        identifier = projector.native_component_context['benchmark']['isolated_profile_id']
    if identifier is None:
        return None
    cache = projector.module_profiles
    if identifier not in cache:
        cache[identifier] = verified_profile(projector.compute_surface.database,identifier)
    profile = cache[identifier]
    previous = profile['request']
    if not same_training_plan(previous,request) and not same_training_plan(previous,isolated_request(request)):
        before=effective_execution_config(previous)
        after=effective_execution_config(isolated_request(request))
        differences={key:dict(measured=before.get(key),requested=after.get(key))
            for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
        if previous['decoder_chunks']!=request['decoder_chunks']:
            differences['decoder_chunks']=dict(measured=previous['decoder_chunks'],requested=request['decoder_chunks'])
        raise ValueError(dict(status='NATIVE_COMPONENT_PLAN_MISMATCH',profile_id=identifier,differences=differences))
    actual = sorted(str(row.get('uuid',row.get('gpu_uuid'))).removeprefix('GPU-').lower()
        for row in projector.topology.devices)
    measured = sorted(str(row['gpu_uuid']).removeprefix('GPU-').lower() for row in profile['hardware'])
    if actual!=measured:
        raise ValueError('成对组件成本对应不同的实际GPU')
    return profile


def bind_components(projector, tasks):
    context = projector.native_component_context
    if context is None:
        return
    rows = {key:dict(logical_rank=task.rank,kind=task.task_type.value,region=task.region_id,
        microbatch=task.microbatch_id,metadata=task.metadata) for key,task in tasks.items()}
    units = context['benchmark']['units']
    for key,task in tasks.items():
        identity = component_key(rows[key],rows)
        if identity is None or not task.duration:
            continue
        unit = units[identity]
        if not math.isclose(task.duration,unit['isolated_ms'],rel_tol=1e-8,abs_tol=1e-7):
            raise ValueError('组件基准成本与成对测量的隔离成本不同')
        task.metadata['native_component_overlap'] = dict(profile_id=context['id'],unit=identity,
            gamma=unit['gamma'],scope='exact_plan_conditional_composite_service_cost',
            isolated_profile_id=context['benchmark']['isolated_profile_id'])


def project_components(search, request, identifier):
    candidate = search.rewriter.candidate(request)
    graph = search.rewriter.compiler.observation(candidate,[],[])['training_graph']
    projector = search.predictor.projector
    projector.forced_component_profile_id = identifier
    try:
        report = projector.project(request,candidate_graph=graph,cost_mode=search.config.predictor_cost_mode)
    finally:
        projector.forced_component_profile_id = None
    rows = report['execution_graph']['tasks']
    result = {}
    for row in rows.values():
        identity = component_key(row,rows)
        if identity is not None and row['duration_ms']>0:
            assert identity not in result
            result[identity] = row['duration_ms']
    assert result
    return result


def execute_paired_profile(search, parent):
    identifier = parent['request_id']
    request = parent['execution_request']
    available = context_record(search.predictor.projector, request)
    if available is not None:
        benchmark = available['benchmark']
        profiles = [verified_profile(search.predictor.database, identifier) for identifier in (
            benchmark['isolated_profile_id'], *benchmark['concurrent_profile_ids'])]
        finished = search.store.update_profile_request(identifier, state='COMPLETED', result=dict(
            status='PASS', protocol=PROTOCOL, component_profile_id=available['id'],
            reused_measured_profiles=[profile['id'] for profile in profiles],
            additional_parent_gpu_hours=0, reused_existing_components=True))
        search.synchronize_cost_revision()
        search.refresh_profile_trials(finished)
        return next(row for row in search.store.profile_requests() if row['request_id'] == identifier)
    control_errors = native_overlap_errors(isolated_request(request))
    if request.get('execution_backend') == 'megatron_fsdp' and request.get('fsdp_prefetch', False):
        control_errors.append('当前原生FSDP组件隔离协议要求关闭参数预取')
    if control_errors:
        return search.store.update_profile_request(identifier, state='FAILED', result=dict(
            status='UNSUPPORTED_NATIVE_ISOLATION_CONTROL', native_constraint_errors=control_errors,
            target_strategy_supported=True, additional_parent_gpu_hours=0,
            child_request_ids=[row['request_id'] for row in search.store.profile_requests()
                if row['request_id'].startswith(identifier+'-')],
            reason='成对采集的隔离控制与原生FSDP异步分片约束不兼容；目标训练策略保留，成本状态保持未知'))
    search.store.update_profile_request(identifier,state='RUNNING')
    children = []
    for phase in ('concurrent-before','isolated','concurrent-after'):
        execution = dict(request)
        for key in ('native_overlap_protocol','native_overlap_trace_backend'):
            execution.pop(key,None)
        if request.get('execution_backend') == 'megatron_fsdp':
            execution['paramorph_fsdp_component_services'] = True
            execution['paramorph_fsdp_component_isolation'] = phase == 'isolated'
        if phase=='isolated':
            execution = isolated_request(execution)
        child_id = identifier+'-'+phase
        execution.update(run_id=child_id,port=35000+int(fingerprint(child_id)[-4:],16)%1000,
            profile=False,paramorph_trace=False,paramorph_calibration=True,
            paramorph_optimizer_calibration=True,paramorph_operator_calibration=False,
            paramorph_memory_audit=True,max_allocation_seconds=None)
        child = search.store.put_profile_request(ProfilingRequest(request_id=child_id,
            trial_ids=parent['trial_ids'],cost_keys=parent['cost_keys'],
            measurement_type='native_training_cost_profile',execution_request=execution,
            reason='相同训练合同下的组件服务时间成对测量：'+phase,
            estimated_gpu_hours=parent['estimated_gpu_hours']/3,priority=parent['priority']))
        previous = None
        stored_run = child_id + ('-'+search.config.run_namespace if search.config.run_namespace else '')
        existing_run = search.output/'profiling'/'runs'/stored_run/'request.json'
        if phase == 'concurrent-before' and child['state'] != 'COMPLETED' and not existing_run.exists():
            from predictor.profile_reuse import reusable_training_profile
            previous = reusable_training_profile(search, execution)
        if previous is not None:
            profile_id, profile, measurement = previous
            finished = search.store.update_profile_request(child_id, state='COMPLETED', result=dict(
                measurement=measurement, profile_id=profile_id, additional_gpu_hours=0,
                reused_existing_profile=True, source_directory=profile['source_directory']))
            search.store.add_evidence(child_id, 'MEASURED', finished['result'])
        else:
            finished = search.execute_profile(child['request_id'],refresh_predictions=False)
        children.append(finished)
        if finished['state']!='COMPLETED':
            return search.store.update_profile_request(identifier,state='FAILED',
                result=dict(status='PAIRED_COMPONENT_CALIBRATION_INCOMPLETE',children=children))
    profiles = [verified_profile(search.predictor.database,row['result']['profile_id']) for row in children]
    reference = profiles[1]
    for index,profile in enumerate(profiles):
        expected = isolated_request(request) if index==1 else request
        assert same_training_plan(profile['request'],expected)
        assert profile['source_hashes']==reference['source_hashes']
        for key in ('data','random_seed','data_random_seed','training_data_contract','optimizer_hyperparameters','optimizer_schedule'):
            assert profile['request'].get(key)==reference['request'].get(key)
    if request.get('execution_backend') == 'megatron_fsdp':
        for rank in range(request['node_count']*request['gpus_per_node']):
            controls = [json.loads((Path(profile['source_directory'])/'artifacts'/
                f'rank{rank}.paramorph-fsdp-services.json').read_text()) for profile in profiles]
            assert len({row['counters']['gradient_parameters'] for row in controls}) == 1
            measurements = [json.loads((Path(profile['source_directory'])/'artifacts'/
                f'rank{rank}.measurements.json').read_text()) for profile in profiles]
            reference_steps = measurements[0]['steps']
            for measurement in measurements[1:]:
                assert len(measurement['steps']) == len(reference_steps)
                for baseline, observed in zip(reference_steps, measurement['steps'], strict=True):
                    assert baseline['optimizer_update_successful'] and observed['optimizer_update_successful']
                    assert baseline['loss'].keys() == observed['loss'].keys()
                    for name, value in baseline['loss'].items():
                        assert math.isclose(value, observed['loss'][name], rel_tol=0.01, abs_tol=1e-5), (
                            rank, name, value, observed['loss'][name])
                    if baseline['grad_norm'] is not None:
                        assert observed['grad_norm'] is not None
                        assert math.isclose(baseline['grad_norm'], observed['grad_norm'],
                            rel_tol=0.01, abs_tol=1e-5), (rank, baseline['grad_norm'], observed['grad_norm'])
    observations = [project_components(search,request,profile['id']) for profile in profiles]
    assert set(observations[0])==set(observations[1])==set(observations[2])
    units = {}
    for key,isolated in observations[1].items():
        concurrent = [observations[0][key],observations[2][key]]
        assert isolated>0 and all(math.isfinite(value) and value>0 for value in concurrent)
        units[key] = dict(isolated_ms=isolated,concurrent_ms=concurrent,
            gamma=max(1.0,mean(concurrent)/isolated),unclamped_ratio=mean(concurrent)/isolated,
            before_after_relative_difference=abs(concurrent[1]-concurrent[0])/mean(concurrent))
    benchmark = dict(source=SOURCE,protocol=PROTOCOL,target_plan_signature=plan_signature(request),
        isolated_profile_id=reference['id'],concurrent_profile_ids=[profiles[0]['id'],profiles[2]['id']],
        hardware_uuids=sorted(str(row['gpu_uuid']).removeprefix('GPU-').lower() for row in reference['hardware']),
        controlled_variables=(dict(communication_executor_protocol=FSDP_CONTROL_PROTOCOL,
            paramorph_fsdp_component_isolation=True) if request.get('execution_backend') == 'megatron_fsdp'
            else CONTROL),units=units,source_hashes=reference['source_hashes'],
        sampling_finished_unix=max((Path(profile['source_directory'])/'result.json').stat().st_mtime for profile in profiles),
        scope='相同完整配置的并发开关干预；隔离组件成本与并发条件服务成本，未使用整步吞吐拟合',
        prediction_scope='exact_training_plan_only_no_cross_configuration_gamma_reuse')
    evidence = identifier+':components'
    search.predictor.database.add_overlap(evidence,benchmark)
    search.store.add_evidence(evidence,'MEASURED',benchmark)
    finished = search.store.update_profile_request(identifier,state='COMPLETED',result=dict(status='PASS',
        protocol=PROTOCOL,component_profile_id=evidence,child_request_ids=[row['request_id'] for row in children],
        component_count=len(units),additional_parent_gpu_hours=0))
    search.synchronize_cost_revision()
    search.refresh_profile_trials(finished)
    return next(row for row in search.store.profile_requests() if row['request_id']==identifier)
