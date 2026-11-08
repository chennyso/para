import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paramorph'))
from schedule_validator import ScheduleValidator
from hybrid_node import launch_geometry
PODS = tuple(os.environ.get('PARAMELD_TRAINER_PODS', '').split())
REMOTE = Path(os.environ.get('PARAMELD_REMOTE_RUNS', '/workspace/frontier/hybrid-performance-20261008'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def cluster_environment():
    environment = os.environ.copy()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(key, None)
    environment['KUBECTL_REMOTE_COMMAND_WEBSOCKETS'] = 'false'
    environment['GODEBUG'] = environment.get('GODEBUG', '') + ',http2client=0'
    return environment


def kubectl(*arguments, timeout_seconds=None):
    seconds = timeout_seconds if timeout_seconds is not None else (120 if arguments[0] == 'cp' else 30)
    command = ['kubectl', '--context', 'prd', '--request-timeout=' + str(seconds) + 's', *map(str, arguments)]
    for attempt in range(3):
        result = subprocess.run(command, check=False, capture_output=True, text=True,
            timeout=seconds + 30, env=cluster_environment())
        if result.returncode == 0:
            return result.stdout
        transient = ('TLS handshake timeout' in result.stderr
            or 'i/o timeout' in result.stderr or 'connection reset by peer' in result.stderr
            or 'Client.Timeout exceeded while awaiting headers' in result.stderr
            or 'context deadline exceeded' in result.stderr
            or (arguments[0] == 'cp' and result.stderr.strip() in ('', 'Error from server:', 'unexpected EOF')))
        if not transient or attempt == 2:
            if result.stderr:
                print(result.stderr, file=sys.stderr, flush=True)
            result.check_returncode()
        time.sleep(2)
    raise RuntimeError('kubectl重试状态异常')


def transfer_runtime_bundle(directory, names, remote):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as stream:
        for name in names:
            stream.add(directory / name, arcname=name)
    payload = archive.getvalue()
    targets = tuple(reversed(PODS[:2]))
    for attempt in range(3):
        pod = targets[attempt % len(targets)]
        command = ['kubectl', '--context', 'prd', '--request-timeout=30s', 'exec', '-i', '-n', 'default',
            pod, '-c', 'trainer', '--', 'sh', '-c', 'mkdir -p "$1" && tar -xzf - -C "$1"', 'sh', str(remote)]
        try:
            result = subprocess.run(command, input=payload, capture_output=True, timeout=45, env=cluster_environment())
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
        else:
            if result.returncode == 0:
                return
            error = result.stderr.decode(errors='replace').strip()
            transient = error in ('', 'Error from server:', 'unexpected EOF') or any(message in error for message in (
                'context deadline exceeded', 'i/o timeout', 'TLS handshake timeout', 'connection reset by peer'))
            if not transient or attempt == 2:
                print(error, file=sys.stderr, flush=True)
                result.check_returncode()
        print(json.dumps(dict(event='RUNTIME_TRANSFER_RETRY', pod=pod, destination=str(remote))),
            file=sys.stderr, flush=True)


def shared_pvc_exec(*arguments, timeout_seconds=None):
    targets = tuple(reversed(PODS[:2]))
    for index, pod in enumerate(targets):
        try:
            return kubectl('exec', '-n', 'default', pod, '-c', 'trainer', '--', *arguments,
                timeout_seconds=timeout_seconds)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = str(error.stderr or '').strip()
            transport = isinstance(error, subprocess.TimeoutExpired) or detail in ('', 'Error from server:') or any(
                message in detail for message in ('i/o timeout', 'TLS handshake timeout', 'context deadline exceeded',
                    'connection reset by peer', 'Client.Timeout', 'Unable to connect to the server', 'unexpected EOF',
                    'error dialing backend', 'error upgrading connection', 'stream error'))
            if not transport or index == len(targets) - 1:
                raise


def download_artifact(remote, destination, expected_digest):
    partial = destination.with_suffix('.partial')
    targets = tuple(reversed(PODS[:2]))
    for attempt in range(3):
        pod = targets[attempt % len(targets)]
        with partial.open('wb') as target:
            result = subprocess.run(['kubectl', '--context', 'prd', '--request-timeout=90s',
                'exec', '-n', 'default', pod, '-c', 'trainer', '--', 'cat', str(remote)],
                stdout=target, stderr=subprocess.PIPE, timeout=100, env=cluster_environment())
        if result.returncode == 0:
            with partial.open('rb') as target:
                digest = hashlib.file_digest(target, 'sha256').hexdigest()
            if digest == expected_digest:
                partial.replace(destination)
                return
            if attempt == 2:
                raise ValueError('实验归档SHA256检查失败')
        elif attempt == 2:
            result.check_returncode()


def node_action(node, request, action, run):
    return json.loads(kubectl('exec', '-n', 'default', node['pod'], '-c', 'trainer', '--', 'python',
        run / 'code/hybrid_node.py', action, '--run', run, '--node', node['index'],
        '--master', request['master'], '--ib', node['ib']))


def observe_training_node(node, request, run):
    try:
        state = node_action(node, request, 'status', run)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        # 状态查询失败无法证明训练已经退出；Pod内的执行器仍负责运行时限。
        print(json.dumps(dict(event='TRAINING_STATUS_UNAVAILABLE', node=node['index'],
            pod=node['pod'], failure=type(error).__name__, run=str(run))), file=sys.stderr, flush=True)
        return None
    if state.get('status') not in ('running', 'complete', 'failed'):
        raise ValueError(f'远端训练状态无效：{state}')
    return state


def launch_training_node(node, request, run):
    for attempt in range(3):
        try:
            state = node_action(node, request, 'start', run)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = str(error.stderr or '').strip()
            transport = isinstance(error, subprocess.TimeoutExpired) or detail in ('', 'Error from server:') or any(
                message in detail for message in ('i/o timeout', 'TLS handshake timeout', 'connection reset by peer',
                    'Client.Timeout', 'Unable to connect to the server', 'unexpected EOF', 'error dialing backend',
                    'error upgrading connection', 'stream error'))
            if not transport or attempt == 2:
                raise
            # 启动回执丢失时只重发同一轮的节点请求，远端锁保证不启动第二个进程。
            print(json.dumps(dict(event='TRAINING_LAUNCH_RECHECK', node=node['index'],
                pod=node['pod'], attempt=attempt + 1, run=str(run))), file=sys.stderr, flush=True)
            time.sleep(2 ** (attempt + 1))
            continue
        if state.get('status') not in ('started', 'already_started'):
            raise ValueError(f'远端训练启动回执无效：{state}')
        return state


def execute(request, output, nodes, resume_existing=False, extra_sources=None, metadata_archive=None,
            release_allocation=None, source_overrides=None, collection_mode=None):
    node_count = request.get('node_count', 2)
    world = node_count * request.get('gpus_per_node', 8)
    workers, devices_per_worker = launch_geometry(request)
    if len(nodes) != workers or (devices_per_worker != request.get('gpus_per_node', 8)
            and len({row['node'] for row in nodes}) != node_count):
        raise ValueError('实际Worker和物理节点数量与训练申请不一致')
    if request.get('prediction_gate') is not None:
        from paramorph.predictor.profile_database import plan_signature
        prediction = request['prediction_gate']
        if prediction['candidate_id'] != request['run_id']:
            raise ValueError('预测准入记录与提交候选身份不一致')
        if prediction['plan_signature'] != plan_signature(request):
            raise ValueError('预测配置与实际提交配置不一致')
        if prediction['memory']['admission'] != 'WITHIN_ESTIMATED_CAPACITY':
            raise ValueError('候选显存预测尚未通过提交准入')
    directory = output / 'runs' / request['run_id']
    if (directory / 'result.json').exists():
        stored = json.loads((directory / 'request.json').read_text())
        if stored != request:
            raise ValueError('A cached hybrid trial has a different request')
        return json.loads((directory / 'result.json').read_text()) | {
            'reused_cached_run': True, 'gpu_allocation_seconds': 0.0}
    remote = REMOTE / request['run_id']
    if resume_existing:
        if json.loads((directory / 'request.json').read_text()) != request:
            raise ValueError('Existing hybrid trial differs from the collection request')
        sources = json.loads((directory / 'source_hashes.json').read_text())
        def staged_transfer():
            return json.loads(shared_pvc_exec(
                'python', '-c',
                'import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); '
                'print(json.dumps(dict(request=(p/"request.json").is_file(), '
                'launched=any(p.glob("node*.pid")) or any(p.glob("node*.exit.json")), '
                'code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/"code").glob("*.py")})))', remote))

        transfer = staged_transfer()
        if not transfer['request'] or any(transfer['code'].get(name) != digest for name, digest in sources.items()):
            if transfer['launched'] or (directory / 'allocation.json').exists():
                raise ValueError('已有执行记录的训练缺少完整原始远端文件')
            for name, digest in sources.items():
                if hashlib.sha256((directory / 'code' / name).read_bytes()).hexdigest() != digest:
                    raise ValueError('待接续传输的源文件与原始记录不同')
            transfer_runtime_bundle(directory, ('code', 'request.json', 'source_hashes.json'), remote)
            transfer = staged_transfer()
            if not transfer['request'] or any(transfer['code'].get(name) != digest for name, digest in sources.items()):
                raise ValueError('恢复传输后的远端源文件与原始记录不同')
        stored_digest = shared_pvc_exec('sha256sum', remote / 'request.json').split()[0]
        if stored_digest != hashlib.sha256((directory / 'request.json').read_bytes()).hexdigest():
            raise ValueError('Remote hybrid trial differs from the recorded request')
    else:
        if (directory / 'request.json').exists():
            raise ValueError('An existing hybrid trial must be collected with resume_existing=True')
        directory.mkdir(parents=True, exist_ok=True)
        code = directory / 'code'
        code.mkdir(exist_ok=True)
        runtime_sources = {'native_hybrid_train.py': ROOT / 'experiments/native_hybrid_train.py',
            'hybrid_training_entry.py': ROOT / 'hybrid_training_entry.py',
            'checkpoint_io_guard.py': ROOT / 'checkpoint_io_guard.py',
            'frontier_hybrid_provider.py': ROOT / 'frontier_hybrid_provider.py',
            'frontier_falcon_cut.py': ROOT / 'frontier_falcon_cut.py',
            'frontier_hybrid_router.py': ROOT / 'frontier_hybrid_router.py',
            'native_frontier_checkpoint.py': ROOT / 'integration/megatron/core/frontier/megatron/native_checkpoint.py',
            'hybrid_artifact_export.py': ROOT / 'experiments/hybrid_artifact_export.py',
            'hybrid_node.py': ROOT / 'experiments/hybrid_node.py'}
        if set(source_overrides or {}) - set(runtime_sources):
            raise ValueError('历史运行代码包含未注册的源文件')
        runtime_sources.update(source_overrides or {})
        for name, source in runtime_sources.items():
            shutil.copyfile(source, code / name)
        for name, source in (extra_sources or {}).items():
            if Path(name).name != name or (code / name).exists():
                raise ValueError('Additional runtime source has an invalid or duplicate name')
            shutil.copyfile(source, code / name)
        write(directory / 'request.json', request)
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code.iterdir()}
        write(directory / 'source_hashes.json', sources)
        transfer_runtime_bundle(directory, ('code', 'request.json', 'source_hashes.json'), remote)
    started = time.monotonic()
    allocation_path = directory / 'allocation.json'
    completed_path = directory / 'allocation-completed.json'
    execution_deadline_reached = (json.loads(completed_path.read_text()).get('execution_deadline_reached', False)
        if resume_existing and completed_path.is_file() else False)
    if resume_existing:
        launch_records = {Path(path).name for path in shared_pvc_exec(
            'find', remote, '-maxdepth', '1', '-type', 'f').splitlines()}
        start_nodes = [node for node in nodes if not any(
            f'node{node["index"]}.{suffix}' in launch_records
            for suffix in ('pid', 'helper.pid', 'exit.json'))]
    else:
        start_nodes = nodes
    if len(start_nodes) == len(nodes):
        write(allocation_path, dict(started_unix_seconds=time.time(), allocated_devices=world))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, workers)) as pool:
        if start_nodes:
            list(pool.map(lambda node: node_action(node, request, 'preflight', remote), start_nodes))
            list(pool.map(lambda node: launch_training_node(node, request, remote), start_nodes))
        start_walltime = json.loads(allocation_path.read_text())['started_unix_seconds']
        remote_limit = request.get('max_allocation_seconds')
        peer_failed = False
        while True:
            states = list(pool.map(lambda node: observe_training_node(node, request, remote), nodes))
            if all(state is not None for state in states):
                if all(state['status'] != 'running' for state in states):
                    execution_deadline_reached |= any(state.get('execution_deadline_reached', False) for state in states)
                    break
                failed_now = any(state.get('returncode', 0) != 0 for state in states)
                if failed_now and peer_failed:
                    running = [node for node, state in zip(nodes, states, strict=True) if state['status'] == 'running']
                    list(pool.map(lambda node: node_action(node, request, 'stop', remote), running))
                peer_failed = failed_now
            if remote_limit is not None and time.time() > start_walltime + remote_limit + 180:
                raise TimeoutError(f'远端执行时限已过，无法确认全部节点退出；保留本次目录以继续收集：{remote}')
            time.sleep(5)
    gpu_allocation_seconds = time.monotonic() - started
    if resume_existing and allocation_path.is_file():
        if completed_path.is_file():
            gpu_allocation_seconds = json.loads(completed_path.read_text())['gpu_allocation_seconds']
        else:
            gpu_allocation_seconds = time.time() - json.loads(allocation_path.read_text())['started_unix_seconds']
    if not completed_path.is_file():
        write(completed_path, dict(completed_unix_seconds=time.time(),
            gpu_allocation_seconds=gpu_allocation_seconds,
            execution_deadline_reached=execution_deadline_reached,
            timing_scope='client_observed_native_exit_before_artifact_collection'))
    if release_allocation is not None:
        release_allocation()
    if ((collection_mode == 'summary' or request.get('result_collection') == 'summary') and not request.get('instrumented')
            and not request.get('profile') and not request.get('nccl_profile')
            and not request.get('compute_communication_profile') and not request.get('communication_test')):
        helper = REMOTE.parent / 'hybrid_result_export.py'
        transfer_runtime_bundle(ROOT / 'experiments', ('hybrid_result_export.py',), REMOTE.parent)
        payload = json.loads(shared_pvc_exec(
            'python', helper, '--run', remote, '--gpu-seconds', str(gpu_allocation_seconds), timeout_seconds=120))
        artifacts = directory / 'artifacts'
        artifacts.mkdir(exist_ok=True)
        for name, content in payload['files'].items():
            if Path(name).name != name:
                raise ValueError('结果文件名超出当前实验目录')
            (artifacts / name).write_text(content)
        record = payload['result']
        record['execution_deadline_reached'] = (
            execution_deadline_reached or record.get('execution_deadline_reached', False))
        record['pvc_raw_artifacts'] = record['raw_artifacts']
        record['raw_artifacts'] = str(artifacts)
        write(directory / 'result.json', record)
        print(json.dumps({key: record.get(key) for key in ('run_id', 'status', 'tokens_per_second',
            'step_seconds', 'peak_allocated_gib', 'gpu_allocation_seconds')}), flush=True)
        return record
    artifacts = directory / 'artifacts'
    metadata = directory / 'metadata-export.tar.gz'
    if metadata_archive is None:
        exported = json.loads(shared_pvc_exec('python', remote / 'code/hybrid_artifact_export.py',
            '--run', remote, timeout_seconds=600))
        download_artifact(exported['archive_path'], metadata, exported['sha256'])
        expected_digest = exported['sha256']
    else:
        if not resume_existing or Path(metadata_archive).resolve() != metadata.resolve():
            raise ValueError('指定的归档需要属于当前已结束实验')
        expected_digest = shared_pvc_exec('sha256sum', remote / 'metadata-export.tar.gz').split()[0]
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != expected_digest:
        raise ValueError('The downloaded training metadata archive is incomplete')
    with tarfile.open(metadata, 'r:gz') as package:
        package.extractall(artifacts, filter='data')
    for name, digest in sources.items():
        if hashlib.sha256((artifacts / 'code' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Hybrid runtime source changed during the trial')
    codes = [state['returncode'] for state in states]
    record = request | {'returncodes': codes, 'elapsed_seconds': time.monotonic() - started,
        'gpu_allocation_seconds': gpu_allocation_seconds,
        'execution_deadline_reached': execution_deadline_reached,
        'raw_artifacts': str(artifacts), 'source_hashes': sources, 'performance_eligible': False}
    if resume_existing:
        record['collection_elapsed_seconds'] = record.pop('elapsed_seconds')
        record['collected_existing_run'] = True
    if request.get('profile', False):
        record['remote_trace_directory'] = str(remote)
        record['nsys_index'] = json.loads((artifacts / 'nsys-index.json').read_text())
    if codes != [0] * len(nodes):
        logs = ''.join((artifacts / f'node{index}.log').read_text() for index in range(len(nodes)))
        allocation_failed = ('out of memory' in logs.lower()
            or 'Failed to CUDA calloc' in logs or 'Failed to CUDA malloc' in logs)
        record.update(status='CUDA_OOM' if allocation_failed else 'FAILED')
        host_failures = [state['host_memory'] for state in states if state.get('host_memory')]
        if host_failures:
            record.update(status='HOST_MEMORY_BUDGET_EXCEEDED', host_memory_failures=host_failures)
    elif request.get('compute_communication_profile', False):
        overlap = json.loads((artifacts / 'compute-communication-overlap.json').read_text())
        if (overlap['source'] != 'real_gpu_compute_communication_microbenchmark'
                or overlap['collective'] != request['overlap_collective']
                or overlap['group_ranks'] != request['overlap_group_ranks']
                or overlap['payload_bytes'] != request['overlap_payload_bytes']
                or overlap['compute_shape'] != request['overlap_compute_shape']
                or overlap['gamma'] < 1 or not overlap['samples']):
            raise ValueError('计算通信并发测量没有覆盖请求的Kernel Shape与Collective')
        record.update(status='PASS', overlap_benchmark=overlap,
            validation_scope='actual_compute_communication_overlap_microbenchmark',
            performance_eligible=False)
    elif request.get('nccl_profile', False):
        benchmark = json.loads((artifacts / 'nccl-costs.json').read_text())
        if (benchmark['world_size'] != world or benchmark['source'] != 'actual_nccl_cuda_event_microbenchmark'
            or not benchmark['records'] or any(not row['samples'] or
                any(value <= 0 for value in row['samples']) for row in benchmark['records'])):
            raise ValueError('NCCL成本测量没有覆盖实际参与设备和有效计时')
        record.update(status='PASS', nccl_benchmark=benchmark,
                      nccl_batch_cost_keys=request.get('nccl_batch_cost_keys', []),
                      nccl_batch_profile_request_ids=request.get('nccl_batch_profile_request_ids', []),
                      validation_scope='actual_nccl_cuda_event_microbenchmark', performance_eligible=False)
    elif request.get('communication_test', False):
        rows = [json.loads((artifacts / f'rank{rank}.protocol-result.json').read_text())
                for rank in range(request['pp'])]
        if any(row['rank'] != rank or row['status'] != 'PASS' or row['source'] != 'actual_nccl'
               or not row['progress']['valid'] for rank, row in enumerate(rows)):
            raise ValueError('实际NCCL通信测试没有完成进展验证')
        record.update(status='PASS', validation_scope=rows[0]['validation_scope'], protocol_results=rows)
    else:
        rows = [json.loads((artifacts / f'rank{rank}.measurements.json').read_text()) for rank in range(world)]
        manifests = [json.loads((artifacts / f'rank{rank}.manifest.json').read_text()) for rank in range(world)]
        if request.get('recompute') == 'selective':
            modules = request['memory_policy']['recompute_modules']
            for module in modules:
                if sum(manifest['native_selective_recomputation']['enabled_modules'][module]
                        for manifest in manifests) == 0:
                    raise ValueError(f'原生重计算模块没有实际启用：{module}')
            if any(manifest['native_selective_recomputation']['requested_modules'] != modules
                    for manifest in manifests):
                raise ValueError('原生重计算模块与完整策略不同')
            record['native_selective_recompute_verified'] = True
        for rank, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
            if manifest['world'] != world:
                raise ValueError('运行时Rank数量与训练申请不一致')
            if row['rank'] != rank or row['status'] != 'complete' or row['updates'] != request['warmup'] + request['measured']:
                raise ValueError('The hybrid measurement did not complete the recorded protocol')
            if (row['warmup'] != request['warmup'] or row['measured'] != request['measured']
                    or row['consumed_samples'] != (request['warmup'] + request['measured']) * request['global_batch']):
                raise ValueError('The hybrid timing or input accounting differs from the request')
            if not all(step['optimizer_update_successful'] for step in row['steps']):
                raise ValueError('A hybrid optimizer update failed')
            for key in ('tp', 'cp', 'pp', 'ep', 'vpp', 'sequence', 'micro_batch', 'global_batch'):
                if key == 'cp' and key not in manifest:
                    if manifest['world'] != manifest['tp'] * manifest['pp']:
                        raise ValueError('历史记录缺少CP，现有Rank数量无法唯一确定CP')
                    actual_value = 1
                    record.setdefault('parallel_configuration_proofs', []).append(dict(rank=rank,
                        cp=1, dp=1, constraint='world = tp * pp * cp * dp; world = tp * pp'))
                else:
                    actual_value = manifest[key]
                if actual_value != request[key]:
                    raise ValueError('The actual hybrid parallel configuration differs from the request')
            for key, expected in (
                ('etp', request.get('etp', request['tp'])),
                ('expert_dp', world // (request.get('etp', request['tp']) * request['ep'] * request['pp'])),
                ('dp', request['dp']), ('optimizer_sharding', request.get('optimizer_sharding', 'distributed'))):
                if key in manifest and manifest[key] != expected:
                    raise ValueError('实际Expert并行、DP或者Optimizer Sharding与执行计划不同')
        if request.get('record_native_schedule'):
            from scheduling.native_microbatches import verify_native_schedule
            record['native_pipeline_schedule'] = verify_native_schedule(request, manifests)
        window = max(row['window_seconds'] for row in rows)
        record.update(status='PASS', performance_eligible=not (request.get('profile', False)
            or request.get('instrumented', False) or request.get('validation_mode')),
            tokens_per_second=request['measured'] * request['global_batch'] * request['sequence'] / window,
            step_seconds=window / request['measured'], peak_allocated_gib=max(row['allocated_gib'] for row in rows),
            validation_scope=rows[0]['validation_scope'], numerical_equivalence_certified=False)
        if request.get('paramorph_execution_plan') is not None:
            costs = [json.loads((artifacts / f'rank{rank}.paramorph-cost.json').read_text())
                     for rank in range(world)]
            progress = [json.loads((artifacts / f'rank{rank}.paramorph-communication.json').read_text())
                        for rank in range(world)]
            expected_updates = request['warmup'] + request['measured']
            for rank, payload in enumerate(costs):
                plans = payload['executed_plans']
                if (payload['rank'] != rank or len(plans) != expected_updates
                    or any(not iteration['matches_plan']
                           or iteration['task_order'] != iteration['planned_order'] for iteration in plans)):
                    raise ValueError('真实训练任务顺序没有完整执行ParaMorph计划')
            for payload in progress:
                if len(payload['iterations']) != expected_updates:
                    raise ValueError('真实训练通信记录没有覆盖全部迭代')
            communication_validation = ScheduleValidator().validate_progress([
                iteration for payload in progress for iteration in payload['iterations']])
            record['execution_plan_verified'] = True
            record['communication_progress_verified'] = True
            record['verified_communication_messages'] = communication_validation['matched_messages']
        if request.get('paramorph_numerical_reference', False):
            references = [json.loads((artifacts / f'rank{rank}.paramorph-numerical-reference.json').read_text())
                          for rank in range(world)]
            for rank, reference in enumerate(references):
                if (reference['rank'] != rank or reference['status'] != 'PASS'
                    or reference['source'] != 'same_loaded_weights_and_recorded_c4_inputs'
                    or reference['compared_parameter_elements'] <= 0
                    or not reference['gradients']
                    or not reference.get('full_training_equivalence_certified', False)
                    or len(reference.get('iterations', [])) != request['measured']):
                    raise ValueError('完整训练数值对照没有覆盖全部Rank及正式验证更新')
                for index, iteration in enumerate(reference['iterations']):
                    if (iteration['step'] != request['warmup'] + index
                        or not iteration['full_training_equivalence_certified']
                        or not iteration['boundary_tensors'] or not iteration['optimizer_states']
                        or iteration['forward_initial_state'] != 'restored_same_model_parameters_rng_and_all_registered_buffers'
                        or any(not row['within_tolerance'] for row in iteration['gradients']
                            + iteration['losses'] + iteration['boundary_tensors'] + iteration['optimizer_states']
                            + iteration['model_buffers'])):
                        raise ValueError('Forward、DGRAD、WGRAD或者完整Optimizer状态数值对照未通过')
            record['numerical_equivalence_certified'] = True
            record['numerical_equivalence_scope'] = 'same_weights_inputs_full_region_outputs_dgrad_accumulated_wgrad_optimizer_states_updated_parameters_multiple_updates'
            record['numerical_compared_parameter_elements'] = sum(
                reference['compared_parameter_elements'] for reference in references)
        if request['recompute'] == 'selective-mlp':
            if any(manifest['recompute_granularity'] != 'selective' or manifest['recompute_modules'] != ['mlp']
                or manifest['cp'] != request['cp'] for manifest in manifests):
                raise ValueError('The actual selective recomputation or context parallel setting differs from the request')
            record['actual_mlp_forward_calls'] = sum(sum(row['mlp_forward_calls'].values()) for row in rows)
            record['recomputation_executed'] = True
            if request.get('mlp_recompute_every', 1) == 2:
                if any(manifest['recompute_mlp_every'] != 2
                    or any(identifier % 2 != 0 for identifier in manifest['recomputed_mlp_layer_ids'])
                    for manifest in manifests):
                    raise ValueError('The actual partial MLP layer selection differs from the request')
                record['recomputed_mlp_layer_ids'] = sorted({identifier for manifest in manifests
                    for identifier in manifest['recomputed_mlp_layer_ids']})
        if request['name'] == 'Falcon-H1-7B':
            record['actual_attention_backend_calls'] = [row.get('attention_backend_calls', {}) for row in rows]
            if request['recompute'] == 'full':
                calls = [row['falcon_checkpoint_calls'] for row in rows]
                executions = [row['falcon_layer_executions'] for row in rows]
                if any(call <= 0 or execution <= call for call, execution in zip(calls, executions, strict=True)):
                    raise ValueError('Falcon recomputation lacks actual forward and backward execution evidence')
                record.update(checkpoint_forward_calls=sum(calls), actual_layer_executions=sum(executions),
                    recomputation_executed=True)
    archive = directory / 'artifacts.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as package:
        package.add(artifacts, arcname='artifacts')
    record['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    write(directory / 'result.json', record)
    print(json.dumps({key: record[key] for key in ('run_id', 'status', 'returncodes',
        'elapsed_seconds', 'performance_eligible', 'tokens_per_second', 'step_seconds',
        'peak_allocated_gib', 'validation_scope', 'raw_artifacts') if key in record}, ensure_ascii=False), flush=True)
    return record


def main():
    global PODS
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-states', type=Path, nargs='*', default=[])
    parser.add_argument('--mistral-falcon', action='store_true')
    parser.add_argument('--falcon-recompute', action='store_true')
    parser.add_argument('--long-context', action='store_true')
    parser.add_argument('--mistral-partial', action='store_true')
    parser.add_argument('--campaign', default='h30')
    parser.add_argument('--pods', nargs='+', default=list(PODS))
    parser.add_argument('--gpu-lock', type=Path, required=True)
    parser.add_argument('--paper-matrix', action='store_true')
    args = parser.parse_args()
    if len(args.pods) < 2:
        parser.error('--pods or PARAMELD_TRAINER_PODS must name at least two trainer pods')
    PODS = tuple(args.pods)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'state.json', {'status': 'waiting_for_context_trials', 'completed': []})
    while any(not path.exists() or json.loads(path.read_text())['status'] != 'completed' for path in args.after_states):
        time.sleep(5)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    os.environ['KUBECTL_REMOTE_COMMAND_WEBSOCKETS'] = 'true'
    nodes = []
    for index, pod in enumerate(PODS):
        info = json.loads(kubectl('get', 'pod', '-n', 'default', pod, '-o', 'json'))
        network = next(item for item in json.loads(info['metadata']['annotations']['k8s.v1.cni.cncf.io/network-status']) if item['interface'] == 'net1')
        nodes.append({'pod': pod, 'index': index, 'ib': network['device-info']['pci']['rdma-device'], 'ip': network['ips'][0]})
    manifest = json.loads(kubectl('exec', '-n', 'default', PODS[0], '-c', 'trainer', '--', 'cat',
        '/workspace/paper-repros-20261003/context-batch-manifest.json'))
    models = {item['id']: item for item in manifest['models']}
    cases = [('nemotronh4', 2, 4, 1, vpp, sequence, False) for vpp in (2, 3) for sequence in (4096, 8192)]
    cases += [('falconh1', 1, pp, 1, 1, sequence, False) for pp in (4, 8) for sequence in (1024, 2048, 4096)]
    cases += [('lightning30', 2, 4, 2, 2, 1024, False), ('lightning30', 2, 4, 2, 2, 1024, True)]
    if args.mistral_falcon:
        cases = [('mistralnemo12', 4, 4, 1, 2, 4096, False),
            ('falconh1', 1, 8, 1, 1, 2048, False), ('falconh1', 1, 8, 1, 1, 4096, False),
            ('mistralnemo12', 4, 4, 1, 2, 8192, False), ('mistralnemo12', 2, 8, 1, 2, 4096, False)]
    if args.falcon_recompute:
        cases = [('falconh1', 1, 8, 1, 1, sequence, False) for sequence in (2048, 4096)]
    cases = [(model_id, tp, 1, pp, ep, vpp, sequence, offload,
        'full' if args.falcon_recompute else 'none')
        for model_id, tp, pp, ep, vpp, sequence, offload in cases]
    if args.long_context:
        cases = [
            ('falconh1', 1, 1, 8, 1, 1, 8192, False, 'full'),
            ('falconh1', 1, 1, 4, 1, 1, 4096, False, 'full'),
            ('falconh1', 1, 1, 4, 1, 1, 8192, False, 'full'),
            ('mistralnemo12', 4, 1, 4, 1, 2, 8192, False, 'selective-mlp'),
            ('mistralnemo12', 2, 2, 4, 1, 2, 8192, False, 'selective-mlp'),
            ('nemotronh4', 2, 1, 4, 1, 1, 8192, False, 'none'),
        ]
    if args.mistral_partial:
        cases = [('mistralnemo12', 4, 1, 4, 1, 2, 8192, False, 'selective-mlp')]
    if args.paper_matrix:
        parallelism = {
            'mimo7': (2, 1, 4, 1, 2, 'full'),
            'nemotronh4': (2, 1, 4, 1, 2, 'none'),
            'falconh1': (1, 1, 8, 1, 1, 'full'),
            'mistralnemo12': (2, 1, 4, 1, 2, 'selective-mlp'),
            'lightning30': (1, 1, 4, 2, 2, 'full'),
            'moonlight16': (1, 1, 4, 2, 2, 'full'),
        }
        cases = [(model_id, *parallelism[model_id][:5], sequence, False,
                  parallelism[model_id][5])
                 for model_id in parallelism for sequence in (2048, 4096, 8192, 16384)]
    records = []
    with args.gpu_lock.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for index, (model_id, tp, cp, pp, ep, vpp, sequence, offload, recompute) in enumerate(cases):
            model = models[model_id]
            request = {key: model[key] for key in ('name', 'model', 'checkpoint', 'data')}
            world = len(nodes) * 8
            request.update(tp=tp, cp=cp, pp=pp, ep=ep, etp=tp, vpp=vpp,
                dp=world // (tp * cp * pp),
                expert_dp=world // (tp * ep * pp), sequence=sequence,
                offload=offload, recompute=recompute, micro_batch=1, global_batch=16,
                warmup=5 if args.paper_matrix else 1,
                measured=10 if args.paper_matrix else 30,
                master=nodes[0]['ip'], port=32500 + index,
                node_count=len(nodes), gpus_per_node=8,
                rank_mapping=list(range(world)), group=pp if vpp > 1 else 1,
                optimizer_sharding='distributed', execution_backend='megatron_ddp',
                megatron_fsdp_version=1, pipeline_schedule=(
                    'interleaved_1f1b' if vpp > 1 else '1f1b'),
                max_allocation_seconds=1200)
            if args.mistral_partial:
                request['mlp_recompute_every'] = 2
            request['run_id'] = f'frontier-{model_id}-tp{tp}-cp{cp}-pp{pp}-ep{ep}-vpp{vpp}-s{sequence}-off{int(offload)}-20261008-{args.campaign}'
            same = [record for record in records if record['name'] == model['name'] and record['tp'] == tp
                and record['cp'] == cp and record['pp'] == pp and record['vpp'] == vpp
                and record['offload'] == offload and record['recompute'] == recompute]
            if same and same[-1]['status'] != 'PASS' and sequence > same[-1]['sequence']:
                record = request | {'status': 'SKIPPED', 'performance_eligible': False, 'reason': 'The shorter context did not complete'}
            elif offload and records[-1]['status'] != 'CUDA_OOM':
                record = request | {'status': 'SKIPPED', 'performance_eligible': False, 'reason': 'CPU offload is unnecessary or the failure is unrelated to memory'}
            else:
                write(args.output / 'state.json', {'status': 'running', 'run_id': request['run_id'], 'completed': records})
                record = execute(request, args.output, nodes)
            records.append(record)
            write(args.output / 'results.json', records)
    write(args.output / 'state.json', {'status': 'completed', 'completed': records})


if __name__ == '__main__':
    main()
