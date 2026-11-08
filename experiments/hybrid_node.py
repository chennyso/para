import argparse
import hashlib
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def write_exit(path, record):
    pending = path.with_name(path.name + f'.{os.getpid()}.partial')
    pending.write_text(json.dumps(record) + '\n')
    pending.replace(path)


def launch_geometry(request):
    physical_nodes = request.get('node_count', 2)
    devices_per_node = request.get('gpus_per_node', 8)
    devices_per_worker = request.get('gpus_per_worker', devices_per_node)
    if (type(devices_per_worker) is not int or devices_per_worker < 1
            or devices_per_node % devices_per_worker):
        raise ValueError('每个Worker的GPU数必须整除单台物理节点的GPU数')
    return physical_nodes * devices_per_node // devices_per_worker, devices_per_worker


def nccl_environment(request, ib):
    connections = request.get('cuda_device_max_connections',
        2 if request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp' else 1)
    workers, devices_per_worker = launch_geometry(request)
    single_node = workers == 1
    environment = dict(CUDA_DEVICE_MAX_CONNECTIONS=str(connections),
        NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1' if single_node else '0',
        NCCL_SOCKET_IFNAME='lo' if single_node else 'net1',
        NCCL_IB_GID_INDEX='0', NCCL_IB_TC='136', NCCL_IB_QPS_PER_CONNECTION='4', NCCL_NET_GDR_LEVEL='5',
        NCCL_MIN_NCHANNELS='4', NCCL_IB_HCA=ib, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if devices_per_worker != request.get('gpus_per_node', 8):
        if not ib:
            raise ValueError('分Pod训练要求每个Worker具有实际IB设备')
        environment.update(NCCL_SHM_DISABLE='1', NCCL_NET='IB', NCCL_IB_HCA='=' + ib)
    return environment


def terminate_process_tree(process):
    owned = [process, *process.children(recursive=True)]
    for child in reversed(owned):
        if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.terminate()
    deadline = time.monotonic() + 10
    remaining = [child for child in owned if child.is_running() and child.status() != psutil.STATUS_ZOMBIE]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [child for child in remaining if child.is_running() and child.status() != psutil.STATUS_ZOMBIE]
    for child in remaining:
        child.kill()
    deadline = time.monotonic() + 10
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [child for child in remaining if child.is_running() and child.status() != psutil.STATUS_ZOMBIE]
    if remaining:
        raise RuntimeError('当前实验的训练进程未在限定时间内退出')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'profile_rank':
        output = Path(__file__).resolve().parent.parent
        rank = int(os.environ['RANK'])
        os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
        request = json.loads((output / 'request.json').read_text())
        world = int(os.environ['WORLD_SIZE'])
        selected = request.get('profile_ranks', list(range(world)))
        if len(set(selected)) != len(selected) or not selected or any(value not in range(world) for value in selected):
            raise ValueError('Nsight rank selection must contain distinct actual training ranks')
        if rank not in selected:
            command = ['/workspace/runs/downloaded-model-tests-20261005/env/bin/python',
                str(output / 'code/hybrid_training_entry.py'), *sys.argv[2:]]
            os.execv(command[0], command)
        command = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda,nvtx,nccl',
            '--sample=none', '--cpuctxsw=none', '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop', '--force-overwrite=true',
            '--output=' + str(output / f'rank{rank}.profile'),
            '/workspace/runs/downloaded-model-tests-20261005/env/bin/python',
            str(output / 'code/hybrid_training_entry.py'), *sys.argv[2:]]
        os.execv(command[0], command)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'start', 'run', 'status', 'stop'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--node', type=int, required=True)
    parser.add_argument('--master', required=True)
    parser.add_argument('--ib', required=True)
    args = parser.parse_args()
    output = args.run
    exit_path = output / f'node{args.node}.exit.json'
    pid_path = output / f'node{args.node}.pid'
    helper_path = output / f'node{args.node}.helper.pid'
    cache = Path('/workspace/frontier/hybrid-cache') / os.environ['HOSTNAME']
    if args.action == 'preflight':
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / 'training.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        rows = [dict(index=int(row[0]), uuid=row[1], used_mib=float(row[2]), total_mib=float(row[3]))
            for row in csv.reader(devices.stdout.splitlines(), skipinitialspace=True)]
        apps = [dict(gpu_uuid=row[0], pid=int(row[1]), used_memory_mib=row[2])
            for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True)]
        _, expected = launch_geometry(json.loads((output / 'request.json').read_text()))
        state = dict(node=args.node, gpu_count=len(rows), devices=rows, compute_processes=apps,
            idle_memory_limit_mib=1024, checked_at_unix_seconds=time.time())
        (output / f'node{args.node}.cuda-preflight.json').write_text(json.dumps(state, indent=2) + '\n')
        if len(rows) != expected or len({row['uuid'] for row in rows}) != expected:
            raise ValueError('CUDA初始检查的实际GPU数量或UUID不符合申请')
        if apps or any(row['used_mib'] > 1024 for row in rows):
            raise RuntimeError(dict(status='CUDA_RESOURCES_OCCUPIED', **state))
        print(json.dumps(dict(status='available', **state)))
        return
    if args.action == 'start':
        launch_lock = output / f'node{args.node}.launch.lock'
        with launch_lock.open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if any(path.exists() for path in (helper_path, pid_path, exit_path)):
                print(json.dumps({'status': 'already_started'}))
                return
            with (output / f'node{args.node}.log').open('wb') as log:
                helper = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run', '--run', str(output),
                    '--node', str(args.node), '--master', args.master, '--ib', args.ib],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            helper_path.write_text(str(helper.pid))
            print(json.dumps({'status': 'started'}))
        return
    if args.action == 'status':
        log_path = output / f'node{args.node}.log'
        if not exit_path.exists() and log_path.exists():
            log_text = log_path.read_text(errors='replace')
            if 'torch.OutOfMemoryError:' in log_text or 'torch.cuda.OutOfMemoryError:' in log_text:
                print(json.dumps({'status': 'running', 'returncode': 1,
                    'reason': '训练日志已经记录CUDA显存不足，需要终止本次分布式运行'}))
                return
        if not exit_path.exists() and helper_path.exists():
            helper_pid = int(helper_path.read_text())
            if not psutil.pid_exists(helper_pid) or psutil.Process(helper_pid).status() == psutil.STATUS_ZOMBIE:
                write_exit(exit_path, {'status': 'failed', 'returncode': 1,
                    'reason': '远端训练启动进程已经退出，未产生完成记录'})
        print(exit_path.read_text() if exit_path.exists() else json.dumps({'status': 'running'}))
        return
    if args.action == 'stop':
        if exit_path.exists() or (not pid_path.exists() and not helper_path.exists()):
            print(json.dumps({'status': 'inactive'}))
            return
        pid = int((pid_path if pid_path.exists() else helper_path).read_text())
        if not psutil.pid_exists(pid):
            print(json.dumps({'status': 'inactive'}))
            return
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            print(json.dumps({'status': 'inactive'}))
            return
        command = process.cmdline()
        if (str(output / 'code/hybrid_training_entry.py') not in command
                and str(output / 'code/pp_vpp_communication_test.py') not in command
                and str(output / 'code/nccl_topology_benchmark.py') not in command
                and not (str(output / 'code/hybrid_node.py') in command
                         and ('profile_rank' in command or 'run' in command))):
            raise ValueError('The stop target does not belong to this hybrid trial')
        terminate_process_tree(process)
        write_exit(exit_path, {'status': 'failed', 'returncode': -15,
                              'reason': '当前实验已经终止'})
        print(json.dumps({'status': 'terminated'}))
        return
    request = json.loads((output / 'request.json').read_text())
    worker_count, devices_per_worker = launch_geometry(request)
    for name in ('extensions', 'inductor', 'triton', 'temporary'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    environment['MAX_JOBS'] = '2'
    environment.update(nccl_environment(request, args.ib))
    if devices_per_worker != request.get('gpus_per_node', 8):
        environment['PARAMELD_PHYSICAL_NODE'] = os.environ['POD_NODE_NAME']
    environment.update(PYTHONPATH='/workspace/overlays/qwen3-convert-20260920:/workspace/tools/transformers-v5.15.0/src:/workspace/tools/Megatron-Bridge-86bab368/src:/workspace/tools/Megatron-Bridge-86bab368/3rdparty/Megatron-LM',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
        TMPDIR=str(cache / 'temporary'), TORCH_EXTENSIONS_DIR=str(cache / 'extensions'),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
        TRITON_CACHE_AUTOTUNING='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if request.get('training_data_contract', {}).get('workers', 0) > 0:
        # 数据加载进程通过Unix套接字传输张量，地址必须短于系统上限。
        identifier = hashlib.sha256(f'{output}:{args.node}'.encode()).hexdigest()[:16]
        socket_directory = Path('/dev/shm') / f'pm-{identifier}'
        socket_directory.mkdir(mode=0o700, exist_ok=True)
        environment['TMPDIR'] = str(socket_directory)
    if request.get('pinned_host_allocator') is not None:
        if request['pinned_host_allocator'] != 'cuda_host_register':
            raise ValueError('当前执行后端仅支持显式选择cuda_host_register锁页路径')
        register_threads = request['pinned_host_register_threads']
        if type(register_threads) is not int or not 1 <= register_threads <= 4:
            raise ValueError('锁页预映射线程数必须符合Worker的四核CPU上限')
        options = ('expandable_segments:True,pinned_use_cuda_host_register:True,'
                   f'pinned_num_register_threads:{register_threads}')
        environment['PYTORCH_ALLOC_CONF'] = environment['PYTORCH_CUDA_ALLOC_CONF'] = options
        print(f'OPTIMIZER_HOST_ALLOCATOR {options}', flush=True)
    for key in tuple(environment):
        if key.startswith('NCCL_HOOK_') or key == 'LD_PRELOAD':
            del environment[key]
    if request.get('paramorph_numerical_reference', False):
        environment['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    command = ['/workspace/runs/downloaded-model-tests-20261005/env/bin/python', '-m', 'torch.distributed.run',
        f'--nnodes={worker_count}', f'--nproc-per-node={devices_per_worker}', f'--node_rank={args.node}', f'--master_addr={args.master}',
        f'--master_port={request["port"]}', '--rdzv_backend=static', str(output / 'code/hybrid_training_entry.py'),
        '--model', request['model'], '--checkpoint', request['checkpoint'], '--data', request['data'],
        '--output', str(output), '--tp', str(request['tp']), '--pp', str(request['pp']), '--ep', str(request['ep']),
        '--cp', str(request['cp']), '--vpp', str(request['vpp']), '--group', str(request.get('group', request['pp'])),
        '--sequence', str(request['sequence']), '--micro-batch', str(request['micro_batch']),
        '--global-batch', str(request['global_batch']),
        '--warmup', str(request['warmup']), '--measured', str(request['measured'])]
    if request['offload']:
        command.append('--cpu-offload')
    if request.get('load_hf_weights', False):
        command.append('--load-hf-weights')
    if request.get('hf_fused_mamba_norm', False):
        command.append('--hf-fused-mamba-norm')
    if request.get('diagnose', False):
        command.append('--diagnose')
    if request.get('deterministic_embedding', False):
        command.append('--deterministic-embedding')
    if request['recompute'] == 'full':
        command.append('--recompute')
    if request['recompute'] == 'selective-mlp':
        command.append('--recompute-mlp')
        command.extend(['--recompute-mlp-every', str(request.get('mlp_recompute_every', 1))])
    if request['name'] == 'Falcon-H1-7B':
        command.append('--unfused-bda')
    if request.get('profile', False):
        position = command.index(str(output / 'code/hybrid_training_entry.py'))
        command = [*command[:position], '--no-python',
            '/workspace/runs/downloaded-model-tests-20261005/env/bin/python',
            str(output / 'code/hybrid_node.py'), 'profile_rank', *command[position + 1:]]
    if request.get('communication_test', False):
        if devices_per_worker != request.get('gpus_per_node', 8):
            raise ValueError('旧流水通信微基准需要完整节点Worker；分Pod使用NCCL实际进程组测量')
        node_count = 1 if request.get('communication_single_node', False) else 2
        command = ['/workspace/runs/downloaded-model-tests-20261005/env/bin/python', '-m',
            'torch.distributed.run', f'--nnodes={node_count}', f'--nproc-per-node={request["pp"] // node_count}',
            f'--node_rank={args.node}', f'--master_addr={args.master}',
            f'--master_port={request["port"]}', '--rdzv_backend=static',
            str(output / 'code/pp_vpp_communication_test.py'), '--output', str(output)]
    if request.get('nccl_profile', False):
        environment.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,NET',
                           NCCL_DEBUG_FILE=str(output / 'nccl-%h-%p.log'))
        command = ['/workspace/runs/downloaded-model-tests-20261005/env/bin/python', '-m',
            'torch.distributed.run', f'--nnodes={worker_count}',
            f'--nproc-per-node={devices_per_worker}', f'--node_rank={args.node}',
            f'--master_addr={args.master}', f'--master_port={request["port"]}', '--rdzv_backend=static',
            str(output / 'code/nccl_topology_benchmark.py'), '--output', str(output / 'nccl-costs.json'),
            '--group-sizes', ','.join(map(str, request['nccl_group_sizes'])),
            '--warmup', str(request['warmup']), '--iterations', str(request['measured']),
            '--dtype', request.get('nccl_dtype', 'bfloat16')]
        if request.get('nccl_payloads_bytes'):
            command.extend(['--payloads-bytes', ','.join(map(str, request['nccl_payloads_bytes']))])
        else:
            command.extend(['--payloads-mib', ','.join(map(str, request['nccl_payloads_mib']))])
    if request.get('compute_communication_profile', False):
        command = ['/workspace/runs/downloaded-model-tests-20261005/env/bin/python', '-m',
            'torch.distributed.run', f'--nnodes={worker_count}',
            f'--nproc-per-node={devices_per_worker}', f'--node_rank={args.node}',
            f'--master_addr={args.master}', f'--master_port={request["port"]}', '--rdzv_backend=static',
            str(output / 'code/compute_communication_overlap_benchmark.py'),
            '--output', str(output / 'compute-communication-overlap.json'),
            '--group-ranks', ','.join(map(str, request['overlap_group_ranks'])),
            '--collective', request['overlap_collective'],
            '--payload-bytes', str(request['overlap_payload_bytes']),
            '--m', str(request['overlap_compute_shape']['M']),
            '--n', str(request['overlap_compute_shape']['N']),
            '--k', str(request['overlap_compute_shape']['K']),
            '--warmup', str(request['warmup']), '--iterations', str(request['measured'])]
    if request.get('nccl_profile', False) and (
            request.get('nccl_rank_groups') or request.get('nccl_payloads_bytes')):
        command.extend(['--rank-groups-json', str(output / 'request.json')])
    with (cache / 'training.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(command, env=environment)
        pid_path.write_text(str(process.pid))
        timed_out = False
        memory_limit_path = Path('/sys/fs/cgroup/memory.max')
        memory_current_path = Path('/sys/fs/cgroup/memory.current')
        memory_stat_path = Path('/sys/fs/cgroup/memory.stat')
        memory_limit_text = memory_limit_path.read_text().strip()
        memory_limit = int(memory_limit_text) if memory_limit_text != 'max' else None
        memory_budget = (memory_limit - max(16 * 1024**3, memory_limit // 20)
            if memory_limit is not None else None)
        raw_memory_budget = (memory_limit - max(4 * 1024**3, memory_limit // 20)
            if memory_limit is not None else None)
        if 'host_working_set_budget_bytes' in request:
            requested_budget = request['host_working_set_budget_bytes']
            if (type(requested_budget) is not int or memory_budget is None or raw_memory_budget is None
                    or not memory_budget <= requested_budget < raw_memory_budget):
                raise ValueError('指定的主机工作集阈值必须低于容器总量保护值')
            memory_budget = requested_budget
        memory_peak = 0
        working_set_peak = 0
        host_memory_failure = None
        deadline = (time.monotonic() + request['max_allocation_seconds']
            if request.get('max_allocation_seconds') is not None else float('inf'))
        while process.poll() is None:
            memory_current = int(memory_current_path.read_text())
            stat = {name: int(value) for name, value in
                (line.split() for line in memory_stat_path.read_text().splitlines())}
            inactive_file_bytes = min(memory_current, stat['inactive_file'],
                max(0, stat['file'] - stat['shmem']))
            working_set_bytes = memory_current - inactive_file_bytes
            node_available = psutil.virtual_memory().available
            memory_peak = max(memory_peak, memory_current)
            working_set_peak = max(working_set_peak, working_set_bytes)
            pressure = ('physical_node_headroom' if node_available < 32 * 1024**3 else
                'container_raw_memory' if raw_memory_budget is not None and memory_current >= raw_memory_budget else
                'worker_working_set' if memory_budget is not None and working_set_bytes >= memory_budget else None)
            if pressure is not None:
                host_memory_failure = dict(status='HOST_MEMORY_BUDGET_EXCEEDED',
                    observed_bytes=memory_current, peak_observed_bytes=memory_peak,
                    budget_bytes=memory_budget, container_limit_bytes=memory_limit,
                    source='actual_cgroup_v2_memory.current', includes_page_cache=True,
                    pressure=pressure, working_set_bytes=working_set_bytes,
                    working_set_peak_bytes=working_set_peak, reclaimable_inactive_file_bytes=inactive_file_bytes,
                    container_raw_budget_bytes=raw_memory_budget,
                    node_available_bytes=node_available, minimum_node_available_bytes=32 * 1024**3,
                    timestamp=time.time())
                write_exit(output / f'node{args.node}.host-memory.json', host_memory_failure)
                terminate_process_tree(psutil.Process(process.pid))
                break
            if time.monotonic() >= deadline:
                timed_out = True
                terminate_process_tree(psutil.Process(process.pid))
                break
            time.sleep(0.1)
        code = -15 if timed_out or host_memory_failure else process.wait()
    write_exit(exit_path, {'status': 'complete' if code == 0 else 'failed', 'returncode': code,
        'host_memory': host_memory_failure, 'host_memory_peak_observed_bytes': memory_peak,
        'host_memory_working_set_peak_bytes': working_set_peak, 'execution_deadline_reached': timed_out})


if __name__ == '__main__':
    main()
