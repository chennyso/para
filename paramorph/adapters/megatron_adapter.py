import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from dataclasses import asdict
from core_data_models import MemoryPolicy
from graph.capability_registry import tensor_parallel_limit

class MegatronAdapter:
    def __init__(self, config):
        self.config = config
        self.output = Path(config.output_directory)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
        import run_hybrid_performance
        self.runner = run_hybrid_performance
        self.runner.PODS = tuple(config.pods)
        self.nodes = []

    def preflight(self):
        from hybrid_node import launch_geometry
        if self.config.cluster_context != 'prd' or self.config.namespace != 'default':
            raise ValueError('现有Megatron训练入口需要prd/default集群环境')
        self.nodes = []
        request = self.config.request
        workers, devices_per_worker = launch_geometry(request)
        if len(self.config.pods) != workers or len(set(self.config.pods)) != workers:
            raise ValueError('训练Pod需要覆盖不同的Worker')
        def read_pod(pod):
            return json.loads(self.runner.kubectl('get', 'pod', '-n', self.config.namespace, pod, '-o', 'json'))
        with ThreadPoolExecutor(max_workers=min(8, workers)) as executor:
            infos = list(executor.map(read_pod, self.config.pods))
        for index, (pod, info) in enumerate(zip(self.config.pods, infos, strict=True)):
            if info['status']['phase'] != 'Running':
                raise ValueError(f'训练Pod未运行：{pod}')
            if info['spec']['nodeName'] not in ('g5', 'g6'):
                raise ValueError(f'训练节点超出授权范围：{info["spec"]["nodeName"]}')
            trainer = next(container for container in info['spec']['containers']
                           if container['name'] == 'trainer')
            if int(trainer['resources']['limits']['nvidia.com/gpu']) != devices_per_worker:
                raise ValueError('训练Pod的实际GPU数量与Worker配置不同')
            if workers == 1:
                ip, ib = '127.0.0.1', ''
            else:
                network = next(row for row in json.loads(info['metadata']['annotations']['k8s.v1.cni.cncf.io/network-status']) if row['interface'] == 'net1')
                ip, ib = network['ips'][0], network['device-info']['pci']['rdma-device']
            self.nodes.append(dict(pod=pod, node=info['spec']['nodeName'], index=index,
                uid=info['metadata']['uid'],
                ib=ib, ip=ip,
                resources=trainer['resources']))
        per_node = request['gpus_per_node'] // devices_per_worker
        groups = [self.nodes[start:start + per_node] for start in range(0, workers, per_node)]
        if (any(len({worker['node'] for worker in group}) != 1 for group in groups)
                or len({group[0]['node'] for group in groups}) != request['node_count']):
            raise ValueError('同一物理节点的Worker必须占用连续Rank且覆盖规定的GPU数量')
        return self.nodes

    def discover_hardware(self):
        from hybrid_node import nccl_environment
        if not self.nodes:
            self.preflight()
        allocation = [{key: node[key] for key in ('pod', 'node', 'uid', 'ib')} for node in self.nodes]
        collector = self.runner.ROOT / 'experiments' / 'profile_topology.py'
        identity = hashlib.sha256(json.dumps(dict(allocation=allocation,
            collector_sha256=hashlib.sha256(collector.read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()
        shared = Path(self.config.gpu_lock_path).parent
        cache = shared / ('allocation-hardware-' + identity + '.json')
        with (shared / 'hardware-discovery.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache.exists():
                document = json.loads(cache.read_text())
                if document['allocation'] != allocation:
                    raise ValueError('硬件记录与当前Pod及IB资源分配不同')
                records = document['hardware']
            else:
                script = self.runner.REMOTE.parent / 'profile_topology.py'
                first = self.nodes[0]['pod']
                self.runner.kubectl('exec', '-n', self.config.namespace, first, '-c', 'trainer',
                    '--', 'mkdir', '-p', script.parent)
                self.runner.kubectl('cp', '-c', 'trainer', collector, f'default/{first}:{script}')
                def collect(node):
                    output = self.runner.REMOTE.parent / ('hardware-' + node['pod'] + '.json')
                    self.runner.kubectl('exec', '-n', self.config.namespace, node['pod'], '-c', 'trainer', '--',
                        '/workspace/runs/downloaded-model-tests-20261005/env/bin/python', script, '--output', output)
                    return json.loads(self.runner.kubectl('exec', '-n', self.config.namespace, node['pod'],
                        '-c', 'trainer', '--', 'cat', output))
                with ThreadPoolExecutor(max_workers=min(8, len(self.nodes))) as executor:
                    records = list(executor.map(collect, self.nodes))
                cache.write_text(json.dumps(dict(allocation=allocation, hardware=records), indent=2, allow_nan=False) + '\n')
        for node, record in zip(self.nodes, records, strict=True):
            record['node'] = node['node']
            record['pod_uid'] = node['uid']
            record['pod_resources'] = node['resources']
            record['launch_environment'] = record['environment'] | nccl_environment(self.config.request, node['ib'])
            record['launch_conditions_source'] = 'actual_node_environment_and_hybrid_node_nccl_environment'
        physical_nodes = self.config.request['node_count']
        if len(records) == physical_nodes:
            return records
        per_node = len(records) // physical_nodes
        merged = []
        for start in range(0, len(records), per_node):
            group = records[start:start + per_node]
            devices, adapters, environments = [], {}, {}
            for index, record in enumerate(group, start=start):
                if len(record['devices']) != 1:
                    raise ValueError('分Pod拓扑采集需要单卡Worker')
                gpu = record['devices'][0]
                devices.append(gpu | dict(index=index - start, nvidia_index=gpu['index'],
                    worker_pod=record['pod'], worker_pod_uid=record['pod_uid']))
                selection = self.nodes[index]['ib']
                selected = [adapter for adapter in record['adapters'] if adapter['name'] == selection]
                if len(selected) != 1:
                    raise ValueError('实际Worker的RDMA设备不在sysfs拓扑中')
                adapters[selection] = selected[0]
                environments[str(index)] = record['launch_environment']
            if len({row['uuid'] for row in devices}) != per_node:
                raise ValueError('物理节点内出现重复分配的GPU')
            merged.append(dict(group[0], hostname=group[0]['node'], devices=devices,
                adapters=list(adapters.values()), environment_per_rank=environments,
                worker_pods=[row['pod'] for row in group],
                pod_resources=dict(scope='per_gpu_worker', worker_count=per_node,
                    worker_resources=[row['pod_resources'] for row in group]),
                topology_source='physical_host_grouped_from_verified_per_gpu_workers'))
        return merged

    def execute(self, request, prediction=None, profile=False, numerical=False, bounded_adam_source=None):
        maximum_tp = tensor_parallel_limit(request)
        if type(request['tp']) is not int or not 1 <= request['tp'] <= maximum_tp:
            raise ValueError(f'实际训练的Tensor并行度必须满足1≤TP≤{maximum_tp}')
        if not self.nodes:
            self.preflight()
        request = dict(request)
        if request.get('memory_policy') is not None:
            request['memory_policy'] = asdict(MemoryPolicy(**request['memory_policy']))
        if (request.get('validation_mode') is None
                and self.config.request.get('max_allocation_seconds') is not None):
            request['max_allocation_seconds'] = min(
                request.get('max_allocation_seconds', self.config.request['max_allocation_seconds']),
                self.config.request['max_allocation_seconds'])
        if request.get('execution_backend') == 'megatron_fsdp':
            request['load_hf_weights'] = True
            request['checkpoint_format'] = 'fsdp_dtensor'
        if request.get('nccl_profile'):
            database = self.output / 'experiments.sqlite'
            if database.is_file():
                from adapters.nccl_batch import batch_nccl_request
                with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
                    profiles = [json.loads(row[0]) for row in connection.execute(
                        'SELECT document FROM profiling_requests')]
                    trials = [json.loads(row[0]) for row in connection.execute(
                        'SELECT document FROM trials')]
                request = batch_nccl_request(request, profiles, trials)
        collection_mode = request.pop('result_collection', None)
        if self.config.run_namespace:
            request['run_id'] = request['run_id'] + '-' + self.config.run_namespace
        request['master'] = self.nodes[0]['ip']
        if prediction is not None:
            request['prediction_gate'] = prediction
        if profile:
            request.update(instrumented=True, paramorph=True, paramorph_calibration=True,
                paramorph_layer_backward_calibration=True, paramorph_link_calibration=True,
                paramorph_optimizer_calibration=True,
                paramorph_operator_calibration=request.get('paramorph_operator_calibration', True))
        if numerical:
            request.update(instrumented=True, paramorph=True, paramorph_numerical_reference=True)
        output = self.output / ('numerical' if numerical else 'profiling' if profile else 'training')
        directory = output / 'runs' / request['run_id']
        if (directory / 'request.json').exists():
            stored = json.loads((directory / 'request.json').read_text())
            compared = dict(request)
            if prediction is not None:
                compared['prediction_gate'] = stored.get('prediction_gate')
            if compared != stored:
                raise ValueError('恢复执行的真实训练配置发生变化')
            request = stored
        sources = {}
        if 'fp16_bounded_adam_sha256' in request:
            if bounded_adam_source is None:
                raise ValueError('本次执行缺少登记的分批FusedAdam源文件')
            bounded_adam_source = Path(bounded_adam_source)
            if hashlib.sha256(bounded_adam_source.read_bytes()).hexdigest() != request['fp16_bounded_adam_sha256']:
                raise ValueError('分批FusedAdam源文件与执行请求的SHA-256不一致')
            sources['bounded_fused_adam.py'] = bounded_adam_source
        elif bounded_adam_source is not None:
            raise ValueError('执行请求未登记分批FusedAdam源文件的SHA-256')
        if request.get('nccl_profile', False):
            sources['nccl_topology_benchmark.py'] = self.runner.ROOT / 'experiments' / 'nccl_topology_benchmark.py'
        if request.get('compute_communication_profile', False):
            sources['compute_communication_overlap_benchmark.py'] = (
                self.runner.ROOT / 'experiments' / 'compute_communication_overlap_benchmark.py')
        if profile or numerical or request.get('paramorph', False) or request.get('paramorph_parameter_ledger', False):
            root = Path(__file__).resolve().parents[1]
            sources.update({'paramorph_runtime.py': root / 'runtime.py',
                'paramorph_communication_calibration.py': root / 'communication_calibration.py'})
            if numerical:
                sources['numerical_reference.py'] = root / 'numerical_reference.py'
            if request.get('paramorph_memory_audit', False):
                sources['memory_audit.py'] = root / 'memory_audit.py'
            if request.get('paramorph_operator_calibration', False):
                sources['operator_profiling.py'] = root / 'operator_profiling.py'
        attempt_id = request['run_id'] + ':' + str(time.time_ns())
        queued_at = time.monotonic()

        def record_allocation_event(event):
            document = dict(event=event, attempt_id=attempt_id, run_id=request['run_id'],
                at_unix_seconds=time.time(), since_queued_seconds=time.monotonic() - queued_at,
                allocated_devices=request['node_count'] * request['gpus_per_node'])
            with (self.output / 'allocation-queue.jsonl').open('a') as stream:
                stream.write(json.dumps(document, allow_nan=False) + '\n')

        with Path(self.config.gpu_lock_path).open('a') as lock:
            record_allocation_event('enqueued')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            record_allocation_event('execution_lock_acquired')
            released = False

            def release_allocation():
                nonlocal released
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                released = True
                record_allocation_event('execution_lock_released')

            try:
                result = self.runner.execute(request, output, self.nodes, extra_sources=sources,
                    collection_mode=collection_mode, release_allocation=release_allocation,
                    resume_existing=(directory / 'request.json').exists() and not (directory / 'result.json').exists())
            finally:
                if not released:
                    release_allocation()
            record_allocation_event('result_collected')
        return result, directory
