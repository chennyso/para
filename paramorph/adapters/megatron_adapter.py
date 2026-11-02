import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
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
        if self.config.cluster_context != 'prd' or self.config.namespace != 'default':
            raise ValueError('现有Megatron训练入口需要prd/default集群环境')
        self.nodes = []
        seen_nodes = set()
        for index, pod in enumerate(self.config.pods):
            info = json.loads(self.runner.kubectl('get', 'pod', '-n', self.config.namespace, pod, '-o', 'json'))
            if info['status']['phase'] != 'Running':
                raise ValueError(f'训练Pod未运行：{pod}')
            if info['spec']['nodeName'] not in ('g5', 'g6'):
                raise ValueError(f'训练节点超出授权范围：{info["spec"]["nodeName"]}')
            if info['spec']['nodeName'] in seen_nodes:
                raise ValueError('同一物理节点不能重复登记为多个GPU分配')
            seen_nodes.add(info['spec']['nodeName'])
            trainer = next(container for container in info['spec']['containers']
                           if container['name'] == 'trainer')
            if int(trainer['resources']['limits']['nvidia.com/gpu']) != 8:
                raise ValueError('训练Pod必须使用节点的八张实际GPU')
            network = next(row for row in json.loads(info['metadata']['annotations']['k8s.v1.cni.cncf.io/network-status']) if row['interface'] == 'net1')
            self.nodes.append(dict(pod=pod, node=info['spec']['nodeName'], index=index,
                uid=info['metadata']['uid'],
                ib=network['device-info']['pci']['rdma-device'], ip=network['ips'][0],
                resources=trainer['resources']))
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
                records = []
                for node in self.nodes:
                    script = self.runner.REMOTE.parent / 'profile_topology.py'
                    output = self.runner.REMOTE.parent / ('hardware-' + node['pod'] + '.json')
                    self.runner.kubectl('exec', '-n', self.config.namespace, node['pod'], '-c', 'trainer',
                        '--', 'mkdir', '-p', script.parent)
                    self.runner.kubectl('cp', '-c', 'trainer', collector,
                        f'default/{node["pod"]}:{script}')
                    self.runner.kubectl('exec', '-n', self.config.namespace, node['pod'], '-c', 'trainer', '--',
                        '/workspace/runs/downloaded-model-tests-20261005/env/bin/python', script, '--output', output)
                    records.append(json.loads(self.runner.kubectl('exec', '-n', self.config.namespace, node['pod'],
                        '-c', 'trainer', '--', 'cat', output)))
                cache.write_text(json.dumps(dict(allocation=allocation, hardware=records), indent=2, allow_nan=False) + '\n')
        for node, record in zip(self.nodes, records, strict=True):
            record['node'] = node['node']
            record['pod_uid'] = node['uid']
            record['pod_resources'] = node['resources']
            record['launch_environment'] = record['environment'] | nccl_environment(self.config.request, node['ib'])
            record['launch_conditions_source'] = 'actual_node_environment_and_hybrid_node_nccl_environment'
        return records

    def execute(self, request, prediction=None, profile=False, numerical=False):
        maximum_tp = tensor_parallel_limit(request)
        if type(request['tp']) is not int or not 1 <= request['tp'] <= maximum_tp:
            raise ValueError(f'实际训练的Tensor并行度必须满足1≤TP≤{maximum_tp}')
        if not self.nodes:
            self.preflight()
        request = dict(request)
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
