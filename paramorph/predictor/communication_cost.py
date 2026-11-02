import math
import statistics
from copy import deepcopy
from uuid import UUID

import numpy as np

from cost.cost_model import MissingCalibration
from contracts import fingerprint
from predictor.physical_paths import communication_resources
from predictor.profile_database import resolved_cuda_connections


PAYLOAD_DEFINITIONS = {
    'p2p_send_recv': '发送端传输到接收端的一份Tensor字节数',
    'all_reduce': '每个Rank完整输入及输出Tensor的字节数',
    'all_gather': '每个Rank本地输入Tensor的字节数；完整输出为参与者数量乘以该值',
    'reduce_scatter': '每个Rank输出分片的字节数；完整输入为参与者数量乘以该值',
    'all_to_all': '均匀交换时每个源Rank发送到每个目的Rank的分片字节数',
}


def bind_candidate_execution_conditions(topology, request):
    result = deepcopy(topology)
    value = resolved_cuda_connections(request)
    if not isinstance(result, dict):
        raise ValueError('候选运行环境需要包含具体设备和环境的通信拓扑')
    # 查询身份使用训练启动器实际采用的后端默认值。
    result['environment'] = dict(result.get('environment', {}), CUDA_DEVICE_MAX_CONNECTIONS=str(value))
    for scope in ('environment_per_node', 'environment_per_rank'):
        if scope in result:
            result[scope] = {key: dict(environment, CUDA_DEVICE_MAX_CONNECTIONS=str(value))
                             for key, environment in result[scope].items()}
    return result


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}需要有限数值')
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name}超出合法范围')
    return float(value)


def _matrix(specification, ranks):
    matrix = specification.get('send_bytes_matrix') if specification else None
    if matrix is None:
        return None
    if len(matrix) != len(ranks) or any(len(row) != len(ranks) for row in matrix):
        raise ValueError('All-to-All发送字节矩阵必须按照physical_ranks提供完整方阵')
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError('All-to-All发送字节矩阵需要非负整数字节数')
    return matrix


def _measured_distribution_matches(row, collective, payload_bytes, matrix):
    if collective != 'all_to_all' or matrix is None:
        return True
    measured_matrix = row.get('send_bytes_matrix')
    if measured_matrix is not None:
        return measured_matrix == matrix
    return all(value == payload_bytes for source, values in enumerate(matrix)
               for destination, value in enumerate(values) if source != destination)


class CommunicationCurve:
    def __init__(self, benchmark, collective, ranks, rows=None):
        self.benchmark = benchmark
        self.collective = collective
        self.ranks = tuple(ranks)
        self.rows = rows if rows is not None else [
            row for row in benchmark['records']
            if row.get('collective', row.get('kind')) == collective and tuple(row['ranks']) == self.ranks
        ]
        if not self.rows:
            raise MissingCalibration('通信曲线缺少对应操作与实际进程组的测量')
        grouped = {}
        for row in self.rows:
            samples = row['samples']
            if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError('通信曲线缺少合法实际计时')
            unit = row.get('sample_unit', benchmark.get('sample_unit', 'microseconds'))
            if unit not in ('microseconds', 'milliseconds'):
                raise ValueError('通信计时单位必须明确为microseconds或milliseconds')
            scale = 1000 if unit == 'microseconds' else 1
            grouped.setdefault(row['payload_bytes'], []).extend(value / scale for value in samples)
        self.sizes = sorted(grouped)
        self.values = [statistics.median(grouped[size]) for size in self.sizes]
        self.dispersion = max(statistics.pstdev(values) for values in grouped.values())

    def estimate(self, payload_bytes, benchmark_identity):
        if benchmark_identity != self.benchmark['hardware_fingerprint']:
            raise ValueError('通信预测设备及软件环境与实测曲线不一致')
        if not self.sizes[0] <= payload_bytes <= self.sizes[-1]:
            raise MissingCalibration('通信消息大小超出实测范围，需要补充测量')
        exact = payload_bytes in self.sizes
        return dict(status='MEASURED_EXACT' if exact else 'INTERPOLATED',
            milliseconds=float(np.interp(payload_bytes, self.sizes, self.values)),
            uncertainty_ms=self.dispersion, source='measured_piecewise_latency_curve',
            uncertainty_scope='observed_sample_dispersion_not_statistical_interval',
            collective=self.collective, ranks=list(self.ranks), sample_count=sum(len(row['samples']) for row in self.rows),
            timing_scopes=sorted({row.get('timing_scope', 'leader_rank_duration') for row in self.rows}),
            payload_definition=self.benchmark.get('payload_definitions', {}).get(
                self.collective, PAYLOAD_DEFINITIONS[self.collective]),
            resource_ids=sorted({resource for row in self.rows for resource in row.get('resource_ids', [])}),
            topology_scope='exact_measured_physical_rank_group_and_hardware_software_identity',
            concurrency_context=self.benchmark.get('concurrency_context', {
                'scheduled_operations': self.benchmark.get('concurrent_operations', []),
                'external_contention': 'unobserved',
            }))

    def extrapolate(self, payload_bytes, benchmark_identity):
        if benchmark_identity != self.benchmark['hardware_fingerprint']:
            raise ValueError('通信预测设备及软件环境与实测曲线不一致')
        if self.sizes[0] <= payload_bytes <= self.sizes[-1]:
            return self.estimate(payload_bytes, benchmark_identity)
        if len(self.sizes) == 1:
            slope = self.values[0] / self.sizes[0]
            boundary_size, boundary_value = self.sizes[0], self.values[0]
        elif payload_bytes < self.sizes[0]:
            slope = max(0.0, (self.values[1] - self.values[0]) /
                (self.sizes[1] - self.sizes[0]))
            boundary_size, boundary_value = self.sizes[0], self.values[0]
        else:
            slope = max(0.0, (self.values[-1] - self.values[-2]) /
                (self.sizes[-1] - self.sizes[-2]))
            boundary_size, boundary_value = self.sizes[-1], self.values[-1]
        milliseconds = max(0.0, boundary_value + slope * (payload_bytes - boundary_size))
        identity = dict(collective=self.collective, physical_ranks=list(self.ranks),
            payload_bytes=payload_bytes, measured_payload_range=[self.sizes[0], self.sizes[-1]])
        return dict(status='ANALYTICAL_CURVE_EXTRAPOLATED', milliseconds=milliseconds,
            uncertainty_ms=None, source='measured_piecewise_curve_boundary_extrapolation',
            collective=self.collective, ranks=list(self.ranks), sample_count=sum(
                len(row['samples']) for row in self.rows),
            payload_definition=self.benchmark.get('payload_definitions', {}).get(
                self.collective, PAYLOAD_DEFINITIONS[self.collective]),
            resource_ids=sorted({resource for row in self.rows for resource in row.get('resource_ids', [])}),
            topology_scope='measured_group_with_payload_outside_measured_range',
            measured_payload_range=[self.sizes[0], self.sizes[-1]],
            assumptions=['使用最近实测分段的非负局部斜率进行消息范围外推',
                '外推条件需要进入排序敏感性与针对性NCCL Profiling'],
            profiling_request=dict(kind='nccl', cost_key=fingerprint(identity), identity=identity))


class TopologyAwareNCCLCostSurface:
    def __init__(self, database):
        self.database = database

    @staticmethod
    def execution_conditions(environment):
        ignored = {'NCCL_DEBUG', 'NCCL_DEBUG_FILE', 'NCCL_DEBUG_SUBSYS', 'NCCL_VERSION'}
        cuda_runtime = {'CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_ORDER',
                        'CUDA_MODULE_LOADING', 'CUDA_DRIVER_VERSION'}
        return {key: str(value) for key, value in environment.items()
                if (key.startswith('NCCL_') and key not in ignored) or key in cuda_runtime or key.startswith('UCX_')}

    @staticmethod
    def _devices(topology):
        devices = {}
        for row in topology['devices']:
            rank = row.get('physical_rank', row.get('rank'))
            identifier = row.get('uuid', row.get('gpu_uuid'))
            if type(rank) is not int or rank < 0 or rank in devices or not identifier:
                raise ValueError('通信设备身份需要明确且唯一的物理Rank和真实GPU UUID')
            identifier = str(UUID(str(identifier).removeprefix('GPU-')))
            devices[rank] = dict(uuid=identifier, node=row.get('node', topology.get('hostname')),
                node_index=row.get('node_index'),
                numa_node=row.get('numa_node'),
                pci_bus_id=str(row['pci_bus_id']).lower() if row.get('pci_bus_id') else None,
                pci_path=row.get('pci_path'))
        if len({row['uuid'] for row in devices.values()}) != len(devices):
            raise ValueError('同一真实GPU UUID不能对应多个物理设备')
        return devices

    @staticmethod
    def _software(software):
        if isinstance(software, list):
            if not software or any(row != software[0] for row in software):
                return None
            software = software[0]
        if not isinstance(software, dict) or any(key not in software for key in ('torch', 'cuda', 'nccl')):
            return None
        return dict(torch=software['torch'], cuda=software['cuda'],
                    nccl=list(software['nccl']) if isinstance(software['nccl'], (list, tuple)) else software['nccl'])

    @staticmethod
    def _paths(paths):
        if paths is None:
            return None
        if not isinstance(paths, list):
            raise ValueError('通信路径身份需要明确的实际路径记录列表')
        ignored = {'source', 'captured_at', 'observed_at', 'description'}
        return sorted(({key: value for key, value in row.items() if key not in ignored} for row in paths), key=fingerprint)

    @classmethod
    def topology_key(cls, topology):
        devices = cls._devices(topology)
        return fingerprint(dict(devices=[dict(physical_rank=rank, **row) for rank, row in sorted(devices.items())],
            software=cls._software(topology.get('software')), environment=cls.execution_conditions(topology.get('environment', {})),
            software_per_rank={str(key): cls._software(value)
                               for key, value in topology.get('software_per_rank', {}).items()},
            environment_per_node={str(key): cls.execution_conditions(value)
                                  for key, value in topology.get('environment_per_node', {}).items()},
            environment_per_rank={str(key): cls.execution_conditions(value)
                                  for key, value in topology.get('environment_per_rank', {}).items()},
            paths=cls._paths(topology.get('paths'))))

    @classmethod
    def _group_environments(cls, topology, devices, ranks):
        per_rank = topology.get('environment_per_rank')
        if per_rank is not None:
            if any(str(rank) not in per_rank for rank in ranks):
                return None
            return [cls.execution_conditions(per_rank[str(rank)]) for rank in ranks]
        per_node = topology.get('environment_per_node')
        if per_node is not None:
            keys = [str(devices[rank]['node_index']) if devices[rank]['node_index'] is not None
                    else str(devices[rank]['node']) for rank in ranks]
            if any(key not in per_node for key in keys):
                return None
            return [cls.execution_conditions(per_node[key]) for key in keys]
        nodes = {row['node'] for row in devices.values()}
        if len(nodes) > 1 and any(devices[rank]['node'] != topology.get('hostname') for rank in ranks):
            return None
        if 'environment' not in topology:
            return None
        return [cls.execution_conditions(topology['environment']) for _ in ranks]

    @classmethod
    def _group_matches(cls, requested, ranks, benchmark, measured_ranks):
        if isinstance(requested, str):
            return requested == benchmark['hardware_fingerprint'] and tuple(ranks) == tuple(measured_ranks)
        requested_devices = cls._devices(requested)
        measured_devices = cls._devices(benchmark)
        if any(rank not in requested_devices for rank in ranks) or any(rank not in measured_devices for rank in measured_ranks):
            return False
        query_group = [requested_devices[rank] for rank in ranks]
        measured_group = [measured_devices[rank] for rank in measured_ranks]
        if [row['uuid'] for row in query_group] != [row['uuid'] for row in measured_group]:
            return False
        software = cls._software(requested.get('software'))
        if software is None or software != cls._software(benchmark.get('software')):
            return False
        measured_software = benchmark.get('software_per_rank')
        if measured_software is not None and any(
                str(rank) not in measured_software or cls._software(measured_software[str(rank)]) != software
                for rank in measured_ranks):
            return False
        environments = cls._group_environments(requested, requested_devices, ranks)
        if environments is None or environments != cls._group_environments(benchmark, measured_devices, measured_ranks):
            return False
        if any(row['node'] is None for row in query_group + measured_group):
            return False
        # 节点名称允许采集工具使用不同标签，实际参与者的同节点关系必须保持一致。
        query_nodes = [[index for index, peer in enumerate(query_group) if peer['node'] == row['node']] for row in query_group]
        measured_nodes = [[index for index, peer in enumerate(measured_group) if peer['node'] == row['node']] for row in measured_group]
        if query_nodes != measured_nodes:
            return False
        for query_device, measured_device in zip(query_group, measured_group):
            for key in ('pci_bus_id', 'pci_path'):
                if measured_device[key] is not None and query_device[key] != measured_device[key]:
                    return False
        measured_paths = cls._paths(benchmark.get('paths'))
        if measured_paths is not None and cls._paths(requested.get('paths')) != measured_paths:
            return False
        return True

    @classmethod
    def _same_topology(cls, first, second):
        if isinstance(first, str) and isinstance(second, str):
            return first == second
        first_key = first if isinstance(first, str) else cls.topology_key(first)
        second_key = second if isinstance(second, str) else cls.topology_key(second)
        return first_key == second_key

    @classmethod
    def _group_topology_class(cls, topology, ranks):
        if isinstance(topology, str):
            return None
        devices = cls._devices(topology)
        if any(rank not in devices or devices[rank]['node'] is None for rank in ranks):
            return None
        nodes = {}
        for rank in ranks:
            row = devices[rank]
            node = row['node_index'] if row['node_index'] is not None else row['node']
            nodes.setdefault(str(node), []).append(row['numa_node'])
        node_shapes = []
        for values in nodes.values():
            numa_counts = {}
            for value in values:
                numa_counts[str(value)] = numa_counts.get(str(value), 0) + 1
            node_shapes.append((len(values), tuple(sorted(numa_counts.values()))))
        return dict(group_size=len(ranks), node_shapes=sorted(node_shapes),
            crosses_nodes=len(nodes) > 1)

    @classmethod
    def _cluster_matches(cls, requested, benchmark):
        if isinstance(requested, str):
            return requested == benchmark.get('hardware_fingerprint')
        requested_devices = cls._devices(requested)
        measured_devices = cls._devices(benchmark)
        requested_uuids = set(row['uuid'] for row in requested_devices.values())
        measured_uuids = set(row['uuid'] for row in measured_devices.values())
        if not measured_uuids or not measured_uuids <= requested_uuids:
            return False
        requested_software = cls._software(requested.get('software'))
        measured_software = cls._software(benchmark.get('software'))
        if requested_software is None or requested_software != measured_software:
            return False
        requested_ranks = sorted(requested_devices)
        measured_ranks = sorted(measured_devices)
        requested_environments = cls._group_environments(requested, requested_devices, requested_ranks)
        measured_environments = cls._group_environments(benchmark, measured_devices, measured_ranks)
        if requested_environments is None or measured_environments is None:
            return False
        requested_by_uuid = {requested_devices[rank]['uuid']: environment
            for rank, environment in zip(requested_ranks, requested_environments)}
        measured_by_uuid = {measured_devices[rank]['uuid']: environment
            for rank, environment in zip(measured_ranks, measured_environments)}
        return all(requested_by_uuid[uuid] == environment
            for uuid, environment in measured_by_uuid.items())

    @classmethod
    def _benchmark_ranks_on_requested(cls, requested, benchmark, measured_ranks):
        if isinstance(requested, str):
            return list(measured_ranks)
        requested_devices = cls._devices(requested)
        measured_devices = cls._devices(benchmark)
        requested_by_uuid = {row['uuid']: rank for rank, row in requested_devices.items()}
        mapped = []
        for rank in measured_ranks:
            if rank not in measured_devices or measured_devices[rank]['uuid'] not in requested_by_uuid:
                return None
            mapped.append(requested_by_uuid[measured_devices[rank]['uuid']])
        return mapped

    @classmethod
    def _measured_topology_class(cls, requested, benchmark, measured_ranks):
        mapped = cls._benchmark_ranks_on_requested(requested, benchmark, measured_ranks)
        if mapped is None:
            return None
        return cls._group_topology_class(requested, mapped)

    @classmethod
    def _compatible_topology_family(cls, requested, requested_ranks, measured, measured_ranks):
        requested_class = cls._group_topology_class(requested, requested_ranks)
        measured_class = cls._measured_topology_class(requested, measured, measured_ranks)
        if requested_class is None or measured_class is None:
            return False
        requested_counts = sorted(row[0] for row in requested_class['node_shapes'])
        measured_counts = sorted(row[0] for row in measured_class['node_shapes'])
        return (len(requested_counts) == len(measured_counts)
                and requested_class['crosses_nodes'] == measured_class['crosses_nodes']
                and len(set(requested_counts)) == 1
                and len(set(measured_counts)) == 1)

    @staticmethod
    def _algorithm_terms(collective, payload_bytes, group_size):
        if group_size < 2:
            raise ValueError('通信解析需要至少两个参与者')
        if collective == 'p2p_send_recv':
            return 1, float(payload_bytes)
        if collective == 'all_reduce':
            return 2 * (group_size - 1), 2 * (group_size - 1) * payload_bytes / group_size
        if collective in ('all_gather', 'reduce_scatter', 'all_to_all'):
            return group_size - 1, float((group_size - 1) * payload_bytes)
        raise ValueError(f'通信解析不支持操作：{collective}')

    @classmethod
    def _ratio_calibrated_estimate(cls, curve, collective, payload_bytes, ranks,
                                   topology_identity, profile_id, measured_ranks):
        measured_rounds = []
        measured_wire_bytes = []
        for size in curve.sizes:
            rounds, wire_bytes = cls._algorithm_terms(collective, size, len(measured_ranks))
            measured_rounds.append(rounds)
            measured_wire_bytes.append(wire_bytes)
        if len(set(measured_wire_bytes)) < 2:
            return None
        slope = max(0.0, (curve.values[-1] - curve.values[0]) /
                    (measured_wire_bytes[-1] - measured_wire_bytes[0]))
        fixed = statistics.median(value - slope * wire_bytes
            for value, wire_bytes in zip(curve.values, measured_wire_bytes))
        latency_per_round_ms = max(0.0, fixed / measured_rounds[0])
        target_rounds, target_wire_bytes = cls._algorithm_terms(
            collective, payload_bytes, len(ranks))
        milliseconds = latency_per_round_ms * target_rounds + slope * target_wire_bytes
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            return None
        fitted = [latency_per_round_ms * rounds + slope * wire_bytes
            for rounds, wire_bytes in zip(measured_rounds, measured_wire_bytes)]
        observed_residual_ms = max(abs(value - estimate)
            for value, estimate in zip(curve.values, fitted))
        topology_class = cls._group_topology_class(topology_identity, ranks)
        identity = dict(collective=collective, payload_bytes=payload_bytes,
            physical_ranks=list(ranks), dtype=curve.benchmark.get('dtype', 'bfloat16'),
            topology_class=topology_class)
        profiling_request = dict(kind='nccl', cost_key=fingerprint(identity), identity=identity,
            reason='实测锚定比例校准尚未经过当前目标通信组独立验证',
            sensitivity_scenario_ms=observed_residual_ms + curve.dispersion,
            estimated_service_time_ms=milliseconds)
        phase_duration = milliseconds / target_rounds
        phase_bytes = target_wire_bytes / target_rounds
        if collective == 'all_reduce':
            phase_kinds = ['reduce_scatter_round'] * (target_rounds // 2) + [
                'all_gather_round'] * (target_rounds // 2)
        else:
            phase_kind = {'all_gather': 'all_gather_round',
                'reduce_scatter': 'reduce_scatter_round',
                'all_to_all': 'pairwise_exchange_round',
                'p2p_send_recv': 'direct_transfer'}[collective]
            phase_kinds = [phase_kind] * target_rounds
        resource_ids = cls._class_resource_ids(topology_identity, ranks)
        communication_phases = [dict(phase_id=f'phase:{index}', kind=kind,
            duration_ms=phase_duration, wire_bytes=phase_bytes,
            resource_ids=resource_ids,
            dependencies=[] if index == 0 else [f'phase:{index - 1}'])
            for index, kind in enumerate(phase_kinds)]
        return dict(status='RATIO_CALIBRATED', milliseconds=milliseconds,
            time_source='RATIO_CALIBRATED', uncertainty='UNCALIBRATED', uncertainty_ms=None,
            uncertainty_scope='observed_anchor_residual_is_not_target_prediction_error_bound',
            observed_anchor_residual_ms=observed_residual_ms,
            observed_sample_dispersion_ms=curve.dispersion,
            collective=collective, ranks=list(ranks), measured_ranks=list(measured_ranks),
            source='measured_anchored_analytical_ratio', evidence_id=profile_id,
            sample_count=sum(len(row['samples']) for row in curve.rows),
            algorithm='direct' if collective == 'p2p_send_recv' else
                'pairwise' if collective == 'all_to_all' else 'ring',
            rounds=target_rounds, algorithm_wire_bytes=target_wire_bytes,
            latency_per_round_ms=latency_per_round_ms,
            milliseconds_per_wire_byte=slope,
            payload_definition=curve.benchmark.get('payload_definitions', {}).get(
                collective, PAYLOAD_DEFINITIONS[collective]),
            topology_class=topology_class,
            topology_scope='same_physical_cluster_software_nccl_environment_and_balanced_node_count',
            calibration_domain=dict(
                measured_payload_range=[curve.sizes[0], curve.sizes[-1]],
                measured_group_size=len(measured_ranks), target_group_size=len(ranks),
                dtype=identity['dtype'], hardware_fingerprint=curve.benchmark['hardware_fingerprint']),
            resource_ids=resource_ids, communication_phases=communication_phases,
            assumptions=['使用相同物理集群、软件、NCCL环境及相同节点数量的实测通信曲线',
                '根据通信算法轮次与逻辑传输字节数将实测锚点投影到目标通信组',
                'DAG资源Replay负责计算共享通信资源竞争，当前数值表示独立操作服务时间'],
            profiling_request=profiling_request, profiling_targets=[profiling_request])

    @classmethod
    def _class_resource_ids(cls, topology, ranks):
        return communication_resources(topology, ranks)['resource_ids']

    @classmethod
    def _network_estimate(cls, collective, payload_bytes, ranks, topology_identity, concurrency, spec):
        required = ('source', 'assumptions', 'algorithm', 'physical_ranks', 'topology_identity', 'links', 'resources')
        missing = [name for name in required if name not in spec]
        if missing:
            raise MissingCalibration(f'解析通信模型缺少显式参数：{missing}')
        if not isinstance(spec['source'], str) or not spec['source'].strip():
            raise ValueError('解析通信参数必须记录来源')
        if not isinstance(spec['assumptions'], list) or any(
                not isinstance(value, str) or not value.strip() for value in spec['assumptions']):
            raise ValueError('解析通信假设必须使用明确的字符串列表')
        if list(ranks) != spec['physical_ranks'] or not cls._same_topology(topology_identity, spec['topology_identity']):
            raise MissingCalibration('解析链路参数与当前物理进程组及拓扑不一致')
        if concurrency and spec.get('concurrency_assumption') != 'exclusive_serialization':
            raise MissingCalibration('并发通信需要明确共享资源串行占用假设')
        algorithm = spec['algorithm']
        expected = 'direct' if collective == 'p2p_send_recv' else 'pairwise' if collective == 'all_to_all' else 'ring'
        if algorithm != expected:
            raise MissingCalibration(f'当前通信解析仅支持{collective}的{expected}模型')
        resources = {}
        for identifier, resource in spec['resources'].items():
            if not isinstance(identifier, str) or not identifier or not resource.get('source'):
                raise ValueError('共享通信资源需要名称和参数来源')
            resources[identifier] = _positive(resource['bandwidth_bytes_per_second'], '资源带宽')
        links = {}
        for link in spec['links']:
            pair = (link['source_rank'], link['destination_rank'])
            if pair[0] not in ranks or pair[1] not in ranks or pair[0] == pair[1] or pair in links:
                raise ValueError('解析通信链路需要唯一且合法的物理端点')
            if not link.get('source') or not link.get('resource_ids'):
                raise MissingCalibration('解析通信链路缺少参数来源或共享资源标识')
            if len(set(link['resource_ids'])) != len(link['resource_ids']):
                raise ValueError('单条物理路径不能重复计数同一个资源')
            if any(identifier not in resources for identifier in link['resource_ids']):
                raise MissingCalibration('解析通信路径引用的共享资源缺少带宽参数')
            links[pair] = dict(
                bandwidth=_positive(link['bandwidth_bytes_per_second'], '路径有效带宽'),
                latency_us=_positive(link['latency_us'], '路径固定延迟', allow_zero=True),
                resource_ids=link['resource_ids'],
            )
        size = len(ranks)
        matrix = _matrix(spec, ranks) if collective == 'all_to_all' else None
        if collective == 'p2p_send_recv':
            if size != 2:
                raise ValueError('P2P解析通信需要两个物理参与者')
            rounds = [[(ranks[0], ranks[1], payload_bytes)]]
        elif collective == 'all_to_all':
            if matrix is None:
                if spec.get('send_distribution') != 'uniform_per_peer':
                    raise MissingCalibration('All-to-All解析需要实际发送字节矩阵或明确均匀分片假设')
                matrix = [[payload_bytes if source != destination else 0 for destination in range(size)]
                          for source in range(size)]
            rounds = [[(ranks[source], ranks[(source + shift) % size], matrix[source][(source + shift) % size])
                       for source in range(size)] for shift in range(1, size)]
        else:
            chunk_bytes = payload_bytes / size if collective == 'all_reduce' else payload_bytes
            round_count = 2 * (size - 1) if collective == 'all_reduce' else size - 1
            rounds = [[(ranks[source], ranks[(source + 1) % size], chunk_bytes)
                       for source in range(size)] for _ in range(round_count)]
        total_seconds = 0.0
        resource_loads = {}
        used_resources = set()
        round_details = []
        for transfers in rounds:
            loads = {}
            latency_seconds = 0.0
            transfer_seconds = 0.0
            for source_rank, destination_rank, amount in transfers:
                if amount == 0:
                    continue
                pair = (source_rank, destination_rank)
                if pair not in links:
                    raise MissingCalibration(f'解析通信缺少实际物理路径：{pair}')
                link = links[pair]
                latency_seconds = max(latency_seconds, link['latency_us'] / 1_000_000)
                transfer_seconds = max(transfer_seconds, amount / link['bandwidth'])
                for identifier in link['resource_ids']:
                    loads[identifier] = loads.get(identifier, 0) + amount
                    resource_loads[identifier] = resource_loads.get(identifier, 0) + amount
                    used_resources.add(identifier)
            resource_seconds = max((amount / resources[identifier] for identifier, amount in loads.items()), default=0)
            duration_seconds = latency_seconds + max(transfer_seconds, resource_seconds)
            total_seconds += duration_seconds
            round_details.append(dict(milliseconds=duration_seconds * 1000, resource_loads_bytes=loads))
        assumptions = list(spec['assumptions']) + [
            f'NCCL通信算法假设为{algorithm}，实际实现可能选择其他通信算法',
            '每个通信轮次使用最大路径耗时和共享资源累计字节估计；各轮次顺序执行',
            '带宽参数需要明确表示对应路径和资源的有效吞吐能力',
        ]
        if concurrency:
            assumptions.append('共享通信资源由DAG事件模拟排他占用；当前数值表示独立操作的服务时间')
        result = dict(
            status='ANALYTICAL', milliseconds=total_seconds * 1000,
            uncertainty_ms=None, uncertainty_scope='explicit_network_assumptions_without_measured_error_bound',
            collective=collective, ranks=list(ranks), source=spec['source'], assumptions=assumptions,
            algorithm=algorithm, rounds=len(rounds), round_details=round_details,
            resource_ids=sorted(used_resources), resource_loads_bytes=resource_loads,
            bytes_scope='algorithm_derived_path_traffic_not_measured_wire_bytes',
            payload_definition=PAYLOAD_DEFINITIONS[collective], concurrent_operations=list(concurrency),
            topology_scope='explicit_directional_paths_and_shared_resource_capacities',
            communication_phases=[dict(phase_id=f'phase:{index}', kind=(
                'reduce_scatter_round' if collective == 'all_reduce' and index < len(rounds) // 2 else
                'all_gather_round' if collective == 'all_reduce' else
                'all_gather_round' if collective == 'all_gather' else
                'reduce_scatter_round' if collective == 'reduce_scatter' else
                'pairwise_exchange_round' if collective == 'all_to_all' else 'direct_transfer'),
                duration_ms=row['milliseconds'], resource_loads_bytes=row['resource_loads_bytes'],
                resource_ids=sorted(row['resource_loads_bytes']),
                dependencies=[] if index == 0 else [f'phase:{index - 1}'])
                for index, row in enumerate(round_details)],
        )
        if matrix is not None:
            result.update(
                send_bytes_matrix=matrix,
                payload_definition='send_bytes_matrix按physical_ranks排列；元素为源Rank发送到目的Rank的字节数',
                remote_send_bytes_per_rank=[sum(value for destination, value in enumerate(row) if source != destination)
                                            for source, row in enumerate(matrix)],
                remote_receive_bytes_per_rank=[sum(matrix[source][destination] for source in range(size)
                                                  if source != destination) for destination in range(size)],
            )
        return result

    def query(self, collective, payload_bytes, ranks, topology_identity, concurrency=(), analytical_spec=None, mode='hybrid', dtype='bfloat16'):
        if mode not in ('hybrid', 'measured_only', 'analytical_only'):
            raise ValueError('通信成本模式需要hybrid、measured_only或analytical_only')
        if type(payload_bytes) is not int or payload_bytes <= 0:
            raise ValueError('NCCL成本查询需要正整数消息字节数')
        if len(ranks) < 2 or len(set(ranks)) != len(ranks) or any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError('NCCL成本查询需要不同的合法物理参与者')
        collective = 'p2p_send_recv' if collective == 'p2p' else collective
        if collective not in PAYLOAD_DEFINITIONS:
            raise ValueError(f'通信成本模型不支持操作：{collective}')
        if collective == 'p2p_send_recv' and len(ranks) != 2:
            raise ValueError('P2P通信成本查询需要两个物理参与者')
        if not isinstance(dtype, str) or not dtype:
            raise ValueError('通信成本查询需要明确Tensor精度')
        matrix = _matrix(analytical_spec, ranks) if collective == 'all_to_all' else None
        matches = []
        topology_class_matches = []
        dtype_transfer_matches = []
        ratio_calibrated_matches = []
        if mode != 'analytical_only':
            for profile in self.database.communication_profiles():
                benchmark = profile['benchmark']
                rows = [row for row in benchmark['records']
                        if row.get('collective', row.get('kind')) == collective
                        and row.get('concurrent_operations',
                            benchmark.get('concurrent_operations', [])) == list(concurrency)
                        and self._group_matches(topology_identity, ranks, benchmark, row['ranks'])
                        and row.get('dtype', benchmark.get('dtype', 'bfloat16')) == dtype
                        and _measured_distribution_matches(row, collective, payload_bytes, matrix)]
                if not rows:
                    continue
                curve = CommunicationCurve(benchmark, collective, ranks, rows)
                estimate = (curve.estimate(payload_bytes, benchmark['hardware_fingerprint'])
                    if curve.sizes[0] <= payload_bytes <= curve.sizes[-1]
                    else curve.extrapolate(payload_bytes, benchmark['hardware_fingerprint']))
                estimate.update(evidence_id=profile['id'], concurrent_operations=list(concurrency))
                estimate['concurrency_context'] = rows[0].get('concurrency_context',
                    benchmark.get('concurrency_context'))
                estimate['source_plan_signature'] = benchmark.get('source_plan_signature')
                estimate['dtype'] = dtype
                estimate['time_source'] = 'MEASURED' if estimate['status'] == 'MEASURED_EXACT' else 'CALIBRATED'
                estimate['uncertainty'] = estimate.get('uncertainty_ms')
                estimate['topology_class'] = self._group_topology_class(topology_identity, ranks)
                estimate['calibration_domain'] = 'exact_ordered_physical_rank_group'
                estimate['measured_ranks'] = list(rows[0]['ranks'])
                estimate['identity_scope'] = dict(gpu_uuid='exact_ordered_physical_devices',
                    node_placement='exact_same_node_relationships', software='exact_torch_cuda_nccl',
                    environment='exact_participant_nccl_ucx_and_cuda_runtime_conditions',
                    paths='exact_recorded_paths' if benchmark.get('paths') is not None else 'not_recorded_in_measurement',
                    pci='checked_when_recorded_in_measurement')
                estimate['communication_phases'] = [dict(phase_id='phase:0',
                    kind='measured_collective_service', duration_ms=estimate['milliseconds'],
                    resource_ids=estimate.get('resource_ids', []), dependencies=[])]
                matches.append(estimate)
                continue
            requested_class = self._group_topology_class(topology_identity, ranks)
            if requested_class is not None:
                for profile in self.database.communication_profiles():
                    benchmark = profile['benchmark']
                    if not self._cluster_matches(topology_identity, benchmark):
                        continue
                    rows = [row for row in benchmark['records']
                        if row.get('collective', row.get('kind')) == collective
                        and row.get('concurrent_operations',
                            benchmark.get('concurrent_operations', [])) == list(concurrency)
                        and self._measured_topology_class(
                            topology_identity, benchmark, row['ranks']) == requested_class
                        and row.get('dtype', benchmark.get('dtype', 'bfloat16')) == dtype
                        and _measured_distribution_matches(row, collective, payload_bytes, matrix)]
                    if not rows:
                        continue
                    curve = CommunicationCurve(benchmark, collective, ranks, rows)
                    estimate = (curve.estimate(payload_bytes, benchmark['hardware_fingerprint'])
                        if curve.sizes[0] <= payload_bytes <= curve.sizes[-1]
                        else curve.extrapolate(payload_bytes, benchmark['hardware_fingerprint']))
                    status = ('MEASURED_TOPOLOGY_CLASS' if payload_bytes in curve.sizes else
                        'INTERPOLATED_TOPOLOGY_CLASS' if curve.sizes[0] <= payload_bytes <= curve.sizes[-1]
                        else 'ANALYTICAL_TOPOLOGY_CLASS_EXTRAPOLATED')
                    estimate.update(status=status,
                        evidence_id=profile['id'], concurrent_operations=list(concurrency),
                        source_plan_signature=benchmark.get('source_plan_signature'),
                        dtype=dtype, measured_ranks=sorted({tuple(row['ranks']) for row in rows}),
                        resource_ids=self._class_resource_ids(topology_identity, ranks),
                        topology_scope='same_collective_group_size_node_distribution_numa_distribution_software_and_dtype',
                        time_source='MEASURED' if status == 'MEASURED_TOPOLOGY_CLASS' else 'CALIBRATED',
                        uncertainty=estimate.get('uncertainty_ms'), topology_class=requested_class,
                        calibration_domain='same_cluster_group_size_node_and_numa_distribution',
                        assumptions=['复用相同节点分布与NUMA分布类别的实测NCCL分段曲线',
                            '动态资源Replay使用候选实际Node类别资源身份计算并发带宽共享'])
                    estimate['concurrency_context'] = rows[0].get('concurrency_context',
                        benchmark.get('concurrency_context'))
                    estimate['communication_phases'] = [dict(phase_id='phase:0',
                        kind='measured_topology_class_collective_service',
                        duration_ms=estimate['milliseconds'],
                        resource_ids=estimate['resource_ids'], dependencies=[])]
                    topology_class_matches.append(estimate)
            if requested_class is not None:
                for profile in self.database.communication_profiles():
                    benchmark = profile['benchmark']
                    if not self._cluster_matches(topology_identity, benchmark):
                        continue
                    rows = [row for row in benchmark['records']
                        if row.get('collective', row.get('kind')) == collective
                        and row.get('concurrent_operations',
                            benchmark.get('concurrent_operations', [])) == list(concurrency)
                        and self._measured_topology_class(
                            topology_identity, benchmark, row['ranks']) == requested_class
                        and row.get('dtype', benchmark.get('dtype', 'bfloat16')) != dtype
                        and _measured_distribution_matches(row, collective, payload_bytes, matrix)]
                    if not rows:
                        continue
                    source_dtype = rows[0].get('dtype', benchmark.get('dtype', 'bfloat16'))
                    if any(row.get('dtype', benchmark.get('dtype', 'bfloat16')) != source_dtype
                           for row in rows):
                        raise ValueError('跨精度通信校准记录包含多个源精度')
                    curve = CommunicationCurve(benchmark, collective, ranks, rows)
                    estimate = (curve.estimate(payload_bytes, benchmark['hardware_fingerprint'])
                        if curve.sizes[0] <= payload_bytes <= curve.sizes[-1]
                        else curve.extrapolate(payload_bytes, benchmark['hardware_fingerprint']))
                    identity = dict(collective=collective, payload_bytes=payload_bytes,
                        physical_ranks=list(ranks), dtype=dtype, topology_class=requested_class)
                    profiling_request = dict(kind='nccl', cost_key=fingerprint(identity), identity=identity,
                        reason='通信传输使用相同字节数的其他精度实测曲线，目标归约精度需要独立验证',
                        sensitivity_scenario_ms=curve.dispersion,
                        estimated_service_time_ms=estimate['milliseconds'])
                    estimate.update(status='ANALYTICAL_DTYPE_TRANSFER_CALIBRATED',
                        time_source='ANALYTICAL', uncertainty='UNCALIBRATED', uncertainty_ms=None,
                        uncertainty_scope='wire_transfer_calibrated_local_reduction_dtype_unvalidated',
                        source='measured_same_collective_bytes_and_topology_with_different_dtype',
                        source_dtype=source_dtype, dtype=dtype, evidence_id=profile['id'],
                        measured_ranks=sorted({tuple(row['ranks']) for row in rows}),
                        resource_ids=self._class_resource_ids(topology_identity, ranks),
                        topology_class=requested_class,
                        topology_scope='same_cluster_group_size_node_numa_distribution_and_collective',
                        calibration_domain=dict(source_dtype=source_dtype, target_dtype=dtype,
                            measured_payload_range=[curve.sizes[0], curve.sizes[-1]],
                            hardware_fingerprint=benchmark['hardware_fingerprint']),
                        assumptions=['物理传输字节数、Collective、通信组拓扑及NCCL环境保持一致',
                            '目标精度的本地归约成本尚未独立校准'],
                        profiling_request=profiling_request, profiling_targets=[profiling_request])
                    estimate['communication_phases'] = [dict(phase_id='phase:0',
                        kind='dtype_transfer_calibrated_collective_service',
                        duration_ms=estimate['milliseconds'],
                        resource_ids=estimate['resource_ids'], dependencies=[])]
                    dtype_transfer_matches.append(estimate)
            for profile in self.database.communication_profiles():
                benchmark = profile['benchmark']
                if not self._cluster_matches(topology_identity, benchmark):
                    continue
                grouped_rows = {}
                for row in benchmark['records']:
                    measured_collective = row.get('collective', row.get('kind'))
                    measured_dtype = row.get('dtype', benchmark.get('dtype', 'bfloat16'))
                    measured_ranks = tuple(row['ranks'])
                    if (measured_collective != collective or measured_dtype != dtype
                            or row.get('concurrent_operations',
                                benchmark.get('concurrent_operations', [])) != list(concurrency)
                            or len(measured_ranks) == len(ranks)
                            or not self._compatible_topology_family(
                                topology_identity, ranks, benchmark, measured_ranks)
                            or not _measured_distribution_matches(
                                row, collective, payload_bytes, matrix)):
                        continue
                    grouped_rows.setdefault(measured_ranks, []).append(row)
                for measured_ranks, rows in grouped_rows.items():
                    curve = CommunicationCurve(benchmark, collective, measured_ranks, rows)
                    estimate = self._ratio_calibrated_estimate(curve, collective, payload_bytes,
                        ranks, topology_identity, profile['id'], measured_ranks)
                    if estimate is not None:
                        estimate.update(dtype=dtype, concurrent_operations=list(concurrency))
                        ratio_calibrated_matches.append(estimate)
        if matches:
            return max(matches, key=lambda row: (row['status'] == 'MEASURED_EXACT', row['sample_count']))
        if topology_class_matches:
            return max(topology_class_matches, key=lambda row: row['sample_count'])
        if dtype_transfer_matches:
            return max(dtype_transfer_matches, key=lambda row: row['sample_count'])
        if ratio_calibrated_matches:
            return max(ratio_calibrated_matches, key=lambda row: (
                row['calibration_domain']['measured_payload_range'][0] <= payload_bytes <=
                    row['calibration_domain']['measured_payload_range'][1],
                -abs(row['calibration_domain']['measured_group_size'] - len(ranks)),
                row['sample_count']))
        reason = '当前操作、消息范围、物理进程组、拓扑或并发条件缺少适用的真实NCCL成本'
        if mode != 'measured_only' and analytical_spec is not None:
            try:
                estimate = self._network_estimate(collective, payload_bytes, ranks, topology_identity, concurrency, analytical_spec)
                identity = dict(collective=collective, payload_bytes=payload_bytes,
                    physical_ranks=list(ranks), topology_identity=topology_identity,
                    concurrent_operations=list(concurrency), dtype=dtype)
                profiling_request = dict(kind='nccl', cost_key=fingerprint(identity), identity=identity,
                    reason='解析通信成本尚未经过当前目标通信组独立验证')
                estimate.update(dtype=dtype, time_source='ANALYTICAL', uncertainty='UNCALIBRATED',
                    topology_class=self._group_topology_class(topology_identity, ranks),
                    calibration_domain='explicit_directional_physical_paths_and_resource_capacities',
                    profiling_request=profiling_request, profiling_targets=[profiling_request])
                return estimate
            except MissingCalibration as missing:
                reason = str(missing)
        elif mode in ('analytical_only', 'hybrid'):
            reason += '；解析评价需要具有来源的实际路径与资源参数'
        identity = dict(collective=collective, payload_bytes=payload_bytes, physical_ranks=list(ranks),
            topology_identity=topology_identity, concurrent_operations=list(concurrency), dtype=dtype)
        if matrix is not None:
            identity['send_bytes_matrix'] = matrix
        return dict(status='NEEDS_PROFILE', milliseconds=None, identity=identity, mode=mode, reason=reason,
            time_source='PROFILE_REQUIRED', uncertainty='UNAVAILABLE',
            topology_class=self._group_topology_class(topology_identity, ranks), calibration_domain=None,
            resource_ids=[],
            profiling_request=dict(kind='nccl', cost_key=fingerprint(identity), identity=identity))
