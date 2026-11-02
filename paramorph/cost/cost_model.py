from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import statistics

from core_data_models import Task, TaskType


class MissingCalibration(ValueError):
    pass


@dataclass
class TaskCost:
    task_type: TaskType
    estimated_ms: float
    confidence: float
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PairCost:
    compute_task_type: TaskType
    comm_task_type: TaskType
    compute_alone_ms: float
    comm_alone_ms: float
    overlapped_ms: float
    benefit_ms: float
    confidence: float
    source: str


class ExecutionCostModel:
    def __init__(self, path=None):
        self.samples = defaultdict(list)
        self.pairs = defaultdict(list)
        self.memory = {}
        self.evidence = defaultdict(list)
        if path is not None:
            payload = json.loads(Path(path).read_text())
            self.samples.update(payload['samples'])
            self.pairs.update(payload['pairs'])
            self.memory.update(payload['memory'])
            self.evidence.update(payload['evidence'])

    @staticmethod
    def key(task, config):
        if config is None:
            raise MissingCalibration('成本查询必须提供完整并行配置')
        identity = {name: task.metadata.get(name) for name in (
            'operation', 'layer_ids', 'shape', 'dtype', 'group_ranks', 'link',
            'direction', 'includes', 'rank_class', 'send_next', 'send_prev', 'recv_prev',
            'recv_next', 'completion_waited')}
        return json.dumps([config.normalized_key(), task.task_type.value, identity],
                          sort_keys=True, separators=(',', ':'))

    @property
    def version(self):
        return hashlib.sha256(json.dumps(dict(samples=self.samples, pairs=self.pairs,
            memory=self.memory), sort_keys=True).encode()).hexdigest()[:16]

    def update(self, measurement):
        required = ('task', 'config', 'milliseconds', 'source', 'evidence_id')
        if any(name not in measurement for name in required):
            raise ValueError('成本测量缺少身份、配置、耗时或证据')
        value = measurement['milliseconds']
        if not math.isfinite(value) or value < 0:
            raise ValueError('成本测量必须为有限非负数值')
        key = self.key(measurement['task'], measurement['config'])
        self.samples[key].append(value)
        self.evidence[key].append({name: measurement[name] for name in ('source', 'evidence_id')})

    def get_task_cost(self, task_type, config=None, topology=None, metadata=None):
        return self.estimate(Task('query', 0, task_type, metadata=metadata or {}), config)

    def estimate(self, task, config):
        key = self.key(task, config)
        if not self.samples.get(key):
            raise MissingCalibration(f'缺少对应配置的任务校准：{key}')
        values = self.samples[key]
        median = statistics.median(values)
        dispersion = statistics.pstdev(values)
        confidence = 1 / (1 + dispersion / median) if median else float(dispersion == 0)
        return TaskCost(task.task_type, median, confidence, 'measured',
            {'sample_count': len(values), 'standard_deviation_ms': dispersion,
             'minimum_ms': min(values), 'maximum_ms': max(values), 'evidence': self.evidence[key]})

    def estimate_compute(self, task, config):
        return self.estimate(task, config)

    def estimate_communication(self, task, topology, config):
        if not topology.validate():
            raise ValueError('硬件拓扑设备数量不一致')
        return self.estimate(task, config)

    def update_pair(self, compute, communication, config, compute_ms, comm_ms,
                    overlapped_ms, evidence_id):
        if any(not math.isfinite(value) or value < 0 for value in (compute_ms, comm_ms, overlapped_ms)):
            raise ValueError('配对耗时必须为有限非负数值')
        if not evidence_id:
            raise ValueError('并发配对必须具有测量证据')
        key = json.dumps([self.key(compute, config), self.key(communication, config)])
        self.pairs[key].append(dict(compute_ms=compute_ms, comm_ms=comm_ms,
                                   overlapped_ms=overlapped_ms, evidence_id=evidence_id))

    def estimate_pair(self, compute, communication, config):
        key = json.dumps([self.key(compute, config), self.key(communication, config)])
        if not self.pairs.get(key):
            raise MissingCalibration('缺少对应Shape、配置和设备的并发配对校准')
        rows = self.pairs[key]
        compute_ms = statistics.median(row['compute_ms'] for row in rows)
        comm_ms = statistics.median(row['comm_ms'] for row in rows)
        overlap = statistics.median(row['overlapped_ms'] for row in rows)
        return PairCost(compute.task_type, communication.task_type, compute_ms, comm_ms,
                        overlap, compute_ms + comm_ms - overlap, 1.0, 'measured')

    def get_pair_cost(self, compute_type, comm_type, config=None):
        return self.estimate_pair(Task('compute', 0, compute_type), Task('comm', 0, comm_type), config)

    def estimate_memory(self, plan):
        if not plan.task_times or not plan.tasks:
            raise ValueError('显存预测需要完整任务集合与执行时间')
        if any(not task.metadata.get('memory_evidence') for task in plan.tasks.values()):
            raise MissingCalibration('任务激活与Workspace显存缺少测量证据')
        peaks = {}
        for rank in plan.rank_schedules:
            if rank not in plan.persistent_bytes:
                raise MissingCalibration(f'缺少rank{rank}的常驻显存测量')
            events = []
            for task in plan.tasks.values():
                if task.rank != rank:
                    continue
                start, end = plan.task_times[task.task_id]
                events.extend([(start, 1, max(task.activation_delta_bytes, 0) + task.workspace_bytes),
                               (end, 0, min(task.activation_delta_bytes, 0) - task.workspace_bytes)])
            used = peak = plan.persistent_bytes[rank]
            for _, _, delta in sorted(events):
                used += delta
                if used < plan.persistent_bytes[rank]:
                    raise ValueError('激活释放超过已申请数量')
                peak = max(peak, used)
            peaks[rank] = peak
        return peaks

    def save(self, path):
        Path(path).write_text(json.dumps(dict(samples=self.samples, pairs=self.pairs,
            memory=self.memory, evidence=self.evidence, version=self.version),
            indent=2, allow_nan=False) + '\n')

    def get_missing_measurements(self, plan=None):
        result = dict(calibrated_task_keys=len(self.samples), calibrated_pair_keys=len(self.pairs))
        if plan is None:
            return result | {'requirements_not_provided': True}
        tasks = [key for key, task in plan.tasks.items()
                 if not self.samples.get(self.key(task, plan.parallel_config))]
        pairs = []
        for left, right in plan.concurrent_pairs:
            compute, communication = plan.tasks[left], plan.tasks[right]
            if not compute.is_compute:
                compute, communication = communication, compute
            key = json.dumps([self.key(compute, plan.parallel_config),
                              self.key(communication, plan.parallel_config)])
            if not self.pairs.get(key):
                pairs.append([left, right])
        return result | dict(unmeasured_tasks=tasks, unmeasured_pairs=pairs,
            unmeasured_persistent_memory=[rank for rank in plan.rank_schedules
                                          if rank not in plan.persistent_bytes])
