from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import statistics
import math

from core_data_models import Bottleneck, Measurement, CriticReport
from cost.runtime_calibration import RuntimeCalibration
from paramorph.trace.trace_parser import TraceParser
from paramorph.trace.dependency_graph import ExecutionGraphBuilder
from scheduling.critical_path import CriticalPathAnalyzer


class CriticAgent:
    def __init__(self, decision_service=None):
        self.decision_service = decision_service
        self.calibration = None
        self.critical_path = []
        self.bottlenecks = []
        self.layer_forward_costs_ms = {}

    @staticmethod
    def attach_task_trace(evidence, profile_directory):
        directory = Path(profile_directory)
        request = json.loads((directory / 'request.json').read_text())
        result = json.loads((directory / 'summary.json').read_text())
        if request['plan'] != evidence['plan'] or request['mode'] != 'profile' or result['returncodes'] != [0, 0]:
            raise ValueError('任务Trace与实际训练策略不一致')
        ranks = {}
        for rank in range(request['plan']['world']):
            artifact = directory / 'artifacts'
            identity_path = artifact / f'rank{rank}.task-events.json'
            trace_path = artifact / f'rank{rank}.trace.json'
            identities = json.loads(identity_path.read_text())
            parser = TraceParser()
            events = parser.associate_nvtx_ranges(parser.parse_pytorch_trace(trace_path, rank))
            ranges = defaultdict(list)
            kernels_by_range = defaultdict(list)
            launches = {event.event_id: event for event in events}
            for event in sorted(events, key=lambda item: item.start_ns):
                if event.name.startswith('ParaMorph/'):
                    ranges[event.name].append(event)
                if event.nvtx_domain is not None:
                    kernels_by_range[event.nvtx_domain].append(event)
            tasks = []
            for identity in identities['events']:
                label = identity['label']
                if not ranges[label]:
                    raise ValueError('任务事件缺少实际Profiler区间')
                scope = ranges[label].pop(0)
                kernels = [event for event in kernels_by_range[label] if
                    scope.start_ns <= launches[event.kernel_args['launch_event_id']].start_ns <= scope.end_ns]
                tasks.append(dict(identity=identity['identity'], label=label,
                    host_start_ns=scope.start_ns, host_end_ns=scope.end_ns,
                    host_duration_ms=scope.duration_ms,
                    correlated_gpu_events=[dict(event_id=event.event_id, name=event.name,
                        start_ns=event.start_ns, end_ns=event.end_ns,
                        duration_ms=event.duration_ms, stream_id=event.stream_id)
                        for event in kernels]))
            ranks[str(rank)] = dict(tasks=tasks,
                trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                identities_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
        return evidence | dict(task_trace=dict(ranks=ranks, profile_directory=str(directory),
            cross_rank_clock_alignment='unknown', performance_eligible=False,
            missing_measurements=['跨Rank时钟校准', 'Collective进程组与消息身份',
                '通信发布与完成等待的独立身份', '完整CPU卸载传输依赖']))

    @staticmethod
    def collect_training_evidence(directories, allow_validation=False):
        evidence = []
        for directory in map(Path, directories):
            request_path = directory / 'request.json'
            result_path = directory / 'summary.json'
            if not result_path.exists():
                result_path = directory / 'exit.json'
            request = json.loads(request_path.read_text())
            result = json.loads(result_path.read_text())
            identifier = hashlib.sha256(request_path.read_bytes() + result_path.read_bytes()).hexdigest()
            item = dict(evidence_id=identifier, directory=str(directory),
                plan=request['plan'], status=result['status'], returncodes=result['returncodes'],
                warmup=request['warmup'], measured=request['measured'],
                precision='bfloat16', instrumented=result.get('instrumented'),
                scope='actual_training_result', cause_attribution='unavailable')
            item['implementation_sources'] = json.loads((directory / 'launch.json').read_text())['sources']
            actual_ranks = {}
            for path in sorted((directory / 'artifacts').glob('rank*.manifest.json')):
                manifest = json.loads(path.read_text())
                rank = manifest['rank']
                if manifest['plan'] != request['plan'] or not 0 <= rank < request['plan']['world']:
                    raise ValueError('实际并行组记录与训练请求不一致')
                if str(rank) in actual_ranks:
                    raise ValueError('实际并行组记录包含重复Rank')
                semantics = dict(manifest['training_semantics'])
                semantics.pop('optimizer_cpu_classes', None)
                actual_ranks[str(rank)] = dict(process_groups=manifest['process_groups'],
                    training_semantics=semantics, decoder_layer_ids=manifest['decoder_layer_ids'],
                    manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                semantic_path = path.with_name(f'rank{rank}.semantic-region-graphs.json')
                if semantic_path.exists():
                    actual_ranks[str(rank)]['semantic_graph_path'] = str(semantic_path)
                    actual_ranks[str(rank)]['semantic_graph_sha256'] = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
                boundary_path = path.with_name(f'rank{rank}.moe-boundary.json')
                if boundary_path.exists():
                    actual_ranks[str(rank)]['module_boundary'] = json.loads(boundary_path.read_text())
                    actual_ranks[str(rank)]['boundary_sha256'] = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
            item['actual_execution'] = dict(ranks=actual_ranks,
                scope='runtime_manifest_and_executed_module_boundary',
                complete_rank_coverage=len(actual_ranks) == request['plan']['world'],
                kernel_dependencies_available=False)
            if result['status'] == 'complete':
                if (result['returncodes'] != [0, 0] or result['instrumented'] or
                        (request['mode'] != 'formal' and not (allow_validation and request['mode'] == 'validate'))):
                    raise ValueError('训练证据不是有效正式测量')
                rank_hashes = {}
                for rank in range(request['plan']['world']):
                    path = directory / 'artifacts' / f'rank{rank}.measurements.json'
                    measurement = json.loads(path.read_text())
                    if (measurement['rank'] != rank or measurement['plan'] != request['plan'] or
                            measurement['completed_optimizer_updates'] != request['warmup'] + request['measured'] or
                            len(measurement['steps']) != request['warmup'] + request['measured']):
                        raise ValueError('实际Rank记录与训练请求不一致')
                    for step in measurement['steps']:
                        if (not step['optimizer_update_successful'] or not math.isfinite(step['grad_norm']) or
                                any(not math.isfinite(float(value)) for value in step['loss'].values())):
                            raise ValueError('实际训练更新或者有限数值检查失败')
                    rank_hashes[str(rank)] = hashlib.sha256(path.read_bytes()).hexdigest()
                item.update(tokens_per_second=result['tokens_per_second'],
                    step_seconds=result['update_seconds'], peak_allocated_gib=result['peak_allocated_gib'],
                    rank_measurement_sha256=rank_hashes,
                    numerical_scope='finite_loss_grad_norm_and_successful_updates')
                item['performance_eligible'] = request['mode'] == 'formal'
                if request['mode'] == 'validate':
                    checks = []
                    for rank in range(request['plan']['world']):
                        record = json.loads((directory / 'artifacts' / f'rank{rank}.validation.json').read_text())
                        if len(record['training_state_checks']) != request['warmup'] + request['measured']:
                            raise ValueError('数值验证缺少连续更新状态检查')
                        checks.extend(record['tensor_checks'])
                    if not checks or any(not check['passed'] for check in checks):
                        raise ValueError('实际完整Tensor数值对照没有通过')
                    item.update(status='validated', tokens_per_second=None, step_seconds=None,
                        numerical_scope='complete_saved_tensor_and_optimizer_state_comparison',
                        numerical_errors=dict(tensor_checks=len(checks),
                            max_absolute_error=max(check.get('max_absolute_error', 0) for check in checks),
                            max_relative_l2_error=max(check.get('relative_l2_error', 0) for check in checks)))
            else:
                if result['status'] not in ('cuda_oom', 'failed'):
                    raise ValueError('证据记录尚未完成实际验收')
                logs = sorted((directory / 'artifacts').glob('node*.training.log'))
                if not logs:
                    raise ValueError('失败证据缺少实际训练日志')
                item['log_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in logs}
                termination = directory / 'termination.json'
                if termination.exists():
                    item['termination'] = json.loads(termination.read_text())
                item['first_error_excerpts'] = {}
                for path in logs:
                    lines = path.read_text().splitlines()
                    first = next((index for index, line in enumerate(lines)
                                  if 'Pretrain failed on rank=' in line), None)
                    if first is not None:
                        item['first_error_excerpts'][path.name] = lines[first:first + 4]
            evidence.append(item)
        return evidence

    def load_runtime(self, directory):
        self.calibration = RuntimeCalibration(directory)
        return self.calibration

    def diagnose(self):
        if self.calibration is None:
            raise ValueError('Critic必须读取实际训练成本文件')
        rank_loads = {}
        evidence = {}
        layer_values = defaultdict(list)
        for path, payload in zip(self.calibration.paths, self.calibration.payloads):
            totals = defaultdict(float)
            request = payload['request']
            for event in payload['events']:
                if event['step'] < request['warmup']:
                    continue
                if request.get('paramorph_trace') and event['step'] == request['warmup'] + request['measured'] - 1:
                    continue
                if event['kind'] in ('forward_compute', 'backward_compute'):
                    if event['cuda_ms'] is None:
                        raise ValueError('Stage负载分析必须具有CUDA计时')
                    totals[event['step']] += event['cuda_ms']
                if event['operation'] == 'layer_forward':
                    layer_values[event['identity']['layer_id']].append(event['cuda_ms'])
            if not totals:
                raise ValueError('缺少实际Stage Forward和Backward成本')
            rank_loads[payload['rank']] = statistics.median(totals.values())
            evidence[payload['rank']] = hashlib.sha256(path.read_bytes()).hexdigest()
        slowest = max(rank_loads, key=rank_loads.get)
        fastest = min(rank_loads, key=rank_loads.get)
        fraction = (rank_loads[slowest] - rank_loads[fastest]) / rank_loads[slowest]
        self.bottlenecks = []
        self.layer_forward_costs_ms = {key: statistics.median(values)
                                     for key, values in layer_values.items()}
        if fraction > 0.05:
            self.bottlenecks.append(Bottleneck('STAGE_IMBALANCE', fraction,
                [slowest, fastest],
                f'实际Region Forward与Backward累计耗时相差{fraction:.2%}，包含Region内部Collective',
                [evidence[slowest], evidence[fastest]], 1.0))
        return self.bottlenecks

    def parse_gpu_traces(self, trace_paths, source_type='nsys', clock_calibration=None):
        bundle = TraceParser().parse_multi_rank(trace_paths, source_type, clock_calibration)
        dag = ExecutionGraphBuilder().build(bundle)
        analyzer = CriticalPathAnalyzer()
        self.critical_path = analyzer.identify_critical_path(dag)
        return dag

    def find_critical_frontier(self, dag, task_budget=8, reduction_fraction=0.1):
        if type(task_budget) is not int or task_budget <= 0 or not 0 < reduction_fraction < 1:
            raise ValueError('敏感任务预算和耗时变化比例不合法')
        if dag.metadata.get('coverage') != 'complete_iteration' or dag.metadata.get('data_gaps'):
            raise ValueError('敏感任务分析需要完整迭代的真实执行依赖和资源顺序')
        analyzer = CriticalPathAnalyzer()
        analyzer.identify_critical_path(dag)
        baseline = analyzer.get_critical_path_length()
        candidates = sorted((key for key, task in dag.tasks.items()
            if task.duration > 0 and analyzer.is_on_critical_path(key)),
            key=lambda key: (-dag.tasks[key].duration, key))[:task_budget]
        frontier = []
        for key in candidates:
            task = dag.tasks[key]
            changed = deepcopy(dag)
            reduction = task.duration * reduction_fraction
            changed.tasks[key].duration -= reduction
            comparison = CriticalPathAnalyzer()
            comparison.identify_critical_path(changed)
            saving = baseline - comparison.get_critical_path_length()
            frontier.append(dict(task_id=key, rank=task.rank, region_id=task.region_id,
                operation=task.task_type.value, duration_ms=task.duration,
                duration_reduction_ms=reduction, predicted_saving_ms=saving,
                sensitivity=saving / reduction,
                dependencies=list(task.dependencies), metadata=dict(task.metadata)))
        return dict(tasks=sorted(frontier, key=lambda item: (-item['sensitivity'], item['task_id'])),
            baseline_critical_path_ms=baseline, evaluated_task_count=len(frontier),
            task_budget=task_budget, graph_metadata=dag.metadata,
            scope='measured_trace_dependency_sensitivity',
            uncertainty=['变换后的Kernel成本', '变换后的通信竞争和动态路由'],
            complete_candidate_speed_prediction=False)

    def generate_critic_report(self, parallel_config, dag=None):
        if self.calibration is None or parallel_config != self.calibration.config:
            raise ValueError('Critic配置与实际训练配置不一致')
        bottlenecks = self.diagnose()
        measurements = []
        windows = []
        for payload in self.calibration.payloads:
            request = payload['request']
            values = [(item['host_end_ns'] - item['host_start_ns']) / 1e6
                for item in payload['windows'] if item['step'] >= request['warmup']
                and not (request.get('paramorph_trace') and
                    item['step'] == request['warmup'] + request['measured'] - 1)]
            windows.append(statistics.median(values))
        iteration_ms = max(windows)
        measurements.append(Measurement('iteration_time_ms', iteration_ms, 'ms',
            'actual_training_iteration_wall_time', False))
        gaps = ['独立TP/CP/EP/PP通信成本', '实测计算通信并发配对', '跨rank关键路径']
        critical = []
        critical_ms = None
        if dag is not None and dag.metadata.get('coverage') == 'complete_iteration' and not dag.metadata.get('data_gaps'):
            analyzer = CriticalPathAnalyzer()
            critical = analyzer.identify_critical_path(dag)
            critical_ms = analyzer.get_critical_path_length()
            gaps.remove('跨rank关键路径')
        report = CriticReport(parallel_config, critical, critical_ms,
            bottlenecks, measurements, gaps, layer_forward_costs_ms=self.layer_forward_costs_ms)
        if self.decision_service is not None:
            report.agent_decision = self.decision_service('critic', report.to_dict())
        return report

    def generate_report(self):
        if self.calibration is None:
            raise ValueError('尚未读取实际运行证据')
        return json.dumps(self.generate_critic_report(self.calibration.config).to_dict(),
                          ensure_ascii=False, indent=2, allow_nan=False)
