from dataclasses import dataclass
import hashlib
import json
import math


@dataclass
class SearchState:
    gpu_hours_used: float = 0.0
    iteration: int = 0
    candidates_evaluated: int = 0
    best_candidate_id: str = None


class CPGCoupledSearch:
    def __init__(self, evaluator, compiler, benchmark, calibrate, architect,
                 search_budget_hours, output):
        if search_budget_hours <= 0:
            raise ValueError('搜索GPU预算必须为正数')
        self.evaluator = evaluator
        self.compiler = compiler
        self.benchmark = benchmark
        self.calibrate = calibrate
        self.architect = architect
        self.search_budget_hours = search_budget_hours
        self.output = output
        self.state = SearchState()
        self.history = []

    @staticmethod
    def normalized_key(candidate):
        payload = [candidate.parallel_config.normalized_key(),
                   [(region.layer_start, region.layer_end) for region in candidate.region_layout or []],
                   candidate.execution_plan_sha256]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def record(self, candidate, result, prediction=None):
        if result['status'] == 'PASS' and not result.get('performance_eligible', False):
            raise ValueError('Profiler或校准结果不能成为正式最佳候选')
        elapsed = result['gpu_allocation_seconds']
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('搜索成本必须来自实际运行时间')
        self.state.gpu_hours_used += candidate.parallel_config.world_size * elapsed / 3600
        candidate.execution_status = 'measured' if result['status'] == 'PASS' else 'failed'
        if result['status'] == 'PASS':
            candidate.actual_throughput = result['tokens_per_second']
            candidate.actual_iteration_time = result['step_seconds']
        entry = dict(candidate=candidate.to_dict(), result=result, prediction=prediction,
                     key=self.normalized_key(candidate), gpu_hours=self.state.gpu_hours_used)
        self.history.append(entry)
        successful = [item for item in self.history if item['result']['status'] == 'PASS']
        if successful:
            self.state.best_candidate_id = max(successful,
                key=lambda item: item['result']['tokens_per_second'])['candidate']['candidate_id']
        self.output.write_text(json.dumps(dict(history=self.history, state=self.state.__dict__),
                                         indent=2, allow_nan=False) + '\n')

    def search(self, candidates, critic):
        pending = list(candidates)
        seen = set()
        while pending and self.state.gpu_hours_used < self.search_budget_hours:
            candidates = []
            for candidate in pending:
                if self.state.gpu_hours_used >= self.search_budget_hours:
                    break
                key = self.normalized_key(candidate)
                if key in seen or not candidate.is_valid:
                    continue
                seen.add(key)
                program = self.compiler.compile(candidate)
                calibration = self.calibrate(candidate, program)
                if calibration['status'] != 'PASS':
                    self.record(candidate, calibration | {'performance_eligible': False}, None)
                    continue
                self.state.gpu_hours_used += candidate.parallel_config.world_size * calibration['gpu_allocation_seconds'] / 3600
                prediction = self.evaluator(candidate, calibration)
                candidate.predicted_critical_path_ms = prediction['predicted_iteration_time_ms']
                candidates.append((prediction['predicted_iteration_time_ms'], candidate, program, prediction))
            pending = []
            for _, candidate, program, prediction in sorted(candidates, key=lambda item: item[0]):
                if self.state.gpu_hours_used >= self.search_budget_hours:
                    break
                result = self.benchmark(candidate, program)
                self.record(candidate, result, prediction)
                self.state.iteration += 1
                self.state.candidates_evaluated += 1
                if result['status'] == 'PASS':
                    pending.extend(self.architect.generate_candidates(candidate, critic(result)))
        return dict(history=self.history, state=self.state.__dict__)
