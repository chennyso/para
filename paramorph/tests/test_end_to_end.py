import argparse
import json
from pathlib import Path

from cost.cost_model import ExecutionCostModel
from cost.runtime_calibration import RuntimeCalibration
from agents.program_agent import ProgramAgent
from core_data_models import Candidate, HardwareTopology, ModelSpec, LayerType
from region_partition.region_partitioner import RegionPartitioner
from validation.parallel_validator import ParallelValidator
from scheduling.schedule_simulator import ScheduleCostEvaluator


def run(directory, maximum_relative_error):
    calibration = RuntimeCalibration(directory)
    request = calibration.payloads[0]['request']
    model_config = json.loads((directory / 'model-config.json').read_text())
    pattern = model_config['layers_block_type']
    model = ModelSpec(request['name'], model_config['num_hidden_layers'],
        model_config['hidden_size'], model_config['num_attention_heads'],
        [LayerType(item) for item in pattern], list(range(len(pattern))),
        num_experts=model_config['n_routed_experts'])
    hardware = HardwareTopology(16, 2, 8, 31.36)
    regions = RegionPartitioner(model, calibration.config).from_counts(request['decoder_chunks'])
    candidate = Candidate(request['run_id'], None, calibration.config, regions, ['measured_runtime'])
    validator = ParallelValidator(model, hardware)
    valid, errors = validator.validate_config(candidate.parallel_config)
    layout_valid, layout_errors = validator.validate_region_layout(candidate.parallel_config, regions)
    valid = valid and layout_valid
    errors += layout_errors
    if not valid:
        raise ValueError(errors)
    program = ProgramAgent().compile(candidate)
    costs = ExecutionCostModel()
    report = calibration.ingest(costs)
    plan = calibration.execution_plan(costs)
    prediction = ScheduleCostEvaluator(costs).evaluate(plan, hardware)
    if abs(prediction.predicted_iteration_time_ms - report['predicted_iteration_time_ms']) > 1e-6:
        raise ValueError('完整执行图预测与阶段成本累计不一致')
    if abs(report['relative_error']) > maximum_relative_error:
        raise ValueError(f'完整迭代预测误差超过验收范围：{report["relative_error"]}')
    for payload in calibration.payloads:
        rank = payload['rank']
        measurement = json.loads((directory / f'rank{rank}.measurements.json').read_text())
        if measurement['status'] != 'complete' or measurement['updates'] != request['warmup'] + request['measured']:
            raise ValueError('实际训练没有完成全部Optimizer更新')
        if calibration.config.vpp > 1 and any(table != [list(pair) for pair in program['schedule_table']]
                for table in payload['schedule_tables']):
            raise ValueError('实际VPP执行表与Program编译结果不一致')
    costs.save(directory / 'execution-cost-model.json')
    (directory / 'end-to-end-validation.json').write_text(json.dumps(
        dict(status='PASS', scope='complete_measured_phase_cost_model_and_runtime',
             prediction=report, program=program,
             predicted_peak_allocated_mib=prediction.predicted_peak_memory_mb),
        indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', relative_error=report['relative_error'],
                          data_gaps=report['data_gaps']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--maximum-relative-error', type=float, default=0.1)
    args = parser.parse_args()
    run(args.directory, args.maximum_relative_error)
