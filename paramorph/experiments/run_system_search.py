import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts import SystemConfig
from search.gpu_budget_policy import unified_gpu_budget
from search.native_calibration_search import CalibratedDAGAgentSearch
from search.system_search import EvidenceGuidedSearch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=('full', 'multi_agent', 'seed', 'random_region', 'config_only', 'graph_aware', 'graph_impact'), default='full')
    parser.add_argument('--selection', choices=('investigation', 'topk', 'diversity'), default=None)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--cost-mode', choices=('hybrid', 'measured_only', 'analytical_only'), default=None)
    parser.add_argument('--hardware-parameters', type=Path)
    parser.add_argument('--hardware-topology-path', type=Path, action='append')
    args = parser.parse_args()
    document = json.loads(args.config.read_text())
    document['output_directory'] = str(args.output.resolve())
    document['evaluation_mode'] = args.method
    if args.selection is not None:
        document['selection_policy'] = args.selection
    if args.no_gpu:
        document['gpu_execution_enabled'] = False
    if args.cost_mode is not None:
        document['predictor_cost_mode'] = args.cost_mode
    if args.hardware_parameters is not None:
        document['predictor_hardware_parameters'] = json.loads(args.hardware_parameters.read_text())
    if args.hardware_topology_path:
        document['hardware_topology_paths'] = [str(path.resolve())
            for path in args.hardware_topology_path]
    config = SystemConfig.model_validate(document)
    config = config.model_copy(update=dict(budget=unified_gpu_budget(config.budget)))
    args.output.mkdir(parents=True, exist_ok=True)
    stored = args.output / 'system-config.json'
    if stored.is_file():
        previous = SystemConfig.model_validate_json(stored.read_text())
        previous = previous.model_copy(update=dict(budget=unified_gpu_budget(previous.budget)))
        if previous != config:
            raise ValueError('同一搜索目录中的模型、方法和预算配置必须保持一致')
    stored.write_text(config.model_dump_json(indent=2) + '\n')

    dag_agent_mode = (
        config.planning_control_enabled
        and config.planning_selector == 'agent'
        and config.planning_partition_engine == 'path'
        and config.agent_execution_enabled
        and config.evaluation_mode in ('full', 'multi_agent')
        and not config.single_decision_round
    )
    search = CalibratedDAGAgentSearch(config) if dag_agent_mode else EvidenceGuidedSearch(config)
    report = search.run()
    print(json.dumps(dict(best_plan=report['best_measured']['plan_hash'] if report['best_measured'] else None,
        usage=report['usage'], gpu_execution_enabled=report['gpu_execution_enabled'])))
