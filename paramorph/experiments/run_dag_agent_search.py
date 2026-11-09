import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts import SystemConfig
from search.dag_agent_control import DAGAgentSearch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--hardware-topology-path', type=Path, action='append')
    args = parser.parse_args()

    document = json.loads(args.config.read_text())
    document['output_directory'] = str(args.output.resolve())
    if args.no_gpu:
        document['gpu_execution_enabled'] = False
    if args.hardware_topology_path:
        document['hardware_topology_paths'] = [
            str(path.resolve()) for path in args.hardware_topology_path]
    config = SystemConfig.model_validate(document)
    args.output.mkdir(parents=True, exist_ok=True)
    stored = args.output / 'system-config.json'
    if stored.exists() and SystemConfig.model_validate_json(stored.read_text()) != config:
        raise ValueError('现有搜索目录中的模型、算法和预算配置与本次请求不同')
    stored.write_text(config.model_dump_json(indent=2) + '\n')

    report = DAGAgentSearch(config).run()
    print(json.dumps(dict(
        best_plan=report['best_measured']['plan_hash']
            if report['best_measured'] else None,
        usage=report['usage'],
        gpu_execution_enabled=report['gpu_execution_enabled'])))
