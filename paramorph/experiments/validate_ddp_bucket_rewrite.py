import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from core_data_models import Task, TaskType
from predictor.ddp_buckets import split_measured_regions
from scheduling.schedule_simulator import topological_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outcomes = []
    for path in args.prediction:
        report = json.loads(path.read_text())
        tasks = {key: Task(key, row['logical_rank'], TaskType(row['kind']),
            duration=row['duration_ms'], region_id=row['region'], microbatch_id=row['microbatch'],
            dependencies=list(row['dependencies']), resource_requirements=tuple(row['resources']),
            metadata=row['metadata']) for key, row in report['execution_graph']['tasks'].items()}
        before, after = defaultdict(float), defaultdict(float)
        for task in tasks.values():
            before[task.rank] += task.duration
        original_count = len(tasks)
        positions = split_measured_regions(tasks, report['evidence']['costs'])
        topological_order(tasks)
        for task in tasks.values():
            after[task.rank] += task.duration
            component_total = sum(row['milliseconds'] * row['occurrences'] for row in task.metadata['cost_components'])
            if not math.isclose(task.duration, component_total, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('DDP层任务展开后成本分量不守恒')
        if any(not math.isclose(value, after[rank], rel_tol=1e-9, abs_tol=1e-6) for rank, value in before.items()):
            raise ValueError('DDP层任务展开改变了Rank总服务时间')
        outcomes.append(dict(candidate_id=report['candidate_id'], source=str(path),
            original_tasks=original_count, expanded_tasks=len(tasks), layer_positions=len(positions),
            maximum_rank_service_difference_ms=max(abs(value - after[rank]) for rank, value in before.items()),
            dag_valid=True, cost_components_complete=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(outcomes, ensure_ascii=False))


if __name__ == '__main__':
    main()
