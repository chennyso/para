import argparse
import hashlib
import json
from pathlib import Path

from cost.runtime_calibration import configuration
from graph.model_graph import read_model
from region_partition.region_partitioner import RegionPartitioner
from scheduling.schedule_simulator import topological_order
from scheduling.task_graph_builder import TaskGraphBuilder


def verify_run(run, model):
    request = json.loads((run / 'request.json').read_text())
    config = configuration(request)
    regions = RegionPartitioner(model, config).from_counts(request['decoder_chunks'])
    dag = TaskGraphBuilder().build(model, config, regions, config.num_microbatches)
    topological_order(dag.tasks)
    manifests = [json.loads((run / 'artifacts' / f'rank{rank}.manifest.json').read_text())
        for rank in range(config.world_size)]
    log = (run / 'artifacts/node0.log').read_text()
    if 'clip_grad=1.0' not in log or 'loss_scale=None' not in log:
        raise ValueError('真实训练日志没有确认当前Optimizer裁剪与Loss Scale配置')
    reductions = [task for task in dag.tasks.values()
        if task.metadata.get('operation') == 'optimizer_gradient_norm_all_reduce']
    by_rank = {rank: task for task in reductions for rank in task.metadata['group_ranks']}
    if len(by_rank) != config.world_size:
        raise ValueError('梯度范数通信没有完整覆盖实测Rank')
    rows = []
    for rank, manifest in enumerate(manifests):
        communication = by_rank[rank]
        group = manifest['process_groups']['intra_dist_opt']
        if communication.metadata['group_ranks'] != group:
            raise ValueError(f'Rank {rank}的Optimizer统计组与实际Manifest不同')
        if (communication.metadata['payload_bytes'] != 4
                or communication.metadata['dtype'] != 'float32'
                or communication.metadata['reduce_op'] != 'sum'):
            raise ValueError('全局L2范数归约消息与部署源码不一致')
        if set(communication.dependencies) != {f'ONORM:{other}' for other in group}:
            raise ValueError('全局范数归约没有等待全部实际参与者')
        expected = {
            f'ONORM:{rank}': [f'OPREP:{rank}'],
            f'ONORM_POST:{rank}': [communication.task_id],
            f'OCLIP:{rank}': [f'ONORM_POST:{rank}'],
            f'OPT:{rank}': [f'OCLIP:{rank}'],
        }
        if f'FINAL:{rank}' not in dag.tasks[f'OPREP:{rank}'].dependencies:
            raise ValueError('Optimizer准备任务没有等待梯度同步完成')
        for key, dependencies in expected.items():
            if dag.tasks[key].dependencies != dependencies:
                raise ValueError(f'{key}的Optimizer执行依赖错误')
        rows.append(dict(rank=rank, actual_grad_stats_group=group,
            main_grads_dtype=manifest['optimizer_state_dtypes']['main_grads'],
            parameter_elements=manifest['parameter_elements']))
    return dict(run_id=request['run_id'], tp=config.tp, pp=config.pp, dp=config.dp, cp=config.cp,
        scalar_collective_count=len(reductions), parameter_and_gradient_norm_sources='actual_rank_manifests',
        checked_rank_count=len(rows), ranks=rows, gpu_execution_performed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    search = args.workspace / 'outputs/qwen3-14b-full-search'
    settings = json.loads((search / 'system-config.json').read_text())
    model, _ = read_model(settings['model_config_path'])
    runs = ('initial-188d58c4b05fb971-df8fea2d8565',
        'paramorph-qwen3-14b-ddp-tp2pp8cp1dp1-global-20261011')
    sources = args.workspace / 'work/optimizer-cost-campaign/deployed-optimizer'
    hashes = {name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
        for name in ('optimizer.py', 'clip_grads.py', 'distrib_optimizer.py', '__init__.py')}
    results = [verify_run(search / 'training/runs' / run_id, model) for run_id in runs]
    result = dict(status='PASS', deployed_optimizer_source_hashes=hashes, runs=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', checked_runs=len(results),
        checked_rank_count=sum(row['checked_rank_count'] for row in results)), ensure_ascii=False))


if __name__ == '__main__':
    main()
