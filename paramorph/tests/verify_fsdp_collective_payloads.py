import argparse
import json
from pathlib import Path

from core_data_models import TaskType
from cost.runtime_calibration import configuration
from graph.model_graph import read_model
from region_partition.region_partitioner import RegionPartitioner
from scheduling.schedule_simulator import topological_order
from scheduling.task_graph_builder import TaskGraphBuilder


def verify_run(run, model):
    request = json.loads((run / 'request.json').read_text())
    manifests = {rank: json.loads((run / 'artifacts' / f'rank{rank}.manifest.json').read_text())
        for rank in range(request['node_count'] * request['gpus_per_node'])}
    ownership = {rank: json.loads((run / 'artifacts' / f'rank{rank}.ownership.json').read_text())
        for rank in manifests}
    config = configuration(request)
    regions = RegionPartitioner(model, config).from_counts(request['decoder_chunks'])
    dag = TaskGraphBuilder().build(model, config, regions, config.num_microbatches)
    topological_order(dag.tasks)
    checked = 0
    for task in dag.tasks.values():
        if task.task_type not in (TaskType.FSDP_UNSHARD, TaskType.FSDP_GRAD_REDUCE_SCATTER):
            continue
        manifest, metadata = manifests[task.rank], task.metadata
        expected_dtype = (manifest['params_dtype'] if task.task_type == TaskType.FSDP_UNSHARD
            else manifest['optimizer_state_dtypes']['main_grads']).removeprefix('torch.')
        if metadata['communication_dtype'] != expected_dtype:
            raise ValueError(f'{task.task_id}通信精度与实际Manifest不同')
        if metadata['group_ranks'] != manifest['process_groups']['dp_cp']:
            raise ValueError(f'{task.task_id}分片进程组与实际Manifest不同')
        participants = len(metadata['group_ranks'])
        full, shards = metadata['communication_full_payloads_bytes'], metadata['communication_payloads']
        if full != [value * participants for value in shards]:
            raise ValueError(f'{task.task_id}完整Buffer与本Rank分片关系错误')
        if metadata['payload_bytes'] != max(full):
            raise ValueError(f'{task.task_id}显存Buffer错误地使用分片大小')
        checked += 1
    if checked == 0:
        raise ValueError('真实FSDP配置没有生成通信任务')

    prefetch_request = dict(request, fsdp_prefetch=True)
    prefetch_config = configuration(prefetch_request)
    prefetch_dag = TaskGraphBuilder().build(model, prefetch_config, regions, config.num_microbatches)
    topological_order(prefetch_dag.tasks)
    prefetch_count = 0
    for task in prefetch_dag.tasks.values():
        if task.task_type != TaskType.FSDP_PREFETCH:
            continue
        wait = prefetch_dag.tasks[task.task_id.replace('FSP:', 'FSU:', 1)]
        for field in ('communication_payloads', 'communication_dtype', 'payload_bytes'):
            if task.metadata[field] != wait.metadata[field]:
                raise ValueError(f'{task.task_id}预取与完成事件的Buffer契约不同')
        prefetch_count += 1
    if not prefetch_count:
        raise ValueError('启用Prefetch后没有生成实际通信任务')
    parameter_checks = []
    for rank, manifest in manifests.items():
        chunks = ownership[rank]['chunks']
        module_rows = [row for chunk in chunks for row in chunk['regions']]
        for row in module_rows:
            has_qk_norm = all(name in row['parameter_keys'] for name in (
                'self_attention.q_layernorm.weight', 'self_attention.k_layernorm.weight'))
            if has_qk_norm != model.finalize_grad_semantics['qk_layernorm']:
                raise ValueError(f'Rank {rank}的QK Norm模型语义与实际模块参数不同')
        modeled = TaskGraphBuilder.region_parameter_components(
            model, config, [row['layer_id'] for row in module_rows])['dense']
        for chunk in chunks:
            if chunk['pre_process']:
                modeled += manifest['vocab_size'] * model.hidden_size // config.tp
            if chunk['post_process']:
                modeled += manifest['vocab_size'] * model.hidden_size // config.tp + model.hidden_size
        measured = manifest['parameter_elements']
        if modeled != measured:
            raise ValueError(f'Rank {rank}参数量与实际Manifest不同：{modeled} != {measured}')
        parameter_checks.append(dict(rank=rank, runtime_parameter_elements=measured,
            modeled_parameter_elements=modeled, difference=measured - modeled))
    return dict(run_id=request['run_id'], vpp=request['vpp'], checked_collective_tasks=checked,
        checked_prefetch_tasks=prefetch_count, parameter_dtype='bfloat16', gradient_dtype='float32',
        group_size=4, qk_layernorm=True, parameter_checks=parameter_checks,
        verification_scope='recorded_model_manifest_and_compiled_collective_contract')


def verify_contracts():
    builder = TaskGraphBuilder()
    for collective, dtype, expected in (
            ('all_gather', 'bfloat16', 165155968),
            ('reduce_scatter', 'float32', 330311936)):
        result = builder.fsdp_collective_metadata([330311936], [0, 1, 2, 3], dtype, collective)
        if result['communication_payloads'] != [expected] or result['payload_bytes'] != expected * 4:
            raise ValueError('Qwen3-14B单层的通信字节契约错误')
        single = builder.fsdp_collective_metadata([330311936], [0], dtype, collective)
        if single['communication_payloads'] != single['communication_full_payloads_bytes']:
            raise ValueError('单Rank通信存在无依据的分片节省')
    outer = builder.fsdp_collective_metadata([330311936], [0, 8], 'float32', 'reduce_scatter')
    if outer['communication_payloads'] != [660623872] or outer['payload_bytes'] != 1321247744:
        raise ValueError('Outer ReduceScatter的完整状态Buffer与输出分片关系错误')
    outer_parameter = builder.fsdp_collective_metadata([330311936], [0, 8], 'bfloat16', 'all_gather')
    if outer_parameter['communication_payloads'] != [330311936]:
        raise ValueError('Outer AllGather的输入分片关系错误')
    padded = builder.fsdp_collective_metadata([330311937], [0, 1, 2, 3], 'float32', 'reduce_scatter')
    if padded['communication_padding_bytes'] != [12] or padded['actual_padding_measured']:
        raise ValueError('解析等分补齐没有正确记录来源及字节数')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    search = args.workspace / 'outputs/qwen3-14b-full-search'
    settings = json.loads((search / 'system-config.json').read_text())
    model, _ = read_model(settings['model_config_path'])
    runs = (
        'paramorph-qwen3-14b-fsdpv1-tp1pp4cp2dp2-smoke-20261011-r4',
        'paramorph-qwen3-14b-fsdpv1-tp1pp4cp2dp2-vpp2-smoke-20261011')
    verify_contracts()
    results = [verify_run(search / 'training/runs' / run_id, model) for run_id in runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS', runs=results), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', runs=results), ensure_ascii=False))


if __name__ == '__main__':
    main()
