import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapters.megatron_adapter import MegatronAdapter
from contracts import SystemConfig
from predictor.profile_database import plan_signature


MEMORY_METRICS = ('allocated_bytes', 'reserved_bytes', 'allocator_unused_reserved_bytes',
    'device_used_bytes', 'device_outside_allocator_bytes', 'peak_allocated_bytes',
    'peak_reserved_bytes', 'communication_storage_bytes')


def allocator_lifetimes(path):
    document = json.loads(path.read_text())
    actions, lifetimes = Counter(), []
    initial_free_events = 0
    for device, events in enumerate(document['device_traces']):
        active = {}
        for index, event in enumerate(events):
            action = event['action']
            actions[action] += 1
            if action == 'alloc':
                address = event['addr']
                if address in active:
                    raise ValueError('CUDA allocator在已有活动分配上重复分配相同地址')
                row = dict(device=device, allocation_event_index=index,
                    address=address, bytes=event['size'], stream=event.get('stream'),
                    free_requested_event_index=None, free_completed_event_index=None,
                    allocation_time_us=event.get('time_us'), free_completed_time_us=None)
                active[address] = row
                lifetimes.append(row)
            elif action == 'free_requested' and event['addr'] in active:
                active[event['addr']]['free_requested_event_index'] = index
            elif action == 'free_completed':
                address = event['addr']
                if address in active:
                    row = active.pop(address)
                    row['free_completed_event_index'] = index
                    row['free_completed_time_us'] = event.get('time_us')
                else:
                    initial_free_events += 1
    return dict(source='actual_pytorch_cuda_allocator_device_traces',
        trace_action_counts=dict(actions), allocation_count=len(lifetimes),
        completed_lifetime_count=sum(row['free_completed_event_index'] is not None for row in lifetimes),
        active_at_history_end_count=sum(row['free_completed_event_index'] is None for row in lifetimes),
        frees_without_recorded_allocation_count=initial_free_events,
        lifetime_coordinate='actual_device_trace_event_order',
        time_scope='time_us_is_preserved_only_when_present_in_actual_allocator_event',
        allocations=lifetimes,
        raw_history_path=str(path.resolve()), raw_history_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summarize_memory_run(directory, source_plan_signature=None):
    directory = Path(directory)
    request = json.loads((directory / 'request.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    if result['status'] != 'PASS' or result['performance_eligible']:
        raise ValueError('独立显存测量必须完成真实训练并登记为仪器测量')
    records = [json.loads(path.read_text()) for path in
               sorted((directory / 'artifacts').glob('rank*.paramorph-memory-audit.json'))]
    world = request['node_count'] * request['gpus_per_node']
    if len(records) != world or {record['rank'] for record in records} != set(range(world)):
        raise ValueError('独立显存测量没有完整覆盖全部训练Rank')
    summary, allocator = {}, {}
    for record in records:
        selected = [(index, row) for index, row in enumerate(record['samples'])
                    if row['step'] >= request['warmup']]
        if not selected:
            raise ValueError('独立显存测量缺少预热结束后的采样')
        maxima, observations = {}, {}
        for metric in MEMORY_METRICS:
            index, row = max(selected, key=lambda item: item[1][metric])
            historical = metric in ('peak_allocated_bytes', 'peak_reserved_bytes')
            maxima[metric] = row[metric]
            observations[metric] = dict(value_bytes=row[metric], sample_index=index,
                step=row['step'], phase=row['phase'], snapshot=dict(row),
                observation_scope='historical_allocator_peak_counter_observed_at_this_boundary' if historical
                    else 'current_value_and_all_other_components_at_this_operation_boundary',
                counter_peak_time_observed=False if historical else None)
        transport_peak = observations['communication_storage_bytes']['snapshot']
        storages = transport_peak.get('communication_storages')
        aliases = None
        if storages is not None:
            addresses = [row['storage_pointer'] for row in storages]
            if len(addresses) != len(set(addresses)):
                raise ValueError('通信显存峰值采样包含重复Storage地址')
            if sum(row['storage_bytes'] for row in storages) != transport_peak['communication_storage_bytes']:
                raise ValueError('通信显存峰值与去重后的实际Storage字节数不同')
            aliases = dict(unique_storage_count=len(storages),
                tensor_reference_count=sum(len(row['reference_ids']) for row in storages),
                aliased_storage_count=sum(len(row['reference_ids']) > 1 or len(row['roles']) > 1 for row in storages),
                storages=storages, scope='actual_live_transport_references_deduplicated_within_one_snapshot')
        references = record.get('communication_lifetimes', [])
        lifetimes = dict(reference_count=len(references),
            release_observed_count=sum(row.get('release_observed_sample') is not None for row in references),
            reference_alive_at_final_sample_count=sum(row.get('release_observed_sample') is None for row in references),
            role_counts=dict(Counter(row['role'] for row in references)),
            scope='observed_transport_tensor_reference_lifetimes_at_rank_local_operation_boundaries',
            raw_reference_lifetimes=references)
        history = directory / 'artifacts' / f'rank{record["rank"]}.paramorph-allocator-history.json'
        if history.is_file():
            allocator[record['rank']] = allocator_lifetimes(history)
            allocator[record['rank']]['history_may_be_truncated'] = record.get('allocator_history_may_be_truncated')
        summary[record['rank']] = dict(logical_rank=record['rank'],
            physical_rank=record.get('physical_rank', record['rank']), gpu_uuid=record.get('gpu_uuid'),
            maxima=maxima, maxima_observations=observations,
            communication_peak_storage_aliases=aliases, communication_lifetime_summary=lifetimes,
            state_inventory=record['state_inventory'],
            state_inventory_scope='unique_actual_state_storages_at_training_end',
            sample_count=len(selected), clock_domain='rank_local_monotonic',
            allocator_history_available=history.is_file())
    return dict(status='PASS', run_id=request['run_id'], ranks=summary,
        source_plan_signature=source_plan_signature,
        measured_plan_signature=plan_signature(request), result=result, raw_directory=str(directory.resolve()),
        allocator_lifetimes=allocator,
        gpu_hours=result.get('gpu_allocation_seconds', result['elapsed_seconds']) * world / 3600,
        performance_scope='instrumented_memory_measurement',
        device_peak_scope='maximum_of_synchronized_operation_boundary_current_value_samples',
        clock_alignment_scope='rank_local_monotonic_clocks_without_cross_rank_alignment',
        memory_accounting_rules=[
            '每个峰值保存对应操作边界的完整显存采样记录',
            'Parameter、Gradient、Optimizer和通信Storage属于allocator allocated覆盖的分配',
            '通信Storage按照同一采样内的真实Storage地址去重，Tensor别名共享同一次分配',
            '同一采样中device_used包含allocator reserved及该设备其他分配',
            '各字段独立最大值具有各自的采样时刻，禁止将这些独立最大值相加',
            '历史allocator peak计数器的观测时刻可能晚于峰值发生时刻',
            '训练结束状态清单和操作边界峰值分别保存，禁止直接相加或者相减推断动态峰值'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pods', nargs=2, required=True)
    parser.add_argument('--suffix', required=True)
    args = parser.parse_args()
    document = json.loads((args.search / 'system-config.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.search / 'search-result.json').read_text())
    expected_pods = {'chenny-g5-8gpu-ib-20261001', 'chenny-g6-8gpu-ib-20261010'}
    if set(args.pods) != expected_pods:
        raise ValueError('独立显存测量必须使用g5和g6各八张GPU的现有训练Pod')
    document.update(output_directory=str(args.output.resolve()), pods=args.pods,
        gpu_lock_path=str(args.output.resolve() / 'gpu.lock'))
    config = SystemConfig.model_validate(document)
    request = dict(report['best_measured']['plan']['backend_config'])
    if request['node_count'] != 2 or request['gpus_per_node'] != 8:
        raise ValueError('独立显存测量要求训练计划完整使用g5和g6的十六张GPU')
    source_signature = plan_signature(request)
    request.update(run_id=request['run_id'] + '-memory-' + args.suffix,
        warmup=2, measured=3, port=35031, paramorph_memory_audit=True,
        paramorph_serial_calibration_steps=0, paramorph_memory_history_limit=5000)
    result, directory = MegatronAdapter(config).execute(request, profile=True)
    if result['status'] != 'PASS':
        raise ValueError(f'真实显存测量训练失败：{directory}')
    output = summarize_memory_run(directory, source_signature)
    (args.output / 'memory-measurement.json').write_text(
        json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', run_id=request['run_id'], ranks=len(output['ranks']),
        gpu_hours=output['gpu_hours'], accounting='coincident_samples_and_distinct_maxima')))
