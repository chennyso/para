from collections import Counter, defaultdict
import math

from core_data_models import Task, TaskType
from graph.bottleneck_analysis import BottleneckGraph
from scheduling.schedule_simulator import union_length


DECISION_REPORT_REVISION = 'execution_decisions_v2_native_origins'


def duration_ms(record):
    value = record.get('duration_ms')
    if value is None:
        value = (record.get('cost') or {}).get('duration_ms')
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
        raise ValueError('执行分析需要有限非负任务成本')
    return value


def measured_result(trial):
    result = trial.get('measurement') or {}
    return result if result.get('status') == 'PASS' and result.get('performance_eligible') is True else None


def execution_analysis(trial, structural_graph, prediction_graph, decision_index,
                       native_config, resource_model):
    prediction = trial.get('prediction') or {}
    graph = prediction_graph if prediction_graph and prediction_graph.get('tasks') else structural_graph
    tasks = graph['tasks']
    durations = {key: duration_ms(row) for key, row in tasks.items()}
    unresolved = [key for key, value in durations.items() if value is None]
    measured = measured_result(trial)
    report = dict(revision=DECISION_REPORT_REVISION, trial_id=trial['trial_id'],
        graph_evidence_id=structural_graph['graph_evidence_id'],
        prediction_evidence_id=prediction.get('prediction_id'),
        measured_evidence_id=None if measured is None else measured['run_id'],
        measured_step_ms=None if measured is None else measured['step_seconds'] * 1000,
        status='PARTIAL_COSTS' if unresolved else 'COSTED_EXECUTION_GRAPH',
        task_count=len(tasks), known_cost_tasks=len(tasks) - len(unresolved),
        unresolved_task_count=len(unresolved), unresolved_task_samples=unresolved[:12],
        predicted_step_ms=prediction.get('performance', {}).get('step_time_ms'),
        resources=dict(Counter(resource for row in tasks.values() for resource in
            row.get('resources', row.get('resource_requirements', []))).most_common(16)),
        critical_tasks=[], bottlenecks=[], rank_occupancy=[],
        missing_measurements=prediction.get('evidence', {}).get('missing_profiles', [])[:8],
        scope='native_training_tasks_with_explicit_opaque_region_internal_collectives')
    groups = defaultdict(list)
    for key, row in tasks.items():
        groups[(row.get('region', row.get('region_id')), row.get('kind', row.get('operation')))].append(key)
    report['unresolved_groups'] = [dict(region_id=region, operation=operation,
        unresolved_tasks=sum(durations[key] is None for key in keys))
        for (region, operation), keys in groups.items() if any(durations[key] is None for key in keys)][:16]
    report['known_work'] = sorted([dict(region_id=region, operation=operation,
        known_work_ms=sum(durations[key] for key in keys if durations[key] is not None),
        known_tasks=sum(durations[key] is not None for key in keys), total_tasks=len(keys))
        for (region, operation), keys in groups.items()],
        key=lambda row: (-row['known_work_ms'], str(row['region_id']), row['operation']))[:12]
    report['known_work_scope'] = 'known_components_only_not_full_iteration_ranking'
    if unresolved:
        report['decision_status'] = 'USE_MEASURED_RESULTS_AND_NATIVE_STRUCTURE_OR_REQUEST_COSTS'
        report['counterfactual_status'] = 'REQUIRES_COMPLETE_COSTS'
        return report
    graph_query = BottleneckGraph(graph, [prediction.get('prediction_id') or structural_graph['graph_evidence_id']])
    replay = graph_query.replay(config=native_config, resource_model=resource_model)
    report['replay_count'] = 1
    report['raw_replayed_step_ms'] = replay['step_time_ms']
    report['measurement_alignment'] = dict(
        measured_step_ms=None if measured is None else measured['step_seconds'] * 1000,
        raw_replayed_step_ms=replay['step_time_ms'],
        raw_relative_residual=(None if measured is None else replay['step_time_ms']
            / (measured['step_seconds'] * 1000) - 1),
        compatible_anchor_id=prediction.get('evidence', {}).get('compatible_anchor_id'),
        scope='task_paths_come_from_profiling_replay_iteration_anchor_is_a_separate_measurement')
    critical = set(replay['critical_task_ids'])
    report['critical_tasks'] = [dict(task_id=key, duration_ms=durations[key],
        rank=tasks[key]['rank'], region_id=tasks[key].get('region', tasks[key].get('region_id')),
        resources=tasks[key].get('resources', [])) for key in
        sorted(critical, key=lambda key: (-durations[key], key))[:12]]
    occupancy = defaultdict(lambda: dict(compute=[], communication=[], ready_delays=[]))
    for key, row in tasks.items():
        start, end = replay['task_times'][key]
        ready = max((replay['task_times'][parent][1] for parent in row['dependencies']), default=0)
        operation = row.get('kind', row.get('operation'))
        task = Task(key, row['rank'], TaskType(operation))
        category = 'communication' if task.is_communication else 'compute'
        if end > start:
            occupancy[row['rank']][category].append((start, end))
        occupancy[row['rank']]['ready_delays'].append(max(0, start - ready))
    report['rank_occupancy'] = [dict(rank=rank, compute_busy_ms=union_length(value['compute']),
        communication_busy_ms=union_length(value['communication']),
        ready_to_start_delay_ms=sum(value['ready_delays'])) for rank, value in sorted(occupancy.items())]
    ordered_groups = sorted(groups, key=lambda group: (
        -sum(durations[key] for key in groups[group] if key in critical), str(group)))
    for region, operation in ordered_groups[:3]:
        selected = groups[(region, operation)]
        if not any(key in critical for key in selected):
            continue
        targets = graph_query.closure(selected, dependency_depth=0, include_resources=False)
        closure = graph_query.causal_closure(targets['task_ids'], replay, dependency_depth=1)
        impact = graph_query.epsilon_impact(targets['task_ids'], 0.1,
            config=native_config, resource_model=resource_model, baseline_replay=replay)
        report['replay_count'] += impact['replay_count']
        controlled = []
        affected = set(targets['task_ids'])
        for name, entry in decision_index.items():
            native_affected = set(entry['affected_task_ids'])
            overlap = {key for key in affected if key in native_affected
                or tasks[key].get('metadata', {}).get('origin_task') in native_affected}
            if overlap:
                controlled.append(dict(parameter=name, affected_target_count=len(overlap),
                    coupled_parameters=entry['hard_constraint_dependencies'].get(name, [])))
        report['bottlenecks'].append(dict(region_id=region, operation=operation,
            target_task_ids=selected[:8], target_task_count=len(selected),
            resources=sorted({resource for key in selected for resource in tasks[key].get('resources', [])}),
            controlling_parameters=controlled, epsilon=0.1,
            local_cost_reduction_ms=sum(durations[key] for key in targets['task_ids']) * 0.1,
            context_task_count=len(closure['task_ids']),
            blocking_context=[dict(task_id=key, reasons=closure['inclusion_reasons'][key])
                for key in closure['task_ids'] if key not in targets['task_ids']][:8],
            target_wait_causes={key: dict(
                dependency_ready_ms=impact['target_wait_causes'][key]['dependency_ready_ms'],
                resource_wait_ms=impact['target_wait_causes'][key]['resource_wait_ms'],
                dependency_task_ids=impact['target_wait_causes'][key]['dependency_task_ids'][:2],
                resource_blocker_count=len(impact['target_wait_causes'][key]['resource_blockers']),
                resource_blockers=sorted(impact['target_wait_causes'][key]['resource_blockers'],
                    key=lambda row: row['start_ms'] - row['end_ms'])[:2]) for key in selected[:1]},
            full_iteration_impact_ms=impact['impact_ms'],
            evidence_type='PREDICTED_COUNTERFACTUAL', measured_strategy_gain=False))
    report['bottlenecks'].sort(key=lambda row: (-row['full_iteration_impact_ms'], str(row['region_id']), row['operation']))
    report['counterfactual_status'] = 'COMPLETE_RESOURCE_AWARE_REPLAY'
    report['decision_status'] = 'COMPARE_COMPLETE_PREDICTIONS_AND_COUNTERFACTUAL_TARGETS'
    return report


def candidate_decisions(parent, candidates):
    base = measured_result(parent)
    base_request = parent['plan']['backend_config']
    rows = []
    for trial in candidates:
        prediction = trial.get('prediction') or {}
        result = measured_result(trial)
        validation = (trial.get('validation') or {}).get('load_validation') or {}
        request = trial['plan']['backend_config']
        elapsed = prediction.get('performance', {}).get('step_time_ms')
        admission = prediction.get('memory', {}).get('admission')
        failure = trial.get('failure')
        state = ('MEASURED' if result is not None else 'EXCLUDED' if failure or admission == 'REJECT_OOM'
            else 'READY_FOR_BENCHMARK' if validation.get('status') == 'PASS'
            else 'PREDICTED' if elapsed is not None else 'UNRESOLVED')
        rows.append(dict(trial_id=trial['trial_id'], state=state,
            predicted_step_ms=elapsed, memory_admission=admission,
            measured_tokens_per_second=None if result is None else result['tokens_per_second'],
            measured_gain_vs_parent=None if result is None or base is None else
                result['tokens_per_second'] / base['tokens_per_second'] - 1,
            measured_step_ratio_vs_parent=None if result is None or base is None else
                result['step_seconds'] / base['step_seconds'],
            measured_comparison=None if result is None or base is None else (
                'HIGHER_THROUGHPUT' if result['tokens_per_second'] > base['tokens_per_second']
                else 'LOWER_THROUGHPUT' if result['tokens_per_second'] < base['tokens_per_second']
                else 'EQUAL_THROUGHPUT'),
            measured_peak_gib=None if result is None else result.get('peak_allocated_gib'),
            load_validation_status=validation.get('status'),
            changed_settings={name: dict(before=base_request.get(name), after=request.get(name))
                for name in ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'expert_dp', 'vpp', 'micro_batch',
                    'group', 'decoder_chunks', 'recompute', 'memory_policy', 'pipeline_schedule',
                    'overlap_p2p', 'optimizer_sharding', 'optimizer_offload_fraction',
                    'optimizer_transfer_overlap', 'execution_backend',
                    'dense_sharding_strategy', 'expert_sharding_strategy',
                    'dp_inner_size', 'dp_outer_size', 'expert_dp_inner_size', 'expert_dp_outer_size',
                    'rank_mapping')
                if base_request.get(name) != request.get(name)},
            missing_measurements=prediction.get('evidence', {}).get('missing_profiles', [])[:4],
            evidence_refs=[value for value in (prediction.get('prediction_id'),
                (result or {}).get('run_id'), validation.get('run_id')) if value]))
    return dict(revision=DECISION_REPORT_REVISION, parent_trial_id=parent['trial_id'],
        parent_measurement_evidence_id=None if base is None else base['run_id'], candidates=rows,
        predicted_order=[row['trial_id'] for row in sorted(rows,
            key=lambda row: (row['predicted_step_ms'] is None, row['predicted_step_ms'] or 0, row['trial_id']))
            if row['state'] == 'PREDICTED'],
        ready_for_benchmark=[row['trial_id'] for row in rows if row['state'] == 'READY_FOR_BENCHMARK'],
        unresolved_candidates=[row['trial_id'] for row in rows if row['state'] == 'UNRESOLVED'],
        excluded_candidates=[row['trial_id'] for row in rows if row['state'] == 'EXCLUDED'],
        ranking_scope='predicted_order_excludes_unknown_costs_and_actual_failures',
        selection_owner='ExperimentInvestigationAgent')
