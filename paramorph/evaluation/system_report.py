import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.prediction_metrics import (
    comparable_identity, measured_trials, prediction_metrics, prediction_provenance, prediction_samples,
)


def _search_efficiency(store, trials):
    charges = [dict(order=row[0], charge_id=row[1], gpu_hours=row[2]) for row in
               store.connection.execute("SELECT rowid, id, amount FROM charges WHERE resource='gpu_hours' ORDER BY rowid")]
    if any(not math.isfinite(row['gpu_hours']) or row['gpu_hours'] < 0 for row in charges):
        raise ValueError('搜索效率曲线需要有限非负的真实GPU费用')
    trial_by_charge = {row['plan_hash'] + ':gpu': row for row in trials}
    profile_by_charge = {row['request_id'] + ':gpu': row for row in store.profile_requests()}
    charged_ids = {row['charge_id'] for row in charges}
    successful = measured_trials(trials)
    eligible_ids = {row['trial_id'] for row in successful}
    reused = [row for row in successful if row['plan_hash'] + ':gpu' not in charged_ids
              and row.get('parent_trial_id') is None and row.get('hypothesis') is None]
    unaccounted = [row['trial_id'] for row in successful if row['plan_hash'] + ':gpu' not in charged_ids
                   and row not in reused]
    groups, best, events = {}, {}, []

    def group_for(trial):
        identity = comparable_identity(trial)
        if identity not in groups:
            groups[identity] = dict(group_id=f'group_{len(groups) + 1}',
                condition_identity=dict(zip(('model_fingerprint', 'hardware_fingerprint',
                                             'workload_fingerprint', 'backend_fingerprint'), identity)),
                model=trial['plan']['backend_config'].get('model'), points=[])
        return identity

    for row in reused:
        identity = group_for(row)
        throughput = row['measurement']['tokens_per_second']
        best[identity] = max(best.get(identity, 0), throughput)
        events.append(dict(kind='reused_initial_measurement', trial_id=row['trial_id'],
                           run_id=row['measurement']['run_id'], gpu_hours=0,
                           cumulative_gpu_hours=0, accounting_scope='prior_measurement_outside_current_search_budget'))
    for identity, value in best.items():
        groups[identity]['points'].append(dict(cumulative_gpu_hours=0, best_measured_tokens_per_second=value))
    cumulative = 0
    for charge in charges:
        cumulative += charge['gpu_hours']
        event = dict(charge, cumulative_gpu_hours=cumulative, kind='unclassified_gpu_charge')
        trial = trial_by_charge.get(charge['charge_id'])
        profile = profile_by_charge.get(charge['charge_id'])
        if trial is not None:
            measurement = trial.get('measurement') or {}
            event.update(kind='full_training' if trial['trial_id'] in eligible_ids else 'full_training_failure_or_ineligible',
                         trial_id=trial['trial_id'], run_id=measurement.get('run_id'),
                         status=measurement.get('status', trial['state']),
                         gpu_allocation_seconds=measurement.get('gpu_allocation_seconds'))
            if trial['trial_id'] in eligible_ids:
                identity = group_for(trial)
                best[identity] = max(best.get(identity, 0), measurement['tokens_per_second'])
        elif profile is not None:
            result = profile.get('result') or {}
            measurement = result.get('measurement', result)
            event.update(kind='profiling', request_id=profile['request_id'], state=profile['state'],
                         measurement_type=profile['measurement_type'], run_id=measurement.get('run_id'),
                         status=measurement.get('status'), gpu_allocation_seconds=measurement.get('gpu_allocation_seconds'))
        for identity, value in best.items():
            groups[identity]['points'].append(dict(cumulative_gpu_hours=cumulative,
                best_measured_tokens_per_second=value, charge_id=charge['charge_id']))
        events.append(event)
    return dict(x_axis='cumulative_gpu_hours', y_axis='best_measured_tokens_per_second',
        cost_source='ExperimentStore.charges.resource=gpu_hours_in_insertion_order',
        chronology_scope='GPU_cost_completion_record_order',
        cost_scope='all_recorded_gpu_allocations_including_profiling_and_failed_runs',
        profiling_gpu_hours_role='subset_of_gpu_hours_not_added_twice',
        cumulative_gpu_hours=cumulative, charge_count=len(charges),
        reused_initial_measurements=len(reused), unaccounted_measured_trial_ids=unaccounted,
        unclassified_gpu_charge_count=sum(row['kind'] == 'unclassified_gpu_charge' for row in events),
        comparable_groups=list(groups.values()), events=events)


def generate_report(store, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence_order = {identifier: order for order, identifier in
                      store.connection.execute('SELECT rowid, id FROM evidence ORDER BY rowid')}
    trials = [dict(row, prediction_provenance_assessment=prediction_provenance(row, evidence_order))
              for row in store.trials()]
    measured = measured_trials(trials)
    tested = prediction_samples(trials)
    independent = [row for row in tested if row['prediction_provenance_assessment']['independent']]
    metrics = prediction_metrics(trials)
    independent_metrics = prediction_metrics(independent)
    throughput_errors = [1000 * row['measurement']['step_seconds'] / row['prediction']['performance']['step_time_ms'] - 1
                         for row in independent]
    efficiency = _search_efficiency(store, trials)
    report = dict(real_measurements=len(measured), recorded_predictions=len(tested),
        independent_predictions=len(independent),
        blind_predictions=sum(row['prediction_provenance_assessment'].get('blind', False) for row in independent),
        time_mape=independent_metrics['step_time_mape'],
        throughput_mape=statistics.mean(abs(value) for value in throughput_errors) if throughput_errors else None,
        pairwise_count=independent_metrics['pairwise_count'],
        pairwise_ranking_accuracy=independent_metrics['pairwise_ranking_accuracy'],
        prediction_provenance=[dict(trial_id=row['trial_id'], prediction_id=row['prediction']['prediction_id'],
                                   **row['prediction_provenance_assessment']) for row in tested],
        usage=store.usage(), best_measured=max(measured,
            key=lambda row: row['measurement']['tokens_per_second'], default=None),
        numerical_equivalence_certified=all(row['measurement'].get('numerical_equivalence_certified', False) for row in measured) if measured else False,
        coverage_scope='recorded_real_trials_only', prediction_metrics=metrics,
        independent_prediction_metrics=independent_metrics, search_efficiency=efficiency)
    if tested:
        figure, axis = plt.subplots(figsize=(5, 4))
        actual = [row['measurement']['step_seconds'] for row in tested]
        predicted = [row['prediction']['performance']['step_time_ms'] / 1000 for row in tested]
        axis.scatter(actual, predicted, label='Recorded predictions')
        bounds = [min(actual + predicted), max(actual + predicted)]
        axis.plot(bounds, bounds, color='black', linewidth=1)
        axis.set(xlabel='Measured Step Time (s)', ylabel='Predicted Step Time (s)')
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / 'prediction-accuracy.pdf')
        figure.savefig(output / 'prediction-accuracy.png', dpi=180)
        plt.close(figure)
    else:
        for name in ('prediction-accuracy.pdf', 'prediction-accuracy.png'):
            (output / name).unlink(missing_ok=True)
    curves = [group for group in efficiency['comparable_groups'] if group['points']]
    if curves:
        figure, axis = plt.subplots(figsize=(6, 4))
        for group in curves:
            axis.step([point['cumulative_gpu_hours'] for point in group['points']],
                      [point['best_measured_tokens_per_second'] for point in group['points']],
                      where='post', label=group['group_id'])
        axis.set(xlabel='Cumulative GPU Hours', ylabel='Best Measured Tokens/s')
        if len(curves) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / 'search-efficiency.pdf')
        figure.savefig(output / 'search-efficiency.png', dpi=180)
        plt.close(figure)
    else:
        for name in ('search-efficiency.pdf', 'search-efficiency.png'):
            (output / name).unlink(missing_ok=True)
    with (output / 'search-efficiency.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group_id', 'cumulative_gpu_hours', 'best_measured_tokens_per_second', 'charge_id'])
        writer.writeheader()
        for group in curves:
            for point in group['points']:
                writer.writerow(dict(group_id=group['group_id'], **point))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report
