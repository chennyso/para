import argparse
import csv
import json
from pathlib import Path
import sqlite3

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_progress(directory):
    database = directory / 'experiments.sqlite'
    connection = sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True)
    trials = {row[0]: json.loads(row[1]) for row in connection.execute('SELECT id, document FROM trials')}
    rounds = [json.loads(row[0]) for row in connection.execute('SELECT document FROM search_rounds ORDER BY id')]
    initial = [json.loads(row[0]) for row in connection.execute(
        "SELECT document FROM search_state_repository WHERE kind='INITIAL_CONFIGURATION'")]
    connection.close()
    if len(initial) != 1:
        raise ValueError(f'搜索记录需要唯一初始配置：{directory}')
    initial_trial = next(row for row in trials.values() if row['plan_hash'] == initial[0]['plan_hash'])
    points = []
    best = 0.0
    seen = set()
    events = [(0, [initial_trial['trial_id']])]
    for row in rounds:
        if row['round_index'] < 0 or row['stage'] != 'COMPLETE':
            continue
        identifiers = row.get('measured_candidate_ids')
        if identifiers is None:
            refs = set(row.get('feedback', {}).get('evidence_refs', []))
            identifiers = [identifier for identifier in row.get('candidate_ids', [])
                if (trials[identifier].get('measurement') or {}).get('run_id') in refs]
        events.append((row['round_index'] + 1, identifiers))
    for iteration, identifiers in events:
        measured = []
        failed = []
        for identifier in identifiers:
            if identifier in seen:
                continue
            trial = trials[identifier]
            measurement = trial.get('measurement')
            if measurement is None:
                continue
            seen.add(identifier)
            if measurement['status'] == 'PASS' and measurement.get('performance_eligible', False):
                value = float(measurement['tokens_per_second'])
                if value <= 0:
                    raise ValueError('正式实测吞吐必须大于零')
                measured.append(dict(trial_id=identifier, tokens_per_second=value))
            else:
                failed.append(dict(trial_id=identifier, status=measurement['status']))
        if measured:
            best = max(best, max(row['tokens_per_second'] for row in measured))
        points.append(dict(iteration=iteration, best_tokens_per_second=best or None,
            measurements=measured, failures=failed))
    request = initial_trial['plan']['backend_config']
    initial_value = points[0]['best_tokens_per_second']
    for point in points:
        point['normalized_best'] = (point['best_tokens_per_second'] / initial_value
            if initial_value and point['best_tokens_per_second'] else None)
    return dict(directory=str(directory.resolve()), model=request['name'],
        sequence=request['sequence'], global_batch=request['global_batch'],
        gpu_count=request['node_count'] * request['gpus_per_node'],
        initial_source=initial[0].get('source'), points=points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    curves = [load_progress(path) for path in args.search]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'search-progress.json').write_text(json.dumps(curves, indent=2, allow_nan=False) + '\n')
    with (args.output / 'search-progress.csv').open('w', newline='') as stream:
        fields = ['model', 'sequence', 'gpu_count', 'iteration', 'best_tokens_per_second', 'normalized_best']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for curve in curves:
            for point in curve['points']:
                writer.writerow({key: curve.get(key, point.get(key)) for key in fields})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'savefig.dpi': 300})
    for normalized in (False, True):
        columns = min(4, len(curves))
        rows = (len(curves) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.2 * rows), squeeze=False,
            layout='constrained')
        for ax, curve in zip(axes.flat, curves):
            key = 'normalized_best' if normalized else 'best_tokens_per_second'
            points = [point for point in curve['points'] if point[key] is not None]
            ax.set_title(f"{curve['model']}\n{curve['sequence'] // 1024}K · {curve['gpu_count']} GPUs")
            if points:
                x, y = [point['iteration'] for point in points], [point[key] for point in points]
                ax.step(x, y, where='post', color='#e44242', linewidth=1.8, label='Best so far')
                ax.scatter(x, y, color='#e44242', s=24, zorder=3)
                ax.axhline(max(y), color='#3995de', linestyle='--', linewidth=1, label='Best observed')
                ax.set_xlim(-0.15, max(1, max(x)) + 0.15)
                if normalized:
                    ax.set_ylim(min(0.98, min(y) - 0.02), max(y) + 0.04)
                ax.annotate(f'{y[-1]:.3f}×' if normalized else f'{y[-1]:,.0f}',
                    (x[-1], y[-1]), xytext=(-5, 8), textcoords='offset points', ha='right', fontsize=9)
                ax.legend(frameon=False, fontsize=8, loc='lower right')
            else:
                ax.text(0.5, 0.5, 'Awaiting valid initial measurement' if normalized
                    else 'Awaiting valid measurement', transform=ax.transAxes, ha='center', fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Search iteration (0 = initial configuration)')
            ax.set_ylabel('Normalized best measured throughput' if normalized else 'Best measured throughput (tokens/s)')
            ax.grid(color='#dddddd', alpha=0.55, linewidth=0.6)
            ax.set_axisbelow(True)
        for ax in list(axes.flat)[len(curves):]:
            ax.set_visible(False)
        name = 'search-progress-normalized' if normalized else 'search-progress-throughput'
        for extension in ('png', 'pdf'):
            fig.savefig(args.output / f'{name}.{extension}', bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(dict(output=str(args.output.resolve()),
        completed_iterations=[len(curve['points']) - 1 for curve in curves])), flush=True)


if __name__ == '__main__':
    main()
