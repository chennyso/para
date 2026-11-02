import argparse
import json
from itertools import product
import math
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1] / 'experiments')]

from run_physical_stage_comparison import prepare
from search.physical_stage_master import NativeEventMaster, PartitionDomain
from search.physical_stage_search import NativeCompositePricingOracle


def longest_path(tasks, columns):
    durations = {key: value for column in columns for key, value in column.durations.items()}
    ends = {}
    while len(ends) < len(tasks):
        ready = [key for key, row in tasks.items() if key not in ends
            and all(parent in ends for parent in row['dependencies'])]
        assert ready
        for key in ready:
            ends[key] = durations[key] + max((ends[parent] for parent in tasks[key]['dependencies']), default=0)
    return max(ends.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictor, rewriter, request = prepare(args.workspace, 'mimo', args.output)
    encoded = predictor.database.connection.execute('SELECT document FROM profiles WHERE id=?',
        ('paramorph-mimo-optimizer-cost-20261011-r1:raw',)).fetchone()[0]
    profile = json.loads(encoded)
    request = dict(profile['request'])
    request.update(main_grads_dtype='float32', main_params_dtype='float32',
        exp_avg_dtype='float32', exp_avg_sq_dtype='float32')
    full_domain = PartitionDomain.around(request, 1)
    variable_boundaries = {1, 2, 3, 5, 6, 7}
    domain = PartitionDomain(full_domain.pp, full_domain.vpp,
        tuple(values if index in variable_boundaries else (full_domain.seed[index],)
            for index, values in enumerate(full_domain.boundaries)), full_domain.seed)
    exhaustive_layouts = [values for values in product(*domain.boundaries)
        if all(a < b for a, b in zip(values, values[1:]))]
    assert domain.layout_count() == len(exhaustive_layouts)
    for stage in range(domain.pp):
        projected = {domain.intervals(values, stage) for values in exhaustive_layouts}
        enumerated = set(domain.columns(stage))
        assert projected == enumerated
        for intervals in enumerated:
            witness = domain.witness(stage, intervals)
            assert domain.intervals(witness, stage) == intervals
    oracle = NativeCompositePricingOracle(domain, request, predictor, rewriter, args.output, profile)
    master = NativeEventMaster(domain, oracle.tasks, oracle.config.world_size, oracle.revision)
    layouts = [domain.seed]
    for index in (2, 6):
        endpoints = list(domain.seed)
        endpoints[index] += 1
        layouts.append(tuple(endpoints))
    endpoints = list(domain.seed)
    endpoints[2] += 1
    endpoints[6] += 1
    layouts.append(tuple(endpoints))
    for layout in layouts:
        for stage in range(domain.pp):
            master.add(oracle.column(stage, domain.intervals(layout, stage)))
    # 同一物理阶段的成本不得依赖其他阶段端点。
    invariant_checks = 0
    for layout in layouts[1:]:
        report = oracle.snapshot(layout)
        for stage in range(domain.pp):
            if domain.intervals(layout, stage) != domain.intervals(domain.seed, stage):
                continue
            for key, task in oracle.tasks.items():
                if task['rank'] // master.lanes == stage:
                    assert math.isclose(task['duration_ms'],
                        report['execution_graph']['tasks'][key]['duration_ms'], abs_tol=1e-7)
                    invariant_checks += 1
    lp = master.solve_lp()
    integer = master.solve_integer()
    buckets = [[column for column in master.columns.values() if column.stage == stage]
        for stage in range(domain.pp)]
    legal = []
    for columns in product(*buckets):
        fragments = {stage: column.intervals for stage, column in enumerate(columns)}
        joined = [fragments[j % domain.pp][j // domain.pp] for j in range(domain.pp * domain.vpp)]
        if joined[0][0] == 0 and joined[-1][1] == domain.seed[-1] and all(
                left[1] == right[0] for left, right in zip(joined, joined[1:])):
            legal.append((longest_path(oracle.tasks, columns), [column.identifier for column in columns]))
    assert legal and math.isclose(integer['objective_ms'], min(value for value, _ in legal), abs_tol=1e-5)
    assert lp['objective_ms'] <= integer['objective_ms'] + 1e-5
    assert math.isclose(integer['objective_ms'],
        oracle.snapshot(integer['endpoints'])['performance']['step_time_ms'], abs_tol=1e-5)
    following = master.solve_integer([integer['column_ids']])
    assert following is not None and set(following['column_ids']) != set(integer['column_ids'])
    assert math.isclose(master.solve_lp()['objective_ms'], lp['objective_ms'], abs_tol=1e-7)
    assert all(master.reduced_cost(column, lp['prices']) >= -1e-5 for column in master.columns.values())
    complete = oracle.complete(integer['endpoints'], [oracle.revision])
    assert complete['prediction_status'] == 'NEEDS_PROFILE'
    assert complete['step_time_ms'] is None and complete['missing_cost_request_count'] > 0
    seed_complete = oracle.complete(domain.seed, [oracle.revision])
    assert seed_complete['decoder_chunks'] == request['decoder_chunks']
    result = dict(status='PASS', complete_layout_count=domain.layout_count(),
        physical_columns=len(master.columns), enumerated_compatible_combinations=len(legal),
        event_count=len(oracle.tasks), event_cost_invariance_checks=invariant_checks,
        lp_objective_ms=lp['objective_ms'], integer_objective_ms=integer['objective_ms'],
        reduced_cost_solver_residual=lp['maximum_reduced_cost_residual'],
        integer_exclusion_preserves_pricing_lp=True, native_compilation=complete,
        new_gpu_runs=0, data_source='actual_MiMo_composite_profiles_and_native_pipeline',
        full_checker_preserves_missing_cost_status=True)
    (args.output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'native_compilation'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
