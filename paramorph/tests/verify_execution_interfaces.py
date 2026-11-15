import argparse
from itertools import product
import json
import math
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1])]

from contracts import fingerprint
from search.impact_neighborhood import cross_stage_interfaces, execution_neighborhood
from search.physical_stage_master import NativeEventMaster, PartitionDomain, PhysicalStageColumn


def counts(report):
    regions = {}
    for task in report['execution_graph']['tasks'].values():
        if task['kind'] == 'forward_compute':
            layers = tuple(task['metadata']['layer_ids'])
            if task['region'] in regions:
                assert regions[task['region']] == layers
            regions[task['region']] = layers
    assert set(regions) == set(range(len(regions)))
    return [len(regions[index]) for index in range(len(regions))]


def path_time(tasks,columns):
    durations = {key:value for column in columns for key,value in column.durations.items()}
    finished = {}
    while len(finished) != len(tasks):
        ready = [key for key,row in tasks.items() if key not in finished
            and all(parent in finished for parent in row['dependencies'])]
        assert ready
        for key in ready:
            finished[key] = durations[key]+max((finished[parent]
                for parent in tasks[key]['dependencies']),default=0)
    return max(finished.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in (args.parent,args.candidate)]
    request = json.loads((args.parent.parent/'inputs.json').read_text())['request']
    pp,vpp = request['pp'],request['vpp']
    assert len(counts(reports[0])) == pp*vpp
    endpoints = []
    for report in reports:
        boundaries = [0]
        for value in counts(report):
            boundaries.append(boundaries[-1]+value)
        endpoints.append(tuple(boundaries))
    domain = PartitionDomain(pp,vpp,tuple(tuple(sorted({row[index] for row in endpoints}))
        for index in range(len(endpoints[0]))),endpoints[0])
    tasks = reports[0]['execution_graph']['tasks']
    world = 1+max(row['rank'] for row in tasks.values())
    revision = fingerprint([row['cost_database_revision'] for row in reports])
    master = NativeEventMaster(domain,tasks,world,revision)
    for report,layout in zip(reports,endpoints):
        graph = report['execution_graph']
        assert set(graph['tasks']) == set(tasks)
        assert all(set(graph['tasks'][key]['dependencies']) == set(tasks[key]['dependencies'])
            for key in tasks)
        interfaces = cross_stage_interfaces(graph,master.lanes)
        assert interfaces
        for stage in range(pp):
            intervals = domain.intervals(layout,stage)
            local = {key:row['duration_ms'] for key,row in graph['tasks'].items()
                if row['rank']//master.lanes == stage}
            column = PhysicalStageColumn(fingerprint(dict(stage=stage,intervals=intervals,
                prediction_id=report['prediction_id'])),stage,intervals,local,{},True,
                dict(cost_revision=revision,cross_stage_interfaces={key:value
                    for key,value in interfaces.items() if stage in value['stages']}))
            master.add(column)
    lp = master.solve_lp()
    integer = master.solve_integer()
    feasible = []
    buckets = [[column for column in master.columns.values() if column.stage==stage] for stage in range(pp)]
    for selected in product(*buckets):
        intervals = [selected[index % pp].intervals[index // pp] for index in range(pp*domain.vpp)]
        if any(left[1] != right[0] for left,right in zip(intervals,intervals[1:])):
            continue
        compatible = all(len({column.evidence['cross_stage_interfaces'][key]['signature']
            for column in selected if column.stage in stages}) == 1
            for key,stages in master.interfaces.items())
        if compatible:
            feasible.append(path_time(tasks,selected))
    assert feasible and math.isclose(integer['objective_ms'],min(feasible),abs_tol=1e-5)
    assert lp['maximum_reduced_cost_residual'] < 1e-5
    parent_request = dict(request,decoder_chunks=counts(reports[0]))
    candidate_request = dict(parent_request,decoder_chunks=counts(reports[1]))
    neighborhood = execution_neighborhood(*reports,parent_request,candidate_request,revision).report()
    assert neighborhood['live_buffers'] and neighborhood['collective_interfaces']
    assert neighborhood['requires_complete_replay']
    result = dict(status='PASS',source_predictions=[str(args.parent),str(args.candidate)],
        cross_stage_interfaces=len(master.interfaces),interface_equality_rows=len(master.interface_rows),
        compatible_combinations=len(feasible),integer_objective_ms=integer['objective_ms'],
        reduced_cost_solver_residual=lp['maximum_reduced_cost_residual'],
        buffer_records=len(neighborhood['live_buffers']),boundary_size=neighborhood['boundary_size'],
        new_gpu_runs=0)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
