import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import time


SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / 'paramorph'), str(SOURCE / 'experiments'), str(SOURCE)]

from megatron import core
core.__path__.append(str(SOURCE / 'integration/megatron/core'))

from search.measurement_review import structural_validation_key
from search.graph_action_compiler import GraphActionCompiler
from search.investigation_control import PlanningInvestigationBatch
from search.system_search import EvidenceGuidedSearch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--primary-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--agent-records', type=Path)
    parser.add_argument('--joint-receipt', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    original = Path(config['output_directory'])
    for source, name in ((original / 'experiments.sqlite', 'experiments.sqlite'),
                         (Path(config['profile_database']), 'costs.sqlite')):
        with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as reader:
            with sqlite3.connect(args.output / name) as writer:
                reader.backup(writer)
    config.update(output_directory=str(args.output), profile_database=str(args.output / 'costs.sqlite'),
        gpu_execution_enabled=False, hardware_topology_paths=[str(original / 'hardware-topology.json')])

    class VerifiedContextSearch(EvidenceGuidedSearch):
        def planning_leader_rate(self):
            with sqlite3.connect((args.primary_case / 'experiments.sqlite').resolve().as_uri() + '?mode=ro',
                                 uri=True) as database:
                leader = database.execute("SELECT MAX(json_extract(document,'$.measurement.tokens_per_second')) "
                    "FROM trials WHERE json_extract(document,'$.measurement.status')='PASS' "
                    "AND json_extract(document,'$.measurement.performance_eligible')=1").fetchone()[0]
            return max(super().planning_leader_rate() or 0, leader or 0)

    search = VerifiedContextSearch(config)
    parent = search.store.trial(config['request']['run_id'])
    recorded_round = json.loads((original / 'round-reports/round-001.json').read_text())['checkpoint']
    identifiers = recorded_round['candidate_ids']
    round_index = recorded_round['round_index']
    candidates = [search.store.trial(identifier) for identifier in identifiers]
    keys = [structural_validation_key(parent, trial, search.backend_identity,
        search.domains, search.topology.num_gpus, search.constraint_implementation) for trial in candidates]
    assert len(set(keys)) == len(candidates)
    before = search.store.usage()
    leader = search.planning_leader_rate()
    rejections = {trial['trial_id']: search.planning_benchmark_rejection(trial, leader,
        {trial['plan_hash']}) for trial in candidates}
    benchmark_ids = {identifier for identifier, reason in rejections.items() if reason is None}
    profiles = [row for row in search.store.profile_requests() if row['state'] == 'PENDING'
                and set(identifiers).intersection(row['trial_ids'])]
    started = time.perf_counter()
    first = search.review_measurement_candidates(parent, candidates, benchmark_ids, profiles, set(), round_index)
    first_seconds = time.perf_counter() - started
    count = len(search.store.search_states('PLANNING_NATIVE_CONFIGURATION_VALIDATION'))
    started = time.perf_counter()
    second = search.review_measurement_candidates(parent, candidates, benchmark_ids, profiles, set(), round_index)
    reused_seconds = time.perf_counter() - started
    assert first == second
    assert count == len(candidates)
    assert len(search.store.search_states('PLANNING_NATIVE_CONFIGURATION_VALIDATION')) == count
    assert search.store.usage().get('agent_calls', 0) == before.get('agent_calls', 0)
    assert search.store.usage().get('gpu_hours', 0) == before.get('gpu_hours', 0)
    assert first['formal_measurement_gate']['leader_tokens_per_second'] == leader
    failures = [row for row in search.store.trials() if (row.get('failure') or {}).get('failure_type') == 'CAPACITY']
    assert failures and all(not search.available_for_joint_measurement(row) for row in failures)
    joint_shapes = []
    if args.agent_records is not None:
        records = json.loads(gzip.decompress(args.agent_records.read_bytes()))['records']
        context = next(row['observation'] for row in records
            if row['original_request']['role'] == 'JointStrategyAgent'
            and 'graph_strategy_impact' in row['observation'])
        response = json.loads(args.joint_receipt.read_text())
        batch = PlanningInvestigationBatch.model_validate(response['response'])
        assert response['ephemeral'] and batch.actions and not batch.tool_requests
        search._planning_graph_impact = context['graph_strategy_impact']
        search._planning_quantitative_evidence = context['graph_quantitative_evidence']
        for action in batch.actions:
            shapes, rejected = search.investigation_shapes(action, parent)
            assert shapes
            for shape in shapes:
                GraphActionCompiler(search._planning_graph_impact).compile(action.graph_intent,
                    parent['plan']['backend_config'], shape,
                    action.parallel_ranges.model_dump(exclude_none=True),
                    action.runtime_ranges.model_dump(exclude_none=True))
            joint_shapes.append(dict(native_shapes=len(shapes), rejected_shapes=len(rejected),
                parallel_ranges=action.parallel_ranges.model_dump(mode='json'),
                runtime_ranges=action.runtime_ranges.model_dump(mode='json')))
    report = dict(status='PASS', real_candidates=identifiers, leader_tokens_per_second=leader,
        current_formal_rejections=rejections, first_native_validation_seconds=first_seconds,
        reused_native_validation_seconds=reused_seconds, unchanged_configuration_certificates=count,
        actual_capacity_failures_excluded=[row['trial_id'] for row in failures],
        authenticated_joint_actions=joint_shapes,
        new_agent_calls=0, new_gpu_runs=0, source_databases_modified=False,
        formal_gate_recomputed=True, runtime_limit_seconds=config['budget']['max_walltime_seconds'])
    (args.output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
