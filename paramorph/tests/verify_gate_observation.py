import argparse
import json
from pathlib import Path
import sqlite3
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'paramorph'),str(SOURCE/'experiments'),str(SOURCE)]

from megatron import core
core.__path__.append(str(SOURCE/'integration/megatron/core'))

from agents.context_budget import role_observation,compact_role_observation
from search.system_search import EvidenceGuidedSearch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config = json.loads(args.config.read_text())
    original = Path(config['output_directory'])
    for source,target in ((Path(config['profile_database']),'costs.sqlite'),
            (original/'experiments.sqlite','experiments.sqlite')):
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as reader:
            with sqlite3.connect(args.output/target) as writer:
                reader.backup(writer)
    config.update(output_directory=str(args.output),profile_database=str(args.output/'costs.sqlite'),
        gpu_execution_enabled=False,hardware_topology_paths=[str(original/'hardware-topology.json')])
    search = EvidenceGuidedSearch(config)
    parent = search.store.trial(config['request']['run_id'])
    observation = search.planning_observation(parent)
    gate = observation['formal_measurement_gate']
    expected = search.store.connection.execute("SELECT MAX(json_extract(document,'$.measurement.tokens_per_second')) "
        "FROM trials WHERE json_extract(document,'$.measurement.status')='PASS' "
        "AND json_extract(document,'$.measurement.performance_eligible')=1").fetchone()[0]
    assert gate['leader_tokens_per_second']==expected
    assert gate['parent_measured_tokens_per_second']==parent['measurement']['tokens_per_second']
    raw_observation = dict(observation)
    del raw_observation['formal_measurement_gate']
    for role in ('PerformanceAnalysisAgent','JointStrategyAgent','ConstraintValidationAgent','ExperimentInvestigationAgent'):
        context = search.agent_context(raw_observation,role)
        assert context['formal_measurement_gate']==gate
        assert role_observation(role,context)['formal_measurement_gate']==gate
        assert compact_role_observation(role,context)['formal_measurement_gate']==gate
    review = search.agent_context(dict(parent_trial_id=parent['trial_id'],candidate_review_catalog=[]),
        'ConstraintValidationAgent')
    assert review['formal_measurement_gate']==gate
    assert search.with_formal_measurement_gate(raw_observation)['formal_measurement_gate']==gate
    report = dict(status='PASS',gate=gate,verified_roles=4,new_gpu_runs=0,
        runtime_limit_seconds=config['budget']['max_walltime_seconds'])
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
