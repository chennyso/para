import argparse
import json
from pathlib import Path
import sqlite3

from predictor.profile_database import profile_execution_request
from search.agent_decision_protocol import exact_native_observation
from search.system_search import evidenced_selection_schema, profile_hardware_matches
from storage.experiment_store import ExperimentStore


def open_read_only(path):
    return sqlite3.connect(path.resolve(strict=True).as_uri() + '?mode=ro', uri=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--feedback', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    directory = args.directory
    config = json.loads((directory / 'system-config.json').read_text())
    hardware = json.loads(Path(config['hardware_topology_paths'][0]).read_text())
    devices = [row | dict(physical_rank=node * config['request']['gpus_per_node'] + row['index'])
        for node, topology in enumerate(hardware) for row in topology['devices']]
    with open_read_only(directory / 'experiments.sqlite') as db:
        feedback = json.loads(db.execute('SELECT document FROM evidence WHERE id=?',
            (args.feedback,)).fetchone()[0])
        response = next(row['response'] for (encoded,) in db.execute(
            "SELECT document FROM search_state_repository WHERE kind='AGENT_RESPONSE' ORDER BY sequence DESC")
            for row in (json.loads(encoded),) if row['role'] == 'ExperimentInvestigationAgent'
            and row['wire_schema'] == 'InvestigationSelectionFinalDecision')
        omitted = {identifier for row in feedback['result']['conflicts']
            for identifier in row.get('candidate_ids', [])}
        assessed = {row['trial_id'] for row in response['candidate_assessments']}
        assert omitted and not omitted & assessed
        count = len(assessed | omitted)
        original = len(assessed)
        schema = evidenced_selection_schema(sorted(assessed | omitted)).model_json_schema()['properties']['candidate_assessments']
        assert schema['minItems'] == schema['maxItems'] == count > original
        trial = json.loads(db.execute('SELECT document FROM trials WHERE id=?',
            (args.candidate,)).fetchone()[0])
        persisted_review = db.execute(
            "SELECT document FROM search_state_repository WHERE kind='AGENT_RESPONSE' "
            "AND json_extract(document, '$.wire_schema')='CandidateReview' ORDER BY sequence DESC LIMIT 1").fetchone()
    if persisted_review:
        actual_review = json.loads(persisted_review[0])
        check_store = ExperimentStore(':memory:')
        key = check_store.add_search_state('PLANNING_CANDIDATE_REVIEW', actual_review['response'])
        assert check_store.search_states('PLANNING_CANDIDATE_REVIEW')[0]['state_id'] == key
        check_store.connection.close()
    with open_read_only(Path(config['profile_database'])) as db:
        observations = []
        for identifier, encoded, source in db.execute(
                'SELECT profiles.id, document, directory FROM profiles LEFT JOIN profile_sources ON profiles.id=profile_sources.profile_id'):
            profile = json.loads(encoded)
            profile['request'] = profile_execution_request(profile, source)
            if not profile_hardware_matches(profile, devices):
                continue
            result = exact_native_observation(trial['plan']['backend_config'], identifier, profile)
            if result is not None and result['stable_step_count'] >= 2:
                observations.append(result)
    measurement = max(observations, key=lambda row: row['stable_step_count'])
    assert len(measurement['peak_per_rank_bytes']) == len(devices)
    assert measurement['independent_prediction_error_eligible'] is False
    print(json.dumps(dict(recorded_assessments=original, required_assessments=count, omitted=sorted(omitted),
        candidate=args.candidate, calibration=measurement,
        formal_rate=trial['prediction']['performance']['tokens_per_second']), ensure_ascii=False))


if __name__ == '__main__':
    main()
