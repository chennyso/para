import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--case', type=Path, required=True)
parser.add_argument('--mailbox', type=Path, required=True)
parser.add_argument('--round', type=int, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.source / 'paramorph'), str(args.source)]

from agents.context_budget import deduplicate_observation, REFERENCE_INSTRUCTIONS

report = json.loads((args.case / 'round-reports' / f'round-{args.round:03d}.json').read_text())
checkpoint = report['checkpoint']
start, end = checkpoint['started_unix_seconds'], checkpoint['completed_unix_seconds']
connection = sqlite3.connect((args.case / 'experiments.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
connection.execute('BEGIN')
trials = {row['trial_id']: row for encoded, in connection.execute('SELECT document FROM trials')
          for row in [json.loads(encoded)]}
calls = {}
for identifier, encoded in connection.execute(
        'SELECT c.id,a.document FROM charges c JOIN actual_cost_checkpoints a ON a.identifier=c.id '
        "WHERE c.resource='agent_calls' ORDER BY a.sequence"):
    observed = datetime.fromisoformat(json.loads(encoded)['at_utc']).timestamp()
    if start <= observed <= end:
        calls.setdefault(identifier, observed)
records = []
for identifier, observed in calls.items():
    directory = args.case / 'agents' / identifier
    context = json.loads((directory / 'request-context.json').read_text())
    request = json.loads((args.mailbox / (identifier + '.request.json')).read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    assert hashlib.sha256(request['prompt'].encode()).hexdigest() == provenance['prompt_sha256']
    assert request['schema'] == context['schema'] and receipt['id'] == identifier
    observation = context['observation']
    packed, packing = deduplicate_observation(observation)
    original_json = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert request['prompt'].endswith(original_json)
    revised = dict(request, prompt=request['prompt'][:-len(original_json)] + REFERENCE_INSTRUCTIONS + '\n'
        + json.dumps(packed, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
    catalog = observation.get('candidate_review_catalog', [])
    records.append(dict(original_request=request, revised_request=revised, receipt=receipt,
        observation=observation, packed_observation=packed, packing=packing,
        deterministic_review_catalog=catalog, observed_unix_seconds=observed))
cost = checkpoint['call_cost_report']
assert len(records) == cost['llm_calls']
assert sum(row['receipt']['usage']['input_tokens'] for row in records) == cost['input_tokens']
bundle = dict(case=args.case.name, round=args.round, cost_report=cost, records=records,
    raw_records_preserved=True, new_gpu_runs=0)
sys.stdout.buffer.write(gzip.compress(json.dumps(bundle, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
