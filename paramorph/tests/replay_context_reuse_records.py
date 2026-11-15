import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import jsonref
import jsonschema
import tiktoken


SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / 'paramorph'), str(SOURCE)]

from agents import pvc_transport
from agents.context_budget import deduplicate_observation, REFERENCE_INSTRUCTIONS
from search.measurement_review import native_review_binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codex', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(gzip.decompress(args.records.read_bytes()))
    encoding = tiktoken.get_encoding('o200k_base')
    rows, current_review = [], None
    for index, record in enumerate(bundle['records']):
        original, revised = record['original_request'], record['revised_request']
        expanded = jsonref.replace_refs(record['packed_observation'], proxies=False)
        assert expanded == record['observation']
        assert original['schema'] == revised['schema']
        directory = args.output / f'{index:02d}-{original["role"]}'
        directory.mkdir()
        catalog = record['observation'].get('candidate_review_catalog', [])
        skip = bool(catalog)
        if skip:
            current_review = native_review_binding(catalog)
            current_review['evidence_id'] = hashlib.sha256(pvc_transport.encoded(current_review).encode()).hexdigest()
        effective_observation = deepcopy(record['observation'])
        if original['role'] == 'ExperimentInvestigationAgent' and current_review is not None:
            effective_observation['candidate_review'] = current_review
        packed, packing = deduplicate_observation(effective_observation)
        assert jsonref.replace_refs(packed, proxies=False) == effective_observation
        original_json = pvc_transport.encoded(record['observation'])
        assert original['prompt'].endswith(original_json)
        revised = dict(revised, prompt=original['prompt'][:-len(original_json)]
            + REFERENCE_INSTRUCTIONS + '\n' + pvc_transport.encoded(packed))
        row = dict(source_request_id=original['id'], role=original['role'],
            source_input_tokens=record['receipt']['usage']['input_tokens'],
            original_prompt_tokens=len(encoding.encode(original['prompt'])),
            revised_prompt_tokens=len(encoding.encode(revised['prompt'])),
            schema_tokens=len(encoding.encode(json.dumps(original['schema']))),
            skipped_duplicate_review=skip, exact_observation_roundtrip=True,
            packing=packing)
        if skip:
            result = current_review
            (directory / 'revalidation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            request = dict(revised, id=hashlib.sha256(
                (original['id'] + str(time.time_ns())).encode()).hexdigest())
            (directory / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
            if args.codex is not None:
                pvc_transport.MODEL = request['model']
                pvc_transport.ARTIFACT_DIRECTORY = directory
                receipt = pvc_transport.infer(str(args.codex), request)
                jsonschema.Draft202012Validator(request['schema']).validate(receipt['response'])
                (directory / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
                row.update(actual_usage=receipt['usage'], actual_seconds=receipt['elapsed_seconds'],
                    actual_request_id=receipt['id'])
                response = receipt['response']
            else:
                response = record['receipt']['response']
        rows.append(row)
        (directory / 'verification.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
        print(json.dumps({key: value for key, value in row.items() if key not in ('packing',)}, ensure_ascii=False), flush=True)
    original_api_seconds = sum(row['receipt']['elapsed_seconds'] for row in bundle['records'])
    summary = dict(status='PASS', source='same_authenticated_round_records',
        model=bundle['case'], round=bundle['round'], original_calls=len(rows),
        revised_calls=sum(not row['skipped_duplicate_review'] for row in rows),
        original_actual_input_tokens=bundle['cost_report']['input_tokens'],
        original_prompt_tokens=sum(row['original_prompt_tokens'] for row in rows),
        revised_prompt_tokens=sum(row['revised_prompt_tokens'] for row in rows if not row['skipped_duplicate_review']),
        original_authenticated_api_seconds=original_api_seconds,
        original_provider_wait_seconds=bundle['cost_report']['llm_seconds'],
        revised_actual_input_tokens=sum(row.get('actual_usage', {}).get('input_tokens', 0) for row in rows)
            if args.codex else None,
        revised_authenticated_api_seconds=sum(row.get('actual_seconds', 0) for row in rows) if args.codex else None,
        exact_observation_roundtrip=True, schemas_unchanged=True,
        new_gpu_runs=0, runtime_limit_seconds=None, rows=rows)
    (args.output / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
