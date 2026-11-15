import argparse
import gzip
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'paramorph'))

from agents.pvc_transport import encode_transport_payload, decode_transport_payload, encoded

parser = argparse.ArgumentParser()
parser.add_argument('--records', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
records = json.loads(gzip.decompress(args.records.read_bytes()))['records']
rows = []
for record in records:
    request = record['original_request']
    envelope = encode_transport_payload(request)
    assert decode_transport_payload(envelope) == request
    rows.append(dict(source_request_id=request['id'], raw_bytes=envelope['raw_bytes'],
        transport_bytes=len(encoded(envelope).encode()), sha256=envelope['sha256']))
report = dict(status='PASS', authenticated_requests=len(rows),
    raw_bytes=sum(row['raw_bytes'] for row in rows),
    transport_bytes=sum(row['transport_bytes'] for row in rows),
    decoded_requests_unchanged=True, new_gpu_runs=0, rows=rows)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'rows'}), flush=True)
