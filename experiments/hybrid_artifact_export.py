import argparse
import hashlib
import json
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
run = args.run.resolve()
reports = [{'name': path.name, 'bytes': path.stat().st_size, 'remote_path': str(path)}
    for path in sorted(run.glob('rank*.profile.nsys-rep'))]
index = {'reports': reports, 'storage': 'PVC', 'cuda_data_verified': False,
    'description': 'Native Nsight reports retained in the original run directory'}
(run / 'nsys-index.json').write_text(json.dumps(index, indent=2) + '\n')
profiler_reports = [{'name': path.name, 'bytes': path.stat().st_size,
    'remote_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(run.glob('rank*.paramorph-trace.json'))]
(run / 'paramorph-trace-index.json').write_text(json.dumps(
    dict(reports=profiler_reports, storage='PVC'), indent=2) + '\n')
archive = run / 'metadata-export.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=1) as package:
    for path in sorted(run.rglob('*')):
        if (path.is_file() and not path.name.endswith('.paramorph-trace.json')
                and path.suffix in ('.py', '.json', '.jsonl', '.log', '.pid', '.csv', '.tsv')):
            package.add(path, arcname=str(path.relative_to(run)))
print(json.dumps({'archive_path': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'trace_reports': len(reports)}), flush=True)
