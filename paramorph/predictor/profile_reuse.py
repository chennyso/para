import hashlib
import json
from pathlib import Path

from adapters.megatron_adapter import prepare_execution_request, profile_source_paths
from predictor.profile_database import same_training_plan


def reusable_training_profile(search, execution):
    request = prepare_execution_request(execution)
    if 'fp16_bounded_adam_sha256' in request:
        return None
    files = search.adapter.runner.runtime_source_paths() | profile_source_paths(request)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    actual = sorted(str(row.get('uuid', row.get('gpu_uuid'))).removeprefix('GPU-').lower()
        for row in search.topology.devices)
    matches = []
    for identifier, profile in search.predictor.database.measured_profiles():
        previous = profile['request']
        if not same_training_plan(previous, request) or profile['source_hashes'] != hashes:
            continue
        flags = ('paramorph_operator_calibration', 'paramorph_memory_audit',
            'paramorph_fsdp_component_services', 'paramorph_fsdp_component_isolation')
        if any(bool(previous.get(name)) != bool(request.get(name)) for name in flags):
            continue
        if request.get('paramorph_optimizer_calibration') and not previous.get('paramorph_optimizer_calibration'):
            continue
        if previous.get('paramorph_operator_scope', 'all') != request.get('paramorph_operator_scope', 'all'):
            continue
        if previous.get('warmup', 0) != request['warmup'] or previous.get('measured', 0) != request['measured']:
            continue
        measured = sorted(str(row['gpu_uuid']).removeprefix('GPU-').lower() for row in profile['hardware'])
        if actual != measured or any(len(rows) < 3 for rows in profile['measured_windows'].values()):
            continue
        directory = Path(profile['source_directory'])
        result = json.loads((directory/'result.json').read_text())
        if result['status'] != 'PASS':
            continue
        for name, digest in hashes.items():
            with (directory/'code'/name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        for name, digest in profile['evidence_files'].items():
            with (directory/'artifacts'/name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        matches.append((identifier, profile, result))
    return max(matches, key=lambda row: row[0], default=None)
