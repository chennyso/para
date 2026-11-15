from pathlib import Path
import json
import re

from contracts import TrainingPlan, fingerprint


def checkpoint_read_crashes(directory, logs, request):
    if directory is None:
        return []
    world = request.get('node_count', 0) * request.get('gpus_per_node', 0)
    mapping = request.get('rank_mapping', list(range(world)))
    crashes = {}
    for path, log in logs.items():
        for match in re.finditer(r'^\s*rank\s*:\s*(\d+)[^\n]*\n\s*exitcode\s*:\s*-11\b', log, re.M):
            rank = int(match[1])
            if not 0 <= rank < world or rank in crashes:
                continue
            events = directory / f'rank{rank}.checkpoint-read.jsonl'
            if not events.is_file():
                continue
            reads = [json.loads(line) for line in events.read_text().splitlines() if line]
            if not reads:
                continue
            last = reads[-1]
            if (last['rank'] != rank or last['source'] != 'actual_torch_dcp_read_plan'
                    or last['status'] != 'ACQUIRED' or not last['requested_tensor_bytes']):
                continue
            crashes[rank] = dict(rank=rank, physical_rank=mapping[rank], signal='SIGSEGV',
                node_log=path.name, read_receipt=events.name, reader_lane=last['reader_lane'],
                readers_per_node=last['io_readers_per_node'], checkpoint=last['checkpoint'],
                serialized_slice_bytes=last['serialized_slice_bytes'], root_cause='UNCONFIRMED')
    return [crashes[rank] for rank in sorted(crashes)]


def classify_failure(result, plan, phase):
    status = result['status']
    if status == 'PASS' and (phase != 'benchmark' or result.get('performance_eligible') is True):
        return None
    raw = result.get('raw_artifacts')
    directory = Path(raw) if raw else None
    paths = sorted(directory.glob('node*.log')) if directory else []
    logs = {path: path.read_text(errors='replace') for path in paths}
    checkpoint_crashes = checkpoint_read_crashes(directory, logs, plan['backend_config'])
    oom_events = []
    if directory is not None:
        for path in sorted(directory.glob('node*.exit.json')):
            row = json.loads(path.read_text())
            before = row.get('cgroup_memory_events_before', {})
            after = row.get('cgroup_memory_events_after', {})
            if 'oom_kill' in before and after.get('oom_kill', 0) > before['oom_kill']:
                oom_events.append(dict(file=path.name, count=after['oom_kill'] - before['oom_kill']))
    patterns = (
        ('EXECUTION_CONFIG_MISMATCH', 'runtime_configuration', r'EXECUTION_CONFIG_MISMATCH:'),
        ('CAPACITY', 'checkpoint_host_memory', r'CHECKPOINT_HOST_MEMORY_ADMISSION:'),
        ('NUMERICAL_CORRECTNESS', 'optimizer_update', r'FSDP优化器没有取得任何真实梯度|non.finite (?:loss|grad)|nonfinite|NaN.*(?:loss|grad)|Unexpected result (?:inf|nan).*grad norm|found (?:Inf|NaN) in local grad norm'),
        ('ENVIRONMENT', 'distributed_initialization', r'EADDRINUSE|DistNetworkError|NCCL.*(?:unhandled system error|remote process exited)|watchdog.*timed out'),
        ('CHECKPOINT_FAILURE', 'checkpoint_load', r'CheckpointException|checkpoint.*(?:not found|missing key|shape mismatch)'),
        ('DETERMINISTIC_CONFIGURATION', 'backend_validation', r'AssertionError:.*(?:not supported|must be divisible)|ValueError:.*(?:不支持|必须整除)'),
    )
    observed = []
    for path, log in logs.items():
        lines = log.splitlines()
        for kind, stage, pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.I):
                    observed.append((kind, stage, dict(file=path.name, line=line.strip()[:500])))
                    break
    if status == 'HOST_MEMORY_ADMISSION_REJECTED':
        kind, stage = 'CAPACITY', 'prelaunch_host_memory'
    elif status in ('CUDA_OOM', 'HOST_MEMORY_BUDGET_EXCEEDED') or oom_events:
        kind, stage = 'CAPACITY', phase
    elif result.get('runtime_validation_kind') == 'EXECUTION_CONFIG_MISMATCH':
        kind, stage = 'EXECUTION_CONFIG_MISMATCH', 'runtime_configuration'
    elif checkpoint_crashes:
        kind, stage = 'CHECKPOINT_READ_FAILURE', 'checkpoint_tensor_read'
    elif result.get('execution_deadline_reached'):
        kind, stage = 'ENVIRONMENT', 'execution_deadline'
    elif observed:
        kind, stage, _ = min(observed, key=lambda row: next(i for i, spec in enumerate(patterns) if spec[0] == row[0]))
    elif status in ('PASS', 'VALIDATION_FAILED'):
        kind, stage = 'INVALID_MEASUREMENT', 'performance_window'
    else:
        kind, stage = 'UNCLASSIFIED', phase
    retry = {'ENVIRONMENT': 'requires_changed_environment', 'CHECKPOINT_READ_FAILURE': 'requires_changed_environment',
        'INVALID_MEASUREMENT': 'requires_changed_measurement',
        'UNCLASSIFIED': 'requires_investigation'}.get(kind, 'requires_changed_execution')
    document = dict(kind='terminal_execution_failure', status=status, failure_type=kind,
        failure_stage=stage, effective_plan_hash=TrainingPlan.model_validate(plan).effective_plan_hash,
        backend_source_hashes=result.get('source_hashes'), checkpoint_identity=result.get('checkpoint'),
        returncodes=result.get('returncodes'), evidence_refs=[result['run_id']],
        observed_error_lines=[evidence for category, _, evidence in observed if category == kind][:4],
        retry_eligibility=retry,
        gpu_allocation_seconds=result.get('gpu_allocation_seconds'))
    if kind == 'CHECKPOINT_READ_FAILURE':
        document['checkpoint_read_crashes'] = checkpoint_crashes
    if kind == 'CAPACITY' and oom_events:
        document['cgroup_oom_kill_events'] = oom_events
    if kind == 'NUMERICAL_CORRECTNESS' and directory is not None:
        document['gradient_bucket_evidence'] = [json.loads(line) for path in sorted(directory.glob('rank*.numerical-failure.jsonl'))
            for line in path.read_text().splitlines() if line]
    if kind == 'EXECUTION_CONFIG_MISMATCH':
        document['runtime_config_mismatch'] = result.get('runtime_config_mismatch', [])
        if not document['runtime_config_mismatch']:
            observed_configs = []
            for path in paths:
                for line in path.read_text(errors='replace').splitlines():
                    match = re.search(r'ValueError: EXECUTION_CONFIG_MISMATCH: (\{.*\})$', line)
                    if match:
                        observed_configs.append(dict(file=path.name, **json.loads(match[1])))
            document['runtime_config_mismatch'] = observed_configs
    return dict(document, witness_id=fingerprint(document))
