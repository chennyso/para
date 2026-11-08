from pathlib import Path
import re

from contracts import TrainingPlan, fingerprint


def classify_failure(result, plan, phase):
    status = result['status']
    if status == 'PASS' and (phase != 'benchmark' or result.get('performance_eligible') is True):
        return None
    raw = result.get('raw_artifacts')
    paths = sorted(Path(raw).glob('node*.log')) if raw else []
    patterns = (
        ('NUMERICAL_CORRECTNESS', 'optimizer_update', r'FSDP优化器没有取得任何真实梯度|non.finite (?:loss|grad)|nonfinite|NaN.*(?:loss|grad)'),
        ('ENVIRONMENT', 'distributed_initialization', r'EADDRINUSE|DistNetworkError|NCCL.*(?:unhandled system error|remote process exited)|watchdog.*timed out'),
        ('CHECKPOINT_FAILURE', 'checkpoint_load', r'CheckpointException|checkpoint.*(?:not found|missing key|shape mismatch)'),
        ('DETERMINISTIC_CONFIGURATION', 'backend_validation', r'AssertionError:.*(?:not supported|must be divisible)|ValueError:.*(?:不支持|必须整除)'),
    )
    observed = []
    for path in paths:
        lines = path.read_text(errors='replace').splitlines()
        for kind, stage, pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.I):
                    observed.append((kind, stage, dict(file=path.name, line=line.strip()[:500])))
                    break
    if status in ('CUDA_OOM', 'HOST_MEMORY_BUDGET_EXCEEDED'):
        kind, stage = 'CAPACITY', phase
    elif result.get('execution_deadline_reached'):
        kind, stage = 'ENVIRONMENT', 'execution_deadline'
    elif observed:
        kind, stage, _ = min(observed, key=lambda row: next(i for i, spec in enumerate(patterns) if spec[0] == row[0]))
    elif status in ('PASS', 'VALIDATION_FAILED'):
        kind, stage = 'INVALID_MEASUREMENT', 'performance_window'
    else:
        kind, stage = 'UNCLASSIFIED', phase
    retry = {'ENVIRONMENT': 'requires_changed_environment',
        'INVALID_MEASUREMENT': 'requires_changed_measurement',
        'UNCLASSIFIED': 'requires_investigation'}.get(kind, 'requires_changed_execution')
    document = dict(kind='terminal_execution_failure', status=status, failure_type=kind,
        failure_stage=stage, effective_plan_hash=TrainingPlan.model_validate(plan).effective_plan_hash,
        backend_source_hashes=result.get('source_hashes'), checkpoint_identity=result.get('checkpoint'),
        returncodes=result.get('returncodes'), evidence_refs=[result['run_id']],
        observed_error_lines=[evidence for category, _, evidence in observed if category == kind][:4],
        retry_eligibility=retry,
        gpu_allocation_seconds=result.get('gpu_allocation_seconds'))
    return dict(document, witness_id=fingerprint(document))
