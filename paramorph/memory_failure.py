import json
import re
from pathlib import Path

from contracts import fingerprint


SIZE_PATTERN = re.compile(r'(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>GiB|MiB|KiB|bytes?)', re.IGNORECASE)
RANK_PATTERNS = (re.compile(r'\[rank(?P<rank>[0-9]+)\]'),
    re.compile(r'\brank[=:](?P<rank>[0-9]+)\b'))


def _bytes(value, unit):
    scales = {'gib': 2**30, 'mib': 2**20, 'kib': 2**10, 'byte': 1, 'bytes': 1}
    return int(float(value) * scales[unit.lower()])


def _extract_size(text, prefix):
    match = re.search(prefix + r'\s*([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB|KiB|bytes?)',
        text, re.IGNORECASE)
    return None if match is None else _bytes(match.group(1), match.group(2))


def _rank(text):
    for pattern in RANK_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group('rank'))
    return None


def _phase(text):
    lower = text.lower()
    if 'param_and_grad_buffer.py' in lower and 'self.grad_data = torch.zeros(' in lower:
        return 'initialization', 'LOAD_OOM'
    if any(value in lower for value in ('load_checkpoint', 'load_state_dict', 'checkpoint load',
            'pretrained_checkpoint', 'dist_checkpointing')):
        return 'checkpoint_load', 'LOAD_OOM'
    if any(value in lower for value in ('optimizer.step', 'step_with_ready_grads',
            'initialize_state', 'optimizer state')):
        return 'optimizer_step', 'TRAIN_OOM'
    if any(value in lower for value in ('train_step', 'forward_step', 'backward_step',
            'forward_backward', 'backward')):
        return 'forward_backward', 'TRAIN_OOM'
    return 'initialization', 'LOAD_OOM'


def oom_allocation_site(text):
    if 'param_and_grad_buffer.py' in text and 'self.grad_data = torch.zeros(' in text:
        return 'ddp_gradient_buffer_initialization'
    return None


def memory_failure_witness(result, plan, directory, prediction=None, world_size=None):
    directory = Path(directory)
    artifact_directory = directory / 'artifacts'
    logs = sorted(artifact_directory.glob('node*.log'))
    records = []
    for path in logs:
        text = path.read_text(errors='replace')
        positions = [position for marker in ('CUDA out of memory', 'OutOfMemoryError',
            'Failed to CUDA calloc async', 'MemoryError', 'OOMKilled')
            if (position := text.find(marker)) >= 0]
        if positions:
            position = min(positions)
            records.append((path, position, text[max(0, position - 12000):position + 4000]))
    if not records:
        raise ValueError('真实OOM结果缺少可定位的原始错误日志')
    path, _, excerpt = min(records, key=lambda row: (row[0].name, row[1]))
    phase, status = _phase(excerpt)
    if not any(marker in excerpt for marker in (
            'CUDA out of memory', 'OutOfMemoryError', 'Failed to CUDA calloc async')):
        status = 'HOST_OOM'
    first_rank = _rank(excerpt)
    requested = _extract_size(excerpt, r'Tried to allocate')
    allocated = _extract_size(excerpt, r'Of the allocated memory')
    unallocated_reserved = _extract_size(excerpt, r'and')
    total_capacity = _extract_size(excerpt, r'total capacity of')
    free = _extract_size(excerpt, r'of which')
    process_used = _extract_size(excerpt, r'this process has')
    reserved = None if allocated is None or unallocated_reserved is None else allocated + unallocated_reserved
    gpu_count = world_size or result.get('node_count', 1) * result.get('gpus_per_node', 1)
    gpu_hours = gpu_count * result['gpu_allocation_seconds'] / 3600
    predicted_phase = None
    if prediction is not None and first_rank is not None:
        breakdown = prediction.get('memory', {}).get('breakdown_per_rank', {})
        row = breakdown.get(first_rank, breakdown.get(str(first_rank)))
        if row is not None:
            predicted_phase = dict(source='PREDICTED', phase=phase,
                predicted_bytes=(row.get('gpu_phases') or {}).get(phase),
                ownership_components=row.get('ownership_components'),
                activation_upper_bound_bytes=row.get('activation_upper_bound_bytes'),
                kernel_workspace_bytes=row.get('kernel_workspace_bytes'),
                reserve_bytes=row.get('reserve_bytes'))
    observed = dict(first_exception_rank=first_rank, requested_bytes=requested,
        allocated_bytes=allocated, reserved_bytes=reserved,
        unallocated_reserved_bytes=unallocated_reserved, device_capacity_bytes=total_capacity,
        device_free_bytes=free, process_device_used_bytes=process_used,
        host_memory_bytes=None, gpu_allocation_seconds=result['gpu_allocation_seconds'],
        gpu_hours=gpu_hours, status=status, phase=phase,
        allocation_site=oom_allocation_site(excerpt),
        source_log=str(path), source='actual_runtime_exception')
    witness = dict(witness_id='', run_id=result['run_id'], status=status,
        complete_plan=plan, observed=observed,
        dag_phase_hypothesis=predicted_phase,
        hypotheses=[] if predicted_phase is None else [
            '预测阶段构成用于提出后续联合策略，真实错误只证明对应Rank和阶段发生内存申请失败'],
        evidence_refs=[result['run_id'], str(path)],
        evidence_scope='runtime_observation_separated_from_dag_based_hypothesis')
    witness['witness_id'] = 'memory-failure-' + fingerprint(witness)[:20]
    return witness


def write_memory_failure_witness(result, plan, directory, prediction=None, world_size=None):
    witness = memory_failure_witness(result, plan, directory, prediction, world_size)
    path = Path(directory) / 'memory-failure-witness.json'
    path.write_text(json.dumps(witness, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return witness
