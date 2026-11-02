import hashlib
import json
import math
from pathlib import Path


class RunHistory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []
        identifiers = [row['program_id'] for row in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('run history contains duplicate real program evaluations')

    @staticmethod
    def identifier(program):
        payload = json.dumps(program, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def contains(self, program):
        identifier = self.identifier(program)
        return any(row['program_id'] == identifier for row in self.records)

    def append(self, record):
        if self.contains(record['program']):
            raise ValueError('program was already evaluated or rejected')
        if record['program_id'] != self.identifier(record['program']):
            raise ValueError('history identity differs from the executable program')
        if record['status'] == 'completed':
            if record['origin'] not in ('seed', 'agent_trial'):
                raise ValueError('search history accepts actual seed or agent trial records')
            if not math.isfinite(record['measured_step_us']) or record['measured_step_us'] <= 0:
                raise ValueError('completed search evaluation requires actual measured step time')
            if not record['raw_run_id'] or not record['source_archive_sha256']:
                raise ValueError('completed search evaluation requires raw GPU measurement identity')
        self.records.append(record)
        pending = self.path.with_suffix('.pending')
        pending.write_text(json.dumps(self.records, indent=2, allow_nan=False) + '\n')
        pending.replace(self.path)

    @property
    def real_trials(self):
        return sum(bool(row['real_run_started']) for row in self.records)

    def measured(self):
        return tuple(row for row in self.records if row['status'] == 'completed')
