import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InterferenceProfile:
    left_resource_class: str
    right_resource_class: str
    payload_class: str
    topology_class: str
    isolated_left_us: float
    isolated_right_us: float
    paired_left_us: float
    paired_right_us: float
    run_id: str
    profile_sha256: str

    def __post_init__(self):
        durations = (self.isolated_left_us, self.isolated_right_us, self.paired_left_us, self.paired_right_us)
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError('interference profiles require positive actual measured durations')
        if not self.run_id or len(self.profile_sha256) != 64:
            raise ValueError('interference profiles require raw measurement identity')

    @property
    def slowdown_left(self):
        return self.paired_left_us / self.isolated_left_us

    @property
    def slowdown_right(self):
        return self.paired_right_us / self.isolated_right_us


class InterferenceDatabase:
    def __init__(self, records):
        self.records = {}
        for record in records:
            key = (record.left_resource_class, record.right_resource_class, record.payload_class, record.topology_class)
            if key in self.records:
                raise ValueError('interference profiles require distinct resource and topology conditions')
            self.records[key] = record

    def lookup(self, left, right, payload, topology):
        return self.records[(left, right, payload, topology)]

    def confidence(self, left, right, payload, topology):
        return 'measured' if (left, right, payload, topology) in self.records else 'high_uncertainty'
