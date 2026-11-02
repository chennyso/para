import statistics

from contracts import fingerprint


class OverlapDatabase:
    def __init__(self, records):
        self.records = records

    def estimate(self, compute_signature, communication_signature, hardware_signature):
        key = fingerprint(dict(compute=compute_signature, communication=communication_signature,
            hardware=hardware_signature))
        rows = [row for row in self.records if row['signature'] == key]
        if not rows:
            return dict(status='overlap_unknown', milliseconds=None, profiling_request=dict(
                kind='compute_communication_pair', compute=compute_signature,
                communication=communication_signature, hardware=hardware_signature))
        if any(row['source'] != 'real_gpu_microbenchmark' for row in rows):
            raise ValueError('并发成本需要真实GPU微基准证据')
        samples = [value for row in rows for value in row['overlapped_ms']]
        return dict(status='measured', milliseconds=statistics.median(samples),
            variability_ms=statistics.pstdev(samples), evidence_ids=[row['evidence_id'] for row in rows])
