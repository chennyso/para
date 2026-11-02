import hashlib
import json
from pathlib import Path

from cost.runtime_calibration import configuration


class TraceMapper:
    def map_runtime(self, directory):
        directory = Path(directory)
        request = json.loads((directory / 'request.json').read_text())
        config = configuration(request)
        ranks = {}
        clock_verified = True
        for rank in range(config.world_size):
            path = directory / 'artifacts' / f'rank{rank}.paramorph-cost.json'
            payload = json.loads(path.read_text())
            if payload['rank'] != rank or not payload['forward_backward_coverage_verified']:
                raise ValueError('TraceMapper需要完整实际训练Rank记录')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            clock_verified = clock_verified and bool(payload.get('cross_rank_clock_calibrated'))
            rows = []
            for index, event in enumerate(payload['events']):
                if event['step'] < request['warmup']:
                    continue
                identity = event['identity']
                task = identity.get('task_id')
                if task is None and event['kind'] in ('forward_compute', 'backward_compute'):
                    task = f'{event["kind"]}:{rank}:{identity["vp_stage"]}:{identity["microbatch"]}'
                mapped = task is not None
                rows.append(dict(trace_event_id=f'{digest}:{index}', task_id=task, rank=rank,
                    region_id=identity.get('region_id', identity['vp_stage'] * config.pp + payload['pp_rank'])
                        if 'vp_stage' in identity else identity.get('region_id'),
                    event_type=event['kind'], measurement_source=str(path),
                    microbatch=identity.get('microbatch'), virtual_stage=identity.get('vp_stage'),
                    stream=event.get('stream_id'), mapped=mapped,
                    stream_identity_domain=event.get('stream_identity_domain'),
                    step=event['step'], cuda_ms=event['cuda_ms'],
                    host_start_ns=event['host_start_ns'], host_end_ns=event['host_end_ns'],
                    diagnostic_granularity='actual_region_cuda_events'))
            ranks[str(rank)] = dict(trace_sha256=digest, tasks=rows,
                clock_calibration=payload.get('clock_calibration'),
                cuda_origin_host_anchors=payload.get('cuda_origin_host_anchors'))
        return dict(ranks=ranks, measured_plan=request,
            cross_rank_clock_alignment='measured_monotonic_offsets_with_uncertainty' if clock_verified else 'unmeasured_rank_local_only',
            cross_rank_wait_ms=None, source='actual_training_cuda_events',
            missing_measurements=['跨Rank时钟校准', '独立Collective完成事件与Stream身份'])
