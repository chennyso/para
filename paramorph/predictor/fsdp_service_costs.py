import math


def exclusive_service_events(payload):
    waits = {}
    for event in payload['events']:
        if event['kind'] == 'fsdp_service_wait':
            waits.setdefault(event['step'], []).append((event['cuda_start_ms'],
                event['cuda_start_ms']+event['cuda_ms']))
    assert waits
    events = []
    for event in payload['events']:
        row = dict(event)
        if event['kind'] in ('layer_compute', 'forward_compute', 'backward_compute'):
            # 采用独立事件对的耗时，避免长时间轴浮点差分误差。
            start = event['cuda_start_ms']
            end = start+event['cuda_ms']
            intervals = []
            for left, right in sorted(waits.get(event['step'], [])):
                left, right = max(start, left), min(end, right)
                if right <= left:
                    continue
                if intervals and left <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(right, intervals[-1][1]))
                else:
                    intervals.append((left, right))
            excluded = sum(right-left for left, right in intervals)
            value = event['cuda_ms']-excluded
            assert math.isfinite(value) and value >= -1e-5, (event['operation'], value)
            row.update(cuda_ms=max(0.0, value), inclusive_cuda_ms=event['cuda_ms'],
                explicit_fsdp_wait_ms=excluded)
        events.append(row)
    return dict(payload, events=events)
