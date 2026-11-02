from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from cost.cost_model import MissingCalibration


class LayerCalibration:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.payloads = [json.loads(path.read_text()) for path in sorted(
            self.directory.glob('rank*.paramorph-cost.json'))]
        if not self.payloads:
            raise MissingCalibration('逐层成本需要真实训练校准文件')

    def extract(self, num_layers):
        samples = defaultdict(list)
        evidence = defaultdict(set)
        region_rows = defaultdict(list)
        memory = {}
        for payload in self.payloads:
            request = payload['request']
            for event in payload['events']:
                if event['step'] < request['warmup']:
                    continue
                if (request.get('paramorph_serial_calibration_steps', 0) >= request['warmup']
                    and event['step'] > request['paramorph_serial_calibration_steps']):
                    continue
                if request.get('paramorph_trace') and event['step'] == request['warmup'] + request['measured'] - 1:
                    continue
                identifier = event['identity'].get('layer_id')
                if event['kind'] == 'layer_compute':
                    if event['operation'] == 'layer_backward' and not event['identity'].get(
                            'backward_input_gradient_observed', False):
                        continue
                    if event['cuda_ms'] is None:
                        raise MissingCalibration('逐层事件没有实际CUDA耗时')
                    key = (identifier, event['operation'])
                    samples[key].append(event['cuda_ms'])
                    evidence[key].add(f'{request["run_id"]}:rank{payload["rank"]}:{event["step"]}:{event["host_start_ns"]}')
                elif event['kind'] in ('forward_compute', 'backward_compute'):
                    region_rows[(payload['rank'], event['identity']['vp_stage'], event['kind'])].append(event['cuda_ms'])
        for path in self.directory.glob('rank*.paramorph-static.json'):
            static = json.loads(path.read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for layer in static['layers']:
                if layer['gradient_bytes'] is None:
                    raise MissingCalibration('逐层梯度显存尚未测量')
                memory[layer['layer_id']] = dict(weight_bytes=layer['weight_bytes'],
                    gradient_bytes=layer['gradient_bytes'], evidence_id=digest)
        result = {}
        for index in range(num_layers):
            if not samples[(index, 'layer_forward')] or not samples[(index, 'layer_backward')]:
                raise MissingCalibration(f'Layer {index}缺少完整Forward或者Backward测量')
            if index not in memory:
                raise MissingCalibration(f'Layer {index}缺少实际显存数据')
            forward = statistics.median(samples[(index, 'layer_forward')])
            backward = statistics.median(samples[(index, 'layer_backward')])
            recomputed = samples[(index, 'layer_recompute_forward')]
            recompute = statistics.median(recomputed) if recomputed else 0.0
            result[index] = dict(forward_ms=forward, backward_ms=backward,
                recompute_ms=recompute, training_compute_ms=forward + backward + recompute,
                backward_timing_scope='recomputed_layer_output_gradient_to_input_gradient',
                recompute_timing_scope='recomputed_layer_forward_before_gradient_propagation',
                recompute_executed=bool(recomputed), **memory[index],
                compute_evidence_ids=sorted(set().union(*(evidence[(index, operation)]
                    for operation in ('layer_forward', 'layer_backward', 'layer_recompute_forward')))))
        return dict(layers=result, regions={str(key): statistics.median(values)
            for key, values in region_rows.items()},
            source='actual_training_layer_cuda_events_and_loaded_parameters',
            independent_layer_backward_measured=True,
            cross_layout_prediction_requires_region_overhead_and_communication_calibration=True)
