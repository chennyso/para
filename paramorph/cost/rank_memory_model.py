import json
from pathlib import Path

import numpy as np

from cost.cost_model import MissingCalibration


class RankMemoryModel:
    def __init__(self, directory, config):
        self.config = config
        self.directory = Path(directory)
        self.layers = {}
        self.capacities = {}
        self.evidence_ids = []
        static = [json.loads(path.read_text()) for path in self.directory.glob('rank*.paramorph-static.json')]
        traces = {row['rank']: row for row in (json.loads(path.read_text())
            for path in self.directory.glob('rank*.paramorph-cost.json'))}
        if len(static) != config.world_size or len(traces) != config.world_size:
            raise MissingCalibration('每个物理Rank需要实际常驻与迭代显存测量')
        features, values = [], []
        for row in static:
            rank = row['rank']
            physical = traces[rank]['pp_rank']
            loaded = 0
            for layer in row['layers']:
                if layer['gradient_bytes'] is None:
                    raise MissingCalibration('Layer梯度显存记录缺失')
                size = layer['weight_bytes'] + layer['gradient_bytes']
                self.layers[layer['layer_id']] = size
                loaded += size
            windows = [window for window in traces[rank]['windows']
                       if window['step'] >= traces[rank]['request']['warmup']]
            persistent = max(window['persistent_bytes'] for window in windows)
            features.append([loaded / 2**30, 1.0, float(physical == 0), float(physical == config.pp - 1)])
            values.append(persistent / 2**30)
            if 'allocator_limit_bytes' not in row:
                raise MissingCalibration('Rank显存准入缺少实际CUDA allocator容量限制')
            self.capacities[rank] = min(row['device_capacity_bytes'], row['allocator_limit_bytes'])
            self.evidence_ids.append(f'{traces[rank]["request"]["run_id"]}:rank{rank}')
        matrix = np.asarray(features)
        if np.linalg.matrix_rank(matrix) < 4:
            raise MissingCalibration('实际Rank数据不足以区分Layer与输入输出常驻显存')
        coefficients, _, _, _ = np.linalg.lstsq(matrix, np.asarray(values), rcond=None)
        if coefficients[0] <= 0:
            raise ValueError('实际显存成本拟合产生非法Layer系数')
        self.coefficients = coefficients.tolist()
        self.maximum_residual_bytes = float(np.abs(matrix @ coefficients - np.asarray(values)).max()) * 2**30
        self.fitting_source = 'actual_loaded_parameters_gradients_and_post_warmup_rank_allocator'

    def persistent_prediction(self, physical, layer_ids):
        slope, intercept, embedding, output = self.coefficients
        predicted = slope * sum(self.layers[index] for index in layer_ids)
        predicted += (intercept + embedding * (physical == 0) + output *
                      (physical == self.config.pp - 1)) * 2**30
        return predicted + self.maximum_residual_bytes

    def admit_partial(self, slot, start, end, counts):
        physical = slot % self.config.pp
        identifiers = list(range(start, end))
        offset = 0
        for index, count in enumerate(counts):
            if index % self.config.pp == physical:
                identifiers.extend(range(offset, offset + count))
            offset += count
        lanes = self.config.tp * self.config.cp * self.config.dp
        capacity = min(self.capacities[physical * lanes + lane] for lane in range(lanes))
        return self.persistent_prediction(physical, identifiers) < capacity

    def describe(self):
        return dict(coefficients=self.coefficients, maximum_residual_bytes=self.maximum_residual_bytes,
            source=self.fitting_source, evidence_ids=self.evidence_ids,
            rank_capacity_bytes=self.capacities,
            gpu_admission_requires_actual_training=True)
