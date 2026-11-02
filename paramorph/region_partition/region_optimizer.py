from dataclasses import asdict
import json

from cost.cost_model import MissingCalibration
from region_partition.region_partitioner import RegionPartitioner


class RegionOptimizer:
    def __init__(self, model, config, layer_measurements, rank_memory_model,
                 evaluate_complete_plan, validate_gpu_candidate):
        self.model, self.config = model, config
        self.layer_measurements = layer_measurements
        self.rank_memory_model = rank_memory_model
        self.evaluate_complete_plan = evaluate_complete_plan
        self.validate_gpu_candidate = validate_gpu_candidate
        self.history = []

    def search(self, beam_width=128, validation_budget=4):
        rows = self.layer_measurements['layers']
        if {int(index) for index in rows} != set(range(self.model.num_layers)):
            raise MissingCalibration('RegionOptimizer需要全部模型层的实际成本')
        if not self.layer_measurements['independent_layer_backward_measured']:
            raise MissingCalibration('RegionOptimizer缺少逐层Backward成本')
        rows = {int(index): row for index, row in rows.items()}
        partitioner = RegionPartitioner(self.model, self.config)
        partitioner.set_layer_costs({index: row['training_compute_ms'] for index, row in rows.items()})

        def memory_admission(slot, start, end, counts):
            return self.rank_memory_model.admit_partial(slot, start, end, counts)

        def score(regions):
            return self.evaluate_complete_plan(regions)['predicted_iteration_time_ms']

        layouts = partitioner.beam_search(score, beam_width, memory_admission)
        legal = []
        for regions in layouts[:validation_budget]:
            prediction = self.evaluate_complete_plan(regions)
            measured = self.validate_gpu_candidate(regions, prediction)
            record = dict(region_layout=[asdict(region) for region in regions],
                          prediction=prediction, measurement=measured)
            self.history.append(record)
            if measured['status'] == 'PASS':
                legal.append((measured['step_seconds'], regions, measured))
        legal.sort(key=lambda row: row[0])
        return dict(validated_layouts=[row[1] for row in legal], history=self.history,
                    best_layout=legal[0][1] if legal else None)

    def save_history(self, path):
        path.write_text(json.dumps(self.history, indent=2, allow_nan=False) + '\n')
