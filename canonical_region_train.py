import global_megatron_train as training


original_start = training.TrainingEvidence.on_train_start


def train_start(self, context):
    original_start(self, context)
    rows = [{'region_ids': [f'layer:{layer.layer_number - 1}:{kind}'
                            for layer in model.decoder.layers for kind in ('ATTENTION', 'MLP')]}
            for model in self.models]
    training.write_json(training.OUTPUT / f'rank{self.rank}.region-program.json',
                        {'actual_ownership': rows, 'boundary_lowering': 'native_canonical',
                         'fine_grained_cut_lowering': 'not_requested'})


training.TrainingEvidence.on_train_start = train_start
training.main()
