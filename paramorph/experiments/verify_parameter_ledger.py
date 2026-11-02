import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost.runtime_calibration import configuration
from graph.model_graph import read_model
from predictor.parameter_metadata import checkpoint_parameter_counts, candidate_rank_ownership
from predictor.native_parameter_layout import bind_runtime_order, candidate_buffer_layouts
from region_partition.region_partitioner import RegionPartitioner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--checkpoint-shapes', type=Path, required=True)
    parser.add_argument('--historical-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-run', type=Path)
    args = parser.parse_args()
    model, graph = read_model(args.model_config, args.checkpoint_shapes)
    artifacts = args.historical_run / 'artifacts'
    request = json.loads((artifacts / 'request.json').read_text())
    request.setdefault('precision', 'bfloat16')
    request.setdefault('main_grads_dtype', 'float32')
    config = configuration(request)
    regions = RegionPartitioner(model, config).from_counts(request['decoder_chunks'])
    ledger = graph['parameter_metadata']
    counts = checkpoint_parameter_counts(ledger, config, regions, request)
    results = []
    for rank in range(config.world_size):
        manifest = json.loads((artifacts / f'rank{rank}.manifest.json').read_text())
        ownership = json.loads((artifacts / f'rank{rank}.ownership.json').read_text())
        actual_layers = {layer for chunk in ownership['chunks'] for layer in chunk['layer_ids']}
        expected_layers = {layer for region in regions if region.pp_rank == rank // (config.world_size // config.pp)
            for layer in range(region.layer_start, region.layer_end)}
        if actual_layers != expected_layers:
            raise ValueError(f'历史Rank层归属与候选重新分区不一致：rank={rank}')
        predicted = counts['dense'][rank] + counts['expert'][rank]
        results.append(dict(rank=rank, predicted_parameter_elements=predicted,
            observed_parameter_elements=manifest['parameter_elements'],
            difference=predicted - manifest['parameter_elements'],
            components=counts['parameter_components'][rank]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'module-parameter-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (args.output / 'historical-rank-comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    owners = candidate_rank_ownership(ledger, config, regions, request)
    (args.output / 'historical-candidate-ownership.json').write_text(json.dumps(owners) + '\n')
    print(json.dumps(results, indent=2))
    if any(row['difference'] for row in results):
        raise ValueError('逐参数Ledger与历史实际Rank参数量不一致')
    if args.runtime_run is not None:
        documents = [json.loads((args.runtime_run / 'artifacts' / f'rank{rank}.parameter-ledger.json').read_text())
            for rank in range(config.world_size)]
        runtime_request = documents[0]['request']
        runtime_config = configuration(runtime_request)
        runtime_regions = RegionPartitioner(model, runtime_config).from_counts(runtime_request['decoder_chunks'])
        runtime_owners = candidate_rank_ownership(ledger, runtime_config, runtime_regions, runtime_request)
        order = bind_runtime_order(ledger, documents, graph['module_structure'], runtime_owners)
        ledger['native_parameter_order'] = order
        layouts = candidate_buffer_layouts(ledger, runtime_config, runtime_regions, runtime_request)
        for document in documents:
            if layouts[document['rank']] != document['native_buffer_layouts']:
                raise ValueError(f'原生Optimizer布局与Ledger重新计算不同：rank={document["rank"]}')
        snapshot = json.loads(args.checkpoint_shapes.read_text())
        snapshot['native_parameter_order'] = order
        (args.output / 'verified-checkpoint-shapes.json').write_text(json.dumps(snapshot, indent=2) + '\n')
        (args.output / 'module-parameter-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
        (args.output / 'runtime-tensor-and-buffer-comparison.json').write_text(json.dumps(dict(
            parameter_shapes=order['validation'], native_buffer_layouts='MATCH_ALL_RANKS'), indent=2) + '\n')
        print(json.dumps(dict(runtime_ranks=len(documents), native_buffer_layouts='MATCH_ALL_RANKS')))


if __name__ == '__main__':
    main()
