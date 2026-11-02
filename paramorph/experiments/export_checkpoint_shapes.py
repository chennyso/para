import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def export_shapes(directory):
    directory = Path(directory)
    config_bytes = (directory / 'config.json').read_bytes()
    index_bytes = (directory / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)
    tensors = {}
    for filename in sorted(set(index['weight_map'].values())):
        with safe_open(directory / filename, framework='pt', device='cpu') as checkpoint:
            for name in checkpoint.keys():
                if index['weight_map'].get(name) != filename or name in tensors:
                    raise ValueError(f'Checkpoint索引与Tensor归属不一致：{name}')
                tensor = checkpoint.get_slice(name)
                tensors[name] = dict(shape=tensor.get_shape(), dtype=tensor.get_dtype(), shard=filename)
    if set(tensors) != set(index['weight_map']):
        raise ValueError('Checkpoint Shape记录没有覆盖索引中的全部Tensor')
    return dict(version=1, source='safetensors.safe_open.get_slice',
        checkpoint_directory=str(directory), config=json.loads(config_bytes),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(), tensors=tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = export_shapes(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), tensor_count=len(document['tensors']))))


if __name__ == '__main__':
    main()
