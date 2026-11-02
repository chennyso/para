import argparse
import hashlib
import json
from pathlib import Path
import re
import unittest

from predictor.state_sharding import _widths


def digest(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def measured_request(directory):
    directory = Path(directory).resolve()
    request_path = directory / 'request.json'
    manifest_path = directory / 'artifacts' / 'rank0.manifest.json'
    request = json.loads(request_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    optimizer = manifest['optimizer_state_dtypes']
    fields = dict(precision=manifest['params_dtype'], main_grads_dtype=optimizer['main_grads'],
        main_params_dtype=optimizer['main_params'], exp_avg_dtype=optimizer['exp_avg'],
        exp_avg_sq_dtype=optimizer['exp_avg_sq'])
    evidence = [digest(request_path), digest(manifest_path)]
    if request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp':
        log_path = directory / 'artifacts' / 'node0.log'
        evidence.append(digest(log_path))
        matches = re.findall(r'Setting up Megatron-FSDP MixedPrecisionPolicy with config '
            r'MixedPrecisionPolicy\(main_params_dtype=(torch\.\w+),', log_path.read_text())
        if len(set(matches)) != 1:
            raise ValueError('实际FSDP日志缺少唯一主权重精度')
        fields['megatron_fsdp_main_params_dtype'] = matches[0]
    return request | fields, dict(source_directory=str(directory), fields=fields, evidence=evidence)


def verify(fsdp_directory, reference_directories, output):
    case = unittest.TestCase()
    request, source = measured_request(fsdp_directory)
    widths = _widths(request)
    case.assertEqual(widths, dict(parameter=2, gradient=2, master=4, first_moment=2, second_moment=2))
    case.assertEqual(request['main_params_dtype'], 'torch.float16')
    case.assertEqual(request['megatron_fsdp_main_params_dtype'], 'torch.float32')
    missing_master = {key: value for key, value in request.items() if key != 'megatron_fsdp_main_params_dtype'}
    with case.assertRaisesRegex(ValueError, 'megatron_fsdp_main_params_dtype'):
        _widths(missing_master)
    for field in ('main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
        with case.assertRaisesRegex(ValueError, field):
            _widths({key: value for key, value in request.items() if key != field})
    ranks = []
    artifacts = Path(fsdp_directory) / 'artifacts'
    for rank in range(request['node_count'] * request['gpus_per_node']):
        cost_path = artifacts / f'rank{rank}.paramorph-cost.json'
        audit_path = artifacts / f'rank{rank}.paramorph-memory-audit.json'
        payload = json.loads(cost_path.read_text())
        inventory = json.loads(audit_path.read_text())['state_inventory']
        components = {event['identity']['component_index']: event['identity']['owned_elements']
            for event in payload['events'] if event['step'] == request['warmup']
            and event['kind'] == 'optimizer_local_update'}
        if not components:
            raise ValueError('原始Profile没有本地Optimizer参数数量证据')
        elements = sum(components.values())
        storages = inventory['storages']
        observed = dict(
            master=sum(row['bytes'] for row in storages if any('main_weight_buffer' in key
                for key in row['tensor_keys'])),
            first_moment=sum(row['bytes'] for row in storages if any(key.endswith(':exp_avg')
                for key in row['tensor_keys'])),
            second_moment=sum(row['bytes'] for row in storages if any(key.endswith(':exp_avg_sq')
                for key in row['tensor_keys'])))
        parameter_bytes = inventory['logical_tensor_bytes']['parameters']
        case.assertEqual(parameter_bytes % widths['parameter'], 0)
        buffer_elements = parameter_bytes // widths['parameter']
        case.assertGreaterEqual(buffer_elements, elements)
        expected = dict(master=buffer_elements * widths['master'],
            first_moment=elements * widths['first_moment'],
            second_moment=elements * widths['second_moment'])
        case.assertEqual(observed, expected)
        case.assertEqual(sum(expected.values()), inventory['logical_tensor_bytes']['optimizer'])
        case.assertEqual(inventory['logical_tensor_bytes']['gradients'], buffer_elements * widths['gradient'])
        ranks.append(dict(rank=rank, owned_elements=elements, optimizer_bytes=sum(expected.values()),
            optimizer_component_bytes=observed,
            allocated_master_elements=buffer_elements, master_padding_elements=buffer_elements - elements,
            master_bytes_above_float16=buffer_elements * (widths['master'] - 2),
            evidence=[digest(cost_path), digest(audit_path)]))
    case.assertEqual(ranks[0]['owned_elements'], 1908342784)
    case.assertEqual(ranks[0]['optimizer_bytes'], 15266742272)
    case.assertEqual(ranks[0]['master_bytes_above_float16'], 3816685568)
    references = []
    for directory in reference_directories:
        recorded, provenance = measured_request(directory)
        result = _widths(recorded)
        case.assertEqual(result, dict(parameter=2, gradient=4, master=4, first_moment=4, second_moment=4))
        references.append(dict(widths=result, **provenance))
    report = dict(status='PASS', source='actual_backend_configuration_and_buffer_storage',
        fsdp_widths=widths, fsdp_dtype_evidence=source, ranks=ranks, references=references,
        missing_backend_dtype_rejected=True, optimizer_config_does_not_override_fsdp_master=True,
        gpu_execution=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsdp-directory', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.fsdp_directory, args.reference_directory, args.output)
    print(json.dumps(dict(status=report['status'], fsdp_widths=report['fsdp_widths'],
        verified_ranks=len(report['ranks']), verified_references=len(report['references']),
        rank0_optimizer_bytes=report['ranks'][0]['optimizer_bytes'],
        rank0_master_bytes_above_float16=report['ranks'][0]['master_bytes_above_float16']), ensure_ascii=False))
