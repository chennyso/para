import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from predictor.measured_modules import (
    OPTIMIZER_EXCLUDED_OPERATIONS, annotate_measurement_semantics,
    enrich_module_profile, optimizer_local_cost_evidence, optimizer_component_cost_evidence)
from predictor.profile_database import ProfileDatabase, backend_reuse_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend-source', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted_profiles = ProfileDatabase(args.output.with_suffix('.sqlite'))
    rows = []
    directory = args.workspace / 'outputs/joint-synthesis-verification/cost-bindings-predictions'
    for model in ('mimo', 'qwen14', 'falcon'):
        database = directory / f'{model}-costs.sqlite'
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            profiles = [json.loads(row[0]) for row in connection.execute('SELECT document FROM profiles')]
        for original in profiles:
            annotated = annotate_measurement_semantics(original)
            preserved = dict(annotated)
            preserved.pop('measurement_semantics')
            assert preserved == original
            accepted_profiles.add(original)
            accepted_profiles.add(annotated)
            assert annotated['measurement_semantics']['anchor']['task_costs_rescaled'] is False
            assert annotated['measurement_semantics']['layer']['local_service_time_verified'] is False
            evidence = optimizer_local_cost_evidence(annotated)
            assert evidence['verified'] is False
            assert evidence['costs_by_rank'] == {}
            for operation in ('optimizer_prepare_grads', 'optimizer_local_grad_norm',
                    'optimizer_local_grad_norm_finalize', 'optimizer_local_grad_clip'):
                assert optimizer_component_cost_evidence(annotated, operation)['verified'] is False
            enriched = enrich_module_profile(annotated)
            for layer, observation in original['layers'].items():
                for field in ('forward_ms', 'backward_ms', 'state_bytes'):
                    assert enriched['layers'][layer][field] == observation[field]
            assert enriched['optimizer'] == original['optimizer']
            assert enriched['module_parameters']
            assert enrich_module_profile(enriched) == enriched
            # 仅声明纯成本时，仍须检查真实事件中的组件范围。
            declaration = deepcopy(annotated)
            declaration['optimizer_local_costs'] = dict(
                scope='local_optimizer_update_and_copy',
                excluded_operations=sorted(OPTIMIZER_EXCLUDED_OPERATIONS))
            declared_evidence = optimizer_local_cost_evidence(declaration)
            assert declared_evidence['verified'] is False
            assert declared_evidence['costs_by_rank'] == {}
            missing = dict(original['request'])
            missing.pop('backend_fingerprint', None)
            assert backend_reuse_evidence(missing, original)['status'] == 'UNVERIFIED'
            source_hash = hashlib.sha256(json.dumps(original['source_hashes'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            assert backend_reuse_evidence(dict(missing, backend_fingerprint=source_hash),
                original)['status'] == 'MATCHED'
            rows.append(dict(model=model, profile_id=original['id'],
                raw_measurements_preserved=True, structure_available=True,
                optimizer_local_cost_verified=evidence['verified'],
                metadata_only_declaration_rejected=True,
                backend_identity_without_target='UNVERIFIED',
                paired_step_comparison=annotated['measurement_semantics']['paired_step_comparison']))
    report = dict(status='PASS', profile_count=len(rows), profiles=rows,
        gpu_runs=0, external_llm_calls=0)
    if args.backend_source is not None:
        sources = {name: (args.backend_source / name).read_text()
            for name in ('optimizer.py', 'distrib_optimizer.py', 'clip_grads.py')}
        trees = {name: ast.parse(source) for name, source in sources.items()}

        def method(filename, class_name, method_name):
            cls = next(node for node in trees[filename].body
                if isinstance(node, ast.ClassDef) and node.name == class_name)
            return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)

        local_step = method('optimizer.py', 'MixedPrecisionOptimizer', 'step_with_ready_grads')
        local_calls = {ast.unparse(node.func): node.lineno for node in ast.walk(local_step)
            if isinstance(node, ast.Call)}
        assert local_calls['self.optimizer.step'] < local_calls['self._copy_main_params_to_model_params']
        distributed_step = method('distrib_optimizer.py', 'DistributedOptimizer', 'step_with_ready_grads')
        distributed_calls = {ast.unparse(node.func): node.lineno for node in ast.walk(distributed_step)
            if isinstance(node, ast.Call)}
        assert distributed_calls['super().step_with_ready_grads'] < distributed_calls[
            'self.start_param_sync_for_bucket_group_subset']
        norm = next(node for node in trees['clip_grads.py'].body
            if isinstance(node, ast.FunctionDef) and node.name == 'get_grad_norm_fp32')
        assert [argument.arg for argument in norm.args.args] == [
            'grads_for_norm', 'norm_type', 'grad_stats_parallel_group']
        clip = next(node for node in trees['clip_grads.py'].body
            if isinstance(node, ast.FunctionDef) and node.name == 'clip_grad_by_total_norm_fp32')
        assert not any(isinstance(node, ast.Call) and ast.unparse(node.func).endswith('all_reduce')
            for node in ast.walk(clip))
        report['deployed_backend_call_boundaries'] = dict(verified=True,
            source_sha256={name: hashlib.sha256(source.encode()).hexdigest()
                for name, source in sources.items()},
            update_line=local_calls['self.optimizer.step'],
            copy_line=local_calls['self._copy_main_params_to_model_params'],
            gather_line=distributed_calls['self.start_param_sync_for_bucket_group_subset'])
        fsdp_path = args.backend_source.parent / 'deployed-distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py'
        if fsdp_path.is_file():
            fsdp_source = fsdp_path.read_text()
            fsdp_tree = ast.parse(fsdp_source)
            buffer_class = next(node for node in fsdp_tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'DataParallelBuffer')
            methods = {node.name: node for node in buffer_class.body if isinstance(node, ast.FunctionDef)}
            assert any(isinstance(node, ast.Assign) and any(ast.unparse(target) == 'self.data'
                for target in node.targets) for node in ast.walk(methods['init_data']))
            assert any(isinstance(node, ast.Return) and ast.unparse(node.value) == 'self.data[start:end]'
                for node in ast.walk(methods['get_item']))
            assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'setattr' and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'orig_param'
                for node in ast.walk(fsdp_tree))
            report['deployed_fsdp_local_storage'] = dict(verified=True,
                source_sha256=hashlib.sha256(fsdp_source.encode()).hexdigest(),
                storage_owner='DataParallelBuffer.data',
                parameter_view='DataParallelBuffer.get_item',
                optimizer_parameter_mapping='optimizer_named_parameters.orig_param')
    project = Path(__file__).resolve().parents[1]
    runtime_tree = ast.parse((project / 'runtime.py').read_text())
    fsdp_export_calls = [node for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'self.fsdp_layer_state_rows']
    assert len(fsdp_export_calls) == 1
    assert ast.unparse(fsdp_export_calls[0].args[0]) == 'context.model'
    storage_calls = [node for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'self.register_parameter_storages']
    assert len(storage_calls) == 1
    assert isinstance(storage_calls[0].args[0], ast.IfExp)
    assert ast.unparse(storage_calls[0].args[0].body) == 'context.model'
    assert ast.unparse(storage_calls[0].args[0].orelse) == 'callback.models'
    report['fsdp_wrapper_entrypoints'] = dict(storage='context.model', state_export='context.model')
    for filename in ('runtime.py', 'predictor/measured_modules.py', 'predictor/fast_predictor.py'):
        compile((project / filename).read_text(), str(project / filename), 'exec')
    accepted_profiles.connection.close()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', profile_count=len(rows), report=str(args.output)),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
