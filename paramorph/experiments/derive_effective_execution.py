import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import TrainingPlan, fingerprint
from predictor.profile_database import normalized_execution, plan_signature, resolved_clip_grad, signature


def file_evidence(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_database_document(path, table, identifier):
    if table not in ('trials', 'profiles'):
        raise ValueError('只允许读取训练计划与成本记录')
    database = Path(path).resolve()
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        row = connection.execute(f'SELECT document FROM {table} WHERE id=?', (identifier,)).fetchone()
    if row is None:
        raise ValueError(f'来源数据库缺少记录：{identifier}')
    return json.loads(row[0])


def inspect_execution(directory):
    directory = Path(directory).resolve()
    request_path, result_path = directory / 'request.json', directory / 'result.json'
    request, result = json.loads(request_path.read_text()), json.loads(result_path.read_text())
    if result['run_id'] != request['run_id'] or result['status'] != 'PASS':
        raise ValueError('实际执行证据需要同一运行的成功结果')
    files = [file_evidence(request_path), file_evidence(result_path)]
    observed, records, logged_nodes = set(), [], []
    pattern = re.compile(r'(?:^|[,(]\s*)clip_grad=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=,|\))')
    for node in range(request['node_count']):
        log = directory / 'artifacts' / f'node{node}.log'
        files.append(file_evidence(log))
        found = []
        for line_number, line in enumerate(log.read_text().splitlines(), start=1):
            if 'Setting up optimizer with config OptimizerConfig(' not in line:
                continue
            matches = pattern.findall(line)
            if len(matches) != 1:
                raise ValueError('OptimizerConfig日志缺少唯一clip_grad字段')
            value = resolved_clip_grad(dict(clip_grad=float(matches[0])))
            observed.add(value)
            found.append(dict(node_index=node, line_number=line_number, clip_grad=value,
                record_sha256=hashlib.sha256(line.encode()).hexdigest(), record=line))
        if found:
            logged_nodes.append(node)
        records.extend(found)
    if len(observed) != 1:
        raise ValueError('实际OptimizerConfig记录缺失或clip_grad不一致')
    effective = observed.pop()
    return dict(run_id=request['run_id'], source_directory=str(directory), source_files=files,
        requested_clip_grad=resolved_clip_grad(request), effective_clip_grad=effective,
        optimizer_config_records=records,
        optimizer_config_logged_nodes=logged_nodes,
        observation_scope='actual_optimizer_configuration_in_available_backend_logs',
        measurement_reference=dict(run_id=result['run_id'], **file_evidence(result_path))), request


def build_derivation(search_directory):
    search_directory = Path(search_directory).resolve()
    config_path = search_directory / 'system-config.json'
    config = json.loads(config_path.read_text())
    trial_database = search_directory / 'experiments.sqlite'
    trial_id = config['request']['run_id']
    trial = read_database_document(trial_database, 'trials', trial_id)
    profile = read_database_document(config['profile_database'], 'profiles', config['profile_id'])
    baseline_evidence, baseline_request = inspect_execution(config['baseline_directory'])
    profile_evidence, profile_request = inspect_execution(profile['source_directory'])
    if baseline_evidence['effective_clip_grad'] != profile_evidence['effective_clip_grad']:
        raise ValueError('Baseline与Profile的实际梯度裁剪阈值不同')
    if normalized_execution(trial['plan']['backend_config']) != normalized_execution(baseline_request):
        raise ValueError('保存的Baseline计划与原始请求不一致')
    if normalized_execution(profile['request']) != normalized_execution(profile_request):
        raise ValueError('成本数据库请求与原始Profile请求不一致')
    source_records = [dict(database=str(trial_database), table='trials', record_id=trial_id,
        content_fields=['plan', 'measurement'],
        content_sha256=fingerprint({key: trial[key] for key in ('plan', 'measurement')})),
        dict(database=str(Path(config['profile_database']).resolve()), table='profiles',
            record_id=config['profile_id'], content_fields=None, content_sha256=fingerprint(profile))]
    core = dict(schema_version=1, derivation_kind='observed_optimizer_execution',
        field='clip_grad', effective_clip_grad=baseline_evidence['effective_clip_grad'],
        source_search_config=file_evidence(config_path), source_database_records=source_records,
        baseline=baseline_evidence, profile=profile_evidence,
        measurement_status='reused_original_measurement_with_observed_execution_semantics')
    identity = fingerprint(core)
    request = dict(trial['plan']['backend_config'], clip_grad=core['effective_clip_grad'])
    plan_document = deepcopy(trial['plan'])
    plan_document.update(plan_id=trial_id + ':effective:' + identity[:16], backend_config=request,
        optimizer_configuration=dict(plan_document['optimizer_configuration'],
            clip_grad=core['effective_clip_grad']))
    plan = TrainingPlan.model_validate(plan_document)
    derived_plan = dict(derivation_id=identity, source_trial_id=trial_id,
        source_stored_plan_hash=trial['plan_hash'], plan=plan.model_dump(mode='json'),
        plan_hash=plan.plan_hash, execution_signature=signature(request),
        plan_signature=plan_signature(request), measurement_reference=baseline_evidence['measurement_reference'])
    derived_profile = deepcopy(profile)
    derived_profile.update(id=profile['id'] + ':effective:' + identity[:16],
        request=dict(profile['request'], clip_grad=core['effective_clip_grad']),
        execution_semantics_evidence=dict(derivation_id=identity,
            source_profile_id=profile['id'], source_profile_sha256=fingerprint(profile),
            requested_clip_grad=profile_evidence['requested_clip_grad'],
            effective_clip_grad=core['effective_clip_grad']))
    return core | dict(derivation_id=identity), derived_plan, derived_profile


def write_derivation(search_directory, output_directory):
    sidecar, plan, profile = build_derivation(search_directory)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ('effective-plan.json', 'effective-profile.json', 'execution-semantics.json')}
    if any(path.exists() for path in paths.values()):
        raise ValueError('派生结果已经存在，请验证现有证据或指定新的输出目录')
    for name, document in (('effective-plan.json', plan), ('effective-profile.json', profile)):
        with paths[name].open('x') as stream:
            stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    sidecar['derived_files'] = {name: file_evidence(paths[name])
        for name in ('effective-plan.json', 'effective-profile.json')}
    with paths['execution-semantics.json'].open('x') as stream:
        stream.write(json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return load_verified_derivation(paths['execution-semantics.json'])


def load_verified_derivation(sidecar_path):
    sidecar = json.loads(Path(sidecar_path).read_text())
    core = {key: value for key, value in sidecar.items() if key not in ('derived_files', 'derivation_id')}
    if fingerprint(core) != sidecar['derivation_id']:
        raise ValueError('派生执行证据的内容与身份不一致')
    files = [sidecar['source_search_config'], *sidecar['baseline']['source_files'],
        *sidecar['profile']['source_files'], *sidecar['derived_files'].values()]
    for evidence in files:
        if file_evidence(evidence['path']) != evidence:
            raise ValueError(f'执行证据文件的SHA256发生变化：{evidence["path"]}')
    for source in sidecar['source_database_records']:
        document = read_database_document(source['database'], source['table'], source['record_id'])
        compared = ({key: document[key] for key in source['content_fields']}
            if source['content_fields'] is not None else document)
        if fingerprint(compared) != source['content_sha256']:
            raise ValueError('原始数据库证据内容发生变化')
    plan = json.loads(Path(sidecar['derived_files']['effective-plan.json']['path']).read_text())
    profile = json.loads(Path(sidecar['derived_files']['effective-profile.json']['path']).read_text())
    validated = TrainingPlan.model_validate(plan['plan'])
    if (validated.plan_hash != plan['plan_hash']
            or resolved_clip_grad(validated.backend_config) != sidecar['effective_clip_grad']
            or resolved_clip_grad(profile['request']) != sidecar['effective_clip_grad']):
        raise ValueError('派生计划、成本与实际执行配置不一致')
    measurement = json.loads(Path(plan['measurement_reference']['path']).read_text())
    return dict(sidecar=sidecar, plan=plan, profile=profile, original_measurement=measurement)


def verified_baseline_documents(sidecar_path):
    sidecar_path = Path(sidecar_path).resolve()
    verified = load_verified_derivation(sidecar_path)
    request = deepcopy(verified['plan']['plan']['backend_config'])
    result = deepcopy(verified['original_measurement'])
    if request['run_id'] != result['run_id']:
        raise ValueError('派生Baseline必须保留原始实测run_id')
    result['raw_artifacts'] = str(Path(verified['sidecar']['baseline']['source_directory'], 'artifacts'))
    result['execution_semantics_evidence'] = dict(path=str(sidecar_path),
        derivation_id=verified['sidecar']['derivation_id'],
        requested_clip_grad=verified['sidecar']['baseline']['requested_clip_grad'],
        effective_clip_grad=verified['sidecar']['effective_clip_grad'],
        measurement_origin='existing_measured_run',
        source_result=verified['sidecar']['baseline']['measurement_reference'])
    return verified, request, result


def materialize_verified_baseline(sidecar_path, output_directory):
    verified, request, result = verified_baseline_documents(sidecar_path)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ('request.json', 'result.json', 'derived-baseline.json')]
    if any(path.exists() for path in paths):
        raise ValueError('派生Baseline目录已经包含记录')
    for path, document in zip(paths[:2], (request, result)):
        with path.open('x') as stream:
            stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    manifest = dict(schema_version=1, sidecar_path=str(Path(sidecar_path).resolve()),
        derivation_id=verified['sidecar']['derivation_id'],
        request=file_evidence(paths[0]), result=file_evidence(paths[1]),
        load_requirement='load_verified_baseline_before_reusing_measurement')
    with paths[2].open('x') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return load_verified_baseline(output)


def load_verified_baseline(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'derived-baseline.json').read_text())
    verified, expected_request, expected_result = verified_baseline_documents(manifest['sidecar_path'])
    if manifest['derivation_id'] != verified['sidecar']['derivation_id']:
        raise ValueError('派生Baseline的证据身份不同')
    for name, expected in (('request', expected_request), ('result', expected_result)):
        path = directory / (name + '.json')
        if file_evidence(path) != manifest[name] or json.loads(path.read_text()) != expected:
            raise ValueError('派生Baseline内容与已验证的原始实测证据不同')
    return dict(request=expected_request, result=expected_result, derivation=verified)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--baseline-from', type=Path)
    args = parser.parse_args()
    if args.baseline_from is not None:
        if args.search_directory is not None or args.verify is not None or args.output is None:
            parser.error('--baseline-from需要--output且不能同时创建或验证其他派生证据')
        result = materialize_verified_baseline(args.baseline_from, args.output)['derivation']
    elif args.verify is not None:
        if args.search_directory is not None or args.output is not None:
            parser.error('--verify不能与派生参数同时使用')
        result = load_verified_derivation(args.verify)
    else:
        if args.search_directory is None or args.output is None:
            parser.error('创建派生证据需要--search-directory和--output')
        result = write_derivation(args.search_directory, args.output)
    print(json.dumps(dict(derivation_id=result['sidecar']['derivation_id'],
        effective_clip_grad=result['sidecar']['effective_clip_grad'],
        plan_hash=result['plan']['plan_hash'], profile_id=result['profile']['id'],
        source_measurement_run_id=result['original_measurement']['run_id']), ensure_ascii=False))
