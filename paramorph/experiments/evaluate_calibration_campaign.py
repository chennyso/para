import argparse
import hashlib
import json
from pathlib import Path

from experiments.validate_predictor_corrections import configure, cost_summary, graph_summary, save_json
from predictor.profile_database import plan_signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=['mimo', 'qwen14', 'falcon'])
    parser.add_argument('--campaign', action='append', default=[], type=Path)
    parser.add_argument('--profile-directory', action='append', default=[], type=Path)
    parser.add_argument('--communication-file', action='append', default=[], type=Path)
    parser.add_argument('--hardware-topology', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictor, requests, comparisons, snapshot = configure(
        args.workspace.resolve(), args.model, output, args.hardware_topology)
    registered, measured = [], {}
    for path in args.communication_file:
        payload = path.read_bytes()
        benchmark = json.loads(payload)
        predictor.database.add_communication(path.parent.parent.name + ':' + hashlib.sha256(payload).hexdigest(), benchmark)
    for campaign_path in args.campaign:
        campaign = json.loads(campaign_path.read_text())
        profile = Path(campaign['profile_directory'])
        anchor = Path(campaign['training_directory']) if campaign.get('training_directory') else None
        identifier = predictor.ingest(profile, anchor)
        registered.append(identifier)
        if anchor is not None:
            request = json.loads((anchor / 'request.json').read_text())
            measurement = json.loads((anchor / 'result.json').read_text())
            if measurement['status'] != 'PASS' or not measurement['performance_eligible']:
                raise ValueError('普通训练参照没有通过真实性能验收')
            measured[plan_signature(request)] = dict(path=str(anchor),
                step_time_ms=measurement['step_seconds'] * 1000,
                tokens_per_second=measurement['tokens_per_second'],
                peak_allocated_gib=measurement['peak_allocated_gib'])
    for path in args.profile_directory:
        registered.append(predictor.ingest(path))
    if not registered:
        raise ValueError('本轮评价至少需要一份通过真实训练的新增Profile')
    predictor.set_preferred_profile(registered[-1])
    rows, missing_communication = [], {}
    for candidate, previous in comparisons.items():
        request = requests[candidate]
        if args.model == 'qwen14' and request.get('execution_backend', 'megatron_ddp') != 'megatron_ddp':
            continue
        compatible = [item for item in predictor.database.reusable_compute_profiles(request, blind=False)
            if item['profile_id'] in registered]
        if compatible:
            documents = [json.loads(predictor.database.connection.execute(
                'SELECT document FROM profiles WHERE id=?', (item['profile_id'],)).fetchone()[0])
                for item in compatible]
            selected = max(documents, key=lambda row: (
                plan_signature(row['request']) == plan_signature(request),
                row['request']['decoder_chunks'] == request['decoder_chunks'], row['id']))
            predictor.set_preferred_profile(selected['id'])
        for environment in predictor.projector.parameters['network']['environment_per_node'].values():
            environment['CUDA_DEVICE_MAX_CONNECTIONS'] = str(request['cuda_device_max_connections'])
        report = predictor.evaluate_projection(request, None, None, persist=False, blind=False)
        for cost in report['evidence']['costs']:
            identity = cost.get('identity', {})
            if cost['status'] != 'NEEDS_PROFILE' or identity.get('collective') is None:
                continue
            value = {key: identity[key] for key in ('collective', 'payload_bytes', 'physical_ranks', 'dtype')}
            key = json.dumps(value, sort_keys=True)
            item = missing_communication.setdefault(key, dict(value, candidate_ids=[]))
            if candidate not in item['candidate_ids']:
                item['candidate_ids'].append(candidate)
        path = output / f'{candidate}.prediction.json'
        save_json(path, report, compact=True)
        current = measured.get(plan_signature(request))
        observed = previous['measured_ms'] if current is None else current['step_time_ms']
        predicted = report['performance'].get('raw_resource_aware_step_time_ms')
        row = dict(model=args.model, candidate_id=candidate,
            prediction_status=report['prediction_status'], raw_step_time_ms=predicted,
            measured_ms=observed, historical_measured_ms=previous['measured_ms'],
            current_measurement=current, signed_error_percent=None if predicted is None else 100 * (predicted / observed - 1),
            profiling_sources=registered, prediction_path=str(path),
            calibration_scope='实际模块与Optimizer独立组件成本标定；完整普通训练Step Time未用于预测锚定',
            missing_profiles=report['evidence']['missing_profiles'],
            cost_sources=cost_summary(report), graph=graph_summary(report),
            memory_admission=report['memory']['admission'],
            profiling_requests=report['evidence']['profiling_requests'])
        rows.append(row)
        save_json(output / 'comparison.json', rows)
        save_json(output / 'required-nccl.json', list(missing_communication.values()))
        print(json.dumps({key: row[key] for key in ('model', 'candidate_id', 'prediction_status',
            'raw_step_time_ms', 'measured_ms', 'signed_error_percent', 'missing_profiles')}, ensure_ascii=False), flush=True)
    save_json(output / 'provenance.json', dict(original_database=snapshot,
        registered_profile_ids=registered, current_measurements=measured,
        gpu_jobs_started_by_evaluation=0, external_llm_requests=0))


if __name__ == '__main__':
    main()
