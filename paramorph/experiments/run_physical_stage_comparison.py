import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_data_models import HardwareTopology
from graph.capability_registry import CapabilityRegistry
from graph.complete_rewrite import CompleteAndRewrite
from graph.model_graph import read_model
from predictor.fast_predictor import FastPredictor
from predictor.measured_modules import optimizer_local_cost_evidence
from search.physical_stage_master import PartitionDomain
from search.physical_stage_search import (
    AgentPricer, DeterministicPricer, JointSearchBudget,
    NativeCompositePricingOracle, PhysicalStageJointSearch, PhysicalStageOracle,
)


def prepare(workspace, model_name, directory):
    campaign = workspace / 'outputs' / ('mimo-full-search' if model_name == 'mimo' else 'qwen3-14b-full-search')
    settings = json.loads((campaign / 'system-config.json').read_text())
    frozen_directory = workspace / 'outputs/predictor-measurement-campaign' / f'{model_name}-current-hardware'
    frozen = json.loads((frozen_directory / f'{model_name}-frozen-inputs.json').read_text())
    validation = frozen_directory / f'{model_name}-costs.sqlite'
    target = directory / 'costs.sqlite'
    if target.exists():
        raise ValueError('新实验目录已经包含成本数据库')
    with sqlite3.connect(validation.resolve().as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(target) as destination:
            original.backup(destination)
    predictor = FastPredictor(target)
    scalar_costs = workspace / 'outputs/optimizer-cost-campaign/qwen14-scalar-nccl/costs.sqlite'
    with sqlite3.connect(scalar_costs.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        for identifier, encoded in connection.execute('SELECT id, document FROM communication_profiles'):
            predictor.database.add_communication(identifier, json.loads(encoded))
    profiling = workspace / 'outputs/optimizer-cost-campaign' / model_name / 'profiling/runs'
    run = profiling / ('paramorph-mimo-optimizer-cost-20261011-r1' if model_name == 'mimo'
        else 'paramorph-qwen3-14b-tp2pp8-optimizer-cost-20261011-r3')
    profile_id = run.name + ':raw'
    stored = predictor.database.connection.execute('SELECT document FROM profiles WHERE id=?', (profile_id,)).fetchone()
    if stored is None:
        predictor.ingest(run)
    else:
        profile = json.loads(stored[0])
        if not optimizer_local_cost_evidence(profile)['verified']:
            raise ValueError('已有成本记录没有通过独立Optimizer事件检查')
    model, graph = read_model(settings['model_config_path'])
    hardware = frozen['hardware']
    devices = [device | dict(node=node.get('node', node['hostname']), node_index=index,
        physical_rank=index * 8 + device['index']) for index, node in enumerate(hardware)
        for device in node['devices']]
    topology = HardwareTopology(16, 2, 8, min(device['capacity_bytes'] for device in devices) / 2**30,
        devices=devices, network_paths=[adapter | dict(node_index=index)
            for index, node in enumerate(hardware) for adapter in node['adapters']],
        software=hardware[0]['software'],
        environment_per_node={str(index): node['launch_environment'] for index, node in enumerate(hardware)})
    parameters = frozen['predictor_parameters']
    parameters.setdefault('network', {})['environment_per_node'] = topology.environment_per_node
    predictor.configure_projection(model, graph, topology, parameters)
    predictor.set_preferred_profile(profile_id)
    with sqlite3.connect((campaign / 'experiments.sqlite').as_uri() + '?mode=ro', uri=True) as connection:
        trials = [json.loads(row[0]) for row in connection.execute('SELECT document FROM trials')]
    seed_id = ('paramorph-mimo-16gpu-baseline-20261011-cfd9dffb3f97' if model_name == 'mimo'
        else 'paramorph-qwen3-14b-ddp-tp2pp4cp1dp2-global-20261011')
    request = dict(next(row['plan']['backend_config'] for row in trials if row['trial_id'] == seed_id))
    manifests = [json.loads((run / 'artifacts' / f'rank{rank}.manifest.json').read_text())
        for rank in range(16)]
    widths = manifests[0]['optimizer_state_dtypes']
    if any(row['optimizer_state_dtypes'] != widths or row['pipeline_dtype'] != manifests[0]['pipeline_dtype']
            for row in manifests):
        raise ValueError('实际Rank的训练状态精度不一致')
    for field, name in (('main_grads_dtype', 'main_grads'), ('main_params_dtype', 'main_params'),
            ('exp_avg_dtype', 'exp_avg'), ('exp_avg_sq_dtype', 'exp_avg_sq')):
        request[field] = widths[name].removeprefix('torch.')
    request['pipeline_dtype'] = manifests[0]['pipeline_dtype'].removeprefix('torch.')
    request.update(execution_backend='megatron_ddp', optimizer_sharding='distributed',
        precision_aware_optimizer=manifests[0]['precision_aware_optimizer'])
    request['expert_dp'] = 16 // (request['etp'] * request['ep'] * request['pp'])
    request['dp_inner_size'], request['dp_outer_size'] = request['dp'] * request['cp'], 1
    request['expert_dp_inner_size'], request['expert_dp_outer_size'] = request['expert_dp'], 1
    request['outer_sharding_strategy'], request['expert_outer_sharding_strategy'] = 'no_shard', 'no_shard'
    registry = CapabilityRegistry(graph['module_structure'])
    rewriter = CompleteAndRewrite(model, topology, registry,
        registry.parallel_domains(model, topology, request))
    manifest = dict(model_name=model_name, request=request, profile_id=profile_id,
        model_config_path=settings['model_config_path'], hardware_path=str(campaign / 'hardware-topology.json'),
        historical_component_run=str(run), database_input=str(validation),
        database_input_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
        hardware_sha256=hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest(),
        frozen_input=str(frozen_directory / f'{model_name}-frozen-inputs.json'),
        module_kinds=[kind.value for kind in model.layer_types],
        source='archived_actual_component_measurements_and_existing_analytical_projection',
        state_dtype_evidence=[dict(rank=row['rank'], dtypes=row['optimizer_state_dtypes']) for row in manifests])
    (directory / 'inputs.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return predictor, rewriter, request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-name', choices=('mimo', 'qwen14'), required=True)
    parser.add_argument('--method', choices=('deterministic', 'agent'), required=True)
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--column-queries', type=int, default=96)
    parser.add_argument('--complete-evaluations', type=int, default=12)
    parser.add_argument('--wall-seconds', type=float, default=1200)
    parser.add_argument('--provider-source', type=Path)
    parser.add_argument('--agent-model')
    parser.add_argument('--pricing-costs', choices=('full_projection', 'native_composite'), default='native_composite')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictor, rewriter, request = prepare(args.workspace, args.model_name, args.output)
    domain = PartitionDomain.around(request, args.radius)
    if args.pricing_costs == 'native_composite':
        identifier = ('paramorph-mimo-optimizer-cost-20261011-r1:raw' if args.model_name == 'mimo'
            else 'paramorph-qwen3-14b-tp2pp4-optimizer-cost-20261011-r1:raw')
        encoded = predictor.database.connection.execute('SELECT document FROM profiles WHERE id=?', (identifier,)).fetchone()
        if encoded is None:
            raise ValueError('复合定价需要已指定的实际完整家族采集记录')
        profile = json.loads(encoded[0])
        request = dict(profile['request'])
        for field, default in (('main_grads_dtype', 'float32'), ('main_params_dtype', 'float32'),
                ('exp_avg_dtype', 'float32'), ('exp_avg_sq_dtype', 'float32')):
            request[field] = request.get(field, default)
        domain = PartitionDomain.around(request, args.radius)
        oracle = NativeCompositePricingOracle(domain, request, predictor, rewriter, args.output, profile)
    else:
        oracle = PhysicalStageOracle(domain, request, predictor, rewriter, args.output)
    if args.method == 'agent':
        if args.provider_source is None:
            raise ValueError('Agent实验必须提供现有真实Provider代码目录')
        sys.path.insert(0, str(args.provider_source))
        from megatron.core.frontier.agents.provider import CodexJSONProvider
        provider = CodexJSONProvider(args.output / 'agents', model=args.agent_model)
        pricer = AgentPricer(provider.generate)
    else:
        pricer = DeterministicPricer()
    budget = JointSearchBudget(column_queries=args.column_queries,
        complete_evaluations=args.complete_evaluations, rounds=args.rounds,
        wall_seconds=args.wall_seconds, columns_per_round=max(1, 24 // request['pp']))
    (args.output / 'protocol.json').write_text(json.dumps(dict(method=args.method, replicate=args.replicate,
        request=request, domain=asdict(domain), budget=asdict(budget),
        cost_revision=oracle.revision, candidate_space_sha256=hashlib.sha256(
            json.dumps(asdict(domain), sort_keys=True).encode()).hexdigest(),
        implementation_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in
            sorted(Path(__file__).resolve().parents[1].rglob('*.py'))},
        new_training=False, pricing_costs=args.pricing_costs), ensure_ascii=False, indent=2) + '\n')
    result = PhysicalStageJointSearch(oracle, pricer, budget).run(
        f'{args.model_name}-{args.method}-{args.replicate}')
    best = result['best_prediction']
    print(json.dumps(dict(run=result['run_id'], usage=result['usage'],
        seed_ms=result['seed_step_time_ms'], best_ms=None if best is None else best['step_time_ms'],
        best_chunks=None if best is None else best['decoder_chunks'], stop=result['stop']), ensure_ascii=False))


if __name__ == '__main__':
    main()
