import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_data_models import HardwareTopology
from cost.cost_model import ExecutionCostModel
from cost.runtime_calibration import RuntimeCalibration
from scheduling.schedule_simulator import ScheduleCostEvaluator
from agents.critic_agent import CriticAgent


def system_command(args):
    from contracts import SystemConfig
    from search.system_search import EvidenceGuidedSearch
    from trace.trace_mapper import TraceMapper
    config = SystemConfig.model_validate_json(args.config.read_text())
    system = EvidenceGuidedSearch(config)
    request = config.request
    if args.request is not None:
        request = json.loads(args.request.read_text())
    if args.command == 'inspect':
        report = dict(model_graph=system.model_graph, capabilities=system.registry.describe(),
            backend_domains=config.capability_domains, profile_id=system.profile_id)
    elif args.command == 'validate':
        candidate = system.rewriter.candidate(request)
        report = dict(candidate=candidate.to_dict(), plan_hash=system.plan(request).plan_hash,
            validation_scope='structural_backend_legality')
    elif args.command == 'predict':
        report = system.predictor.evaluate(request)
    elif args.command == 'compare':
        requests = json.loads(args.candidates.read_text())
        predictions = [system.predictor.evaluate(row) for row in requests]
        report = dict(candidates=sorted(predictions, key=lambda row: row['performance']['step_time_ms']
            if row['performance']['step_time_ms'] is not None else float('inf')),
            ranking_probes=system.predictor.ranking_probes(requests[0], requests[1]) if len(requests) == 2 else None)
    elif args.command == 'joint-partition':
        from search.physical_stage_search import AgentPricer, DeterministicPricer, JointSearchBudget
        if args.pricing == 'agent':
            if not config.agent_execution_enabled:
                raise ValueError('当前配置没有启用真实Agent')
            if config.llm_provider == 'codex':
                from megatron.core.frontier.agents.provider import CodexJSONProvider
                provider = CodexJSONProvider(system.output / 'pricing-agents', model=config.agent_model)
            else:
                from megatron.core.frontier.agents.direct_provider import DirectAPIProvider
                provider = DirectAPIProvider(system.output / 'pricing-agents', model=config.agent_model,
                    base_url=config.llm_base_url)
            pricer = AgentPricer(provider.generate)
        else:
            pricer = DeterministicPricer()
        budget = JointSearchBudget(column_queries=args.column_queries,
            complete_evaluations=args.complete_evaluations, rounds=args.pricing_rounds,
            wall_seconds=config.budget.max_walltime_seconds)
        report = system.solve_physical_stage_partition(request, pricer, budget, args.boundary_radius,
            pricing_costs=args.pricing_costs, profile_id=args.profile_id or config.profile_id)
    elif args.command == 'profile':
        result, directory = system.adapter.execute(request, profile=True)
        if result['status'] != 'PASS':
            raise ValueError('真实成本采集执行失败')
        profile_id = system.predictor.ingest(directory)
        report = dict(measurement=result, profile_id=profile_id, trace=TraceMapper().map_runtime(directory))
    elif args.command == 'verify':
        prediction = system.predictor.evaluate(request)
        result, directory = system.adapter.execute(request, prediction)
        report = dict(prediction=prediction, measurement=result, directory=str(directory))
    elif args.command in ('optimize', 'resume'):
        report = system.run()
    else:
        from evaluation.system_report import generate_report
        report = generate_report(system.store, system.output / 'evaluation')
    system.output.mkdir(parents=True, exist_ok=True)
    (system.output / f'{args.command}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(command=args.command, output=str(system.output), status='COMPLETE'), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(prog='paramorph')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'validate', 'profile', 'predict', 'compare', 'optimize', 'verify', 'resume', 'evaluate', 'report', 'joint-partition'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--request', type=Path)
        command.add_argument('--candidates', type=Path)
        if name == 'joint-partition':
            command.add_argument('--pricing', choices=('agent', 'deterministic'), required=True)
            command.add_argument('--boundary-radius', type=int, default=2)
            command.add_argument('--column-queries', type=int, default=96)
            command.add_argument('--complete-evaluations', type=int, default=12)
            command.add_argument('--pricing-rounds', type=int, default=4)
            command.add_argument('--pricing-costs', choices=('native_composite', 'full_projection'), default='full_projection')
            command.add_argument('--profile-id')
    diagnose = commands.add_parser('diagnose')
    diagnose.add_argument('directory', type=Path)
    diagnose.add_argument('--output', type=Path, required=True)
    validate = commands.add_parser('validate-cost')
    validate.add_argument('directory', type=Path)
    validate.add_argument('--nodes', type=int, required=True)
    validate.add_argument('--gpus-per-node', type=int, required=True)
    validate.add_argument('--memory-gib', type=float, required=True)
    validate.add_argument('--held-out', type=int, default=1)
    validate.add_argument('--maximum-relative-error', type=float, default=0.1)
    validate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command not in ('diagnose', 'validate-cost'):
        system_command(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'diagnose':
        critic = CriticAgent()
        critic.load_runtime(args.directory)
        report = critic.generate_critic_report(critic.calibration.config).to_dict()
        (args.output / 'critic.json').write_text(json.dumps(report,
            ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    else:
        calibration = RuntimeCalibration(args.directory)
        costs = ExecutionCostModel()
        report = calibration.ingest(costs, args.held_out)
        topology = HardwareTopology(calibration.config.world_size, args.nodes,
                                   args.gpus_per_node, args.memory_gib)
        prediction = ScheduleCostEvaluator(costs).evaluate(
            calibration.execution_plan(costs, args.held_out), topology)
        report['schedule_prediction'] = asdict(prediction)
        costs.save(args.output / 'execution-cost-model.json')
        (args.output / 'cost-validation.json').write_text(json.dumps(report,
            ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if abs(report['relative_error']) > args.maximum_relative_error:
            raise ValueError('独立迭代预测误差超过指定验收范围')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
