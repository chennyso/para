import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]

from run_physical_stage_comparison import prepare_pricing_oracle
from search.frontier_pricer import FrontierPricer, PricingPolicy
from search.physical_stage_search import JointSearchBudget, PhysicalStageJointSearch
from search.pricing_program_learning import LearnedPricingProgram, train_pricing_programs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calls',type=int,default=8)
    args = parser.parse_args()
    protocol = json.loads(args.cases.read_text())
    budget = JointSearchBudget(**protocol['search_budget'])
    if budget.wall_seconds is not None:
        raise ValueError('离线程序训练使用明确的候选查询数量，不设置运行时限')

    def factory(case,directory):
        return prepare_pricing_oracle(args.workspace,case['model_name'],directory,
            case.get('radius',1),case.get('pricing_costs','native_composite'),case.get('decoder_chunks'))

    manifest = train_pricing_programs(protocol['training_cases'],factory,budget,args.calls,args.output)
    learned = LearnedPricingProgram(args.output)
    heldout = []
    for index,case in enumerate(protocol['heldout_cases']):
        directory = args.output/f'heldout-{index:03d}'
        directory.mkdir()
        oracle = factory(case,directory)
        proposal = learned.propose(oracle)
        from contracts import fingerprint
        graph_id = fingerprint(oracle.seed_report['execution_graph'])
        if graph_id in proposal['training_graphs']:
            raise ValueError('独立执行图验证与训练执行图重复')
        policy = PricingPolicy.model_validate(proposal['policy'])
        result = PhysicalStageJointSearch(oracle,FrontierPricer(policy,policy_model=args.output),budget).run(directory.name)
        heldout.append(dict(case=case,proposal=proposal,usage=result['usage'],
            best_prediction=result['best_prediction'],best_pricing_proxy=result['best_pricing_proxy']))
    report = dict(status='PASS',training_graphs=len(manifest['training_rows']),
        heldout_graphs=len(heldout),heldout=heldout,model_evidence_id=manifest['evidence_id'],
        search_budget=asdict(budget),new_gpu_runs=0,training_speedup_claimed=False)
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='heldout'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
