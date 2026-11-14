import argparse
import json
import math
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1]/'experiments')]

from contracts import ModuleRelation
from run_physical_stage_comparison import prepare_pricing_oracle
from search.frontier_pricer import FrontierPricer,PricingPolicy
from search.linear_pricing_bound import LinearPricingBound
from search.module_relations import complete_partition_boundaries,prefix_support
from search.physical_stage_master import NativeEventMaster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    oracle = prepare_pricing_oracle(args.workspace,'mimo',args.output,1)
    domain = oracle.domain
    assert domain.pp==4 and domain.vpp==2 and domain.seed[-1]==36
    relations = [ModuleRelation(relation=kind,module_ids=ids,virtual_slots=slots,
        evidence_refs=[oracle.profile_id],rationale='真实MiMo原生分区上的模块归属约束')
        for kind,ids,slots in [('same_virtual',[3,4],[]),('same_physical',[0,10],[]),
            ('different_physical',[0,1],[]),('allowed_virtual',[35],[7])]]
    assert prefix_support(oracle.request['decoder_chunks'],36,4,2,relations)
    oracle.module_relations = tuple(relations)
    master = NativeEventMaster(domain,oracle.tasks,oracle.config.world_size,oracle.revision,relations)
    for stage in range(domain.pp):
        assert master.add(oracle.column(stage,domain.intervals(domain.seed,stage)))
    for stage in range(domain.pp):
        for intervals in domain.columns(stage):
            if oracle.witness(stage,intervals) is not None:
                master.add(oracle.column(stage,intervals))
    lp = master.solve_lp()
    integer = master.solve_integer()
    assert prefix_support(integer['decoder_chunks'],36,4,2,relations)
    fixed = {index:value for index,value in enumerate(integer['endpoints'])}
    assert complete_partition_boundaries(domain,fixed,relations) == tuple(integer['endpoints'])
    bound_checks = 0
    for stage in range(domain.pp):
        bound = LinearPricingBound(master,oracle,lp,stage)
        for intervals in domain.columns(stage):
            cost = master.reduced_cost(oracle.estimated_column(master,stage,intervals),lp['prices'])
            assert math.isclose(bound.bound(intervals),cost,abs_tol=1e-5)
            assert bound.bound(intervals[:1]) <= cost+1e-5
            bound_checks += 1
    pricing = FrontierPricer(PricingPolicy(reason='原生模块关系与实际LP对偶价格验证',evidence_refs=[oracle.revision]))
    rows,stats = pricing.propose(master,oracle,lp,4,dict(evaluations=[]))
    assert rows and not stats['pricing_certified']
    report = dict(status='PASS',profile_id=oracle.profile_id,physical_columns=len(master.columns),
        module_equality_rows=len(master.module_eq_rows),module_inequality_rows=len(master.module_ub_rows),
        bound_checks=bound_checks,integer_decoder_chunks=integer['decoder_chunks'],
        reduced_cost_solver_residual=lp['maximum_reduced_cost_residual'],
        no_solver_time_limit=True,new_gpu_runs=0)
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
