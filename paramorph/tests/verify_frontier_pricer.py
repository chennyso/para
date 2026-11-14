import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1] / 'experiments')]

from run_physical_stage_comparison import prepare
from search.frontier_pricer import FrontierPricer, PricingPolicy
from search.impact_neighborhood import execution_neighborhood
from search.physical_stage_master import NativeEventMaster, PartitionDomain
from search.physical_stage_search import DeterministicPricer, NativeCompositePricingOracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--agent-transport',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    predictor,rewriter,request = prepare(args.workspace,'mimo',args.output)
    encoded = predictor.database.connection.execute('SELECT document FROM profiles WHERE id=?',
        ('paramorph-mimo-optimizer-cost-20261011-r1:raw',)).fetchone()[0]
    profile = json.loads(encoded)
    request = dict(profile['request'],main_grads_dtype='float32',main_params_dtype='float32',
        exp_avg_dtype='float32',exp_avg_sq_dtype='float32')
    domain = PartitionDomain.around(request,1)
    oracle = NativeCompositePricingOracle(domain,request,predictor,rewriter,args.output,profile)
    master = NativeEventMaster(domain,oracle.tasks,oracle.config.world_size,oracle.revision)
    for stage in range(domain.pp):
        assert master.add(oracle.column(stage,domain.intervals(domain.seed,stage)))
    lp = master.solve_lp()
    policy = PricingPolicy(reason='归档MiMo原生事件与真实LP价格验证',
        evidence_refs=[oracle.revision],max_frontier_expansions=4096)
    frontier = FrontierPricer(policy)
    oracle.fragment_cache = None
    started = time.perf_counter()
    first,stats = frontier.propose(master,oracle,lp,6,dict(evaluations=[]))
    first_seconds = time.perf_counter() - started
    first_states = stats['frontier_states']
    assert len(first) == 6 and stats['provider_calls'] == 0
    assert not stats['pricing_certified'] and stats['pending_frontier_states'] > 0
    actual = {(row['stage'],tuple(tuple(pair) for pair in row['intervals'])) for row in first}
    frontier_cache = oracle.cost_fragments().report()
    oracle.fragment_cache = None
    started = time.perf_counter()
    reference,enumeration = DeterministicPricer().propose(master,oracle,lp,6,dict(evaluations=[]))
    enumeration_seconds = time.perf_counter() - started
    expected = {(stage,intervals) for stage in range(domain.pp) for intervals in domain.columns(stage)}
    assert enumeration['enumerated_columns'] == len(expected)
    assert all(math.isclose(left,right,abs_tol=1e-5) for left,right in zip(
        sorted(row['estimated_reduced_cost'] for row in first),
        sorted(row['estimated_reduced_cost'] for row in reference)))
    bound_checks = 0
    for stage,intervals in expected:
        value = master.reduced_cost(oracle.estimated_column(master,stage,intervals),lp['prices'])
        bound = frontier.pricing[stage]
        assert math.isclose(bound.bound(intervals),value,abs_tol=1e-5)
        for length in range(len(intervals)):
            assert bound.bound(intervals[:length]) <= value+1e-5
            bound_checks += 1
    iterations = 0
    frontier_states = stats['frontier_states']
    while frontier.pending:
        rows,stats = frontier.propose(master,oracle,lp,32,dict(evaluations=[]))
        selected = {(row['stage'],tuple(tuple(pair) for pair in row['intervals'])) for row in rows}
        assert not actual & selected
        actual.update(selected)
        frontier_states += stats['frontier_states']
        iterations += 1
        assert iterations <= len(expected)
    assert actual == expected and stats['enumeration_complete']
    assert stats['dominance_pruned'] == 0 and not stats['pricing_certified']
    selected = first[0]
    witness = domain.witness(selected['stage'],selected['intervals'])
    for stage in range(domain.pp):
        master.add(oracle.column(stage,domain.intervals(witness,stage)))
    solved_lp = master.solve_lp()
    assert solved_lp['maximum_reduced_cost_residual'] < 1e-5
    integer = master.solve_integer()
    replay = oracle.snapshot(integer['endpoints'])
    assert math.isclose(integer['objective_ms'],replay['performance']['step_time_ms'],abs_tol=1e-5)
    candidate_request = dict(request,decoder_chunks=[b-a for a,b in zip(witness,witness[1:])])
    changed = execution_neighborhood(oracle.seed_report,oracle.snapshot(witness),request,
        candidate_request,oracle.revision).report()
    assert changed['requires_complete_replay'] and not changed['independence_certified']
    assert changed['cost_revision'] == oracle.revision
    identical = execution_neighborhood(oracle.seed_report,oracle.seed_report,request,
        request,oracle.revision).report()
    assert identical['affected_task_ids'] == ()
    result = dict(status='CPU_VERIFIED_PENDING_AGENT' if args.agent_transport is not None else 'PASS',
        data_source=profile['id'],domain=asdict(domain),
        physical_columns=len(expected),frontier_columns=len(actual),frontier_states=frontier_states,
        first_batch_frontier_states=first_states,
        first_batch_seconds=first_seconds,enumeration_batch_seconds=enumeration_seconds,
        first_batch_reference_columns=enumeration['enumerated_columns'],
        first_batch_fragment_cache=frontier_cache,
        cross_stage_interface_count=len(master.interfaces),
        reduced_cost_solver_residual=solved_lp['maximum_reduced_cost_residual'],
        integer_matches_complete_event_replay=True,neighborhood_boundary_size=changed['boundary_size'],
        pricing_certified=False,dominance_pruned=0,new_gpu_runs=0)
    result['surrogate_bound_checks'] = bound_checks
    (args.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if args.agent_transport is not None:
        sys.path.insert(0,str(args.agent_transport))
        os.environ['PARAMELD_SOURCE_ROOT'] = str(Path(__file__).resolve().parents[2])
        import agent_transport
        from openai.lib._pydantic import to_strict_json_schema
        calls = []

        def generate(role,instructions,observation,schema):
            body = dict(role='JointStrategyAgent',model=agent_transport.bridge.MODEL,effort='low',
                prompt=instructions+'\n'+agent_transport.bridge.encoded(observation),
                schema=to_strict_json_schema(schema))
            from contracts import fingerprint
            request = dict(body,id=fingerprint(dict(body,created=time.time_ns())),deadline=None)
            (args.output/'agent-request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
            response = agent_transport.bridge.infer('codex',request)
            (args.output/'agent-response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
            calls.append(request['id'])
            return schema.model_validate(response['response']),response

        agent_pricer = FrontierPricer(generate=generate)
        observation = dict(evaluations=[],profile_id=oracle.profile_id,cost_revision=oracle.revision,
            family=dict(pp=domain.pp,vpp=domain.vpp),boundaries=domain.boundaries,
            module_costs=oracle.module_costs,prices=master.price_observation(solved_lp),
            capacity_bytes=next(iter(master.columns.values())).evidence['capacity_bytes'])
        rows,initial = agent_pricer.propose(master,oracle,solved_lp,2,observation)
        following,next_stats = agent_pricer.propose(master,oracle,solved_lp,2,observation)
        assert calls and len(calls) == 1 and initial['provider_calls'] == 1
        assert next_stats['provider_calls'] == 0
        assert not {(row['stage'],tuple(map(tuple,row['intervals']))) for row in rows} & {
            (row['stage'],tuple(map(tuple,row['intervals']))) for row in following}
        result['agent_policy'] = dict(status='PASS',actual_calls=len(calls),
            request_ids=calls,policy=agent_pricer.policy.model_dump(mode='json'),
            first_batch=len(rows),second_batch=len(following),new_gpu_runs=0)
        result['status'] = 'PASS'
        (args.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result['agent_policy'],ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
