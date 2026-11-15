import argparse
import json
from pathlib import Path
import sqlite3
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE),str(SOURCE/'paramorph'),str(SOURCE/'experiments'),str(SOURCE/'integration')]

from megatron import core
core.__path__.append(str(SOURCE/'integration/megatron/core'))

from agents.pvc_provider import PVCStructuredProvider
from search.conditional_partition import ConditionalPartitionSolver
from search.system_search import EvidenceGuidedSearch


def copy_database(source,target):
    with sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True) as original:
        with sqlite3.connect(target) as destination:
            original.backup(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bridge-root',type=Path,required=True)
    parser.add_argument('--policy-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--isolated-mailbox',action='store_true')
    parser.add_argument('--resume-output',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume_output)
    config = json.loads(args.config.read_text())
    original_output = Path(config['output_directory'])
    if not args.resume_output:
        copy_database(config['profile_database'],args.output/'costs.sqlite')
        copy_database(original_output/'experiments.sqlite',args.output/'experiments.sqlite')
    config.update(output_directory=str(args.output),profile_database=str(args.output/'costs.sqlite'),
        hardware_topology_paths=[str(original_output/'hardware-topology.json')],
        gpu_execution_enabled=False,planning_partition_engine='frontier',
        planning_pricing_program_path=str(args.policy_model),
        planning_frontier_lp_rounds=2,planning_frontier_column_queries=12,
        investigation_max_seconds=None)
    sys.path.insert(0,str(args.bridge_root))
    import agent_transport
    if args.isolated_mailbox:
        from agents.pvc_mailbox import OwnedMailbox
        agent_transport.bridge.MAILBOX = args.output/'agent-mailbox'
        OwnedMailbox(agent_transport.bridge.MAILBOX).activate()
    search = EvidenceGuidedSearch(config)
    search.provider = PVCStructuredProvider(agent_transport.bridge,args.output/'agents')
    trials = [row for row in search.store.trials() if (row.get('prediction') or {}).get(
        'prediction_status') == 'PREDICTED_CALIBRATED' and (row.get('prediction') or {}).get(
        'memory', {}).get('admission')=='WITHIN_ESTIMATED_CAPACITY']
    trial = min(trials,key=lambda row:row['prediction']['performance']['step_time_ms'])
    request = trial['plan']['backend_config']
    prediction = search.predict_request(request,parent_request=request)
    before = search.store.usage()
    profiles = [json.loads(row[0]) for row in search.predictor.database.connection.execute('SELECT document FROM profiles')]
    solver = ConditionalPartitionSolver(search.rewriter,profiles,backend_source_hashes=search.backend_identity,
        boundary_index=search.execution_boundary_index(),backend_source_resolver=search.native_profile_source_hashes)
    profile = solver._profile(request)

    def evaluate(candidate,candidate_graph=None):
        return search.evaluate_planning_request(candidate,request,candidate_graph=candidate_graph)

    result = search.run_frontier_investigation(request,prediction,profile,evaluate,(),2,'controller-native-verification')
    after = search.store.usage()
    assert after.get('cpu_predictions',0)-before.get('cpu_predictions',0) <= 2
    receipts = list((args.output/'agents').glob('*/provenance.json'))
    assert receipts and all(json.loads(path.read_text())['provider']=='authenticated_codex_app_server_pvc_bridge'
        for path in receipts)
    assert search.store.search_states('PHYSICAL_STAGE_PROGRAM')
    assert config['budget'].get('max_walltime_seconds') is None
    report = dict(status='PASS',case=trial['trial_id'],native_program_status=result['status'],
        actual_agent_calls=result['usage']['provider_calls'],authenticated_agent_receipts=len(receipts),new_gpu_runs=0,
        newly_charged={key:after.get(key,0)-before.get(key,0) for key in set(before)|set(after)},
        native_search=result,original_pod_unchanged=True,no_runtime_limit=True)
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='native_search'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
