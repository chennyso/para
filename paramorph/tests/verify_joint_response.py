import argparse
import json
from pathlib import Path
import sqlite3
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'paramorph'),str(SOURCE/'experiments'),str(SOURCE)]

from megatron import core
core.__path__.append(str(SOURCE/'integration/megatron/core'))

from agents.pvc_mailbox import OwnedMailbox
from agents.pvc_provider import PVCStructuredProvider
from search.graph_action_compiler import GraphActionCompiler
from search.investigation_control import PlanningInvestigationBatch
from search.system_search import EvidenceGuidedSearch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bridge-root',type=Path,required=True)
    parser.add_argument('--policy-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--failed-context',type=Path,required=True)
    parser.add_argument('--isolated-mailbox',action='store_true',required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config = json.loads(args.config.read_text())
    original = Path(config['output_directory'])
    for source,target in ((Path(config['profile_database']),'costs.sqlite'),
            (original/'experiments.sqlite','experiments.sqlite')):
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as reader:
            with sqlite3.connect(args.output/target) as writer:
                reader.backup(writer)
    config.update(output_directory=str(args.output),profile_database=str(args.output/'costs.sqlite'),
        gpu_execution_enabled=False,hardware_topology_paths=[str(original/'hardware-topology.json')])
    context = json.loads(args.failed_context.read_text())
    search = EvidenceGuidedSearch(config)
    search._planning_graph_impact = context['observation']['graph_strategy_impact']
    search._planning_quantitative_evidence = context['observation']['graph_quantitative_evidence']
    sys.path.insert(0,str(args.bridge_root))
    import agent_transport
    agent_transport.bridge.MAILBOX = args.output/'agent-mailbox'
    OwnedMailbox(agent_transport.bridge.MAILBOX).activate()
    search.provider = PVCStructuredProvider(agent_transport.bridge,args.output/'agents')
    schema = search.scoped_planning_schema(PlanningInvestigationBatch)
    assert schema.model_json_schema()['properties']['reason']['maxLength']==600
    intent = schema.model_fields['actions'].annotation.__args__[0].model_fields['graph_intent'].annotation
    assert 'affected_regions' not in intent.model_fields and 'target_resources' not in intent.model_fields
    parent = search.store.trial(context['observation']['parent_trial_id'])
    before = search.store.usage()
    batch,identifier = search.invoke_with_tools('JointStrategyAgent',context['instructions'],context['observation'],schema)
    assert batch.actions and not batch.tool_requests
    results = []
    for action in batch.actions:
        shapes,rejected = search.investigation_shapes(action,parent)
        assert shapes
        for shape in shapes:
            GraphActionCompiler(search._planning_graph_impact).compile(action.graph_intent,
                parent['plan']['backend_config'],shape,action.parallel_ranges.model_dump(exclude_none=True),
                action.runtime_ranges.model_dump(exclude_none=True))
        results.append(dict(action=action.model_dump(mode='json'),native_shapes=len(shapes),rejected_shapes=len(rejected)))
    after = search.store.usage()
    report = dict(status='PASS',source='actual_failed_observation_and_authenticated_Agent',
        evidence_id=identifier,actions=results,reason=batch.reason,new_gpu_runs=0,
        usage={key:after.get(key,0)-before.get(key,0) for key in set(before)|set(after)},
        runtime_limit_seconds=config['budget']['max_walltime_seconds'])
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('actions','reason')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
