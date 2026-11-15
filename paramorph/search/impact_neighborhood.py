from dataclasses import asdict, dataclass
from itertools import accumulate

import networkx as nx

from contracts import fingerprint
from graph.bottleneck_analysis import BottleneckGraph
from search.physical_stage_master import task_stage


@dataclass(frozen=True)
class ExecutionNeighborhood:
    changed_variables: tuple
    affected_task_ids: tuple
    boundary_dependencies: tuple
    boundary_resources: tuple
    live_buffers: tuple
    collective_interfaces: tuple
    fixed_decisions: tuple
    cost_revision: str
    parent_graph_id: str
    candidate_graph_id: str
    task_membership_changed: bool
    constraints: tuple
    decision_hypergraph: dict

    def report(self):
        document = asdict(self)
        return dict(document, neighborhood_id=fingerprint(document),
            boundary_size=len(self.boundary_dependencies) + len(self.boundary_resources)
                + len(self.live_buffers) + len(self.collective_interfaces),
            independence_certified=False, requires_complete_replay=True,
            scope='complete_observed_graph_interfaces_and_conservative_shared_memory_constraints')


def decision_constraint_hypergraph(parent_request,candidate_request,graphs):
    old_ends = tuple(accumulate(parent_request['decoder_chunks'],initial=0))
    new_ends = tuple(accumulate(candidate_request['decoder_chunks'],initial=0))
    globals_changed = sorted(key for key in set(parent_request)|set(candidate_request)
        if key not in ('run_id','decoder_chunks') and parent_request.get(key) != candidate_request.get(key))
    boundaries = ([index for index in range(1,len(old_ends)-1) if old_ends[index] != new_ends[index]]
        if len(old_ends)==len(new_ends) else list(range(1,max(len(old_ends),len(new_ends))-1)))
    variables = globals_changed+[f'boundary:{index}' for index in boundaries]
    edges = []
    for label,inventory in zip(('parent','candidate'),graphs):
        associations = {}
        for key,row in inventory.records.items():
            region = row.get('region',row.get('region_id'))
            associations[key] = set(globals_changed)
            if region is not None:
                associations[key].update(f'boundary:{index}' for index in boundaries
                    if index in (region,region+1))
        rank_tasks = {}
        for key,row in inventory.records.items():
            rank_tasks.setdefault(row['rank'],set()).add(key)
        constraints = [('resource',str(resource),ids) for resource,ids in inventory.resource_tasks.items()]
        constraints += [('memory_rank',str(rank),ids) for rank,ids in rank_tasks.items()]
        constraints += [('collective',fingerprint(identity),ids)
            for identity,ids in inventory.collective_tasks.items()]
        constraints += [('dependency',fingerprint((parent,child)),{parent,child})
            for child,parents in inventory.predecessors.items() for parent in parents]
        for kind,identity,task_ids in constraints:
            members = sorted(set().union(*(associations[key] for key in task_ids)))
            if members:
                edges.append(dict(id=fingerprint(dict(graph=label,kind=kind,identity=identity)),
                    graph=label,kind=kind,identity=identity,variables=members,task_ids=sorted(task_ids)))
    graph = nx.Graph()
    graph.add_nodes_from(('variable',name) for name in variables)
    for edge in edges:
        graph.add_edges_from((('variable',name),('constraint',edge['id'])) for name in edge['variables'])
    components = []
    for nodes in nx.connected_components(graph):
        names = sorted(value for kind,value in nodes if kind=='variable')
        ids = sorted(value for kind,value in nodes if kind=='constraint')
        components.append(dict(variables=names,constraint_ids=ids,boundary_state_weight=len(ids),
            native_repair=dict(engines=['conditional_partition','physical_stage_joint_search'],
                allowed_schedules=['1f1b','interleaved_1f1b'],arbitrary_event_reordering=False)))
    return dict(variables=variables,hyperedges=edges,
        repair_components=sorted(components,key=lambda row:row['variables']),
        decomposition_scope='observed_complete_native_constraints',
        recombination_requires_complete_replay=True)


def cross_stage_interfaces(graph, lanes):
    inventory = BottleneckGraph(graph)
    result = {}
    for identity,ids in inventory.collective_tasks.items():
        stages = sorted({task_stage(inventory.records[key],lanes) for key in ids})
        if len(stages) < 2:
            continue
        tasks = {key:{name:value for name,value in inventory.records[key].items()
            if name not in ('start_ms','end_ms','duration_ms')} for key in sorted(ids)}
        result[fingerprint(identity)] = dict(stages=stages,signature=fingerprint(tasks))
    return result


def execution_neighborhood(parent_report, candidate_report, parent_request, candidate_request,
                           cost_revision):
    if not cost_revision:
        raise ValueError('执行影响范围需要明确成本版本')
    graphs = [BottleneckGraph(report['execution_graph'])
        for report in (parent_report, candidate_report)]
    previous,current = graphs
    variables = tuple(sorted(key for key in set(parent_request) | set(candidate_request)
        if key != 'run_id' and parent_request.get(key) != candidate_request.get(key)))
    affected = set(previous.records) ^ set(current.records)
    for key in previous.records.keys() & current.records.keys():
        def semantic(row):
            return {name:value for name,value in row.items() if name not in ('start_ms','end_ms')}
        if semantic(previous.records[key]) != semantic(current.records[key]):
            affected.add(key)
    if variables and not affected:
        # 未反映为任务差异的配置变化仍然需要完整检查。
        affected.update(current.records)
    for graph in graphs:
        seeds = affected & graph.records.keys()
        if seeds:
            affected.update(graph.closure(seeds,dependency_depth=1,
                include_resources=False)['task_ids'])
    dependencies,resources,buffers,collectives,constraints = [],[],[],[],[]
    for label,graph,report in zip(('parent','candidate'),graphs,(parent_report,candidate_report)):
        selected = affected & graph.records.keys()
        ranks = {graph.records[key]['rank'] for key in selected}
        for child,parents in graph.predecessors.items():
            for parent in parents:
                if (child in selected) != (parent in selected):
                    dependencies.append(dict(graph=label,parent=parent,child=child))
        for resource,ids in sorted(graph.resource_tasks.items()):
            if ids & selected:
                row = dict(graph=label,resource_id=resource,
                    inside_task_ids=sorted(ids & selected),outside_task_ids=sorted(ids - selected))
                resources.append(row)
                constraints.append(dict(graph=label,kind='shared_resource',variables=variables,
                    identity=resource,task_ids=sorted(ids)))
        for identity,ids in graph.collective_tasks.items():
            if ids & selected:
                collectives.append(dict(graph=label,identity=identity,task_ids=sorted(ids),
                    participant_ranks=sorted({rank for key in ids for rank in graph._participants(key)}),
                    dependencies={key:sorted(graph.predecessors[key]) for key in sorted(ids)}))
        for row in report['memory'].get('buffer_lifetimes', []):
            if set(row['physical_ranks']) & ranks:
                buffers.append(dict(graph=label,**row))
        for rank in sorted(ranks):
            constraints.append(dict(graph=label,kind='shared_memory',identity=rank,
                variables=variables,buffer_ids=sorted(row['buffer_id'] for row in buffers
                    if row['graph'] == label and rank in row['physical_ranks']),
                persistent_state=report['memory'].get('breakdown_per_rank', {}).get(
                    rank,report['memory'].get('breakdown_per_rank', {}).get(str(rank)))))
    fixed = tuple((key,parent_request[key]) for key in sorted(parent_request)
        if key != 'run_id' and key not in variables)
    return ExecutionNeighborhood(variables,tuple(sorted(affected)),tuple(dependencies),
        tuple(resources),tuple(buffers),tuple(collectives),fixed,cost_revision,
        previous.graph_id,current.graph_id,set(previous.records) != set(current.records),tuple(constraints),
        decision_constraint_hypergraph(parent_request,candidate_request,graphs))
