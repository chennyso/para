import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict

from megatron.core.frontier.planner.obligations import (
    CollectiveOrderConflict, DeadlineMiss, DependencyViolation, EventOrderViolation, MemoryDeficit,
    ProgramCoverageConflict, ResourceConflict, TensorLifetimeConflict, UpdateIncomplete,
)
from megatron.core.frontier.planner.verifier import StructuralVerifier, VerificationReport, json_value


class TemporalVerifier:
    def check_program(self, structure, graph, dag, intervals, persistent_bytes, memory_limits,
            capability_catalog, pipeline=None, repairs=(), expected_communications=(), repair_result=None):
        structural = StructuralVerifier(capability_catalog).check(structure, graph, pipeline, repairs)
        if not structural.legal:
            return VerificationReport('program', structural.program_id, graph.scope,
                structural.obligations, structural.checks)
        if repair_result is not None:
            if repair_result.status != 'LEGAL_STRUCTURAL' or repair_result.after.program_id != structural.program_id:
                obligation = ProgramCoverageConflict('completed_mandatory_repair',
                    (structural.program_id, 'LEGAL_STRUCTURAL'), (repair_result.after.program_id, repair_result.status))
                return VerificationReport('program', structural.program_id, graph.scope,
                    (*repair_result.after.obligations, obligation), structural.checks)
            if repair_result.state_migrations and repair_result.migration_receipt is None:
                obligation = ProgramCoverageConflict('actual_training_state_migration', ('completed_migration_receipt',), ())
                return VerificationReport('program', structural.program_id, graph.scope, (obligation,), structural.checks)
        elif graph.scope == 'MODEL':
            obligation = ProgramCoverageConflict('completed_mandatory_repair', (structural.program_id,), ())
            return VerificationReport('program', structural.program_id, graph.scope, (obligation,), structural.checks)
        groups = {group.key: group.ranks for mapping in structure.region_mappings
            for _, rows in mapping.process_groups for group in rows}
        temporal = self.check(dag, intervals, graph.expected_parameter_keys, persistent_bytes,
            memory_limits, scope=graph.scope, process_groups=groups, expected_communications=expected_communications)
        obligations = (*temporal.obligations, *self.coverage_obligations(structure, graph, dag))
        identifier = hashlib.sha256((structural.program_id + ':' + temporal.program_id).encode()).hexdigest()
        unique = {item.identifier(): item for item in obligations}
        return VerificationReport('program', identifier, graph.scope, tuple(unique.values()),
            (*structural.checks, *temporal.checks, 'model_bound_microbatch_and_rank_coverage',
             'model_bound_gradient_groups_and_optimizer_ownership'))

    def coverage_obligations(self, structure, graph, dag):
        obligations = []
        microbatches = tuple(range(structure.num_microbatches))
        regions = {region.region_id: region for region in graph.regions}

        def require_microbatches(component, tasks):
            actual = tuple(sorted(task.microbatch_id for task in tasks if task.microbatch_id is not None))
            if actual != microbatches or len(tasks) != len(microbatches):
                obligations.append(ProgramCoverageConflict(component, tuple(map(str, microbatches)), tuple(map(str, actual))))

        for region in graph.regions:
            contract = region.training_contract
            ranks = {rank for value in contract.forward_outputs for rank in value.physical_ranks}
            for rank in ranks:
                tasks = [node for node in dag.nodes.values() if node.region_id == region.region_id and node.rank == rank]
                forwards = [node for node in tasks if node.node_type in ('REGION_FWD', 'FORWARD')]
                require_microbatches(region.region_id + ':forward:rank:' + str(rank), forwards)
                if any(value.value.gradient_eligible for value in contract.forward_inputs):
                    require_microbatches(region.region_id + ':dgrad:rank:' + str(rank),
                        [node for node in tasks if node.node_type in ('REGION_BWD', 'DGRAD')])
                for backward in (node for node in tasks if node.node_type in ('REGION_BWD', 'DGRAD', 'WGRAD')):
                    preceding = [node for node in forwards if node.microbatch_id == backward.microbatch_id]
                    if len(preceding) != 1 or not dag.reachable(preceding[0].node_id, backward.node_id):
                        obligations.append(ProgramCoverageConflict('forward_backward_readiness:' + backward.node_id,
                            tuple(node.node_id for node in preceding), backward.dependencies))
            for parameter in contract.parameter_contracts:
                for rank in parameter.parameter.storage_owners:
                    tasks = [node for node in dag.nodes.values() if parameter.parameter_key in node.parameter_keys
                        and node.region_id == region.region_id and node.rank == rank]
                    split = [node for node in tasks if node.node_type == 'WGRAD']
                    require_microbatches(parameter.parameter_key + ':wgrad:rank:' + str(rank), split if split else
                        [node for node in tasks if node.node_type == 'REGION_BWD'])
                for rank in parameter.gradient.storage_owners:
                    reducers = [node for node in dag.nodes.values() if parameter.parameter_key in node.parameter_keys
                        and node.rank == rank and node.node_type in ('GRADIENT_REDUCE', 'GRADIENT_READY')]
                    if not reducers:
                        obligations.append(UpdateIncomplete(parameter.parameter_key, 'gradient_readiness:rank:' + str(rank)))
                    for reducer in reducers:
                        valid = [group for group in parameter.gradient_reduce_groups if rank in group.ranks]
                        if reducer.node_type == 'GRADIENT_READY':
                            if not valid or any(len(group.ranks) != 1 for group in valid):
                                obligations.append(UpdateIncomplete(parameter.parameter_key, reducer.node_id + ':distributed_gradient_reduction'))
                        elif not any(group.key == reducer.process_group_key for group in valid):
                            obligations.append(UpdateIncomplete(parameter.parameter_key, reducer.node_id + ':gradient_process_group'))
                optimizer_ranks = {rank for state in parameter.optimizer_states for rank in state.storage_owners}
                for rank in optimizer_ranks:
                    updates = [node for node in dag.nodes.values() if parameter.parameter_key in node.parameter_keys
                        and node.rank == rank and node.node_type in ('OPTIMIZER', 'OPTIMIZER_STEP')]
                    if len(updates) != 1:
                        obligations.append(UpdateIncomplete(parameter.parameter_key, 'exactly_one_optimizer_update:rank:' + str(rank)))
        for node in dag.nodes.values():
            if node.region_id is not None and node.region_id not in regions:
                obligations.append(ProgramCoverageConflict('task_region:' + node.node_id, tuple(regions), (node.region_id,)))
            if node.node_type in ('WGRAD', 'REGION_BWD'):
                owner = regions.get(node.region_id)
                if owner is None or not set(node.parameter_keys) <= set(owner.parameter_keys):
                    obligations.append(ProgramCoverageConflict('task_parameter_owner:' + node.node_id,
                        () if owner is None else owner.parameter_keys, node.parameter_keys))
        return tuple(obligations)

    def check(self, dag, intervals, parameter_keys, persistent_bytes, memory_limits,
            scope='MODEL', process_groups=None, expected_communications=()):
        if not dag.nodes:
            raise ValueError('A complete update DAG must contain executable tasks')
        if len(set(parameter_keys)) != len(parameter_keys):
            raise ValueError('The update parameter inventory must contain distinct identities')
        if set(intervals) != set(dag.nodes):
            obligation = ProgramCoverageConflict('complete_update_schedule', tuple(sorted(dag.nodes)), tuple(sorted(intervals)))
            return VerificationReport('temporal', '', scope, (obligation,), ('complete_schedule_coverage',))
        for identity, (start, end) in intervals.items():
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError('Temporal verification requires finite complete task intervals')
        devices = {node.resource.device for node in dag.nodes.values()}
        if not devices <= set(persistent_bytes) or not devices <= set(memory_limits):
            raise ValueError('Every scheduled device requires its actual persistent allocation and memory limit')
        if any(persistent_bytes[device] < 0 or memory_limits[device] <= 0 for device in devices):
            raise ValueError('Persistent memory and device capacities require valid byte counts')
        obligations = []
        if scope == 'MODEL' and not parameter_keys:
            obligations.append(ProgramCoverageConflict('checkpoint_parameter_inventory', ('nonempty_model_parameters',), ()))
        resources, collectives, communication = defaultdict(list), defaultdict(list), defaultdict(list)
        for identity, node in dag.nodes.items():
            start, end = intervals[identity]
            for predecessor in node.dependencies:
                if intervals[predecessor][1] > start:
                    obligations.append(DependencyViolation(predecessor, identity, intervals[predecessor][1], start))
            for predecessor in node.finish_dependencies:
                if intervals[predecessor][1] > end:
                    obligations.append(EventOrderViolation(predecessor, identity, 'FINISH', 'FINISH',
                        intervals[predecessor][1], end))
            for predecessor in node.launch_dependencies:
                if intervals[predecessor][0] > end:
                    obligations.append(EventOrderViolation(predecessor, identity, 'START', 'FINISH',
                        intervals[predecessor][0], end))
            for resource in node.resource.resources:
                if resource != 'GPU_MEMORY':
                    resources[(node.resource.device, resource)].append((start, end, identity))
            if node.deadline_us is not None and end > node.deadline_us:
                obligations.append(DeadlineMiss(identity, node.deadline_us, end))
            if node.process_group_key is not None and node.node_type not in ('SEND', 'PP_SEND', 'RECV', 'PP_RECV'):
                if node.rank is None or node.collective_index is None:
                    raise ValueError('Collective tasks require actual rank and sequence identities')
                collectives[node.process_group_key].append(node)
            if node.communication_id is not None and node.node_type in ('SEND', 'PP_SEND', 'RECV', 'PP_RECV'):
                communication[node.communication_id].append(node)
            if node.node_type in ('SEND', 'PP_SEND', 'RECV', 'PP_RECV') and node.communication_id is None:
                obligations.append(ProgramCoverageConflict('p2p_identity:' + identity, ('communication_id',), ()))
            if node.node_type in ('SEND', 'PP_SEND', 'RECV', 'PP_RECV') and (node.rank is None
                    or process_groups is None or node.process_group_key not in process_groups
                    or node.rank not in process_groups[node.process_group_key]):
                obligations.append(ProgramCoverageConflict('p2p_group_membership:' + identity, ('actual_rank_and_process_group',), ()))
            unknown = set(node.parameter_keys) - set(parameter_keys)
            if unknown:
                obligations.append(ProgramCoverageConflict('update_parameter_inventory:' + identity,
                    tuple(sorted(parameter_keys)), tuple(sorted(unknown))))
        for (device, resource), tasks in resources.items():
            tasks.sort()
            for index, (start, end, identity) in enumerate(tasks):
                for next_start, next_end, next_identity in tasks[index + 1:]:
                    if next_start >= end:
                        break
                    if min(end, next_end) > max(start, next_start):
                        obligations.append(ResourceConflict(device, resource, identity, next_identity))
        for group, nodes in collectives.items():
            if process_groups is None or group not in process_groups:
                obligations.append(ProgramCoverageConflict('collective_group:' + group, (), tuple(sorted(node.node_id for node in nodes))))
                continue
            ranks = process_groups[group]
            if not ranks or len(set(ranks)) != len(ranks):
                raise ValueError('Collective verification requires distinct actual process group members')
            traces = {}
            for rank in ranks:
                actual = sorted((node for node in nodes if node.rank == rank), key=lambda node: (intervals[node.node_id][0], node.node_id))
                traces[rank] = tuple((node.collective_index, node.node_type, node.communication_id) for node in actual)
            if any(node.rank not in ranks for node in nodes):
                obligations.append(ProgramCoverageConflict('collective_members:' + group, tuple(map(str, ranks)), tuple(map(str, sorted({node.rank for node in nodes})))))
            reference = traces[ranks[0]]
            for rank in ranks[1:]:
                if traces[rank] != reference:
                    obligations.append(CollectiveOrderConflict(group, repr(reference), repr(traces[rank])))
            if tuple(item[0] for item in reference) != tuple(sorted({item[0] for item in reference})):
                obligations.append(CollectiveOrderConflict(group, 'unique_sequence_indices', repr(reference)))
        for identity in set(expected_communications) | set(communication):
            nodes = communication.get(identity, [])
            sends = [node for node in nodes if node.node_type in ('SEND', 'PP_SEND')]
            receives = [node for node in nodes if node.node_type in ('RECV', 'PP_RECV')]
            if len(sends) != 1 or len(receives) != 1 or not (dag.reachable(sends[0].node_id, receives[0].node_id)
                    or sends[0].node_id in receives[0].launch_dependencies):
                obligations.append(ProgramCoverageConflict('matched_p2p:' + identity, ('send', 'receive'), tuple(node.node_type for node in nodes)))
            elif sends[0].rank == receives[0].rank or sends[0].process_group_key != receives[0].process_group_key:
                obligations.append(ProgramCoverageConflict('p2p_endpoints:' + identity,
                    ('distinct_ranks_in_one_process_group',), tuple(str(node.rank) for node in nodes)))
        for parameter in parameter_keys:
            writers = [node for node in dag.nodes.values() if parameter in node.parameter_keys and node.node_type in ('WGRAD', 'REGION_BWD')]
            reductions = [node for node in dag.nodes.values() if parameter in node.parameter_keys and node.node_type in ('GRADIENT_REDUCE', 'GRADIENT_READY')]
            optimizers = [node for node in dag.nodes.values() if parameter in node.parameter_keys and node.node_type in ('OPTIMIZER', 'OPTIMIZER_STEP')]
            if not writers or not reductions or not optimizers:
                obligations.append(UpdateIncomplete(parameter, 'wgrad/gradient_reduce/optimizer_step'))
            for writer in writers:
                if not any((writer.rank is None or reducer.rank is None or writer.rank == reducer.rank)
                        and dag.reachable(writer.node_id, reducer.node_id) for reducer in reductions):
                    obligations.append(UpdateIncomplete(parameter, writer.node_id + ':gradient_reduce_readiness'))
            for reducer in reductions:
                local_writers = [writer for writer in writers if writer.rank is None or reducer.rank is None or writer.rank == reducer.rank]
                if not local_writers or not all(dag.reachable(writer.node_id, reducer.node_id) for writer in local_writers):
                    obligations.append(UpdateIncomplete(parameter, reducer.node_id + ':complete_wgrad_accumulation'))
                if not any((reducer.rank is None or optimizer.rank is None or reducer.rank == optimizer.rank)
                        and dag.reachable(reducer.node_id, optimizer.node_id) for optimizer in optimizers):
                    obligations.append(UpdateIncomplete(parameter, reducer.node_id + ':optimizer_completion'))
            for optimizer in optimizers:
                local_reductions = [reducer for reducer in reductions if reducer.rank is None or optimizer.rank is None or reducer.rank == optimizer.rank]
                if not local_reductions or not all(dag.reachable(reducer.node_id, optimizer.node_id) for reducer in local_reductions):
                    obligations.append(UpdateIncomplete(parameter, optimizer.node_id + ':optimizer_readiness'))
        timelines = defaultdict(list)
        for node in dag.nodes.values():
            start, end = intervals[node.node_id]
            if node.workspace_bytes and end > start:
                timelines[node.resource.device].extend(((start, node.workspace_bytes), (end, -node.workspace_bytes)))
            if node.alloc_bytes:
                if len(node.produces) != 1:
                    raise ValueError('Allocated storage requires one explicit logical storage identity')
                storage = node.produces[0]
                consumers = [consumer for consumer in dag.nodes.values() if storage in consumer.consumes and consumer.node_type != 'BUFFER_RELEASE']
                last_use = max((intervals[consumer.node_id][1] for consumer in consumers), default=end)
                releases = [consumer for consumer in dag.nodes.values() if storage in consumer.consumes and consumer.node_type == 'BUFFER_RELEASE']
                if len(releases) != 1:
                    obligations.append(TensorLifetimeConflict(storage, 'exactly_one_buffer_release', node.node_id))
                lifetime_end = max((intervals[release.node_id][1] for release in releases), default=max(end for _, end in intervals.values()))
                for release in releases:
                    for consumer in consumers:
                        if not dag.reachable(consumer.node_id, release.node_id) or intervals[release.node_id][0] < intervals[consumer.node_id][1]:
                            obligations.append(TensorLifetimeConflict(storage, release.node_id, consumer.node_id))
                lifetime_end = max(last_use, lifetime_end)
                if lifetime_end > start:
                    timelines[node.resource.device].extend(((start, node.alloc_bytes), (lifetime_end, -node.alloc_bytes)))
        for device in devices:
            current = persistent_bytes[device]
            deltas = defaultdict(int)
            for time, delta in timelines[device]:
                deltas[time] += delta
            times = sorted(set(deltas) | {0.0, max(end for _, end in intervals.values())})
            for index, start in enumerate(times[:-1]):
                current += deltas[start]
                if current > memory_limits[device] and times[index + 1] > start:
                    obligations.append(MemoryDeficit(device, (start, times[index + 1]), current - memory_limits[device]))
        descriptor = {'nodes': [asdict(dag.nodes[identity]) for identity in dag.order], 'intervals': intervals,
            'parameters': parameter_keys, 'persistent_bytes': persistent_bytes, 'memory_limits': memory_limits,
            'groups': process_groups, 'communications': expected_communications, 'scope': scope}
        identifier = hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=json_value, allow_nan=False).encode()).hexdigest()
        unique = {obligation.identifier(): obligation for obligation in obligations}
        return VerificationReport('temporal', identifier, scope, tuple(unique.values()),
            ('complete_schedule_coverage', 'dependencies', 'resource_conflicts', 'collective_order',
             'p2p_matching', 'wgrad_deadlines', 'gradient_reduce_and_optimizer_readiness', 'storage_last_use_and_memory'))
