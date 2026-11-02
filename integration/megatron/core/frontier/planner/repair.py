from dataclasses import dataclass, replace
from math import prod

import torch.distributed as dist

from megatron.core.frontier.ir.contract import DistributedTensorContract, ParameterContract, ProcessGroupContract
from megatron.core.frontier.lowering.base import LoweringContext
from megatron.core.frontier.lowering.standard import StandardNCCLFunction, StandardNCCLLowering
from megatron.core.frontier.runtime.boundary import BoundarySchema
from megatron.core.frontier.state import (
    DistributedState, ObligationKind, Placement, PlacementKind, TransitionObligation,
)
from megatron.core.frontier.planner.verifier import StructuralVerifier, gradient_interface
from megatron.core.frontier.planner.obligations import ProgramCoverageConflict, UnsupportedImplementation


def local_shape(contract):
    if contract.storage_intervals:
        raise ValueError('A flattened optimizer storage interval cannot be used as an activation collective payload')
    axes = dict(contract.logical_mesh)
    shape = list(contract.value.logical_shape)
    if any(not isinstance(dimension, int) for dimension in shape):
        raise ValueError('This collective repair requires a concrete compiled local shape')
    for axis, dimension in contract.sharding:
        shape[dimension] //= axes[axis]
    return tuple(shape)


def axis_group_matches(contract, axis, group):
    axes = dict(contract.logical_mesh)
    if len(group.ranks) != axes[axis] or not set(group.ranks) <= set(contract.physical_ranks):
        return False
    coordinates = {rank: index for index, rank in enumerate(contract.physical_ranks)}
    index = [name for name, _ in contract.logical_mesh].index(axis)
    stride = prod(size for _, size in contract.logical_mesh[index + 1:])
    offsets = [coordinates[rank] for rank in group.ranks]
    base = offsets[0] - ((offsets[0] // stride) % axes[axis]) * stride
    return offsets == [base + position * stride for position in range(axes[axis])]


@dataclass(frozen=True)
class CollectiveRepair:
    producer: str
    consumer: str
    forward_source: DistributedTensorContract
    forward_destination: DistributedTensorContract
    backward_source: DistributedTensorContract
    backward_destination: DistributedTensorContract
    axis: str
    primitive: str
    groups: tuple[ProcessGroupContract, ...]
    producer_output: int = 0
    consumer_input: int = 0

    def __post_init__(self):
        if self.producer_output < 0 or self.consumer_input < 0:
            raise ValueError('Communication repair requires concrete nonnegative tensor interface indices')
        if self.primitive not in ('AllReduce', 'ReduceScatter', 'AllGather'):
            raise ValueError('The requested native collective primitive is unregistered')
        contracts = (self.forward_source, self.forward_destination, self.backward_source, self.backward_destination)
        if any(contract.storage_intervals for contract in contracts):
            raise ValueError('Activation collective repair cannot execute flattened optimizer storage intervals')
        first = self.forward_source
        if any(contract.logical_mesh != first.logical_mesh or contract.physical_ranks != first.physical_ranks
                or contract.value.logical_shape != first.value.logical_shape for contract in contracts):
            raise ValueError('Collective repair requires one complete same-mesh forward and backward domain')
        if first.value.dtype != self.forward_destination.value.dtype or self.backward_source.value.dtype != self.backward_destination.value.dtype:
            raise ValueError('Collective repair cannot silently change forward or backward precision')
        if any(contract.storage_owners != first.storage_owners or contract.value.logical_slice != first.value.logical_slice
                or contract.value.presence_policy != first.value.presence_policy
                or not contract.value.gradient_eligible for contract in contracts):
            raise ValueError('Native collective repair cannot change storage ownership or payload projection semantics')
        axes = dict(first.logical_mesh)
        if self.axis not in axes:
            raise ValueError('Collective repair requires an actual tensor mesh axis')
        if self.primitive == 'AllGather':
            if first.sharding != ((self.axis, 0),) or first.partial:
                raise ValueError('AllGather repair requires a dimension-zero shard axis')
        elif dict(first.partial).get(self.axis) != 'sum':
            raise ValueError('Reduction repair requires an actual sum-partial axis')
        destination = self.forward_destination
        if self.primitive in ('AllReduce', 'AllGather'):
            expected_sharding = first.sharding
            if self.primitive == 'AllGather':
                expected_sharding = tuple(item for item in first.sharding if item[0] != self.axis)
            expected_replication = set(first.replication) | {self.axis}
        else:
            expected_sharding = (*first.sharding, (self.axis, 0))
            expected_replication = set(first.replication)
        if (set(destination.sharding) != set(expected_sharding)
                or set(destination.replication) != expected_replication
                or dict(destination.partial) != {axis: operation for axis, operation in first.partial if axis != self.axis}
                or first.layout != destination.layout):
            raise ValueError('Reduction repair changes an unrelated layout property')
        gradient_input, gradient_output = self.backward_source, self.backward_destination
        output_replication = set(first.replication) if self.primitive == 'AllGather' else set(first.replication) | {self.axis}
        if (set(gradient_output.sharding) != set(first.sharding)
                or set(gradient_output.replication) != output_replication
                or gradient_output.partial != destination.partial
                or set(gradient_input.sharding) != set(destination.sharding)
                or set(gradient_input.replication) != set(destination.replication)
                or gradient_input.partial != destination.partial
                or gradient_input.layout != first.layout or gradient_output.layout != first.layout):
            raise ValueError('Reduction repair has an incompatible declared backward adjoint')
        members = [rank for group in self.groups for rank in group.ranks]
        if (set(members) != set(first.physical_ranks) or len(members) != len(first.physical_ranks)
                or any(group.backend != 'nccl' or len(group.ranks) != axes[self.axis]
                    or self.primitive not in group.collective_semantics
                    or self.adjoint not in group.collective_semantics for group in self.groups)):
            raise ValueError('Collective repair requires complete native forward and adjoint process groups')
        coordinates = {rank: index for index, rank in enumerate(first.physical_ranks)}
        names = [name for name, _ in first.logical_mesh]
        axis_index = names.index(self.axis)
        stride = prod(size for _, size in first.logical_mesh[axis_index + 1:])
        for group in self.groups:
            offsets = [coordinates[rank] for rank in group.ranks]
            base = offsets[0] - ((offsets[0] // stride) % axes[self.axis]) * stride
            if offsets != [base + index * stride for index in range(axes[self.axis])]:
                raise ValueError('Collective process group ordering differs from the tensor mesh axis')

    @property
    def adjoint(self):
        return {'AllReduce': 'AllReduce', 'ReduceScatter': 'AllGather', 'AllGather': 'ReduceScatter'}[self.primitive]

    def bind(self, rank, process_groups):
        group, = (group for group in self.groups if rank in group.ranks)
        handle = process_groups[group.key]
        if (dist.get_backend(handle) != group.backend
                or tuple(dist.get_process_group_ranks(handle)) != group.ranks):
            raise ValueError('Actual collective membership differs from the repaired program')
        shape = local_shape(self.forward_source)
        target = Placement(PlacementKind.SHARD, 0) if self.primitive == 'ReduceScatter' else Placement(PlacementKind.REPLICATE)
        source_placement = Placement(PlacementKind.SHARD, 0) if self.primitive == 'AllGather' else Placement(PlacementKind.PARTIAL)
        obligation = ObligationKind.REPARTITION if self.primitive == 'AllGather' else ObligationKind.REDUCE
        source = DistributedState(source_placement, group.key, shape,
            TransitionObligation(obligation, target, group.key), self.forward_source.value.dtype)
        schema = BoundarySchema(0, source, source.resolved(target), 'standard_nccl', shape,
            group.ranks, group.ranks, pg_key=group.key)
        return BoundCollectiveRepair(self, LoweringContext(schema, 0, 0, process_groups=process_groups))


@dataclass(frozen=True)
class BoundCollectiveRepair:
    repair: CollectiveRepair
    context: LoweringContext

    def forward(self, tensor):
        if str(tensor.dtype).removeprefix('torch.') != self.repair.forward_source.value.dtype:
            raise ValueError('Actual forward precision differs from the repaired tensor contract')
        return StandardNCCLFunction.apply(tensor, self.context)

    def backward(self, gradient):
        expected = self.repair.backward_source
        if tuple(gradient.shape) != local_shape(expected) or str(gradient.dtype).removeprefix('torch.') != expected.value.dtype:
            raise ValueError('Actual backward input differs from the repaired gradient contract')
        return StandardNCCLLowering().backward(gradient, self.context)


class MandatoryRepair:
    def close(self, structure, graph, capability_catalog, pipeline=None, context=None, state_backend=None, source_graph=None,
            initialize_from_backend=False):
        verifier = StructuralVerifier(capability_catalog)
        before = verifier.check(structure, graph, pipeline)
        mappings = {mapping.region_id: mapping for mapping in structure.region_mappings}
        rebuilt, actions, reasons, state_migrations = [], [], [], []
        for region in graph.regions:
            mapping = mappings.get(region.region_id)
            if mapping is None or not capability_catalog.has_region_backend(mapping.implementation_key):
                rebuilt.append(region)
                reasons.append(region.region_id + ': no registered region compiler')
                continue
            backend = capability_catalog.region_backend(mapping.implementation_key)
            if graph.scope not in backend.scopes or not backend.supports(mapping, region, graph.scope):
                rebuilt.append(region)
                reasons.append(region.region_id + ': requested mapping is unsupported by the actual backend')
                continue
            contract = backend.compile_contract(mapping, region, context)
            if contract is None:
                raise ValueError('A registered supported backend returned no training contract')
            rebuilt.append(replace(region, training_contract=contract,
                input_contracts=tuple(value.value for value in contract.forward_inputs),
                output_contracts=tuple(value.value for value in contract.forward_outputs)))
            actions.append(region.region_id + ': rebuild forward, backward, parameter, gradient and optimizer contracts')
        repaired_graph = replace(graph, regions=tuple(rebuilt))
        original = graph if source_graph is None else source_graph
        source_conflicts = parameter_ownership_conflicts(original)
        destination_conflicts = parameter_ownership_conflicts(repaired_graph)
        old_inventory = {} if source_conflicts else parameter_inventory(original)
        new_inventory = {} if destination_conflicts else parameter_inventory(repaired_graph)
        declared_source = set(original.expected_parameter_keys if original.expected_parameter_keys is not None else
            (key for region in original.regions for key in region.parameter_keys))
        initial = initialize_from_backend and source_graph is None and all(region.training_contract is None for region in graph.regions)
        if initialize_from_backend and not initial:
            raise ValueError('Initial backend registration requires a wholly uninitialized graph without an original training program')
        missing_source = not initial and set(old_inventory) != declared_source
        if source_conflicts or destination_conflicts or missing_source:
            reasons.append('The requested repair lacks a complete unambiguous original training state inventory')
        if not initial and not source_conflicts and not destination_conflicts:
            if set(old_inventory) != set(new_inventory):
                reasons.append('The source and destination training programs have different complete parameter inventories')
            else:
                state_migrations.extend(training_state_transitions(original, repaired_graph))
        regions = {region.region_id: region for region in repaired_graph.regions}
        repairs = []
        for connection in repaired_graph.connections():
            if connection.producer not in mappings or connection.consumer not in mappings:
                continue
            producer = regions[connection.producer].training_contract
            consumer = regions[connection.consumer].training_contract
            if producer is None or consumer is None:
                continue
            left = producer.forward_outputs[connection.producer_output]
            right = consumer.forward_inputs[connection.consumer_input]
            backward_left = gradient_interface(consumer, connection.consumer_input, False)
            backward_right = gradient_interface(producer, connection.producer_output, True)
            if left.compatible_with(right):
                if ((backward_left is None) != (backward_right is None)
                        or backward_left is not None and not backward_left.compatible_with(backward_right)):
                    reasons.append(connection.producer + ' -> ' + connection.consumer
                        + ': no registered backward-only redistribution for the requested adjoint')
                continue
            gather_axis = self.gather_axis(left, right, backward_left, backward_right)
            if gather_axis is None and not self.supports_reduction(left, right, backward_left, backward_right):
                reasons.append(connection.producer + ' -> ' + connection.consumer + ': no registered bidirectional redistribution')
                continue
            if gather_axis is not None:
                axis = gather_axis
            else:
                axis, = set(dict(left.partial)) - set(dict(right.partial))
            candidates = {group.key: group for group in dict(mappings[connection.producer].process_groups).get(axis, ())}
            groups = tuple(group for group in candidates.values()
                if axis_group_matches(left, axis, group)
                and group.key in {item.key for item in (*producer.required_process_groups, *consumer.required_process_groups)})
            members = [rank for group in groups for rank in group.ranks]
            if set(members) != set(left.physical_ranks) or len(members) != len(left.physical_ranks):
                reasons.append(connection.producer + ' -> ' + connection.consumer + ': reduction axis lacks complete native groups')
                continue
            repair = CollectiveRepair(connection.producer, connection.consumer, left, right,
                backward_left, backward_right, axis, 'AllGather', groups) if gather_axis is not None else self.reduction(
                connection.producer, connection.consumer, left, right, backward_left, backward_right, groups)
            repairs.append(replace(repair, producer_output=connection.producer_output,
                consumer_input=connection.consumer_input))
            actions.append(connection.producer + ' -> ' + connection.consumer + ': ' + repair.primitive + '/' + repair.adjoint)
        after = verifier.check(structure, repaired_graph, pipeline, tuple(repairs))
        after = replace(after, obligations=(*after.obligations, *source_conflicts, *destination_conflicts))
        if missing_source:
            after = replace(after, obligations=(*after.obligations, ProgramCoverageConflict(
                'original_training_state_inventory', tuple(sorted(declared_source)), tuple(sorted(old_inventory)))))
        if not initial and set(old_inventory) != set(new_inventory):
            after = replace(after, obligations=(*after.obligations, ProgramCoverageConflict(
                'preserved_training_state_inventory', tuple(sorted(old_inventory)), tuple(sorted(new_inventory)))))
        pending = ()
        if state_migrations:
            supported = state_backend is not None and state_backend.supports(tuple(state_migrations))
            if not supported:
                reasons.append('No actual state migration backend supports the complete requested training state transition')
            pending = tuple(UnsupportedImplementation(
                migration.region, mappings[migration.region], migration.destination.parameter_key
                + ': training_state_migration must complete before program execution')
                for migration in state_migrations)
            after = replace(after, obligations=(*after.obligations, *pending))
        other_unsupported = any(isinstance(item, UnsupportedImplementation) and item not in pending for item in after.obligations)
        status = 'UNSUPPORTED' if reasons or other_unsupported else 'READY_STATE_MIGRATION' if pending and len(
            pending) == len(after.obligations) else 'LEGAL_STRUCTURAL' if after.legal else 'ILLEGAL'
        return RepairResult(status, repaired_graph, tuple(repairs), before, after, tuple(actions), tuple(reasons),
            tuple(state_migrations), state_backend.key if state_backend is not None else None, pending)

    def execute_state_migrations(self, result, state_backend, source_bindings, destination_bindings,
            checkpoint_directory, global_group, process_groups, update_boundary):
        if result.status != 'READY_STATE_MIGRATION' or result.state_backend_key != state_backend.key:
            raise ValueError('State migration execution requires a structurally closed supported repair plan')
        receipt = state_backend.migrate(result.state_migrations, source_bindings, destination_bindings,
            checkpoint_directory, global_group, process_groups, update_boundary)
        if receipt.plan_id != state_backend.plan_id(result.state_migrations):
            raise ValueError('The state migration receipt does not belong to this repaired program')
        after = replace(result.after, obligations=tuple(item for item in result.after.obligations
            if item not in result.state_migration_obligations))
        if not after.legal:
            raise ValueError('Completed state migration left unresolved structural obligations')
        return replace(result, status='LEGAL_STRUCTURAL', after=after, migration_receipt=receipt)

    def gather_axis(self, source, destination, backward_source, backward_destination):
        if backward_source is None or backward_destination is None:
            return None
        removed = set(dict(source.sharding)) - set(dict(destination.sharding))
        if len(removed) != 1:
            return None
        axis, = removed
        if source.sharding != ((axis, 0),) or source.partial or any(not isinstance(size, int) for size in source.value.logical_shape):
            return None
        expected = replace(source, value=replace(source.value, placement=destination.value.placement),
            sharding=tuple(item for item in source.sharding if item[0] != axis),
            replication=(*source.replication, axis))
        if (not expected.compatible_with(destination) or not destination.compatible_with(backward_source)
                or not source.compatible_with(backward_destination)):
            return None
        return axis

    def supports_reduction(self, source, destination, backward_source, backward_destination):
        if backward_source is None or backward_destination is None:
            return False
        if any(contract.storage_owners != source.storage_owners
                or contract.value.logical_slice != source.value.logical_slice
                or contract.value.presence_policy != source.value.presence_policy
                or not contract.value.gradient_eligible for contract in (source, destination, backward_source, backward_destination)):
            return False
        if any(contract.logical_mesh != source.logical_mesh or contract.physical_ranks != source.physical_ranks
                or contract.value.logical_shape != source.value.logical_shape for contract in (
                    destination, backward_source, backward_destination)):
            return False
        removed = set(dict(source.partial)) - set(dict(destination.partial))
        if len(removed) != 1 or any(not isinstance(size, int) for size in source.value.logical_shape):
            return False
        axis, = removed
        if dict(source.partial)[axis] != 'sum' or source.layout != destination.layout:
            return False
        if axis in destination.replication:
            expected_sharding = source.sharding
            expected_replication = set(source.replication) | {axis}
        elif (axis, 0) in destination.sharding:
            expected_sharding = (*source.sharding, (axis, 0))
            expected_replication = set(source.replication)
        else:
            return False
        partial = tuple(item for item in source.partial if item[0] != axis)
        return (source.value.dtype == destination.value.dtype
            and backward_source.value.dtype == backward_destination.value.dtype
            and set(destination.sharding) == set(expected_sharding)
            and set(destination.replication) == expected_replication and destination.partial == partial
            and set(backward_source.sharding) == set(destination.sharding)
            and set(backward_source.replication) == set(destination.replication)
            and backward_source.partial == partial and backward_source.layout == source.layout
            and set(backward_destination.sharding) == set(source.sharding)
            and set(backward_destination.replication) == set(source.replication) | {axis}
            and backward_destination.partial == partial and backward_destination.layout == source.layout)

    def reduction(self, producer, consumer, source, destination, backward_source, backward_destination, groups):
        removed = set(dict(source.partial)) - set(dict(destination.partial))
        if len(removed) != 1:
            raise ValueError('Each registered reduction repair closes one explicit partial axis')
        axis, = removed
        if axis in destination.replication:
            primitive = 'AllReduce'
        elif (axis, 0) in destination.sharding:
            primitive = 'ReduceScatter'
        else:
            raise ValueError('No validated native reduction implementation matches the requested layout')
        return CollectiveRepair(producer, consumer, source, destination, backward_source,
            backward_destination, axis, primitive, groups)


@dataclass(frozen=True)
class TrainingStateMigration:
    region: str
    source: ParameterContract
    destination: ParameterContract
    source_region: str | None = None


def parameter_inventory(graph):
    inventory = {}
    for region in graph.regions:
        if region.training_contract is None:
            continue
        for parameter in region.training_contract.parameter_contracts:
            if parameter.parameter_key in inventory:
                raise ValueError('The source training state has multiple owners for one logical parameter')
            inventory[parameter.parameter_key] = (region.region_id, parameter)
    return inventory


def parameter_ownership_conflicts(graph):
    owners = {}
    for region in graph.regions:
        if region.training_contract is not None:
            for parameter in region.training_contract.parameter_contracts:
                owners.setdefault(parameter.parameter_key, []).append(region.region_id)
    return tuple(ProgramCoverageConflict('unique_training_parameter_owner:' + key,
        ('exactly_one_region',), tuple(regions)) for key, regions in sorted(owners.items()) if len(regions) != 1)


def training_state_transitions(source_graph, destination_graph):
    source = parameter_inventory(source_graph)
    destination = parameter_inventory(destination_graph)
    if set(source) != set(destination):
        raise ValueError('State transitions must preserve the complete logical parameter inventory')
    return tuple(TrainingStateMigration(destination[key][0], source[key][1], destination[key][1], source[key][0])
        for key in sorted(source) if source[key][0] != destination[key][0]
        or training_state_changed(source[key][1], destination[key][1]))


def training_state_changed(source, destination):
    source_states = {state.value.logical_value: state for state in source.optimizer_states}
    destination_states = {state.value.logical_value: state for state in destination.optimizer_states}
    return (not source.parameter.compatible_with(destination.parameter)
        or not source.gradient.compatible_with(destination.gradient)
        or source.gradient_reduce_groups != destination.gradient_reduce_groups
        or source.replica_groups != destination.replica_groups
        or source_states.keys() != destination_states.keys()
        or any(not state.compatible_with(destination_states[name]) for name, state in source_states.items()
            if name in destination_states))


@dataclass(frozen=True)
class RepairResult:
    status: str
    graph: object
    repairs: tuple[CollectiveRepair, ...]
    before: object
    after: object
    actions: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]
    state_migrations: tuple[TrainingStateMigration, ...] = ()
    state_backend_key: str | None = None
    state_migration_obligations: tuple[UnsupportedImplementation, ...] = ()
    migration_receipt: object | None = None

    def bind_collectives(self, rank, process_groups):
        if self.status != 'LEGAL_STRUCTURAL' or not self.after.legal:
            raise ValueError('Training execution requires a fully closed structural repair result')
        if self.state_migrations and self.migration_receipt is None:
            raise ValueError('Training execution requires completed actual state migration')
        bound = {}
        for repair in self.repairs:
            connection = (repair.producer, repair.producer_output, repair.consumer, repair.consumer_input)
            if connection in bound:
                raise ValueError('A training tensor connection has multiple communication repair implementations')
            if rank in repair.forward_source.physical_ranks:
                bound[connection] = repair.bind(rank, process_groups)
        return bound
