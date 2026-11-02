import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from megatron.core.frontier.planner.obligations import (
    GradientOwnershipConflict, InterfaceBindingConflict, LayoutMismatch,
    ProcessGroupUnsatisfied, ProgramCoverageConflict, UnsupportedImplementation,
)
from megatron.core.frontier.planner.native_ownership import check_native_training_ownership


def json_value(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f'Unsupported verifier metadata type: {type(value).__name__}')


@dataclass(frozen=True)
class VerificationReport:
    phase: str
    program_id: str
    scope: str
    obligations: tuple
    checks: tuple[str, ...]

    @property
    def legal(self):
        return not self.obligations and self.scope in ('OPERATOR', 'MODEL')

    def machine_report(self):
        descriptor = {'phase': self.phase, 'program_id': self.program_id, 'scope': self.scope,
            'legal': self.legal, 'checks': self.checks,
            'conflicts': tuple({'id': conflict.identifier(), 'kind': type(conflict).__name__,
                'properties': asdict(conflict)} for conflict in self.obligations)}
        return json.loads(json.dumps(descriptor, sort_keys=True, default=json_value, allow_nan=False))


def gradient_interface(contract, forward_index, output):
    forward = contract.forward_outputs if output else contract.forward_inputs
    if not forward[forward_index].value.gradient_eligible:
        return None
    index = sum(value.value.gradient_eligible for value in forward[:forward_index])
    backward = contract.backward_gradient_inputs if output else contract.backward_gradient_outputs
    return backward[index]


def group_satisfies(actual, required):
    return (actual is not None and actual.key == required.key and actual.ranks == required.ranks
        and actual.backend == required.backend
        and set(required.collective_semantics) <= set(actual.collective_semantics))


class StructuralVerifier:
    def __init__(self, capability_catalog):
        self.capability_catalog = capability_catalog

    def check_native_ownership(self, records, world_size):
        program_id, obligations = check_native_training_ownership(records, world_size)
        return VerificationReport('native_training_ownership', program_id, 'MODEL', obligations,
            ('actual_rank_coverage', 'native_parameter_fragment_layout',
             'actual_gradient_buffer_layout', 'actual_reduction_group_membership',
             'complete_native_adam_state_inventory', 'complete_optimizer_interval_ownership'))

    def check(self, structure, graph, pipeline=None, repairs=()):
        regions = {region.region_id: region for region in graph.regions}
        mappings = {mapping.region_id: mapping for mapping in structure.region_mappings}
        obligations, checks = [], []
        if set(regions) != set(mappings):
            obligations.append(ProgramCoverageConflict('region_mappings', tuple(sorted(regions)), tuple(sorted(mappings))))
        stage_owners = {}
        if pipeline is None:
            if structure.pp_degree == 1:
                stage_owners = {identity: 0 for identity in regions}
            else:
                obligations.append(ProgramCoverageConflict('physical_pipeline', tuple(regions), ()))
        else:
            actual = tuple(identity for chunk in sorted(pipeline.chunks, key=lambda item: (item.vp_stage, item.pp_rank))
                for identity in chunk.region_ids)
            if actual != tuple(regions) or pipeline.pp_size != structure.pp_degree:
                obligations.append(ProgramCoverageConflict('physical_pipeline', tuple(regions), actual))
            stage_owners = {identity: chunk.pp_rank for chunk in pipeline.chunks for identity in chunk.region_ids}
        keys = tuple(key for region in graph.regions for key in region.parameter_keys)
        if graph.expected_parameter_keys is None:
            obligations.append(ProgramCoverageConflict('checkpoint_parameter_inventory', (), tuple(sorted(keys))))
        elif set(keys) != set(graph.expected_parameter_keys):
            obligations.append(ProgramCoverageConflict('checkpoint_parameters', tuple(sorted(graph.expected_parameter_keys)), tuple(sorted(keys))))
        checks.append('complete_region_and_checkpoint_parameter_coverage')

        group_definitions = {}
        for identity, mapping in mappings.items():
            for _, groups in mapping.process_groups:
                for group in groups:
                    previous = group_definitions.get(group.key)
                    if previous is not None and previous != group:
                        obligations.append(ProcessGroupUnsatisfied(identity, mapping, previous.ranks))
                    else:
                        group_definitions[group.key] = group
        checks.append('program_wide_process_group_identity')

        for identity, region in regions.items():
            if identity not in mappings:
                continue
            mapping = mappings[identity]
            if not self.capability_catalog.has_region_backend(mapping.implementation_key):
                obligations.append(UnsupportedImplementation(identity, mapping, 'No executable region backend is registered'))
            else:
                backend = self.capability_catalog.region_backend(mapping.implementation_key)
                if graph.scope not in backend.scopes or not backend.supports(mapping, region, graph.scope):
                    obligations.append(UnsupportedImplementation(identity, mapping, 'The actual backend rejects this mapping or verification scope'))
            contract = region.training_contract
            if contract is None or not contract.forward_inputs or not contract.forward_outputs:
                obligations.append(UnsupportedImplementation(identity, mapping, 'The region lacks complete distributed training contracts'))
                continue
            if contract.implementation_id != mapping.implementation_key:
                obligations.append(UnsupportedImplementation(identity, mapping,
                    'The training contract belongs to a different executable implementation'))
            if tuple(value.value for value in contract.forward_inputs) != region.input_contracts or tuple(
                    value.value for value in contract.forward_outputs) != region.output_contracts:
                obligations.append(ProgramCoverageConflict(identity + ':tensor_interfaces',
                    tuple(value.logical_value for value in region.input_contracts + region.output_contracts),
                    tuple(value.value.logical_value for value in contract.forward_inputs + contract.forward_outputs)))
            actual_keys = tuple(parameter.parameter_key for parameter in contract.parameter_contracts)
            if set(actual_keys) != set(region.parameter_keys):
                obligations.append(ProgramCoverageConflict(identity + ':parameter_ownership', tuple(sorted(region.parameter_keys)), tuple(sorted(actual_keys))))
            available = {group.key: group for _, groups in mapping.process_groups for group in groups}
            for group in contract.required_process_groups:
                actual = available.get(group.key)
                if not group_satisfies(actual, group):
                    obligations.append(ProcessGroupUnsatisfied(identity, mapping,
                        () if actual is None else actual.ranks))
            if identity in stage_owners and stage_owners[identity] < len(structure.physical_stage_ranks):
                stage_ranks = structure.physical_stage_ranks[stage_owners[identity]]
                values = (*contract.forward_inputs, *contract.forward_outputs,
                    *contract.backward_gradient_inputs, *contract.backward_gradient_outputs)
                if any(not set(value.physical_ranks) <= set(stage_ranks) for value in values):
                    obligations.append(ProcessGroupUnsatisfied(identity, mapping, stage_ranks))
                for parameter in contract.parameter_contracts:
                    states = (('parameter', parameter.parameter), ('gradient', parameter.gradient),
                        *(('optimizer', state) for state in parameter.optimizer_states))
                    for kind, state in states:
                        if not set(state.storage_owners) <= set(stage_ranks):
                            obligations.append(GradientOwnershipConflict(parameter.parameter_key,
                                stage_ranks, state.storage_owners, kind))
                    for group in (*parameter.gradient_reduce_groups, *parameter.replica_groups):
                        requested = available.get(group.key)
                        if not group_satisfies(requested, group):
                            obligations.append(GradientOwnershipConflict(parameter.parameter_key,
                                stage_ranks if requested is None else requested.ranks, group.ranks,
                                'gradient' if group in parameter.gradient_reduce_groups else 'parameter'))
        checks.extend(('backend_support', 'forward_backward_region_interfaces',
            'process_group_membership', 'parameter_gradient_optimizer_ownership'))

        bindings = {(identity, index): [] for identity, region in regions.items()
            for index in range(len(region.input_contracts))}
        for identity, index in graph.external_inputs:
            bindings[(identity, index)].append('external_input')
        consumed_repairs = set()
        for connection in graph.connections():
            source = regions[connection.producer].training_contract
            destination = regions[connection.consumer].training_contract
            bindings[(connection.consumer, connection.consumer_input)].append(connection.producer)
            if (source is None or destination is None or not source.forward_outputs or not destination.forward_inputs
                    or connection.producer not in mappings or connection.consumer not in mappings):
                continue
            left = source.forward_outputs[connection.producer_output]
            right = destination.forward_inputs[connection.consumer_input]
            backward_left = gradient_interface(destination, connection.consumer_input, False)
            backward_right = gradient_interface(source, connection.producer_output, True)
            matched = [index for index, repair in enumerate(repairs)
                if repair.producer == connection.producer and repair.consumer == connection.consumer
                and repair.producer_output == connection.producer_output
                and repair.consumer_input == connection.consumer_input
                and repair.forward_source == left and repair.forward_destination == right
                and repair.backward_source == backward_left and repair.backward_destination == backward_right]
            if len(matched) > 1:
                obligations.append(InterfaceBindingConflict(connection.consumer, connection.consumer_input,
                    tuple('repair:' + str(index) for index in matched)))
            if len(matched) == 1:
                consumed_repairs.add(matched[0])
                for group in repairs[matched[0]].groups:
                    available = {candidate.key: candidate for mapping in (mappings[connection.producer], mappings[connection.consumer])
                        for _, groups in mapping.process_groups for candidate in groups}
                    if not group_satisfies(available.get(group.key), group):
                        obligations.append(ProcessGroupUnsatisfied(connection.producer, mappings[connection.producer], ()))
            else:
                if not left.compatible_with(right):
                    obligations.append(LayoutMismatch(connection.producer, connection.consumer, left, right, 'forward'))
                if (backward_left is None) != (backward_right is None):
                    obligations.append(ProgramCoverageConflict('backward_gradient_edge',
                        (connection.producer,) if backward_right is not None else (),
                        (connection.consumer,) if backward_left is not None else ()))
                elif backward_left is not None and not backward_left.compatible_with(backward_right):
                    obligations.append(LayoutMismatch(connection.consumer, connection.producer,
                        backward_left, backward_right, 'backward'))
        for (identity, index), providers in bindings.items():
            if len(providers) != 1:
                obligations.append(InterfaceBindingConflict(identity, index, tuple(providers)))
        if consumed_repairs != set(range(len(repairs))):
            obligations.append(ProgramCoverageConflict('bound_repairs', tuple(map(str, range(len(repairs)))), tuple(map(str, sorted(consumed_repairs)))))
        checks.extend(('forward_and_backward_edge_closure', 'communication_repair_matching', 'complete_input_bindings'))
        descriptor = {'structure': asdict(structure), 'graph': asdict(graph), 'repairs': [asdict(repair) for repair in repairs],
            'backend_revisions': {mapping.implementation_key: self.capability_catalog.region_backend(mapping.implementation_key).source_revision
                for mapping in mappings.values() if self.capability_catalog.has_region_backend(mapping.implementation_key)}}
        program_id = hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=json_value,
            allow_nan=False).encode()).hexdigest()
        unique = {obligation.identifier(): obligation for obligation in obligations}
        return VerificationReport('structural', program_id, graph.scope, tuple(unique.values()), tuple(checks))
