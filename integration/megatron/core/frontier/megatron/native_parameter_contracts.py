from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

from megatron.core.frontier.ir.contract import (
    DistributedTensorContract, ParameterContract, Placement, PlacementKind,
    ProcessGroupContract, ValueContract,
)
from megatron.core.frontier.planner.native_ownership import check_native_training_ownership, parameter_identity


@dataclass(frozen=True)
class NativeParameterDomain:
    identity: tuple
    contract: ParameterContract
    checkpoint_fragments: tuple
    optimizer_devices: tuple


def compile_native_parameter_domains(records, world_size):
    _, conflicts = check_native_training_ownership(records, world_size)
    if conflicts:
        raise ValueError('Native parameter contract compilation requires complete actual ownership without conflicts')
    declarations, owners = {}, defaultdict(list)
    for record in records:
        for parameter in record['parameters']:
            declarations.setdefault(parameter_identity(parameter), parameter)
        for owner in record['owners']:
            owners[parameter_identity(owner)].append((record['rank'], owner))
    domains = []
    for identity, parameter in sorted(declarations.items()):
        key, ranks, fragments = identity
        domain_id = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        group = ProcessGroupContract('native-reduction:' + domain_id, ranks, 'nccl', ('ReduceScatter', 'AllGather'))
        mesh = (('native_dp', len(ranks)),)
        shape = tuple(parameter['parameter_shape'])
        value = ValueContract(key, shape, Placement(PlacementKind.REPLICATE), group.key,
            dtype=parameter['parameter_dtype'].removeprefix('torch.'))
        weight = DistributedTensorContract(value, mesh, ranks, (), ('native_dp',),
            'native-local-parameter', ranks, domain_id + ':parameter')
        rows = sorted(owners[identity])
        gradient_dtypes = {owner['gradient_dtype'] for _, owner in rows}
        if len(gradient_dtypes) != 1:
            raise ValueError('Native parameter domain replicas require one actual gradient precision')
        gradient = DistributedTensorContract(
            ValueContract(key + ':gradient', shape, value.placement, group.key,
                dtype=gradient_dtypes.pop().removeprefix('torch.')),
            mesh, ranks, (), ('native_dp',), 'native-main-grad', ranks, domain_id + ':gradient')
        intervals = tuple((rank, *owner['optimizer_flattened_range']) for rank, owner in rows)
        state_owners = tuple(rank for rank, _ in rows)
        states = tuple(DistributedTensorContract(
            ValueContract(name, shape, value.placement, group.key, dtype='float32'),
            mesh, ranks, (), ('native_dp',), 'native-flat-optimizer', state_owners,
            domain_id + ':' + name, storage_intervals=intervals)
            for name in ('param', 'exp_avg', 'exp_avg_sq'))
        contract = ParameterContract(key, weight, gradient, states, (group,), (group,))
        devices = tuple((rank, tuple((name, state['device']) for name, state in sorted(owner['states'].items())))
            for rank, owner in rows)
        domains.append(NativeParameterDomain(identity, contract, fragments, devices))
    return tuple(domains)


def bind_native_parameter_domains(records, domains):
    by_identity = {domain.identity: domain for domain in domains}
    bindings = []
    covered = set()
    for record in records:
        rank = record['rank']
        if not record.get('model_bindings'):
            raise ValueError('Actual native model parameter names are required to bind training contracts')
        candidates = defaultdict(list)
        for parameter in record['parameters']:
            identity = parameter_identity(parameter)
            if identity not in by_identity:
                raise ValueError('The actual native model contains an uncompiled parameter domain')
            candidates[(parameter['parameter_key'], tuple(parameter['parameter_shape']), identity[2])].append(identity)
        local_names = set()
        for binding in record['model_bindings']:
            name = (rank, binding['chunk'], binding['native_name'])
            if name in local_names:
                raise ValueError('The native model parameter binding is declared multiple times')
            local_names.add(name)
            if not binding.get('checkpoint_fragments'):
                raise ValueError('Native expert bindings require their actual checkpoint fragments and global offsets')
            fragments = tuple(sorted((fragment['key'], tuple(fragment['global_shape']),
                tuple(fragment['global_offset']), tuple(fragment['local_shape']), fragment['prepend_axis_num'])
                for fragment in binding['checkpoint_fragments']))
            matched = candidates[(binding['parameter_key'], tuple(binding['parameter_shape']), fragments)]
            if len(matched) != 1:
                raise ValueError('The native model parameter name lacks an unambiguous actual checkpoint domain: '
                    + str({'rank': rank, 'binding': binding, 'candidate_count': len(matched)}))
            identity, = matched
            domain = by_identity[identity]
            covered.add((rank, identity))
            bindings.append({'rank': rank, 'chunk': binding['chunk'], 'native_name': binding['native_name'],
                'parameter_key': binding['parameter_key'], 'storage_id': domain.contract.parameter.storage_id})
    expected = {(record['rank'], parameter_identity(parameter))
        for record in records for parameter in record['parameters']}
    if covered != expected:
        raise ValueError('Actual native model parameter names do not cover every compiled training contract')
    return tuple(bindings)
