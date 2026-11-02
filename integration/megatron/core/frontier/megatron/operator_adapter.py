from dataclasses import replace

import torch
import torch.distributed as dist

from megatron.core.frontier.capability.catalog import RegionBackendImplementation
from megatron.core.frontier.contracts import ParallelPolicy, RegionContract, RegionKind, StatePattern
from megatron.core.frontier.ir.contract import DistributedTensorContract, ParameterContract, Placement, PlacementKind, ValueContract
from megatron.core.frontier.state import PlacementKind as PatternPlacementKind


class NativeLinearContractBackend:
    def __init__(self, key, parameter_key, module, optimizer, gradient, input_shape, group, source_revision):
        if not isinstance(module, torch.nn.Linear) or module.bias is not None:
            raise ValueError('This executable backend requires an actual bias-free torch Linear module')
        if module.weight not in optimizer.state or gradient.shape != module.weight.shape:
            raise ValueError('Linear contracts require actual initialized optimizer state and gradient ownership')
        self.key, self.parameter_key, self.module = key, parameter_key, module
        self.optimizer, self.gradient, self.input_shape = optimizer, gradient, input_shape
        self.group, self.source_revision = group, source_revision
        if tuple(dist.get_process_group_ranks(dist.group.WORLD)) != group.ranks:
            raise ValueError('Operator verification requires the explicitly observed world process group')

    def supports(self, mapping, region, scope):
        dense = mapping.policy.dense
        return (scope == 'OPERATOR' and dense.tp == len(self.group.ranks) and dense.cp == dense.pp == dense.dp == 1
            and mapping.policy.expert is None and mapping.policy.state_sharding == 'replicated'
            and mapping.implementation_key == self.key and region.parameter_keys == (self.parameter_key,))

    def tensor(self, name, shape, dtype, partial=False):
        ranks = self.group.ranks
        value = ValueContract(name, tuple(shape), Placement(PlacementKind.PARTIAL if partial else PlacementKind.REPLICATE),
            self.group.key, dtype=str(dtype).removeprefix('torch.'))
        return DistributedTensorContract(value, (('tp', len(ranks)),), ranks, (),
            () if partial else ('tp',), 'contiguous', ranks, name,
            partial=(('tp', 'sum'),) if partial else ())

    def compile_contract(self, mapping, region, context):
        if not self.supports(mapping, region, 'OPERATOR'):
            raise ValueError('The actual Linear backend cannot compile the requested parallel mapping')
        parameter = self.tensor(self.parameter_key, self.module.weight.shape, self.module.weight.dtype)
        gradient = self.tensor(self.parameter_key + ':gradient', self.gradient.shape, self.gradient.dtype)
        states = tuple(self.tensor(self.parameter_key + ':' + name, tensor.shape, tensor.dtype)
            for name, tensor in self.optimizer.state[self.module.weight].items())
        ownership = ParameterContract(self.parameter_key, parameter, gradient, states, (self.group,), (self.group,))
        inputs = (replace(self.tensor('hidden', self.input_shape, self.module.weight.dtype, partial=True), storage_id='linear.input'),)
        outputs = (replace(self.tensor('hidden', (*self.input_shape[:-1], self.module.out_features),
            self.module.weight.dtype, partial=True), storage_id='linear.output'),)
        backward_inputs = (replace(outputs[0], value=replace(outputs[0].value,
            placement=Placement(PlacementKind.REPLICATE), logical_value='linear.output_gradient'),
            partial=(), replication=('tp',), storage_id='linear.output_gradient'),)
        backward_outputs = (replace(inputs[0], value=replace(inputs[0].value,
            placement=Placement(PlacementKind.REPLICATE), logical_value='linear.input_gradient'),
            partial=(), replication=('tp',), storage_id='linear.input_gradient'),)
        return RegionContract(region.region_id, RegionKind.MLP,
            StatePattern(PatternPlacementKind.PARTIAL, allow_pending=True), StatePattern(PatternPlacementKind.PARTIAL, allow_pending=True),
            ParallelPolicy(tp=len(self.group.ranks)), self.key, self.key + ':profile',
            forward_inputs=inputs, forward_outputs=outputs, backward_gradient_inputs=backward_inputs,
            backward_gradient_outputs=backward_outputs, parameter_contracts=(ownership,), required_process_groups=(self.group,))

    def registration(self):
        return RegionBackendImplementation(self.key, self.source_revision, ('OPERATOR',), self.supports, self.compile_contract)


class NativeIdentityContractBackend:
    def __init__(self, key, shape, dtype, group, source_revision):
        self.key, self.shape, self.dtype = key, shape, dtype
        self.group, self.source_revision = group, source_revision
        self.module = torch.nn.Identity()

    def supports(self, mapping, region, scope):
        dense = mapping.policy.dense
        return (scope == 'OPERATOR' and dense.tp == len(self.group.ranks) and dense.cp == dense.pp == dense.dp == 1
            and mapping.policy.expert is None and mapping.policy.state_sharding == 'replicated'
            and mapping.implementation_key == self.key and not region.parameter_keys)

    def compile_contract(self, mapping, region, context):
        if not self.supports(mapping, region, 'OPERATOR'):
            raise ValueError('The actual Identity backend cannot compile the requested parallel mapping')
        ranks = self.group.ranks
        value = ValueContract('hidden', self.shape, Placement(PlacementKind.REPLICATE),
            self.group.key, dtype=str(self.dtype).removeprefix('torch.'))
        tensor = DistributedTensorContract(value, (('tp', len(ranks)),), ranks, (), ('tp',), 'contiguous', ranks, 'identity.input')
        output = tensor
        gradient = replace(tensor, value=replace(value, logical_value='identity.gradient'), storage_id='identity.gradient')
        return RegionContract(region.region_id, RegionKind.LAYER, StatePattern(PatternPlacementKind.REPLICATE),
            StatePattern(PatternPlacementKind.REPLICATE), ParallelPolicy(tp=len(ranks)), self.key, self.key + ':profile',
            forward_inputs=(tensor,), forward_outputs=(output,), backward_gradient_inputs=(gradient,),
            backward_gradient_outputs=(gradient,), required_process_groups=(self.group,))

    def registration(self):
        return RegionBackendImplementation(self.key, self.source_revision, ('OPERATOR',), self.supports, self.compile_contract)
