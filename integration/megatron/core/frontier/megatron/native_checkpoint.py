from dataclasses import dataclass, replace
from pathlib import Path
from socket import gethostname
from time import perf_counter

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace, nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory, apply_factories
from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map, make_sharded_optimizer_tensor
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, MegatronOptimizer


@dataclass(frozen=True)
class GradientBinding:
    tensor: torch.Tensor
    flattened_range: tuple[int, int] | None = None


class NodeSerializedLoadStrategy(TorchDistLoadShardedStrategy):
    def load(self, sharded_state_dict, checkpoint_dir, async_strategy='mcore'):
        hosts = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, gethostname())
        peers = tuple(rank for rank, host in enumerate(hosts) if host == hosts[dist.get_rank()])
        wave = peers.index(dist.get_rank())
        wave_count = max(hosts.count(host) for host in set(hosts))
        result = None
        # 原生读取使用 no_dist，同一节点逐个进程读取以限制文件解码临时内存。
        for current in range(wave_count):
            if wave == current:
                result = super().load(sharded_state_dict, checkpoint_dir, async_strategy)
            dist.barrier()
        if result is None:
            raise RuntimeError('The native checkpoint load did not execute for this rank')
        return result


class NativeMegatronTrainingCheckpoint:
    def __init__(self, model_chunks, optimizer, gradient_attribute='main_grad'):
        self.model_chunks = tuple(model_chunks)
        self.optimizer = optimizer
        self.gradient_attribute = gradient_attribute
        if not self.model_chunks or not isinstance(optimizer, MegatronOptimizer):
            raise ValueError('Native migration requires actual Megatron model chunks and optimizer')
        if any(not callable(model.sharded_state_dict) or not callable(model.load_state_dict) for model in self.model_chunks):
            raise ValueError('Every actual model chunk must provide native sharded checkpoint methods')

    def components(self):
        def visit(optimizer):
            if isinstance(optimizer, ChainedOptimizer):
                return tuple(child for component in optimizer.chained_optimizers for child in visit(component))
            return (optimizer,)
        return visit(self.optimizer)

    def runtime_tensors(self):
        tensors, names = {}, {}
        for chunk, model in enumerate(self.model_chunks):
            for name, parameter in model.named_parameters():
                key = 'model:' + str(chunk) + ':' + name
                tensors[key] = parameter
                names[parameter] = key
        for index, component in enumerate(self.components()):
            if not isinstance(component, DistributedOptimizer):
                if not isinstance(component, Float16OptimizerWithFloat16Params):
                    raise ValueError('完整状态绑定缺少当前Optimizer的原生接口')
                bindings = {parameter: parameter for parameter in names}
                if hasattr(component, 'float16_groups'):
                    for source_group, main_group in zip(component.float16_groups,
                            component.fp32_from_float16_groups, strict=True):
                        for source, main in zip(source_group, main_group, strict=True):
                            bindings[main] = source
                for group in component.optimizer.param_groups:
                    for main in group['params']:
                        if main not in bindings:
                            raise ValueError('复制Optimizer参数无法对应实际模型参数')
                        source = bindings[main]
                        key = 'optimizer:' + str(index) + ':' + names[source]
                        tensors[key + ':param'] = main
                        for name, tensor in component.optimizer.state[main].items():
                            if isinstance(tensor, torch.Tensor):
                                tensors[key + ':' + name] = tensor
                        tensors[key + ':gradient'] = getattr(source, self.gradient_attribute)
                continue
            for buffer_ranges in component.gbuf_ranges:
                for buckets in buffer_ranges.values():
                    for bucket in buckets:
                        for parameter, ranges in bucket['param_map'].items():
                            key = 'optimizer:' + str(index) + ':' + names[parameter]
                            for name, tensor in component._get_main_param_and_optimizer_states(parameter).items():
                                tensors[key + ':' + name] = tensor
                            interval = ranges['param']
                            tensors[key + ':gradient'] = getattr(parameter, self.gradient_attribute).view(-1)[interval.start:interval.end]
        return tensors

    def runtime_scalar_states(self):
        bindings = {}
        scalar_types = (int, float, bool, str, tuple, list, dict, type(None))
        for component_index, component in enumerate(self.components()):
            for group_index, group in enumerate(component.optimizer.param_groups):
                for name, value in group.items():
                    if name != 'params' and isinstance(value, scalar_types):
                        key = f'optimizer:{component_index}:group:{group_index}:{name}'
                        bindings[key] = (group, name)
            for parameter_index, state in enumerate(component.optimizer.state.values()):
                for name, value in state.items():
                    if isinstance(value, scalar_types):
                        key = f'optimizer:{component_index}:state:{parameter_index}:{name}'
                        bindings[key] = (state, name)
        return bindings

    def runtime_ownership(self):
        model_state = [model.sharded_state_dict() for model in self.model_chunks]
        parameters = tuple(dict.fromkeys(parameter for model in self.model_chunks
            for parameter in model.parameters() if parameter.requires_grad))
        metadata = get_param_id_to_sharded_param_map(model_state, parameters)
        if set(metadata) != set(range(len(parameters))):
            raise ValueError('Actual parameter ownership requires complete native checkpoint metadata')
        identities = {parameter: metadata[index] for index, parameter in enumerate(parameters)}
        fragments = {}
        for parameter, identity in identities.items():
            native = identity.build() if isinstance(identity, ShardedTensorFactory) else {'parameter': identity}
            pieces = tuple(value for value in nested_values(native) if isinstance(value, ShardedTensor))
            if not pieces or sum(value.data.numel() for value in pieces) != parameter.numel():
                raise ValueError('Native checkpoint fragments do not cover the actual parameter storage')
            fragments[parameter] = tuple({'key': value.key, 'global_shape': value.global_shape,
                'global_offset': value.global_offset, 'local_shape': value.local_shape,
                'prepend_axis_num': value.prepend_axis_num, 'replica_id': value.replica_id}
                for value in pieces)
        model_bindings = []
        for chunk, model in enumerate(self.model_chunks):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    model_bindings.append({'chunk': chunk, 'native_name': name,
                        'parameter_key': identities[parameter].key,
                        'parameter_shape': tuple(parameter.shape),
                        'checkpoint_fragments': fragments[parameter]})
        owners, parameter_records, registered = [], [], set()
        for index, component in enumerate(self.components()):
            if not isinstance(component, DistributedOptimizer):
                raise ValueError('Actual ownership observation requires the native distributed optimizer')
            group = component.data_parallel_group
            members = tuple(dist.get_process_group_ranks(group))
            backend = dist.get_backend(group)
            if dist.get_rank() not in members:
                raise ValueError('The actual optimizer reduction group omits its owning rank')
            for buffer in component.buffers:
                for parameter in buffer.params:
                    if parameter in registered:
                        raise ValueError('An actual trainable parameter has multiple optimizer buffer owners')
                    registered.add(parameter)
                    parameter_records.append({'component': index, 'parameter_key': identities[parameter].key,
                        'parameter_shape': tuple(parameter.shape), 'parameter_dtype': str(parameter.dtype),
                        'checkpoint_fragments': fragments[parameter], 'reduction_group_ranks': members,
                        'reduction_group_backend': backend})
            for buffer_ranges in component.gbuf_ranges:
                for buckets in buffer_ranges.values():
                    for bucket in buckets:
                        for parameter, ranges in bucket['param_map'].items():
                            identity = identities[parameter]
                            interval = ranges['param']
                            states = component._get_main_param_and_optimizer_states(parameter)
                            parameter_replica = None
                            if 'master_param' in states:
                                proxy = states['param']
                                parameter_replica = {'shape': tuple(proxy.shape), 'dtype': str(proxy.dtype),
                                    'device': str(proxy.device), 'numel': proxy.numel()}
                                states = dict(states)
                                states['param'] = states.pop('master_param')
                            scalar_states = {}
                            if 'step' in states:
                                step = states['step']
                                if step.numel() != 1:
                                    raise ValueError('Native optimizer step counter must be scalar')
                                scalar_states['step'] = {'shape': tuple(step.shape), 'dtype': str(step.dtype),
                                    'device': str(step.device), 'numel': step.numel(), 'value': step.item()}
                                states = {name: tensor for name, tensor in states.items() if name != 'step'}
                            if set(states) != {'param', 'exp_avg', 'exp_avg_sq'}:
                                raise ValueError('Native Adam ownership requires all registered parameter-shaped states: '
                                    + str({name: tuple(tensor.shape) for name, tensor in states.items()}))
                            gradient = getattr(parameter, self.gradient_attribute)
                            if (gradient is None or gradient.shape != parameter.shape
                                    or any(tensor.numel() != interval.end - interval.start
                                        for tensor in states.values())):
                                detail = {'parameter': identity.key, 'shape': tuple(parameter.shape),
                                    'gradient_shape': None if gradient is None else tuple(gradient.shape),
                                    'interval': (interval.start, interval.end),
                                    'states': {name: tuple(tensor.shape) for name, tensor in states.items()}}
                                raise ValueError('Native optimizer ownership differs from its actual tensor storage: ' + str(detail))
                            owners.append({'component': index, 'parameter_key': identity.key,
                                'checkpoint_fragments': fragments[parameter],
                                'parameter_shape': tuple(parameter.shape), 'parameter_dtype': str(parameter.dtype),
                                'gradient_shape': tuple(gradient.shape), 'gradient_dtype': str(gradient.dtype),
                                'optimizer_flattened_range': (interval.start, interval.end),
                                'reduction_group_ranks': members, 'reduction_group_backend': backend,
                                'optimizer_scalar_states': scalar_states,
                                'optimizer_parameter_replica': parameter_replica,
                                'states': {name: {'shape': tuple(tensor.shape), 'dtype': str(tensor.dtype),
                                    'device': str(tensor.device), 'numel': tensor.numel()}
                                    for name, tensor in states.items()}})
        if registered != set(parameters):
            raise ValueError('Actual optimizer buffers do not cover the complete native training model')
        return {'rank': dist.get_rank(), 'world_size': dist.get_world_size(),
            'parameters': parameter_records, 'owners': owners, 'model_bindings': model_bindings}

    def gradient_state(self, model_state, is_loading):
        parameters = tuple(dict.fromkeys(parameter for model in self.model_chunks for parameter in model.parameters()
            if parameter.requires_grad))
        metadata = get_param_id_to_sharded_param_map(model_state, parameters)
        if set(metadata) != set(range(len(parameters))):
            raise ValueError('The native checkpoint does not cover every actual trainable model parameter')
        by_parameter = {parameter: metadata[index] for index, parameter in enumerate(parameters)}
        gradients, bindings, covered, distributed_parameters, local_distributed_parameters = {}, {}, set(), set(), set()
        for component in self.components():
            if isinstance(component, DistributedOptimizer):
                distributed_parameters.update(parameter for buffer in component.buffers for parameter in buffer.params)
                if not is_loading:
                    group = component.data_parallel_group
                    dp_rank, dp_size = dist.get_rank(group), dist.get_world_size(group)
                    for buffer in component.buffers:
                        for bucket_index, bucket in enumerate(buffer.buckets):
                            data = bucket.grad_data.view(-1)
                            if data.numel() % dp_size:
                                raise ValueError('Native gradient bucket does not divide over its actual reduction group')
                            shard = data.chunk(dp_size)[dp_rank].contiguous()
                            full = torch.empty_like(data)
                            dist.all_gather_into_tensor(full, shard, group=group)
                            if dp_rank == 0:
                                host = full.cpu()
                                for parameter, (start, end, index) in buffer.param_index_map.items():
                                    if index != bucket_index:
                                        continue
                                    gradient = host[start - bucket.offset:end - bucket.offset].view(parameter.shape)
                                    identity = str(len(gradients))
                                    gradients[identity] = make_sharded_optimizer_tensor(by_parameter[parameter], gradient, 'frontier.gradient')
                                    bindings[identity] = GradientBinding(getattr(parameter, self.gradient_attribute))
                                    covered.add(parameter)
                            del full
                    continue
                for native_buffer in component.buffers:
                    for parameter in native_buffer.params:
                        buffer = getattr(parameter, self.gradient_attribute)
                        if buffer is None or tuple(buffer.shape) != tuple(parameter.shape):
                            raise ValueError('Distributed gradient buffers must exist with their actual model parameter shape')
                        identity = str(len(gradients))
                        gradients[identity] = make_sharded_optimizer_tensor(by_parameter[parameter], buffer, 'frontier.gradient')
                        bindings[identity] = GradientBinding(buffer)
                        covered.add(parameter)
                        local_distributed_parameters.add(parameter)
            else:
                owned = tuple(parameter for group in (*component.float16_groups, *component.fp32_from_fp32_groups)
                    for parameter in group) if isinstance(component, Float16OptimizerWithFloat16Params) else component.get_parameters()
                for parameter in owned:
                    if parameter not in by_parameter:
                        continue
                    gradient = getattr(parameter, self.gradient_attribute)
                    if gradient is None:
                        raise ValueError('Every complete training checkpoint requires the actual retained gradient')
                    identity = str(len(gradients))
                    gradients[identity] = make_sharded_optimizer_tensor(by_parameter[parameter], gradient, 'frontier.gradient')
                    bindings[identity] = GradientBinding(gradient)
                    covered.add(parameter)
        required_local = (set(parameters) - distributed_parameters) | local_distributed_parameters
        if is_loading and required_local != covered or not is_loading and not required_local <= covered:
            raise ValueError('The native optimizer gradient checkpoint does not cover every model parameter')
        inventory = (tuple(sorted(value.key for value in by_parameter.values())),
            tuple(sorted(value.key.removeprefix('frontier.gradient.') for value in gradients.values())))
        inventories = [None] * dist.get_world_size()
        dist.all_gather_object(inventories, inventory)
        if {key for model_keys, _ in inventories for key in model_keys} != {
                key for _, gradient_keys in inventories for key in gradient_keys}:
            raise ValueError('The complete native distributed gradient inventory omits actual model parameters')
        return gradients, bindings

    def snapshot(self, is_loading):
        model_state = [model.sharded_state_dict() for model in self.model_chunks]
        optimizer_state = self.optimizer.sharded_state_dict(model_state, is_loading=is_loading,
            metadata={'distrib_optim_sharding_type': 'fully_reshardable'})
        gradients, bindings = self.gradient_state(model_state, is_loading)
        return {'models': model_state, 'optimizer': optimizer_state, 'gradients': gradients}, bindings

    def observe(self):
        state, _ = self.snapshot(is_loading=False)
        parameters = tuple(dict.fromkeys(parameter for model in self.model_chunks for parameter in model.parameters()
            if parameter.requires_grad))
        metadata = get_param_id_to_sharded_param_map(state['models'], parameters)
        inventory = tuple(sorted(value.key for value in metadata.values()))
        apply_factories(state)
        shards = []
        for category in ('models', 'optimizer', 'gradients'):
            for value in nested_values(state[category]):
                if isinstance(value, ShardedTensor):
                    interval = value.flattened_range
                    shards.append({'category': category, 'key': value.key,
                        'global_shape': value.global_shape, 'global_offset': value.global_offset,
                        'local_shape': value.local_shape, 'payload_shape': tuple(value.data.shape),
                        'dtype': str(value.data.dtype), 'device': str(value.data.device),
                        'replica_id': value.replica_id, 'prepend_axis_num': value.prepend_axis_num,
                        'flattened_range': None if interval is None else (interval.start, interval.stop, interval.step),
                        'storage_bytes': value.data.untyped_storage().nbytes()})
        local = {'rank': dist.get_rank(), 'trainable_parameter_keys': inventory,
            'local_trainable_numel': sum(value.numel() for value in parameters),
            'model_classes': tuple(type(model).__module__ + '.' + type(model).__name__ for model in self.model_chunks),
            'optimizer_class': type(self.optimizer).__module__ + '.' + type(self.optimizer).__name__,
            'shards': tuple(shards)}
        observations = [None] * dist.get_world_size()
        dist.all_gather_object(observations, local)
        return {'world_size': dist.get_world_size(),
            'logical_trainable_parameter_keys': tuple(sorted({key for observation in observations
                for key in observation['trainable_parameter_keys']})), 'ranks': tuple(observations)}

    def save(self, directory, update_boundary):
        if not update_boundary:
            raise ValueError('Native state checkpointing requires a completed optimizer update')
        destination = Path(directory)
        if dist.get_rank() == 0:
            destination.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        torch.cuda.synchronize()
        state, _ = self.snapshot(is_loading=False)
        started = perf_counter()
        dist_checkpointing.save(state, str(destination), sharded_strategy=('torch_dist', 1),
            validate_access_integrity=True)
        return perf_counter() - started

    def load(self, directory, update_boundary):
        if not update_boundary:
            raise ValueError('Native state publication requires an inactive target program at an update boundary')
        state, gradient_bindings = self.snapshot(is_loading=True)

        def allocate(value):
            if isinstance(value, (ShardedTensor, ShardedTensorFactory)):
                return value
            if isinstance(value, torch.Tensor):
                return value if value.device.type == 'cpu' else torch.empty_like(value, device='cpu')
            return value

        staging = dict_list_map_inplace(allocate, state)
        started = perf_counter()
        loaded = dist_checkpointing.load(staging, str(directory),
            sharded_strategy=NodeSerializedLoadStrategy(), validate_access_integrity=True, strict='raise_all')
        if (set(loaded) != {'models', 'optimizer', 'gradients'}
                or len(loaded['models']) != len(self.model_chunks)
                or set(loaded['gradients']) != set(gradient_bindings)):
            raise ValueError('Native loading returned an incomplete model, optimizer or gradient inventory')
        prepared = {}
        for key, target in gradient_bindings.items():
            value = loaded['gradients'][key]
            if target.flattened_range is not None:
                start, end = target.flattened_range
                value = value.view(-1)[start:end]
            if value.shape != target.tensor.shape or value.dtype != target.tensor.dtype:
                raise ValueError('A loaded native gradient differs from its actual target buffer')
            prepared[key] = value
        torch.cuda.synchronize()
        dist.barrier()
        for model, values in zip(self.model_chunks, loaded['models'], strict=True):
            model.load_state_dict(values, strict=True)
        self.optimizer.load_state_dict(loaded['optimizer'])
        with torch.no_grad():
            for key, gradient in gradient_bindings.items():
                gradient.tensor.copy_(prepared[key])
        torch.cuda.synchronize()
        dist.barrier()
        return perf_counter() - started
