import json
import weakref
import time
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def tensor_storage_inventory(bindings, *, device_type='cuda'):
    storages, logical, views = {}, {}, set()
    for key, category, tensor in bindings:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        if local.device.type != device_type or local.numel() == 0:
            continue
        storage = local.untyped_storage()
        address = storage.data_ptr()
        if storage.nbytes() == 0:
            raise ValueError('显存状态清单引用了已经释放的非空Tensor')
        identity = (str(local.device), address)
        row = storages.setdefault(identity, dict(storage_pointer=address,
            device=str(local.device), bytes=storage.nbytes(), categories=[], tensor_keys=[]))
        if row['bytes'] != storage.nbytes():
            raise ValueError('同一状态Storage地址对应不同分配大小')
        if category not in row['categories']:
            row['categories'].append(category)
        row['tensor_keys'].append(key)
        view = (category, str(local.device), local.data_ptr(), local.numel(), local.dtype)
        if view not in views:
            logical[category] = logical.get(category, 0) + local.numel() * local.element_size()
            views.add(view)
    return dict(logical_tensor_bytes=logical,
        unique_storage_bytes=sum(row['bytes'] for row in storages.values()),
        storages=list(storages.values()),
        logical_scope='distinct_tensor_views_per_category_with_cross_category_aliases_preserved',
        physical_scope='unique_allocated_storage_per_device')


class MemoryAudit:
    def __init__(self, evidence):
        self.evidence = evidence
        self.samples = []
        self.state_inventory = None
        self.communication_tensors = []
        self.communication_lifetimes = []
        self.history_limit = int(self.evidence.request.get('paramorph_memory_history_limit', 10000))
        if not 1000 <= self.history_limit <= 50000:
            raise ValueError('CUDA allocator历史容量必须位于1000到50000条事件之间')
        self.history_context = self.evidence.request.get('paramorph_memory_history_context')
        if self.history_context not in (None, 'state', 'alloc', 'all'):
            raise ValueError('CUDA分配历史的调用位置设置无效')
        torch.cuda.memory._record_memory_history(enabled='all', context=self.history_context, stacks='python',
                                                max_entries=self.history_limit, clear_history=True)

    def track_communication(self, tensors, role):
        for tensor in tensors:
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            row = dict(step=self.evidence.step, role=role, storage_pointer=storage.data_ptr(),
                storage_bytes=storage.nbytes(), first_sample=len(self.samples), last_live_sample=None,
                first_observed_host_ns=time.perf_counter_ns(), last_observed_host_ns=None,
                release_observed_sample=None, release_observed_host_ns=None,
                reference_id=len(self.communication_lifetimes),
                lifetime_scope='transport_tensor_reference_observed_at_operation_boundaries')
            self.communication_tensors.append((weakref.ref(tensor), row))
            self.communication_lifetimes.append(row)

    def snapshot(self, phase):
        torch.cuda.synchronize()
        query_start = time.perf_counter_ns()
        free, capacity = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        communication = self.evidence.plan_executor.communication if self.evidence.plan_executor else None
        storages = {}
        alive = []
        for reference, row in self.communication_tensors:
            tensor = reference()
            if tensor is not None:
                storage = tensor.untyped_storage()
                item = storages.setdefault(storage.data_ptr(), dict(storage_bytes=storage.nbytes(), roles=[], reference_ids=[]))
                if item['storage_bytes'] != storage.nbytes():
                    raise ValueError('同一存活Storage地址具有不同分配大小')
                if row['role'] not in item['roles']:
                    item['roles'].append(row['role'])
                item['reference_ids'].append(row['reference_id'])
                row['last_live_sample'] = len(self.samples)
                row['last_observed_host_ns'] = time.perf_counter_ns()
                alive.append((reference, row))
            else:
                row['release_observed_sample'] = len(self.samples)
                row['release_observed_host_ns'] = time.perf_counter_ns()
        self.communication_tensors = alive
        if communication is not None:
            tensors = list(communication.receive_buffers.values())
            tensors.extend(state['tensor'] for state in communication.states.values()
                           if state.get('tensor') is not None)
            for tensor in tensors:
                storage = tensor.untyped_storage()
                storages.setdefault(storage.data_ptr(), dict(storage_bytes=storage.nbytes(),
                    roles=['paramorph_executor_transport'], reference_ids=[]))
        query_end = time.perf_counter_ns()
        self.samples.append(dict(step=self.evidence.step, phase=phase,
            sample_index=len(self.samples), snapshot_host_ns=query_end,
            query_start_host_ns=query_start, query_end_host_ns=query_end,
            clock_domain='rank_local_monotonic',
            allocated_bytes=allocated, reserved_bytes=reserved,
            allocator_unused_reserved_bytes=reserved - allocated,
            device_used_bytes=capacity - free,
            device_capacity_bytes=capacity,
            device_outside_allocator_bytes=max(0, capacity - free - reserved),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            communication_storage_bytes=sum(row['storage_bytes'] for row in storages.values()),
            communication_storage_count=len(storages),
            communication_storages=[dict(storage_pointer=address, **row) for address, row in sorted(storages.items())],
            snapshot_scope='synchronized_operation_boundary_current_values',
            peak_counter_scope='allocator_counter_since_latest_runtime_peak_reset'))

    def inventory(self, models, optimizer):
        from native_frontier_checkpoint import NativeMegatronTrainingCheckpoint

        tensors = NativeMegatronTrainingCheckpoint(models, optimizer).runtime_tensors()
        storages = {}
        logical = {}
        for key, tensor in tensors.items():
            if tensor.device.type != 'cuda':
                continue
            category = ('parameters' if key.startswith('model:') else
                        'gradients' if key.endswith(':gradient') else 'optimizer')
            logical[category] = logical.get(category, 0) + tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            identifier = storage.data_ptr()
            if identifier not in storages:
                storages[identifier] = dict(storage_pointer=identifier, bytes=storage.nbytes(), categories=[])
            if category not in storages[identifier]['categories']:
                storages[identifier]['categories'].append(category)
        self.state_inventory = dict(logical_tensor_bytes=logical,
            unique_storage_bytes=sum(row['bytes'] for row in storages.values()),
            storages=list(storages.values()),
            source='actual_model_gradient_and_optimizer_tensor_storage')

    def inventory_fsdp(self, models, optimizer):
        bindings = []
        categories = {
            'model_weight_buffer': 'parameters',
            'transpose_weight_buffer': 'parameters',
            'main_weight_buffer': 'optimizer',
            'main_grad_buffer': 'gradients',
            'hfsdp_helper_wbuf': 'parameters',
            'hfsdp_helper_wtbuf': 'parameters',
            'hfsdp_helper_gbuf': 'gradients',
        }
        for vp, (_, buffer) in enumerate(self.evidence.fsdp_model_bindings(models)):
            for group_index, group in enumerate(buffer.parameter_groups):
                for name, category in categories.items():
                    owner = getattr(group, name)
                    if owner is not None and owner.data is not None:
                        bindings.append((f'fsdp:vp{vp}:group{group_index}:{name}', category, owner.data))
                if group.sharding_strategy == 'no_shard' and group.model_weight_buffer is None:
                    bindings.extend((f'fsdp:vp{vp}:group{group_index}:parameter{index}',
                        'parameters', parameter) for index, parameter in enumerate(group.params))
        pending, component_index = [optimizer], 0
        while pending:
            component = pending.pop(0)
            children = getattr(component, 'chained_optimizers', None)
            if children is not None:
                pending.extend(children)
                continue
            if getattr(component, 'is_stub_optimizer', False):
                continue
            for group_index, group in enumerate(component.optimizer.param_groups):
                for parameter_index, parameter in enumerate(group['params']):
                    for name, value in component.optimizer.state.get(parameter, {}).items():
                        if isinstance(value, torch.Tensor):
                            key = f'optimizer:{component_index}:group{group_index}:parameter{parameter_index}:{name}'
                            bindings.append((key, 'optimizer', value))
            component_index += 1
        self.state_inventory = dict(tensor_storage_inventory(bindings),
            source='actual_megatron_fsdp_local_buffers_and_optimizer_state_storage')
        if not self.state_inventory['storages']:
            raise ValueError('FSDP显存清单没有实际CUDA状态存储')

    def save(self):
        if self.state_inventory is None or not self.samples:
            raise ValueError('独立显存测量缺少实际状态或者采样记录')
        payload = dict(rank=dist.get_rank(), request=self.evidence.request,
            physical_rank=self.evidence.request.get('rank_mapping', list(range(dist.get_world_size())))[dist.get_rank()],
            gpu_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            device_index=torch.cuda.current_device(), hostname=os.uname().nodename,
            samples=self.samples, state_inventory=self.state_inventory,
            communication_lifetimes=self.communication_lifetimes,
            communication_bytes_scope='unique_live_storages_referenced_by_actual_pipeline_transport',
            source='synchronized_cuda_allocator_and_device_memory_queries',
            device_peak_scope='maximum_of_synchronized_operation_boundary_samples',
            device_outside_allocator_scope='all_device_allocations_outside_this_rank_allocator')
        path = self.evidence.output / f'rank{dist.get_rank()}.paramorph-memory-audit.json'
        allocator = torch.cuda.memory._snapshot()
        history_path = self.evidence.output / f'rank{dist.get_rank()}.paramorph-allocator-history.json'
        history_path.write_text(json.dumps(allocator, indent=2, allow_nan=False) + '\n')
        payload.update(allocator_history_path=str(history_path), allocator_history_max_entries=self.history_limit,
            allocator_history_context=self.history_context,
            allocator_history_may_be_truncated=any(len(events) >= self.history_limit
                                                  for events in allocator['device_traces']),
            buffer_lifetime_scope='actual_allocator_events_and_observed_transport_tensor_references',
            allocator_snapshot_scope='allocator_events_recorded_from_training_start_with_initial_live_segments',
            coincident_component_scope='each_sample_contains_current_allocator_device_and_transport_storage_values',
            accounting_rule='communication_storage_is_a_subset_of_allocator_storage_and_is_not_added_to_allocated_bytes')
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        torch.cuda.memory._record_memory_history(enabled=None)
