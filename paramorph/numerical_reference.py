from copy import deepcopy
import json
import random
import hashlib
import os
import mmap

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from native_frontier_checkpoint import NativeMegatronTrainingCheckpoint


class RecordedIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.batches = []
        self.replaying = False
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.replaying:
            if self.position >= len(self.batches):
                raise ValueError('原生调度请求了自研调度没有使用的训练输入')
            batch = deepcopy(self.batches[self.position])
            self.position += 1
            return batch
        batch = next(self.iterator)
        self.batches.append(deepcopy(batch))
        return batch

    def __getattr__(self, name):
        return getattr(self.iterator, name)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.random.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                tracker=deepcopy(get_cuda_rng_tracker().get_states()))


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'])
    get_cuda_rng_tracker().set_states(deepcopy(state['tracker']))


class NumericalReference:
    def __init__(self, evidence):
        self.evidence = evidence
        self.boundaries = {}
        self.boundary_records = []
        self.tensor_files = []

    def save_tensor(self, key, tensor):
        if not tensor.is_contiguous():
            raise ValueError('完整数值快照需要实际连续Tensor存储')
        path = self.directory / (hashlib.sha256(key.encode()).hexdigest() + '.npy')
        if path.exists():
            raise ValueError('数值快照身份重复')
        flat = tensor.detach().reshape(-1)
        disk = flat.view(torch.int16) if tensor.dtype == torch.bfloat16 else flat
        dtype = disk[:0].cpu().numpy().dtype
        stored = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(flat.numel(),))
        with path.open('rb') as handle:
            for offset in range(0, flat.numel(), 1024 * 1024):
                stored[offset:offset + 1024 * 1024] = disk[offset:offset + 1024 * 1024].cpu().numpy()
                if offset % (16 * 1024 * 1024) == 15 * 1024 * 1024 or offset + 1024 * 1024 >= flat.numel():
                    stored.flush()
                    stored._mmap.madvise(mmap.MADV_DONTNEED)
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.fsync(handle.fileno())
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        del stored
        self.tensor_files.append(path)
        return dict(path=path, shape=list(tensor.shape), dtype=tensor.dtype, elements=tensor.numel())

    def tensor_chunk(self, array, descriptor, offset, count):
        value = torch.from_numpy(np.array(array[offset:offset + count], copy=True))
        return value.view(torch.bfloat16) if descriptor['dtype'] == torch.bfloat16 else value

    def restore_tensor(self, descriptor, tensor):
        if list(tensor.shape) != descriptor['shape'] or tensor.dtype != descriptor['dtype']:
            raise ValueError('数值快照与实际Tensor状态不同')
        stored = np.load(descriptor['path'], mmap_mode='r', allow_pickle=False)
        flat = tensor.detach().reshape(-1)
        for offset in range(0, flat.numel(), 1024 * 1024):
            flat[offset:offset + 1024 * 1024].copy_(self.tensor_chunk(stored, descriptor,
                offset, min(1024 * 1024, flat.numel() - offset)))
        del stored

    def compare_tensor(self, key, descriptor, tensor, absolute=None, relative=None):
        if list(tensor.shape) != descriptor['shape'] or tensor.dtype != descriptor['dtype']:
            raise ValueError(f'数值对照形状或者精度不同：{key}')
        absolute = self.evidence.request.get('numerical_absolute_tolerance', 1e-4) if absolute is None else absolute
        relative = self.evidence.request.get('numerical_relative_tolerance', 1e-2) if relative is None else relative
        stored = np.load(descriptor['path'], mmap_mode='r', allow_pickle=False)
        flat = tensor.detach().reshape(-1)
        maximum_absolute = maximum_relative = error_sum = reference_sum = 0.0
        accepted = True
        for offset in range(0, flat.numel(), 1024 * 1024):
            count = min(1024 * 1024, flat.numel() - offset)
            left = self.tensor_chunk(stored, descriptor, offset, count).double()
            right = flat[offset:offset + count].cpu().double()
            difference = (left - right).abs()
            accepted = accepted and bool(torch.isfinite(left).all() and torch.isfinite(right).all()
                and (difference <= absolute + relative * right.abs()).all())
            maximum_absolute = max(maximum_absolute, difference.max().item())
            maximum_relative = max(maximum_relative, (difference / right.abs().clamp_min(1e-12)).max().item())
            error_sum += difference.square().sum().item()
            reference_sum += right.square().sum().item()
        del stored
        relative_l2 = (error_sum / max(reference_sum, 1e-24)) ** 0.5
        l2_error = error_sum ** 0.5
        l2_limit = absolute * flat.numel() ** 0.5 + relative * reference_sum ** 0.5
        accepted = accepted and l2_error <= l2_limit
        return dict(tensor=key, elements=flat.numel(), max_absolute_error=maximum_absolute,
            max_relative_error=maximum_relative, relative_l2_error=relative_l2,
            absolute_l2_error=l2_error, absolute_l2_tolerance=l2_limit,
            absolute_tolerance=absolute, relative_tolerance=relative, within_tolerance=accepted)

    def capture(self, kind, identity, output):
        tensors = output if isinstance(output, (tuple, list)) else [output]
        for index, tensor in enumerate(tensors):
            key = f'{kind}:{identity["vp_stage"]}:{identity["microbatch"]}:{index}'
            if self.pass_name == 'custom':
                if key in self.boundaries:
                    raise ValueError('训练边界Tensor身份重复')
                self.boundaries[key] = None if tensor is None else self.save_tensor(key, tensor)
            else:
                if key not in self.boundaries:
                    raise ValueError('原生执行产生了自研执行缺失的边界Tensor')
                descriptor = self.boundaries[key]
                if descriptor is None or tensor is None:
                    if descriptor is not None or tensor is not None:
                        raise ValueError('原生和自研边界梯度的存在状态不同')
                    row = dict(tensor=key, elements=0, within_tolerance=True)
                else:
                    row = self.compare_tensor(key, descriptor, tensor)
                self.boundary_records.append(row)
                if not row['within_tolerance']:
                    payload = dict(rank=dist.get_rank(), step=self.evidence.step, status='FAILED',
                        source='same_loaded_weights_and_recorded_c4_inputs', first_difference=row,
                        full_training_equivalence_certified=False)
                    (self.evidence.output / f'rank{dist.get_rank()}.paramorph-numerical-reference.json').write_text(
                        json.dumps(payload, indent=2, allow_nan=False) + '\n')
                    raise ValueError(f'原生与自研执行首次边界数值差异：{key}')

    def verify_optimizer(self, original, arguments, keywords):
        checkpoint = NativeMegatronTrainingCheckpoint(self.models, self.evidence.optimizer)
        scalar_bindings = checkpoint.runtime_scalar_states()
        initial_scalars = {key: deepcopy(mapping[name]) for key, (mapping, name) in scalar_bindings.items()}
        before = checkpoint.runtime_tensors()
        before = {key: tensor for key, tensor in before.items() if not key.endswith(':gradient')}
        initial = {key: self.save_tensor('optimizer-before:' + key, tensor) for key, tensor in before.items()}
        torch.cuda.synchronize()
        dist.barrier()
        result = original(*arguments, **keywords)
        if not result[0]:
            raise ValueError('自研训练Optimizer Update没有成功')
        torch.cuda.synchronize()
        updated = checkpoint.runtime_tensors()
        updated_scalar_bindings = checkpoint.runtime_scalar_states()
        if updated_scalar_bindings.keys() != scalar_bindings.keys():
            raise ValueError('Optimizer Update改变了标量状态身份集合')
        actual_scalars = {key: deepcopy(mapping[name]) for key, (mapping, name) in updated_scalar_bindings.items()}
        if {key for key in updated if not key.endswith(':gradient')} != before.keys():
            raise ValueError('Optimizer Update改变了完整状态身份集合')
        actual = {key: self.save_tensor('optimizer-custom:' + key, tensor) for key, tensor in updated.items()}
        for key, tensor in before.items():
            self.restore_tensor(initial[key], tensor)
        for key, (mapping, name) in scalar_bindings.items():
            mapping[name] = deepcopy(initial_scalars[key])
        for key, parameter in self.parameters.items():
            self.restore_tensor(self.native_gradients[key], parameter.main_grad)
        torch.cuda.synchronize()
        dist.barrier()
        reference_result = original(*arguments, **keywords)
        if not reference_result[0]:
            raise ValueError('原生梯度参考的Optimizer Update没有成功')
        torch.cuda.synchronize()
        compared = checkpoint.runtime_tensors()
        compared_scalar_bindings = checkpoint.runtime_scalar_states()
        if compared_scalar_bindings.keys() != scalar_bindings.keys():
            raise ValueError('原生Optimizer参考改变了标量状态身份集合')
        scalar_records = []
        for key, (mapping, name) in compared_scalar_bindings.items():
            scalar_records.append(dict(state=key, actual=actual_scalars[key], reference=deepcopy(mapping[name]),
                within_tolerance=actual_scalars[key] == mapping[name]))
            mapping[name] = deepcopy(actual_scalars[key])
        records = []
        for key, descriptor in actual.items():
            tensor = compared[key]
            parameter_state = key.startswith('model:') or key.endswith(':param') or key.endswith(':master_param')
            exact_counter = key.endswith(':step')
            absolute, relative = (0.0, 0.0) if exact_counter else (
                (1e-8, 1e-6) if parameter_state and tensor.dtype == torch.float32 else (None, None))
            records.append(self.compare_tensor(key, descriptor, tensor, absolute, relative))
            self.restore_tensor(descriptor, tensor)
        if compared.keys() != updated.keys():
            raise ValueError('原生参考Optimizer状态与实际更新状态身份不同')
        torch.cuda.synchronize()
        accepted = all(row['within_tolerance'] for row in records + scalar_records)
        self.payload.update(status='PASS' if accepted else 'FAILED', optimizer_states=records,
            optimizer_scalar_states=scalar_records,
            full_training_equivalence_certified=accepted,
            optimizer_validation='same_initial_model_optimizer_state_with_native_and_custom_full_gradients',
            continued_training_state='restored_actual_custom_optimizer_update')
        self.save_report()
        certified = torch.tensor([int(accepted)], device='cuda')
        dist.all_reduce(certified, op=dist.ReduceOp.MIN)
        if not certified.item():
            raise ValueError('完整Optimizer状态或者更新后的模型参数数值对照未通过')
        self.evidence.numerical_optimizer_audit = None
        for path in self.tensor_files:
            path.unlink()
        self.directory.rmdir()
        return result

    def save_report(self):
        self.evidence.numerical_reports[self.evidence.step] = self.payload
        payload = dict(self.payload, iterations=list(self.evidence.numerical_reports.values()))
        (self.evidence.output / f'rank{dist.get_rank()}.paramorph-numerical-reference.json').write_text(
            json.dumps(payload, indent=2, allow_nan=False) + '\n')

    def run(self, custom, native, arguments, keywords):
        if arguments:
            raise ValueError('数值对照需要使用真实调度入口的具名参数')
        iterators = [RecordedIterator(iterator) if iterator is not None else None
                     for iterator in keywords['data_iterator']]
        selected = dict(keywords, data_iterator=iterators)
        self.directory = self.evidence.output / f'rank{dist.get_rank()}.numerical-tensors' / f'step{self.evidence.step}'
        self.directory.mkdir(parents=True, exist_ok=False)
        models = keywords['model']
        model_buffers = {f'{vp}:{name}': tensor for vp, model in enumerate(models)
                         for name, tensor in model.named_buffers()}
        before_buffers = {key: self.save_tensor('buffer-before:' + key, tensor)
                          for key, tensor in model_buffers.items()}
        self.pass_name = 'custom'
        self.evidence.numerical_capture = self.capture
        before = rng_state()
        output = custom(**selected)
        after = rng_state()
        actual_buffers = {key: self.save_tensor('buffer-custom:' + key, tensor)
                          for key, tensor in model_buffers.items()}
        buffer_updates = [self.compare_tensor(key, before_buffers[key], tensor, 0.0, 0.0)
                          for key, tensor in model_buffers.items()]
        gradients = {}
        parameters = {}
        seen = set()
        block_elements = 1024 * 1024
        for vp, model in enumerate(models):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    key = f'{vp}:{name}'
                    parameters[key] = parameter
                    gradient = parameter.main_grad.detach().reshape(-1)
                    if gradient.dtype != torch.float32:
                        raise ValueError('当前完整梯度对照要求实际FP32 main_grad')
                    gradients[key] = self.save_tensor('custom-gradient:' + key, parameter.main_grad)['path']
        torch.cuda.synchronize()
        dist.barrier()
        for model in models:
            model.zero_grad_buffer()
        for key, tensor in model_buffers.items():
            self.restore_tensor(before_buffers[key], tensor)
        for iterator in iterators:
            if iterator is not None:
                iterator.replaying = True
        restore_rng(before)
        self.evidence.reference_phase = True
        self.pass_name = 'native'
        reference_output = native(**selected)
        self.evidence.reference_phase = False
        self.evidence.numerical_capture = None
        torch.cuda.synchronize()
        buffer_records = []
        for key, tensor in model_buffers.items():
            buffer_records.append(self.compare_tensor(key, actual_buffers[key], tensor, 0.0, 0.0))
            self.restore_tensor(actual_buffers[key], tensor)
        records = []
        absolute_tolerance = self.evidence.request.get('numerical_absolute_tolerance', 1e-4)
        relative_tolerance = self.evidence.request.get('numerical_relative_tolerance', 1e-2)
        self.native_gradients = {}
        for key, parameter in parameters.items():
            self.native_gradients[key] = self.save_tensor('native-gradient:' + key, parameter.main_grad)
            actual = np.load(gradients[key], mmap_mode='r', allow_pickle=False)
            reference = parameter.main_grad.detach().reshape(-1)
            maximum_absolute = maximum_relative = squared_error = squared_reference = 0.0
            accepted = True
            for offset in range(0, reference.numel(), block_elements):
                right = reference[offset:offset + block_elements].cpu()
                left = torch.from_numpy(np.array(actual[offset:offset + right.numel()], copy=True))
                difference = (left - right).abs()
                accepted = accepted and bool(torch.isfinite(left).all() and torch.isfinite(right).all()
                    and (difference <= absolute_tolerance + relative_tolerance * right.abs()).all())
                maximum_absolute = max(maximum_absolute, difference.max().item())
                maximum_relative = max(maximum_relative,
                    (difference / right.abs().clamp_min(1e-12)).max().item())
                squared_error += difference.double().square().sum().item()
                squared_reference += right.double().square().sum().item()
            relative_l2 = (squared_error / max(squared_reference, 1e-24)) ** 0.5
            l2_error = squared_error ** 0.5
            l2_limit = absolute_tolerance * reference.numel() ** 0.5 + relative_tolerance * squared_reference ** 0.5
            accepted = accepted and l2_error <= l2_limit
            records.append(dict(parameter=key, elements=reference.numel(),
                max_absolute_error=maximum_absolute, max_relative_error=maximum_relative,
                relative_l2_error=relative_l2, within_tolerance=accepted))
            records[-1].update(absolute_l2_error=l2_error, absolute_l2_tolerance=l2_limit,
                absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)
            destination = parameter.main_grad.reshape(-1)
            for offset in range(0, destination.numel(), block_elements):
                destination[offset:offset + block_elements].copy_(torch.from_numpy(
                    np.array(actual[offset:offset + block_elements], copy=True)))
            del actual
        losses = []
        if len(output) != len(reference_output):
            raise ValueError('实际训练Loss数量与原生参考不同')
        for index, (actual, reference) in enumerate(zip(output, reference_output, strict=True)):
            if actual.keys() != reference.keys():
                raise ValueError('实际训练Loss字段与原生参考不同')
            for name in actual:
                left, right = actual[name].detach().cpu(), reference[name].detach().cpu()
                if left.shape != right.shape:
                    raise ValueError('实际训练Loss形状与原生参考不同')
                losses.append(dict(microbatch=index, name=name,
                    max_absolute_error=(left - right).abs().max().item(),
                    within_tolerance=torch.allclose(left, right, atol=absolute_tolerance,
                                                   rtol=relative_tolerance)))
        for iterator in iterators:
            if iterator is not None and iterator.position != len(iterator.batches):
                raise ValueError('原生参考没有消费全部实际训练输入')
        restore_rng(after)
        if len(self.boundary_records) != len(self.boundaries):
            raise ValueError('完整前向输出和DGRAD对照没有覆盖全部边界')
        accepted = all(row['within_tolerance'] for row in records + losses + self.boundary_records + buffer_records)
        payload = dict(rank=dist.get_rank(), step=self.evidence.step,
            status='PASS' if accepted else 'FAILED', source='same_loaded_weights_and_recorded_c4_inputs',
            absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
            compared_parameter_elements=sum(row['elements'] for row in records),
            snapshot_storage='npy_memory_mapped_full_gradient_arrays',
            comparison_block_elements=block_elements,
            gradients=records, losses=losses,
            boundary_tensors=self.boundary_records,
            model_buffers=buffer_records,
            changed_training_buffers=[row for row in buffer_updates if not row['within_tolerance']],
            forward_initial_state='restored_same_model_parameters_rng_and_all_registered_buffers',
            full_training_equivalence_certified=False,
            optimizer_input='restored_actual_custom_schedule_gradients')
        self.payload = payload
        self.models, self.parameters = models, parameters
        self.evidence.numerical_optimizer_audit = self
        self.save_report()
        result = torch.tensor([int(accepted)], device='cuda')
        dist.all_reduce(result, op=dist.ReduceOp.MIN)
        if not result.item():
            raise ValueError('自研调度与原生调度的完整Gradient或者Loss数值对照未通过')
        return output
