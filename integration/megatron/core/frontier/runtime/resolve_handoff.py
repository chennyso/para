from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PeerSlice:
    peer_rank: int
    start: int
    end: int

    def __post_init__(self):
        if self.peer_rank < 0 or self.start < 0 or self.end <= self.start:
            raise ValueError('peer slice must have a valid rank and nonempty interval')


@dataclass(frozen=True)
class ResolveHandoffPlan:
    send_slices: tuple[PeerSlice, ...]
    recv_sources: tuple[int, ...]
    local_shape: tuple[int, ...]
    dim: int
    group_key: str

    def __post_init__(self):
        if not self.local_shape or min(self.local_shape) < 1 or not 0 <= self.dim < len(self.local_shape):
            raise ValueError('handoff requires a valid shape and projection dimension')
        if bool(self.send_slices) == bool(self.recv_sources):
            raise ValueError('handoff rank must have exactly one source or destination role')
        peers = tuple(item.peer_rank for item in self.send_slices) or self.recv_sources
        if len(set(peers)) != len(peers) or min(peers) < 0:
            raise ValueError('handoff peers must be distinct nonnegative ranks')
        if self.send_slices:
            slices = sorted(self.send_slices, key=lambda item: item.start)
            if slices[0].start != 0 or slices[-1].end != self.local_shape[self.dim]:
                raise ValueError('projection slices must cover the source dimension')
            if any(left.end != right.start for left, right in zip(slices, slices[1:])):
                raise ValueError('projection slices must be disjoint and contiguous')
            if len({item.end - item.start for item in slices}) != 1:
                raise ValueError('this lowering requires equal destination shard sizes')

    @property
    def projected_shape(self):
        shape = list(self.local_shape)
        if self.send_slices:
            shape[self.dim] = self.send_slices[0].end - self.send_slices[0].start
        return tuple(shape)


class ResolveOnHandoffRuntime:
    def __init__(self, plan, process_group, buffer_pool):
        if process_group is None or not dist.is_initialized():
            raise RuntimeError('an initialized process group is required for handoff')
        if dist.get_backend(process_group) != 'nccl' or not torch.cuda.is_available():
            raise RuntimeError('ResolveOnHandoff requires CUDA and an NCCL process group')
        rank = dist.get_rank()
        members = dist.get_process_group_ranks(process_group)
        peers = tuple(item.peer_rank for item in plan.send_slices) or plan.recv_sources
        if rank not in members or any(peer not in members or peer == rank for peer in peers):
            raise ValueError('handoff rank mapping differs from process group membership')
        count, shape, dtype, device = buffer_pool.specification(0)
        if count != len(peers) or shape != plan.projected_shape or device.type != 'cuda':
            raise ValueError('preallocated buffers must match the compiled handoff shape and peers')
        self.plan = plan
        self.process_group = process_group
        self.buffer_pool = buffer_pool
        self._stream = torch.cuda.Stream(device=device)
        self._ready = torch.cuda.Event()
        self._output = torch.empty(plan.local_shape, dtype=dtype, device=device)
        self._dtype = dtype
        self._device = device
        self._direction = None
        self._source_pending = False
        self._output_live = False
        self._retain_output = True

    def _check_tensor(self, tensor, shape):
        if tuple(tensor.shape) != shape or tensor.dtype != self._dtype or tensor.device != self._device:
            raise ValueError('tensor differs from the compiled handoff contract')

    def _launch(self, operations):
        with torch.cuda.stream(self._stream):
            requests = dist.batch_isend_irecv(operations)
            for request in requests:
                request.wait()
            self._ready.record(self._stream)
        return requests, self._ready

    def forward_source(self, tensor, retain_for_backward=True):
        if not self.plan.send_slices:
            raise ValueError('forward_source requires the source role')
        self._check_tensor(tensor, self.plan.local_shape)
        if self._source_pending:
            raise RuntimeError('source lifetime slot must complete backward before another forward')
        self._source_pending = retain_for_backward
        buffers = self.buffer_pool.acquire_many(0, len(self.plan.send_slices), self.plan.projected_shape)
        self._direction = 'forward_source'
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.no_grad(), torch.cuda.stream(self._stream), torch.cuda.nvtx.range('Frontier/PROJECT_PP_SEND'):
            for item, buffer in zip(self.plan.send_slices, buffers, strict=True):
                buffer.copy_(tensor.narrow(self.plan.dim, item.start, item.end - item.start))
            operations = [dist.P2POp(dist.isend, buffer, item.peer_rank, self.process_group)
                          for item, buffer in zip(self.plan.send_slices, buffers, strict=True)]
            requests, event = self._launch(operations)
        return requests, buffers, event

    def forward_destination(self, slot_id, retain_for_backward=True):
        if self.plan.send_slices or slot_id != 0:
            raise ValueError('forward_destination requires a compiled destination slot')
        if self._output_live:
            raise RuntimeError('destination output remains live until its backward adjoint completes')
        self._retain_output = retain_for_backward
        buffers = self.buffer_pool.acquire_many(slot_id, len(self.plan.recv_sources), self.plan.local_shape)
        self._direction = 'forward_destination'
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        operations = [dist.P2POp(dist.irecv, buffer, peer, self.process_group)
                      for peer, buffer in zip(self.plan.recv_sources, buffers, strict=True)]
        requests, _ = self._launch(operations)
        return buffers, requests

    def finish_destination(self, buffers, requests, slot_id=0):
        if slot_id != 0 or self._direction not in ('forward_destination', 'backward_destination'):
            raise ValueError('no matching receive operation is active')
        with torch.no_grad(), torch.cuda.stream(self._stream), torch.cuda.nvtx.range('Frontier/RESOLVE_HANDOFF'):
            if self._direction == 'forward_destination':
                self._output.copy_(buffers[0])
                for buffer in buffers[1:]:
                    self._output.add_(buffer)
            else:
                for item, buffer in zip(self.plan.send_slices, buffers, strict=True):
                    self._output.narrow(self.plan.dim, item.start, item.end - item.start).copy_(buffer)
            self._ready.record(self._stream)
        torch.cuda.current_stream(self._device).wait_event(self._ready)
        self.buffer_pool.release(slot_id)
        if self._direction == 'forward_destination':
            self._output_live = self._retain_output
        else:
            self._source_pending = False
        self._direction = None
        return self._output

    def finish_source(self, requests, event=None):
        if self._direction not in ('forward_source', 'backward_source'):
            raise ValueError('no matching send operation is active')
        torch.cuda.current_stream(self._device).wait_event(self._ready)
        if self.buffer_pool.is_in_use(0):
            self.buffer_pool.release(0)
        if self._direction == 'backward_source':
            self._output_live = False
        self._direction = None

    def backward_source(self, grad):
        if self.plan.send_slices:
            raise ValueError('backward_source requires the forward destination role')
        if not self._output_live:
            raise RuntimeError('destination adjoint requires a retained forward value')
        self._check_tensor(grad, self.plan.local_shape)
        buffers = self.buffer_pool.acquire_many(0, len(self.plan.recv_sources), self.plan.local_shape)
        self._direction = 'backward_source'
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.no_grad(), torch.cuda.stream(self._stream), torch.cuda.nvtx.range('Frontier/HANDOFF_ADJOINT_SEND'):
            for buffer in buffers:
                buffer.copy_(grad)
            operations = [dist.P2POp(dist.isend, buffer, peer, self.process_group)
                          for peer, buffer in zip(self.plan.recv_sources, buffers, strict=True)]
            requests, event = self._launch(operations)
        return requests, buffers, event

    def backward_destination(self, slot_id):
        if not self.plan.send_slices or slot_id != 0:
            raise ValueError('backward_destination requires the forward source role')
        if not self._source_pending:
            raise RuntimeError('source adjoint requires a preceding forward contribution')
        buffers = self.buffer_pool.acquire_many(slot_id, len(self.plan.send_slices), self.plan.projected_shape)
        self._direction = 'backward_destination'
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        operations = [dist.P2POp(dist.irecv, buffer, item.peer_rank, self.process_group)
                      for item, buffer in zip(self.plan.send_slices, buffers, strict=True)]
        requests, _ = self._launch(operations)
        return buffers, requests
