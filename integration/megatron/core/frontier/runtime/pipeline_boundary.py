from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryDispatchResult:
    handled: bool
    tensor: object | None


class PipelineBoundaryRuntime:
    def __init__(self, resolve_runtimes=(), *, pp_stage=0, vp_stage=0) -> None:
        self.resolve_runtimes = tuple(resolve_runtimes)
        self.coordinates = pp_stage, vp_stage
        self._live = {}
        self._free = list(range(len(self.resolve_runtimes)))
        if len({id(runtime) for runtime in self.resolve_runtimes}) != len(self.resolve_runtimes):
            raise ValueError('each live microbatch requires a distinct preallocated runtime')
        if self.resolve_runtimes and any(runtime.plan != self.resolve_runtimes[0].plan
                                         for runtime in self.resolve_runtimes):
            raise ValueError('microbatch runtimes must implement the same compiled boundary')

    def _key(self, pp_stage, vp_stage, microbatch_id):
        if (pp_stage, vp_stage) != self.coordinates or microbatch_id < 0:
            raise ValueError('boundary invocation differs from compiled pipeline coordinates')
        return pp_stage, vp_stage, microbatch_id

    def _begin(self, pp_stage, vp_stage, microbatch_id, source):
        key = self._key(pp_stage, vp_stage, microbatch_id)
        if key in self._live:
            raise RuntimeError('microbatch already has a live boundary value')
        if not self._free:
            raise RuntimeError('compiled boundary lifetime capacity is exhausted')
        index = self._free[0]
        runtime = self.resolve_runtimes[index]
        if bool(runtime.plan.send_slices) != source:
            raise ValueError('boundary call differs from the compiled source or destination role')
        self._free.pop(0)
        self._live[key] = index
        return runtime

    def _backward(self, pp_stage, vp_stage, microbatch_id, source):
        key = self._key(pp_stage, vp_stage, microbatch_id)
        if key not in self._live:
            raise RuntimeError('boundary adjoint requires the matching live microbatch')
        runtime = self.resolve_runtimes[self._live[key]]
        if bool(runtime.plan.send_slices) != source:
            raise ValueError('boundary adjoint differs from the compiled rank role')
        return key, runtime

    def _release(self, key):
        self._free.append(self._live.pop(key))

    @staticmethod
    def _shape(runtime, tensor_shape):
        if tuple(tensor_shape) != runtime.plan.local_shape:
            raise ValueError('receive shape differs from compiled boundary ownership')

    def forward_send(self, *, tensor, pp_stage, vp_stage, microbatch_id):
        if not self.resolve_runtimes:
            return BoundaryDispatchResult(False, tensor)
        runtime = self._begin(pp_stage, vp_stage, microbatch_id, True)
        requests, keepalive, event = runtime.forward_source(tensor)
        runtime.finish_source(requests, event)
        return BoundaryDispatchResult(True, None)

    def forward_recv(self, *, pp_stage, vp_stage, microbatch_id, tensor_shape):
        if not self.resolve_runtimes:
            return BoundaryDispatchResult(False, None)
        runtime = self._begin(pp_stage, vp_stage, microbatch_id, False)
        self._shape(runtime, tensor_shape)
        buffers, requests = runtime.forward_destination(0)
        tensor = runtime.finish_destination(buffers, requests, 0)
        return BoundaryDispatchResult(True, tensor.detach().requires_grad_(True))

    def backward_send(self, *, grad, pp_stage, vp_stage, microbatch_id):
        if not self.resolve_runtimes:
            return BoundaryDispatchResult(False, grad)
        key, runtime = self._backward(pp_stage, vp_stage, microbatch_id, False)
        requests, keepalive, event = runtime.backward_source(grad)
        runtime.finish_source(requests, event)
        self._release(key)
        return BoundaryDispatchResult(True, None)

    def backward_recv(self, *, pp_stage, vp_stage, microbatch_id, tensor_shape):
        if not self.resolve_runtimes:
            return BoundaryDispatchResult(False, None)
        key, runtime = self._backward(pp_stage, vp_stage, microbatch_id, True)
        self._shape(runtime, tensor_shape)
        buffers, requests = runtime.backward_destination(0)
        tensor = runtime.finish_destination(buffers, requests, 0)
        self._release(key)
        return BoundaryDispatchResult(True, tensor)
