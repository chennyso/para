from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class _RankMetrics:
    variant: str
    calls: int = 0
    custom_calls: int = 0
    sent_bytes: int = 0
    received_bytes: int = 0
    elapsed_ms: float = 0.0
    by_direction_ms: dict[str, float] = field(default_factory=dict)

    def add(self, direction: str, elapsed_ms: float, sent_bytes: int, received_bytes: int, custom: bool) -> None:
        self.calls += 1
        self.custom_calls += int(custom)
        self.sent_bytes += sent_bytes
        self.received_bytes += received_bytes
        self.elapsed_ms += elapsed_ms
        self.by_direction_ms[direction] = self.by_direction_ms.get(direction, 0.0) + elapsed_ms


class FrontierP2PAdapter:
    def __init__(self, communicator) -> None:
        self.communicator = communicator
        self.variant = os.environ.get("FRONTIER_VARIANT", "canonical")
        self.metrics = _RankMetrics(self.variant)
        # Megatron attaches the received tensor to the current autograd graph.
        # Reusing a tensor across microbatches would connect unrelated graphs.
        self._stream = None

    def communicate(self, original, *, tensor_send_next, tensor_send_prev, recv_prev, recv_next, tensor_shape, wait_on_reqs=True):
        if self.variant in {"canonical", "disabled"} or os.environ.get("FRONTIER_CUSTOM_P2P", "0") != "1":
            started = time.perf_counter()
            result = original(
                tensor_send_next=tensor_send_next,
                tensor_send_prev=tensor_send_prev,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                wait_on_reqs=wait_on_reqs,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            received_bytes = sum(self._tensor_bytes(tensor) for tensor in result[:2] if tensor is not None)
            direction = self._direction(tensor_send_next, tensor_send_prev, recv_prev, recv_next)
            self.metrics.add(direction, elapsed_ms, 0, received_bytes, custom=False)
            return result
        if self.communicator.config.variable_seq_lengths or self.communicator.config.mtp_standalone:
            return original(
                tensor_send_next=tensor_send_next,
                tensor_send_prev=tensor_send_prev,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                wait_on_reqs=wait_on_reqs,
            )

        started = time.perf_counter()
        result = self._issue(
            tensor_send_next,
            tensor_send_prev,
            recv_prev,
            recv_next,
            tensor_shape,
            wait_on_reqs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sent_bytes = sum(self._tensor_bytes(tensor) for tensor in (tensor_send_next, tensor_send_prev))
        received_bytes = sum(self._tensor_bytes(tensor) for tensor in result[:2] if tensor is not None)
        direction = self._direction(tensor_send_next, tensor_send_prev, recv_prev, recv_next)
        self.metrics.add(direction, elapsed_ms, sent_bytes, received_bytes, custom=True)
        return result

    def _issue(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next, tensor_shape, wait_on_reqs):
        recv_prev_tensor = self._receive_buffer(tensor_shape, "prev") if recv_prev else None
        recv_next_tensor = self._receive_buffer(tensor_shape, "next") if recv_next else None
        operations = []
        if tensor_send_next is not None:
            operations.append(dist.P2POp(dist.isend, tensor_send_next.contiguous(), self.communicator.next_rank, self.communicator.pp_group))
        if recv_prev_tensor is not None:
            operations.append(dist.P2POp(dist.irecv, recv_prev_tensor, self.communicator.prev_rank, self.communicator.pp_group))
        if tensor_send_prev is not None:
            operations.append(dist.P2POp(dist.isend, tensor_send_prev.contiguous(), self.communicator.prev_rank, self.communicator.pp_group))
        if recv_next_tensor is not None:
            operations.append(dist.P2POp(dist.irecv, recv_next_tensor, self.communicator.next_rank, self.communicator.pp_group))
        requests = dist.batch_isend_irecv(operations) if operations else []
        if wait_on_reqs:
            for request in requests:
                request.wait()
            requests = []
        return recv_prev_tensor, recv_next_tensor, requests

    def _receive_buffer(self, shape, side):
        normalized = tuple(int(value) for value in shape)
        dtype = self.communicator.config.pipeline_dtype
        return torch.empty(
            normalized,
            device=torch.cuda.current_device(),
            dtype=dtype,
            requires_grad=True,
        )

    @staticmethod
    def _tensor_bytes(tensor) -> int:
        return 0 if tensor is None else int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _direction(tensor_send_next, tensor_send_prev, recv_prev, recv_next) -> str:
        return "+".join(
            name
            for name, enabled in (
                ("send_next", tensor_send_next is not None),
                ("send_prev", tensor_send_prev is not None),
                ("recv_prev", recv_prev),
                ("recv_next", recv_next),
            )
            if enabled
        )

    def flush(self) -> None:
        output_dir = os.environ.get("FRONTIER_METRICS_DIR")
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        path = os.path.join(output_dir, f"p2p-rank-{rank}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.metrics.__dict__, handle, sort_keys=True, indent=2)


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def install_p2p_adapter() -> None:
    if os.environ.get("FRONTIER_RUNTIME", "0") != "1":
        return
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    if getattr(P2PCommunicator, "_frontier_adapter_installed", False):
        return
    original_init = P2PCommunicator.__init__
    original_communicate = P2PCommunicator._communicate

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._frontier_adapter = FrontierP2PAdapter(self)
        atexit.register(self._frontier_adapter.flush)

    def communicate(self, **kwargs):
        adapter = getattr(self, "_frontier_adapter", None)
        if adapter is None:
            return original_communicate(self, **kwargs)
        return adapter.communicate(original_communicate.__get__(self, P2PCommunicator), **kwargs)

    P2PCommunicator.__init__ = init
    P2PCommunicator._communicate = communicate
    P2PCommunicator._frontier_adapter_installed = True
