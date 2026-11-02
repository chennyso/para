from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BufferSlot:
    tensors: list[torch.Tensor]
    in_use: bool = False


class BoundaryBufferPool:
    def __init__(self) -> None:
        self._slots: dict[int, BufferSlot] = {}

    def register_slot(self, slot_id: int, count: int, shape: tuple[int, ...], dtype, device) -> None:
        if slot_id in self._slots:
            raise KeyError(f"buffer slot already registered: {slot_id}")
        if count < 1:
            raise ValueError("buffer slot count must be positive")
        tensors = [torch.empty(shape, dtype=dtype, device=device) for _ in range(count)]
        self._slots[slot_id] = BufferSlot(tensors=tensors)

    def acquire_many(self, slot_id: int, count: int, shape: tuple[int, ...]) -> list[torch.Tensor]:
        slot = self._slots[slot_id]
        if slot.in_use:
            raise RuntimeError(f"buffer slot is already in use: {slot_id}")
        if count < 1 or len(slot.tensors) < count:
            raise ValueError("requested buffer count exceeds slot capacity")
        if tuple(slot.tensors[0].shape) != shape:
            raise ValueError("requested buffer shape differs from registered shape")
        slot.in_use = True
        return slot.tensors[:count]

    def release(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        if not slot.in_use:
            raise RuntimeError(f"buffer slot is not in use: {slot_id}")
        slot.in_use = False

    def is_in_use(self, slot_id: int) -> bool:
        return self._slots[slot_id].in_use

    def specification(self, slot_id: int):
        slot = self._slots[slot_id]
        tensor = slot.tensors[0]
        return len(slot.tensors), tuple(tensor.shape), tensor.dtype, tensor.device
