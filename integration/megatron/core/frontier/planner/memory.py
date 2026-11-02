from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lifetime:
    value_id: str
    start_event: int
    end_event: int
    bytes_required: int

    def __post_init__(self) -> None:
        if self.end_event < self.start_event or self.bytes_required < 0:
            raise ValueError("invalid memory lifetime")


@dataclass
class MemorySlot:
    slot_id: int
    capacity_bytes: int
    last_end_event: int


def color_lifetimes(lifetimes: tuple[Lifetime, ...]):
    ordered = sorted(lifetimes, key=lambda item: (item.start_event, item.end_event, item.value_id))
    slots: list[MemorySlot] = []
    assignment: dict[str, int] = {}
    for lifetime in ordered:
        selected = next((slot for slot in slots if slot.last_end_event < lifetime.start_event and slot.capacity_bytes >= lifetime.bytes_required), None)
        if selected is None:
            selected = MemorySlot(len(slots), lifetime.bytes_required, lifetime.end_event)
            slots.append(selected)
        else:
            selected.last_end_event = lifetime.end_event
        assignment[lifetime.value_id] = selected.slot_id
    return assignment, tuple(slots)
