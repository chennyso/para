from __future__ import annotations


class FeasibilityChecker:
    def __init__(self, memory_limit_bytes: int, memory_reserve_bytes: int, capability_catalog) -> None:
        if memory_limit_bytes <= memory_reserve_bytes:
            raise ValueError("memory limit must exceed reserved memory")
        self.memory_limit_bytes = memory_limit_bytes
        self.memory_reserve_bytes = memory_reserve_bytes
        self.capability_catalog = capability_catalog

    def region_mode_feasible(self, mode, shape_context) -> bool:
        lower_bound = self._memory_lower_bound(mode, shape_context)
        return lower_bound <= self.memory_limit_bytes - self.memory_reserve_bytes and bool(mode.implementation_key)

    def _memory_lower_bound(self, mode, shape_context) -> int:
        if callable(shape_context):
            return int(shape_context(mode))
        if isinstance(shape_context, int):
            return shape_context
        return int(shape_context.get(mode.mode_id, 0))
