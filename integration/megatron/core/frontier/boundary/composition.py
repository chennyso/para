from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CompositionRule:
    key: str
    left_impl: str
    right_impl: str
    supports: Callable
    build_fused_op: Callable


class CompositionRegistry:
    def __init__(self) -> None:
        self._rules: list[CompositionRule] = []

    def register(self, rule: CompositionRule) -> None:
        if any(existing.key == rule.key for existing in self._rules):
            raise KeyError(f"composition rule already registered: {rule.key}")
        self._rules.append(rule)

    def match(self, left, right, context) -> tuple[CompositionRule, ...]:
        return tuple(rule for rule in self._rules if rule.left_impl == left.implementation_key and rule.right_impl == right.implementation_key and rule.supports(left, right, context))


def compose_adjacent(program, left_index: int, registry: CompositionRegistry, context):
    if left_index < 0 or left_index + 1 >= len(program.ops):
        raise IndexError("composition index is outside the operation chain")
    left, right = program.ops[left_index:left_index + 2]
    rules = registry.match(left, right, context)
    return tuple(rule.build_fused_op(program, left_index, right_index=left_index + 1) for rule in rules)
