from __future__ import annotations

from dataclasses import dataclass, replace

from megatron.core.frontier.ir.contract import ObligationKind
from megatron.core.frontier.ir.boundary import BoundaryProgram


@dataclass(frozen=True)
class MobilityRule:
    region_mode_id: str
    obligation_kind: ObligationKind
    output_obligation_kind: ObligationKind | None
    adjoint_key: str


class MobilityRegistry:
    def __init__(self) -> None:
        self._rules: dict[tuple[str, ObligationKind], MobilityRule] = {}

    def register(self, rule: MobilityRule) -> None:
        key = (rule.region_mode_id, rule.obligation_kind)
        if key in self._rules:
            raise KeyError(f"mobility rule already registered: {key}")
        self._rules[key] = rule

    def get(self, region_mode_id: str, obligation_kind: ObligationKind) -> MobilityRule | None:
        return self._rules.get((region_mode_id, obligation_kind))


def move_resolve_across_region(
    program: BoundaryProgram,
    op_index: int,
    region_mode,
    registry: MobilityRegistry,
) -> tuple[BoundaryProgram, ...]:
    op = program.ops[op_index]
    obligation = op.input_contract.obligation
    if obligation is None:
        return ()
    rule = registry.get(region_mode.mode_id, obligation.kind)
    if rule is None:
        return ()
    moved_ops = list(program.ops)
    moved = moved_ops.pop(op_index)
    insert_at = min(len(moved_ops), op_index + 1)
    moved_ops.insert(insert_at, replace(moved, movable_before_region=region_mode.region_id if hasattr(region_mode, "region_id") else None))
    return (replace(program, ops=tuple(moved_ops)),)
