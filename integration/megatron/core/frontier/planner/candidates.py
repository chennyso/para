from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainCandidate:
    start_region: int
    end_region: int
    policy_key: str
    mode_ids: tuple[str, ...]
    local_forward_us: float
    local_backward_us: float
    memory_lower_bound: int
    input_signature: tuple
    output_signature: tuple
    left_boundary_lb_us: float = 0.0
    right_boundary_lb_us: float = 0.0

    @property
    def lower_bound_us(self) -> float:
        return self.local_forward_us + self.local_backward_us + self.left_boundary_lb_us + self.right_boundary_lb_us
