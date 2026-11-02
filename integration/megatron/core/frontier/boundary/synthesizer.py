from __future__ import annotations

from collections import deque

from megatron.core.frontier.boundary.composition import compose_adjacent
from megatron.core.frontier.boundary.demand import can_factorize_reduce, factorize_reduce
from megatron.core.frontier.boundary.mobility import move_resolve_across_region


class BoundarySynthesizer:
    def __init__(self, verifier, mobility_registry, composition_registry, max_candidates: int = 5):
        self.verifier = verifier
        self.mobility_registry = mobility_registry
        self.composition_registry = composition_registry
        self.max_candidates = max_candidates

    def synthesize(self, canonical, region_path, consumer_demand, context) -> tuple:
        queue = deque([canonical])
        visited = set()
        valid = []
        while queue:
            program = queue.popleft()
            signature = program.signature()
            if signature in visited:
                continue
            visited.add(signature)
            if self.verifier.verify(program):
                valid.append(program)
            for index, op in enumerate(program.ops):
                for mode in region_path:
                    queue.extend(move_resolve_across_region(program, index, mode, self.mobility_registry))
                if op.kind.name == "RESOLVE" and consumer_demand is not None and can_factorize_reduce(op, consumer_demand):
                    queue.append(factorize_reduce(program, index, consumer_demand))
            for index in range(len(program.ops) - 1):
                queue.extend(compose_adjacent(program, index, self.composition_registry, context))
        valid.sort(key=lambda item: (item.latency_lower_bound_us, item.scratch_bytes, len(item.ops)))
        return tuple(valid[:self.max_candidates])
