from __future__ import annotations

from megatron.core.frontier.ir.boundary import BoundaryOpKind


class BoundaryVerifier:
    def __init__(self, semantic_registry, adjoint_registry, capability_catalog) -> None:
        self.semantic_registry = semantic_registry
        self.adjoint_registry = adjoint_registry
        self.capability_catalog = capability_catalog

    def verify(self, program) -> bool:
        return self._verify_contract_chain(program) and self._verify_adjoint_chain(program) and self._verify_executable(program)

    def _verify_contract_chain(self, program) -> bool:
        if not program.ops or program.ops[-1].kind not in {BoundaryOpKind.TRANSFER, BoundaryOpKind.FUSED}:
            return False
        current = program.source_contract
        for op in program.ops:
            if op.input_contract != current:
                return False
            current = op.output_contract
        return current == program.destination_contract

    def _verify_adjoint_chain(self, program) -> bool:
        return all(self.adjoint_registry.has(op.implementation_key) for op in program.ops)

    def _verify_executable(self, program) -> bool:
        witness = program.executable_witness
        if witness is None or program.executable_witness_key != witness.identifier():
            return False
        for op in program.ops:
            implementation = self.capability_catalog.boundary_impl(op.implementation_key)
            if not implementation.supports(program):
                return False
        return True
