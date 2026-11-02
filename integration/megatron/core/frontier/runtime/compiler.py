from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledBoundary:
    boundary_id: int
    implementation_keys: tuple[str, ...]
    buffer_slot: int
    runtime_descriptors: tuple = ()


@dataclass(frozen=True)
class CompiledProgram:
    program: object
    boundaries: tuple[CompiledBoundary, ...]


class FrontierCompiler:
    def __init__(self, capability_catalog, buffer_assignment: dict[int, int] | None = None) -> None:
        self.capability_catalog = capability_catalog
        self.buffer_assignment = buffer_assignment or {}

    def compile(self, program) -> CompiledProgram:
        compiled = []
        for boundary in program.boundaries:
            witness = boundary.executable_witness
            if witness is None or boundary.executable_witness_key != witness.identifier():
                raise ValueError("boundary compilation requires a complete executable witness")
            descriptors = []
            for op in boundary.ops:
                implementation = self.capability_catalog.boundary_impl(op.implementation_key)
                if not implementation.supports(boundary):
                    raise ValueError(f"unsupported boundary operation: {op.implementation_key}")
                descriptor = implementation.compile_runtime(boundary)
                if descriptor is None or not (callable(getattr(descriptor, "instantiate", None))
                                               or callable(getattr(descriptor, "forward", None))):
                    raise ValueError("boundary capability did not produce an executable runtime descriptor")
                descriptors.append(descriptor)
            compiled.append(CompiledBoundary(boundary.boundary_id, tuple(op.implementation_key for op in boundary.ops),
                                              self.buffer_assignment.get(boundary.boundary_id, boundary.boundary_id),
                                              tuple(descriptors)))
        return CompiledProgram(program, tuple(compiled))
