from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from megatron.core.frontier.ir.boundary import BoundaryProgram
from megatron.core.frontier.ir.region import RegionMode


@dataclass(frozen=True)
class BoundaryImplementation:
    key: str
    supports: Callable[[BoundaryProgram], bool]
    compile_runtime: Callable


@dataclass(frozen=True)
class RegionBackendImplementation:
    key: str
    source_revision: str
    scopes: tuple[str, ...]
    supports: Callable
    compile_contract: Callable

    def __post_init__(self):
        if (not self.key or not self.source_revision or not self.scopes
                or not set(self.scopes) <= {'OPERATOR', 'MODEL'}
                or not callable(self.supports) or not callable(self.compile_contract)):
            raise ValueError('Region backends require a source revision, explicit scope and actual compiler callbacks')


class CapabilityCatalog:
    def __init__(self) -> None:
        self._region_modes: dict[str, RegionMode] = {}
        self._boundary_impls: dict[str, BoundaryImplementation] = {}
        self._region_backends: dict[str, RegionBackendImplementation] = {}

    def register_region_mode(self, mode: RegionMode) -> None:
        if mode.mode_id in self._region_modes:
            raise KeyError(f"region mode already registered: {mode.mode_id}")
        self._region_modes[mode.mode_id] = mode

    def register_boundary_impl(self, implementation: BoundaryImplementation) -> None:
        if implementation.key in self._boundary_impls:
            raise KeyError(f"boundary implementation already registered: {implementation.key}")
        self._boundary_impls[implementation.key] = implementation

    def region_mode(self, mode_id: str) -> RegionMode:
        return self._region_modes[mode_id]

    def boundary_impl(self, key: str) -> BoundaryImplementation:
        return self._boundary_impls[key]

    def all_region_modes(self) -> tuple[RegionMode, ...]:
        return tuple(self._region_modes.values())

    def register_region_backend(self, implementation: RegionBackendImplementation):
        if implementation.key in self._region_backends:
            raise KeyError(f'Region backend already registered: {implementation.key}')
        self._region_backends[implementation.key] = implementation

    def has_region_backend(self, key):
        return key in self._region_backends

    def region_backend(self, key):
        return self._region_backends[key]
