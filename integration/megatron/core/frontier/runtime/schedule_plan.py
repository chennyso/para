from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierLayerSchedulePlan:
    logical_layer_id: int
    pre_boundary: object | None = None
    post_boundary: object | None = None


@dataclass(frozen=True)
class FrontierModelChunkSchedulePlan:
    layers: tuple[FrontierLayerSchedulePlan, ...]
    compiled_program: object | None = None

    def layer(self, logical_layer_id: int) -> FrontierLayerSchedulePlan:
        return self.layers[logical_layer_id]
