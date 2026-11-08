from dataclasses import dataclass


@dataclass(frozen=True)
class RecomputeEffect:
    layer_id: int
    policy: str
    modules: tuple
    forward_ms: float | None
    backward_ms: float | None
    saved_activation_bytes: int | None
    evidence_ids: tuple
    cost_status: str
    memory_status: str


def resolve_recompute_effect(request, layer_id, profile=None):
    mode = request['recompute']
    modules = (tuple((request.get('memory_policy') or {}).get('recompute_modules', []))
        if mode == 'selective' else ('mlp',) if mode == 'selective-mlp' else ())
    measured = None if profile is None else profile['layers'].get(str(layer_id))
    return RecomputeEffect(layer_id, mode, modules,
        None if measured is None else measured['forward_ms'],
        None if measured is None else measured['backward_ms'], None,
        () if profile is None else (profile['id'],),
        'NEEDS_PROFILE' if measured is None else 'MEASURED_COMPOSITE_INCLUDING_RECOMPUTATION',
        'NEEDS_SAVED_TENSOR_EVIDENCE')
