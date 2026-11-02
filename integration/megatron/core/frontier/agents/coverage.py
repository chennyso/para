import torch


def select_coverage_seeds(features, required_indices, budget=6):
    values = torch.as_tensor(features, dtype=torch.float64)
    if values.ndim != 2 or len(values) < 1 or not torch.isfinite(values).all():
        raise ValueError('coverage requires finite structural features for legal programs')
    if budget < len(required_indices) or budget > 6 or len(set(required_indices)) != len(required_indices):
        raise ValueError('coverage budget must include the distinct mandatory legal candidates')
    if any(index < 0 or index >= len(values) for index in required_indices):
        raise ValueError('mandatory coverage candidate is outside the legal space')
    selected = list(required_indices)
    if not selected:
        selected.append(0)
    scale = (values.amax(0) - values.amin(0)).clamp_min(1.0)
    normalized = values / scale
    while len(selected) < min(budget, len(values)):
        distances = torch.cdist(normalized, normalized[selected]).amin(1)
        distances[selected] = -1
        selected.append(int(distances.argmax()))
    return tuple(selected)
