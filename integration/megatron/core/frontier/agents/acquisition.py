import torch


def select_trials(features, measured_features, predicted_real_us, residual_std_us,
                  best_measured_us, gpu_seconds, beta=2.0, diversity_weight=0.1, minimum_distance=0.1):
    x = torch.as_tensor(features, dtype=torch.float64)
    known = torch.as_tensor(measured_features, dtype=torch.float64)
    mean = torch.as_tensor(predicted_real_us, dtype=torch.float64)
    deviation = torch.as_tensor(residual_std_us, dtype=torch.float64)
    cost = torch.as_tensor(gpu_seconds, dtype=torch.float64)
    if x.ndim != 2 or known.ndim != 2 or x.shape[1] != known.shape[1] or len(known) < 1:
        raise ValueError('acquisition requires compatible candidate and actual measured features')
    if mean.shape != (len(x),) or deviation.shape != mean.shape or cost.shape != mean.shape:
        raise ValueError('acquisition vectors must match the candidate count')
    if not all(torch.isfinite(value).all() for value in (x, known, mean, deviation, cost)):
        raise ValueError('acquisition inputs must be finite')
    if (mean <= 0).any() or (cost <= 0).any() or (deviation < 0).any():
        raise ValueError('invalid predicted duration, uncertainty or GPU cost')
    distance = torch.cdist(x, known).amin(1)
    eligible = distance > 1e-12
    if not eligible.any():
        return (), {'distance': distance.tolist()}
    exploitation_scores = mean.clone()
    exploitation_scores[~eligible] = torch.inf
    exploitation = int(exploitation_scores.argmin())
    lcb = mean - beta * deviation
    utility = (best_measured_us - lcb) / cost + diversity_weight * distance
    distinct = eligible & (distance >= minimum_distance)
    distinct[exploitation] = False
    selected = [exploitation]
    if distinct.any():
        scores = utility.clone()
        scores[~distinct] = -torch.inf
        selected.append(int(scores.argmax()))
    return tuple(selected), {'lcb': lcb.tolist(), 'utility': utility.tolist(), 'distance': distance.tolist()}
