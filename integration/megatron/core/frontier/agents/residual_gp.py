import torch


class ResidualGaussianProcess:
    def __init__(self, noise=1e-4, length_scale=1.0, amplitude=1.0):
        if min(noise, length_scale, amplitude) <= 0:
            raise ValueError('GP noise, length scale and amplitude must be positive')
        self.noise = float(noise)
        self.length_scale = float(length_scale)
        self.amplitude = float(amplitude)
        self.features = None

    def _kernel(self, left, right):
        distance = torch.cdist(left / self.length_scale, right / self.length_scale).square()
        return self.amplitude**2 * torch.exp(-0.5 * distance)

    def fit(self, features, simulated_us, measured_us):
        x = torch.as_tensor(features, dtype=torch.float64)
        predicted = torch.as_tensor(simulated_us, dtype=torch.float64)
        actual = torch.as_tensor(measured_us, dtype=torch.float64)
        if x.ndim != 2 or not 1 <= x.shape[0] <= 20 or predicted.shape != actual.shape or actual.shape != (x.shape[0],):
            raise ValueError('GP requires one to twenty complete measured program records')
        if not all(torch.isfinite(value).all() for value in (x, predicted, actual)):
            raise ValueError('GP training records must be finite')
        if torch.unique(x, dim=0).shape[0] != x.shape[0]:
            raise ValueError('duplicate structural feature vectors must be combined before GP fitting')
        self.mean = x.mean(0)
        self.scale = x.std(0, correction=0).clamp_min(1e-12)
        self.features = (x - self.mean) / self.scale
        residual = actual - predicted
        self.residual_mean = residual.mean()
        self.residual_scale = residual.std(correction=0).clamp_min(1.0)
        target = (residual - self.residual_mean) / self.residual_scale
        covariance = self._kernel(self.features, self.features)
        covariance += torch.eye(len(x), dtype=torch.float64) * self.noise
        self.factor = torch.linalg.cholesky(covariance)
        self.alpha = torch.cholesky_solve(target[:, None], self.factor)

    def predict(self, features):
        if self.features is None:
            raise RuntimeError('residual GP requires measured training records')
        x = torch.as_tensor(features, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.features.shape[1] or not torch.isfinite(x).all():
            raise ValueError('GP candidate features differ from the fitted feature contract')
        normalized = (x - self.mean) / self.scale
        cross = self._kernel(normalized, self.features)
        posterior = (cross @ self.alpha).squeeze(1)
        projection = torch.linalg.solve_triangular(self.factor, cross.T, upper=False)
        variance = self.amplitude**2 - projection.square().sum(0)
        if (variance < -1e-8).any():
            raise ValueError('GP posterior covariance is numerically invalid')
        return posterior * self.residual_scale + self.residual_mean, variance.clamp_min(0).sqrt() * self.residual_scale
