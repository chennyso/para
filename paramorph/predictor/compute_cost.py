import math
import statistics

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from contracts import fingerprint
from cost.analytical_model import estimate_hardware_cost


def compute_family(identity):
    family = {key: value for key, value in identity.items() if key != 'features'}
    if isinstance(family.get('layout'), dict):
        family['layout'] = {key: value for key, value in family['layout'].items()
            if key not in ('tp', 'cp', 'etp', 'ep')}
    if identity.get('features'):
        if identity.get('expert_token_counts') is not None and all(f'expert_{index}_tokens' in identity['features']
                for index in range(len(identity['expert_token_counts']))):
            family.pop('expert_token_counts', None)
        family['inputs'] = [dict(dtype=item['dtype'], rank=len(item['shape']),
            contiguous=item.get('contiguous'), strides=item.get('stride') if not item.get('contiguous') else None)
            for item in identity.get('inputs', [])]
        family['parameter_shapes'] = [dict(name=item['name'], dtype=item['dtype'], rank=len(item['shape']))
            for item in identity.get('parameter_shapes', [])]
    if identity.get('granularity') == 'gpu_kernel':
        family.pop('model', None)
    return fingerprint(family)


class KernelCostSurface:
    def __init__(self, database):
        self.database = database
        self.family_cache = {}
        self.model_cache = {}

    def family_costs(self, family_key):
        if family_key not in self.family_cache:
            self.family_cache[family_key] = self.database.operator_family_aggregates(family_key)
        return self.family_cache[family_key]

    def family_models(self, family_key):
        if family_key not in self.model_cache:
            self.model_cache[family_key] = self.database.operator_models(family_key)
        return self.model_cache[family_key]

    def query(self, identity, analytical_spec=None, mode='hybrid', calibration_run_ids=None):
        if mode not in ('hybrid', 'measured_only', 'analytical_only'):
            raise ValueError('计算成本模式必须为hybrid、measured_only或者analytical_only')
        if calibration_run_ids is not None and (
                not isinstance(calibration_run_ids, (list, tuple, set))
                or any(not isinstance(identifier, str) or not identifier for identifier in calibration_run_ids)):
            raise ValueError('独立标定Run身份必须为明确的非空文本集合')
        prefixes = None if calibration_run_ids is None else tuple(identifier + ':' for identifier in calibration_run_ids)
        if mode != 'analytical_only':
            measured = self._measured_query(identity, prefixes)
            if measured is not None:
                return dict(measured, mode=mode)
            if mode == 'measured_only':
                return dict(self.missing(identity, '缺少适用的实测成本或者实测形状插值范围'), mode=mode)
            tiled = self._tile_query(identity, prefixes)
            if tiled is not None:
                return dict(tiled, mode=mode)
        analytical = estimate_hardware_cost(identity, analytical_spec)
        if analytical['status'] != 'NEEDS_PROFILE':
            calibrated = self._efficiency_query(identity, analytical_spec, analytical, prefixes)
            return dict(calibrated if calibrated is not None else analytical, mode=mode)
        return dict(self.missing(identity, analytical['reason']), mode=mode,
                    calibration=analytical['calibration'], assumptions=analytical['assumptions'])

    def _efficiency_query(self, identity, analytical_spec, target_base, prefixes):
        if analytical_spec is None or analytical_spec.get('operation') == 'composite':
            return None
        hardware = analytical_spec.get('hardware', {})
        if hardware.get('throughput_kind') != 'theoretical_peak':
            return None
        features = identity.get('features')
        if not isinstance(features, dict) or not features or any(
                type(value) not in (int, float) or value <= 0 for value in features.values()):
            return None
        rows = [row for row in self.family_costs(compute_family(identity))
            if self._allowed_evidence(row.get('evidence_id'), prefixes)]
        neighbors = []
        for row in rows:
            sample_features = row['identity'].get('features')
            if not isinstance(sample_features, dict) or set(sample_features) != set(features):
                continue
            if any(type(sample_features[name]) not in (int, float) or sample_features[name] <= 0
                    for name in features):
                continue
            distance = max(abs(math.log2(features[name] / sample_features[name])) for name in features)
            if distance > 2:
                continue
            sample_identity = dict(identity, features=sample_features)
            sample_spec = dict(analytical_spec, features=sample_features)
            sample_base = estimate_hardware_cost(sample_identity, sample_spec)
            if sample_base['status'] == 'NEEDS_PROFILE' or sample_base['milliseconds'] <= 0:
                continue
            efficiency = row['milliseconds'] / sample_base['milliseconds']
            if not math.isfinite(efficiency) or efficiency <= 0:
                raise ValueError('实测Kernel效率修正系数必须为有限正数')
            neighbors.append((distance, efficiency, row['evidence_id']))
        if not neighbors:
            return None
        selected = sorted(neighbors)[:8]
        factor = statistics.median(row[1] for row in selected)
        return dict(target_base,
            status='ANALYTICAL_SHAPE_CALIBRATED',
            milliseconds=target_base['milliseconds'] * factor,
            source='shape_conditioned_measured_efficiency_correction',
            calibration=dict(kind='neighbor_efficiency_ratio', factor=factor,
                maximum_log2_shape_distance=max(row[0] for row in selected),
                evidence_ids=[row[2] for row in selected]),
            assumptions=list(dict.fromkeys(target_base.get('assumptions', []) +
                ['效率修正仅使用相同GPU、Kernel、精度、Layout和阶段的成本家族',
                 '相邻形状的每个维度比例不超过四倍'])))

    @staticmethod
    def _allowed_evidence(identifier, prefixes):
        return prefixes is None or (isinstance(identifier, str) and identifier.startswith(prefixes))

    def _measured_query(self, identity, prefixes=None):
        exact = [row for row in self.database.operator_costs(identity)
                 if self._allowed_evidence(row.get('evidence_id'), prefixes)]
        if exact:
            return self.aggregate(exact, 'MEASURED_EXACT', identity)
        features = identity.get('features')
        if not features:
            return None
        names = sorted(features)
        if any(type(features[name]) not in (int, float) or not math.isfinite(features[name]) or features[name] < 0 for name in names):
            raise ValueError('计算成本的形状特征必须为有限非负数值')
        samples = [row for row in self.family_costs(compute_family(identity))
                   if self._allowed_evidence(row.get('evidence_id'), prefixes)]
        grouped = {}
        for row in samples:
            values = row['identity'].get('features', {})
            if set(values) != set(features):
                continue
            key = tuple(values[name] for name in names)
            grouped.setdefault(key, []).append(row)
        if len(grouped) > 1:
            points = np.asarray(list(grouped), dtype=float)
            target = np.asarray([features[name] for name in names], dtype=float)
            spread = np.ptp(points, axis=0)
            active = spread > 0
            fixed_match = np.all(points[0, ~active] == target[~active])
            inside_bounds = np.all(target >= points.min(axis=0)) and np.all(target <= points.max(axis=0))
            if fixed_match and inside_bounds and np.any(active):
                coordinates = (points[:, active] - points[:, active].min(axis=0)) / spread[active]
                desired = (target[active] - points[:, active].min(axis=0)) / spread[active]
                durations = np.asarray([statistics.median(row['milliseconds'] for row in rows) for rows in grouped.values()])
                basis_points = coordinates - coordinates[0]
                dimensions = int(np.linalg.matrix_rank(basis_points))
                _, _, axes = np.linalg.svd(basis_points, full_matrices=False)
                basis = axes[:dimensions].T
                displacement = desired - coordinates[0]
                if not np.allclose(displacement, displacement @ basis @ basis.T, rtol=1e-8, atol=1e-10):
                    return None
                desired = displacement @ basis
                coordinates = basis_points @ basis
                prediction = None
                if dimensions == 1:
                    order = np.argsort(coordinates[:, 0])
                    prediction = float(np.interp(desired[0], coordinates[order, 0], durations[order]))
                elif len(coordinates) >= dimensions + 1 and np.linalg.matrix_rank(coordinates - coordinates[0]) == dimensions:
                    hull = Delaunay(coordinates)
                    if hull.find_simplex(desired) >= 0:
                        prediction = float(LinearNDInterpolator(hull, durations)(desired).item())
                if prediction is not None:
                    if not math.isfinite(prediction) or prediction < 0:
                        raise ValueError('计算成本插值产生非法时延')
                    return dict(status='INTERPOLATED', milliseconds=prediction, identity=identity,
                        cost_scope=identity.get('granularity', 'module_composite'),
                        source='within_measured_shape_convex_hull',
                        calibration=dict(kind='measured_local_interpolation', family_key=compute_family(identity),
                            shape_feature_names=names, shape_min=points.min(axis=0).tolist(),
                            shape_max=points.max(axis=0).tolist()),
                        evidence_ids=[row['evidence_id'] for rows in grouped.values() for row in rows],
                        assumptions=['相同GPU、Kernel实现、精度和Layout条件内的分段形状插值'])
        return None

    def _tile_query(self, identity, prefixes=None):
        features = identity.get('features')
        if not features:
            return None
        for model in self.family_models(compute_family(identity)):
            if prefixes is not None and (not model.get('evidence_refs') or any(
                    not self._allowed_evidence(identifier, prefixes) for identifier in model['evidence_refs'])):
                continue
            if model['model_kind'] != 'gemm_tile' or identity.get('granularity') != 'gpu_kernel':
                continue
            if set(features) != {'M', 'N', 'K'}:
                continue
            if any(not model['shape_bounds'][name][0] <= features[name] <= model['shape_bounds'][name][1] for name in features):
                continue
            tile = model['kernel_tile']
            count = math.ceil(features['M'] / tile['B_M']) * math.ceil(features['N'] / tile['B_N'])
            waves = math.ceil(count / model['concurrent_tiles'])
            steps = math.ceil(features['K'] / tile['B_K'])
            duration = model['launch_ms'] + waves * steps * model['wave_k_step_ms']
            if not math.isfinite(duration) or duration < 0:
                raise ValueError('标定后的Kernel Tile模型产生非法时延')
            return dict(status='ANALYTICAL_CALIBRATED_TILE', milliseconds=duration, identity=identity,
                cost_scope='gpu_kernel', tile_count=count, waves=waves, k_steps=steps,
                evidence_ids=model['evidence_refs'], model_id=model['model_id'],
                source='calibrated_kernel_tile_model',
                calibration=dict(kind='calibrated_tile', kernel_configuration_source=model['kernel_configuration_source'],
                                 shape_bounds=model['shape_bounds'], evidence_ids=model['evidence_refs']),
                assumptions=['使用已登记Kernel Tile配置、并发Tile数量与标定后的波次参数'])
        return None

    @staticmethod
    def aggregate(rows, status, identity):
        values = [row['milliseconds'] for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('计算成本缺少合法的真实计时')
        return dict(status=status, milliseconds=statistics.median(values), sample_count=len(values),
            observed_dispersion_ms=statistics.pstdev(values), identity=identity,
            cost_scope=identity.get('granularity', 'module_composite'), evidence_ids=[row['evidence_id'] for row in rows],
            source='actual_operator_cost_records',
            calibration=dict(kind='measured_exact', evidence_ids=[row['evidence_id'] for row in rows]),
            assumptions=['仅复用完全相同执行身份的计时，成本范围保持测量记录的原始粒度'])

    @staticmethod
    def missing(identity, reason):
        return dict(status='NEEDS_PROFILE', milliseconds=None, identity=identity, reason=reason,
            cost_scope=identity.get('granularity', 'module_composite'), source='missing_applicable_compute_cost',
            calibration=None, assumptions=[],
            profiling_request=dict(kind='operator_cost', cost_key=fingerprint(identity), identity=identity))
