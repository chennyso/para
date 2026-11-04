import hashlib
import json
import math

from .profile_database import signature
from cost.decomposition import assess_reuse


def profile_key(profile, operation, rank, vp, layer_ids):
    identity = dict(execution=signature(profile['request']), hardware=profile['hardware'][rank],
        backend=profile['source_hashes'], operation=operation, rank=rank, vp=vp,
        evidence=profile['evidence_files'],
        layer_ids=layer_ids,
        shapes=[profile['layer_signatures'][str(layer)] for layer in layer_ids]
            if 'layer_signatures' in profile else None)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def graph_delta(parent, candidate):
    execution_equal = signature(parent) == signature(candidate)
    if not execution_equal:
        assessment = assess_reuse(parent, candidate)
        reuse = assessment.reuse_layer_costs and not assessment.requires_new_profiling
        return dict(
            reusable_template=reuse,
            reuse_scope='atomic_layer_and_link_costs' if reuse else 'none',
            reuse_layer_costs=assessment.reuse_layer_costs,
            reuse_link_costs=assessment.reuse_link_costs,
            reuse_schedule=not assessment.rebuild_schedule,
            rebuild_schedule=assessment.rebuild_schedule,
            requires_new_profiling=assessment.requires_new_profiling,
            reuse_assessment=assessment.as_dict(),
            changed_execution_fields=sorted(key for key in set(parent) | set(candidate)
                if parent.get(key) != candidate.get(key)),
            changed_regions=None,
            evidence_scope='structural_cost_reuse_classification_requires_matching_real_measurement_conditions',
        )
    previous, current = [], []
    offset = 0
    for count in parent['decoder_chunks']:
        previous.append(tuple(range(offset, offset + count)))
        offset += count
    offset = 0
    for count in candidate['decoder_chunks']:
        current.append(tuple(range(offset, offset + count)))
        offset += count
    if len(previous) != len(current) or sum(parent['decoder_chunks']) != sum(candidate['decoder_chunks']):
        raise ValueError('相同执行模板需要完整且一致的模型层集合')
    changed = [index for index, (left, right) in enumerate(zip(previous, current)) if left != right]
    return dict(
        reusable_template=True,
        reuse_scope='identical_execution',
        reuse_layer_costs=True,
        reuse_link_costs=True,
        reuse_schedule=not bool(changed),
        rebuild_schedule=bool(changed),
        requires_new_profiling=False,
        reuse_assessment=None,
        changed_execution_fields=['decoder_chunks'] if changed else [],
        changed_regions=changed,
        reused_regions=[index for index in range(len(previous)) if index not in changed],
        changes=[dict(region=index, previous_layers=previous[index], candidate_layers=current[index])
                 for index in changed],
        evidence_scope='same_execution_signature_atomic_cost_reuse_only',
    )


def anchor_score(parent, candidate, target, baseline_throughput, measured_throughput,
                 coupling_penalty=0.0, epsilon=1e-12):
    dimensions = ('tp', 'pp', 'cp', 'ep', 'dp', 'vpp', 'micro_batch')
    values = [baseline_throughput, measured_throughput, coupling_penalty, epsilon]
    values.extend(parent[key] for key in dimensions)
    values.extend(candidate[key] for key in dimensions)
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Anchor评分需要有限的实测数值')
    if target not in dimensions or coupling_penalty < 0 or epsilon <= 0:
        raise ValueError('Anchor评分参数无效')
    changes = {}
    for key in dimensions:
        if parent[key] <= 0 or candidate[key] <= 0:
            raise ValueError('Anchor并行配置必须为正数')
        changes[key] = math.log2(candidate[key] / parent[key])
    if changes[target] == 0 or min(baseline_throughput, measured_throughput) <= 0:
        raise ValueError('Anchor评分需要目标参数变化及真实正吞吐')
    return dict(score=(measured_throughput - baseline_throughput) / (abs(changes[target]) + epsilon)
        - coupling_penalty * sum(abs(value) for key, value in changes.items() if key != target),
        log2_changes=changes, scope='measured_constrained_joint_change')
