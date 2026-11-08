from copy import deepcopy
from dataclasses import dataclass

from contracts import RewriteIntent
from predictor.profile_database import normalized_execution
from search.conditional_partition import REWRITE_FIELDS
from search.family_state import GLOBAL_FIELDS, global_identity


@dataclass(frozen=True)
class JointScope:
    parent_plan_id: str
    family_id: str
    global_values: dict
    fixed_values: dict
    allowed_ranges: dict
    adjustable: tuple
    evidence_refs: tuple


class JointScopeCompiler:
    def compile(self, parent, family, intent):
        intent = RewriteIntent.model_validate(intent)
        values = family['values']
        if set(values) != set(GLOBAL_FIELDS) or global_identity(dict(parent, **values)) != family['family_id']:
            raise ValueError('联合调查的全局家族身份或字段不完整')
        conflicts = [name for name in GLOBAL_FIELDS if name in intent.values and intent.values[name] != values[name]]
        if conflicts:
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=conflicts))
        opened = set(intent.adjustable)
        redundant = {name for name in opened & set(intent.values)
            if name not in intent.allowed_ranges or intent.allowed_ranges[name] == [intent.values[name]]}
        opened -= redundant
        ranges = {name: values for name, values in intent.allowed_ranges.items() if name not in redundant}
        if 'recompute' in opened and 'memory_policy' not in opened:
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['memory_policy'],
                reason='开放重计算需要共同开放对应的模块存储策略'))
        if 'decoder_chunks' not in opened:
            counts = intent.values.get('decoder_chunks', parent['decoder_chunks'])
            virtual = ranges.get('vpp', [intent.values.get('vpp', parent['vpp'])])
            if any(len(counts) != values['pp'] * value for value in virtual):
                raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['decoder_chunks'],
                    reason='改变流水线区间数量需要开放模块分区或提交相容的固定分区'))
        if opened & (set(intent.preserve) | set(intent.values)) or opened & set(GLOBAL_FIELDS):
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['joint_scope_open_and_fixed']))
        if opened - set(REWRITE_FIELDS) or set(ranges) - opened:
            raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['joint_scope_ranges']))
        normalized = normalized_execution(parent)
        if 'execution_backend' not in opened or 'execution_backend' in ranges:
            backends = ranges.get('execution_backend', [
                intent.values.get('execution_backend', normalized['execution_backend'])])
            if ('optimizer_offload_fraction' not in opened
                    or 'optimizer_offload_fraction' in ranges):
                fractions = ranges.get('optimizer_offload_fraction', [
                    intent.values.get('optimizer_offload_fraction', normalized['optimizer_offload_fraction'])])
                if set(backends) == {'megatron_fsdp'} and fractions and all(value > 0 for value in fractions):
                    raise ValueError(dict(status='CONSTRAINT_CONFLICT',
                        conflicts=['execution_backend', 'optimizer_offload_fraction'],
                        reason='当前Megatron后端的优化器CPU卸载只支持DDP分布式优化器；'
                        'FSDP全分片候选必须将优化器CPU卸载比例设为零。'))
        fixed = {}
        derived = {'dp_inner_size', 'dp_outer_size', 'expert_dp_inner_size', 'expert_dp_outer_size'}
        backend_varies = ('execution_backend' in opened or
            intent.values.get('execution_backend', normalized['execution_backend']) != normalized['execution_backend'])
        backend_fields = {'dense_sharding_strategy', 'expert_sharding_strategy',
            'outer_sharding_strategy', 'expert_outer_sharding_strategy'}
        offload_varies = ('optimizer_offload_fraction' in opened or
            intent.values.get('optimizer_offload_fraction', normalized['optimizer_offload_fraction'])
            != normalized['optimizer_offload_fraction'])
        for name in REWRITE_FIELDS:
            if name in GLOBAL_FIELDS or name in opened or name in derived:
                continue
            if name not in intent.values and name not in intent.preserve and (
                    (backend_varies and name in backend_fields) or (offload_varies and name == 'offload')):
                continue
            if (parent.get('execution_backend', 'megatron_ddp') == 'megatron_ddp'
                    and not backend_varies and name not in intent.values
                    and (name.startswith('fsdp_') or name.startswith('megatron_fsdp_'))):
                continue
            if (backend_varies and name not in intent.values and name not in intent.preserve
                    and (name.startswith('fsdp_') or name.startswith('megatron_fsdp_'))):
                continue
            value = intent.values.get(name, parent.get(name, normalized.get(name)))
            if value is not None:
                fixed[name] = deepcopy(value)
        return JointScope(intent.parent_plan_id or parent['run_id'], family['family_id'],
            deepcopy(values), fixed, deepcopy(ranges), tuple(sorted(opened)),
            tuple(intent.evidence_refs))
