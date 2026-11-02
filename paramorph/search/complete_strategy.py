from copy import deepcopy
from pydantic import TypeAdapter

from contracts import RewriteIntent, fingerprint
from search.conditional_partition import REWRITE_FIELDS
from search.family_state import global_identity
from core_data_models import MemoryPolicy


COMPLETE_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp', 'micro_batch',
                   'group', 'decoder_chunks', 'recompute', 'pipeline_schedule',
                   'overlap_p2p', 'rank_mapping', 'execution_backend', 'optimizer_sharding')


def compile_complete_strategy(rewriter, parent, intent):
    intent = RewriteIntent.model_validate(intent)
    if intent.allowed_ranges or intent.module_relations or intent.adjustable:
        raise ValueError('完整策略提交需要确定取值')
    unsupported = set(intent.values) - set(REWRITE_FIELDS)
    if unsupported:
        raise ValueError({'unsupported_strategy_fields': sorted(unsupported)})
    missing = [name for name in COMPLETE_FIELDS if name not in intent.values]
    if missing:
        raise ValueError({'missing_complete_fields': missing})
    submitted = deepcopy(parent)
    submitted.update(intent.values)
    # 原生候选验证检查全部维度、布局、微批和后端条件。
    try:
        rewriter.candidate(submitted)
    except ValueError as error:
        raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['backend_validation'],
            reason=error.args[0], revision_owner='JointStrategyAgent')) from error
    rewrite = rewriter.complete_and_rewrite(parent, intent, limit=1, allow_unchanged=True)[0]
    def canonical_value(name, value):
        if name == 'memory_policy':
            adapter = TypeAdapter(MemoryPolicy)
            return adapter.dump_python(adapter.validate_python(value))
        return value

    changed = {name: dict(submitted=value, compiled=rewrite['request'][name])
        for name, value in intent.values.items()
        if canonical_value(name, rewrite['request'][name]) != canonical_value(name, value)}
    if changed:
        raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=['compiler_normalization'],
            changed_fields=changed, revision_owner='JointStrategyAgent',
            reason='完整策略必须提交原生编译后保持一致的取值'))
    rewrite['proposal_identity'] = fingerprint(intent.model_dump(mode='json'))
    rewrite['proposal_kind'] = 'complete_strategy'
    rewrite['global_family_id'] = global_identity(rewrite['request'])
    rewrite['family_search_scope'] = 'unrestricted_independent_continuation'
    return rewrite


def complete_values(request):
    missing = [name for name in COMPLETE_FIELDS if name not in request]
    if missing:
        raise ValueError({'missing_complete_fields': missing})
    return {name: deepcopy(value) for name, value in request.items()
            if name in REWRITE_FIELDS and name != 'offload'}


def family_continuation_intent(request, evidence_ref):
    # 完整候选只提供一个可评价点，继续分区时开放全部原生布局。
    return RewriteIntent(operation='JointTransform',
        values={}, adjustable=[], preserve=[], allowed_ranges={}, module_relations=[],
        hypothesis='在该全局条件下继续完整条件分区，保留候选域',
        evidence_refs=[evidence_ref], parent_plan_id=request['run_id'])
