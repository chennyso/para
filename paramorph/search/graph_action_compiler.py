from pydantic import Field, StrictInt

from contracts import Contract
from predictor.profile_database import effective_execution_config


class GraphGuidedIntent(Contract):
    source_impact_id: str | None = Field(default=None,min_length=1,max_length=256,
        description='当前父策略DAG影响分析的完整身份，由当前Schema绑定')
    target_task_ids: list[str] = Field(min_length=1, max_length=24)
    affected_regions: list[StrictInt] = Field(default_factory=list, max_length=32)
    target_resources: list[str] = Field(default_factory=list, max_length=24)
    movable_boundaries: list[StrictInt] = Field(default_factory=list, max_length=32,
        description='当前DAG中虚拟阶段槽位之间的边界索引，取值从1至PP乘VPP减1；交由完整分区搜索时使用空列表')
    evidence_refs: list[str] = Field(min_length=1)
    expected_effects: str = Field(min_length=1,max_length=180)


class GraphActionCompiler:
    def __init__(self, impact):
        self.impact = impact

    def compile(self, intent, reference, shape, parallel_ranges, runtime_ranges):
        if intent.source_impact_id is not None and intent.source_impact_id!=self.impact['impact_id']:
            raise ValueError('联合意图的DAG影响分析身份与当前父策略不一致')
        if intent.source_impact_id!=self.impact['impact_id'] and self.impact['impact_id'] not in intent.evidence_refs:
            raise ValueError('联合意图必须引用当前任务影响分析')
        known = {row['task_id']: row for row in self.impact['tasks']}
        unknown = sorted(set(intent.target_task_ids) - known.keys())
        if unknown:
            raise ValueError(dict(code='DAG_TARGET_EVIDENCE_MISSING',
                graph_id=self.impact['graph_id'], impact_id=self.impact['impact_id'],
                invalid_task_ids=unknown, available_task_ids=list(known),
                reason='联合意图引用了当前执行证据中不存在的任务'))
        records = [known[key] for key in intent.target_task_ids]
        regions = sorted({row['region'] for row in records if type(row['region']) is int})
        resources = sorted({resource for row in records for resource in row['resources']})
        if not set(intent.affected_regions) <= set(regions) or not set(intent.target_resources) <= set(resources):
            raise ValueError(dict(reason='联合意图的模块区域或资源没有对应的任务证据',
                selected_task_regions=regions, selected_task_resources=resources,
                instruction='保留 Task ID，区域与资源可提交空列表，由后端按任务证据补全。'))
        affected_regions = intent.affected_regions or regions
        target_resources = intent.target_resources or resources
        slots = reference['pp'] * reference['vpp']
        affected_boundaries = {index for region in affected_regions for index in (region, region + 1)
            if 0 < index < slots}
        if not set(intent.movable_boundaries) <= affected_boundaries:
            raise ValueError('优先修改的边界需要连接已经确认的模块区域')
        same_schedule = effective_execution_config(reference) == effective_execution_config(shape)
        return dict(source_impact_id=self.impact['impact_id'], target_task_ids=intent.target_task_ids,
            affected_regions=affected_regions, target_resources=target_resources,
            parallel_changes=parallel_ranges, runtime_changes=runtime_ranges,
            expected_effects=intent.expected_effects,
            preferred_boundaries=sorted(set(intent.movable_boundaries) or affected_boundaries) if same_schedule else [],
            timing='LOCAL_REFERENCE_PATH' if same_schedule else 'REBUILD_AND_REPLAY_TARGET_DAG')
