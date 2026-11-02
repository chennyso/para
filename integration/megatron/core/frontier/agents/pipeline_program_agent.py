from megatron.core.frontier.agents.history_filter import filter_target_expert_history
from megatron.core.frontier.agents.neighborhoods import PipelineNeighborhood


class PipelineProgramAgent:
    def __init__(self, provider):
        self.provider = provider

    def propose(self, context):
        allowed = ('workload_key', 'parallel_neighborhood', 'region_graph', 'readiness_dag', 'stage_trace',
                   'boundary_catalog', 'memory_timeline', 'interference_profile', 'capabilities', 'evidence_ids')
        observation = {key: context[key] for key in allowed if key in context}
        observation['run_history'] = filter_target_expert_history(context['run_history'], context['workload_key'])
        result, provenance = self.provider.generate('PipelineProgramAgent',
            'Select legal cut windows, virtual chunk templates and boundary alternatives using the supplied '
            'readiness, trace and memory evidence. The deterministic planner chooses the final executable program.',
            observation, PipelineNeighborhood)
        if not set(result.evidence_ids) <= set(context['evidence_ids']):
            raise ValueError('pipeline proposal cites unknown measured evidence')
        known_regions = context['region_graph']['region_ids']
        positions = {region: index for index, region in enumerate(known_regions)}
        for window in result.cut_windows:
            if window.begin_region not in positions or window.end_region not in positions:
                raise ValueError('pipeline proposal names an unregistered cut interface')
            if positions[window.begin_region] > positions[window.end_region]:
                raise ValueError('pipeline cut window is reversed')
        if not set(result.boundary_alternatives) <= set(context['capabilities']['boundary_alternatives']):
            raise ValueError('pipeline proposal uses an unavailable boundary implementation')
        return result, provenance
