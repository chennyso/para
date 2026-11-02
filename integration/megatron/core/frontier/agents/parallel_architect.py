from megatron.core.frontier.agents.history_filter import filter_target_expert_history
from megatron.core.frontier.agents.neighborhoods import GlobalParallelNeighborhood


class ParallelArchitect:
    def __init__(self, provider):
        self.provider = provider

    def propose(self, context):
        allowed = ('workload_key', 'model_graph', 'hardware_topology', 'memory_admission',
                   'anchor_sensitivity', 'current_best', 'capabilities', 'evidence_ids')
        observation = {key: context[key] for key in allowed if key in context}
        observation['run_history'] = filter_target_expert_history(context['run_history'], context['workload_key'])
        result, provenance = self.provider.generate('ParallelArchitect',
            'Select a justified neighborhood of supported global parallel policies and domain splits. '
            'Use the measured anchor sensitivity and memory admission. Do not select a final training trial.',
            observation, GlobalParallelNeighborhood)
        if not set(result.evidence_ids) <= set(context['evidence_ids']):
            raise ValueError('parallel proposal cites unknown measured evidence')
        if not set(result.domain_split_regions) <= set(context['model_graph']['region_ids']):
            raise ValueError('parallel proposal names an unregistered region')
        policies = context['capabilities']['global_parallel']
        for field in ('tp', 'cp', 'pp'):
            allowed_degrees = {row['policy']['dense'][field] for row in policies}
            if not set(getattr(result, field)) <= allowed_degrees:
                raise ValueError('parallel proposal exceeds the supported policy dimensions')
        return result, provenance
