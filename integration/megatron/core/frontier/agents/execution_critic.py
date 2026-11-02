from megatron.core.frontier.agents.neighborhoods import CriticEvidence
from megatron.core.frontier.agents.repair_critique import RepairCritique


class ExecutionCritic:
    def __init__(self, provider):
        self.provider = provider

    def explain_conflicts(self, report, allowed_changes):
        machine = report.machine_report()
        conflicts = {conflict['id'] for conflict in machine['conflicts']}
        if not conflicts or not allowed_changes:
            raise ValueError('Repair critique requires actual machine conflicts and a registered edit space')
        observation = {'verification': machine, 'allowed_changes': tuple(allowed_changes)}
        result, provenance = self.provider.generate('ExecutionCritic',
            'Analyse every supplied machine conflict using its exact identifier and recorded properties. '
            'Recommend only permitted changes and assign them to Architect or Program. '
            'The Verifier determines legality; recommendations cannot declare a program legal or claim performance gains.',
            observation, RepairCritique)
        identifiers = tuple(advice.conflict_id for advice in result.advice)
        if (result.program_id != machine['program_id'] or set(identifiers) != conflicts
                or len(identifiers) != len(conflicts)):
            raise ValueError('Repair critique must cover the exact actual program and every machine conflict once')
        if any(not set(advice.requested_changes) <= set(allowed_changes) for advice in result.advice):
            raise ValueError('Repair critique requested an unregistered program edit')
        return result, provenance

    def critique(self, predicted_trace, measured_trace, evidence_ids):
        observation = {'predicted_trace': predicted_trace, 'measured_trace': measured_trace,
                       'evidence_ids': evidence_ids}
        result, provenance = self.provider.generate('ExecutionCritic',
            'Compare predicted and measured event classes. Explain residuals using actual evidence. '
            'New causal explanations have evidence_level HYPOTHESIS. Do not edit programs or select GPU trials.',
            observation, CriticEvidence)
        if not set(result.evidence_ids) <= set(evidence_ids):
            raise ValueError('critic cites unknown measured evidence')
        if result.evidence_level not in ('HYPOTHESIS', 'OBSERVED'):
            raise ValueError('evidence promotion requires independent repeated measurements')
        return result, provenance
