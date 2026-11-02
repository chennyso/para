import hashlib
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from megatron.core.frontier.agents.parallel_architect import ParallelArchitect
from megatron.core.frontier.agents.pipeline_program_agent import PipelineProgramAgent
from megatron.core.frontier.agents.provider import CodexJSONProvider


class ProgramPrediction(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    program_id: str
    predicted_step_seconds: float = Field(gt=0)


class CompetingHypothesis(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    identifier: str
    explanation: str
    probability: float = Field(gt=0, le=1)
    evidence_ids: list[str]
    predictions: list[ProgramPrediction]


class DiscoveryReasoning(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    hypotheses: list[CompetingHypothesis] = Field(min_length=2, max_length=4)
    evidence_ids: list[str]
    rationale: str

    @model_validator(mode='after')
    def check_hypotheses(self):
        if len({row.identifier for row in self.hypotheses}) != len(self.hypotheses):
            raise ValueError('competing hypotheses require distinct identities')
        if not math.isclose(sum(row.probability for row in self.hypotheses), 1.0, abs_tol=1e-6):
            raise ValueError('hypothesis probabilities must sum to one')
        return self


def program_identity(candidate):
    descriptor = {key: candidate[key] for key in ('plan', 'variant', 'source_pp', 'source_vp', 'logical_layer')}
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class MeasuredProgramDiscovery:
    def __init__(self, output, workload=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.workload = workload or {'model': 'Qwen3-30B-A3B', 'sequence_length': 1024, 'layers': 48,
            'architecture': 'Qwen3 sparse MoE', 'region_kinds': ('ATTENTION', 'PRE_DISPATCH', 'ROUTER', 'DISPATCH', 'EXPERT', 'COMBINE'),
            'boundary_keys': {'canonical': 'native_moe_attention_PP_cut', 'move': 'native_moe_router_MOVE_typed_PP'},
            'cut_region_kinds': {'canonical': 'ATTENTION', 'move': 'ROUTER'},
            'tp': 1, 'cp': 2, 'pp': 4, 'dp': 2, 'ep': 4, 'etp': 1, 'edp': 1}

    def select(self, candidates, records):
        if not candidates or not records:
            raise ValueError('agent discovery requires legal executable candidates and actual completed GPU measurements')
        evidence_ids = [row['run_id'] for row in records]
        workload = self.workload
        if any(row['plan']['model'] != workload['model'] or row['plan']['sequence'] != workload['sequence_length']
               or any(row['plan'][key] != workload[key] for key in ('tp', 'cp', 'pp', 'ep')) for row in candidates):
            raise ValueError('agent candidates differ from the registered workload and executable global policy')
        history = []
        for row in records:
            if row['status'] != 'PASS' or not row['performance_eligible'] or not row['source_archive_sha256']:
                raise ValueError('agent evidence must originate from an actual completed performance run')
            history.append(row | {'model': workload['model'], 'sequence_length': workload['sequence_length'],
                                  'origin': 'agent_trial' if row.get('agent_round') is not None else 'seed'})
        regions = [f'layer:{layer}:{kind}' for layer in range(workload['layers']) for kind in workload['region_kinds']]
        catalog = [{key: row[key] for key in ('plan', 'variant', 'source_pp', 'source_vp', 'logical_layer',
                                             'predicted_stage_peaks_bytes', 'gpu_memory_bytes')}
                   | {'program_id': program_identity(row)} for row in candidates]
        boundary_keys = workload['boundary_keys']
        best = min(history, key=lambda row: row['step_seconds'])
        context = {
            'workload_key': {'model': workload['model'], 'sequence_length': workload['sequence_length']},
            'model_graph': {'region_ids': regions, 'decoder_layers': workload['layers'], 'architecture': workload['architecture']},
            'region_graph': {'region_ids': regions},
            'hardware_topology': {'node_count': 2, 'world_size': 16, 'devices': 'RTX5090', 'nodes': ['g5', 'g6']},
            'memory_admission': {'candidate_count': len(catalog), 'status': 'PASS', 'actual_peak_gib': best['peak_allocated_gib']},
            'current_best': best, 'run_history': history, 'evidence_ids': evidence_ids,
            'anchor_sensitivity': {'actual_measurements': history, 'causal_scope': 'complete_program_measurements'},
            'stage_trace': {'available_metrics': ['complete_step_seconds', 'tokens_per_second', 'peak_allocated_gib'],
                            'actual_measurements': history, 'event_breakdown_available': False},
            'memory_timeline': {'scope': 'measured_allocated_peak_and_analytic_candidate_bounds', 'candidate_bounds': catalog},
            'boundary_catalog': {'candidates': catalog, 'implementations': boundary_keys},
            'capabilities': {'global_parallel': [{'policy': {'dense': {key: workload[key] for key in ('tp', 'cp', 'pp', 'dp')},
                                                              'expert': {key: workload[key] for key in ('etp', 'ep', 'edp')}}}],
                             'state_sharding': ['distributed_optimizer'], 'mapping_templates': ['actual_device_order'],
                             'vpp': sorted({row['vpp'] for row in candidates}), 'chunk_templates': ['BOUNDARY_AWARE'],
                             'boundary_alternatives': list(boundary_keys.values()),
                             'recompute': ['none'], 'activation_offload': ['none'], 'candidate_programs': catalog},
        }
        directory = self.output / f'round-{len(records):02d}'
        provider = CodexJSONProvider(directory / 'llm')
        global_space, global_provenance = ParallelArchitect(provider).propose(context)
        if any(getattr(global_space, key) != [workload[key]] for key in ('tp', 'cp', 'pp', 'ep', 'etp')):
            raise ValueError('agent proposed a global policy outside the executable candidate catalog')
        context['parallel_neighborhood'] = global_space.model_dump(mode='json')
        pipeline_space, pipeline_provenance = PipelineProgramAgent(provider).propose(context)
        positions = {region: index for index, region in enumerate(regions)}
        eligible = [row for row in candidates if row['vpp'] in pipeline_space.vpp
                    and boundary_keys[row['variant']] in pipeline_space.boundary_alternatives
                    and any(positions[window.begin_region] <= positions[f'layer:{row["logical_layer"]}:'
                            + workload['cut_region_kinds'][row['variant']]]
                            <= positions[window.end_region] for window in pipeline_space.cut_windows)]
        if not eligible:
            raise ValueError('actual agent neighborhood contains no remaining executable candidate')
        reasoning, critic_provenance = provider.generate('ExecutionCritic',
            'Use only actual completed GPU results and the supplied executable candidates. Form two to four '
            'competing falsifiable performance hypotheses. Predict complete training step seconds for EVERY '
            'eligible candidate under EACH hypothesis. These are uncalibrated predictions, never measurements. '
            'Probabilities must sum to one. Update beliefs using prior observed outcomes when supplied. '
            'Do not claim a bottleneck is proven without event evidence. Cite only supplied evidence IDs.',
            {'actual_measurements': history, 'eligible_candidates': [row for row in catalog
              if row['program_id'] in {program_identity(candidate) for candidate in eligible}],
             'evidence_ids': evidence_ids,
             'previous_rounds': [json.loads(path.read_text()) for path in sorted(self.output.glob('round-*/selection.json'))
                                 if int(path.parent.name.split('-')[-1]) < len(records)]},
            DiscoveryReasoning)
        identifiers = {program_identity(row) for row in eligible}
        if not set(reasoning.evidence_ids) <= set(evidence_ids):
            raise ValueError('agent reasoning cites unknown actual measurements')
        for hypothesis in reasoning.hypotheses:
            if (not set(hypothesis.evidence_ids) <= set(evidence_ids)
                    or {row.program_id for row in hypothesis.predictions} != identifiers
                    or len(hypothesis.predictions) != len(identifiers)):
                raise ValueError('hypothesis predictions omitted a candidate or cited unknown evidence')
        scores = []
        best_seconds = best['step_seconds']
        for candidate in eligible:
            identifier = program_identity(candidate)
            values = [(hypothesis.probability, next(row.predicted_step_seconds for row in hypothesis.predictions
                                                   if row.program_id == identifier)) for hypothesis in reasoning.hypotheses]
            mean = sum(probability * value for probability, value in values)
            deviation = sum(probability * (value - mean)**2 for probability, value in values)
            cost = best_seconds * (best['warmup'] + best['measured']) * 16
            scores.append({'program_id': identifier, 'predicted_step_seconds': mean,
                           'improvement_per_estimated_gpu_second': sum(probability * max(0.0, best_seconds - value)
                                                                      for probability, value in values) / cost,
                           'prediction_disagreement_per_estimated_gpu_second': deviation / (best_seconds**2 * cost),
                           'estimated_gpu_seconds': cost})
        improvement = max(scores, key=lambda row: (row['improvement_per_estimated_gpu_second'], -row['predicted_step_seconds']))
        selected = [improvement['program_id']]
        diagnostic = [row for row in scores if row['program_id'] not in selected]
        if diagnostic:
            choice = max(diagnostic, key=lambda row: row['prediction_disagreement_per_estimated_gpu_second'])
            if choice['prediction_disagreement_per_estimated_gpu_second'] > 0:
                selected.append(choice['program_id'])
        result = {'parallel_input': context, 'parallel_output': global_space.model_dump(mode='json'),
                  'pipeline_output': pipeline_space.model_dump(mode='json'), 'critic_output': reasoning.model_dump(mode='json'),
                  'provenance': [global_provenance, pipeline_provenance, critic_provenance], 'scores': scores,
                  'selected_program_ids': selected, 'consumed_by_training_scheduler': False,
                  'prediction_scope': 'uncalibrated_LLM_hypotheses_for_real_experiment_selection'}
        selection_path = directory / 'selection.json'
        if selection_path.exists():
            existing = json.loads(selection_path.read_text())
            for key in ('parallel_output', 'pipeline_output', 'critic_output', 'scores', 'selected_program_ids'):
                if existing[key] != result[key]:
                    raise ValueError('authenticated agent replay changed the recorded experiment selection')
        else:
            selection_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return tuple(next(row for row in eligible if program_identity(row) == identifier) for identifier in selected), directory
