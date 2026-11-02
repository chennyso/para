import json
from pathlib import Path

import torch

from megatron.core.frontier.agents.acquisition import select_trials
from megatron.core.frontier.agents.history import RunHistory
from megatron.core.frontier.agents.history_filter import filter_target_expert_history
from megatron.core.frontier.agents.residual_gp import ResidualGaussianProcess


class FrontierCoordinator:
    def __init__(self, parallel_architect, pipeline_agent, execution_critic, planner, runner, history_path):
        self.parallel_architect = parallel_architect
        self.pipeline_agent = pipeline_agent
        self.execution_critic = execution_critic
        self.planner = planner
        self.runner = runner
        self.history = RunHistory(history_path)
        self.gp = ResidualGaussianProcess()
        self.round_path = Path(history_path).parent / 'rounds'
        self.round_path.mkdir(parents=True, exist_ok=True)

    def _evaluate(self, candidate, workload, origin):
        if self.history.contains(candidate['program']):
            raise ValueError('duplicate or known rejected executable program')
        if candidate['admission_status'] != 'PASS' or not candidate['executable_witness']:
            raise ValueError('real trial requires memory admission and an executable witness')
        result = self.runner.run(candidate, warmup=10, measured=30)
        if result['warmup'] != 10 or result['measured'] != 30:
            raise ValueError('search runner changed the registered training protocol')
        record = result | {'program': candidate['program'], 'program_id': self.history.identifier(candidate['program']),
                           'model': workload['model'], 'sequence_length': workload['sequence_length'],
                           'origin': origin, 'features': candidate['features'],
                           'simulated_step_us': candidate['simulated_step_us']}
        self.history.append(record)
        return record

    def optimize(self, context, max_trials=20, seed_budget=6):
        if not 1 <= max_trials <= 20 or not 1 <= seed_budget <= min(6, max_trials):
            raise ValueError('search requires at most twenty actual evaluations, including coverage seeds')
        workload = context['workload_key']
        if any(key in context for key in ('expert_configuration', 'expert_current', 'expert_trajectory')):
            raise ValueError('expert configuration must remain outside the search coordinator')
        seeds = self.planner.coverage_seeds(context, budget=seed_budget)
        for candidate in seeds:
            if self.history.real_trials >= max_trials:
                break
            if not self.history.contains(candidate['program']):
                self._evaluate(candidate, workload, 'seed')
        round_index = 0
        while self.history.real_trials < max_trials:
            measured = self.history.measured()
            if not measured:
                raise RuntimeError('agent search requires actual successful coverage measurements')
            best = min(measured, key=lambda row: row['measured_step_us'])
            observation = self.runner.collect_context(best, context)
            observation['workload_key'] = workload
            observation['run_history'] = filter_target_expert_history(self.history.records, workload)
            observation['current_best'] = best
            global_space, global_provenance = self.parallel_architect.propose(observation)
            observation['parallel_neighborhood'] = global_space.model_dump(mode='json')
            pipeline_space, pipeline_provenance = self.pipeline_agent.propose(observation)
            candidates = self.planner.synthesize(global_space, pipeline_space, observation)
            candidates = tuple(candidate for candidate in candidates
                               if not self.history.contains(candidate['program'])
                               and candidate['admission_status'] == 'PASS' and candidate['executable_witness'])
            if not candidates:
                return {'status': 'legal_neighborhood_exhausted', 'best': best,
                        'real_trials': self.history.real_trials}
            self.gp.fit([row['features'] for row in measured],
                        [row['simulated_step_us'] for row in measured], [row['measured_step_us'] for row in measured])
            features = [candidate['features'] for candidate in candidates]
            residual, uncertainty = self.gp.predict(features)
            predicted = torch.tensor([candidate['simulated_step_us'] for candidate in candidates],
                                     dtype=torch.float64) + residual
            selected, acquisition = select_trials(features, [row['features'] for row in measured], predicted,
                                                  uncertainty, best['measured_step_us'],
                                                  [candidate['estimated_gpu_seconds'] for candidate in candidates])
            evidence = {'parallel_proposal': global_space.model_dump(mode='json'),
                        'pipeline_proposal': pipeline_space.model_dump(mode='json'),
                        'parallel_provenance': global_provenance, 'pipeline_provenance': pipeline_provenance,
                        'candidate_count': len(candidates), 'selected': selected, 'acquisition': acquisition,
                        'measurements': []}
            for index in selected:
                if self.history.real_trials >= max_trials:
                    break
                candidate = candidates[index]
                result = self._evaluate(candidate, workload, 'agent_trial')
                if result['status'] == 'completed':
                    critic, provenance = self.execution_critic.critique(candidate['predicted_trace'],
                                                                       result['measured_trace'], result['evidence_ids'])
                    evidence['measurements'].append({'run': result, 'critic': critic.model_dump(mode='json'),
                                                     'critic_provenance': provenance})
                else:
                    evidence['measurements'].append({'run': result})
            (self.round_path / f'round-{round_index}.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
            round_index += 1
            if not selected:
                return {'status': 'measured_neighborhood_exhausted', 'best': best,
                        'real_trials': self.history.real_trials}
        best = min(self.history.measured(), key=lambda row: row['measured_step_us'])
        return {'status': 'budget_completed', 'best': best, 'real_trials': self.history.real_trials}
