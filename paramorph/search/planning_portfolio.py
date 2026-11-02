from collections import defaultdict
import json
from pathlib import Path
import statistics

from contracts import fingerprint
from predictor.dag_projection import PROJECTION_REVISION
from predictor.profile_database import plan_signature
from search.conditional_partition import ConditionalPartitionSolver
from search.communication_placement import placement_families
from scheduling.native_microbatches import valid_microbatch_group


GLOBAL_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'expert_dp', 'vpp', 'group')
RUNTIME_FIELDS = ('micro_batch', 'recompute', 'optimizer_sharding')


class PortfolioControl:
    @staticmethod
    def planning_family_key(request):
        return tuple(request[name] for name in GLOBAL_FIELDS) + (request.get('overlap_p2p', False),) + tuple(
            request.get(name) for name in RUNTIME_FIELDS) + (fingerprint(request.get('rank_mapping')),)

    @staticmethod
    def planning_action_key(action, request):
        return tuple(request[name] for name in GLOBAL_FIELDS) + (
            request.get('overlap_p2p', False),) + tuple(request.get(name) for name in RUNTIME_FIELDS) + (
            fingerprint(request.get('rank_mapping')),)

    def planning_families(self, parent):
        prototype = self.deterministic_investigation(parent, 2)
        ranges = type(prototype.parallel_ranges)(**{name: list(self.domains[name])
            for name in ('tp', 'pp', 'cp', 'ep', 'etp', 'vpp')})
        prototype = prototype.model_copy(update=dict(module_scope='all', movable_module_ids=[], parallel_ranges=ranges))
        profiles = [json.loads(row[0]) for row in self.predictor.database.connection.execute('SELECT document FROM profiles')]
        solver = ConditionalPartitionSolver(self.rewriter, profiles, backend_source_hashes=self.backend_identity)
        original = parent['plan']['backend_config']
        cache_key = plan_signature(original)
        cached = getattr(self, '_planning_shape_index', {})
        if self.config.planning_full_strategy_enabled and cache_key not in cached:
            micro = self.domains.get('micro_batch', self.domains.get('micro_batch_size', [original['micro_batch']]))
            axes = dict(micro_batch=list(micro), recompute=['none', 'full', 'selective-mlp'],
                optimizer_sharding=['distributed', 'replicated']
                    if original.get('execution_backend', 'megatron_ddp') == 'megatron_ddp'
                    else [original['optimizer_sharding']])
            runtimes = {tuple(original[key] for key in RUNTIME_FIELDS)}
            for key, values in axes.items():
                for value in values:
                    changed = dict(original, **{key: value})
                    runtimes.add(tuple(changed[name] for name in RUNTIME_FIELDS))
            for profile in profiles:
                request = profile['request']
                if request['model'] == original['model'] and all(request.get(key) in values for key, values in axes.items()):
                    runtimes.add(tuple(request[key] for key in RUNTIME_FIELDS))
            placed, rejected = {}, []
            for values in sorted(runtimes):
                runtime = prototype.runtime_ranges.model_copy(update={key: [value]
                    for key, value in zip(RUNTIME_FIELDS, values)})
                action = prototype.model_copy(update=dict(runtime_ranges=runtime))
                shapes, failures = self.investigation_shapes(action, parent)
                rejected.extend(failures)
                for shape in shapes:
                    profile = solver._profile(shape)
                    variants = [shape] + ([row['request'] for row in placement_families(shape)]
                        if profile is not None else [])
                    for variant in variants:
                        microbatches = variant['global_batch'] // (variant['dp'] * variant['micro_batch'])
                        groups = (self.domains.get('group', [variant['group']])
                            if variant['vpp'] > 1 else [variant['group']])
                        for group in groups:
                            if valid_microbatch_group(microbatches, variant['pp'], variant['vpp'], group):
                                scheduled = dict(variant, group=group)
                                placed[plan_signature(scheduled)] = (scheduled, None if profile is None else profile['id'])
            cached[cache_key] = (list(placed.values()), len(rejected))
            self._planning_shape_index = cached
        if self.config.planning_full_strategy_enabled:
            indexed, rejected_count = cached[cache_key]
        else:
            shapes, rejected = self.investigation_shapes(prototype, parent)
            indexed = [(shape, None if (profile := solver._profile(shape)) is None else profile['id']) for shape in shapes]
            rejected_count = len(rejected)
        history = defaultdict(list)
        global_history = defaultdict(int)
        for trial in self.store.trials():
            prediction = trial.get('prediction')
            if (prediction and prediction.get('cost_database_revision') == self.cost_revision
                    and prediction.get('evidence', {}).get('predictor_revision') == PROJECTION_REVISION):
                history[self.planning_family_key(trial['plan']['backend_config'])].append(trial)
                known_request = trial['plan']['backend_config']
                global_history[tuple(known_request[name] for name in GLOBAL_FIELDS if name != 'group')] += 1
        rows = []
        for shape, profile_id in indexed:
            family = self.planning_family_key(shape)
            trials = history[family]
            usable = [trial['prediction']['performance']['step_time_ms'] for trial in trials
                if trial['prediction']['prediction_status'] != 'NEEDS_PROFILE'
                and trial['prediction']['performance'].get('step_time_ms') is not None
                and trial['prediction']['memory'].get('admission') != 'REJECT_OOM']
            rows.append(dict(values=list(family), module_profile_id=profile_id,
                evaluated=len(trials), rankable=len(usable), best_ms=min(usable) if usable else None,
                missing_time=sum(trial['prediction']['prediction_status'] == 'NEEDS_PROFILE'
                    or trial['prediction']['performance'].get('step_time_ms') is None for trial in trials),
                parallel_evaluated=global_history[tuple(shape[name] for name in GLOBAL_FIELDS if name != 'group')],
                changed_dimensions=sum(shape[name] != parent['plan']['backend_config'][name]
                    for name in ('tp', 'pp', 'cp', 'ep', 'etp', 'vpp')) +
                    (sum(shape.get(name) != parent['plan']['backend_config'].get(name) for name in RUNTIME_FIELDS)
                     + (shape['rank_mapping'] != parent['plan']['backend_config']['rank_mapping'])
                     if self.config.planning_full_strategy_enabled else 0),
                request=shape))
        rows.sort(key=lambda row: (row['module_profile_id'] is None,
            bool(row['missing_time'] and not row['rankable']),
            row['parallel_evaluated'] > 0 if self.config.planning_full_strategy_enabled else False,
            row['evaluated'] > 0,
            row['changed_dimensions'], row['values']))
        evidence = dict(kind='native_configuration_family_coverage', columns=list(GLOBAL_FIELDS) + ['overlap_p2p'] + list(RUNTIME_FIELDS) + ['rank_mapping_id'],
            rows=[{key: value for key, value in row.items() if key != 'request'} for row in rows],
            rejected_shape_count=rejected_count, cost_revision=self.cost_revision,
            index_policy='source_supported_runtime_combinations_and_single_axis_unknown_representatives',
            proposal_domains=self.domains,
            unknown_cross_runtime_combinations='available_to_both_selectors_through_the_same_investigation_compiler')
        identifier = fingerprint(evidence)
        self.store.add_evidence(identifier, 'PREDICTED', evidence)
        return rows, identifier

    def portfolio_queue_state(self, pending, origins):
        self._planning_portfolio_pending = pending
        self._planning_portfolio_origins = origins
        self.store.add_search_state('PLANNING_PORTFOLIO_QUEUE', dict(cost_revision=self.cost_revision,
            pending=[action.model_dump(mode='json') for action in pending], origins=origins))

    def portfolio_actions(self, parent, round_index, observation, batch_schema):
        if self.config.planning_fixed_parent_id is not None:
            parent = next(row for row in self.store.trials() if row['trial_id'] == self.config.planning_fixed_parent_id)
            observation = self.planning_observation(parent)
        if not hasattr(self, '_planning_portfolio_pending'):
            self._planning_portfolio_pending = []
            self._planning_portfolio_origins = {}
            saved = self.store.search_states('PLANNING_PORTFOLIO_QUEUE')
            if saved and saved[-1]['document']['cost_revision'] == self.cost_revision:
                state = saved[-1]['document']
                action_type = type(self.deterministic_investigation(parent, round_index))
                self._planning_portfolio_pending = [action_type.model_validate(action) for action in state['pending']]
                self._planning_portfolio_origins = state['origins']
        families, family_evidence = self.planning_families(parent)
        priorities = {self.planning_action_key(None, row['request']): index for index, row in enumerate(families)}
        if not self._planning_portfolio_pending:
            remaining = max(0.0, self.config.budget.max_walltime_seconds - self.store.search_seconds())
            completed_calls = [json.loads(path.read_text())['elapsed_seconds']
                for path in sorted((self.output / 'agents').glob('*/provenance.json'))]
            completed_receipts = [row['document'] for row in self.store.search_states('PLANNING_INVESTIGATION')]
            if self.config.planning_budget_allocation == 'completion_reserve':
                completed_receipts = [row for row in completed_receipts
                    if row.get('budget_allocation') == 'completion_reserve']
            positive = [(row['elapsed_seconds'] if self.config.planning_budget_allocation == 'completion_reserve'
                else row['prediction_seconds']) / row['actual_predictions'] for row in completed_receipts
                if row['actual_predictions'] > 0]
            evaluation_seconds = statistics.median(positive) if positive else self.config.investigation_max_seconds / 4
            generation_seconds = statistics.median(completed_calls) if completed_calls else 0
            required_seconds = generation_seconds + 2 * evaluation_seconds
            if self.config.planning_selector == 'deterministic_portfolio':
                preferred = families[0]
                prototype = self.deterministic_investigation(parent, round_index)
                actions = [prototype.model_copy(update=dict(module_scope='all', movable_module_ids=[],
                    parallel_ranges=type(prototype.parallel_ranges)(**{
                        key: [preferred['request'][key]] for key in GLOBAL_FIELDS if key != 'group'}),
                    runtime_ranges=prototype.runtime_ranges.model_copy(update={
                        **{key: [preferred['request'][key]] for key in RUNTIME_FIELDS},
                        'group': [preferred['request']['group']],
                        'overlap_p2p': [preferred['request'].get('overlap_p2p', False)],
                        'rank_mapping': [preferred['request']['rank_mapping']]}) if self.config.planning_full_strategy_enabled else prototype.runtime_ranges,
                    evidence_refs=[family_evidence], hypothesis='依据原生条件覆盖、已知成本及既有评价选择完整分区调查'))]
            elif remaining > required_seconds:
                visible_families, represented = [], set()
                for row in families:
                    key = tuple(row['request'][name] for name in GLOBAL_FIELDS)
                    if key not in represented:
                        visible_families.append(row)
                        represented.add(key)
                    if len(visible_families) == 24:
                        break
                profile_ids = sorted({row['module_profile_id'] for row in visible_families if row['module_profile_id'] is not None})
                observation = dict(observation,
                    full_strategy_permissions=dict(enabled=self.config.planning_full_strategy_enabled,
                        micro_batch=self.domains.get('micro_batch', self.domains.get('micro_batch_size', [])),
                        group=self.domains.get('group', []),
                        recompute=['none', 'full', 'selective-mlp'], optimizer_sharding=['distributed', 'replicated'],
                        mapping_scope='native_dense_axis_permutations',
                        parent_axis_mappings=[dict(axis_order=row['axis_order'], rank_mapping=row['request']['rank_mapping'])
                            for row in placement_families(parent['plan']['backend_config'])],
                        construction=self.config.planning_partition_engine),
                    configuration_families=dict(evidence_id=family_evidence,
                        columns=list(GLOBAL_FIELDS) + ['overlap_p2p'] + list(RUNTIME_FIELDS) + ['rank_mapping_id', 'source_profile_index', 'evaluated', 'rankable', 'best_ms', 'missing_time'],
                        profiles=profile_ids,
                        rows=[row['values'] + [None if row['module_profile_id'] is None else profile_ids.index(row['module_profile_id']),
                            row['evaluated'], row['rankable'], row['best_ms'], row['missing_time']] for row in visible_families],
                        total_native_families=len(families)),
                    planning_cost_estimates=dict(remaining_seconds=remaining,
                        completed_generation_median_seconds=generation_seconds,
                        new_prediction_median_seconds=evaluation_seconds,
                        desired_evaluation_reserve_seconds=2 * evaluation_seconds))
                observation['evidence_ids'] = observation['evidence_ids'] + [family_evidence]
                batch, _ = self.invoke_with_tools('JointStrategyAgent',
                    '依据真实模型结构和配置覆盖，提交2至4项能够共同使用剩余预算的调查。'
                    '同时考虑尚未调查的并行形状和已有形状下的联合模块，说明各项不同的执行依据。'
                    '已经提供预算、关键任务和成本缺项；工具查询需要说明仍然缺少什么信息。'
                    '所有范围由你选择，不要求指定某个固定并行取值。', observation, batch_schema)
                actions = batch.actions
                if not actions:
                    raise ValueError('组合调查需要至少一项实际规划动作')
            else:
                # 剩余时间用于继续已经生成的调查，避免再次等待无法容纳的模型响应。
                eligible = [row for row in completed_receipts if row['actual_predictions'] > 0
                    and fingerprint(row['action']) in self._planning_portfolio_origins]
                if not eligible:
                    return []
                best = min(eligible, key=lambda row: min([point['step_time_ms']
                    for point in row['completed_evaluations'] if point['step_time_ms'] is not None]
                    or [float('inf')]))
                original = self._planning_portfolio_origins[fingerprint(best['action'])]
                action_type = type(self.deterministic_investigation(parent, round_index))
                actions = [action_type.model_validate(original)]
            expanded, origins = [], dict(self._planning_portfolio_origins)
            for proposal in actions:
                shapes, _ = self.investigation_shapes(proposal,
                    next(row for row in self.store.trials() if row['trial_id'] == proposal.parent_trial_id))
                for shape in shapes:
                    child = proposal.model_copy(update=dict(parallel_ranges=type(proposal.parallel_ranges)(**{
                        key: [shape[key]] for key in GLOBAL_FIELDS if key != 'group'}),
                        partition_seeds=[seed for seed in proposal.partition_seeds
                            if seed.pp == shape['pp'] and seed.vpp == shape['vpp']],
                        runtime_ranges=proposal.runtime_ranges.model_copy(update={
                            **{key: [shape[key]] for key in RUNTIME_FIELDS
                               if getattr(proposal.runtime_ranges, key) is not None},
                            **({'group': [shape['group']]} if proposal.runtime_ranges.group is not None else {}),
                            **({'rank_mapping': [shape['rank_mapping']]} if proposal.runtime_ranges.rank_mapping is not None else {}),
                            **({'overlap_p2p': [shape['overlap_p2p']]} if proposal.runtime_ranges.overlap_p2p is not None else {})})))
                    expanded.append(child)
                    origins[fingerprint(child.model_dump(mode='json'))] = proposal.model_dump(mode='json')
            unique = {fingerprint(action.model_dump(mode='json')): action for action in expanded}
            def action_priority(action):
                actual_parent = next(row for row in self.store.trials() if row['trial_id'] == action.parent_trial_id)
                shapes, _ = self.investigation_shapes(action, actual_parent)
                return min((priorities.get(self.planning_action_key(action, shape), len(priorities))
                    for shape in shapes), default=len(priorities))
            pending = (list(unique.values()) if self.config.planning_selector == 'agent'
                else sorted(unique.values(), key=action_priority))
            if not pending:
                return []
            self.portfolio_queue_state(pending, origins)
        pending = self._planning_portfolio_pending
        selected = pending[0]
        remaining = max(0.0, self.config.budget.max_walltime_seconds - self.store.search_seconds())
        available = max(0, self.config.budget.max_cpu_predictions - int(self.store.usage().get('cpu_predictions', 0)))
        if remaining <= 0 or available <= 0:
            return []
        granted_seconds = min(selected.budget.seconds, remaining / len(pending))
        full_evaluations = min(selected.budget.full_evaluations, max(1, available // len(pending)))
        estimate = None
        if self.config.planning_budget_allocation == 'completion_reserve':
            completed = [row['document'] for row in self.store.search_states('PLANNING_INVESTIGATION')]
            completed = [row for row in completed if row.get('budget_allocation') == 'completion_reserve']
            costs = [row['elapsed_seconds'] / row['actual_predictions'] for row in completed
                if row['actual_predictions'] > 0]
            estimate = statistics.median(costs) if costs else self.config.investigation_max_seconds / 4
            desired_queries = min(selected.budget.full_evaluations, 4, available)
            granted_seconds = min(selected.budget.seconds, remaining,
                max(remaining / len(pending), desired_queries * estimate))
            affordable = max(1, int(granted_seconds / estimate))
            full_evaluations = min(selected.budget.full_evaluations, available, affordable)
        budget = selected.budget.model_copy(update=dict(seconds=granted_seconds,
            full_evaluations=full_evaluations))
        dispatched = selected.model_copy(update=dict(budget=budget))
        origin = self._planning_portfolio_origins[fingerprint(selected.model_dump(mode='json'))]
        self._planning_portfolio_origins[fingerprint(dispatched.model_dump(mode='json'))] = origin
        self.store.add_search_state('PLANNING_PORTFOLIO_DISPATCH', dict(
            action=dispatched.model_dump(mode='json'), proposal=origin,
            pending_count=len(pending), remaining_seconds=remaining,
            allocation=self.config.planning_budget_allocation, estimated_seconds_per_prediction=estimate))
        self.portfolio_queue_state(pending[1:], self._planning_portfolio_origins)
        return [dispatched]
