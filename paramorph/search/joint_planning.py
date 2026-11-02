from copy import deepcopy
from collections import Counter
import gzip
import json
import time

from contracts import RewriteIntent, fingerprint
from graph.complete_rewrite import CompleteAndRewrite
from graph.bottleneck_analysis import BottleneckGraph, missing_anchor_predictions, replay_shared_cost_scenarios
from predictor.profile_database import normalized_execution, plan_signature, signature
from search.conditional_partition import ConditionalPartitionSolver, REWRITE_FIELDS
from search.family_state import FamilyLedger, GLOBAL_FIELDS, global_identity, global_values
from search.seed_planner import SeedPlanner


class JointPlanning:
    def evaluate_planning_request(self, request, parent_request):
        self.synchronize_cost_revision()
        self.rewriter.candidate(request)
        parent_context = self.projection_parent_context(parent_request)
        key = fingerprint(dict(cost_revision=self.cost_revision,
            candidate=plan_signature(request), parent=parent_context))
        cache = getattr(self, '_continuation_predictions', None)
        if cache is None:
            self._continuation_predictions = cache = {}
        if key in cache:
            return cache[key]
        self.check_walltime()
        if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
            return None
        prediction = self.predict_request(request, parent_request=parent_context)
        prediction = self.persist_prediction_graph(prediction)
        planning_evidence_id = fingerprint(dict(kind='planning_prediction',
            projection_id=prediction['prediction_id'], parent=parent_context,
            cost_revision=self.cost_revision, prediction_document=prediction))
        prediction = dict(prediction, planning_evidence_id=planning_evidence_id)
        self.store.add_evidence(planning_evidence_id, 'PREDICTED', prediction)
        identifier = 'planning:' + str(time.time_ns())
        self.store.charge(identifier + ':count', 'cpu_predictions', 1)
        self.store.charge(identifier + ':seconds', 'cpu_seconds', prediction['evaluation_seconds'])
        cache[key] = prediction
        return prediction

    def expand_complete_strategy(self, parent_request, intent, limit, *, continuation_requests=()):
        from search.complete_strategy import compile_complete_strategy
        if type(limit) is not int or limit < 1:
            raise ValueError('完整策略的候选配额必须为正整数')
        intent = RewriteIntent.model_validate(intent)
        proposed = compile_complete_strategy(self.rewriter, parent_request, intent)
        continuation_pool = {plan_signature(proposed['request']): proposed['request']}
        for request in continuation_requests:
            self.require_fixed_training_semantics(parent_request, request)
            self.rewriter.candidate(request)
            if global_identity(request) == global_identity(proposed['request']):
                continuation_pool.setdefault(plan_signature(request), request)
        for trial in self.store.trials():
            request = trial['plan']['backend_config']
            if (signature(request) == signature(proposed['request'])
                    and not trial.get('failure')
                    and (trial.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM'):
                continuation_pool.setdefault(plan_signature(request), request)
        values = global_values(proposed['request'])
        family = dict(family_id=global_identity(proposed['request']), values=values)
        candidates = {plan_signature(proposed['request']): proposed}
        pairing = None
        previous_mapping = parent_request.get('rank_mapping',
            list(range(self.topology.num_gpus)))
        if proposed['request']['rank_mapping'] != previous_mapping:
            from search.complete_strategy import complete_values
            counterpart_request = dict(proposed['request'], rank_mapping=previous_mapping)
            counterpart_intent = RewriteIntent(operation='JointTransform',
                values=complete_values(counterpart_request), adjustable=[], preserve=[],
                hypothesis='保持相同并行形状、模块布局和运行设置，单独检查物理通信放置',
                evidence_refs=intent.evidence_refs, parent_plan_id=parent_request['run_id'])
            counterpart = compile_complete_strategy(self.rewriter, parent_request, counterpart_intent)
            pairing = dict(changed_fields=['rank_mapping'],
                proposed=plan_signature(proposed['request']),
                counterpart=plan_signature(counterpart_request),
                status='PENDING_MATCHED_MEASUREMENT' if limit > 1 else 'OUTPUT_BUDGET',
                performance_advantage_supported=False)
            if limit > 1:
                counterpart['proposal_kind'] = 'placement_control'
                counterpart['control_intent'] = counterpart_intent.model_dump(mode='json')
                candidates[plan_signature(counterpart_request)] = counterpart
        if len(candidates) == limit:
            return dict(candidates=list(candidates.values()), status='UNKNOWN', search_complete=False,
                scope='complete_candidate_only', termination='candidate_limit',
                proofs=[], missing_cost_requests=[], placement_pair=pairing,
                proposed_candidate_id=plan_signature(proposed['request']),
                selected_global_family_id=family['family_id'])
        prediction = self.evaluate_planning_request(proposed['request'], parent_request)
        if prediction is None or prediction['performance']['step_time_ms'] is None:
            return dict(candidates=list(candidates.values()), status='UNKNOWN', search_complete=False,
                scope='complete_candidate_with_unresolved_costs',
                termination='costs_required_for_partition_comparison', proofs=[],
                missing_cost_requests=[] if prediction is None else
                    prediction.get('evidence', {}).get('missing_profiles', []),
                placement_pair=pairing,
                proposed_candidate_id=plan_signature(proposed['request']),
                selected_global_family_id=family['family_id'])
        # 完整策略提供续接布局，数值分区继续展开同一原生域。
        completion = self.solve_conditional_family(parent_request, family, None, limit - len(candidates) + 1,
            continuation_requests=list(continuation_pool.values()))
        for candidate in completion['candidates']:
            if len(candidates) == limit:
                break
            candidates.setdefault(plan_signature(candidate['request']), candidate)
        return dict(completion, candidates=list(candidates.values()),
            continuation_source_plan_ids=list(continuation_pool),
            proposed_candidate_id=plan_signature(proposed['request']),
            selected_global_family_id=family['family_id'], placement_pair=pairing)

    def solve_physical_stage_partition(self, request, pricer, budget, radius=2, run_id=None,
                                       pricing_costs='full_projection', profile_id=None):
        from search.physical_stage_master import PartitionDomain
        from search.physical_stage_search import (
            NativeCompositePricingOracle, PhysicalStageJointSearch, PhysicalStageOracle)
        self.synchronize_cost_revision()
        domain = PartitionDomain.around(request, radius)
        directory = self.output / 'physical-stage-partition'
        if pricing_costs == 'native_composite':
            if profile_id is None:
                raise ValueError('复合事件定价必须指定实际归档成本身份')
            encoded = self.predictor.database.connection.execute(
                'SELECT document FROM profiles WHERE id=?', (profile_id,)).fetchone()
            if encoded is None:
                raise ValueError('指定的归档成本身份不存在')
            oracle = NativeCompositePricingOracle(domain, request, self.predictor,
                self.rewriter, directory, json.loads(encoded[0]))
        elif pricing_costs == 'full_projection':
            oracle = PhysicalStageOracle(domain, request, self.predictor, self.rewriter, directory)
        else:
            raise ValueError('定价成本范围没有注册')
        result = PhysicalStageJointSearch(oracle, pricer, budget).run(run_id or request['run_id'])
        self.store.add_evidence(fingerprint(result), 'PREDICTED', result)
        return result

    def joint_global_families(self, request):
        cache_key = fingerprint(dict(request=request, domains=self.domains))
        if cache_key in self.global_family_cache:
            return deepcopy(self.global_family_cache[cache_key])
        planner = SeedPlanner(self.rewriter, self.profile)
        grouped = {}
        parent = global_values(request)
        for row in planner.global_families(request):
            representative = row['representative_request']
            identifier = global_identity(representative)
            if identifier not in grouped:
                values = global_values(representative)
                grouped[identifier] = dict(family_id=identifier, values=values,
                    representative_request=representative, state_variant_count=0, state_variants=[],
                    parent_distance=sum(values[key] != parent[key] for key in GLOBAL_FIELDS))
            grouped[identifier]['state_variant_count'] += 1
            grouped[identifier]['state_variants'].append({name: representative[name] for name in (
                'execution_backend', 'megatron_fsdp_version', 'optimizer_sharding',
                'dense_sharding_strategy', 'expert_sharding_strategy',
                'optimizer_offload_fraction', 'optimizer_transfer_overlap')})
        result = sorted(grouped.values(), key=lambda row: row['family_id'])
        self.global_family_cache[cache_key] = deepcopy(result)
        return result

    def family_catalog(self, trial, representatives=()):
        self.synchronize_cost_revision()
        families = self.joint_global_families(trial['plan']['backend_config'])
        records = {row['family_id']: row for row in self.family_ledger.records()}
        summaries = []
        for family in families:
            row = records.get(family['family_id'])
            summaries.append(dict(family_id=family['family_id'], values=family['values'],
                state_variant_count=family['state_variant_count'],
                state_variants=family['state_variants'],
                status=row['status'] if row else 'UNKNOWN',
                candidate_count=len(row['candidate_ids']) if row else 0,
                best_predicted_step_ms=(row['best_predicted']['predicted_step_ms']
                    if row and row['best_predicted'] else None),
                best_measured_tokens_per_s=(row['best_measured']['measured_tokens_per_s']
                    if row and row['best_measured'] else None),
                lower_bound_ms=row['lower_bound_ms'] if row else None,
                lower_bound_evidence=row['lower_bound_evidence'] if row else None))
        return dict(total_legal_families=len(families), family_fields=list(GLOBAL_FIELDS),
            value_coverage={key: sorted({json.dumps(row['values'][key], sort_keys=True)
                if key == 'rank_mapping' else row['values'][key] for row in families})
                for key in GLOBAL_FIELDS},
            family_status_counts=dict(Counter(row['status'] for row in summaries)),
            representative_families=summaries[:16],
            representative_family_count=min(16, len(summaries)),
            catalog_access=dict(tool='inspect_global_families', page_limit=32),
            selection_rule='global_H_with_conditional_L_Z_completion_and_measured_feedback',
            cost_revision=self.cost_revision)

    def solve_conditional_family(self, parent_request, family, intent, limit, *,
                                 continuation_requests=()):
        self.synchronize_cost_revision()
        started = time.perf_counter()
        fixed = {}
        rewriter = self.rewriter
        if intent is None:
            values = {key: family['values'][key] for key in GLOBAL_FIELDS}
            evidence_id = fingerprint(dict(global_family=family['family_id'],
                parent=plan_signature(parent_request)))
        else:
            intent = RewriteIntent.model_validate(intent)
            if all(key in intent.values for key in GLOBAL_FIELDS):
                values = {key: intent.values[key] for key in GLOBAL_FIELDS}
            else:
                closed = self.rewriter.complete(parent_request, intent, limit=1)[0]['request']
                values = global_values(closed)
            evidence_id = fingerprint(intent.model_dump(mode='json'))
            normalized = normalized_execution(parent_request)
            for key in REWRITE_FIELDS:
                if key in GLOBAL_FIELDS:
                    continue
                if key in intent.values:
                    fixed[key] = intent.values[key]
                elif key in intent.preserve:
                    if key in parent_request:
                        fixed[key] = parent_request[key]
                    elif key in normalized and normalized[key] is not None:
                        fixed[key] = normalized[key]
            if intent.allowed_ranges:
                domains = deepcopy(self.domains)
                for key, allowed in intent.allowed_ranges.items():
                    if key in domains:
                        domains[key] = [value for value in domains[key] if value in allowed]
                        if not domains[key]:
                            raise ValueError(dict(status='CONSTRAINT_CONFLICT',
                                conflicts=[f'{key}的允许范围与后端约束没有交集']))
                rewriter = CompleteAndRewrite(self.model, self.topology, self.registry, domains)
        profiles = [json.loads(row[0]) for row in self.predictor.database.connection.execute(
            "SELECT document FROM profiles WHERE json_extract(document, '$.request.model')=?",
            (parent_request['model'],))]
        solver = ConditionalPartitionSolver(rewriter, profiles,
            memory_evaluator=self.predictor.evaluate_memory_bound,
            backend_source_hashes=self.backend_identity)
        replay_limit = int(min(self.config.conditional_partition_max_replays,
            max(0, self.config.budget.max_cpu_predictions
                - self.store.usage().get('cpu_predictions', 0) - limit)))
        prediction_seconds = 0.0

        def evaluate_continuation(request):
            nonlocal prediction_seconds
            before = self.store.usage().get('cpu_seconds', 0.0)
            prediction = self.evaluate_planning_request(request, parent_request)
            prediction_seconds += self.store.usage().get('cpu_seconds', 0.0) - before
            return prediction

        result = solver.solve(parent_request, values, evidence_id, fixed_values=fixed, limit=limit,
            allowed_ranges={} if intent is None else intent.allowed_ranges,
            module_relations=() if intent is None else intent.module_relations,
            preserve=() if intent is None else intent.preserve,
            max_states_per_boundary=self.config.conditional_partition_max_states,
            max_expansions=self.config.conditional_partition_max_expansions,
            max_layouts=self.config.conditional_partition_max_layouts,
            continuation_requests=continuation_requests,
            continuation_evaluator=evaluate_continuation if continuation_requests else None,
            max_continuation_evaluations=replay_limit)
        if not fixed and (intent is None or (not intent.allowed_ranges and not intent.module_relations)):
            result['scope'] = 'all_family_completions'
        else:
            result['scope'] = 'conditional_search_domain'
        compiled = []
        for row in result['candidates']:
            if plan_signature(row['request']) == plan_signature(parent_request):
                continue
            rewrite = rewriter.complete_and_rewrite(parent_request, row['intent'], limit=1)[0]
            rewrite['intent'] = row['intent']
            rewrite['partition_evidence'] = row['partition_evidence']
            rewrite['global_family_id'] = global_identity(rewrite['request'])
            compiled.append(rewrite)
        result['candidates'] = compiled
        if 'continuation_search' in result:
            continuation_record = dict(kind='partition_continuation',
                parent=plan_signature(parent_request), cost_revision=self.cost_revision,
                global_values=values, **result['continuation_search'])
            result['continuation_evidence_id'] = fingerprint(continuation_record)
            self.store.add_evidence(result['continuation_evidence_id'], 'PREDICTED', continuation_record)
        result['solve_seconds'] = time.perf_counter() - started
        result['partition_exclusive_seconds'] = result['solve_seconds'] - prediction_seconds
        if result['partition_exclusive_seconds'] < 0:
            raise ValueError('分区预测计费超过实际完整调用时间')
        identifier = fingerprint(dict(parent=plan_signature(parent_request), values=values,
            fixed=fixed, cost_revision=self.cost_revision,
            module_relations=result['module_relations'],
            candidates=[plan_signature(row['request']) for row in compiled],
            counters=result['counters']))
        self.store.charge('partition:' + identifier + ':' + str(time.time_ns()),
            'partition_cpu_seconds', result['partition_exclusive_seconds'])
        self.family_ledger.record_solver(dict(parent_request, **values), result, self.cost_revision)
        return result

    def synchronize_cost_revision(self):
        revision = self.predictor.database.cost_database_revision()
        if revision != self.cost_revision:
            self.cost_revision = revision
            self.family_ledger.invalidate_costs(revision)
        return revision

    def prepare_joint_candidates(self, parent):
        remaining = self.config.budget.max_cpu_predictions - int(self.store.usage().get('cpu_predictions', 0))
        limit = min(max(0, remaining), max(2, self.config.budget.max_candidates_per_round // 2))
        if limit == 0:
            return [], self.family_catalog(parent) | dict(evidence_id=None)
        families = self.joint_global_families(parent['plan']['backend_config'])
        records = {row['family_id']: row for row in self.family_ledger.records()}
        families = [row for row in families if records.get(row['family_id'], {}).get('status') != 'INFEASIBLE']
        def priority(row):
            record = records.get(row['family_id'])
            unseen = record is None or not record['candidate_ids']
            best = None if record is None else record['best_predicted']
            lower_bound = None if record is None else record['lower_bound_ms']
            potential = (best['predicted_step_ms'] if best else
                lower_bound if lower_bound is not None else float('inf'))
            return (not unseen, bool(record and record['search_complete']),
                potential, len(record['pending_cost_keys']) if record else 0,
                -row['parent_distance'], row['family_id'])
        families.sort(key=priority)
        candidates = []
        per_family = min(3, limit)
        family_attempt_limit = max(8, limit * 4)
        attempted_families = 0
        for family in families:
            if len(candidates) >= limit or attempted_families >= family_attempt_limit:
                break
            self.check_walltime()
            attempted_families += 1
            result = self.solve_conditional_family(parent['plan']['backend_config'], family,
                None, min(per_family, limit - len(candidates)))
            for row in result['candidates']:
                if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.max_cpu_predictions:
                    break
                hypothesis = RewriteIntent.model_validate(row['intent'])
                trial = self.predict_candidate(row, parent, hypothesis)
                self.family_ledger.record_trial(trial, self.cost_revision)
                if trial['measurement'] is None:
                    candidates.append(trial)
        candidates = list({row['trial_id']: row for row in candidates}.values())
        catalog = self.family_catalog(parent, candidates)
        catalog['initial_family_attempt_limit'] = family_attempt_limit
        catalog['initial_family_attempted'] = attempted_families
        catalog['initial_family_selection'] = 'best_first_bounded_conditional_completion'
        evidence_id = fingerprint(dict(kind='conditional_global_family_catalog', catalog=catalog))
        self.store.add_evidence(evidence_id, 'PREDICTED', dict(kind='conditional_global_family_catalog', **catalog))
        return candidates, catalog | dict(evidence_id=evidence_id)

    def full_prediction(self, trial):
        prediction = deepcopy(trial['prediction'])
        if prediction is None:
            raise ValueError('DAG成本调查需要已有完整预测')
        path = prediction.get('evidence', {}).get('cost_records_artifact')
        if path:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                artifact = json.load(stream)
            prediction['evidence']['costs'] = artifact['costs']
            prediction['execution_graph'] = artifact['execution_graph']
        elif prediction.get('execution_graph') is not None and 'tasks' not in prediction['execution_graph']:
            prediction['execution_graph'] = self.load_prediction_graph(prediction)
        if prediction.get('execution_graph') is None:
            raise ValueError(dict(status='NEEDS_PROFILE',
                prediction_id=prediction['prediction_id'],
                reason='带成本执行图尚不可用',
                missing_profiles=prediction['evidence'].get('missing_profiles', [])))
        prediction['candidate_id'] = trial['trial_id']
        return prediction

    def trial_cost_keys(self, trial):
        prediction = trial.get('prediction')
        if prediction is None:
            return set()
        keys = {row['cost_key'] for row in prediction.get('evidence', {}).get(
            'profiling_requests', []) if row.get('cost_key')}
        path = prediction.get('evidence', {}).get('cost_records_artifact')
        if path:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                graph = json.load(stream)['execution_graph']
            if graph is not None:
                keys.update(component['cost_key'] for task in graph['tasks'].values()
                    for component in task['metadata'].get('cost_components', []))
        return keys

    def bottleneck_impact(self, trial, task_ids, epsilon=0.1):
        prediction = self.full_prediction(trial)
        graph = BottleneckGraph(prediction['execution_graph'], evidence_refs=[prediction['prediction_id']])
        closure = graph.closure(task_ids, dependency_depth=1, include_resources=True)
        impact = graph.epsilon_impact(closure['task_ids'], epsilon,
            resource_model=self.predictor.projector.parameters.get('resource_replay', {}))
        self.store.charge('impact:' + fingerprint(dict(prediction=prediction['prediction_id'],
            task_ids=task_ids, epsilon=epsilon)), 'counterfactual_replays', 2)
        return dict(closure=graph.page(closure, limit=16, max_chars=6000), impact=impact)

    def bottleneck_graph(self, trial):
        if trial.get('prediction') is not None:
            prediction = self.full_prediction(trial)
            return (BottleneckGraph(prediction['execution_graph'],
                evidence_refs=[prediction['prediction_id']]), 'PREDICTED_COSTED_DAG')
        graph = self.backend_dag(trial)
        tasks = {}
        for task_id, row in graph['tasks'].items():
            metadata = dict(row['metadata'])
            tasks[task_id] = dict(rank=row['rank'], kind=row['operation'],
                region=row['region_id'], microbatch=row['microbatch'],
                dependencies=row['dependencies'], duration_ms=None,
                resources=row.get('resources', row.get('resource_requirements', [])), metadata=metadata)
        structural = dict(tasks=tasks, task_count=len(tasks), graph_sha256=graph['graph_sha256'])
        return (BottleneckGraph(structural, evidence_refs=[graph['graph_sha256']]),
            'STRUCTURAL_DAG')

    def shared_ranking_probes(self, candidates):
        eligible = [trial for trial in candidates if trial['prediction']
            and trial['prediction']['performance'].get('step_time_ms') is not None
            and trial['prediction']['memory']['admission'] != 'REJECT_OOM']
        missing_raw = [trial['trial_id'] for trial in eligible
            if trial['prediction']['performance'].get('raw_resource_aware_step_time_ms') is None]
        if missing_raw:
            return [dict(status='UNAVAILABLE', reason='missing_raw_candidate_prediction',
                cost_key=None, candidate_ids=sorted(missing_raw))]
        reports = [self.full_prediction(trial) for trial in eligible]
        incomplete = []
        for report in reports:
            tasks = (report.get('execution_graph') or {}).get('tasks', {})
            missing_bindings = sum(type(row.get('duration_ms')) in (int, float) and row['duration_ms'] > 0 and
                not row.get('metadata', {}).get('cost_components') for row in tasks.values())
            missing_resources = sum('resources' not in row or type(row.get('duration_ms')) not in (int, float)
                                    for row in tasks.values())
            if not tasks or missing_bindings or missing_resources or not report['evidence'].get('costs'):
                incomplete.append(dict(candidate_id=report['candidate_id'],
                    complete_graph_available=bool(tasks), tasks_missing_cost_bindings=missing_bindings,
                    tasks_missing_replay_semantics=missing_resources))
        if incomplete:
            return [dict(status='UNAVAILABLE', reason='incomplete_candidate_costed_graph', cost_key=None,
                candidate_ids=sorted(row['candidate_id'] for row in incomplete), candidate_issues=incomplete)]
        if len(reports) < 2:
            return []
        known = {report['prediction_id']: report for report in reports}
        required = {report['evidence']['graph_delta_calibration']['prediction_id'] for report in reports
            if report['performance'].get('graph_delta_anchored', False)}
        anchors = {}
        trials = {trial['prediction']['prediction_id']: trial for trial in self.store.trials()
                  if trial.get('prediction') is not None}
        for identity in sorted(required):
            if identity in known:
                anchors[identity] = known[identity]
                continue
            if identity in trials:
                anchors[identity] = self.full_prediction(trials[identity])
                continue
            record = self.predictor.database.connection.execute(
                'SELECT document FROM predictions WHERE id=?', (identity,)).fetchone()
            if record is None:
                record = self.store.connection.execute(
                    "SELECT document FROM evidence WHERE json_extract(document, '$.prediction_id')=?",
                    (identity,)).fetchone()
            if record is None:
                continue
            prediction = json.loads(record[0])
            artifact_path = prediction.get('evidence', {}).get('cost_records_artifact')
            if artifact_path:
                with gzip.open(artifact_path, 'rt', encoding='utf-8') as stream:
                    artifact = json.load(stream)
                prediction['evidence']['costs'] = artifact['costs']
                prediction['execution_graph'] = artifact['execution_graph']
            elif 'tasks' not in prediction.get('execution_graph', {}):
                prediction['execution_graph'] = self.load_prediction_graph(prediction)
            if prediction.get('execution_graph') and prediction['execution_graph'].get('tasks'):
                anchors[identity] = prediction
        missing = missing_anchor_predictions(reports, anchors)
        if missing:
            return [dict(status='UNAVAILABLE', reason='anchor_prediction_not_evaluable', cost_key=None,
                anchor_issues=missing,
                candidate_ids=sorted({row['candidate_id'] for row in missing}),
                missing_anchor_prediction_ids=sorted({row['anchor_prediction_id'] for row in missing
                    if row['reason'] == 'missing_actual_anchor_prediction'}))]
        sources = {report['prediction_id']: report for report in reports}
        sources.update(anchors)
        costs = {}
        for report in sources.values():
            bound_keys = {part['cost_key'] for task in report['execution_graph']['tasks'].values()
                for part in task['metadata'].get('cost_components', [])}
            request_sensitivity = {}
            for request in report['evidence'].get('profiling_requests', []):
                key = request.get('cost_key')
                sensitivity = request.get('sensitivity_scenario_ms')
                if key is not None and sensitivity is not None and sensitivity > 0:
                    request_sensitivity[key] = max(request_sensitivity.get(key, 0.0), sensitivity)
            for row in report['evidence'].get('costs', []):
                key = row.get('cost_key')
                if key not in bound_keys:
                    continue
                spreads = [row[name] for name in ('stddev_ms', 'sampling_stddev_ms', 'uncertainty_ms')
                           if row.get(name) is not None and row[name] > 0]
                value = row.get('milliseconds')
                request_key = row.get('profiling_request', {}).get('cost_key')
                spreads.extend(request_sensitivity[source] for source in (key, request_key)
                               if source in request_sensitivity)
                uncertainty = max(spreads) if spreads else None
                if value is None or uncertainty is None or uncertainty <= 0:
                    continue
                entry = costs.setdefault(key, dict(low=max(0.0, value - uncertainty),
                    high=value + uncertainty, candidates=set(), source_prediction_ids=set()))
                entry['low'] = min(entry['low'], max(0.0, value - uncertainty))
                entry['high'] = max(entry['high'], value + uncertainty)
                entry['source_prediction_ids'].add(report['prediction_id'])
        for report in reports:
            identities = {report['prediction_id']}
            if report['performance'].get('graph_delta_anchored', False):
                identities.add(report['evidence']['graph_delta_calibration']['prediction_id'])
            for entry in costs.values():
                if identities & entry['source_prediction_ids']:
                    entry['candidates'].add(report['candidate_id'])
        selected = sorted(costs, key=lambda key: (-len(costs[key]['candidates']), key))[:4]
        models = {identity: self.predictor.projector.parameters.get('resource_replay', {})
                  for identity in sources}
        result = []
        for key in selected:
            row = costs[key]
            charge_id = fingerprint(dict(cost_revision=self.cost_revision, key=key,
                predictions=[item['prediction_id'] for item in reports], anchors=sorted(anchors),
                scope='shared_candidate_and_anchor_cost_scenario_v2', low=row['low'], high=row['high']))
            evidence_id = 'shared-cost-ranking-' + charge_id
            cached = self.store.connection.execute('SELECT document FROM evidence WHERE id=?',
                (evidence_id,)).fetchone()
            if cached is None:
                report = replay_shared_cost_scenarios(reports, key, row['low'], row['high'], models,
                    self.config.ranking_top_k, anchor_reports=anchors)
                self.store.add_evidence(evidence_id, 'PREDICTED', report)
                self.store.charge('shared-cost:' + charge_id, 'counterfactual_replays',
                    report['evaluation_count'] + report['baseline_replay_count'])
            else:
                report = json.loads(cached[0])
            result.append(report)
        return result
