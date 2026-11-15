from copy import deepcopy
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

from agents.evidence_policy import AblationEvidencePolicy
from contracts import RewriteIntent, fingerprint
from cost.runtime_calibration import configuration
from graph.complete_rewrite import CompleteAndRewrite
from graph.bottleneck_analysis import BottleneckGraph, missing_anchor_predictions, replay_shared_cost_scenarios
from predictor.profile_database import normalized_execution, plan_signature, signature
from search.conditional_partition import ConditionalPartitionSolver, REWRITE_FIELDS
from search.family_state import FamilyLedger, GLOBAL_FIELDS, global_identity, global_values
from search.seed_planner import SeedPlanner
from search.joint_scope import JointScopeCompiler
from search.policy_frontier import PolicyFrontier


class JointPlanning:
    def native_profile_source_hashes(self,profile):
        from adapters.megatron_adapter import profile_source_paths
        paths = self.adapter.runner.runtime_source_paths()
        paths.update(profile_source_paths(profile['request'],
            numerical='numerical_reference.py' in profile['source_hashes']))
        files = tuple((name,str(path),Path(path).stat().st_mtime_ns,Path(path).stat().st_size)
            for name,path in sorted(paths.items()))
        cache = getattr(self,'_native_profile_source_cache',None)
        if cache is None:
            self._native_profile_source_cache = cache = {}
        if files not in cache:
            cache[files] = {name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in paths.items()}
        return cache[files]

    def run_frontier_investigation(self,request,reference_report,module_profile,evaluate,
                                   module_relations,full_evaluations,action_id):
        from search.frontier_pricer import FrontierPricer
        from search.module_relations import complete_partition_boundaries,prefix_support
        from search.physical_stage_master import PartitionDomain
        from search.physical_stage_search import JointSearchBudget,PhysicalStageJointSearch,PhysicalStageOracle
        domain = PartitionDomain.around(request,self.config.planning_frontier_radius)
        if not prefix_support(request['decoder_chunks'],self.model.num_layers,request['pp'],request['vpp'],module_relations):
            witness = complete_partition_boundaries(domain,{},module_relations)
            if witness is None:
                return dict(status='NO_NATIVE_PARTITION_IN_NEIGHBORHOOD',expansions=0,evaluations=[])
            request = dict(request,decoder_chunks=[b-a for a,b in zip(witness,witness[1:])])
            reference_report = evaluate(request)
            domain = PartitionDomain.around(request,self.config.planning_frontier_radius)
        if reference_report is None or module_profile is None:
            return dict(status='NEEDS_PROFILE',expansions=0,evaluations=[],
                reason='当前执行条件缺少完整种子评价或实际模块成本')
        graph = self.load_prediction_graph(reference_report)
        if not graph:
            return dict(status='NEEDS_PROFILE',expansions=0,evaluations=[],
                prediction_id=reference_report['prediction_id'],
                profiling_requests=reference_report.get('evidence', {}).get('profiling_requests', []))
        directory = self.output/'physical-stage-investigations'/action_id/plan_signature(request)
        prediction_start = self.store.usage().get('cpu_predictions',0)

        def scoped_evaluate(candidate,candidate_graph=None):
            if self.store.usage().get('cpu_predictions',0)-prediction_start >= full_evaluations:
                return None
            return evaluate(candidate,candidate_graph=candidate_graph)

        def projection_observer(candidate,report):
            identifier = 'frontier-column:'+str(time.time_ns())
            self.store.charge(identifier+':count','column_cost_projections',1)
            self.store.charge(identifier+':seconds','column_cost_seconds',report['evaluation_seconds'])
            self.store.add_search_state('PHYSICAL_STAGE_COST_PROJECTION',dict(action_id=action_id,
                plan_signature=plan_signature(candidate),prediction=self.persist_prediction_graph(report)))

        def generate(role,instructions,observation,schema):
            before = self.store.usage().get('agent_calls',0)
            self.store.add_evidence(observation['pricing_evidence_id'],'PREDICTED',dict(
                kind='native_lp_pricing_observation',prices=observation['prices'],artifact=observation['pricing_artifact']))
            metric = dict(context_id=fingerprint(dict(pricing_program=observation)),tool_round=0,
                current_tool_return_tokens=0,cumulative_tool_return_tokens=0,
                cumulative_estimated_context_tokens=0,cumulative_actual_input_tokens=0)
            response,identifier = self.invoke('JointStrategyAgent',instructions,observation,schema,metric)
            return response,dict(evidence_id=identifier,role='JointStrategyAgent',wire_schema=schema.__name__,
                actual_provider_calls=self.store.usage().get('agent_calls',0)-before)

        oracle = PhysicalStageOracle(domain,request,self.predictor,self.rewriter,directory,
            seed_report=dict(reference_report,execution_graph=graph),module_profile=module_profile,
            complete_evaluator=lambda candidate,compiled:scoped_evaluate(candidate,candidate_graph=compiled),
            projection_observer=projection_observer,graph_loader=self.load_prediction_graph,
            module_relations=module_relations)
        pricer = FrontierPricer(generate=generate,policy_model=self.config.planning_pricing_program_path)
        budget = JointSearchBudget(column_queries=max(request['pp'],self.config.planning_frontier_column_queries),
            complete_evaluations=max(1,full_evaluations),rounds=self.config.planning_frontier_lp_rounds,
            columns_per_round=max(1,24//request['pp']),wall_seconds=None)
        result = PhysicalStageJointSearch(oracle,pricer,budget).run(action_id)
        for deferred in result['deferred_cost_requests']:
            if scoped_evaluate(deferred['request']) is None:
                break
        self.store.add_search_state('PHYSICAL_STAGE_PROGRAM',dict(action_id=action_id,
            policy=None if pricer.policy is None else pricer.policy.model_dump(mode='json'),
            result=result,artifact=str(directory/'search-result.json')))
        result.update(status='COMPLETE' if result['stop']=='NO_PROGRESS' else 'SEARCHED_DECLARED_NEIGHBORHOOD',
            expansions=result['frontier_search']['frontier_states'])
        return result

    def execution_boundary_index(self):
        from search.boundary_certificates import ExecutionBoundaryIndex
        index = getattr(self,'_execution_boundary_index',None)
        if index is None or index.cost_revision != self.cost_revision:
            self._execution_boundary_index = index = ExecutionBoundaryIndex(self.cost_revision,
                fingerprint(dict(hardware=self.hardware_identity,backend=self.backend_identity,
                    predictor=self.predictor_revision,parameters=self.predictor.projector.parameters)))
            self._boundary_registered_prediction_ids = set()
        for trial in self.store.trials():
            prediction = trial.get('prediction')
            if (prediction is None or prediction.get('prediction_id') in self._boundary_registered_prediction_ids
                    or trial['plan'].get('hardware_fingerprint') != fingerprint(self.hardware_identity)
                    or trial['plan'].get('backend_fingerprint') != fingerprint(self.backend_identity)
                    or prediction.get('cost_database_revision') != self.cost_revision
                    or prediction.get('prediction_status') not in ('PREDICTED','PREDICTED_CALIBRATED')
                    or prediction.get('memory', {}).get('admission') != 'WITHIN_ESTIMATED_CAPACITY'):
                continue
            self._boundary_registered_prediction_ids.add(prediction['prediction_id'])
            graph = self.load_prediction_graph(prediction)
            if graph is not None:
                index.register(trial['plan']['backend_config'],dict(prediction,execution_graph=graph))
        return index

    def solve_joint_scope(self, parent_request, family, intent, limit):
        self.synchronize_cost_revision()
        intent = RewriteIntent.model_validate(intent)
        scope = JointScopeCompiler().compile(parent_request, family, intent)
        measured_signatures = sorted({plan_signature(row['plan']['backend_config']) for row in self.store.trials()
            if (row.get('measurement') or {}).get('status') == 'PASS'
            and row['measurement'].get('performance_eligible') is True
            and all(row['plan']['backend_config'].get(name) == parent_request.get(name)
                for name in ('warmup', 'measured'))})
        key = fingerprint(dict(parent=plan_signature(parent_request), family=family,
            intent=intent.model_dump(mode='json'), limit=limit, cost_revision=self.cost_revision,
            measured_signatures=measured_signatures))
        if not hasattr(self, '_joint_scope_completions'):
            self._joint_scope_completions = {}
        if key in self._joint_scope_completions:
            return deepcopy(self._joint_scope_completions[key])
        closed = intent.model_copy(update=dict(values=dict(scope.global_values, **scope.fixed_values),
            adjustable=list(scope.adjustable), allowed_ranges=scope.allowed_ranges, preserve=[]))
        completion = self.solve_conditional_family(parent_request, family, closed, limit=max(2, limit))
        frontier = PolicyFrontier(self.rewriter).expand(parent_request, completion['candidates'], scope, limit,
            measured_signatures=measured_signatures)
        evaluated = []
        for row in frontier['candidates']:
            prediction = self.evaluate_planning_request(row['request'], parent_request)
            evaluated.append(dict(layout_id=row['policy_layout_id'], plan_signature=plan_signature(row['request']),
                policy=dict(recompute=row['request']['recompute'],
                    modules=(row['request'].get('memory_policy') or {}).get('recompute_modules', []),
                    backend=row['request']['execution_backend'],
                    optimizer_offload=row['request'].get('optimizer_offload_fraction', 0)),
                prediction_id=None if prediction is None else prediction['prediction_id'],
                step_time_ms=None if prediction is None else prediction['performance']['step_time_ms'],
                memory=None if prediction is None else prediction['memory'],
                missing_profiles=[] if prediction is None else prediction['evidence'].get('missing_profiles', []),
                comparison_conditions=None if prediction is None else dict(
                    revision=prediction.get('cost_database_revision'),
                    predictor=prediction.get('evidence', {}).get('predictor_revision'),
                    anchor=prediction.get('evidence', {}).get('compatible_anchor_id')),
                status='PREDICTION_BUDGET' if prediction is None else prediction['prediction_status']))
        retention = PolicyFrontier.retain(evaluated)
        report = dict(kind='policy_frontier', scope=scope.__dict__,
            cost_database_revision=self.cost_revision, layouts=frontier['comparisons'], evaluations=evaluated,
            retention=retention,
            policy_generation_complete=completion['search_complete']
                and all(row['status'] == 'COMPLETE' for row in frontier['comparisons']),
            evaluation_status='PARTIAL' if retention['unresolved'] else 'COMPLETE')
        identifier = fingerprint(report)
        self.store.add_evidence(identifier, 'PREDICTED', report)
        result = dict(completion, candidates=[row for row in frontier['candidates']
            if plan_signature(row['request']) in retention['retained']], policy_frontier_evidence_id=identifier,
            policy_frontier=report)
        self._joint_scope_completions[key] = deepcopy(result)
        return result

    def evaluate_planning_request(self, request, parent_request,candidate_graph=None):
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
        if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.cpu_prediction_ceiling:
            return None
        started = time.perf_counter()
        bound = self.predictor.evaluate_memory_bound(request) if self.predictor.projector is not None else None
        memory = None if bound is None else bound['memory']
        if memory is not None and memory['status'] == 'REJECTED_MEMORY_LOWER_BOUND':
            prediction = self.predictor.needs_profile(request, started, [], persist=False, memory_bound=memory)
            prediction.update(prediction_status='REJECTED_MEMORY_LOWER_BOUND',
                validation=dict(status='REJECTED_MEMORY_LOWER_BOUND', backend_supported=True))
            prediction['performance']['ranking_eligible'] = False
            prediction['evidence'].update(predictor_revision=self.predictor_revision, profiling_requests=[])
        else:
            prediction = self.predict_request(request,candidate_graph=candidate_graph,parent_request=parent_context)
            prediction['evaluation_seconds'] = time.perf_counter() - started
        boundary_index = getattr(self,'_execution_boundary_index',None)
        if boundary_index is not None and boundary_index.cost_revision == self.cost_revision:
            boundary_index.register(request,prediction)
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
        if pricer is None:
            from search.frontier_pricer import FrontierPricer, PricingPolicy
            pricer = FrontierPricer(PricingPolicy(reason='完整事件LP价格指导逐虚拟区间扩展',
                evidence_refs=[oracle.revision]))
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
            values = {key: family['values'][key] for key in GLOBAL_FIELDS}
            conflicts = [key for key in GLOBAL_FIELDS if key in intent.values and intent.values[key] != values[key]]
            if conflicts:
                raise ValueError(dict(status='CONSTRAINT_CONFLICT', conflicts=conflicts))
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
            backend_source_hashes=self.backend_identity,boundary_index=self.execution_boundary_index(),
            backend_source_resolver=self.native_profile_source_hashes)
        replay_limit = int(min(self.config.conditional_partition_max_replays,
            max(0, self.config.budget.cpu_prediction_ceiling
                - self.store.usage().get('cpu_predictions', 0) - limit)))
        prediction_seconds = 0.0

        def evaluate_continuation(request):
            nonlocal prediction_seconds
            before = self.store.usage().get('cpu_seconds', 0.0)
            prediction = self.evaluate_planning_request(request, parent_request)
            prediction_seconds += self.store.usage().get('cpu_seconds', 0.0) - before
            return prediction

        timed_path_reference = None
        if replay_limit and not AblationEvidencePolicy(self.config.agent_ablation).without_graph:
            references = [row.get('prediction') for row in self.store.trials()
                if plan_signature(row['plan']['backend_config']) == plan_signature(parent_request)]
            prediction = next((row for row in references if row
                and row.get('cost_database_revision') == self.cost_revision
                and row.get('performance', {}).get('step_time_ms') is not None), None)
            remaining = self.config.budget.walltime_ceiling - self.store.search_seconds()
            if prediction is not None and remaining > 0 and self.load_prediction_graph(prediction):
                timed_path_reference = dict(request=parent_request, prediction=prediction,
                    evaluate=evaluate_continuation, load_graph=self.load_prediction_graph,
                    deadline=time.perf_counter() + min(remaining, self.config.investigation_max_seconds
                        if self.config.investigation_max_seconds is not None else float('inf')))
                replay_limit -= 1

        result = solver.solve(parent_request, values, evidence_id, fixed_values=fixed, limit=limit,
            allowed_ranges={} if intent is None else intent.allowed_ranges,
            module_relations=() if intent is None else intent.module_relations,
            preserve=() if intent is None else intent.preserve,
            max_states_per_boundary=self.config.conditional_partition_max_states,
            max_expansions=self.config.conditional_partition_max_expansions,
            max_layouts=self.config.conditional_partition_max_layouts,
            continuation_requests=continuation_requests,
            continuation_evaluator=evaluate_continuation if continuation_requests else None,
            max_continuation_evaluations=replay_limit, timed_path_reference=timed_path_reference)
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
        remaining = self.config.budget.cpu_prediction_ceiling - int(self.store.usage().get('cpu_predictions', 0))
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
        covered = set()
        coverage_first = []
        deferred = []
        for family in families:
            values = family['values']
            mechanism = (values['pp'], values['ep'], values['tp'], values['cp'])
            if mechanism in covered:
                deferred.append(family)
            else:
                covered.add(mechanism)
                coverage_first.append(family)
        families = [*coverage_first, *deferred]
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
                if self.store.usage().get('cpu_predictions', 0) >= self.config.budget.cpu_prediction_ceiling:
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
        catalog['initial_family_selection'] = 'unseen_cost_priority_with_parallel_mechanism_coverage'
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
            metadata = Path(path).stat()
            identity = (metadata.st_mtime_ns, metadata.st_size)
            cache = getattr(self, '_artifact_cost_keys', {})
            cached = cache.get(path)
            if cached is None or cached['identity'] != identity:
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    graph = json.load(stream)['execution_graph']
                graph_keys = frozenset(component['cost_key'] for task in (graph or {}).get('tasks', {}).values()
                    for component in task['metadata'].get('cost_components', []))
                cached = dict(identity=identity, keys=graph_keys)
                cache[path] = cached
                self._artifact_cost_keys = cache
            keys.update(cached['keys'])
        return keys

    def bottleneck_impact(self, trial, task_ids, epsilon=0.1):
        graph, scope = self.bottleneck_graph(trial)
        targets = graph.closure(task_ids, dependency_depth=0, include_resources=False)
        closure = graph.closure(task_ids, dependency_depth=1, include_resources=False)
        if scope != 'PREDICTED_COSTED_DAG':
            return dict(status='NEEDS_PROFILE', target_task_ids=targets['task_ids'],
                closure=graph.page(closure, limit=16, max_chars=6000), impact=None)
        resource_model = self.predictor.projector.parameters.get('resource_replay', {})
        baseline = graph.replay(resource_model=resource_model)
        closure = graph.causal_closure(targets['task_ids'], baseline, dependency_depth=1)
        impact = graph.epsilon_impact(targets['task_ids'], epsilon,
            resource_model=resource_model, baseline_replay=baseline)
        self.store.charge('impact:' + fingerprint(dict(graph=graph.graph_id,
            task_ids=task_ids, epsilon=epsilon)), 'counterfactual_replays', 2)
        return dict(status='AVAILABLE', target_task_ids=targets['task_ids'],
            closure=graph.page(closure, limit=16, max_chars=6000), impact=impact)

    def bottleneck_graph(self, trial):
        if trial.get('prediction') is not None:
            prediction = trial['prediction']
            projected = self.load_prediction_graph(prediction)
            if projected is not None and all(row.get('duration_ms') is not None
                    for row in projected['tasks'].values()):
                return (BottleneckGraph(projected,
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

    def planning_component_evidence(self, candidates, profiles):
        known = sorted((row for row in candidates if row['prediction']['performance'].get('step_time_ms') is not None
            and row['prediction'].get('execution_graph')),
            key=lambda row: row['prediction']['performance']['step_time_ms'])[:self.config.ranking_top_k]
        shared = {row['cost_key']: row for row in self.shared_ranking_probes(known)
            if row.get('status') == 'AVAILABLE'}
        cost_ranges = {key: [row['scenarios'][endpoint]['cost_value'] for endpoint in ('low', 'high')]
            for key, row in shared.items()}
        for trial in known:
            for row in self.full_prediction(trial)['evidence'].get('costs', []):
                value = row.get('milliseconds')
                spreads = [row[key] for key in ('stddev_ms', 'sampling_stddev_ms', 'uncertainty_ms')
                    if row.get(key) is not None and row[key] > 0]
                if value is None or not spreads or not row.get('evidence_ids'):
                    continue
                margin = max(spreads)
                key = (row.get('profiling_request') or {}).get('cost_key') or row.get('cost_key')
                if key is None:
                    continue
                interval = [max(0.0, value - margin), value + margin]
                prior = cost_ranges.get(key, interval)
                cost_ranges[key] = [min(prior[0], interval[0]), max(prior[1], interval[1])]
        components, modeled = {}, {}
        for trial in candidates:
            identifier = trial['trial_id']
            prediction = trial['prediction']
            needed = {row['cost_key']: row for row in prediction['evidence'].get('profiling_requests', [])
                if row.get('cost_key') and (row.get('profile_required_occurrence_count')
                    or row['kind'] == 'native_training_cost_profile')}
            if not needed:
                continue
            graph = self.load_decision_graph(prediction)
            tasks = (graph or {}).get('tasks', {})
            by_key = {}
            for task_id, task in tasks.items():
                for key in task['metadata'].get('missing_cost_keys', []):
                    by_key.setdefault(key, []).append(task_id)
            for key, request in needed.items():
                item = components.setdefault(key, dict(affected={}, range_ms=cost_ranges.get(key),
                    ranking_sensitivity=shared.get(key, {}).get('priority_class')))
                affected = by_key.get(key, request.get('affected_task_ids', []))
                item['affected'][identifier] = dict(task_count=len(affected), task_ids=affected[:24])
            if graph and cost_ranges and prediction['performance'].get('step_time_ms') is None:
                scenarios = BottleneckGraph(graph).replay_bounded_costs(cost_ranges,
                    config=configuration(trial['plan']['backend_config']),
                    resource_model=self.predictor.projector.parameters.get('resource_replay', {}))
                if scenarios['replay_count']:
                    modeled[identifier] = scenarios
                    charge = fingerprint(dict(candidate_id=identifier, ranges=cost_ranges,
                        cost_revision=self.cost_revision))
                    self.store.charge('partial-range:' + charge, 'counterfactual_replays', scenarios['replay_count'])
        result = []
        for profile in profiles:
            keys = profile['cost_keys']
            affected = set(profile['trial_ids'])
            priority = min((shared[key]['priority_class'] for key in keys if key in shared), default=4)
            for key in keys:
                affected.update(components.get(key, {}).get('affected', {}))
                affected.update(shared.get(key, {}).get('affected_candidate_ids', ()))
            span = {identifier: modeled[identifier]['step_time_scenario_ms'] for identifier in affected
                if identifier in modeled}
            duration_range = max((high - low for low, high in span.values()), default=None)
            result.append(dict(request_id=profile['request_id'], cost_keys=keys,
                rank_priority=priority, affected_candidate_ids=sorted(affected),
                affected_per_gpu_hour=len(affected) / profile['estimated_gpu_hours'],
                joint_scenario_width_ms=duration_range,
                estimated_gpu_hours=profile['estimated_gpu_hours']))
        result.sort(key=lambda row: (row['rank_priority'],
            -(row['joint_scenario_width_ms'] or 0) / row['estimated_gpu_hours'],
            -row['affected_per_gpu_hour'], row['request_id']))
        return dict(ordered_profiles=result, modeled_time_scenarios=modeled, missing_components=components)

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
