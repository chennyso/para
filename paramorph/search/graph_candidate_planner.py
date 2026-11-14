import json
import time

from contracts import RewriteIntent, fingerprint
from graph.strategy_impact import GraphStrategyImpact
from predictor.profile_database import plan_signature, signature
from search.complete_partition import CompletePartitionSearch
from search.complete_strategy import complete_values
from search.conditional_partition import ConditionalPartitionSolver
from search.experiment_value import experiment_value


class GraphGuidedCandidatePlanner:
    def __init__(self, search):
        self.search = search

    @staticmethod
    def visible(evidence, limit):
        ranked = [row for row in evidence if row['predicted_step_ms'] is not None]
        pending = [row for row in evidence if row['predicted_step_ms'] is None]
        keep = min(len(pending), limit - 1, max(1, limit // 3)) if ranked else limit
        chosen = ranked[:limit - keep] + pending[:keep]
        result = []
        for row in (chosen + ranked[limit - keep:] + pending[keep:])[:limit]:
            changes = row['bottleneck_changes']
            operations = changes['operations']
            ordered = sorted(operations, key=lambda item: (
                -abs(item['delta']['critical_tasks']),
                -abs(item['delta']['ready_wait_sum_ms']),
                -abs(item['delta']['gpu_work_ms']), item['operation']))
            result.append(row | dict(
                affected_task_ids={side: ids[:6] for side, ids in row['affected_task_ids'].items()},
                missing_cost_keys=row['missing_cost_keys'][:8],
                missing_cost_key_count=len(row['missing_cost_keys']),
                bottleneck_changes=changes | dict(operations=ordered[:4], operation_count=len(operations))))
        return result

    def prepare(self, parent, families):
        search = self.search
        baseline = parent['plan']['backend_config']
        reference = search._planning_projection_reference
        available = search.config.budget.cpu_prediction_ceiling - int(search.store.usage().get('cpu_predictions', 0))
        allowance = min(available, search.config.budget.max_candidates_per_round)
        limit = search.config.investigation_max_seconds
        deadline = time.perf_counter() + min(limit if limit is not None else float('inf'),
            max(0.0, search.config.budget.walltime_ceiling - search.store.search_seconds()))
        existing = {plan_signature(row['plan']['backend_config']): row for row in search.store.trials()}
        profiles = [json.loads(row[0]) for row in search.predictor.database.connection.execute('SELECT document FROM profiles')]
        solver = ConditionalPartitionSolver(search.rewriter,profiles,backend_source_hashes=search.backend_identity,
            backend_source_resolver=search.native_profile_source_hashes)
        evaluated, rejections, scopes = {}, [], {}

        def assess(request):
            nonlocal allowance
            key = plan_signature(request)
            if key in evaluated:
                return evaluated[key]['prediction']
            if allowance <= 0 or time.perf_counter() >= deadline:
                return None
            trial = existing.get(key)
            if trial and (trial.get('failure') or trial.get('measurement')):
                return None
            prediction = trial.get('prediction') if trial else None
            if prediction is None or prediction.get('cost_database_revision') != search.cost_revision:
                prediction = search.evaluate_planning_request(request, reference)
                if prediction is None:
                    return None
                allowance -= 1
            if prediction['memory']['admission'] == 'REJECT_OOM':
                rejections.append(dict(plan_id=key, reason=prediction['memory'].get('rejection_basis'),
                    prediction_id=prediction['prediction_id']))
            if trial is None:
                intent = RewriteIntent(operation='JointTransform', values=complete_values(request),
                    adjustable=[], preserve=[], parent_plan_id=baseline['run_id'],
                    hypothesis='在共同负载和原生约束下比较联合策略的资源感知执行图',
                    evidence_refs=[parent['prediction']['prediction_id']])
                rewrite, = search.rewriter.complete_and_rewrite(baseline, intent, limit=1, allow_unchanged=True)
                rewrite['proposal_kind'] = 'graph_guided_preselection'
                trial = search.register_constraint_candidate(rewrite, parent, intent)
            if trial.get('prediction') != prediction:
                trial = search.store.update_trial(trial['trial_id'], prediction=prediction,
                    validation=(trial.get('validation') or {}) | prediction['validation'], state='PROJECTED')
                search.family_ledger.record_trial(trial, search.cost_revision)
            evaluated[key] = trial
            return prediction

        # 轮换未评价的执行条件；已有 Profile 只决定能否使用带成本的分区目标。
        all_rows, unique = [], set()
        for family in families:
            request = family['request']
            key = signature(request)
            if key not in unique:
                all_rows.append(request)
                unique.add(key)
        queues = {}
        for request in all_rows:
            queues.setdefault((request['pp'], request['tp'], request['optimizer_sharding']), []).append(request)
        ordered = [row for group in sorted(queues) for row in queues[group]]
        depths = {group: 0 for group in queues}
        order = {}
        for request in ordered:
            group = (request['pp'], request['tp'], request['optimizer_sharding'])
            order[plan_signature(request)] = depths[group], group
            depths[group] += 1
        for request in sorted(ordered, key=lambda row: (
                (existing.get(plan_signature(row), {}).get('prediction') or {}).get('cost_database_revision')
                    == search.cost_revision, order[plan_signature(row)], plan_signature(row))):
            if allowance <= 0 or time.perf_counter() >= deadline:
                break
            prediction = assess(request)
            if prediction is None or prediction['memory']['admission'] == 'REJECT_OOM':
                continue
            key = plan_signature(request)
            if request['pp'] * request['vpp'] == 1:
                scopes[key] = 'FIXED_LAYOUT'
                continue
            if search.config.planning_partition_engine=='frontier':
                scopes[key] = dict(status='PRICING_PROGRAM_SELECTED_BY_AGENT',
                    engine='frontier',complete_reference_prediction_id=prediction['prediction_id'])
                continue
            profile = solver._profile(request)
            deadline_for_shape = min(deadline, time.perf_counter() + 2) if limit is not None else deadline
            report = CompletePartitionSearch(search.model.num_layers, request, profile, (), assess,
                deadline_for_shape, [request], {key: prediction['performance'].get('step_time_ms') or float('inf')},
                reports={key: prediction}, graph_loader=search.load_prediction_graph,
                engine='path').run(1, 1)
            scopes[key] = dict(statuses=report['statuses'], evaluations=report['evaluations'])
        history = search.store.trials()
        candidates = [row for row in history if not row.get('failure') and row.get('measurement') is None
            and (row.get('rewrite') or {}).get('candidate_source') == 'graph_guided_preselection'
            and (row.get('prediction') or {}).get('cost_database_revision') == search.cost_revision
            and row['prediction']['memory']['admission'] != 'REJECT_OOM']
        measured = search.store.best_measured()
        leader = None if measured is None else measured['measurement']['tokens_per_second']
        evidence = [self.describe(parent, trial, history, leader) for trial in candidates]
        evidence.sort(key=lambda row: (row['predicted_step_ms'] is None,
            -(row['expected_step_savings_ms_per_gpu_hour'] or 0),
            row['predicted_step_ms'] if row['predicted_step_ms'] is not None else float('inf'),
            row['candidate_id']))
        report = dict(kind='deterministic_graph_candidate_portfolio', candidate_evidence=evidence,
            rejected=rejections, layouts=scopes, enumerated_execution_conditions=len(all_rows),
            distinct_evaluated_conditions=len({signature(row['plan']['backend_config']) for row in history
                if (row.get('prediction') or {}).get('cost_database_revision') == search.cost_revision}),
            cost_revision=search.cost_revision)
        identity = fingerprint(report)
        search.store.add_evidence(identity, 'PREDICTED', report)
        return evidence, identity

    def describe(self, parent, trial, history, leader):
        search = self.search
        before, after = parent['prediction'], trial['prediction']
        before_graph = search.load_prediction_graph(before)
        after_graph = search.load_decision_graph(after)
        impact = GraphStrategyImpact.compare_candidates(parent['plan']['backend_config'], before, before_graph,
            trial['plan']['backend_config'], after, after_graph)
        missing = [row for row in after['evidence'].get('profiling_requests', [])
            if row.get('profile_required_occurrence_count', 0) or row['kind'] == 'native_training_cost_profile']
        request = trial['plan']['backend_config']
        value = experiment_value(request, after, history, leader, search.config.budget.estimated_trial_seconds)
        return dict(candidate_id=trial['trial_id'], execution_condition=signature(request),
            complete_strategy=complete_values(request),
            predicted_step_ms=after['performance'].get('step_time_ms'), memory_admission=after['memory']['admission'],
            affected_task_ids=impact['task_ids'], affected_task_count=impact['changed_task_count'],
            bottleneck_changes=impact['changes'],
            missing_cost_keys=sorted({row['cost_key'] for row in missing if row.get('cost_key')}),
            prediction_uncertainty=dict(interval_ms=after['performance'].get('prediction_interval_ms'),
                scope=after['performance'].get('prediction_uncertainty'), comparison_status=impact['status']),
            estimated_measurement_gpu_hours=value['expected_gpu_hours'],
            expected_step_savings_ms_per_gpu_hour=value['expected_step_savings_ms_per_gpu_hour'])
