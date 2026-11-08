from copy import deepcopy


def role_observation(role, observation):
    context = deepcopy(observation)
    deferred = []
    for row in context.get('recent_search_rounds', []):
        if row.get('feedback') is not None and row.get('feedback_evidence_id'):
            row.pop('feedback')
            row['feedback_access'] = dict(tool='get_trial_history', evidence_id=row['feedback_evidence_id'])
            deferred.append('recent_search_rounds.feedback')
    if role == 'JointStrategyAgent' and isinstance(context.get('global_family_search'), dict):
        catalog = context['global_family_search']
        if catalog.get('catalog_access'):
            context['global_family_search'] = {key: catalog[key] for key in
                ('total_legal_families', 'family_fields', 'catalog_access', 'evidence_id',
                 'coverage', 'candidate_count') if key in catalog}
            deferred.append('global_family_search.reference_plans')
    previous = context.get('previous_response')
    history = context.get('tool_history', [])
    if isinstance(previous, dict) and previous.get('tool_requests') and history:
        executed = [compact_query_arguments(row['tool']) for row in history if isinstance(row.get('tool'), dict)]
        remaining = [query for query in previous['tool_requests']
            if compact_query_arguments(query) not in executed]
        if len(remaining) != len(previous['tool_requests']):
            previous['tool_requests'] = remaining
            deferred.append('previous_response.executed_tool_requests')
    context['request_observation_profile'] = dict(version='role_decision_v1', role=role,
        deferred_fields=sorted(set(deferred)), history_details_tool='get_trial_history')
    return context


def compact_query_arguments(value):
    if isinstance(value, dict):
        return {key: compact_query_arguments(item) for key, item in value.items()
                if item is not None and item != [] and item != {}}
    if isinstance(value, list):
        return [compact_query_arguments(item) for item in value]
    return value


def compact_role_observation(role, observation):
    context = role_observation(role, observation)
    changed = False
    for row in context.get('recent_search_rounds', []):
        if row.get('feedback_access'):
            row['feedback_access']['history_view'] = 'rounds'
    catalog = context.get('evidence_catalog')
    if isinstance(catalog, dict) and len(catalog.get('recent_note_evidence_refs', [])) > 4:
        count = len(catalog.pop('recent_note_evidence_refs'))
        catalog['historical_notes_access'] = dict(tool='get_trial_history',
            history_view='notes', evidence_reference_count=count,
            result_field='notes', pagination='offset_and_limit')
        changed = True
    memory = context.get('measured_strategy_memory')
    if isinstance(memory, dict) and 'columns' in memory and 'rows' in memory:
        columns = memory['columns']
        removed = [name for name in ('process_groups', 'module_ownership') if name in columns]
        if removed:
            changed = True
            keep = [index for index, name in enumerate(columns) if name not in removed]
            memory['rows'] = [[row[index] for index in keep] for row in memory['rows']]
            memory['columns'] = [columns[index] for index in keep]
            memory['structure_details_access'] = dict(tool='get_trial_history',
                history_view='structure',
                candidate_id_column='trial_id', deferred_fields=removed,
                scope='historical_process_groups_and_module_ownership')
    failures = {row['trial_id']: row for row in context.get('conditional_failure_memory', [])
        if isinstance(row, dict) and 'trial_id' in row}
    for index, row in enumerate(context.get('recent_history', [])):
        if row.get('trial_id') in failures and failures[row['trial_id']] == row:
            changed = True
            context['recent_history'][index] = dict(trial_id=row['trial_id'],
                same_record_in='conditional_failure_memory')
    receipt = context.get('backend_completion_receipt')
    if isinstance(receipt, dict) and 'result' in receipt:
        for row in context.get('tool_history', []):
            if (row.get('evidence_id') == receipt.get('evidence_id')
                    and row.get('result') == receipt['result']):
                row.pop('result')
                row['result_in'] = 'backend_completion_receipt.result'
                changed = True
    if changed:
        context['request_observation_profile'].update(version='role_compact_v2',
            historical_structure_details_tool='get_trial_history',
            reference_semantics='same_record_in和result_in引用当前观测中已完整提供的记录')
    return context


def columnar_records(value):
    if isinstance(value, dict):
        return {key: columnar_records(item) for key, item in value.items()}
    if not isinstance(value, list):
        return value
    rows = [columnar_records(item) for item in value]
    if (len(value) >= 2 and all(isinstance(item, dict) for item in value)
            and all(set(item) == set(value[0]) for item in value)):
        columns = list(value[0])
        return dict(representation='columnar_records_v1', columns=columns,
            rows=[[row[column] for column in columns] for row in rows])
    return rows


def pack_context(observation, maximum_tokens, count_tokens):
    if maximum_tokens is None:
        maximum_tokens = float('inf')
    context = deepcopy(observation)
    deferred = []
    prior = context.get('prior_episode')
    if isinstance(prior, dict) and isinstance(prior.get('strategies'), list):
        prior['strategy_total_count'] = len(prior['strategies'])
        while len(prior['strategies']) > 1 and count_tokens(context) > maximum_tokens:
            removed = prior['strategies'].pop(0)
            deferred.append(dict(field='prior_episode.strategies',
                reference=removed['source_trial_id'], access='registered_cross_node_history'))
    recent = context.get('recent')
    if isinstance(recent, dict) and isinstance(recent.get('rows'), list):
        recent['total_count'] = len(recent['rows'])
        while len(recent['rows']) > 2 and count_tokens(context) > maximum_tokens:
            removed = recent['rows'].pop()
            deferred.append(dict(field='recent.rows', reference=removed[0], access='get_trial_history'))
    view = context.get('execution_view')
    if count_tokens(context) > maximum_tokens and isinstance(view, dict) and view.get('evidence_id'):
        facts = view.get('facts', {})
        sequences = facts.pop('native_compute_sequences', None)
        if sequences is not None:
            facts['native_compute_sequence_samples'] = [dict(pp_rank=stage,
                physical_ranks=ranks, event_count=len(events),
                first_events=events[:2], last_events=events[-2:])
                for stage, ranks, events in sequences]
            facts['schedule_sample_scope'] = 'first_and_last_two_events_per_stage_not_complete_schedule'
        dependencies = facts.pop('dependency_templates', None)
        if dependencies is not None:
            facts['dependency_template_samples'] = dependencies[:8]
            facts['dependency_template_count'] = len(dependencies)
        groups = facts.pop('physical_groups', None)
        if groups is not None:
            identities, catalog, references = {}, [], []
            for kind, members, cross_node in groups:
                identity = (tuple(members), cross_node)
                if identity not in identities:
                    identities[identity] = len(catalog)
                    catalog.append(dict(group_id=identities[identity],
                        physical_members=members, cross_node=cross_node))
                references.append([kind, identities[identity]])
            facts['physical_group_catalog'] = catalog
            facts['physical_group_references'] = references
            facts['group_columns'] = ['kind', 'group_id']
        if sequences is not None or dependencies is not None:
            deferred.append(dict(field='execution_view.facts',
                reference=dict(evidence_id=view['evidence_id'],
                    trial_id=view['trial_id'], source_graph_sha256=view['source_graph_sha256']),
                access='inspect_dag_subgraph'))
            view['complete_schedule_access'] = dict(tool='inspect_dag_subgraph',
                candidate_ids=[view['trial_id']], evidence_id=view['evidence_id'])
    if isinstance(context.get('deferred_context'), list) and len(context['deferred_context']) > 32:
        context['deferred_context'] = context['deferred_context'][-32:]
    limits = [('proposal_archive', 1, 'inspect_trial_history'),
              ('policy_comparisons', 1, 'inspect_policy_frontier'),
              ('execution_revisions', 1, 'inspect_execution_impact'),
              ('tool_history', 1, 'original_tool_query'),
              ('recent_history', 2, 'get_trial_history'),
              ('recent_search_rounds', 2, 'get_trial_history'),
              ('recent_profiling_results', 1, 'inspect_region_cost')]
    before = count_tokens(context)
    for row in context.get('tool_history', []):
        if not isinstance(row.get('tool'), dict):
            continue
        original = row['tool']
        compact = compact_query_arguments(original)
        for field in ('transformation_intent', 'transformation_intents'):
            values = compact.get(field)
            if values is None:
                continue
            entries = values if isinstance(values, list) else [values]
            entries = [{key: entry[key] for key in ('operation', 'values', 'parent_plan_id')
                        if key in entry} for entry in entries]
            compact[field] = entries if isinstance(values, list) else entries[0]
        if compact != original:
            row['tool'] = compact
            deferred.append(dict(field='tool_history.query_arguments',
                reference=dict(evidence_id=row.get('evidence_id'), tool=original.get('tool')),
                access='backend_original_tool_receipt'))
    for index, row in enumerate(context.get('profiling_results', [])):
        if count_tokens(row) <= 1500:
            continue
        result = row.get('result') or {}
        measurement = result.get('measurement') or {}
        reference = {key: row[key] for key in ('request_id', 'state', 'measurement_type') if key in row}
        context['profiling_results'][index] = dict(reference,
            cost_key_count=len(row.get('cost_keys', [])),
            result={key: result[key] for key in ('profile_id', 'status') if key in result},
            measurement={key: measurement[key] for key in
                ('run_id', 'status', 'returncodes', 'gpu_allocation_seconds', 'raw_artifacts')
                if key in measurement},
            communication_record_count=len(measurement.get('nccl_benchmark', {}).get('records', [])),
            details_access=['inspect_region_cost', 'inspect_communication_flow'])
        deferred.append(dict(field='profiling_results', reference=reference,
            access='backend_profile_record'))
    if deferred:
        context['deferred_context'] = deferred
    previous = context.get('previous_selection')
    if isinstance(previous, dict) and count_tokens(context) > maximum_tokens:
        context['previous_selection'] = {key: value for key, value in previous.items()
            if key not in ('reason', 'missing_measurements')}
        deferred.append(dict(field='previous_selection.reason_and_missing_measurements',
            reference=dict(evidence_id=context.get('previous_selection_evidence_id')),
            access='get_trial_history'))
        context['deferred_context'] = deferred
    for field, minimum, access in limits:
        rows = context.get(field)
        if not isinstance(rows, list):
            continue
        while len(rows) > minimum and count_tokens(context) > maximum_tokens:
            removed = rows.pop(0)
            reference = {key: removed[key] for key in
                ('evidence_id', 'trial_id', 'request_id', 'plan_hash', 'round_index')
                if key in removed}
            if field == 'tool_history':
                reference['query'] = removed.get('tool')
            deferred.append(dict(field=field, reference=reference, access=access))
            context['deferred_context'] = deferred
    compact_fields = [
        ('pending_profiling_requests', 'backend_profile_request'),
        ('candidate_execution_effects', 'inspect_rank_topology'),
        ('measured_strategy_memory', 'get_trial_history'),
        ('global_family_search', 'inspect_global_families'),
        ('conditional_failure_memory', 'inspect_memory'),
        ('communication_placement', 'inspect_communication_flow'),
        ('execution_structure', 'inspect_rank_topology'),
        ('model_module_sequence', 'inspect_region'),
        ('legal_actions', 'inspect_global_families'),
        ('recent_search_rounds', 'get_trial_history'),
        ('queried_global_families', 'inspect_global_families'),
        ('tool_history', 'backend_original_tool_receipt'),
        ('query_receipts', 'backend_original_tool_receipt'),
        ('previous_response', 'backend_previous_agent_response'),
        ('deferred_context', 'backend_original_context'),
    ]
    for field, access in compact_fields:
        if count_tokens(context) <= maximum_tokens or field not in context:
            continue
        value = context[field]
        reference = dict(field=field, access=access)
        if isinstance(value, list):
            reference['record_count'] = len(value)
            reference['sample_ids'] = [row.get('family_id', row.get('trial_id', row.get('round_index')))
                for row in value[:4] if isinstance(row, dict)]
        elif isinstance(value, dict):
            reference['keys'] = sorted(value)[:12]
        else:
            reference['value_type'] = type(value).__name__
        context[field] = dict(summary=reference)
        if field == 'deferred_context':
            deferred = [reference]
        else:
            deferred.append(reference)
        context['deferred_context'] = deferred
    representation_changes = []
    for field in ('performance', 'hardware_topology', 'execution_analysis',
            'candidate_decision_report', 'complete_strategy_comparison', 'execution',
            'execution_revisions', 'policy_comparisons', 'recent_search_outcomes'):
        if field not in context:
            continue
        original_tokens = count_tokens(context[field])
        compacted = columnar_records(context[field])
        compacted_tokens = count_tokens(compacted)
        if compacted_tokens < original_tokens:
            context[field] = compacted
            representation_changes.append(dict(field=field,
                representation='columnar_records_v1', tokens_saved=original_tokens - compacted_tokens))
    after = count_tokens(context)
    if after > maximum_tokens and isinstance(prior, dict) and prior.get('strategies'):
        prior.pop('strategies')
        prior['strategy_access'] = 'registered_cross_node_history'
        after = count_tokens(context)
    for field in ('request_observation_profile', 'ablation_contract', 'deferred_context', 'prior_episode', 'reference_format'):
        if after > maximum_tokens and field in context:
            context.pop(field)
            after = count_tokens(context)
    if after > maximum_tokens:
        raise ValueError(dict(status='CONTEXT_BUDGET_EXCEEDED',
            observation_tokens=after, maximum_observation_tokens=maximum_tokens,
            section_tokens={key: count_tokens(value) for key, value in context.items()}))
    return context, dict(original_observation_tokens=before,
        packed_observation_tokens=after, deferred_records=deferred,
        representation_changes=representation_changes,
        protected_fields=['current_training_plan', 'diagnosis', 'performance',
                          'candidates', 'parallel_legality',
                          'execution_analysis', 'candidate_decision_report'])
