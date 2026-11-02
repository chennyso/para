from copy import deepcopy


def compact_query_arguments(value):
    if isinstance(value, dict):
        return {key: compact_query_arguments(item) for key, item in value.items()
                if item is not None and item != [] and item != {}}
    if isinstance(value, list):
        return [compact_query_arguments(item) for item in value]
    return value


def pack_context(observation, maximum_tokens, count_tokens):
    context = deepcopy(observation)
    deferred = []
    if isinstance(context.get('deferred_context'), list) and len(context['deferred_context']) > 32:
        context['deferred_context'] = context['deferred_context'][-32:]
    limits = [('tool_history', 1, 'original_tool_query'),
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
        deferred.append(reference)
        context['deferred_context'] = deferred
    after = count_tokens(context)
    if after > maximum_tokens:
        raise ValueError(dict(status='CONTEXT_BUDGET_EXCEEDED',
            observation_tokens=after, maximum_observation_tokens=maximum_tokens,
            section_tokens={key: count_tokens(value) for key, value in context.items()}))
    return context, dict(original_observation_tokens=before,
        packed_observation_tokens=after, deferred_records=deferred,
        protected_fields=['current_training_plan', 'diagnosis', 'performance',
                          'candidates', 'parallel_legality', 'previous_response'])
