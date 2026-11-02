import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from openai.lib._pydantic import to_strict_json_schema

from contracts import TrainingPlan, fingerprint
from predictor.profile_database import ProfileDatabase, normalized_execution, plan_signature
from search.family_state import GLOBAL_FIELDS, global_identity
from search.system_search import (
    BottleneckDecision,
    ConstraintValidationDecision,
    ConstraintValidationToolRequest,
    EvidenceGuidedSearch,
    IntentBatch,
    InvestigationSelection,
    InvestigationToolRequest,
    token_count,
)
from storage.experiment_store import ExperimentStore


def backup_database(source, target):
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(target) as destination:
            original.backup(destination)


def import_saved_records(source, target):
    destination = ExperimentStore(target)
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as original:
        for table in ('trials', 'evidence'):
            records = original.execute(f'SELECT * FROM {table}').fetchall()
            destination.connection.executemany(f'INSERT INTO {table} VALUES (?, ?, ?)', records)
    destination.connection.commit()
    destination.connection.close()


def restore_recorded_precision(config, evidence_path, database):
    evidence = json.loads(evidence_path.read_text())
    log_path = Path(evidence['source_log'])
    content = log_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == evidence['sha256']
    assert evidence['execution_backend'] == 'megatron_ddp'
    fields = evidence['fields']
    assert set(fields) == {'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'}
    line = content.decode('utf-8').splitlines()[evidence['optimizer_config_line'] - 1]
    assert 'Setting up optimizer with config OptimizerConfig(' in line
    for name, dtype in fields.items():
        assert f'{name}=torch.{dtype},' in line
    reference = config['request']
    immutable = ('model', 'checkpoint', 'data', 'precision', 'optimizer', 'precision_aware_optimizer')
    restored = []
    with sqlite3.connect(database) as connection:
        recorded = connection.execute('SELECT document FROM trials WHERE id=?',
            (evidence['request_id'],)).fetchone()
        assert recorded is not None
        recorded_request = json.loads(recorded[0])['plan']['backend_config']
        assert plan_signature(reference) == plan_signature(recorded_request)
        for identifier, encoded in connection.execute('SELECT id, document FROM trials').fetchall():
            trial = json.loads(encoded)
            request = trial['plan']['backend_config']
            if request.get('execution_backend', 'megatron_ddp') != 'megatron_ddp':
                continue
            if any(normalized_execution(request).get(key) != normalized_execution(reference).get(key)
                    for key in immutable):
                continue
            assert all(request.get(key, value) == value for key, value in fields.items())
            previous = plan_signature(request)
            request.update(fields)
            # 历史对象保持原有Schema，精度恢复保持完整执行签名。
            assert plan_signature(request) == previous
            connection.execute('UPDATE trials SET document=? WHERE id=?',
                (json.dumps(trial, allow_nan=False), identifier))
            restored.append(identifier)
    reference.update(fields)
    receipt = dict(source=evidence, restored_trial_ids=restored,
        scope='explicit_recorded_ddp_precision_with_unchanged_plan_identity')
    store = ExperimentStore(database)
    store.add_evidence(fingerprint(receipt), 'MEASURED', receipt)
    store.connection.close()
    return receipt


def read_strategy(search):
    identifier = '79b7c68a2aa655b18f8b726167cf39721b2dff95d91868d74dea47785d66484c'
    record = search.store.connection.execute('SELECT document FROM evidence WHERE id=?',
        (identifier,)).fetchone()
    if record is None:
        raise ValueError('验收需要已有MiMo Agent策略证据')
    return identifier, IntentBatch.model_validate(json.loads(record[0]))


def verify_constraints(search, strategy_id, strategy):
    trials = {trial['trial_id']: trial for trial in search.store.trials()}
    parent = trials[strategy.intents[0].parent_plan_id]
    request = ConstraintValidationToolRequest(tool='solve_conditional_partition',
        parent_trial_id=parent['trial_id'], strategy_evidence_id=strategy_id,
        intent_indices=[0, 2], limit=2)
    receipt = search.constraint_tool(request)
    first_counts = (len(search.store.trials()), search.store.usage())
    repeated = search.constraint_tool(request)
    assert receipt == repeated
    assert first_counts == (len(search.store.trials()), search.store.usage())
    assert receipt['result_tokens'] <= 4000
    current = {trial['trial_id']: trial for trial in search.store.trials()}
    normalized_parent = normalized_execution(parent['plan']['backend_config'])
    outcomes = []
    for result in receipt['result']['intent_results']:
        assert result['status'] == 'COMPLETED', result
        intent = strategy.intents[result['intent_index']].executable()
        for candidate_id in result['candidate_ids']:
            candidate = current[candidate_id]
            plan = TrainingPlan.model_validate(candidate['plan'])
            backend = plan.backend_config
            normalized = normalized_execution(backend)
            search.require_fixed_training_semantics(parent['plan']['backend_config'], backend)
            search.rewriter.candidate(backend)
            for key in intent.preserve:
                actual = backend.get(key, normalized.get(key))
                expected = parent['plan']['backend_config'].get(key, normalized_parent.get(key))
                assert actual == expected, (key, actual, expected)
            for key, values in intent.allowed_ranges.items():
                assert backend.get(key, normalized.get(key)) in values, (key, values, backend.get(key))
            with gzip.open(candidate['rewrite']['artifact'], 'rt', encoding='utf-8') as stream:
                artifact = json.load(stream)
            tasks = artifact['execution_graph']['tasks']
            assert tasks and set(range(16)) == {task['rank'] for task in tasks.values()}
            assert all(dependency in tasks for task in tasks.values() for dependency in task['dependencies'])
            assert sum(backend['decoder_chunks']) == search.model.num_layers
            assert backend['tp'] * backend['pp'] * backend['cp'] * backend['dp'] == 16
            outcomes.append(dict(intent_index=result['intent_index'], candidate_id=candidate_id,
                plan_hash=plan.plan_hash,
                family_id=global_identity(backend), tp=backend['tp'], pp=backend['pp'],
                dp=backend['dp'], vpp=backend['vpp'], group=backend['group'],
                decoder_chunks=backend['decoder_chunks'],
                task_count=len(tasks), preserve_checked=list(intent.preserve),
                allowed_ranges_checked=list(intent.allowed_ranges)))
    assert {row['pp'] for row in outcomes} == {2, 4}
    assert len({row['plan_hash'] for row in outcomes}) == len(outcomes)
    candidate_ids = [row['candidate_id'] for row in outcomes]
    validation = search.constraint_tool(ConstraintValidationToolRequest(
        tool='validate_training_plans', parent_trial_id=parent['trial_id'],
        strategy_evidence_id=strategy_id, candidate_ids=candidate_ids, limit=8))
    assert all(row['status'] == 'PASS' for row in validation['result']['validations'])
    return dict(source_strategy_evidence=strategy_id, outcomes=outcomes,
        completion_result_tokens=receipt['result_tokens'], repeated_call_is_idempotent=True,
        validation_result=validation['result'])


def verify_family_catalog(search):
    reference = search.reference_trial()
    families = search.joint_global_families(reference['plan']['backend_config'])
    assert all(set(row['values']) == set(GLOBAL_FIELDS) for row in families)
    assert len({row['family_id'] for row in families}) == len(families)
    mappings = {tuple(row['values']['rank_mapping']) for row in families}
    assert len(mappings) > 1
    assert any(row['state_variant_count'] > 1 for row in families)
    first = reference['plan']['backend_config']
    alternative_state = dict(first, optimizer_sharding='replicated')
    assert global_identity(first) == global_identity(alternative_state)
    return dict(family_count=len(families), rank_mapping_count=len(mappings),
        maximum_state_variants=max(row['state_variant_count'] for row in families),
        family_fields=list(GLOBAL_FIELDS), state_choices_share_global_family=True)


def verify_dag_queries(search, prediction_path):
    archived = json.loads(prediction_path.read_text())
    trial_id = archived['candidate_id']
    trial = next(row for row in search.store.trials() if row['trial_id'] == trial_id)
    trial = search.predict_trial(trial, candidate_graph=search.backend_dag(trial), force=True)
    prediction = trial['prediction']
    assert prediction['cost_database_revision'] == search.cost_revision
    if search.load_prediction_graph(prediction) is None:
        assert prediction['prediction_status'] == 'NEEDS_PROFILE'
        assert prediction['performance']['step_time_ms'] is None
        try:
            search.bottleneck_graph(trial)
        except ValueError as error:
            assert isinstance(error.args[0], dict)
            assert error.args[0]['status'] == 'NEEDS_PROFILE'
        else:
            raise AssertionError('缺少成本时没有拒绝带成本图查询')
        return dict(status='NEEDS_PROFILE', trial_id=trial_id,
            prediction_id=prediction['prediction_id'], cost_revision=search.cost_revision,
            missing_profiles=prediction['evidence']['missing_profiles'],
            profiling_requests=prediction['evidence'].get('profiling_requests', []),
            verification_scope='current_costed_dag_not_available',
            missing_cost_query_fails_explicitly=True)
    persisted = search.persist_prediction_graph(prediction)
    trial = search.store.update_trial(trial_id, prediction=persisted)
    search.family_ledger.record_trial(trial, search.cost_revision)
    query, scope = search.bottleneck_graph(trial)
    assert scope == 'PREDICTED_COSTED_DAG'
    shared = [(key, row) for key, row in query.records.items()
        if key in query.collective_keys and len(query._participants(key)) > 1]
    assert shared
    seed, _ = max(shared, key=lambda item: item[1]['duration_ms'])
    closure = query.closure([seed], dependency_depth=1, include_resources=True)
    expected = set(closure['task_ids'])
    recovered = set()
    offsets = []
    maximum_tokens = 0
    offset = 0
    while offset is not None:
        receipt = search.investigate(InvestigationToolRequest(tool='inspect_dag_subgraph',
            candidate_ids=[trial_id], task_ids=[seed], dependency_depth=1,
            offset=offset, limit=8), [trial])
        result = receipt['result']
        assert result['closure_id'] == closure['closure_id']
        assert result['complete_graph_task_count'] == len(query.records)
        assert result['source_scope'] == 'original_complete_executable_dag'
        assert len(result['tasks']) <= 8
        maximum_tokens = max(maximum_tokens, receipt['result_tokens'])
        assert receipt['result_tokens'] <= 4000
        identifiers = {row['task_id'] for row in result['tasks']}
        assert not identifiers & recovered
        recovered.update(identifiers)
        offsets.append(offset)
        offset = result['next_offset']
    assert recovered == expected
    assert len({query.records[key]['rank'] for key in recovered}) > 1
    participants = set(query._participants(seed))
    collective = query.collective_tasks[query.collective_keys[seed]]
    assert collective <= recovered
    assert participants <= {query.records[key]['rank'] for key in collective}
    return dict(source_prediction=str(prediction_path), trial_id=trial_id,
        archived_cost_revision=archived['cost_database_revision'],
        current_cost_revision=prediction['cost_database_revision'],
        complete_task_count=len(query.records), seed_task_id=seed,
        closure_task_count=len(expected), page_count=len(offsets),
        participant_ranks=sorted(participants), maximum_tool_tokens=maximum_tokens,
        all_closure_tasks_recovered=True, collective_participants_preserved=True)


def verify_ledger(search, alternative_database):
    valid = [row for row in search.store.trials() if row.get('measurement')
        and row['measurement']['status'] == 'PASS'
        and row['measurement']['performance_eligible']]
    assert valid
    for trial in valid:
        search.family_ledger.record_trial(trial, search.cost_revision)
    expected = max(valid, key=lambda row: row['measurement']['tokens_per_second'])
    records = search.family_ledger.records()
    best = max((row['best_measured'] for row in records if row['best_measured']),
        key=lambda row: row['measured_tokens_per_s'])
    assert best['trial_id'] == expected['trial_id']
    assert math.isclose(best['measured_tokens_per_s'], expected['measurement']['tokens_per_second'])
    measured_before = {row['family_id']: row['best_measured'] for row in records}
    predicted = [entry for row in records for entry in row['candidates'].values()
        if entry['predicted_step_ms'] is not None and entry['cost_revision'] == search.cost_revision]
    assert predicted
    other = ProfileDatabase(alternative_database)
    alternative_revision = other.cost_database_revision()
    other.connection.close()
    assert alternative_revision != search.cost_revision
    changed = search.family_ledger.invalidate_costs(alternative_revision)
    after = search.family_ledger.records()
    assert changed
    assert all(row['best_predicted'] is None and row['lower_bound_ms'] is None
        and not row['search_complete'] for row in after)
    assert all(row['best_measured'] == measured_before[row['family_id']] for row in after)
    assert all(entry['cost_revision'] != alternative_revision for row in after
        for entry in row['candidates'].values())
    search.family_ledger.invalidate_costs(search.cost_revision)
    for trial in search.store.trials():
        search.family_ledger.record_trial(trial, search.cost_revision)
    return dict(best_measured_trial_id=best['trial_id'],
        best_measured_tokens_per_s=best['measured_tokens_per_s'],
        actual_cost_revision=search.cost_revision, alternative_actual_cost_revision=alternative_revision,
        invalidated_families=len(changed), measured_results_preserved=True,
        current_prediction_count=len(predicted), previous_predictions_excluded_after_revision_change=True)


def verify_agent_contexts(search, strategy, strategy_id, constraints):
    trials = {row['trial_id']: row for row in search.store.trials()}
    parent = trials[strategy.intents[0].parent_plan_id]
    candidates = [search.predict_trial(trials[row['candidate_id']]) for row in constraints['outcomes']]
    assert len(candidates) >= 2
    assert all(row['prediction']['cost_database_revision'] == search.cost_revision for row in candidates)
    print(json.dumps(dict(stage='new_candidate_prediction', predictions=len(candidates)),
        ensure_ascii=False), flush=True)
    diagnosis_id = '001f1da8db485702dd90fdbfc43279c71013df2616d18b363a820d27c4aa0870'
    document = search.store.connection.execute('SELECT document FROM evidence WHERE id=?',
        (diagnosis_id,)).fetchone()
    assert document
    diagnosis = BottleneckDecision.model_validate(json.loads(document[0]))
    interaction_id = 'df9c223433e1b678f3648c930afac893fa59a52692d2e72f1c9e68931005ee49'
    document = search.store.connection.execute('SELECT document FROM evidence WHERE id=?',
        (interaction_id,)).fetchone()
    assert document
    interactions = json.loads(document[0])
    contexts = dict(
        PerformanceAnalysisAgent=search.performance_agent_context(parent),
        JointStrategyAgent=search.strategy_agent_context(parent, diagnosis, diagnosis_id,
            search.global_family_catalog(parent)),
        ConstraintValidationAgent=search.constraint_agent_context(parent, strategy, strategy_id),
        ExperimentInvestigationAgent=search.investigation_agent_context(
            parent, candidates, strategy, interactions, interaction_id))
    schemas = dict(PerformanceAnalysisAgent=BottleneckDecision,
        JointStrategyAgent=IntentBatch, ConstraintValidationAgent=ConstraintValidationDecision,
        ExperimentInvestigationAgent=InvestigationSelection)
    instructions = dict(PerformanceAnalysisAgent=search.performance_instructions(),
        JointStrategyAgent=search.strategy_instructions(),
        ConstraintValidationAgent=search.constraint_instructions(),
        ExperimentInvestigationAgent='比较完整候选的GraphDelta、关键路径、显存余量、预测误差和实测历史，选择有价值的GPU测量。可通过tool_requests请求inspect_execution_graph、evaluate_memory、compare_candidates、inspect_prediction_error；工具参数使用candidate_ids。accepted_candidate_ids按测量价值降序返回，只能引用提供的候选。load_validation_candidate_ids选择需要先执行Checkpoint Load与单步训练验证的候选。exploration_candidate_ids必须是accepted_candidate_ids的子集；没有选择探索候选时返回空列表。说明缺少的测量和选择依据，收益保持为待验证假设。')
    prompt_names = dict(PerformanceAnalysisAgent='performance_analysis_agent.md',
        JointStrategyAgent='joint_strategy_agent.md',
        ConstraintValidationAgent='constraint_validation_agent.md',
        ExperimentInvestigationAgent='experiment_investigation_agent.md')
    prompt_directory = Path(__file__).resolve().parents[1] / 'agents/prompts'
    measurements = {}
    for role, context in contexts.items():
        visible = search.visible_observation(search.agent_context(context, role))
        knowledge = ('' if role in ('JointStrategyAgent', 'ConstraintValidationAgent') else
            (prompt_directory / 'training_optimization_knowledge.md').read_text() + '\n')
        instruction = (prompt_directory / prompt_names[role]).read_text() + '\n' + knowledge + instructions[role]
        prompt = ('Use only the supplied observation. Do not call tools, access files, edit source code, '
            'or invent measurements. Return one object matching the output schema.\n'
            + instruction + '\n' + json.dumps(visible, ensure_ascii=False, allow_nan=False))
        schema_tokens = token_count(to_strict_json_schema(schemas[role]))
        prompt_tokens = token_count(prompt)
        measurements[role] = dict(prompt_tokens=prompt_tokens, strict_schema_tokens=schema_tokens,
            prompt_and_schema_tokens=prompt_tokens + schema_tokens,
            observation_tokens=token_count(visible),
            observation_field_tokens={key: token_count(value) for key, value in visible.items()})
    print(json.dumps(dict(stage='agent_context_tokens', roles=measurements),
        ensure_ascii=False), flush=True)
    assert all(row['prompt_and_schema_tokens'] <= 10000 for row in measurements.values()), measurements
    return dict(token_encoding='o200k_base', roles=measurements,
        source_diagnosis_evidence=diagnosis_id, source_strategy_evidence=strategy_id,
        new_cpu_predictions=[dict(trial_id=row['trial_id'],
            prediction_id=row['prediction']['prediction_id'],
            step_time_ms=row['prediction']['performance']['step_time_ms'],
            memory_admission=row['prediction']['memory']['admission']) for row in candidates])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recorded-state-evidence', type=Path)
    parser.add_argument('--profile-directory', type=Path, action='append', default=[])
    parser.add_argument('--communication-directory', type=Path, action='append', default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('集成验收目录必须为空，防止混入之前的状态')
    output.mkdir(parents=True, exist_ok=True)
    source = args.workspace / 'outputs/mimo-full-search'
    database = args.workspace / 'outputs/representative-predictor-validation/mimo/costs.sqlite'
    if args.resume:
        assert all((output / name).is_file() for name in
            ('costs.sqlite', 'alternative-costs.sqlite', 'experiments.sqlite'))
    else:
        backup_database(database, output / 'costs.sqlite')
        backup_database(source / 'profiles.sqlite', output / 'alternative-costs.sqlite')
        import_saved_records(source / 'experiments.sqlite', output / 'experiments.sqlite')
    config = json.loads((source / 'system-config.json').read_text())
    precision_receipt = None
    if args.recorded_state_evidence is not None:
        precision_receipt = restore_recorded_precision(config, args.recorded_state_evidence,
            output / 'experiments.sqlite')
    config.update(output_directory=str(output), profile_database=str(output / 'costs.sqlite'),
        hardware_topology_paths=[str(source / 'hardware-topology.json')],
        agent_execution_enabled=False, gpu_execution_enabled=False,
        max_llm_calls_per_round=4, evaluation_mode='multi_agent')
    search = EvidenceGuidedSearch(config)
    ingested_profiles = [search.predictor.ingest(path) for path in args.profile_directory]
    communication_profiles = []
    for directory in args.communication_directory:
        request = json.loads((directory / 'request.json').read_text())
        measurement = json.loads((directory / 'result.json').read_text())
        if measurement['status'] != 'PASS':
            raise ValueError('通信成本需要已经通过的真实测量')
        search.predictor.database.add_communication(request['run_id'], measurement['nccl_benchmark'])
        communication_profiles.append(dict(id=request['run_id'], source=str(directory.resolve()),
            result_sha256=hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest(),
            records=len(measurement['nccl_benchmark']['records'])))
    search.synchronize_cost_revision()
    strategy_id, strategy = read_strategy(search)
    result = dict(status='PASS', scope='saved_real_records_cpu_integration',
        recorded_state_precision=precision_receipt,
        ingested_profile_ids=ingested_profiles,
        ingested_communication_profiles=communication_profiles,
        family_catalog=verify_family_catalog(search),
        constraints=verify_constraints(search, strategy_id, strategy))
    print(json.dumps(dict(stage='constraint_completion', status='PASS'), ensure_ascii=False), flush=True)
    predictions = args.workspace / 'outputs/joint-synthesis-verification/cost-bindings-predictions'
    prediction_path = predictions / 'paramorph-mimo-16gpu-baseline-20261011-cfd9dffb3f97.json'
    result['dag_queries'] = verify_dag_queries(search, prediction_path)
    if result['dag_queries'].get('status') == 'NEEDS_PROFILE':
        result.update(status='PARTIAL', llm_requests=0, gpu_jobs=0,
            agent_contexts=dict(status='NOT_EXECUTED', reason='requires_current_costed_dag'),
            family_ledger=dict(status='NOT_EXECUTED', reason='requires_rankable_current_predictions'),
            measured_local_usage=search.store.usage())
        (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(status=result['status'],
            verified=['family_catalog', 'constraints', 'recorded_state_precision'],
            unavailable=result['dag_queries']['missing_profiles']), ensure_ascii=False))
        return
    result['agent_contexts'] = verify_agent_contexts(search, strategy, strategy_id, result['constraints'])
    result['family_ledger'] = verify_ledger(search, output / 'alternative-costs.sqlite')
    result.update(llm_requests=0, gpu_jobs=0, source_experiments=str(source / 'experiments.sqlite'),
        source_cost_database=str(database), measured_local_usage=search.store.usage())
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
