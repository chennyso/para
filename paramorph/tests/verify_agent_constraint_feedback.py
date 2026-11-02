import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from contracts import fingerprint
from predictor.profile_database import ProfileDatabase, plan_signature
from search.system_search import (AgentRewriteIntent, ConstraintValidationDecision,
    ConstraintValidationToolRequest, EvidenceGuidedSearch, IntentBatch, InvestigationToolRequest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recorded-search', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--derived-baseline', type=Path)
    args = parser.parse_args()
    source, output = args.recorded_search.resolve(), args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'system-config.json').read_text())
    with sqlite3.connect((source / 'experiments.sqlite').as_uri() + '?mode=ro', uri=True) as original:
        saved = [json.loads(row[0]) for row in original.execute('SELECT document FROM agent_sessions')]
        session = next(row for row in saved if row.get('response', {}).get('intents'))
        with sqlite3.connect(output / 'experiments.sqlite') as cloned:
            original.backup(cloned)
    with sqlite3.connect(Path(config['profile_database']).as_uri() + '?mode=ro', uri=True) as original:
        profile = json.loads(original.execute('SELECT document FROM profiles WHERE id=?',
            (config['profile_id'],)).fetchone()[0])
    profiles = ProfileDatabase(output / 'costs.sqlite')
    profiles.add(profile)
    profiles.connection.close()
    config.update(output_directory=str(output), profile_database=str(output / 'costs.sqlite'),
        hardware_topology_paths=[str(source / 'hardware-topology.json')],
        gpu_execution_enabled=False, agent_execution_enabled=False)
    engine = EvidenceGuidedSearch(config)
    strategy = IntentBatch.model_validate(session['response'])
    parent = engine.reference_trial()
    before = deepcopy(parent['plan']['backend_config'])
    feedback = engine.strategy_validation_feedback(strategy)
    assert feedback['result']['status'] == 'REVISION_REQUIRED'
    issues = feedback['result']['issues']
    assert any(row.get('conflicts') == ['intent_count'] and row['actual'] == 5
        and row['expected'] == 6 for row in issues)
    family = next(row for row in issues if row.get('selected_family_id')
        and row.get('preserve_values'))
    assert family['family_values'] and family['reason']
    batch = next(row for row in issues if row.get('relation') == 'GBS % (DP * MBS) == 0')
    assert batch['numerator'] == 8 and batch['denominator'] == 16 and batch['remainder'] == 8
    assert any(row.get('incorrect_interaction_evidence_refs') for row in issues)
    first_query = InvestigationToolRequest(tool='preview_transformation',
        candidate_ids=[parent['trial_id']], transformation_intent=strategy.intents[0])
    receipt = engine.investigate(first_query, engine.store.trials())
    assert receipt['result']['status'] == 'CONSTRAINT_CONFLICT'
    assert receipt['result']['family_values'] == family['family_values']
    assert receipt['result']['preserve_values'] == family['preserve_values']
    assert parent['plan']['backend_config'] == before
    region_intent = next(intent for intent in strategy.intents
        if intent.values.decoder_chunks == [23, 25])
    corrected = AgentRewriteIntent.model_validate(dict(region_intent.model_dump(mode='json'),
        interaction_evidence_ids=[]))
    second_query = InvestigationToolRequest(tool='preview_transformation',
        candidate_ids=[parent['trial_id']], transformation_intent=corrected)
    successful_preview = engine.investigate(second_query, engine.store.trials())
    assert successful_preview['result'].get('status') != 'CONSTRAINT_CONFLICT'
    assert successful_preview['evidence_id'] != receipt['evidence_id']
    revisions = list(strategy.intents)
    legal_index = revisions.index(region_intent)
    revisions[legal_index] = corrected
    revised = strategy.model_copy(update=dict(intents=revisions))
    strategy_id = fingerprint(revised.model_dump(mode='json'))
    engine.store.add_evidence(strategy_id, 'HYPOTHESIS', revised.model_dump(mode='json'))
    completed = engine.constraint_tool(ConstraintValidationToolRequest(tool='complete_training_plans',
        parent_trial_id=parent['trial_id'], strategy_evidence_id=strategy_id,
        intent_indices=list(range(len(revised.intents))), limit=8))
    reports = completed['result']['intent_results']
    assert reports[legal_index]['status'] == 'COMPLETED'
    assert reports[legal_index]['candidate_ids']
    assert any(row['status'] == 'CONSTRAINT_CONFLICT' for row in reports)
    decision = ConstraintValidationDecision.model_validate(dict(
        intent_results=[dict(intent_index=row['intent_index'], status=row['status'],
            candidate_ids=row['candidate_ids'], evidence_refs=[completed['evidence_id']],
            reason=row.get('reason', '确定性工具已生成完整合法计划')) for row in reports],
        evidence_refs=[completed['evidence_id']], reason='根据真实工具结果逐项验证'))
    accepted = engine.validate_constraint_decision(decision, revised, parent, strategy_id)
    assert set(reports[legal_index]['candidate_ids']) <= set(accepted)
    assert engine.config.request['global_batch'] == before['global_batch']
    assert engine.config.request['sequence'] == before['sequence']
    assert engine.provider is None
    candidate = next(row for row in engine.store.trials() if row.get('prediction') and any(
        item.get('kind') == 'optimizer_component'
        for item in row['prediction']['evidence'].get('profiling_requests', [])))
    descriptors = candidate['prediction']['evidence']['profiling_requests']
    optimizer_keys = set()
    for item in descriptors:
        identity = item.get('identity', {})
        if item.get('kind') == 'optimizer_component' or (item.get('kind') == 'operator_cost' and (
                identity.get('operation') in ('local_optimizer_update_and_copy', 'optimizer_step')
                or {'physical_rank', 'parameter_count', 'optimizer_sharding'} <= set(identity))):
            optimizer_keys.add(item.get('cost_key') or fingerprint(identity))
    routed = engine.request_profiles(candidate, candidate['prediction'])
    repeated = engine.request_profiles(candidate, candidate['prediction'])
    native = engine.request_profiles(candidate, candidate['prediction'], native_only=True)
    assert routed['request_id'] == repeated['request_id'] == native['request_id']
    assert routed['measurement_type'] == 'native_training_cost_profile'
    assert set(routed['cost_keys']) == optimizer_keys
    assert routed['execution_request']['paramorph_optimizer_calibration'] is True
    assert routed['execution_request']['decoder_chunks'] == candidate['plan']['backend_config']['decoder_chunks']
    assert plan_signature(routed['execution_request']) == plan_signature(candidate['plan']['backend_config'])
    registered = engine.store.profile_requests()
    assert sum(row['request_id'] == routed['request_id'] for row in registered) == 1
    assert any(row['measurement_type'] == 'nccl' and candidate['trial_id'] in row['trial_ids']
        for row in registered)
    report = dict(status='PASS', validation_mode='control_flow_replay_without_llm_revision',
        recorded_strategy_evidence_id=session['evidence_id'],
        recorded_intent_count=len(strategy.intents), required_intent_count=engine.config.intent_count,
        validation_feedback=feedback, first_tool_result=receipt,
        second_tool_evidence_id=successful_preview['evidence_id'],
        constraint_tool_result=completed, accepted_candidate_ids=accepted,
        correction_source='explicit_constraint_test_input_with_recorded_valid_evidence',
        optimizer_profile_routing=dict(request_id=routed['request_id'],
            combined_cost_count=len(optimizer_keys), exact_plan_preserved=True,
            duplicate_requests_merged=True, nccl_requests_retained=True),
        gpu_runs=0, new_llm_requests=0, original_search_modified=False,
        original_prediction_ranking_claimed=False)
    if args.derived_baseline is not None:
        from experiments.derive_effective_execution import load_verified_baseline
        verified = load_verified_baseline(args.derived_baseline)
        derived_output = output / 'verified-baseline'
        derived_output.mkdir(parents=True, exist_ok=True)
        derived_profile = verified['derivation']['profile']
        derived_database = ProfileDatabase(derived_output / 'costs.sqlite')
        derived_database.add(derived_profile)
        derived_database.connection.close()
        derived_config = dict(config, request=verified['request'],
            output_directory=str(derived_output), baseline_directory=str(args.derived_baseline.resolve()),
            profile_database=str(derived_output / 'costs.sqlite'), profile_id=derived_profile['id'])
        resumed = EvidenceGuidedSearch(derived_config)
        baseline = resumed.initialize()
        assert baseline['plan']['backend_config']['clip_grad'] == 1.0
        assert baseline['measurement']['execution_semantics_evidence']['effective_clip_grad'] == 1.0
        assert baseline['measurement']['step_seconds'] == verified['result']['step_seconds']
        assert baseline['measurement']['run_id'] == verified['result']['run_id']
        changed = dict(verified['request'], clip_grad=0.0)
        try:
            resumed.require_fixed_training_semantics(verified['request'], changed)
        except ValueError as error:
            assert error.args[0]['status'] == 'CONSTRAINT_CONFLICT'
            assert 'clip_grad' in str(error.args[0]['conflicts'])
        else:
            raise AssertionError('固定Optimizer语义没有拒绝clip_grad变化')
        report['verified_baseline'] = dict(run_id=baseline['measurement']['run_id'],
            effective_clip_grad=1.0, original_measurement_reused=True,
            new_gpu_runs=0, clip_grad_change_rejected=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2,
        ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', issues=len(issues), accepted=len(accepted),
        report=str(output / 'report.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
