import argparse
from copy import deepcopy
import json
from pathlib import Path
import unittest

from contracts import TrainingPlan
from experiments.derive_effective_execution import load_verified_derivation
from predictor.profile_database import ProfileDatabase, normalized_execution, plan_signature, signature


def verify(sidecar_path, output):
    evidence = load_verified_derivation(sidecar_path)
    sidecar, actual = evidence['sidecar'], evidence['plan']['plan']
    source = json.loads(Path(sidecar['baseline']['source_directory'], 'request.json').read_text())
    case = unittest.TestCase()
    case.assertEqual(source['clip_grad'], 0.0)
    case.assertEqual(sidecar['baseline']['effective_clip_grad'], 1.0)
    case.assertEqual(sidecar['profile']['effective_clip_grad'], 1.0)
    case.assertEqual(actual['backend_config']['clip_grad'], 1.0)
    case.assertEqual(actual['optimizer_configuration']['clip_grad'], 1.0)
    case.assertEqual(evidence['original_measurement']['run_id'], source['run_id'])
    changed = deepcopy(actual)
    changed['backend_config']['clip_grad'] = 0.0
    changed['optimizer_configuration']['clip_grad'] = 0.0
    actual_plan, requested_plan = TrainingPlan.model_validate(actual), TrainingPlan.model_validate(changed)
    case.assertNotEqual(actual_plan.plan_hash, requested_plan.plan_hash)
    case.assertNotEqual(signature(actual['backend_config']), signature(changed['backend_config']))
    case.assertNotEqual(plan_signature(actual['backend_config']), plan_signature(changed['backend_config']))
    implicit = dict(actual['backend_config'])
    del implicit['clip_grad']
    case.assertEqual(signature(implicit), signature(actual['backend_config']))
    case.assertEqual(signature(dict(implicit, clip_grad=1)), signature(actual['backend_config']))
    inconsistent = deepcopy(actual)
    inconsistent['optimizer_configuration']['clip_grad'] = 0.0
    with case.assertRaisesRegex(ValueError, 'clip_grad不一致'):
        TrainingPlan.model_validate(inconsistent)
    for invalid in (-1, float('inf'), float('nan'), True, None, '1.0'):
        with case.assertRaisesRegex(ValueError, '有限非负数值'):
            normalized_execution(dict(actual['backend_config'], clip_grad=invalid))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    database_path = output / 'derived-costs.sqlite'
    if database_path.exists():
        raise ValueError('验证成本数据库已经存在')
    database = ProfileDatabase(database_path)
    profile = evidence['profile']
    database.add(profile)
    database.register_source(profile['id'], profile['source_directory'])
    case.assertEqual(database.select(actual['backend_config']), profile['id'])
    case.assertIsNone(database.select(changed['backend_config']))
    case.assertTrue(database.matches(profile['id'], actual['backend_config']))
    case.assertFalse(database.matches(profile['id'], changed['backend_config']))
    with case.assertRaisesRegex(ValueError, '匹配的成本记录'):
        database.get(profile['id'], changed['backend_config'])
    case.assertEqual([row['profile_id'] for row in database.reusable_compute_profiles(actual['backend_config'])],
        [profile['id']])
    case.assertEqual(database.reusable_compute_profiles(changed['backend_config']), [])
    database.connection.close()
    load_verified_derivation(sidecar_path)
    report = dict(status='PASS', source='actual_qwen30_baseline_and_profile_logs',
        derivation_id=sidecar['derivation_id'], source_run_id=source['run_id'],
        requested_clip_grad=source['clip_grad'], effective_clip_grad=1.0,
        effective_plan_hash=actual_plan.plan_hash, requested_plan_hash=requested_plan.plan_hash,
        cost_identity_isolated=True, derived_profile_sources_verified=True,
        original_sources_unchanged=True, created_measurement=False,
        database=str(database_path), source_optimizer_records={
            name: [dict(line_number=row['line_number'], node_index=row['node_index'],
                record_sha256=row['record_sha256']) for row in sidecar[name]['optimizer_config_records']]
            for name in ('baseline', 'profile')})
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.sidecar, args.output), ensure_ascii=False))
