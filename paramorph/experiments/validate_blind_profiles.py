import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts import SystemConfig
from predictor import FastPredictor
from predictor.profile_database import plan_signature


def run(config_path, output):
    config = SystemConfig.model_validate_json(config_path.read_text())
    predictor = FastPredictor(config.profile_database)
    request = dict(config.request)
    profile = predictor.database.get(config.profile_id, request)
    target = dict(profile['request'], run_id='blind-alias-' + request['run_id'])
    assert predictor.database.select(target, blind=True) is None
    assert plan_signature(target) == plan_signature(profile['request'])
    assert all(record.get('plan_signature') != plan_signature(target)
        for record in predictor.database.validation_residuals(target))
    counts = list(request['decoder_chunks'])
    assert counts[-1] > 1
    counts[-2] += 1
    counts[-1] -= 1
    candidate = dict(request, run_id='blind-independent-' + request['run_id'], decoder_chunks=counts)
    assert predictor.database.select(candidate, blind=True) is not None
    prediction = predictor.evaluate(candidate, config.profile_id, persist=False, blind=True)
    assert prediction['plan_signature'] == plan_signature(candidate)
    assert prediction['evidence']['compatible_anchor_id'] != candidate['run_id']
    assert prediction['performance']['step_time_ms'] > 0
    repeated = predictor.evaluate(candidate, config.profile_id, persist=False, blind=True)
    assert repeated['performance'] == prediction['performance']
    assert repeated['diagnosis']['prediction_cache_hit']
    report = dict(status='PASS', model=request['model'],
        same_plan_alias_excluded=True, independent_profile_accepted=True,
        prediction=prediction, cached_prediction_seconds=repeated['evaluation_seconds'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', model=request['model'], output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
