import argparse
import json
from pathlib import Path

from search.experiment_value import budgeted_selection_values
from search.schedule_controls import effective_schedule_execution


def read(path):
    return json.loads(path.read_text())


def verify(root):
    common, = (root / 'mistral/common/training/runs').glob('*/request.json')
    baseline = effective_schedule_execution(read(common))
    graph = read(root / 'mistral/full/round-records/round-0001.json')
    group_only, = [row['strategy'] for row in graph['candidates']
        if row['strategy']['vpp'] == 1 and row['strategy']['group'] == 8]
    assert effective_schedule_execution(group_only) == baseline

    trained, = (root / 'mistral/without_dag_evidence/training/runs').glob('candidate-*/result.json')
    result = read(trained)
    assert result['performance_eligible'] and result['native_pipeline_schedule']['overlap_p2p']
    assert effective_schedule_execution(result) != baseline

    archived = read(root / 'qwen8/full/round-records/round-0001.json')
    options = archived['experiment_options']
    assert len(options) == 2 and all(row['memory_admission'] == 'REQUIRE_MEMORY_PROFILE' for row in options)
    expected = [row['candidate_id'] for row in options]
    assert budgeted_selection_values(options, gpu_execution_enabled=True, full_training_trials=0) == (expected, True)
    assert budgeted_selection_values(options, gpu_execution_enabled=True, full_training_trials=1) == (
        ['DEFER', *expected], False)
    assert budgeted_selection_values([], gpu_execution_enabled=True, full_training_trials=0) == (['DEFER'], False)
    print('已用正式回执验证无效分组、实际交错调度和首次探索选项')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    verify(parser.parse_args().campaign)
