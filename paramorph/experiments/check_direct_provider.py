import argparse
import json
from pathlib import Path
import sys

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from megatron.core.frontier.agents.direct_provider import DirectAPIProvider


class ProviderEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    evidence_refs: list[str]
    measured_rank_ids: list[int]
    measurement_status: str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-5.5')
    args = parser.parse_args()
    measurement = json.loads(args.measurement.read_text())
    if measurement['status'] != 'PASS' or not measurement['ranks']:
        raise ValueError('Provider验证需要真实成功的逐Rank显存测量')
    observation = dict(evidence_id=measurement['run_id'], status=measurement['status'],
        measured_rank_ids=sorted(int(rank) for rank in measurement['ranks']),
        source='actual_synchronized_cuda_memory_measurement')
    response, provenance = DirectAPIProvider(args.output, model=args.model).generate(
        'structured_provider_validation',
        '从提供的实际测量中返回证据身份、全部已测量Rank和状态。严格按照Schema输出。',
        observation, ProviderEvidence)
    if (response.evidence_refs != [observation['evidence_id']]
        or response.measured_rank_ids != observation['measured_rank_ids']
        or response.measurement_status != observation['status']):
        raise ValueError('真实Provider响应与提供的测量证据不同')
    result = dict(status='PASS', response=response.model_dump(), provenance=provenance)
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', model=args.model, usage=provenance['usage'])))
