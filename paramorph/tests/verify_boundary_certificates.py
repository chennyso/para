import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1])]

from search.boundary_certificates import ExecutionBoundaryIndex
from search.conditional_partition import PartitionState,strictly_dominates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with gzip.open(args.records,'rt') as stream:
        source = json.load(stream)
    records = source['records']
    revision = records[0]['prediction']['cost_database_revision']
    index = ExecutionBoundaryIndex(revision,source['source_database'])
    for row in records:
        assert index.register(row['request'],row['prediction'])
    states = []
    for row in records:
        request = row['request']
        counts = tuple(request['decoder_chunks'])
        certificate = index.certify(request,counts)
        assert certificate is not None
        states.append(PartitionState(sum(counts),counts,(len(counts),sum(counts)),
            ('complete_native_suffix_domain',certificate.boundary_identity),None,None,None,
            certificate.identifier))
    dominated = []
    for left in states:
        for right in states:
            if left is right:
                continue
            result = strictly_dominates(left,right,index)
            left_time = index.certificates[left.boundary_certificate_id].metrics[0][0]
            right_time = index.certificates[right.boundary_certificate_id].metrics[0][0]
            assert result == (left_time < right_time)
            if result:
                assert not strictly_dominates(left,right)
                dominated.append(dict(kept=list(left.counts),pruned=list(right.counts),
                    kept_step_ms=left_time,pruned_step_ms=right_time))
    assert dominated
    request = records[0]['request']
    assert index.certify(request,()) is None
    incomplete = ExecutionBoundaryIndex(revision,source['source_database'])
    assert incomplete.register(records[0]['request'],records[0]['prediction'])
    assert incomplete.certify(records[1]['request'],tuple(records[1]['request']['decoder_chunks'])) is None
    held = tuple(records[0]['request']['decoder_chunks'])
    assert incomplete.certify(request,held[:-1]) is not None
    report = dict(status='PASS',source_database=source['source_database'],
        real_complete_records=len(records),certified_dominance=dominated,
        incomplete_suffix_domain_retained=True,missing_plan_retained=True,
        certificate_index=index.report(),new_gpu_runs=0)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='certificate_index'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
