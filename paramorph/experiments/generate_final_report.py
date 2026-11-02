import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.system_report import generate_report
from storage.experiment_store import ExperimentStore


def generate_final_report(ledger, output):
    if not Path(ledger).is_file():
        raise ValueError('实验报告需要实际Trial Ledger文件')
    store = ExperimentStore(ledger)
    report = generate_report(store, output)
    report['profiling_requests'] = store.profile_requests()
    report['failures'] = [row for row in store.trials() if row['state'] == 'FAILED']
    report['source_ledger'] = str(Path(ledger).resolve())
    Path(output, 'report.json').write_text(json.dumps(report, ensure_ascii=False,
        indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = generate_final_report(args.ledger, args.output)
    print(json.dumps(dict(real_measurements=report['real_measurements'],
                         source_ledger=report['source_ledger'])))
