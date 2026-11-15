import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


ARMS = ('full', 'without_dag_evidence')
WORKLOAD = ('name', 'model', 'checkpoint', 'data', 'sequence', 'global_batch', 'precision',
    'optimizer', 'node_count', 'gpus_per_node', 'gpus_per_worker', 'warmup', 'measured',
    'optimizer_hyperparameters', 'optimizer_schedule', 'training_data_contract')


def read(path):
    return json.loads(Path(path).read_text())


def paired_interval(ratios):
    if len(ratios) < 3:
        return None
    values = [math.log(value) for value in ratios]
    center = mean(values)
    radius = t.ppf(0.975, len(values) - 1) * stdev(values) / math.sqrt(len(values))
    return dict(estimate_percent=100 * math.expm1(center),
        ci95_percent=[100 * math.expm1(center - radius), 100 * math.expm1(center + radius)],
        method='student_t_interval_on_paired_log_ratios')


def arm_results(paths, budget):
    observations = []
    for name in paths:
        path = Path(name)
        directory = path if path.is_dir() else path.parent
        result = directory / 'result.json'
        allocation = directory / 'allocation-completed.json'
        if result.is_file():
            receipt = read(result)
        else:
            receipt = read(directory / 'request.json') | dict(status='RECEIPT_INCOMPLETE', performance_eligible=False)
            if allocation.is_file():
                receipt.update(gpu_allocation_seconds=read(allocation)['gpu_allocation_seconds'])
        if any(receipt.get(field) for field in ('profile', 'instrumented', 'validation_mode', 'nccl_profile',
                'communication_test', 'compute_communication_profile')):
            raise ValueError('正式上卡次数不能包含采样或加载验证')
        launch = read(directory / 'launch-confirmed.json')
        if launch['run_id'] != receipt['run_id']:
            raise ValueError('GPU 启动记录和训练回执身份不同')
        seconds = receipt.get('gpu_allocation_seconds')
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('正式上卡需要正数 GPU 分配时间')
        hours = None if seconds is None else seconds * receipt['node_count'] * receipt['gpus_per_node'] / 3600
        observations.append((launch['observed_unix_seconds'], receipt, hours))
    observations.sort(key=lambda row: row[0])
    if sum(row[2] for row in observations if row[2] is not None) > budget + 1e-9:
        raise ValueError('正式训练实际 GPU 用量超过登记的共同预算')
    return observations


def evaluate(document):
    budget = document['gpu_hours_budget_per_arm']
    repetitions = document['pairs']
    if not math.isfinite(budget) or budget <= 0 or not repetitions:
        raise ValueError('配对实验需要共同 GPU 预算和登记的配对批次')
    seeds, seen, ratios, records = set(), set(), [], []
    repeated_workload = None
    for pair in repetitions:
        seed = pair.get('seed')
        if seed is not None and (type(seed) is not int or seed <= 0 or seed in seeds):
            raise ValueError('独立配对需要不同的已登记随机种子')
        if seed is not None:
            seeds.add(seed)
        arms = [arm_results(pair[arm], budget) for arm in ARMS]
        for arm in arms:
            for _, row, _ in arm:
                run_id = row['run_id']
                if run_id in seen:
                    raise ValueError('同一个 GPU 运行不能重复用于不同配对')
                seen.add(run_id)
        count = min(map(len, arms))
        best, statuses, total_hours = {}, {}, {}
        eligible_repeat = seed is not None and count > 0
        workload = None
        for name, arm in zip(ARMS, arms, strict=True):
            formal_hours = sum(row[2] for row in arm if row[2] is not None)
            usage_path = pair.get('usage_receipts', {}).get(name)
            usage = None if usage_path is None else read(usage_path)['usage']['gpu_hours']
            if usage is not None and (not math.isfinite(usage) or usage < formal_hours - 1e-8 or usage > budget + 1e-8):
                raise ValueError('搜索总 GPU 用量与正式训练回执或共同预算不一致')
            total_hours[name] = usage
            eligible_repeat &= usage is not None and all(row[2] is not None for row in arm)
            statuses[name] = [dict(run_id=row['run_id'], status=row['status'], gpu_hours=hours)
                for _, row, hours in arm[:count]]
            accepted = []
            for _, row, _ in arm[:count]:
                current = {field: row.get(field) for field in WORKLOAD}
                if workload is None:
                    workload = current
                elif workload != current:
                    raise ValueError('配对训练的工作负载不同')
                if row['status'] == 'PASS' and row.get('performance_eligible') is True:
                    throughput = row['tokens_per_second']
                    if not math.isfinite(throughput) or throughput <= 0:
                        raise ValueError('有效吞吐必须为正数')
                    protocol = row.get('training_protocol_observed', {})
                    eligible_repeat &= protocol.get('training_seed') == seed and protocol.get('training_data_seed') == seed
                    accepted.append(throughput)
            best[name] = max(accepted, default=None)
        if workload is not None:
            if repeated_workload is None:
                repeated_workload = workload
            elif repeated_workload != workload:
                raise ValueError('独立重复必须使用同一工作负载')
        ratio = best['full'] / best['without_dag_evidence'] if all(value is not None for value in best.values()) else None
        if ratio is not None and eligible_repeat:
            ratios.append(ratio)
        records.append(dict(pair_id=pair['pair_id'], seed=seed, matched_formal_attempts=count,
            started={name: len(arm) for name, arm in zip(ARMS, arms, strict=True)}, best_tokens_per_second=best,
            paired_delta_percent=None if ratio is None else 100 * (ratio - 1), attempts=statuses,
            total_gpu_hours=total_hours,
            independent_seed_verified=bool(ratio is not None and eligible_repeat)))
    inference = paired_interval(ratios) if len(ratios) == len(repetitions) and len({
        row['matched_formal_attempts'] for row in records}) == 1 else None
    return dict(gpu_hours_budget_per_arm=budget, paired_results=records, verified_independent_pairs=len(ratios),
        inference=inference, supports_full_advantage=bool(inference and inference['ci95_percent'][0] > 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(read(args.manifest)), ensure_ascii=False, indent=2, allow_nan=False))
