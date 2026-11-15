import argparse
import json
from pathlib import Path


def strategy(plan, world, model_layers, sequence, global_batch):
    reasons = []
    stages = plan['num_stages']
    if stages not in (1, 2, 4, 8) or world % stages:
        return dict(legal=False, reasons=['流水线阶段数不能整除八张卡'])
    width = world // stages
    if plan['num_gpus'] != [width] * stages:
        reasons.append('不同阶段的GPU数量超出共同搜索范围')
    op_counts = plan['num_ops_in_each_stage']
    if len(op_counts) != stages:
        return dict(legal=False, reasons=[*reasons, '原生阶段算子数量与流水线阶段数不同'])
    layers = []
    for index, count in enumerate(op_counts):
        decoder_ops = count - int(index == 0) - 2 * int(index == stages - 1)
        if decoder_ops < 13 or decoder_ops % 13:
            reasons.append('原生阶段边界没有落在完整Decoder层之间')
            break
        layers.append(decoder_ops // 13)
    if sum(layers) != model_layers or plan['seq_length'] != sequence or plan['global_batch_size'] != global_batch:
        reasons.append('模型结构或训练负载与共同协议不同')
    meshes = {}
    for field in ('model_parallel_size_of_each_op', 'data_parallel_size_of_each_op'):
        groups = plan[field]
        values = {value for group in groups for value in group}
        if len(values) != 1 or [len(group) for group in groups] != op_counts:
            reasons.append(f'{field}使用了不同的逐阶段或逐算子并行度')
            continue
        meshes[field], = values
    if plan.get('resharding_stages') != [False] * stages:
        reasons.append('阶段内部张量重新分发超出共同搜索范围')
    if plan['algo_of_each_op'] != [[0] * count for count in op_counts]:
        reasons.append('算子替换超出共同搜索范围')
    modes = []
    for mode in ('none', 'full'):
        expected = [[int(mode == 'full')] * count for count in op_counts]
        expected[0][0] = 0
        expected[-1][-2:] = [0, 0]
        if plan['recompute_ops'] == expected and plan['checkpoint_activations'] == [any(row) for row in expected]:
            modes.append(mode)
    if not modes:
        reasons.append('重计算粒度超出全开或关闭的共同范围')
    tp = meshes.get('model_parallel_size_of_each_op')
    dp = meshes.get('data_parallel_size_of_each_op')
    micro_batch = None
    native_micro = plan['micro_batch_size']
    if tp is None or dp is None or tp * dp != width:
        reasons.append('TP和DP没有组成共同的每阶段设备组')
    elif native_micro % dp or native_micro // dp not in (1, 2, 4) or native_micro > 8 or global_batch % native_micro:
        reasons.append('原生微批与共同工作负载或合法微批范围不同')
    else:
        micro_batch = native_micro // dp
    return dict(legal=not reasons, reasons=reasons, strategy=dict(tp=tp, pp=stages, dp=dp,
        micro_batch_per_dp_replica=micro_batch, decoder_chunks=layers, recompute=modes[0] if modes else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.request:
        payload = json.loads(path.read_text())
        plan = payload.get('config', payload)
        print(json.dumps(dict(run_id=payload.get('run_id'), **strategy(plan, 8, 32, 2048, 1024)),
            ensure_ascii=False))
