def valid_microbatch_group(num_microbatches, pp, vpp, group):
    if any(type(value) is not int or value < 1 for value in (num_microbatches, pp, vpp, group)):
        return False
    if vpp == 1:
        return True
    remainder = num_microbatches % group
    return pp > 1 and pp <= group <= num_microbatches and (remainder == 0 or remainder >= pp)


def native_schedule_table(num_microbatches, pp, vpp, group):
    if not valid_microbatch_group(num_microbatches, pp, vpp, group):
        raise ValueError('微批分组不满足当前Megatron原生调度条件')
    # 原生调度允许最后一组小于group，但交错路径要求其至少包含PP个微批。
    return [(microbatch, virtual) for offset in range(0, num_microbatches, group)
            for virtual in range(vpp)
            for microbatch in range(offset, min(offset + group, num_microbatches))]


def native_schedule_settings(request):
    overlap = request.get('overlap_p2p', False)
    batch = request.get('batch_p2p_comm', not overlap)
    warmup_flush = request.get('overlap_p2p_comm_warmup_flush', False)
    if any(type(value) is not bool for value in (overlap, batch, warmup_flush)):
        raise ValueError('原生P2P调度开关必须为布尔值')
    if overlap and (batch or request['vpp'] == 1):
        raise ValueError('P2P重叠要求交错流水线，且与批量P2P通信互斥')
    if warmup_flush and not overlap:
        raise ValueError('预热和收尾阶段的P2P重叠要求启用P2P重叠')
    divisor = request['dp'] * request['micro_batch']
    if request['global_batch'] % divisor:
        raise ValueError('全局批量必须能被DP与MBS的乘积整除')
    total = request['global_batch'] // divisor
    pp, vpp = request['pp'], request['vpp']
    group = request.get('group', pp)
    table = native_schedule_table(total, pp, vpp, group) if vpp > 1 else [(i, 0) for i in range(total)]
    return dict(pp=pp, vpp=vpp, num_microbatches=total, group=group if vpp > 1 else None,
        overlap_p2p=overlap, batch_p2p_comm=batch, overlap_p2p_comm_warmup_flush=warmup_flush,
        schedule_table=[list(row) for row in table])


def verify_native_schedule(request, manifests):
    expected = native_schedule_settings(request)
    observed = [row.get('native_pipeline_schedule') for row in manifests]
    if not observed or any(record != expected for record in observed):
        raise ValueError('Megatron实测流水线调度与已编译配置不同')
    return expected


def schedule_delta(before, after):
    left, right = native_schedule_settings(before), native_schedule_settings(after)
    return dict(before={key: val for key, val in left.items() if key != 'schedule_table'},
        after={key: val for key, val in right.items() if key != 'schedule_table'},
        schedule_order_changed=left['schedule_table'] != right['schedule_table'],
        first_changed_positions=[dict(index=i, before=a, after=b)
            for i, (a, b) in enumerate(zip(left['schedule_table'], right['schedule_table'])) if a != b][:8],
        table_lengths=[len(left['schedule_table']), len(right['schedule_table'])],
        overlap_permission_changed=left['overlap_p2p'] != right['overlap_p2p'],
        observed_overlap_ms=None, evidence='native_execution_contract_requires_matched_trace_for_overlap_duration')
