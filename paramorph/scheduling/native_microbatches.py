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
