import json


DDP_OVERLAP_FIELDS = ('overlap_grad_reduce', 'overlap_param_gather')


def resolve_ddp_overlap(request):
    ddp = request.get('execution_backend', 'megatron_ddp') == 'megatron_ddp'
    distributed = request.get('optimizer_sharding', 'distributed') == 'distributed'
    values = dict(overlap_grad_reduce=request.get('overlap_grad_reduce', ddp),
        overlap_param_gather=request.get('overlap_param_gather', ddp and distributed))
    if any(type(value) is not bool for value in values.values()):
        raise ValueError('DDP 重叠开关必须使用布尔值')
    if values['overlap_param_gather'] and not distributed:
        raise ValueError('参数收集重叠需要 Distributed Optimizer')
    return values


def verify_ddp_overlap(configuration, expected, *, rank=None):
    differences = {name: dict(requested=value, actual=getattr(configuration, name, None))
        for name, value in expected.items() if getattr(configuration, name, None) is not value}
    if differences:
        raise ValueError('EXECUTION_CONFIG_MISMATCH: ' + json.dumps(dict(rank=rank, differences=differences)))
