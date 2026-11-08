import math

from predictor.profile_database import effective_execution_config


STAGE_MODEL_REVISION = 'measured_stage_service_and_pipeline_bubble_v1'


def aggregate_stage_cost(request, profile):
    if profile is None:
        return None
    previous = profile['request']
    candidate_config = effective_execution_config(request)
    profile_config = effective_execution_config(previous)
    # 阶段内测量只在计算形状、通信端点和优化器所有权相同的条件下复用。
    for field in ('group', 'overlap_p2p', 'batch_p2p_comm', 'overlap_p2p_comm_warmup_flush'):
        candidate_config.pop(field)
        profile_config.pop(field)
    if candidate_config != profile_config or request['decoder_chunks'] != previous['decoder_chunks']:
        return None
    pp, vpp = request['pp'], request['vpp']
    microbatches = request['global_batch'] // (request['dp'] * request['micro_batch'])
    ranks = request['node_count'] * request['gpus_per_node']
    lanes = ranks // pp
    stages = [0.0] * pp
    tails = []
    for rank in range(ranks):
        stage = rank // lanes
        region_cost = 0.0
        layer_start = 0
        for index, width in enumerate(request['decoder_chunks']):
            if index % pp == stage:
                vp = index // pp
                for phase in ('forward', 'backward'):
                    region = profile['regions'].get(f'{rank}:{vp}:{phase}_compute')
                    if region is None:
                        return None
                    region_cost += region['overhead_ms'] + sum(
                        profile['layers'][str(layer)][f'{phase}_ms']
                        for layer in range(layer_start, layer_start + width))
            layer_start += width
        peer_cost = 0.0
        for vp in range(vpp):
            slot = vp * pp + stage
            for direction, adjacent in (('forward', slot + 1), ('backward', slot - 1)):
                if 0 <= adjacent < pp * vpp:
                    peer = adjacent % pp * lanes + rank % lanes
                    measured = profile['links'].get(f'{rank}:{peer}:{direction}')
                    if measured is None:
                        return None
                    peer_cost += measured
        stages[stage] = max(stages[stage], region_cost + peer_cost)
        tail = profile['gradient_finalize'].get(str(rank))
        optimizer = profile['optimizer'].get(str(rank))
        if tail is None or optimizer is None:
            return None
        tails.append(tail + optimizer)
    bubble = (pp - 1) / vpp
    estimate = max(stages) * (microbatches + bubble) + max(tails)
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError('阶段模型计算得到无效时间')
    return dict(step_time_ms=estimate, stage_service_ms=stages, bubble_microbatches=bubble,
        optimizer_and_sync_ms=max(tails), profile_id=profile['id'])
