import math

from predictor.profile_database import effective_execution_config
from region_partition.region_partitioner import RegionPartitioner


STAGE_MODEL_REVISION = 'stage_service_and_pipeline_bubble_v2'


def project_stage_cost(request, projector, *, cost_mode='hybrid'):
    config = projector.prepare_cost_surfaces(request, cost_mode)
    regions = RegionPartitioner(projector.model, config).from_counts(request['decoder_chunks'])
    counts = projector._parameter_counts(config, regions)
    costs = []
    if counts is None:
        return None, costs
    states = projector._state_memory(counts, config, request)
    lanes = config.world_size // config.pp
    stages = [0.0] * config.pp
    tails = []
    width = {'bfloat16': 2, 'float16': 2, 'float32': 4}[projector.pipeline_dtype]
    payload = config.sequence_length * config.micro_batch_size * projector.model.hidden_size * width
    payload //= config.cp * (config.tp if config.use_sequence_parallel else 1)
    fsdp = request.get('execution_backend') == 'megatron_fsdp'
    profile = projector.reusable_profile
    for rank in range(config.world_size):
        stage = rank // lanes
        service, tail = [], []
        for region in (item for item in regions if item.pp_rank == stage):
            for phase in ('forward', 'backward'):
                for layer in range(region.layer_start, region.layer_end):
                    service.append(projector._layer(layer, phase, rank, config, request, costs))
                    if phase == 'backward' and request['recompute'] == 'full' and projector.reusable_profile is None:
                        service.append(projector._layer(layer, 'recomputation', rank, config, request, costs))
                overhead = projector._region_overhead(region.region_id, phase, config, costs)
                if overhead is not None:
                    service.append(overhead)
            for direction, adjacent in (('backward', region.region_id - 1), ('forward', region.region_id + 1)):
                if 0 <= adjacent < len(regions):
                    peer = adjacent % config.pp * lanes + rank % lanes
                    if rank != peer:
                        link = profile['links'].get(f'{rank}:{peer}:{direction}') if profile else None
                        if link is not None and profile.get('pipeline_dtype', projector.dtype) == projector.pipeline_dtype:
                            costs.append(dict(status='MEASURED_REUSED', milliseconds=link,
                                source='actual_stage_point_to_point_link', evidence_ids=[profile['id']]))
                            service.append(link)
                        else:
                            service.append(projector._communication('p2p', payload, [rank, peer], costs,
                                dtype=projector.pipeline_dtype)[0])
        for family, axis in (('dense', 'dp_cp'), ('expert', 'expert_dp')):
            group = projector._group(axis, rank)
            parameters = counts[family][rank]
            if len(group) <= 1 or parameters == 0:
                continue
            distributed = request.get('optimizer_sharding', 'distributed') == 'distributed'
            collective = 'reduce_scatter' if distributed else 'all_reduce'
            tail.append(projector._communication(collective, math.ceil(parameters * 4 /
                (len(group) if distributed else 1)), group, costs, dtype='float32')[0])
            if distributed:
                synchronization = projector._communication('all_gather', math.ceil(parameters * width / len(group)),
                    group, costs, dtype=projector.dtype)[0]
                tail.append(synchronization)
                if fsdp:
                    service.append(synchronization)
        local = states[rank]['optimizer_bytes'] / states[rank]['optimizer_element_bytes']
        optimizer = projector._native_optimizer_cost(rank, counts, config, request, costs)
        if optimizer is None:
            optimizer = projector._elementwise('adam_update', math.ceil(local) * 20, math.ceil(local) * 28,
                'optimizer', rank, config, request, costs, None)
        tail.append(optimizer)
        if any(value is None for value in service + tail):
            return None, costs
        stages[stage] = max(stages[stage], sum(service))
        tails.append(sum(tail))
    bubble = (config.pp - 1) / config.vpp
    step = max(stages) * (config.num_microbatches + bubble) + max(tails)
    if not math.isfinite(step) or step <= 0:
        raise ValueError('跨配置阶段费用无效')
    return dict(step_time_ms=step, stage_service_ms=stages, bubble_microbatches=bubble,
        optimizer_and_sync_ms=max(tails), profile_id=(projector.reusable_profile or {}).get('id'),
        scope='aggregate_compute_communication_and_bubbles_without_execution_dag'), costs


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
