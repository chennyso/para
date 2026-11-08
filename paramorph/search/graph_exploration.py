from collections import Counter

from graph.process_groups import process_groups


SHAPE_FIELDS = ('tp', 'pp', 'cp', 'dp', 'ep', 'etp', 'vpp')


def execution_shape(request):
    return tuple(request.get('etp', request['tp']) if key == 'etp' else request[key] for key in SHAPE_FIELDS)


def physical_parallel_investigations(request, families, analysis, devices, attempted_shapes):
    evidence_id = analysis.get('evidence_id') or analysis.get('graph_evidence_id')
    if not evidence_id:
        return []
    comparable_axes = ('cp', 'dp', 'ep', 'rank_mapping')
    parent_tp, parent_pp = request['tp'], request['pp']
    backend = request.get('execution_backend', 'megatron_ddp')
    parent_peak = max(sum(request['decoder_chunks'][stage::parent_pp]) for stage in range(parent_pp))
    targets = list(dict.fromkeys(row['region_id'] for row in (
        *analysis.get('bottlenecks', []), *analysis.get('critical_tasks', []))
        if type(row.get('region_id')) is int and 0 <= row['region_id'] < len(request['decoder_chunks'])))[:3]
    boundaries = [0]
    for count in request['decoder_chunks']:
        boundaries.append(boundaries[-1] + count)

    def cross_node_layout(plan):
        nodes = {row['physical_rank']: row['node_index'] for row in devices}
        groups = process_groups(plan)['physical']
        pp = plan['pp']
        result = {name: sum(len({nodes[rank] for rank in ranks}) > 1 for ranks in groups[name])
            for name in ('tp', 'ep', 'etp')}
        result['pipeline_links'] = sum(nodes[ranks[index % pp]] != nodes[ranks[(index + 1) % pp]]
            for ranks in groups['pp'] for index in range(len(plan['decoder_chunks']) - 1))
        return result

    by_direction = {1: [], -1: []}
    for family in families:
        values = family['values']
        if any(values[key] != request[key] for key in comparable_axes):
            continue
        if not any(state['execution_backend'] == backend for state in family['state_variants']):
            continue
        if (values['pp'] - parent_pp) * (values['tp'] - parent_tp) >= 0:
            continue
        direction = 1 if values['pp'] > parent_pp else -1
        shape_seen = any(shape[:6] == execution_shape(dict(values, vpp=1))[:6] for shape in attempted_shapes)
        by_direction[direction].append((shape_seen, abs(values['pp'] - parent_pp),
            values['etp'] != request.get('etp', parent_tp), family['family_id'], family))

    result = []
    before_groups = cross_node_layout(request) if devices and any(by_direction.values()) else {}
    for direction, options in by_direction.items():
        if not options:
            continue
        attempted, _, _, _, family = min(options, key=lambda item: item[:4])
        values = family['values']
        stages = values['pp']
        layers = [boundaries[-1] // stages + int(index < boundaries[-1] % stages) for index in range(stages)]
        ends = []
        for count in layers:
            ends.append((ends[-1] if ends else 0) + count)
        starts = [0, *ends[:-1]]
        result.append(dict(evidence_id=evidence_id, family_id=family['family_id'],
            status='STRUCTURAL_HYPOTHESIS', parent_cost_status=analysis.get('status'),
            parallel={key: values[key] for key in SHAPE_FIELDS if key != 'vpp'} | {'expert_dp': values['expert_dp']},
            direction='more_physical_stages_less_tensor_parallelism' if direction > 0 else
                'fewer_physical_stages_more_tensor_parallelism', attempted_in_this_search=attempted,
            balanced_vpp1_layout=layers, max_layers_per_physical_stage=[parent_peak, max(layers)],
            logical_pipeline_boundaries=[len(request['decoder_chunks']) - 1, stages - 1],
            cross_node_groups_and_links=[before_groups,
                cross_node_layout(dict(request, **values, decoder_chunks=layers))] if devices else None,
            critical_region_placement=[dict(parent_region=index, parent_stage=index % parent_pp,
                balanced_stages=[stage for stage, (start, end) in enumerate(zip(starts, ends, strict=True))
                    if start < boundaries[index + 1] and end > boundaries[index]]) for index in targets]))
    return result


def pipeline_investigations(request, module_types, analysis):
    """将父执行图中的区域压力转化为跨流水线形状的结构调查。"""
    if not analysis.get('graph_evidence_id'):
        return []
    chunks = request['decoder_chunks']
    pp, vpp = request['pp'], request['vpp']
    if len(chunks) != pp * vpp or sum(chunks) != len(module_types):
        raise ValueError('流水线调查需要与模型结构一致的完整父布局')
    targets = list(dict.fromkeys(row['region_id'] for row in [
        *analysis.get('bottlenecks', []), *analysis.get('critical_tasks', [])]
        if type(row.get('region_id')) is int and 0 <= row['region_id'] < len(chunks)))
    if not targets:
        return []
    region_counts = []
    offset = 0
    for count in chunks:
        region_counts.append(Counter(module_types[offset:offset + count]))
        offset += count

    def stage_moe(stages):
        return [sum(region_counts[index]['moe'] for index in range(stage, len(chunks), stages))
            for stage in range(stages)]

    tp, cp = request['tp'], request['cp']
    etp, ep = request.get('etp', tp), request['ep']
    world = tp * pp * cp * request['dp']
    parent_peak = max(stage_moe(pp))
    alternatives = []
    for stages in range(pp - 1, 1, -1):
        if len(chunks) % stages or world % (tp * cp * stages):
            continue
        dp = world // (tp * cp * stages)
        if request['global_batch'] % (dp * request['micro_batch']):
            continue
        microbatches = request['global_batch'] // (dp * request['micro_batch'])
        groups = [size for size in range(stages, microbatches + 1) if microbatches % size == 0]
        if not groups or world % (stages * etp):
            continue
        # 模块共同驻留后保持单进程专家归属上界，容量由原生预览另行检查。
        expert_slots = world // (stages * etp)
        moe_peak = max(stage_moe(stages))
        experts = request.get('num_moe_experts')
        partitions = [value for value in range(1, expert_slots + 1)
            if expert_slots % value == 0 and (experts is None or experts % value == 0)
            and (parent_peak == 0 and value == ep
                or parent_peak > 0 and moe_peak * ep <= parent_peak * value)]
        if not partitions:
            continue
        next_ep = min(partitions)
        values = dict(tp=tp, cp=cp, pp=stages, dp=dp, ep=next_ep, etp=etp,
            expert_dp=expert_slots // next_ep, vpp=len(chunks) // stages, group=groups[0])
        alternatives.append(dict(values=values, microbatches=microbatches,
            layout_witness=list(chunks), schedule='interleaved_1f1b',
            graph_evidence_id=analysis['graph_evidence_id'],
            target_region_placements=[dict(region=index, before=index % pp, after=index % stages,
                co_resident_regions=list(range(index % stages, len(chunks), stages))) for index in targets[:4]],
            stage_moe_modules_before=stage_moe(pp), stage_moe_modules_after=stage_moe(stages),
            parent_expert_ownership_units=parent_peak / ep, candidate_expert_ownership_units=moe_peak / next_ep,
            interpretation='结构假设：保持区间顺序，改变物理共驻、微批就绪及专家归属；训练性能和峰值显存待核验'))
    return alternatives


def matching_investigation(candidate, investigations):
    request = candidate['plan']['backend_config']
    return next((row for row in investigations if execution_shape(row['values']) == execution_shape(request)), None)


def exploration_representative(candidates, investigations, attempted_shapes):
    if not investigations:
        return None
    eligible = [row for row in candidates if not row.get('measurement') and not row.get('failure')
        and (row.get('prediction') or {}).get('memory', {}).get('admission') != 'REJECT_OOM'
        and (row.get('rewrite') or {}).get('candidate_source') == 'complete_strategy'
        and execution_shape(row['plan']['backend_config']) not in attempted_shapes]
    if not eligible:
        return None
    # 优先保留对应图调查的提议，并保留Agent对其他组合的选择权。
    return min(eligible, key=lambda row: matching_investigation(row, investigations) is None)


def retain_portfolio(ranked, representatives, unknown, alternative, limit):
    selected = ranked[:1]
    if alternative is not None and len(selected) < limit and alternative not in selected:
        selected.append(alternative)
    if unknown and len(selected) < limit and not any(row in selected for row in unknown):
        selected.append(unknown[0])
    for row in [*representatives, *ranked, *unknown]:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected
