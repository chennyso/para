from collections import defaultdict
from math import prod
from statistics import median

from graph.bottleneck_analysis import BottleneckGraph
from predictor.profile_database import signature


class NativeProfileReplay:
    def __init__(self, rewriter, reference, measured):
        self.rewriter, self.reference, self.measured = rewriter, reference, measured
        self.execution_signature = signature(reference)
        self.reference_counts = reference['decoder_chunks']
        self.layers = measured['layers']
        self.link_costs = {(row['source_rank'], row['target_rank'], row['direction'], row['bytes']): row
            for row in measured['p2p']}
        self.evidence = [measured['run_id'], measured['p2p_id']]

    def project(self, request):
        candidate = self.rewriter.candidate(request)
        config = candidate.parallel_config
        dag = self.rewriter.compiler.graph(candidate)
        if signature(request) != self.execution_signature:
            return dict(prediction_status='NEEDS_PROFILE', performance=dict(step_time_ms=None),
                memory=dict(admission='REQUIRE_MEMORY_PROFILE'), graph=None, replay=None,
                missing_costs=[dict(kind='native_module_and_schedule_components',
                    execution_signature=signature(request), task_count=len(dag.tasks))])
        tasks, missing = {}, []
        stages = request['pp']
        lanes = config.world_size // stages
        for identifier, task in dag.tasks.items():
            rank, region = task.rank, task.region_id
            stage = rank // lanes
            metadata = dict(task.metadata)
            kind = task.task_type.value
            evidence = self.measured['run_id']
            resources = [f'gpu:{rank}:compute']
            duration = None
            if kind in ('forward_compute', 'backward_compute'):
                phase = kind.removesuffix('_compute')
                duration = sum(self.layers[str(layer)][phase + '_ms'] for layer in metadata['layer_ids'])
                duration += self.measured['boundary_ms'][stage][phase]
            elif kind == 'pp_p2p_send':
                sender, receiver = metadata['group_ranks']
                payload = prod(metadata['shape']) * 2
                measured = self.link_costs.get((sender, receiver, metadata['direction'], payload))
                if measured is not None:
                    duration = measured['duration_ms']
                    resources = [f'gpu:{sender}:compute', f'gpu:{receiver}:compute',
                        f'gpu:{sender}:p2p', f'gpu:{receiver}:p2p']
                    evidence = self.measured['p2p_id']
            elif kind == 'dp_reduce' and task.microbatch_id is None:
                measured = self.measured['state_costs'][stage]
                elements = self.measured['decoder_layer_elements']
                baseline = self.reference_counts[stage] * elements + measured['boundary_elements']
                target = request['decoder_chunks'][stage] * elements + measured['boundary_elements']
                duration = measured['gradient_ms'] * target / baseline
                metadata['partition_cost_basis'] = dict(scope='physical_stage', reference_elements=target,
                    elements_per_parameter={'dense': 1.0}, source='measured_native_gradient_ownership_ratio')
            elif kind == 'optimizer_step':
                measured = self.measured['state_costs'][stage]
                elements = self.measured['decoder_layer_elements']
                baseline = self.reference_counts[stage] * elements + measured['boundary_elements']
                target = request['decoder_chunks'][stage] * elements + measured['boundary_elements']
                duration = measured['optimizer_ms'] * target / baseline
                metadata['partition_cost_basis'] = dict(scope='physical_stage', reference_elements=target,
                    elements_per_parameter={'dense': 1.0}, source='measured_native_optimizer_ownership_ratio')
                metadata['aggregate_optimizer_includes'] = ['prepare_gradients', 'gradient_norm',
                    'gradient_norm_all_reduce', 'clip_gradients', 'parameter_update']
            elif kind == 'nccl_generic' and metadata.get('operation') == 'word_embedding_gradient_all_reduce':
                participants = metadata['group_ranks']
                duration = max(self.measured['state_costs'][item // lanes]['embedding_ms'] for item in participants)
                resources = [f'gpu:{item}:compute' for item in participants]
            elif (kind in ('optimizer_prepare_grads', 'optimizer_local_grad_norm',
                    'optimizer_local_grad_norm_finalize', 'optimizer_local_grad_clip')
                    or kind == 'nccl_generic' and metadata.get('operation') == 'optimizer_gradient_norm_all_reduce'):
                duration = 0.0
                evidence = 'causal_alias_in_measured_native_optimizer_aggregate'
            elif kind in ('pp_p2p_recv', 'cuda_sync'):
                duration = 0.0
                evidence = 'explicit_receive_or_synchronization_dependency'
            if duration is None:
                missing.append(dict(task_id=identifier, operation=kind, region=region))
                continue
            metadata['duration_evidence'] = evidence
            tasks[identifier] = dict(rank=rank, logical_rank=rank, kind=kind, region=region,
                microbatch=task.microbatch_id, dependencies=list(task.dependencies),
                metadata=metadata, resources=resources, duration_ms=duration)
        if missing:
            return dict(prediction_status='NEEDS_PROFILE', performance=dict(step_time_ms=None),
                memory=dict(admission='REQUIRE_MEMORY_PROFILE'), graph=None, replay=None,
                missing_costs=missing[:24], missing_task_count=len(missing))
        graph = dict(tasks=tasks)
        replay = BottleneckGraph(graph, self.evidence).replay(config=config)
        timed = dict(tasks={key: row | dict(start_ms=replay['task_times'][key][0], end_ms=replay['task_times'][key][1])
            for key, row in tasks.items()})
        lifetimes, live = [], defaultdict(list)
        for key, row in tasks.items():
            if row['kind'] != 'forward_compute':
                continue
            backwards = 'B' + key[1:]
            if backwards not in tasks:
                continue
            start = replay['task_times'][key][1]
            end = replay['task_times'][backwards][0]
            if end <= start:
                continue
            size = request['sequence'] * request['micro_batch'] * self.rewriter.model.hidden_size * 2
            lifetimes.append(dict(buffer_id=key, bytes=size, start_ms=start, end_ms=end, kind='checkpoint_hidden_boundary'))
            live[row['rank']].extend(((start, size), (end, -size)))
        peaks = {}
        for rank, events in live.items():
            current, peak = 0, 0
            for _, delta in sorted(events):
                current += delta
                peak = max(current, peak)
            peaks[rank] = peak
        persistent = self.measured['decoder_layer_elements'] * 2
        state_bound = {rank: persistent * request['decoder_chunks'][rank // lanes] for rank in range(config.world_size)}
        measured_peaks = self.measured.get('reference_peak_bytes_by_rank', {})
        calibrated_layout = request['decoder_chunks'] == self.reference_counts and bool(measured_peaks)
        peak_projection = {rank: value + self.measured['decoder_layer_elements'] * 14 *
            (request['decoder_chunks'][rank // lanes] - self.reference_counts[rank // lanes])
            for rank, value in measured_peaks.items()}
        capacity = self.measured['device_capacity_bytes']
        memory = dict(admission=('REJECT_OOM' if max(state_bound.values()) >= self.measured['device_capacity_bytes']
            else 'ADMIT_CALIBRATED' if calibrated_layout and max(measured_peaks.values()) < capacity
            else 'WITHIN_ESTIMATED_CAPACITY' if peak_projection and max(peak_projection.values()) < capacity
            else 'REQUIRE_MEMORY_PROFILE'), persistent_lower_bound_bytes=state_bound,
            measured_peak_per_rank_bytes=measured_peaks if calibrated_layout else {},
            estimated_peak_per_rank_bytes=peak_projection,
            peak_estimate_scope='native_reference_peak_plus_fp16_weights_fp32_master_and_adam_state_delta',
            buffer_lifetimes=lifetimes, checkpoint_boundary_peak_bytes=peaks,
            scope='decoder_weights_and_checkpoint_boundary_lifetimes')
        return dict(prediction_status='PREDICTED_PROFILED_NATIVE', graph=graph, timed_graph=timed,
            replay=replay, performance=dict(step_time_ms=replay['step_time_ms'],
                tokens_per_second=request['sequence'] * request['global_batch'] * 1000 / replay['step_time_ms']),
            memory=memory, cost_sources=self.evidence, uncertainty='UNVALIDATED_ON_THIS_LAYOUT')
