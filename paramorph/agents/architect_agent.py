from core_data_models import Candidate
from region_partition.region_partitioner import RegionPartitioner
from validation.parallel_validator import validate_candidate
from agents.strategy_transformations import StrategyTransformationCompiler
from scheduling.native_microbatches import valid_microbatch_group


class ArchitectAgent:
    def __init__(self, model, topology, decision_service=None, transformation_compiler=None):
        self.model = model
        self.topology = topology
        self.decision_service = decision_service
        self.transformation_compiler = transformation_compiler

    def generate_graph_transformations(self, current, evidence, history):
        if self.decision_service is None:
            raise ValueError('图驱动变换需要真实Agent服务')
        compiler = self.transformation_compiler or StrategyTransformationCompiler(self.model, self.topology)
        full_observation = compiler.observation(current, evidence, history)
        observation = compiler.agent_context(full_observation)
        diagnosis = self.decision_service('diagnosis', dict(report=dict(bottlenecks=[], evidence=observation['evidence']),
            training_graph=observation['training_graph'],
            candidates=[], history=history,
            instruction='仅使用实测证据定位优化方向；缺少完整Trace时明确列出缺口，禁止推断未观测的关键路径'),
            query_handler=lambda query: compiler.query(full_observation, query))
        observation = observation | {'diagnosis': diagnosis}
        candidates = []
        seen = set()
        collaboration = []
        known = {current.candidate_id: current}
        for role in ('strategy',):
            snapshots = {key: compiler.observation(candidate, evidence, history)
                         for key, candidate in known.items()}
            query_observation = full_observation | {'candidate_observations': snapshots}
            shared_graphs = {key: compiler.agent_context(snapshot)['training_graph']
                            for key, snapshot in snapshots.items() if key != current.candidate_id}
            proposals = self.decision_service.propose_transformations(role,
                observation | {'shared_proposals': collaboration, 'shared_graphs': shared_graphs},
                query_handler=lambda query: compiler.query(query_observation, query))
            conflicts = [dict(proposal=proposal.model_dump(mode='json'),
                conflicts=compiler.parameter_conflicts(proposal)) for proposal in proposals.proposals
                if compiler.parameter_conflicts(proposal)]
            if conflicts:
                proposals = self.decision_service.propose_transformations(role,
                    observation | {'shared_proposals': collaboration, 'compiler_conflicts': conflicts,
                                   'shared_graphs': shared_graphs},
                    query_handler=lambda query: compiler.query(query_observation, query))
            for proposal in proposals.proposals:
                parent_id = proposal.parent_candidate_id or current.candidate_id
                if parent_id not in known:
                    raise ValueError('Agent引用了未生成的原始候选')
                parent = known[parent_id]
                expanded = [compiler.compile(parent, proposal, evidence)]
                for candidate in expanded:
                    candidate.transformation_record['reference_graph_delta'] = compiler.graph_delta(current, candidate)
                    candidate.transformation_record['lineage'] = parent.transformation_record.get('lineage', []) + (
                        [parent.transformation_record['proposal']] if parent.transformation_record else [])
                    if candidate.candidate_id not in seen:
                        seen.add(candidate.candidate_id)
                        candidates.append(candidate)
                        collaboration.append(compiler.candidate_context(candidate))
                        known[candidate.candidate_id] = candidate
        if candidates:
            review_observation = full_observation | dict(candidate_observations={
                candidate.candidate_id: compiler.observation(candidate, evidence, history)
                for candidate in candidates})
            review = self.decision_service('review', dict(
                report=dict(bottlenecks=[], evidence=observation['evidence']),
                decision_mode='experimental_priority',
                candidates=[compiler.candidate_context(candidate) for candidate in candidates],
                diagnosis=diagnosis,
                instruction='根据Graph Diff审查操作是否影响目标任务，列出计算形状、通信、调度与显存代价；未知成本保留不确定性，排序实验优先级'),
                query_handler=lambda query: compiler.query(review_observation, query))
            by_id = {candidate.candidate_id: candidate for candidate in candidates}
            if len(review['candidate_ids']) != len(set(review['candidate_ids'])):
                raise ValueError('图差异审查包含重复候选')
            candidates = [by_id[key] for key in review['candidate_ids']]
            for candidate in candidates:
                candidate.transformation_record['diagnosis_decision'] = diagnosis
                candidate.transformation_record['counterfactual_review'] = review
        return candidates

    def candidate(self, name, config, parent=None, actions=()):
        regions = RegionPartitioner(self.model, config).generate_uniform_regions()
        valid, errors = validate_candidate(config, self.model, self.topology, regions)
        return Candidate(name, parent, config, regions, list(actions), is_valid=valid,
                         validation_errors=errors)

    def required_candidates(self, baseline):
        matrix = {'baseline': (2, 8, 1, 1), 'A': (2, 8, 1, 2), 'B': (2, 8, 1, 4),
                  'C': (4, 4, 1, 2), 'D': (2, 4, 2, 2), 'E': (4, 2, 2, 1)}
        return [self.candidate(name, baseline.with_updates(tp=tp, pp=pp, dp=dp, cp=1,
            ep=1, etp=tp, vpp=vpp, microbatch_group=pp), actions=['required_experiment'])
            for name, (tp, pp, dp, vpp) in matrix.items()]

    def generate_candidates(self, current, critic):
        if not critic.bottlenecks:
            return []
        if any(not item.evidence_ids for item in critic.bottlenecks):
            raise ValueError('候选扩展必须具有真实瓶颈证据')
        config = current.parallel_config
        generated = []
        for bottleneck in critic.bottlenecks:
            if bottleneck.bottleneck_type in ('STAGE_IMBALANCE', 'PIPELINE_BUBBLE'):
                layer_costs = critic.layer_forward_costs_ms
                if set(layer_costs) == set(range(self.model.num_layers)):
                    partitioner = RegionPartitioner(self.model, config)
                    partitioner.set_layer_costs(layer_costs)
                    regions = partitioner.generate_load_balanced_regions()
                    valid, errors = validate_candidate(config, self.model, self.topology, regions)
                    generated.append(Candidate(f'{current.candidate_id}-nonuniform',
                        current.candidate_id, config, regions,
                        ['measured_layer_forward_proposal', 'full_iteration_calibration_required'],
                        is_valid=valid, validation_errors=errors))
                for vpp in (1, 2, 4):
                    if vpp == config.vpp:
                        continue
                    candidate = self.candidate(f'{current.candidate_id}-vpp{vpp}',
                        config.with_updates(vpp=vpp, microbatch_group=config.pp),
                        current.candidate_id, [bottleneck.bottleneck_type, 'change_vpp'])
                    generated.append(candidate)
                if config.vpp > 1:
                    for group in range(config.pp, config.num_microbatches + 1):
                        if group == config.microbatch_group or not valid_microbatch_group(
                                config.num_microbatches, config.pp, config.vpp, group):
                            continue
                        generated.append(self.candidate(f'{current.candidate_id}-group{group}',
                            config.with_updates(microbatch_group=group), current.candidate_id,
                            [bottleneck.bottleneck_type, 'microbatch_chunk_schedule']))
            if bottleneck.bottleneck_type in ('TP_COMMUNICATION', 'PIPELINE_BUBBLE'):
                for tp in (1, 2, 4):
                    for pp in (2, 4, 8):
                        divisor = tp * pp * config.cp
                        if self.topology.num_gpus % divisor:
                            continue
                        dp = self.topology.num_gpus // divisor
                        if config.global_batch_size % (dp * config.micro_batch_size):
                            continue
                        candidate = self.candidate(f'{current.candidate_id}-tp{tp}-pp{pp}',
                            config.with_updates(tp=tp, etp=tp, pp=pp, dp=dp,
                                                microbatch_group=pp), current.candidate_id,
                            [bottleneck.bottleneck_type, 'joint_tp_pp_dp'])
                        generated.append(candidate)
        if self.decision_service is not None:
            selected = self.decision_service('architect', dict(report=critic.to_dict(),
                candidates=[item.to_dict() for item in generated]))
            names = selected['candidate_ids']
            if not set(names) <= {item.candidate_id for item in generated}:
                raise ValueError('Agent输出包含未通过确定性候选生成的身份')
            generated = [item for item in generated if item.candidate_id in names]
        return generated
