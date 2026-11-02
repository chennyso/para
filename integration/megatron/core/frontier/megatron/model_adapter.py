from __future__ import annotations

from dataclasses import dataclass, replace

from megatron.core.frontier.ir.contract import Placement, PlacementKind, ValueContract
from megatron.core.frontier.ir.region import PipelineRegion, PipelineRegionGraph, RegionKind, TensorConnection
from megatron.core.frontier.runtime.repair import RepairBoundaryExecutor


@dataclass(frozen=True)
class PreparedTrainingProgram:
    program: object
    repair_result: object
    process_groups: dict

    def __post_init__(self):
        result = self.repair_result
        if self.program is None or not isinstance(self.process_groups, dict):
            raise ValueError('Native training preparation requires a complete program and actual group handles')
        if result.graph.scope != 'MODEL' or result.status != 'LEGAL_STRUCTURAL' or not result.after.legal:
            raise ValueError('Native model training requires a complete structurally closed model program')
        if result.state_migrations and result.migration_receipt is None:
            raise ValueError('Native model training preparation requires completed training state migration')


class FrontierModelAdapter:
    def __init__(self, model, program) -> None:
        self.model = model
        self.program = program
        self.boundary_executor = None

    def install_repair(self, result, rank, process_groups):
        self.boundary_executor = RepairBoundaryExecutor(result, rank, process_groups)
        return self.boundary_executor

    def bind_dense_output_repair(self, layer, connection):
        if self.boundary_executor is None:
            raise ValueError('Actual model boundary binding requires an installed executable repair result')
        kind = getattr(layer, 'frontier_region_kind', None)
        identity = f'layer:{layer.layer_number - 1}:{kind}'
        if (kind not in ('ATTENTION', 'MLP') or connection.producer != identity
                or connection.producer_output != 0):
            raise ValueError('Native dense output repair must match the actual partitioned layer interface')
        key = (connection.producer, connection.producer_output, connection.consumer, connection.consumer_input)
        if key not in self.boundary_executor.connections:
            raise ValueError('Native dense output repair is not owned by the current training rank')
        if getattr(layer, 'frontier_output_repair', None) is not None:
            raise ValueError('A native dense region output already has an installed repair')
        layer.frontier_output_repair = (self.boundary_executor, connection)

    def schedule_plan(self):
        return getattr(self.model, "build_schedule_plan")()

    def region_modes(self):
        return tuple(domain.mode_ids for domain in self.program.domains)


def install_dense_training_repair(model_chunks, result, program, rank, process_groups):
    chunks = tuple(model_chunks)
    if not chunks:
        raise ValueError('Native training repair requires actual constructed model chunks')
    adapter = FrontierModelAdapter(chunks, program)
    executor = adapter.install_repair(result, rank, process_groups)
    layers = {}
    for model in chunks:
        for module in model.modules():
            kind = getattr(module, 'frontier_region_kind', None)
            if kind in ('ATTENTION', 'MLP'):
                identity = f'layer:{module.layer_number - 1}:{kind}'
                if identity in layers:
                    raise ValueError('Native model chunks contain duplicate physical region implementations')
                layers[identity] = module
    connections = {(connection.producer, connection.producer_output,
        connection.consumer, connection.consumer_input): connection
        for connection in result.graph.connections()}
    bindings = []
    for key in executor.bindings:
        if key not in connections or key[0] not in layers or key[1] != 0:
            raise ValueError('Requested communication repair lacks a registered actual Dense output implementation')
        layer = layers[key[0]]
        if getattr(layer, 'frontier_output_repair', None) is not None:
            raise ValueError('Native training region already owns an output repair implementation')
        bindings.append((layer, connections[key]))
    if len({id(layer) for layer, _ in bindings}) != len(bindings):
        raise ValueError('Native Dense output implementation requires one repaired consumer per region output')
    for layer, connection in bindings:
        adapter.bind_dense_output_repair(layer, connection)
    return adapter


def build_layer_region_graph(layer, sequence_length, micro_batch_size, process_group_key):
    if getattr(layer, 'frontier_region_kind', None) is not None:
        raise ValueError('full model region registration requires the original native layer before ownership partitioning')
    config = layer.config
    logical_layer = layer.layer_number - 1
    if not callable(layer._forward_attention) or not callable(layer._forward_mlp):
        raise ValueError('registered transformer layer callables are required')
    placement = Placement(PlacementKind.SHARD, 0) if config.sequence_parallel or config.context_parallel_size > 1 else Placement(PlacementKind.REPLICATE)
    hidden = ValueContract('hidden_input', (sequence_length, micro_batch_size, config.hidden_size), placement,
                           process_group_key, process_group_key=process_group_key)
    attention_output = replace(hidden, logical_value='attention_output')
    layer_output = replace(hidden, logical_value='hidden_output')
    parameters = dict(layer.named_parameters())
    attention_parameters = tuple(name for name in parameters
                                 if name.startswith(('input_layernorm.', 'self_attention.')))
    mlp_parameters = tuple(name for name in parameters if name not in attention_parameters)
    is_moe = hasattr(layer.mlp, 'token_dispatcher')
    definitions = []
    if not is_moe:
        definitions = [(RegionKind.ATTENTION, (hidden,), (attention_output,), attention_parameters,
                        type(layer).__module__ + '.' + type(layer).__name__ + '._forward_attention', True),
                       (RegionKind.MLP, (attention_output,), (layer_output,), mlp_parameters,
                        type(layer).__module__ + '.' + type(layer).__name__ + '._forward_mlp', True)]
    else:
        if (config.enable_mhc_connections or config.moe_shared_expert_intermediate_size is not None
                or config.moe_latent_size or config.fp32_residual_connection):
            raise ValueError('six-region MoE registry requires its actual bias-free routed-expert layer')
        local = replace(hidden, logical_value='normalized_hidden')
        probability_dtype = {None: 'bfloat16', 'fp32': 'float32'}[config.moe_router_dtype]
        probs = replace(hidden, logical_value='routing_probabilities', logical_shape=('local_tokens', config.num_moe_experts),
                        dtype=probability_dtype)
        routing = replace(probs, logical_value='routing_map', dtype='bool', gradient_eligible=False, adjoint_key='none')
        dispatched = replace(local, logical_value='dispatched_tokens', logical_shape=('dispatched_tokens', config.hidden_size))
        permuted_probs = replace(probs, logical_value='permuted_probabilities', logical_shape=('dispatched_tokens',))
        routed = replace(dispatched, logical_value='expert_output')
        residual = replace(hidden, logical_value='residual')
        expert_parameters = tuple(name for name in mlp_parameters if name.startswith('mlp.experts.'))
        norm_parameters = tuple(name for name in mlp_parameters if name.startswith('pre_mlp_layernorm.'))
        router_parameters = tuple(name for name in mlp_parameters if name.startswith('mlp.router.'))
        if set(mlp_parameters) != set(expert_parameters + norm_parameters + router_parameters):
            raise ValueError('MoE region graph contains an unregistered actual parameter owner')
        assert all(callable(getattr(layer.mlp, key)) for key in ('route', 'preprocess', 'dispatch',
                                                               'routed_experts_compute', 'combine', 'postprocess'))
        layer_class = type(layer).__module__ + '.' + type(layer).__name__
        moe_class = type(layer.mlp).__module__ + '.' + type(layer.mlp).__name__
        definitions = [(RegionKind.ATTENTION, (hidden,), (attention_output,), attention_parameters,
                        layer_class + '._forward_attention', True),
                       (RegionKind.PRE_DISPATCH, (attention_output,), (local, residual), norm_parameters,
                        layer_class + '._pre_mlp_layernorm_and_residual', False),
                       (RegionKind.ROUTER, (local,), (probs, routing), router_parameters,
                        moe_class + '.route', True),
                       (RegionKind.DISPATCH, (local, probs, routing), (dispatched, permuted_probs), (),
                        moe_class + '.dispatch', False),
                       (RegionKind.EXPERT, (dispatched, permuted_probs), (routed,), expert_parameters,
                        moe_class + '.routed_experts_compute', False),
                       (RegionKind.COMBINE, (routed, residual), (layer_output,), (),
                        moe_class + '.combine', True)]
    regions = []
    for kind, inputs, outputs, names, forward, transportable in definitions:
        identity = f'layer:{logical_layer}:{kind.name}'
        modes = (f'TP{config.tensor_model_parallel_size}', f'CP{config.context_parallel_size}')
        if is_moe and kind in (RegionKind.DISPATCH, RegionKind.EXPERT, RegionKind.COMBINE):
            modes += (f'EP{config.expert_model_parallel_size}', f'ETP{config.expert_tensor_parallel_size}')
        regions.append(PipelineRegion(identity, logical_layer, kind, inputs, outputs,
                                      tuple(f'layer:{logical_layer}:{name}' for name in names), forward,
                                      'torch.autograd.backward',
                                      identity + ':forward', identity + ':backward', transportable,
                                      'megatron_gpt_moe' if is_moe else 'megatron_gpt_dense',
                                      modes,
                                      (moe_class + '.preprocess', moe_class + '.dispatch') if is_moe and kind == RegionKind.DISPATCH
                                      else (moe_class + '.combine', moe_class + '.postprocess', layer_class + '._apply_mlp_bda_step')
                                      if is_moe and kind == RegionKind.COMBINE else (forward,),
                                      parameter_elements=sum(parameters[name].numel() for name in names),
                                      parameter_storage_bytes=sum(parameters[name].numel() * parameters[name].element_size()
                                                                  for name in names)))
    producers = {}
    connections = []
    external = []
    for region in regions:
        for index, contract in enumerate(region.input_contracts):
            if contract.logical_value in producers:
                source, output = producers[contract.logical_value]
                connections.append(TensorConnection(source, output, region.region_id, index))
            elif contract.logical_value == hidden.logical_value:
                external.append((region.region_id, index))
            else:
                raise ValueError('实际Region输入缺少Tensor生产者')
        for index, contract in enumerate(region.output_contracts):
            if contract.logical_value in producers:
                raise ValueError('实际Region包含重复Tensor生产者')
            producers[contract.logical_value] = region.region_id, index
    graph = PipelineRegionGraph(tuple(regions), tuple(dict.fromkeys(
        (connection.producer, connection.consumer) for connection in connections)),
        tuple(connections), tuple(key for region in regions for key in region.parameter_keys),
        tuple(external), scope='OPERATOR')
    assert len({key for region in regions for key in region.parameter_keys}) == len(parameters)
    return graph
