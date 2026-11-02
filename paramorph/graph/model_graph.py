from pathlib import Path
import json

from contracts import fingerprint
from core_data_models import ModelSpec, LayerType
from predictor.parameter_metadata import bind_checkpoint_parameters


def build_operator_graph(model, config):
    tokens = config.sequence_length // config.cp * config.micro_batch_size
    hidden, heads = model.hidden_size, model.num_attention_heads
    dimension = model.head_dimension or hidden // heads
    local_heads = heads // config.tp
    local_kv = max(1, (model.num_query_groups or heads) // config.tp)
    nodes, flows = {}, []
    for layer, kind in enumerate(model.layer_types):
        input_id = f'layer:{layer}:input'
        nodes[input_id] = dict(operator_id=input_id, layer_id=layer, operation='layer_input',
            input_shape=[tokens, hidden], output_shape=[tokens, hidden], parameter_shape=None,
            semantic_source='model_configuration', independently_executable=False)
        composite = model.module_kinds is not None or kind in (LayerType.HYBRID, LayerType.MAMBA)
        if composite:
            operations = [('hybrid_module', [tokens, hidden], [tokens, hidden], None)]
        else:
            ffn = model.ffn_hidden_size
            if ffn is None:
                raise ValueError('静态MLP算子图需要模型FFN维度')
            operations = [
                ('attention_norm', [tokens, hidden], [tokens, hidden], [hidden]),
                ('qkv_projection', [tokens, hidden], [tokens, (local_heads + 2 * local_kv) * dimension],
                    [hidden, (local_heads + 2 * local_kv) * dimension]),
                ('attention_core', [tokens, (local_heads + 2 * local_kv) * dimension],
                    [tokens, local_heads * dimension], None),
                ('output_projection', [tokens, local_heads * dimension], [tokens, hidden],
                    [local_heads * dimension, hidden]),
                ('attention_residual', [tokens, hidden], [tokens, hidden], None),
                ('mlp_norm', [tokens, hidden], [tokens, hidden], [hidden])]
            if kind == LayerType.MOE:
                local_experts = model.num_experts // config.ep
                operations.extend([
                    ('router', [tokens, hidden], [tokens, model.num_experts], [hidden, model.num_experts]),
                    ('dispatch', [tokens, hidden], ['routed_tokens', hidden], None),
                    ('expert_gate_up', ['routed_tokens', hidden], ['routed_tokens', 2 * ffn // config.etp],
                        [local_experts, hidden, 2 * ffn // config.etp]),
                    ('expert_activation', ['routed_tokens', 2 * ffn // config.etp], ['routed_tokens', ffn // config.etp], None),
                    ('expert_down', ['routed_tokens', ffn // config.etp], ['routed_tokens', hidden],
                        [local_experts, ffn // config.etp, hidden]),
                    ('combine', ['routed_tokens', hidden], [tokens, hidden], None)])
            else:
                operations.extend([
                    ('gate_up', [tokens, hidden], [tokens, 2 * ffn // config.tp], [hidden, 2 * ffn // config.tp]),
                    ('activation', [tokens, 2 * ffn // config.tp], [tokens, ffn // config.tp], None),
                    ('down', [tokens, ffn // config.tp], [tokens, hidden], [ffn // config.tp, hidden])])
            operations.append(('mlp_residual', [tokens, hidden], [tokens, hidden], None))
        previous = input_id
        for operation, input_shape, output_shape, parameter_shape in operations:
            identity = f'layer:{layer}:{operation}'
            nodes[identity] = dict(operator_id=identity, layer_id=layer, operation=operation,
                input_shape=input_shape, output_shape=output_shape, parameter_shape=parameter_shape,
                layout=dict(tp=config.tp, cp=config.cp, ep=config.ep, etp=config.etp),
                semantic_source='model_configuration', backend_binding='requires_loaded_module_binding',
                module_kind=model.module_kinds[layer] if model.module_kinds else None,
                independently_executable=False)
            if previous:
                flows.append(dict(flow_id=f'{previous}->{identity}', producer=previous, consumer=identity,
                    tensor_shape=nodes[previous]['output_shape'], direction='forward', adjoint_direction='backward'))
            previous = identity
        if not composite:
            for source, destination, flow_type in (
                (input_id, f'layer:{layer}:attention_residual', 'residual_input'),
                (f'layer:{layer}:attention_residual', f'layer:{layer}:mlp_residual', 'residual_input')):
                flows.append(dict(flow_id=f'{source}->{destination}:skip', producer=source, consumer=destination,
                    tensor_shape=[tokens, hidden], direction='forward', adjoint_direction='backward', flow_type=flow_type))
        if kind == LayerType.MOE and not composite:
            flows.append(dict(flow_id=f'layer:{layer}:normalized_tokens_to_dispatch',
                producer=f'layer:{layer}:mlp_norm', consumer=f'layer:{layer}:dispatch', tensor_shape=[tokens, hidden],
                direction='forward', adjoint_direction='backward', flow_type='expert_token_input'))
            if model.shared_expert_size:
                for name, input_shape, output_shape, weight in (
                    ('shared_gate_up', [tokens, hidden], [tokens, 2 * model.shared_expert_size // config.tp],
                        [hidden, 2 * model.shared_expert_size // config.tp]),
                    ('shared_activation', [tokens, 2 * model.shared_expert_size // config.tp],
                        [tokens, model.shared_expert_size // config.tp], None),
                    ('shared_down', [tokens, model.shared_expert_size // config.tp], [tokens, hidden],
                        [model.shared_expert_size // config.tp, hidden])):
                    identity = f'layer:{layer}:{name}'
                    nodes[identity] = dict(operator_id=identity, layer_id=layer, operation=name,
                        input_shape=input_shape, output_shape=output_shape, parameter_shape=weight,
                        semantic_source='model_configuration', independently_executable=False)
                chain = ['mlp_norm', 'shared_gate_up', 'shared_activation', 'shared_down', 'combine']
                for source, destination in zip(chain, chain[1:]):
                    source, destination = f'layer:{layer}:{source}', f'layer:{layer}:{destination}'
                    flows.append(dict(flow_id=f'{source}->{destination}:shared', producer=source, consumer=destination,
                        tensor_shape=nodes[source]['output_shape'], direction='forward', adjoint_direction='backward'))
        if layer:
            predecessor = (f'layer:{layer - 1}:hybrid_module' if model.module_kinds is not None
                or model.layer_types[layer - 1] in (LayerType.HYBRID, LayerType.MAMBA)
                else f'layer:{layer - 1}:mlp_residual')
            flows.append(dict(flow_id=f'layer:{layer - 1}:output_to_layer:{layer}', producer=predecessor,
                consumer=input_id, tensor_shape=[tokens, hidden],
                direction='forward', adjoint_direction='backward'))
    forward_flows = list(flows)
    for identifier, node in list(nodes.items()):
        identity = identifier + ':backward'
        nodes[identity] = dict(operator_id=identity, layer_id=node['layer_id'], operation=node['operation'] + '_backward',
            input_shape=node['output_shape'], output_shape=node['input_shape'],
            parameter_gradient_shape=node.get('parameter_shape'), phase='backward',
            semantic_source='adjoint_of_configuration_operator', independently_executable=False)
        flows.append(dict(flow_id=identifier + ':saved_state', producer=identifier, consumer=identity,
            tensor_shape=node['input_shape'], flow_type='forward_state_to_backward', direction='forward_to_backward'))
    for flow in forward_flows:
        flows.append(dict(flow_id=flow['flow_id'] + ':adjoint', producer=flow['consumer'] + ':backward',
            consumer=flow['producer'] + ':backward', tensor_shape=flow['tensor_shape'], direction='backward'))
    return dict(nodes=nodes, flows=flows, source='static_model_configuration',
        cost_annotations=False, token_distribution='symbolic_router_output_for_moe',
        representation='configuration_operator_semantics_and_tensor_shapes')


def read_model(path, parameter_metadata_path=None):
    path = Path(path)
    config_text = path.read_text()
    config = json.loads(config_text)
    layers = config['num_hidden_layers']
    experts = config.get('num_experts', config.get('num_local_experts', config.get('n_routed_experts')))
    pattern = config.get('hybrid_layer_pattern', config.get('hybrid_override_pattern'))
    module_kinds = config.get('layers_block_type')
    if config.get('model_type') == 'falcon_h1':
        module_kinds = ['attention_mamba_composite'] * layers
    elif module_kinds is None and pattern is not None:
        symbols = {'M': 'mamba', '*': 'attention', '-': 'mlp', 'E': 'moe'}
        main_pattern = pattern.split('/')[0].replace('|', '')
        if set(main_pattern) - set(symbols):
            raise ValueError('Hybrid Pattern包含尚未绑定执行语义的模块类型')
        module_kinds = [symbols[symbol] for symbol in main_pattern]
    if module_kinds is not None:
        if len(module_kinds) != layers:
            raise ValueError('模型实际Hybrid Pattern与层数不一致')
        kinds = {'attention': LayerType.ATTENTION, 'mamba': LayerType.MAMBA,
            'mlp': LayerType.MLP, 'moe': LayerType.MOE,
            'attention_mamba_composite': LayerType.HYBRID}
        if set(module_kinds) - set(kinds):
            raise ValueError('实际模型包含尚未绑定执行语义的模块类型')
        types = [kinds[kind] for kind in module_kinds]
    elif experts:
        dense = set(config.get('mlp_only_layers', [])) | set(range(config.get('first_k_dense_replace', 0)))
        interval = config.get('decoder_sparse_step', config.get('moe_layer_freq', 1))
        types = [LayerType.MOE if index not in dense and (index + 1) % interval == 0 else LayerType.MLP for index in range(layers)]
    else:
        types = [LayerType.ATTENTION] * layers
    semantic_regions = []
    flows = []
    attention_kinds = config.get('layer_types', ['attention'] * layers)
    if len(attention_kinds) != layers:
        raise ValueError('实际Attention类型列表没有覆盖全部模型层')
    for index, kind in enumerate(types):
        components = ['attention', 'router', 'expert', 'combine'] if kind == LayerType.MOE else ['attention', 'mlp']
        if module_kinds is not None:
            components = [module_kinds[index]]
        for component in components:
            identity = f'layer:{index}:{component}'
            semantic_regions.append(dict(region_id=identity, layer_id=index, component=component,
                attention_kind=attention_kinds[index],
                backend_callable='actual_megatron_layer.forward', independently_executable=False,
                phases=['forward', 'backward', 'supported_recomputation'],
                semantic_source='actual_model_configuration', runtime_binding='requires_loaded_module_record'))
        if kind == LayerType.MOE and module_kinds is None:
            flows.extend([dict(flow_id=f'layer:{index}:dispatch', producer=f'layer:{index}:router',
                consumer=f'layer:{index}:expert', flow_type='expert_token_dispatch',
                token_distribution_source='requires_actual_router_measurement'),
                dict(flow_id=f'layer:{index}:combine', producer=f'layer:{index}:expert',
                    consumer=f'layer:{index}:combine', flow_type='expert_token_combine')])
        if index + 1 < layers:
            flows.append(dict(flow_id=f'layer:{index}:activation', producer_layer=index,
                consumer_layer=index + 1, flow_type='activation_and_adjoint_gradient'))
    mtp_layers = config.get('num_nextn_predict_layers', config.get('mtp_num_layers', 0)) or 0
    position_kind = config.get('position_embedding_type')
    qk_layernorm = config.get('qk_layernorm', config.get('model_type') in ('qwen3', 'qwen3_moe'))
    head_dimension = config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])
    finalize_grad_semantics = dict(
        share_embeddings_and_output_weights=config.get(
            'share_embeddings_and_output_weights', config.get('tie_word_embeddings', False)),
        mtp_num_layers=mtp_layers,
        embedding_requires_grad=not config.get('freeze_word_embeddings', False),
        vocab_size=config.get('padded_vocab_size', config.get('vocab_size')),
        position_embedding_type=position_kind,
        position_embedding_gradient_elements=(
            config.get('max_position_embeddings', 0) * config['hidden_size']
            if position_kind in ('learned_absolute', 'learned') else 0),
        position_embedding_group_policy='first_only',
        non_tensor_parallel_grad_elements_per_layer=2 * config['hidden_size'],
        qk_layernorm_grad_elements_per_layer=2 * head_dimension if qk_layernorm else 0,
        final_layernorm_grad_elements=config['hidden_size'],
        qk_layernorm=qk_layernorm,
        source='actual_model_configuration_and_megatron_finalize_model_grads_semantics')
    model = ModelSpec(config.get('_name_or_path', path.parent.name), layers,
        config['hidden_size'], config['num_attention_heads'], types, list(range(layers)), num_experts=experts,
        ffn_hidden_size=config.get('moe_intermediate_size', config.get('intermediate_size')), head_dimension=config.get('head_dim'),
        num_query_groups=config.get('num_key_value_heads'), moe_top_k=config.get('num_experts_per_tok'),
        shared_expert_size=config.get('shared_expert_intermediate_size',
            config.get('moe_shared_expert_intermediate_size')), module_kinds=module_kinds,
        finalize_grad_semantics=finalize_grad_semantics)
    parameter_metadata = (bind_checkpoint_parameters(
        json.loads(Path(parameter_metadata_path).read_text()), config)
        if parameter_metadata_path is not None else None)
    identity_config = json.loads(config_text, parse_constant=lambda value: {'__numeric_constant__': value})
    return model, dict(model_fingerprint=fingerprint(identity_config), model_source=str(path),
        parameter_metadata=parameter_metadata,
        model_revision=config.get('_commit_hash'), module_structure=config,
        nodes=[dict(layer_id=index, kind=kind.value,
            module_kind=module_kinds[index] if module_kinds is not None else 'transformer_block',
            backend_callable='actual_megatron_layer.forward')
            for index, kind in enumerate(types)],
        forward_dependencies=[[index, index + 1] for index in range(layers - 1)],
        backward_dependencies=[[index + 1, index] for index in range(layers - 1)],
        semantic_regions=semantic_regions, flows=flows,
        special_modules=dict(embedding='first_pipeline_stage', output='last_pipeline_stage',
            loss='last_pipeline_stage', mtp_layers=mtp_layers,
            tied_embeddings=config.get('tie_word_embeddings', False)),
        tensor_specs_source='actual_runtime_static_parameter_records',
        supported_model_operations=['continuous_module_region_partition'],
        scope='ordered_backend_model_modules')
