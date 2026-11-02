from inspect import signature
from types import MethodType

import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.transformer.spec_utils import build_module


def router_forward(self, hidden_states, *args, **kwargs):
    normalized, residual, state = self._pre_mlp_layernorm_and_residual(hidden_states)
    if state:
        raise ValueError('Hybrid Router MOVE requires the original single-stream normalization interface')
    probabilities, routing = self.mlp.route(normalized, kwargs.get('padding_mask'))
    sequence, batch, _ = normalized.shape
    if self.frontier_router_packet == 'sparse-topk':
        count = self.config.moe_router_topk
        torch._assert_async((routing.sum(dim=-1) <= count).all(), 'Router exceeds the configured top-k packet capacity')
        indices = routing.to(torch.int32).topk(count, dim=-1).indices
        selected = routing.gather(-1, indices)
        values = probabilities.gather(-1, indices).masked_fill(~selected, 0)
        encoded_indices = indices.masked_fill(~selected, -1)
        payload = (values.view(sequence, batch, count).float(),
            encoded_indices.view(sequence, batch, count).float())
    else:
        payload = (probabilities.view(sequence, batch, -1).float(),
            routing.view(sequence, batch, -1).float())
    packet = torch.cat((normalized.float(), residual.float(), *payload), dim=-1)
    return packet, kwargs.get('context')


def expert_forward(self, hidden_states, *args, **kwargs):
    packet = hidden_states
    hidden, experts = self.config.hidden_size, self.config.num_moe_experts
    count = self.config.moe_router_topk if self.frontier_router_packet == 'sparse-topk' else experts
    if packet.shape[-1] != 2 * hidden + 2 * count or packet.dtype != torch.float32:
        raise ValueError('Hybrid Router MOVE requires its complete lossless FP32 transport packet')
    normalized = packet[..., :hidden].to(self.config.params_dtype)
    residual_dtype = torch.float32 if self.config.fp32_residual_connection else self.config.params_dtype
    residual = packet[..., hidden:2 * hidden].to(residual_dtype)
    if self.frontier_router_packet == 'sparse-topk':
        values = packet[..., 2 * hidden:2 * hidden + count].reshape(-1, count)
        indices = packet[..., 2 * hidden + count:].reshape(-1, count).long()
        selected = indices >= 0
        torch._assert_async(((indices >= -1) & (indices < experts)).all(), 'Router packet contains an invalid expert index')
        indices = indices.clamp_min(0)
        probabilities = values.new_zeros((values.shape[0], experts)).scatter_add(
            -1, indices, values.masked_fill(~selected, 0))
        routing = torch.zeros_like(probabilities, dtype=torch.int32).scatter_add(
            -1, indices, selected.to(torch.int32)).bool()
    else:
        probabilities = packet[..., 2 * hidden:2 * hidden + experts].reshape(-1, experts)
        routing = packet[..., 2 * hidden + experts:].reshape(-1, experts).bool()
    shared = self.mlp.shared_experts_compute(normalized)
    permuted, probabilities = self.mlp.preprocess(normalized, probabilities, routing)
    dispatched, probabilities = self.mlp.dispatch(permuted, probabilities)
    output, bias = self.mlp.routed_experts_compute(dispatched, probabilities)
    if bias is not None:
        raise ValueError('Hybrid Router MOVE requires the original bias-free routed experts')
    output = self.mlp.postprocess(self.mlp.combine(output), shared)
    return self._apply_mlp_bda_step((output, bias), residual, ()), kwargs.get('context')


def install_hybrid_router_cut(logical_layer, source_pp, packet_schema='dense', boundary_dtypes=False):
    if packet_schema not in ('dense', 'sparse-topk'):
        raise ValueError('Hybrid Router MOVE requires a supported transport schema')
    if getattr(HybridStack, '_frontier_router_cut', None) is not None:
        raise ValueError('One concrete Router MOVE must own Hybrid model construction')
    original = HybridStack.__init__
    constructor = signature(original)

    def construct(self, *args, **kwargs):
        arguments = constructor.bind(self, *args, **kwargs).arguments
        original(self, *args, **kwargs)
        if self.is_mtp_layer:
            return
        config = arguments['config']
        groups = arguments['pg_collection']
        stage = groups.pp.rank()
        if stage == source_pp and self.layers[-1].layer_number - 1 == logical_layer - 1:
            local_config, = validate_segment_layers('E', config)
            layer = build_module(arguments['submodules'].moe_layer, config=local_config,
                layer_number=logical_layer + 1, pg_collection=groups, is_mtp_layer=False,
                add_layer_offset=False)
            layer.mlp.experts = None
            layer.mlp.shared_experts = None
            layer.mlp.use_shared_expert = False
            layer.mlp.token_dispatcher.shared_experts = None
            layer.forward = MethodType(router_forward, layer)
            layer.frontier_router_packet = packet_schema
            layer.frontier_region_kind = 'HYBRID_ROUTER_FRAGMENT'
            keys = tuple(dict(layer.named_parameters()))
            if not keys or any(not key.startswith(('pre_mlp_layernorm.', 'mlp.router.')) for key in keys):
                raise ValueError('Hybrid source must own only original normalization and Router parameters')
            self.layers.append(layer)
            self.num_layers_per_pipeline_rank = len(self.layers)
        elif stage == source_pp + 1 and self.layers[0].layer_number - 1 == logical_layer:
            layer = self.layers[0]
            if layer.config.moe_latent_size:
                raise ValueError('Hybrid Router MOVE requires its original non-latent expert interface')
            layer.mlp.router = None
            layer.pre_mlp_layernorm = None
            layer.forward = MethodType(expert_forward, layer)
            layer.frontier_router_packet = packet_schema
            layer.frontier_region_kind = 'HYBRID_EXPERT_FRAGMENT'

    HybridStack.__init__ = construct
    HybridStack._frontier_router_cut = (logical_layer, source_pp)
    install_hybrid_fp32_transport(boundary_dtypes)


def install_hybrid_fp32_transport(boundary_dtypes=False):
    if getattr(HybridModel, '_frontier_fp32_transport', False):
        raise ValueError('Hybrid FP32 transport must be installed exactly once')
    original_forward = HybridModel.forward
    original_input = HybridModel.set_input_tensor

    def forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        if boundary_dtypes or self.post_process:
            return output
        return output.float()

    def set_input(self, input_tensor):
        values = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        receiver = getattr(self.decoder.layers[0], 'frontier_region_kind', None) == 'HYBRID_EXPERT_FRAGMENT'
        if not receiver:
            values = [None if value is None else value.to(self.config.params_dtype) for value in values]
        return original_input(self, values)

    HybridModel.forward = forward
    HybridModel.set_input_tensor = set_input
    HybridModel._frontier_fp32_transport = True
    if boundary_dtypes:
        install_boundary_dtype_communication()


def install_boundary_dtype_communication():
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator, _p2p_ops

    if getattr(P2PCommunicator, '_frontier_boundary_dtypes', False):
        raise ValueError('Boundary-specific pipeline communication must be installed exactly once')

    def communicate(self, *, tensor_send_next, tensor_send_prev, recv_prev, recv_next,
            tensor_shape, wait_on_reqs=True):
        config = self.config
        if not config.variable_seq_lengths or config.batch_p2p_comm or config.use_ring_exchange_p2p:
            raise ValueError('Boundary-specific transport requires the native variable-shape unbatched P2P path')
        if (recv_prev or recv_next) and tensor_shape is None:
            raise ValueError('Boundary-specific transport requires the native receive shape contract')
        hidden = config.hidden_size
        widths = (hidden, 2 * hidden + 2 * config.num_moe_experts,
            2 * hidden + 2 * config.moe_router_topk)

        def dtype(shape):
            if len(shape) != 3 or shape[-1] not in widths:
                raise ValueError('Pipeline tensor shape does not match a registered Hybrid boundary contract')
            return config.params_dtype if shape[-1] == hidden else torch.float32

        for tensor in (tensor_send_next, tensor_send_prev):
            if tensor is not None and tensor.dtype != dtype(tensor.shape):
                raise ValueError('Pipeline sender precision differs from its boundary contract')
        previous_shape, next_shape = self._communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next)
        previous = torch.empty(previous_shape, requires_grad=True,
            device=torch.cuda.current_device(), dtype=dtype(previous_shape)) if recv_prev else None
        following = torch.empty(next_shape, requires_grad=True,
            device=torch.cuda.current_device(), dtype=dtype(next_shape)) if recv_next else None
        for direction, tensor in (('send_forward', tensor_send_next), ('send_backward', tensor_send_prev),
                ('receive_forward', previous), ('receive_backward', following)):
            if tensor is not None:
                boundary = 'activation' if tensor.shape[-1] == hidden else 'router'
                key = f'{direction}/{boundary}/{tensor.dtype}'
                record = P2PCommunicator._frontier_transport_totals.setdefault(key, {'messages': 0, 'bytes': 0})
                record['messages'] += 1
                record['bytes'] += tensor.numel() * tensor.element_size()
        requests = _p2p_ops(tensor_send_prev=tensor_send_prev, tensor_recv_prev=previous,
            tensor_send_next=tensor_send_next, tensor_recv_next=following, group=self.pp_group,
            prev_pipeline_rank=self.prev_rank, next_pipeline_rank=self.next_rank)
        if wait_on_reqs and requests:
            for request in requests.values():
                request.wait()
            requests = None
        return previous, following, requests

    P2PCommunicator._communicate = communicate
    P2PCommunicator._frontier_boundary_dtypes = True
    P2PCommunicator._frontier_transport_totals = {}
