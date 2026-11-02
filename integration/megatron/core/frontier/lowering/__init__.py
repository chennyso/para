from __future__ import annotations
from megatron.core.frontier.lowering.base import TransitionLowering, LoweringContext, LoweringRegistry, LOWERING_REGISTRY
from megatron.core.frontier.lowering.standard import StandardNCCLLowering
from megatron.core.frontier.lowering.fused_tp_cp import FusedTPCPLowering
from megatron.core.frontier.lowering.reduce_handoff import ReduceOnHandoffLowering
