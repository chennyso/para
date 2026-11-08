"""train_downloaded_model.py 的 VPP 增强版本。

新增参数：
  --vpp VPP            virtual pipeline depth（默认 1，即 1F1B）
  --group GROUP        microbatch group size，VPP>1 时必须整除 microbatches
  --warmup WARMUP      预热步数（覆盖原来的默认值 0，推荐 20）
  --measured MEASURED  正式测量步数（默认 100）

其他改动：
  - VPP>1 时自动计算均匀 decoder_chunks，末尾阶段补余数层
  - VPP>1 时自动开启 overlap_p2p_comm
  - 合法性断言：16 % (tp * pp) == 0，global_batch % (dp * mbs) == 0，
    VPP>1 时 pp <= group <= microbatches 且整除
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from types import MethodType

os.environ["TRITON_CACHE_DIR"] = str(Path(os.environ["TRITON_CACHE_DIR"]) / f"isolated-rank{os.environ['LOCAL_RANK']}")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / f"isolated-rank{os.environ['LOCAL_RANK']}")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge import AutoBridge
from megatron.bridge.models.falcon_h1.falconh1_provider import FalconH1ModelProvider
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.pretrain import pretrain
from megatron.core.utils import unwrap_model as unwrap_megatron_model
from megatron.core import tensor_parallel
from megatron.core.transformer.enums import AttnBackend
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from torch.utils._pytree import tree_leaves
from hybrid_moe_metrics import configure_hybrid_moe_metrics
from hf_fused_mamba_norm import HFFusedRMSNormMambaMixer


def unwrap_model(model):
    current = unwrap_megatron_model(model)
    while not hasattr(current, 'decoder') and hasattr(current, 'module'):
        current = current.module
    return current


def checkpoint_falcon_layer(self, hidden_states, *args, **kwargs):
    if self.training and torch.is_grad_enabled():
        self.frontier_checkpoint_calls += 1

        def execute(value):
            self.frontier_layer_executions += 1
            return self.frontier_original_forward(value, *args, **kwargs)

        return tensor_parallel.checkpoint(execute, False, hidden_states)
    return self.frontier_original_forward(hidden_states, *args, **kwargs)


class Evidence(Callback):
    def __init__(self, args):
        self.args = args
        self.rows = []
        self.traces = set()
        self.falcon_layers = []
        self.attention_backend_calls = {}
        self.mlp_forward_calls = {}

    def on_train_start(self, context):
        self.rank = dist.get_rank()
        self.models = [unwrap_model(model) for model in context.model]
        if self.args.recompute_mlp:
            for model_index, model in enumerate(self.models):
                for layer_index, layer in enumerate(model.decoder.layers):
                    if isinstance(layer, MambaLayer):
                        continue
                    if not layer.recompute_mlp:
                        raise ValueError('Selective MLP recomputation did not reach the actual Transformer layer')
                    layer.recompute_mlp = (layer.layer_number - 1) % self.args.recompute_mlp_every == 0
                    if not layer.recompute_mlp:
                        continue
                    key = f'{model_index}:{layer_index}'
                    self.mlp_forward_calls[key] = 0

                    def record_mlp(instance, inputs, label=key):
                        self.mlp_forward_calls[label] += 1

                    layer.mlp.register_forward_pre_hook(record_mlp)
        if isinstance(context.state.cfg.model, FalconH1ModelProvider):
            for model_index, model in enumerate(self.models):
                if self.args.recompute:
                    for layer in model.decoder.layers:
                        layer.frontier_original_forward = layer.forward
                        layer.frontier_checkpoint_calls = 0
                        layer.frontier_layer_executions = 0
                        layer.forward = MethodType(checkpoint_falcon_layer, layer)
                        self.falcon_layers.append(layer)
                for name, module in model.named_modules():
                    if name.endswith(('.unfused_attention', '.fused_attention', '.flash_attention')):
                        key = f'{model_index}:{name}'
                        self.attention_backend_calls[key] = 0

                        def record_backend(instance, inputs, label=key):
                            self.attention_backend_calls[label] += 1

                        module.register_forward_pre_hook(record_backend)
        if getattr(context.state.cfg.model, "num_moe_experts", None) and getattr(context.state.cfg.model, "mtp_num_layers", 0):
            configure_hybrid_moe_metrics(self.models, context.state.cfg.model)
        if self.args.deterministic_embedding:
            for model in self.models:
                for module in model.modules():
                    if isinstance(module, VocabParallelEmbedding):
                        module.deterministic_mode = True
                        def verify_embedding(instance, inputs):
                            if id(instance) not in self.traces:
                                indices = inputs[0]
                                masked = indices.clone() - instance.vocab_start_index
                                masked[(indices < instance.vocab_start_index) | (indices >= instance.vocab_end_index)] = 0
                                with torch.no_grad():
                                    assert torch.equal(F.embedding(masked, instance.weight), instance.weight[masked])
                                self.traces.add(id(instance))
                                print(f"EMBEDDING_FORWARD_EQUAL rank={self.rank}", flush=True)
                        module.register_forward_pre_hook(verify_embedding)
        if self.args.diagnose:
            for model in self.models:
                for name, parameter in model.named_parameters():
                    assert torch.isfinite(parameter).all().item(), f"Nonfinite parameter: {name}"
                    def check_gradient(gradient, label=name, parameter_ref=parameter):
                        actual = parameter_ref.main_grad if getattr(parameter_ref, "grad_added_to_main_grad", False) else gradient
                        assert torch.isfinite(actual).all().item(), f"Nonfinite accumulated parameter gradient: {label}"
                        maximum = actual.abs().max().item()
                        if label == "embedding.word_embeddings.weight":
                            print(f"EMBEDDING_GRADIENT rank={self.rank} dtype={actual.dtype} abs_max={maximum}", flush=True)
                        if label == "output_layer.weight":
                            print(f"GRADIENT_TRACE rank={self.rank} name={label} dtype={actual.dtype} abs_max={maximum} fp64_norm={actual.double().norm().item()}", flush=True)
                        if maximum > 1e12:
                            print(f"LARGE_GRADIENT rank={self.rank} name={label} abs_max={maximum}", flush=True)
                        return gradient
                    parameter.register_hook(check_gradient)
        manifest = {"model": str(self.args.model), "checkpoint": str(self.args.checkpoint),
                    "rank": self.rank, "world": dist.get_world_size(),
                    "tp": self.args.tp, "pp": self.args.pp, "ep": self.args.ep,
                    "cp": context.state.cfg.model.context_parallel_size,
                    "vpp": self.args.vpp, "group": self.args.group,
                    "optimizer_cpu_offload": self.args.cpu_offload,
                    "deterministic_embedding": self.args.deterministic_embedding,
                    "recompute_granularity": context.state.cfg.model.recompute_granularity,
                    "recompute_modules": context.state.cfg.model.recompute_modules,
                    "recompute_mlp_every": self.args.recompute_mlp_every,
                    "recomputed_mlp_layer_ids": [layer.layer_number - 1 for model in self.models
                        for layer in model.decoder.layers if getattr(layer, 'recompute_mlp', False)],
                    "attention_backend": str(context.state.cfg.model.attention_backend),
                    "falcon_checkpoint_layers": len(self.falcon_layers),
                    "bias_dropout_fusion": context.state.cfg.model.bias_dropout_fusion,
                    "hf_fused_mamba_norm": self.args.hf_fused_mamba_norm,
                    "params_dtype": str(context.state.cfg.model.params_dtype),
                    "pipeline_dtype": str(context.state.cfg.model.pipeline_dtype),
                    "num_moe_experts": getattr(context.state.cfg.model, "num_moe_experts", None),
                    "moe_router_topk": getattr(context.state.cfg.model, "moe_router_topk", None),
                    "moe_token_dispatcher_type": getattr(context.state.cfg.model, "moe_token_dispatcher_type", None),
                    "sequence": self.args.sequence, "global_batch": self.args.global_batch, "micro_batch": self.args.micro_batch,
                    "parameter_elements": sum(parameter.numel() for model in self.models for parameter in model.parameters()),
                    "model_classes": [type(model).__module__ + "." + type(model).__name__ for model in self.models],
                    "hybrid_layer_pattern": getattr(context.state.cfg.model, "hybrid_layer_pattern", None),
                    "mtp_num_layers": getattr(context.state.cfg.model, "mtp_num_layers", 0),
                    "vocab_size": context.state.cfg.model.vocab_size,
                    "layernorm_epsilon": context.state.cfg.model.layernorm_epsilon,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name()}
        (self.args.output / f"rank{self.rank}.manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    def on_train_step_start(self, context):
        assert context.state.train_state.step == len(self.rows)
        if len(self.rows) == self.args.warmup:
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()

    def on_train_step_end(self, context):
        assert context.skipped_iter is False
        losses = {key: float(value) for key, value in context.loss_dict.items()}
        assert all(math.isfinite(value) for value in losses.values())
        norm = None if context.grad_norm is None else float(context.grad_norm)
        assert norm is None or math.isfinite(norm)
        self.rows.append({"update": len(self.rows) + 1, "loss": losses, "grad_norm": norm,
                          "optimizer_update_successful": True})
        if self.rank == 0 and (len(self.rows) == 1 or len(self.rows) % 10 == 0):
            print(f"DOWNLOADED_MODEL_UPDATE update={len(self.rows)} grad_norm={norm}", flush=True)

    def on_train_end(self, context):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        checkpoint_calls = sum(layer.frontier_checkpoint_calls for layer in self.falcon_layers)
        layer_executions = sum(layer.frontier_layer_executions for layer in self.falcon_layers)
        if self.falcon_layers and (checkpoint_calls == 0 or layer_executions <= checkpoint_calls):
            raise ValueError('Falcon activation checkpointing did not execute both forward and backward recomputation')
        if self.args.recompute_mlp:
            expected = len(self.rows) * self.args.global_batch // (self.args.micro_batch *
                (dist.get_world_size() // (self.args.tp * self.args.cp * self.args.pp)))
            if not self.mlp_forward_calls or any(count != 2 * expected for count in self.mlp_forward_calls.values()):
                raise ValueError('Selective MLP recomputation did not execute its forward and backward replay')
        result = {"status": "complete", "rank": self.rank, "updates": len(self.rows),
                  "falcon_checkpoint_calls": checkpoint_calls,
                  "falcon_layer_executions": layer_executions,
                  "attention_backend_calls": self.attention_backend_calls,
                  "mlp_forward_calls": self.mlp_forward_calls,
                  "warmup": self.args.warmup, "measured": self.args.measured,
                  "window_seconds": elapsed, "update_seconds": elapsed / self.args.measured,
                  "tokens_per_second": self.args.global_batch * self.args.sequence * self.args.measured / elapsed,
                  "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "steps": self.rows, "consumed_samples": context.state.train_state.consumed_train_samples,
                  "validation_scope": "真实检查点训练、全部更新成功、loss及grad_norm有限；尚无独立完整数值等价认证"}
        assert len(self.rows) == self.args.warmup + self.args.measured
        (self.args.output / f"rank{self.rank}.measurements.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        dist.barrier()
        print(f"DOWNLOADED_MODEL_COMPLETE rank={self.rank}", flush=True)


def uniform_decoder_chunks(num_layers, pp, vpp):
    """均匀分配层到 pp*vpp 个逻辑 stage；末尾 stage 补余数层。"""
    total_stages = pp * vpp
    base = num_layers // total_stages
    remainder = num_layers % total_stages
    return tuple(base + (1 if i < remainder else 0) for i in range(total_stages))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--vpp", type=int, default=1)
    parser.add_argument("--group", type=int, default=0,
                        help="microbatch group size for VPP schedule; 0 = auto (pp)")
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--deterministic-embedding", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--recompute-mlp", action="store_true")
    parser.add_argument("--recompute-mlp-every", type=int, choices=(1, 2), default=1)
    parser.add_argument("--unfused-bda", action="store_true")
    parser.add_argument("--hf-fused-mamba-norm", action="store_true")
    parser.add_argument("--load-hf-weights", action="store_true")
    args = parser.parse_args()
    if args.vpp < 1:
        raise ValueError('The virtual pipeline count must be positive')
    if args.recompute and args.recompute_mlp:
        raise ValueError('Full and selective MLP recomputation are mutually exclusive')
    if args.recompute_mlp_every != 1 and not args.recompute_mlp:
        raise ValueError('Partial MLP recomputation requires selective MLP recomputation')
    if args.group == 0:
        args.group = args.pp
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(31 * 2**30 / torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    if not args.load_hf_weights:
        assert (args.checkpoint / "iter_0000000/.metadata").is_file()
    assert Path(str(args.data) + ".bin").is_file()
    args.output.mkdir(parents=True, exist_ok=True)
    world = int(os.environ['WORLD_SIZE'])
    if world % (args.tp * args.pp * args.cp):
        raise ValueError('TP*PP*CP must divide the actual GPU count')
    dp = world // (args.tp * args.pp * args.cp)
    microbatches = args.global_batch // (dp * args.micro_batch)
    assert world % (args.tp * args.pp) == 0, f"world={world} not divisible by tp={args.tp} * pp={args.pp}"
    assert args.global_batch % (dp * args.micro_batch) == 0, \
        f"global_batch={args.global_batch} not divisible by dp={dp} * micro_batch={args.micro_batch}"
    if args.vpp > 1:
        assert args.pp <= args.group <= microbatches, \
            f"VPP>1 requires pp <= group <= microbatches, got pp={args.pp} group={args.group} microbatches={microbatches}"
        assert microbatches % args.group == 0, \
            f"microbatches={microbatches} not divisible by group={args.group}"
    os.environ.update(TRAIN_ITERS=str(args.warmup + args.measured), GLOBAL_BATCH_SIZE=str(args.global_batch),
                      MICROBATCH_GROUP_SIZE=str(args.group), OUTPUT_PATH=str(args.output))
    sys.path.append("/workspace/tools")
    from train_qwen3_8b_bridge_vpp import build_config
    cfg = build_config()
    cfg.model = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True).to_megatron_provider(
        load_weights=args.load_hf_weights)
    if args.hf_fused_mamba_norm:
        cfg.model.hybrid_stack_spec = copy.deepcopy(cfg.model._resolve_hybrid_stack_spec())
        cfg.model.hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.module = HFFusedRMSNormMambaMixer
    if isinstance(cfg.model, FalconH1ModelProvider):
        cfg.model.pipeline_dtype = torch.float32
        cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
        cfg.mixed_precision.pipeline_dtype = torch.float32
        # 所有并行配置均允许 Transformer Engine 选择 Attention 实现。
        if getattr(cfg.model, "attention_backend", None) == AttnBackend.unfused:
            cfg.model.attention_backend = AttnBackend.auto
    cfg.model.tensor_model_parallel_size = args.tp
    cfg.model.pipeline_model_parallel_size = args.pp
    num_layers = cfg.model.num_layers
    if args.vpp > 1:
        chunks = uniform_decoder_chunks(num_layers, args.pp, args.vpp)
        cfg.model.virtual_pipeline_model_parallel_size = args.vpp
        cfg.model.decoder_seq_length = args.sequence
        # num_layers_per_virtual_pipeline_stage は均匀の場合のみ整数として渡す
        per_vpp = num_layers // (args.pp * args.vpp)
        if num_layers % (args.pp * args.vpp) == 0:
            cfg.model.num_layers_per_virtual_pipeline_stage = per_vpp
        else:
            cfg.model.num_layers_per_virtual_pipeline_stage = None
            cfg.model.decoder_first_pipeline_num_layers = None
            # decoder_chunks を直接設定する（Bridgeモデルが対応している場合）
            if hasattr(cfg.model, "decoder_chunks"):
                cfg.model.decoder_chunks = chunks
        cfg.model.overlap_p2p_comm = True
        cfg.model.batch_p2p_comm = False
    elif cfg.model.num_layers % args.pp:
        # 非均匀1F1B分区由外层入口依据decoder_chunks构建完整Pipeline Layout。
        cfg.model.num_layers_in_first_pipeline_stage = None
        cfg.model.num_layers_in_last_pipeline_stage = None
        cfg.model.virtual_pipeline_model_parallel_size = None
        cfg.model.overlap_p2p_comm = False
        cfg.model.batch_p2p_comm = True
    else:
        cfg.model.virtual_pipeline_model_parallel_size = None
        cfg.model.overlap_p2p_comm = False
        cfg.model.batch_p2p_comm = True
    cfg.model.context_parallel_size = args.cp
    cfg.model.expert_model_parallel_size = args.ep
    cfg.model.expert_tensor_parallel_size = args.tp
    if args.ep > 1:
        assert cfg.model.num_moe_experts is not None
        cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.sequence_parallel = args.tp > 1
    cfg.model.seq_length = args.sequence
    cfg.model.recompute_granularity = None
    if args.unfused_bda:
        cfg.model.bias_dropout_fusion = False
    if args.recompute:
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "uniform"
        # 设置 uniform 重计算的分组大小。
        vpp_depth = args.vpp if args.vpp > 1 else 1
        cfg.model.recompute_num_layers = 1 if cfg.model.mtp_num_layers else max(1, num_layers // (args.pp * vpp_depth))
    if args.recompute_mlp:
        cfg.model.recompute_granularity = "selective"
        cfg.model.recompute_modules = ["mlp"]
    cfg.train.train_iters = args.warmup + args.measured
    cfg.train.micro_batch_size = args.micro_batch
    cfg.train.global_batch_size = args.global_batch
    cfg.dataset.seq_length = args.sequence
    cfg.dataset.data_path = str(args.data)
    cfg.tokenizer.tokenizer_model = str(args.model)
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True, "use_fast": True}
    cfg.checkpoint.pretrained_checkpoint = None if args.load_hf_weights else str(args.checkpoint)
    if args.cpu_offload:
        cfg.optimizer.optimizer_cpu_offload = True
        cfg.optimizer.optimizer_offload_fraction = 1.0
        cfg.optimizer.use_precision_aware_optimizer = False
        cfg.optimizer.use_torch_optimizer_for_cpu_offload = False
        cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = True
    pretrain(config=cfg, forward_step_func=forward_step, callbacks=[Evidence(args)])


if __name__ == "__main__":
    main()
