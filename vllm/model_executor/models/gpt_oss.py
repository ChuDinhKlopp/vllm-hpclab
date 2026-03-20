# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.utils import rocm_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer, # NOTE(ducct)
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
# NOTE(ducct): 
from vllm.logger import init_logger
import vllm.envs as envs
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.math_utils import round_up

from vllm.model_executor.layers.expert_prefetch import ExpertPredictorModel, ExpertCache

import time
import os
import json
import time
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
from collections.abc import Iterable, Callable

# Hidden-state dump (for correctness checks)
ENABLE_HIDDEN_STATE_DUMP = os.getenv("ENABLE_HIDDEN_STATE_DUMP", "0") == "1"
HIDDEN_STATE_LOG_DIR = Path(
    os.getenv("HIDDEN_STATE_LOG_DIR", "/run/user/1019/ducct/logs/hidden_state/vllm-pref")
)
HIDDEN_STATE_MAX_STEPS_PER_FILE = int(
    os.getenv("HIDDEN_STATE_MAX_STEPS_PER_FILE", "1")
)
HIDDEN_STATE_FLUSH_EVERY_STEP = os.getenv(
    "HIDDEN_STATE_FLUSH_EVERY_STEP", "0"
) == "1"
HIDDEN_STATE_TARGET_LAYER = os.getenv("HIDDEN_STATE_TARGET_LAYER")
if HIDDEN_STATE_TARGET_LAYER is not None and HIDDEN_STATE_TARGET_LAYER != "":
    HIDDEN_STATE_TARGET_LAYER = int(HIDDEN_STATE_TARGET_LAYER)

class HiddenStateLogger:
    def __init__(
        self,
        log_dir: Path,
        max_steps_per_file: int = 1,
        filename_format: str = "hidden_states_device_{device:01d}_{start:06d}_{end:06d}.pt",
    ):
        self.log_dir = log_dir
        self.max_steps_per_file = max_steps_per_file
        self.filename_format = filename_format
        self.buffer: List[Dict] = []
        self.current_start_step: int | None = None
        self.current_device_id: int | None = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_device_id(device: torch.device) -> int:
        if device.type == "cuda":
            return device.index
        return 0  # CPU -> device_0

    def _to_cpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._to_cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [self._to_cpu(v) for v in value]
            return type(value)(converted)
        return value

    def log(
        self,
        step: int,
        layer_id: int,
        tensor: torch.Tensor,
        device: torch.device,
        cached_weights: dict | None = None,
    ) -> None:
        device_id = self._normalize_device_id(device)
        if self.current_device_id is not None and device_id != self.current_device_id:
            self.flush()

        if self.current_start_step is None:
            self.current_start_step = step
            self.current_device_id = device_id

        entry = {
            "step": step,
            "layer": layer_id,
            "tensor": tensor.detach().cpu(),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device_id": device_id,
        }
        if cached_weights is not None:
            entry["cached_weights"] = self._to_cpu(cached_weights)
        self.buffer.append(entry)

        if self.max_steps_per_file:
            if step - self.current_start_step >= self.max_steps_per_file:
                self.flush()
        if HIDDEN_STATE_FLUSH_EVERY_STEP:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        steps = [e["step"] for e in self.buffer]
        start_step = min(steps)
        end_step = max(steps)
        device_id = self.current_device_id or 0
        filename = self.filename_format.format(
            device=device_id, start=start_step, end=end_step
        )
        torch.save(
            {"entries": self.buffer, "step_range": (start_step, end_step)},
            self.log_dir / filename,
        )
        self.buffer = []
        self.current_start_step = None
        self.current_device_id = None


hidden_state_logger = HiddenStateLogger(
    log_dir=HIDDEN_STATE_LOG_DIR,
    max_steps_per_file=HIDDEN_STATE_MAX_STEPS_PER_FILE,
)


class OAIAttention(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type": "yarn",
                "factor": config.rope_scaling["factor"],
                "original_max_position_embeddings": config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                "beta_fast": config.rope_scaling["beta_fast"],
                "beta_slow": config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size, requires_grad=False)
        )

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        # Only apply sliding window to every other layer
        sliding_window = config.sliding_window if self.layer_idx % 2 == 0 else None
        self.attn = Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        v = v.contiguous()
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.router = torch.nn.Linear(config.hidden_size, config.num_local_experts)
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            apply_router_weight_on_input=False,
            has_bias=True,
            activation="swigluoai",
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        if self.is_sequence_parallel:
            x = sequence_parallel_chunk(x)

        if current_platform.is_rocm():
            g = rocm_unquantized_gemm(
                self, x[:, : self.hidden_size], self.router.weight, self.router.bias
            )
        else:
            g = self.router(x)
        x = self.experts(hidden_states=x, router_logits=g)

        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x.contiguous(), 0)
            x = x[:num_tokens]
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.layer_idx = extract_layer_index(prefix)
        self.attn = OAIAttention(
            config,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.mlp = MLPBlock(vllm_config, self.layer_idx, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        # NOTE(ducct): add expert predictor
        # self.expert_predictor = ExpertPredictorModel(
        #     weight_path="/home/ducct/repos/profiling/trace-analysis/vllm-offload/epoch=01-val_acc=0.9493.ckpt",
        #     input_dim=2880,
        #     num_experts=config.num_local_experts,
        #     device="cpu",
        # )
        self.top_k = self.mlp.experts.top_k


    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        with torch.profiler.record_function("ducct::layer_norm+attn"):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.attn(hidden_states, positions)

        # TODO(ducct): Predict + prefetch expert weights for next layer here only if you want to prefectch after attn
        # predictor runs on CPU -> transfer hidden_states back to CPU for computation
        # The output of the expert predictor, predicted_topk_ids is used to form a CPU tensor of size (num_predicted_experts, inter_dim, hidden_dim)
        # Then we copy this tensor to next layer's cache. How do we access next layer's cache here?
        next_layer = getattr(self, "next_layer", None)
        if next_layer is not None and hidden_states.is_cuda:
            prefetch_stream = getattr(self, "_prefetch_stream", None)
            if prefetch_stream is None:
                self._prefetch_stream = torch.cuda.Stream()
                prefetch_stream = self._prefetch_stream

            # 1) D2H copy on prefetch stream
            # with torch.profiler.record_function("expert_prefetch.d2h_hidden_states"):
            with torch.cuda.stream(prefetch_stream):
                hs_cpu = hidden_states.detach().to("cpu", non_blocking=True)
            prefetch_stream.synchronize() # ensure d2h done before CPU prediction

            # 2) NOTE(ducct): implement CPU expert predictor
            moe = next_layer.mlp.experts
            # with torch.profiler.record_function("ducct::expert_predictor"):
            #     predicted_ids = self.expert_predictor.predict_batch(
            #         hs_cpu, top_k=self.top_k
            #     )["indices"]  # CPU
            predicted_ids = torch.tensor([0,4,2,9], device="cpu")

            # NOTE(ducct):Normalize predicted ids to a unique 1D list (cache expects <= num_experts).
            # with torch.profiler.record_function("expert_ids.check_and_normalize"):
            predicted_ids = predicted_ids.reshape(-1)
            predicted_ids = torch.unique(predicted_ids)
            max_cache = moe.expert_cache.ping_buffer.w13_weight.shape[0]
            if predicted_ids.numel() > max_cache:
                predicted_ids = predicted_ids[:max_cache]

            # with torch.profiler.record_function("expert_ids.h2d_cache_ids"):
            if moe.expert_cache.active_buffer == "ping":
                moe.cached_expert_ids_pong = predicted_ids.detach().to("cuda")
            else:
                moe.cached_expert_ids_ping = predicted_ids.detach().to("cuda")

            # 3) H2D prefetch on prefetch stream
            def do_prefetch():
                # with torch.profiler.record_function("expert_prefetch.fetch_on_demand"):
                    inactive_bf = moe.expert_cache.get_inactive_buffer()
                    inactive_bf.fetch_on_demand(moe, predicted_ids)

            with torch.cuda.stream(prefetch_stream):
                with torch.profiler.record_function("ducct::prefetch"):
                    moe.expert_cache.prefetch(predicted_ids, prefetch_fn=do_prefetch, stream=prefetch_stream)


        # Fully Connected
        with torch.profiler.record_function("ducct::layer_norm+moe"):
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            output = self.mlp(hidden_states)

        return output, residual


@support_torch_compile
class GptOssModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: TransformerBlock(
                vllm_config,
                prefix=prefix,
                quant_config=self.quant_config,
            ),
            prefix=f"{prefix}.layers",
        )
        # NOTE(ducct): Link neighboring layers without registering as submodules.
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                continue
            next_layer = self.layers[i + 1] if i + 1 < len(self.layers) else None
            layer.__dict__["next_layer"] = next_layer

        # NOTE(ducct): init expert cache
        self.expert_cache = ExpertCache(
            num_experts=self.config.num_local_experts
        )
        # OLD(ducct): MXFP4-specific cache creation.
        # self.expert_cache.create_cache(
        #     config=self.config,
        #     mxfp4_block=32,
        #     weight_dtype=torch.uint8,
        #     scale_dtype=torch.uint8,
        # )
        owner_fused_moe = next(
            (
                layer.mlp.experts for layer in self.layers
                if not isinstance(layer, PPMissingLayer)
            ),
            None,
        )
        if owner_fused_moe is None:
            raise RuntimeError("Failed to find a local FusedMoE layer for expert cache.")
        self.expert_cache.create_cache(owner_fused_moe=owner_fused_moe)
        # NOTE(ducct): Share the model-level expert cache across all MoE layers.
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            # NOTE(ducct): Keep shared cache reachable without registering as a submodule
            # to avoid per-layer naming prefixes in state_dict/params.
            layer.mlp.experts.__dict__["expert_cache"] = self.expert_cache

        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        # NOTE(ducct): 
        self._hidden_state_step = 0


    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(ducct):
        if ENABLE_HIDDEN_STATE_DUMP:
            self._hidden_state_step += 1
            hidden_state_step = self._hidden_state_step

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.embed_input_ids(input_ids)

            residual = None
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            # NOTE(ducct): log
            envs.LAYER_ID = i

            layer = self.layers[i]
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(x if residual is None else x + residual)
            x, residual = layer(x, positions, residual)
            # NOTE(ducct): Hidden-state dump (for correctness checks)
            if (
                ENABLE_HIDDEN_STATE_DUMP
                and hidden_state_step is not None
                and (
                    HIDDEN_STATE_TARGET_LAYER is None
                    or HIDDEN_STATE_TARGET_LAYER == i
                )
            ):
                hidden = x if residual is None else x + residual
                cached_weights = None
                if getattr(self, "expert_cache", None) is not None:
                    active_buffer = self.expert_cache.get_active_buffer()
                    # OLD(ducct): unconditional MXFP4-only dump payload.
                    # cached_weights = {
                    #     "active_buffer": self.expert_cache.active_buffer,
                    #     "w13_weight": active_buffer.w13_weight,
                    #     "w13_weight_scale": active_buffer.w13_weight_scale,
                    #     "w13_bias": active_buffer.w13_bias,
                    #     "w2_weight": active_buffer.w2_weight,
                    #     "w2_weight_scale": active_buffer.w2_weight_scale,
                    #     "w2_bias": active_buffer.w2_bias,
                    # }
                    cached_weights = {
                        "active_buffer": self.expert_cache.active_buffer,
                    }
                    for param_name in self.expert_cache.cached_parameter_names:
                        cached_weights[param_name] = getattr(active_buffer, param_name)
                hidden_state_logger.log(
                    step=hidden_state_step,
                    layer_id=i,
                    tensor=hidden,
                    device=hidden.device,
                    cached_weights=cached_weights,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x, "residual": residual})
        x, _ = self.norm(x, residual)

        if len(aux_hidden_states) > 0:
            return x, aux_hidden_states
        return x

    def _load_weights_mxfp4(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        mxfp4_block = 32
        use_ep = self.parallel_config.enable_expert_parallel
        num_experts = self.config.num_local_experts
        # NOTE(ducct): add num_global_experts
        global_selected_expert_ids = list(range(num_experts))


        # In MoE, we need to flatten the tensor parallel size across the data
        # parallel size when EP is disabled.
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp(
            tp_size=get_tensor_model_parallel_world_size(),
            dp_size=get_dp_group().world_size,
            dp_rank=get_dp_group().rank_in_group,
        )

        intermediate_size = self.config.intermediate_size
        intermediate_size_block = intermediate_size // mxfp4_block
        per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
        per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        for name, weight in weights:
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            if ".w13_weight_scale" in name:
                # Handle MLP gate and up projection weights scale
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]

                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids


                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                # NOTE(ducct): pre-load layer 0 & 1's expert weights into the cache
                # This means that we should check the name of the weight if it contains "layer_0" or "layer_1". If true, we perform preloading to expert cache.
                if "layers.0." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_ping_{leaf}"
                    # cached_param = params_dict.get(key)
                    cached_param = params_dict.get(key)
                    if cached_param is None and ".mlp.experts." in key:
                        suffix = key.split(".mlp.experts.", 1)[1]
                        cached_param = params_dict.get(
                            f"expert_cache_params.{suffix}"
                        )
                    if cached_param is None:
                        loaded_params.add(name)
                        continue
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                elif "layers.1." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_pong_{leaf}"
                    # cached_param = params_dict.get(key)
                    cached_param = params_dict.get(key)
                    if cached_param is None and ".mlp.experts." in key:
                        suffix = key.split(".mlp.experts.", 1)[1]
                        cached_param = params_dict.get(
                            f"expert_cache_params.{suffix}"
                        )
                    if cached_param is None:
                        loaded_params.add(name)
                        continue
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    # NUM_CACHED_EXPERTS = 16
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                loaded_params.add(name)
                continue
            elif ".w2_weight_scale" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]

                else:
                    narrow_weight = weight[
                        ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                    ]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]


                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                # NOTE(ducct): pre-load layer 1 & 2's expert weights into the cache
                # This means that we should check the name of the weight if it contains "layer_1" or "layer_2". If true, we perform preloading to expert cache.
                if "layers.0." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_ping_{leaf}"
                    # cached_param = params_dict.get(key)
                    cached_param = params_dict.get(key)
                    if cached_param is None and ".mlp.experts." in key:
                        suffix = key.split(".mlp.experts.", 1)[1]
                        cached_param = params_dict.get(
                            f"expert_cache.{suffix}"
                        )
                    if cached_param is None:
                        loaded_params.add(name)
                        continue
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                elif "layers.1." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_pong_{leaf}"
                    cached_param = params_dict.get(key)
                    if cached_param is None and ".mlp.experts." in key:
                        suffix = key.split(".mlp.experts.", 1)[1]
                        cached_param = params_dict.get(
                            f"expert_cache.{suffix}"
                        )
                    if cached_param is None:
                        loaded_params.add(name)
                        continue
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    # NUM_CACHED_EXPERTS = 16
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                loaded_params.add(name)
                continue
            elif ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(
                    num_experts, 2 * intermediate_size, -1
                ).contiguous()

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]

                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids


                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                # NOTE(ducct): pre-load layer 1 & 2's expert weights into the cache
                # This means that we should check the name of the weight if it contains "layer_1" or "layer_2". If true, we perform preloading to expert cache.
                if "layers.0." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_ping_{leaf}"
                    cached_param = params_dict.get(key)
                    if cached_param is None and ".mlp.experts." in key:
                        suffix = key.split(".mlp.experts.", 1)[1]
                        cached_param = params_dict.get(
                            f"expert_cache.{suffix}"
                        )
                    if cached_param is None:
                        loaded_params.add(name)
                        continue
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                elif "layers.1." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_pong_{leaf}"
                    # cached_param = params_dict.get(key)
                    cached_param = params_dict.get(key)
                    if cached_param is None and ".mlp.experts." in key:
                        suffix = key.split(".mlp.experts.", 1)[1]
                        cached_param = params_dict.get(
                            f"expert_cache.{suffix}"
                        )
                    if cached_param is None:
                        loaded_params.add(name)
                        continue
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(
                    num_experts, -1, intermediate_size // 2
                ).contiguous()
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]

                else:
                    narrow_weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids


                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                # NOTE(ducct): pre-load layer 1 & 2's expert weights into the cache
                # This means that we should check the name of the weight if it contains "layer_1" or "layer_2". If true, we perform preloading to expert cache.
                if "layers.0." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_ping_{leaf}"
                    cached_param = params_dict[key]
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                elif "layers.1." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_pong_{leaf}"
                    cached_param = params_dict[key]
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]

                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids


                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                # NOTE(ducct): pre-load layer 1 & 2's expert weights into the cache
                # This means that we should check the name of the weight if it contains "layer_1" or "layer_2". If true, we perform preloading to expert cache.
                if "layers.0." in name:

                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_ping_{leaf}"
                    cached_param = params_dict[key]
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                elif "layers.1." in name:

                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_pong_{leaf}"
                    cached_param = params_dict[key]
                    cached_weight_loader = getattr(cached_param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids[ep_rank_start:ep_rank_end]

                else:
                    # NOTE(ducct):
                    local_selected_expert_ids = global_selected_expert_ids

                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                weight_loader(
                    param, weight, weight_name=name, shard_id=None, expert_id=None
                )
                # NOTE(ducct): pre-load layer 1 & 2's expert weights into the cache
                # This means that we should check the name of the weight if it contains "layer_1" or "layer_2". If true, we perform preloading to expert cache.
                if "layers.0." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_ping_{leaf}"
                    cached_param = params_dict[key]
                    cached_weight_loader = getattr(param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                elif "layers.1." in name:
                    base, leaf = name.rsplit(".", 1)
                    key = f"expert_cache.expert_cache_pong_{leaf}"
                    cached_param = params_dict[key]
                    cached_weight_loader = getattr(param, "cached_weight_loader", default_weight_loader)
                    cached_weight_loader(
                        cached_param,
                        narrow_weight,
                        weight_name=name,
                        shard_id=None,
                        expert_id=None,
                        selected_expert_ids=local_selected_expert_ids,
                        key=key,
                    )

                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def _load_weights_other(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        use_ep = self.parallel_config.enable_expert_parallel

        # In MoE, we need to flatten the tensor parallel size across the data
        # parallel size when EP is disabled.
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp(
            tp_size=get_tensor_model_parallel_world_size(),
            dp_size=get_dp_group().world_size,
            dp_rank=get_dp_group().rank_in_group,
        )

        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        for name, weight in weights:
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            if ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # Extract gate and up projection parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]

                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                param = params_dict[name]
                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                param = params_dict[name]
                param.copy_(weight)
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank
        num_experts = self.config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        quant_method = (
            self.config.quantization_config["quant_method"]
            if hasattr(self.config, "quantization_config")
            else None
        )
        if quant_method == "mxfp4":
            return self._load_weights_mxfp4(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )
        else:
            return self._load_weights_other(
                ep_rank_start,
                ep_rank_end,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )


class GptOssForCausalLM(nn.Module, SupportsPP, SupportsEagle3, SupportsLoRA):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".self_attn.": ".attn.",
        },
        orig_to_new_suffix={
            ".embed_tokens.weight": ".embedding.weight",
            # MoE MXFP4 weights
            ".gate_up_proj_blocks": ".w13_weight",
            ".down_proj_blocks": ".w2_weight",
            ".gate_up_proj_scales": ".w13_weight_scale",
            ".down_proj_scales": ".w2_weight_scale",
            # MoE other weights
            ".gate_up_proj": ".w13_weight",
            ".down_proj": ".w2_weight",
            # MoE Bias
            ".gate_up_proj_bias": ".w13_bias",
            ".down_proj_bias": ".w2_bias",
        },
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, weight scales, activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
