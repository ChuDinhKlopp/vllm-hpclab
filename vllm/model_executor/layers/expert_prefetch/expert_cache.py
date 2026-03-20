import torch
import torch.nn as nn
from collections.abc import Callable

from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe import FusedMoE


class ExpertBuffer(nn.Module):
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w13_bias: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    w2_bias: torch.Tensor

    def __init__(self, num_experts: int, name: str):
        super().__init__()
        self.num_experts = num_experts
        # Keep cached expert IDs on GPU when available to avoid CPU<->GPU hops.
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.cached_expert_ids: torch.Tensor = torch.empty(
            0, dtype=torch.int32, device=device
        )
        self.avail = True
        self.prefetch_event: torch.cuda.Event | None = None

    def record_expert_ids(self, expert_ids: torch.Tensor):
        self.cached_expert_ids = expert_ids

    def get_cached_expert_ids(self):
        return self.cached_expert_ids

    def is_avail(self):
        return self.avail

    def create_buffer(
        self,
        layer: FusedMoE,
        cached_parameter_names: tuple[str, ...],
    ):
        # OLD(ducct): hard-coded MXFP4 cache layout.
        # hidden_size = config.hidden_size
        # intermediate_size = config.intermediate_size
        # tp_size = get_tensor_model_parallel_world_size()
        # assert intermediate_size % tp_size == 0
        # intermediate_size_per_partition = intermediate_size // tp_size
        # intermediate_size_per_partition_after_pad = intermediate_size_per_partition
        # intermediate_size_per_partition_after_pad = round_up(
        #     intermediate_size_per_partition, 128
        # )
        # if current_platform.is_xpu():
        #     hidden_size = round_up(hidden_size, 128)
        # else:
        #     hidden_size = round_up(hidden_size, 256)
        #
        # self.w13_weight: torch.Tensor = torch.nn.Parameter(
        #     torch.zeros(
        #         self.num_experts,
        #         2 * intermediate_size_per_partition_after_pad,
        #         hidden_size // 2,
        #         dtype=weight_dtype,
        #     ),
        #     requires_grad=False,
        # )
        # self.w13_weight_scale: torch.Tensor = torch.nn.Parameter(
        #     torch.zeros(
        #         self.num_experts,
        #         2 * intermediate_size_per_partition_after_pad,
        #         hidden_size // mxfp4_block,
        #         dtype=scale_dtype,
        #     ),
        #     requires_grad=False,
        # )
        # self.w13_bias: torch.Tensor = torch.nn.Parameter(
        #     torch.zeros(
        #         self.num_experts,
        #         2 * intermediate_size_per_partition_after_pad,
        #         dtype=torch.bfloat16,
        #     ),
        #     requires_grad=False,
        # )
        # self.w2_weight: torch.Tensor = torch.nn.Parameter(
        #     torch.zeros(
        #         self.num_experts,
        #         hidden_size,
        #         intermediate_size_per_partition_after_pad // 2,
        #         dtype=weight_dtype,
        #     ),
        #     requires_grad=False,
        # )
        # self.w2_weight_scale: torch.Tensor = torch.nn.Parameter(
        #     torch.zeros(
        #         self.num_experts,
        #         hidden_size,
        #         intermediate_size_per_partition_after_pad // mxfp4_block,
        #         dtype=scale_dtype,
        #     ),
        #     requires_grad=False,
        # )
        # self.w2_bias: torch.Tensor = torch.nn.Parameter(
        #     torch.zeros(
        #         self.num_experts,
        #         hidden_size,
        #         dtype=torch.bfloat16,
        #     ),
        #     requires_grad=False,
        # )

        for param_name in cached_parameter_names:
            source_param = getattr(layer, param_name)
            cached_param = torch.nn.Parameter(
                torch.zeros(
                    (self.num_experts, *source_param.shape[1:]),
                    dtype=source_param.dtype,
                    # device="cuda",
                ),
                requires_grad=False,
            )
            print(f"cached_param: {cached_param.device}")
            setattr(self, param_name, cached_param)

    def _copy_float8_rows(self, dst, src, expert_ids, n):
        # dst: GPU float8 tensor, src: CPU float8 tensor
        dst_u8 = dst[:n].view(torch.uint8)
        src_u8 = src.view(torch.uint8)[expert_ids]
        dst_u8.copy_(src_u8)

    def fetch_on_demand(self, layer, expert_ids, slot_ids: torch.Tensor | None = None):
        expert_ids = expert_ids.reshape(-1)
        if expert_ids.numel() == 0:
            return

        # Map global expert ids -> local expert ids when EP is enabled.
        # Filter out experts not owned by this rank (mapped to -1).
        # with torch.profiler.record_function("expert_ids.map_and_filter"):
        if getattr(layer, "expert_map", None) is not None:
            map_device = layer.expert_map.device
            local_ids = layer.expert_map[
                expert_ids.to(map_device, dtype=torch.long)
            ]
            keep = local_ids >= 0
            if not torch.any(keep):
                return
            local_ids = local_ids[keep]
        else:
            local_ids = expert_ids

        # with torch.profiler.record_function("expert_ids.to_device"):
        local_ids = local_ids.to(layer.w13_weight.device, dtype=torch.long)
        if slot_ids is not None:
            slot_ids = slot_ids.to(layer.w13_weight.device, dtype=torch.long)

        num_expert_ids = local_ids.numel()

        if slot_ids is None:
            slot_ids = torch.arange(
                num_expert_ids,
                device=layer.w13_weight.device,
                dtype=torch.long,
            )
        # NOTE(ducct): This code yields correct result
        # self.w13_weight[slot_ids] = layer.w13_weight[local_ids].to("cuda")
        # self.w13_bias[slot_ids] = layer.w13_bias[local_ids].to("cuda")
        # self.w2_weight[slot_ids] = layer.w2_weight[local_ids].to("cuda")
        # self.w2_bias[slot_ids] = layer.w2_bias[local_ids].to("cuda")

        # # float8 scales: index via uint8 view
        # # Use explicit slot_ids so cache rows align with cached_expert_ids.
        # dst_u8 = self.w13_weight_scale.view(torch.uint8)
        # src_u8 = layer.w13_weight_scale.view(torch.uint8)[local_ids]
        # dst_u8[slot_ids] = src_u8.to("cuda")

        # dst_u8 = self.w2_weight_scale.view(torch.uint8)
        # src_u8 = layer.w2_weight_scale.view(torch.uint8)[local_ids]
        # dst_u8[slot_ids] = src_u8.to("cuda")

        # logger.info(f"layer.w13_weight: {layer.w13_weight.shape}")
        # logger.info(f"layer.w13_weight_scale: {layer.w13_weight_scale.shape}")
        # logger.info(f"layer.w13_bias: {layer.w13_bias.shape}")
        # logger.info(f"layer.w2_weight: {layer.w2_weight.shape}")
        # logger.info(f"layer.w2_weight_scale: {layer.w2_weight_scale.shape}")
        # logger.info(f"layer.w2_bias: {layer.w2_bias.shape}")

        # OLD(ducct): hard-coded MXFP4 cache copy path.
        # for i, expert_id in enumerate(local_ids.tolist()):
        #     self.w13_weight[i].copy_(layer.w13_weight[expert_id].pin_memory(), non_blocking=True)
        #     self.w13_weight_scale[i].copy_(layer.w13_weight_scale[expert_id].pin_memory(), non_blocking=True)
        #     self.w13_bias[i].copy_(layer.w13_bias[expert_id].pin_memory(), non_blocking=True)
        #     self.w2_weight[i].copy_(layer.w2_weight[expert_id].pin_memory(), non_blocking=True)
        #     self.w2_weight_scale[i].copy_(layer.w2_weight_scale[expert_id].pin_memory(), non_blocking=True)
        #     self.w2_bias[i].copy_(layer.w2_bias[expert_id].pin_memory(), non_blocking=True)

        cached_parameter_names = getattr(
            layer.expert_cache,
            "cached_parameter_names",
            (),
        )
        for slot_id, expert_id in zip(slot_ids.tolist(), local_ids.tolist()):
            for param_name in cached_parameter_names:
                cache_param = getattr(self, param_name)
                layer_param = getattr(layer, param_name)
                # print(f"cache_param: {cache_param.device}")
                # print(f"layer_param: {layer_param.device}")
                cache_param[slot_id].copy_(
                    layer_param[expert_id].pin_memory(),
                    non_blocking=True,
                )


class ExpertCache(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.ping_buffer = ExpertBuffer(num_experts=num_experts, name="ping")
        self.pong_buffer = ExpertBuffer(num_experts=num_experts, name="pong")
        self.active_buffer = "ping"
        # Bound at model init so we can reuse FusedMoE's loader logic.
        self.owner_fused_moe: FusedMoE | None = None
        self.cached_parameter_names: tuple[str, ...] = ()

    # NOTE(ducct): custom weight loader for expert cache
    def cached_weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        selected_expert_ids: list[int],
        key: str,
        return_success: bool = False,
    ) -> bool | None:
        owner = self.owner_fused_moe
        if owner is None:
            raise AttributeError(
                "ExpertCache.owner_fused_moe is not set; cannot reuse "
                "FusedMoE.cached_weight_loader."
            )
        return FusedMoE.cached_weight_loader(
            owner,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            selected_expert_ids=selected_expert_ids,
            key=key,
            return_success=return_success,
        )


    def _resolve_cached_parameter_names(self, layer: FusedMoE) -> tuple[str, ...]:
        quant_method_name = layer.quant_method.__class__.__name__
        print(f"quant_method_name: {quant_method_name}")
        if quant_method_name == "Mxfp4MoEMethod":
            return (
                "w13_weight",
                "w13_weight_scale",
                "w13_bias",
                "w2_weight",
                "w2_weight_scale",
                "w2_bias",
            )
        if quant_method_name == "UnquantizedFusedMoEMethod":
            print(f"[ducct] I'm in here brooo")
            names = ["w13_weight", "w2_weight"]
            if hasattr(layer, "w13_bias"):
                names.append("w13_bias")
            if hasattr(layer, "w2_bias"):
                names.append("w2_bias")
            return tuple(names)

        return tuple(
            name
            for name in (
                "w13_weight",
                "w13_weight_scale",
                "w13_bias",
                "w2_weight",
                "w2_weight_scale",
                "w2_bias",
            )
            if hasattr(layer, name)
        )

    def create_cache(self, owner_fused_moe: FusedMoE):
        # OLD(ducct): fixed MXFP4-only cache init signature.
        # def create_cache(
        #     self,
        #     config,
        #     mxfp4_block,
        #     weight_dtype,
        #     scale_dtype,
        # ):
        self.owner_fused_moe = owner_fused_moe
        extra_weight_attrs = {
            "cached_weight_loader": self.cached_weight_loader
        }
        self.cached_parameter_names = self._resolve_cached_parameter_names(
            owner_fused_moe
        )
        if not self.cached_parameter_names:
            raise NotImplementedError(
                "Expert cache could not determine cached parameter layout for "
                f"{owner_fused_moe.quant_method.__class__.__name__}."
            )
        # Avoid reallocating shared buffers if they already exist.
        if (
            self.ping_buffer is not None
            and self.pong_buffer is not None
            and getattr(self.ping_buffer, "w13_weight", None) is not None
            and getattr(self.pong_buffer, "w13_weight", None) is not None
        ):
            return
        self.ping_buffer.create_buffer(
            owner_fused_moe,
            self.cached_parameter_names,
        )
        # OLD(ducct): explicit MXFP4-only registration.
        # self.register_parameter("expert_cache_ping_w13_weight", self.ping_buffer.w13_weight)
        # set_weight_attrs(self.ping_buffer.w13_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w13_weight_scale", self.ping_buffer.w13_weight_scale)
        # set_weight_attrs(self.ping_buffer.w13_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w13_bias", self.ping_buffer.w13_bias)
        # set_weight_attrs(self.ping_buffer.w13_bias, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w2_weight", self.ping_buffer.w2_weight)
        # set_weight_attrs(self.ping_buffer.w2_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w2_weight_scale", self.ping_buffer.w2_weight_scale)
        # set_weight_attrs(self.ping_buffer.w2_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w2_bias", self.ping_buffer.w2_bias)
        # set_weight_attrs(self.ping_buffer.w2_bias, extra_weight_attrs)
        for param_name in self.cached_parameter_names:
            param = getattr(self.ping_buffer, param_name)
            self.register_parameter(f"expert_cache_ping_{param_name}", param)
            set_weight_attrs(param, extra_weight_attrs)

        self.pong_buffer.create_buffer(
            owner_fused_moe,
            self.cached_parameter_names,
        )
        # OLD(ducct): explicit MXFP4-only registration.
        # self.register_parameter("expert_cache_pong_w13_weight", self.pong_buffer.w13_weight)
        # set_weight_attrs(self.pong_buffer.w13_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w13_weight_scale", self.pong_buffer.w13_weight_scale)
        # set_weight_attrs(self.pong_buffer.w13_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w13_bias", self.pong_buffer.w13_bias)
        # set_weight_attrs(self.pong_buffer.w13_bias, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w2_weight", self.pong_buffer.w2_weight)
        # set_weight_attrs(self.pong_buffer.w2_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w2_weight_scale", self.pong_buffer.w2_weight_scale)
        # set_weight_attrs(self.pong_buffer.w2_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w2_bias", self.pong_buffer.w2_bias)
        # set_weight_attrs(self.pong_buffer.w2_bias, extra_weight_attrs)
        for param_name in self.cached_parameter_names:
            param = getattr(self.pong_buffer, param_name)
            self.register_parameter(f"expert_cache_pong_{param_name}", param)
            set_weight_attrs(param, extra_weight_attrs)

    def get_active_buffer(self):
        if self.active_buffer == "ping":
            return self.ping_buffer
        return self.pong_buffer

    def get_inactive_buffer(self):
        if self.active_buffer == "ping":
            return self.pong_buffer
        return self.ping_buffer

    def flip_active_buffer(self):
        self.active_buffer = "pong" if self.active_buffer == "ping" else "ping"

    def prefetch(
        self,
        predicted_expert_ids,
        prefetch_fn: Callable[[], None] | None = None,
        stream: torch.cuda.Stream | None = None,
    ):
        # NOTE(ducct): mark inactive buffer unavailable and record completion event.
        inactive_cache = self.get_inactive_buffer()
        inactive_cache.avail = False
        if torch.cuda.is_available():
            if stream is None:
                stream = torch.cuda.current_stream()
            with torch.cuda.stream(stream):
                if prefetch_fn is not None:
                    prefetch_fn()
                inactive_cache.prefetch_event = torch.cuda.Event()
                inactive_cache.prefetch_event.record(stream)
        else:
            if prefetch_fn is not None:
                prefetch_fn()
            inactive_cache.avail = True
