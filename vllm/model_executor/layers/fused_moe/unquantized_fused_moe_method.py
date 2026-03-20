# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts, fused_experts
else:
    fused_experts = None  # type: ignore

if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore

logger = init_logger(__name__)


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

        self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        if self.rocm_aiter_moe_enabled:
            from .rocm_aiter_fused_moe import rocm_aiter_fused_experts

            self.rocm_aiter_fused_experts = rocm_aiter_fused_experts
        else:
            self.rocm_aiter_fused_experts = None  # type: ignore

        # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
        self.flashinfer_cutlass_moe_enabled = (
            has_flashinfer_cutlass_fused_moe()
            and envs.VLLM_USE_FLASHINFER_MOE_FP16
            and self.moe.moe_parallel_config.use_ep
            and self.moe.moe_parallel_config.dp_size == 1
            and current_platform.get_device_capability()[0] >= 9
        )
        if self.flashinfer_cutlass_moe_enabled:
            logger.info_once(
                "Enabling FlashInfer CUTLASS MoE for UnquantizedFusedMoEMethod"
            )
            from functools import partial

            from .flashinfer_cutlass_moe import flashinfer_cutlass_moe

            self.flashinfer_cutlass_moe = partial(
                flashinfer_cutlass_moe,
                quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
                tp_rank=self.moe.moe_parallel_config.tp_rank,
                tp_size=self.moe.moe_parallel_config.tp_size,
                ep_rank=self.moe.moe_parallel_config.ep_rank,
                ep_size=self.moe.moe_parallel_config.ep_size,
            )
        else:
            if (
                self.moe.moe_parallel_config.use_ep
                and self.moe.moe_parallel_config.dp_size == 1
            ):
                logger.info_once(
                    "FlashInfer CUTLASS MoE is available for EP"
                    " but not enabled, consider setting"
                    " VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.",
                    scope="local",
                )
            elif self.moe.moe_parallel_config.dp_size > 1:
                logger.info_once(
                    "FlashInfer CUTLASS MoE is currently not available for DP.",
                    scope="local",
                )
            self.flashinfer_cutlass_moe = None  # type: ignore

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def maybe_make_prepare_finalize(self) -> FusedMoEPrepareAndFinalize | None:
        if self.rocm_aiter_moe_enabled:
            return None
        else:
            return super().maybe_make_prepare_finalize()

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(self.moe_quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()

        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        # NOTE(ducct):
        layer.expert_cache.ping_buffer.w13_weight.data = self._maybe_pad_weight(layer.expert_cache.ping_buffer.w13_weight.data)
        layer.expert_cache.pong_buffer.w13_weight.data = self._maybe_pad_weight(layer.expert_cache.pong_buffer.w13_weight.data)
        layer.expert_cache.ping_buffer.w2_weight.data = self._maybe_pad_weight(layer.expert_cache.ping_buffer.w2_weight.data)
        layer.expert_cache.pong_buffer.w2_weight.data = self._maybe_pad_weight(layer.expert_cache.pong_buffer.w2_weight.data)


        if self.rocm_aiter_moe_enabled:
            shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data
            )

            layer.w13_weight.data = shuffled_w13
            layer.w2_weight.data = shuffled_w2

        if self.flashinfer_cutlass_moe_enabled:
            # Swap halves to arrange as [w3; w1] (kernel expectation)
            w1_w, w3_w = torch.chunk(layer.w13_weight.data, 2, dim=1)
            w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)
            layer.w13_weight.data = w13_weight_swapped.contiguous()

        if current_platform.is_xpu():
            import intel_extension_for_pytorch as ipex

            ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
            layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
                experts_start_id=ep_rank_start,
            )
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe

            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (
                    envs.VLLM_CPU_SGL_KERNEL
                    and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                    and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)
                ):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight
                    )
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight
                    )
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    layer.cpu_fused_moe = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    layer.cpu_fused_moe = cpu_fused_moe.IPEXFusedMOE(layer)
            else:
                layer.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None

        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_eplb=enable_eplb,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        topk_weights, topk_ids, zero_expert_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
            num_fused_shared_experts=layer.num_fused_shared_experts,
        )

        if self.rocm_aiter_moe_enabled:
            result = self.rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        elif self.flashinfer_cutlass_moe_enabled:
            return self.flashinfer_cutlass_moe(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            # TODO(ducct): checking if selected experts (topk_ids) exist in the current layer's expert cache. The mapper is on CPU -> need to:
            # 1. Get the unique set of topk_ids -> unique_expert_ids: done
            active_cache = layer.expert_cache.get_active_buffer()
            inactive_cache = layer.expert_cache.get_inactive_buffer()
            if layer.expert_cache.active_buffer == "ping":
                # NOTE(ducct): temporary fix
                layer.cached_expert_ids_ping = torch.empty(0, dtype=torch.int32, device="cuda")
                cached_expert_ids = layer.cached_expert_ids_ping
                #layer.cached_expert_ids_ping = torch.empty(0, dtype=torch.int32, device="cuda")

            else:
                # NOTE(ducct):temporary fix
                layer.cached_expert_ids_pong = torch.empty(0, dtype=torch.int32, device="cuda")
                cached_expert_ids = layer.cached_expert_ids_pong
                #layer.cached_expert_ids_pong = torch.empty(0, dtype=torch.int32, device="cuda")
            # 2. Transfer the topk_ids from GPU to CPU to compare with cached_expert_ids: done
            unique_selected_ids = torch.unique(topk_ids.reshape(-1))
            # 3. Do the checking on GPU
            expert_mask = torch.isin(unique_selected_ids, cached_expert_ids, assume_unique=True) # check if unique_selected_ids is subset of cached_expert_ids
            is_subset = expert_mask.all().item()

            if not is_subset:
                missing_values = unique_selected_ids[~expert_mask]
                cached_expert_ids = unique_selected_ids
                # 4. if there is missing ids in predicted ids, fetch to GPU on demand
                # NOTE(ducct): fetch on-demand missing experts into the cache. 
                active_cache.fetch_on_demand(layer, cached_expert_ids.to("cpu"))

            # OLD: map expert IDs -> cache slot indices from cached_expert_ids.
            # num_experts = getattr(layer, "num_experts", None) or layer.global_num_experts
            # cached_expert_ids_dev = cached_expert_ids.to(device=topk_ids.device, dtype=torch.long)
            # lookup = torch.full(
            #     (num_experts,),
            #     -1,
            #     device=topk_ids.device,
            #     dtype=torch.int32,
            # )
            # lookup[cached_expert_ids_dev] = torch.arange(
            #     cached_expert_ids_dev.numel(),
            #     device=topk_ids.device,
            #     dtype=torch.int32,
            # )
            # cached_topk_ids = lookup[topk_ids]
            # if (cached_topk_ids < 0).any():
            #     raise RuntimeError("topk_ids contains experts not present in cache.")
            
            # DEBUG(ducct):
            # for slot_id, selected_id in zip(torch.tensor(list(range(cached_expert_ids.numel()))), cached_expert_ids):
            #     assert torch.allclose(active_cache.w13_weight[slot_id].to("cpu"), layer.w13_weight[selected_id], atol=1e-5, rtol=1e-5)
            #     assert torch.allclose(active_cache.w2_weight[slot_id].to("cpu"), layer.w2_weight[selected_id], atol=1e-5, rtol=1e-5)
            #     assert torch.allclose(active_cache.w13_weight_scale[slot_id].to("cpu"), layer.w13_weight_scale[selected_id], atol=1e-5, rtol=1e-5)
            #     assert torch.allclose(active_cache.w2_weight_scale[slot_id].to("cpu"), layer.w2_weight_scale[selected_id], atol=1e-5, rtol=1e-5)
            #     assert torch.allclose(active_cache.w13_bias[slot_id].to("cpu"), layer.w13_bias[selected_id], atol=1e-5, rtol=1e-5)
            #     assert torch.allclose(active_cache.w2_bias[slot_id].to("cpu"), layer.w2_bias[selected_id], atol=1e-5, rtol=1e-5)

            # DEBUG(ducct): for each cache slot, compare against all layer experts and
            # return the matched layer expert id.
            # layer_w13_cpu = (
            #     layer.w13_bias
            #     if layer.w13_bias.device.type == "cpu"
            #     else layer.w13_bias.to("cpu")
            # )
            # layer_w2_cpu = (
            #     layer.w2_bias
            #     if layer.w2_bias.device.type == "cpu"
            #     else layer.w2_bias.to("cpu")
            # )
            # matched_layer_ids_cpu = torch.full(
            #     (cached_expert_ids.numel(),), -1, dtype=torch.long
            # )
            # for slot_id in range(cached_expert_ids.numel()):
            #     cache_w13_cpu = active_cache.w13_bias[slot_id].to("cpu")
            #     cache_w2_cpu = active_cache.w2_bias[slot_id].to("cpu")
            #     for layer_id in range(layer_w13_cpu.shape[0]):
            #         w13_match = (
            #             torch.allclose(
            #                 cache_w13_cpu,
            #                 layer_w13_cpu[layer_id],
            #                 atol=1e-5,
            #                 rtol=1e-5,
            #             )
            #             if cache_w13_cpu.is_floating_point()
            #             else torch.equal(cache_w13_cpu, layer_w13_cpu[layer_id])
            #         )
            #         if not w13_match:
            #             continue
            #         w2_match = (
            #             torch.allclose(
            #                 cache_w2_cpu,
            #                 layer_w2_cpu[layer_id],
            #                 atol=1e-5,
            #                 rtol=1e-5,
            #             )
            #             if cache_w2_cpu.is_floating_point()
            #             else torch.equal(cache_w2_cpu, layer_w2_cpu[layer_id])
            #         )
            #         if w2_match:
            #             matched_layer_ids_cpu[slot_id] = layer_id
            #             break

            # if (matched_layer_ids_cpu < 0).any():
            #     unmatched_slots = torch.where(matched_layer_ids_cpu < 0)[0].tolist()
            #     raise RuntimeError(
            #         "Could not match active_cache slots to layer expert ids. "
            #         f"Unmatched slots: {unmatched_slots}"
            #     )

            # matched_layer_ids = matched_layer_ids_cpu.to(
            #     device=topk_ids.device, dtype=torch.long
            # )
            # print(f"matched_layer_ids_by_slot: {matched_layer_ids}")
            # # Reverse view: print matched slot id keyed by layer weight id.
            # matched_slot_by_layer_cpu = torch.full(
            #     (layer_w13_cpu.shape[0],), -1, dtype=torch.long
            # )
            # matched_slot_by_layer_cpu[matched_layer_ids_cpu] = torch.arange(
            #     matched_layer_ids_cpu.numel(), dtype=torch.long
            # )
            # for layer_weight_id in torch.where(matched_slot_by_layer_cpu >= 0)[0].tolist():
            #     print(
            #         "matched_id_by_layer_weight_id: "
            #         f"layer_weight_id={layer_weight_id}, "
            #         f"matched_slot_id={matched_slot_by_layer_cpu[layer_weight_id].item()}"
            #     )

            # Use cached ids (not debug-only matched ids) to build topk -> cache-slot lookup.
            num_experts = getattr(layer, "num_experts", None) or layer.global_num_experts
            lookup_expert_ids = cached_expert_ids.to(
                device=topk_ids.device, dtype=torch.long
            )
            lookup = torch.full(
                (num_experts,),
                -1,
                device=topk_ids.device,
                dtype=torch.int32,
            )
            # OLD:
            # lookup[cached_expert_ids] = torch.arange(
            #     cached_expert_ids.numel(),
            #     device=topk_ids.device,
            #     dtype=torch.int32,
            # )
            lookup[lookup_expert_ids] = torch.arange(
                lookup_expert_ids.numel(),
                device=topk_ids.device,
                dtype=torch.int32,
            )
            cached_topk_ids = lookup[topk_ids]
            if (cached_topk_ids < 0).any():
                raise RuntimeError(
                    "topk_ids contains experts not present in matched cache slots."
                )


            result = fused_experts(
                hidden_states=x,
                # w1=layer.w13_weight,
                # w2=layer.w2_weight,
                w1=active_cache.w13_weight,
                w2=active_cache.w2_weight,
                topk_weights=topk_weights,
                # topk_ids=topk_ids,
                topk_ids=cached_topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

            # NOTE(ducct): Check if inactive buffer is available (done prefetching).
            # If not available yet, wait on the prefetch event before flipping.
            if not inactive_cache.is_avail():
                if torch.cuda.is_available() and inactive_cache.prefetch_event is not None:
                    torch.cuda.current_stream().wait_event(inactive_cache.prefetch_event)
                inactive_cache.avail = True
            layer.expert_cache.flip_active_buffer()


        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            enable_eplb is not False
            or expert_load_view is not None
            or logical_to_physical_map is not None
            or logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for CPU.")
        return layer.cpu_fused_moe(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            enable_eplb is not False
            or expert_load_view is not None
            or logical_to_physical_map is not None
            or logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for XPU.")
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function=custom_routing_function,
        )

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert apply_router_weight_on_input is False
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU."
            )
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU."
            )
        assert activation == "silu", f"{activation} is not supported for TPU."
        assert routed_scaling_factor == 1.0, (
            f"routed_scaling_factor {routed_scaling_factor} is not supported for TPU."
        )
        if (
            enable_eplb is not False
            or expert_load_view is not None
            or logical_to_physical_map is not None
            or logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for TPU.")
        return fused_moe_pallas(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk=top_k,
            gating_output=router_logits,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            renormalize=renormalize,
        )

    if current_platform.is_tpu():
        forward_native = forward_tpu
    elif current_platform.is_cpu():
        forward_native = forward_cpu
    elif current_platform.is_xpu():
        forward_native = forward_xpu
    else:
        forward_native = forward_cuda
