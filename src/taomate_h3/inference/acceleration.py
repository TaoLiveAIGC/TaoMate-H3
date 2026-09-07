# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Apply TaoMate-H3's fixed, quality-validated inference acceleration profile."""

from __future__ import annotations

from typing import Any

from taomate_h3.config import DIRECT_W8A8_POLICY


def apply_direct_acceleration(transformer: Any) -> dict[str, Any]:
    from taomate_h3.distributed.linear import RowParallelLinear
    from taomate_h3.inference.fused_kernels import (
        enable_h3_fused_qkv_preprocess,
        enable_h3_sol_engine_fusions,
    )
    from taomate_h3.inference.lora_checkpoint import (
        enable_h3_fused_lora_row_reduce_,
        materialize_h3_lora_bf16_buffers_,
    )
    from taomate_h3.inference.w8a8 import (
        enable_h3_cutlass_w8a8_workspace_,
        quantize_h3_interior_qkv_fc1_,
    )
    from taomate_h3.model.layers import set_inference_packed_dense_flash_attn3

    if getattr(transformer, "_load_time_w8a8_policy", None) != DIRECT_W8A8_POLICY:
        raise RuntimeError("direct W8A8 weights were not prepared during DiT loading")
    weight_loading = getattr(transformer, "_native_weight_load_receipt", None)
    expected_w8a8_count = len(transformer._load_time_w8a8_linear_names)
    if (
        not isinstance(weight_loading, dict)
        or weight_loading.get("streamed_w8a8_linear_count") != expected_w8a8_count
    ):
        raise RuntimeError("direct W8A8 weights were not fully streamed from the checkpoint")
    enable_h3_sol_engine_fusions()
    enable_h3_fused_qkv_preprocess()
    lora_buffers = materialize_h3_lora_bf16_buffers_(transformer)
    fused_lora = enable_h3_fused_lora_row_reduce_(transformer)
    w8a8 = quantize_h3_interior_qkv_fc1_(
        transformer,
        policy=DIRECT_W8A8_POLICY,
    )
    if w8a8["post_load_conversion_linear_count"]:
        raise RuntimeError("direct W8A8 unexpectedly converted BF16 weights after loading")
    w8a8_workspace = enable_h3_cutlass_w8a8_workspace_(transformer)
    transformer.enable_inference_static_timestep_cache()
    set_inference_packed_dense_flash_attn3(True)

    row_modules = [
        module for module in transformer.modules() if isinstance(module, RowParallelLinear)
    ]
    for module in row_modules:
        module.inference_inplace_reduce_enabled = True

    return {
        "dit_weight_loading": weight_loading,
        "dit_w8a8": w8a8,
        "dit_w8a8_workspace": w8a8_workspace,
        "dit_lora_bf16_buffers": lora_buffers,
        "fused_lora_row_reduce": fused_lora,
        "sol_engine_kernels": True,
        "fused_qkv_preprocess": True,
        "static_timestep_adaln_cache": True,
        "packed_dense_fa3": True,
        "inplace_row_parallel_reduce_modules": len(row_modules),
    }


__all__ = ["apply_direct_acceleration"]
