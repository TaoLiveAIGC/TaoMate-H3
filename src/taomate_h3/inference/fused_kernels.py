# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Numerically equivalent Triton fusions used by direct H3 inference."""

from __future__ import annotations

from typing import Any

import torch

_ENABLED = False
_SOL_ENGINE_FUSIONS_ENABLED = False
_FUSED_QKV_PREPROCESS_ENABLED = False


def enable_h3_inference_fused_kernels() -> None:
    global _ENABLED
    if not _TRITON_AVAILABLE:
        raise RuntimeError("H3 fused inference kernels require Triton")
    _ENABLED = True


def enable_h3_sol_engine_fusions() -> None:
    """Enable the lossless MiniMax-H3 fusion pattern adapted from SOL Engine."""

    global _SOL_ENGINE_FUSIONS_ENABLED
    enable_h3_inference_fused_kernels()
    _SOL_ENGINE_FUSIONS_ENABLED = True


def enable_h3_fused_qkv_preprocess() -> None:
    """Enable inference-only in-place Q/K normalization and RoPE for merged QKV."""

    global _FUSED_QKV_PREPROCESS_ENABLED
    if not _TRITON_AVAILABLE:
        raise RuntimeError("H3 fused QKV preprocessing requires Triton")
    _FUSED_QKV_PREPROCESS_ENABLED = True


def h3_inference_fused_kernels_enabled() -> bool:
    return _ENABLED


def h3_sol_engine_fusions_enabled() -> bool:
    return _SOL_ENGINE_FUSIONS_ENABLED


def h3_fused_qkv_preprocess_enabled() -> bool:
    return _FUSED_QKV_PREPROCESS_ENABLED


try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _triton_libdevice
    except ImportError:
        _triton_libdevice = None

    _TRITON_AVAILABLE = True
    _TRITON_LIBDEVICE_AVAILABLE = _triton_libdevice is not None

    @triton.jit
    def _round_bf16(value):
        bits = value.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)

    @triton.jit
    def _video_vae_rgb24_exact_kernel(
        input_ptr,
        output_ptr,
        size,
        channels,
        temporal,
        height,
        width,
        input_stride_batch,
        input_stride_channel,
        input_stride_temporal,
        input_stride_height,
        input_stride_width,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        INPUT_KIND: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        width_index = offsets % width
        position = offsets // width
        height_index = position % height
        position = position // height
        temporal_index = position % temporal
        position = position // temporal
        channel = position % channels
        batch = position // channels
        input_offsets = (
            batch * input_stride_batch
            + channel * input_stride_channel
            + temporal_index * input_stride_temporal
            + height_index * input_stride_height
            + width_index * input_stride_width
        )
        mean = tl.where(
            channel == 0,
            mean0,
            tl.where(channel == 1, mean1, mean2),
        )
        std = tl.where(
            channel == 0,
            std0,
            tl.where(channel == 1, std1, std2),
        )
        value = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        value = value - mean
        if INPUT_KIND == 1:
            value = value.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
        elif INPUT_KIND == 2:
            value = _round_bf16(value)
        value = value / std
        if INPUT_KIND == 1:
            value = value.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
        elif INPUT_KIND == 2:
            value = _round_bf16(value)
        value = tl.maximum(0.0, tl.minimum(1.0, value)) * 255.0
        value = _triton_libdevice.rint(value)
        tl.store(output_ptr + offsets, value.to(tl.uint8), mask=mask)

    @triton.jit
    def _modulate_kernel(
        value_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        output_ptr,
        columns,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        table_row = tl.load(index_ptr + row)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        value = tl.load(value_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + table_row * columns + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        shift = tl.load(shift_ptr + table_row * columns + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        output = _round_bf16(_round_bf16(value * _round_bf16(1.0 + scale)) + shift)
        tl.store(
            output_ptr + row * columns + offsets,
            output.to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _gate_add_kernel(
        residual_ptr,
        gate_ptr,
        branch_ptr,
        index_ptr,
        output_ptr,
        columns,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        table_row = tl.load(index_ptr + row)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        residual = tl.load(residual_ptr + row * columns + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        branch = tl.load(branch_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_row * columns + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        output = _round_bf16(residual + _round_bf16(gate * branch))
        tl.store(
            output_ptr + row * columns + offsets,
            output.to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _rope_kernel(
        input_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        width,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < width
        channel = offsets % head_dim
        head_start = offsets - channel
        half = rotary_dim // 2
        rotating = channel < rotary_dim
        low = channel < half
        value = tl.load(input_ptr + row * width + offsets, mask=mask, other=0.0).to(tl.float32)
        partner = head_start + tl.where(low, channel + half, channel - half)
        paired = tl.load(
            input_ptr + row * width + partner,
            mask=mask & rotating,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(low, -paired, paired)
        cos = _round_bf16(
            tl.load(
                cos_ptr + row * rotary_dim + channel,
                mask=mask & rotating,
                other=0.0,
            ).to(tl.float32)
        )
        sin = _round_bf16(
            tl.load(
                sin_ptr + row * rotary_dim + channel,
                mask=mask & rotating,
                other=0.0,
            ).to(tl.float32)
        )
        output = _round_bf16(_round_bf16(value * cos) + _round_bf16(rotated * sin))
        tl.store(
            output_ptr + row * width + offsets,
            tl.where(rotating, output, value).to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _swiglu_kernel(
        input_ptr,
        output_ptr,
        half,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < half
        gate = tl.load(input_ptr + row * 2 * half + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(input_ptr + row * 2 * half + half + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        activated = _round_bf16(gate * tl.sigmoid(gate))
        output = _round_bf16(activated * up)
        tl.store(
            output_ptr + row * half + offsets,
            output.to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _rms_norm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        columns,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        value = tl.load(
            input_ptr + row * columns + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(value * value, axis=0) / columns
        output = value * tl.rsqrt(variance + eps) * weight
        tl.store(
            output_ptr + row * columns + offsets,
            output.to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _rms_norm_modulate_kernel(
        value_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        output_ptr,
        columns,
        table_stride,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        value = tl.load(
            value_ptr + row * columns + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(value * value, axis=0) / columns
        normalized = _round_bf16(value * tl.rsqrt(variance + eps) * weight)
        table_row = tl.load(index_ptr + row)
        table_offsets = table_row * table_stride + offsets
        scale = tl.load(scale_ptr + table_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offsets, mask=mask, other=0.0).to(tl.float32)
        output = _round_bf16(_round_bf16(normalized * _round_bf16(1.0 + scale)) + shift)
        tl.store(
            output_ptr + row * columns + offsets,
            output.to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _residual_gate_rms_norm_modulate_kernel(
        residual_ptr,
        branch_ptr,
        gate_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        hidden_ptr,
        output_ptr,
        columns,
        table_stride,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        row_offsets = row * columns + offsets
        table_row = tl.load(index_ptr + row)
        table_offsets = table_row * table_stride + offsets
        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = _round_bf16(residual + _round_bf16(gate * branch))
        tl.store(hidden_ptr + row_offsets, hidden.to(tl.bfloat16), mask=mask)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(hidden * hidden, axis=0) / columns
        normalized = _round_bf16(hidden * tl.rsqrt(variance + eps) * weight)
        scale = tl.load(scale_ptr + table_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offsets, mask=mask, other=0.0).to(tl.float32)
        output = _round_bf16(_round_bf16(normalized * _round_bf16(1.0 + scale)) + shift)
        tl.store(output_ptr + row_offsets, output.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _qk_norm_rope_kernel(
        value_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps,
        BLOCK: tl.constexpr,
    ):
        token_head = tl.program_id(0)
        token = token_head // heads
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        base = token_head * head_dim
        value = tl.load(value_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(value * value, axis=0) / head_dim
        normalized = _round_bf16(value * tl.rsqrt(variance + eps) * weight)
        half = rotary_dim // 2
        rotating = offsets < rotary_dim
        low = offsets < half
        partner = tl.where(low, offsets + half, offsets - half)
        paired_value = tl.load(
            value_ptr + base + partner,
            mask=rotating,
            other=0.0,
        ).to(tl.float32)
        paired_weight = tl.load(
            weight_ptr + partner,
            mask=rotating,
            other=0.0,
        ).to(tl.float32)
        paired = _round_bf16(paired_value * tl.rsqrt(variance + eps) * paired_weight)
        rotated = tl.where(low, -paired, paired)
        cos = _round_bf16(
            tl.load(
                cos_ptr + token * rotary_dim + offsets,
                mask=rotating,
                other=1.0,
            ).to(tl.float32)
        )
        sin = _round_bf16(
            tl.load(
                sin_ptr + token * rotary_dim + offsets,
                mask=rotating,
                other=0.0,
            ).to(tl.float32)
        )
        rope = _round_bf16(_round_bf16(normalized * cos) + _round_bf16(rotated * sin))
        output = tl.where(rotating, rope, normalized)
        tl.store(output_ptr + base + offsets, output.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _merged_qkv_preprocess_kernel(
        qkv_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_sin_ptr,
        positions_ptr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        cache_stride,
        eps,
        BLOCK: tl.constexpr,
    ):
        token_head = tl.program_id(0)
        qk = tl.program_id(1)
        token = token_head // heads
        head = token_head - token * heads
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        projection_stride = heads * head_dim
        base = token * 3 * projection_stride + qk * projection_stride + head * head_dim
        value = tl.load(qkv_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        q_weight = tl.load(q_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        k_weight = tl.load(k_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.where(qk == 0, q_weight, k_weight)
        variance = tl.sum(value * value, axis=0) / head_dim
        inverse_rms = tl.rsqrt(variance + eps)
        normalized = _round_bf16(value * inverse_rms * weight)

        half = rotary_dim // 2
        rotating = offsets < rotary_dim
        low = offsets < half
        partner = tl.where(low, offsets + half, offsets - half)
        paired_value = tl.load(
            qkv_ptr + base + partner,
            mask=rotating,
            other=0.0,
        ).to(tl.float32)
        q_paired_weight = tl.load(
            q_weight_ptr + partner,
            mask=rotating,
            other=0.0,
        ).to(tl.float32)
        k_paired_weight = tl.load(
            k_weight_ptr + partner,
            mask=rotating,
            other=0.0,
        ).to(tl.float32)
        paired_weight = tl.where(qk == 0, q_paired_weight, k_paired_weight)
        paired = _round_bf16(paired_value * inverse_rms * paired_weight)
        rotated = tl.where(low, -paired, paired)

        position = tl.load(positions_ptr + token)
        phase = offsets % half
        cache_base = position * cache_stride
        cos = _round_bf16(
            tl.load(
                cos_sin_ptr + cache_base + phase,
                mask=rotating,
                other=1.0,
            ).to(tl.float32)
        )
        sin = _round_bf16(
            tl.load(
                cos_sin_ptr + cache_base + half + phase,
                mask=rotating,
                other=0.0,
            ).to(tl.float32)
        )
        rope = _round_bf16(_round_bf16(normalized * cos) + _round_bf16(rotated * sin))
        output = tl.where(rotating, rope, normalized)
        tl.store(qkv_ptr + base + offsets, output.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _video_vae_swiglu_exact_kernel(
        input_ptr,
        output_ptr,
        input_stride_row,
        output_stride_row,
        half,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < half
        gate = tl.load(
            input_ptr + row * input_stride_row + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr + row * input_stride_row + half + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float16)
        # Match CUDA's expf-based SiLU instead of Triton's approximate tl.exp,
        # then retain eager's FP16 activation and multiply rounding points.
        exponential = _triton_libdevice.exp(-gate)
        one = tl.full(exponential.shape, 1.0, tl.float32)
        denominator = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[one, exponential],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        activated = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[gate, denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        ).to(tl.float16)
        output = tl.inline_asm_elementwise(
            "mul.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[activated, up],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
        tl.store(
            output_ptr + row * output_stride_row + columns,
            output,
            mask=mask,
        )

    @triton.jit
    def _video_vae_mul_rn_f32(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _video_vae_scaled_residual_exact_kernel(
        output_ptr,
        residual_ptr,
        branch_ptr,
        scale_ptr,
        residual_stride_row,
        branch_stride_row,
        hidden_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < hidden_size
        residual = tl.load(
            residual_ptr + row * residual_stride_row + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        branch = tl.load(
            branch_ptr + row * branch_stride_row + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        # Retain eager's multiplication rounding before the residual add.
        updated = residual + _video_vae_mul_rn_f32(branch, scale)
        tl.store(output_ptr + row * hidden_size + columns, updated, mask=mask)

except ImportError:
    _TRITON_AVAILABLE = False
    _TRITON_LIBDEVICE_AVAILABLE = False


def _launch_rows(kernel: Any, first: torch.Tensor, *rest: Any) -> torch.Tensor:
    rows, columns = first.shape
    output = torch.empty_like(first)
    kernel[(rows,)](
        first,
        *rest,
        output,
        columns,
        BLOCK=triton.next_power_of_2(columns),
        num_warps=8,
    )
    return output


def fused_modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    shape = value.shape
    flat = value.reshape(-1, shape[-1]).contiguous()
    output = _launch_rows(
        _modulate_kernel,
        flat,
        scale.contiguous(),
        shift.contiguous(),
        indices.reshape(-1).contiguous(),
    )
    return output.reshape(shape)


def fused_gate_add(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    shape = residual.shape
    flat = residual.reshape(-1, shape[-1]).contiguous()
    output = _launch_rows(
        _gate_add_kernel,
        flat,
        gate.contiguous(),
        branch.reshape(-1, shape[-1]).contiguous(),
        indices.reshape(-1).contiguous(),
    )
    return output.reshape(shape)


def fused_apply_rope(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    tokens, heads, head_dim = value.shape
    rotary_dim = cos.shape[-1]
    flat = value.reshape(tokens, heads * head_dim).contiguous()
    output = torch.empty_like(flat)
    block = 1024
    width = heads * head_dim
    _rope_kernel[(tokens, triton.cdiv(width, block))](
        flat,
        cos.contiguous(),
        sin.contiguous(),
        output,
        width,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        BLOCK=block,
        num_warps=4,
    )
    return output.reshape(tokens, heads, head_dim)


def fused_swiglu(value: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    half = shape[-1] // 2
    flat = value.reshape(-1, shape[-1]).contiguous()
    output = torch.empty((flat.shape[0], half), dtype=value.dtype, device=value.device)
    block = 1024
    _swiglu_kernel[(flat.shape[0], triton.cdiv(half, block))](
        flat,
        output,
        half,
        BLOCK=block,
        num_warps=4,
    )
    return output.reshape(*shape[:-1], half)


def fused_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    shape = value.shape
    flat = value.reshape(-1, shape[-1]).contiguous()
    output = torch.empty_like(flat)
    block = triton.next_power_of_2(shape[-1])
    _rms_norm_kernel[(flat.shape[0],)](
        flat,
        weight.contiguous(),
        output,
        shape[-1],
        eps,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return output.reshape(shape)


def fused_rms_norm_modulate(
    value: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    shape = value.shape
    flat = value.reshape(-1, shape[-1]).contiguous()
    index = indices.reshape(-1).contiguous()
    if int(index.numel()) != int(flat.shape[0]):
        raise ValueError("SOL RMSNorm/AdaLN indices do not cover every row")
    if scale.stride(-1) != 1 or shift.stride(-1) != 1:
        scale = scale.contiguous()
        shift = shift.contiguous()
    output = torch.empty_like(flat)
    block = triton.next_power_of_2(shape[-1])
    _rms_norm_modulate_kernel[(flat.shape[0],)](
        flat,
        weight.contiguous(),
        scale,
        shift,
        index,
        output,
        shape[-1],
        scale.stride(0),
        eps,
        BLOCK=block,
        num_warps=16 if block >= 8192 else (8 if block >= 2048 else 4),
    )
    return output.reshape(shape)


def fused_residual_gate_rms_norm_modulate(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = residual.shape
    columns = shape[-1]
    residual_flat = residual.reshape(-1, columns).contiguous()
    branch_flat = branch.reshape(-1, columns).contiguous()
    index = indices.reshape(-1).contiguous()
    if int(index.numel()) != int(residual_flat.shape[0]):
        raise ValueError("SOL residual/AdaLN indices do not cover every row")
    if any(table.stride(-1) != 1 for table in (gate, scale, shift)):
        gate, scale, shift = (table.contiguous() for table in (gate, scale, shift))
    hidden = torch.empty_like(residual_flat)
    output = torch.empty_like(residual_flat)
    block = triton.next_power_of_2(columns)
    _residual_gate_rms_norm_modulate_kernel[(residual_flat.shape[0],)](
        residual_flat,
        branch_flat,
        gate,
        weight.contiguous(),
        scale,
        shift,
        index,
        hidden,
        output,
        columns,
        gate.stride(0),
        eps,
        BLOCK=block,
        num_warps=16 if block >= 8192 else (8 if block >= 2048 else 4),
    )
    return hidden.reshape(shape), output.reshape(shape)


def fused_qk_norm_rope(
    value: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tokens, heads, head_dim = value.shape
    rotary_dim = int(cos.shape[-1])
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("SOL QKNorm/RoPE requires an even rotary span within a head")
    flat = value.reshape(-1, head_dim).contiguous()
    output = torch.empty_like(flat)
    block = triton.next_power_of_2(head_dim)
    _qk_norm_rope_kernel[(tokens * heads,)](
        flat,
        weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        output,
        heads=heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        BLOCK=block,
        num_warps=4,
    )
    return output.reshape(tokens, heads, head_dim)


def fused_qkv_qk_norm_rope_(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize and rotate Q/K in-place in a merged BF16 QKV projection.

    The arithmetic and explicit BF16 rounding match :func:`fused_qk_norm_rope`.
    V is not read or written.  Keeping the merged allocation intact lets the
    Ulysses path form its all-to-all send layout without an intermediate QKV
    stack.
    """

    if qkv.ndim != 2 or not qkv.is_contiguous():
        raise ValueError("fused QKV preprocessing requires contiguous [tokens, width]")
    if q_weight.shape != k_weight.shape or q_weight.ndim != 1:
        raise ValueError("fused QKV Q/K norm weights differ")
    if any(value.dtype != torch.bfloat16 for value in (qkv, q_weight, k_weight, cos_sin_cache)):
        raise ValueError("fused QKV preprocessing requires BF16 tensors")
    tokens = int(qkv.shape[0])
    head_dim = int(q_weight.numel())
    projection_width, remainder = divmod(int(qkv.shape[-1]), 3)
    heads, head_remainder = divmod(projection_width, head_dim)
    if remainder or head_remainder:
        raise ValueError("fused QKV width does not contain three complete projections")
    positions = positions.reshape(-1).contiguous()
    if int(positions.numel()) != tokens:
        raise ValueError("fused QKV RoPE positions do not cover every token")
    cache = cos_sin_cache.contiguous()
    rotary_dim = int(cache.shape[-1])
    if cache.ndim != 2 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("fused QKV RoPE cache span is invalid")
    block = triton.next_power_of_2(head_dim)
    _merged_qkv_preprocess_kernel[(tokens * heads, 2)](
        qkv,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cache,
        positions,
        heads=heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        cache_stride=cache.stride(0),
        eps=eps,
        BLOCK=block,
        num_warps=4,
    )
    return qkv


def h3_video_vae_exact_fusions_supported(device: torch.device) -> bool:
    """Whether the bit-exact Video VAE kernels are validated on ``device``."""

    if (
        not _TRITON_AVAILABLE
        or not _TRITON_LIBDEVICE_AVAILABLE
        or device.type != "cuda"
        or torch.version.hip is not None
    ):
        return False
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor in {90, 100, 103}


def _video_vae_exact_runtime_enabled(value: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and value.is_cuda
    )


def try_h3_video_vae_swiglu_exact(value: torch.Tensor) -> torch.Tensor | None:
    """Fuse the H3 VAE's FP16 SiLU and gated multiply exactly."""

    if not (
        _video_vae_exact_runtime_enabled(value)
        and value.dtype == torch.float16
        and value.ndim >= 1
        and value.shape[-1] > 0
        and value.shape[-1] % 2 == 0
        and value.is_contiguous()
    ):
        return None
    shape = value.shape
    flat = value.reshape(-1, shape[-1])
    half = shape[-1] // 2
    output = torch.empty((flat.shape[0], half), dtype=value.dtype, device=value.device)
    block = 1024
    _video_vae_swiglu_exact_kernel[(flat.shape[0], triton.cdiv(half, block))](
        flat,
        output,
        flat.stride(0),
        output.stride(0),
        half,
        BLOCK=block,
        num_warps=4,
    )
    return output.reshape(*shape[:-1], half)


def try_h3_video_vae_scaled_residual_exact(
    residual: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor | None:
    """Fuse one H3 VAE FP32 scaled residual update exactly."""

    if not (
        _video_vae_exact_runtime_enabled(residual)
        and branch.is_cuda
        and residual.device == branch.device == scale.device
        and residual.dtype == scale.dtype == torch.float32
        and branch.dtype == torch.float16
        and residual.shape == branch.shape
        and residual.ndim >= 1
        and residual.shape[-1] == 2048
        and scale.shape == residual.shape[-1:]
        and residual.numel() > 0
        and residual.is_contiguous()
        and branch.is_contiguous()
        and scale.is_contiguous()
    ):
        return None
    hidden_size = residual.shape[-1]
    residual_2d = residual.reshape(-1, hidden_size)
    branch_2d = branch.reshape(-1, hidden_size)
    output = torch.empty_like(residual_2d)
    _video_vae_scaled_residual_exact_kernel[(residual_2d.shape[0],)](
        output,
        residual_2d,
        branch_2d,
        scale,
        residual_2d.stride(0),
        branch_2d.stride(0),
        hidden_size=hidden_size,
        BLOCK=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return output.reshape_as(residual)


def try_h3_video_vae_rgb24_exact(
    value: torch.Tensor,
    *,
    mean_values: tuple[float, float, float],
    std_values: tuple[float, float, float],
) -> torch.Tensor | None:
    """Fuse H3 Video VAE denormalize, clamp, and RGB24 quantization."""

    if not (
        _video_vae_exact_runtime_enabled(value)
        and _TRITON_LIBDEVICE_AVAILABLE
        and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and value.ndim == 5
        and int(value.shape[1]) == 3
        and value.numel() > 0
        and len(mean_values) == len(std_values) == 3
    ):
        return None
    output = torch.empty(value.shape, dtype=torch.uint8, device=value.device)
    rounded_mean = tuple(
        float(torch.tensor(item, dtype=value.dtype).item()) for item in mean_values
    )
    rounded_std = tuple(float(torch.tensor(item, dtype=value.dtype).item()) for item in std_values)
    input_kind = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }[value.dtype]
    size = int(value.numel())
    _video_vae_rgb24_exact_kernel[(triton.cdiv(size, 4096),)](
        value,
        output,
        size,
        int(value.shape[1]),
        int(value.shape[2]),
        int(value.shape[3]),
        int(value.shape[4]),
        *value.stride(),
        *rounded_mean,
        *rounded_std,
        INPUT_KIND=input_kind,
        BLOCK=4096,
        num_warps=8,
    )
    return output


__all__ = [
    "enable_h3_fused_qkv_preprocess",
    "enable_h3_inference_fused_kernels",
    "enable_h3_sol_engine_fusions",
    "fused_apply_rope",
    "fused_gate_add",
    "fused_modulate",
    "fused_rms_norm",
    "fused_qk_norm_rope",
    "fused_qkv_qk_norm_rope_",
    "fused_residual_gate_rms_norm_modulate",
    "fused_rms_norm_modulate",
    "fused_swiglu",
    "h3_fused_qkv_preprocess_enabled",
    "h3_inference_fused_kernels_enabled",
    "h3_sol_engine_fusions_enabled",
    "h3_video_vae_exact_fusions_supported",
    "try_h3_video_vae_scaled_residual_exact",
    "try_h3_video_vae_swiglu_exact",
    "try_h3_video_vae_rgb24_exact",
]
