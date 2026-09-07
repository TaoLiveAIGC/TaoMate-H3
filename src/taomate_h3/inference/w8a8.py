# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Selective CUTLASS W8A8 for the validated interior H3 projections."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import torch

INTERIOR_FC1_QKV_POLICY = "interior_fc1_qkv_v1"
_STREAM_LOAD_TEMPORARY_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=1)
def require_vllm_cutlass_w8a8() -> Any:
    try:
        backend = importlib.import_module("vllm._custom_ops")
    except ImportError as exc:
        raise RuntimeError("W8A8 requires vLLM CUDA custom operations") from exc
    for name in ("scaled_int8_quant", "cutlass_scaled_mm"):
        if not callable(getattr(backend, name, None)):
            raise RuntimeError(f"vLLM CUDA custom operations are missing {name}")
    return backend


def resolve_h3_cutlass_w8a8_policy(names: list[str], *, policy: str) -> dict[str, Any]:
    """Select the fixed interior QKV/FC1 projection set."""

    if policy != INTERIOR_FC1_QKV_POLICY:
        raise ValueError(f"unsupported H3 W8A8 policy: {policy}")
    candidates = [
        name
        for name in names
        if name.startswith("blocks.") and name.endswith((".attn.qkv_proj", ".mlp.fc1", ".mlp.fc2"))
    ]
    block_indices = sorted({int(name.split(".", 2)[1]) for name in candidates})
    if not block_indices:
        raise RuntimeError("H3 W8A8 found no DiT projections")
    protected_blocks = set((*block_indices[:2], *block_indices[-3:]))
    selected = [
        name
        for name in candidates
        if name.endswith((".attn.qkv_proj", ".mlp.fc1"))
        and int(name.split(".", 2)[1]) not in protected_blocks
    ]
    selected_set = set(selected)
    return {
        "quantized_linear_names": selected,
        "bf16_protected_linear_names": [name for name in candidates if name not in selected_set],
        "bf16_boundary_block_indices": sorted(protected_blocks),
    }


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.clamp(
        weight.abs().amax(dim=1, keepdim=True).float() / 127.0,
        min=1.0e-12,
    )
    quantized = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def prepare_cutlass_w8a8_stream_load_slot_(
    module: Any,
    *,
    name: str,
) -> None:
    """Replace one uninitialized BF16 parameter with its final W8 storage."""

    weight = module.weight
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"{name} expected an uninitialized BF16 weight")
    if weight.requires_grad is False:
        raise RuntimeError(f"{name} uninitialized BF16 weight is already frozen")
    if module.bias is not None:
        raise RuntimeError(f"{name} expected a bias-free H3 projection")
    module.weight = torch.nn.Parameter(
        torch.empty(weight.shape, dtype=torch.int8, device=weight.device),
        requires_grad=False,
    )
    module.register_buffer(
        "_cutlass_w8a8_weight_scale",
        torch.empty(
            (int(weight.shape[0]), 1),
            dtype=torch.float32,
            device=weight.device,
        ),
        persistent=False,
    )
    module._cutlass_w8a8_stream_load_pending = True
    module._cutlass_w8a8_stream_loaded = False
    module._cutlass_w8a8_source_weight_bytes = int(weight.numel() * weight.element_size())


def load_cutlass_w8a8_stream_weight_(
    module: Any,
    source_weight: torch.Tensor,
    *,
    name: str,
    maximum_temporary_bytes: int = _STREAM_LOAD_TEMPORARY_BYTES,
) -> None:
    """Quantize a CPU BF16 checkpoint tensor directly into final GPU W8 storage.

    Per-output-row weight scaling makes row-wise conversion exactly equivalent
    to converting the full local tensor. The bounded BF16 staging tensor avoids
    ever materializing the complete target matrix in BF16 on the GPU.
    """

    if getattr(module, "_cutlass_w8a8_stream_load_pending", None) is not True:
        raise RuntimeError(f"{name} has no pending CUTLASS W8A8 stream load")
    if module.weight.dtype != torch.int8 or source_weight.dtype != torch.bfloat16:
        raise TypeError(f"{name} streamed CUTLASS W8A8 dtype differs")
    if source_weight.ndim != 2 or tuple(source_weight.shape) != tuple(module.weight.shape):
        raise ValueError(f"{name} streamed CUTLASS W8A8 shape differs")
    if maximum_temporary_bytes <= 0:
        raise ValueError("maximum_temporary_bytes must be positive")
    bytes_per_row = int(source_weight.shape[1]) * int(source_weight.element_size())
    rows_per_chunk = max(1, maximum_temporary_bytes // bytes_per_row)
    for start in range(0, int(source_weight.shape[0]), rows_per_chunk):
        stop = min(start + rows_per_chunk, int(source_weight.shape[0]))
        temporary = source_weight[start:stop].to(
            device=module.weight.device,
            dtype=torch.bfloat16,
        )
        quantized, scale = _quantize_weight(temporary)
        module.weight[start:stop].copy_(quantized)
        module._cutlass_w8a8_weight_scale[start:stop].copy_(scale)
    module._cutlass_w8a8_stream_load_temporary_bytes_limit = int(maximum_temporary_bytes)
    module._cutlass_w8a8_stream_load_pending = False
    module._cutlass_w8a8_stream_loaded = True


class CutlassW8A8Workspace:
    """Grow-only per-stream buffers shared by serial H3 QKV/FC1 calls."""

    def __init__(self) -> None:
        namespace = getattr(getattr(torch, "ops", None), "_C", None)
        self._dynamic_quant = getattr(namespace, "dynamic_scaled_int8_quant", None)
        self._scaled_mm = getattr(namespace, "cutlass_scaled_mm", None)
        if not callable(self._dynamic_quant) or not callable(self._scaled_mm):
            raise RuntimeError("vLLM W8A8 out operators are unavailable")
        self._activation: dict[tuple[int | None, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._outputs: dict[tuple[int | None, int, int], torch.Tensor] = {}

    @staticmethod
    def _stream(inputs: torch.Tensor) -> tuple[int | None, int]:
        stream = torch.cuda.current_stream(device=inputs.device)
        return inputs.device.index, int(stream.cuda_stream)

    def linear(
        self,
        inputs: torch.Tensor,
        weight_t: torch.Tensor,
        weight_scale_t: torch.Tensor,
    ) -> torch.Tensor:
        rows, width = int(inputs.shape[0]), int(inputs.shape[1])
        device_index, stream = self._stream(inputs)
        activation_key = (device_index, stream, width)
        buffers = self._activation.get(activation_key)
        if buffers is None or int(buffers[0].shape[0]) < rows:
            buffers = (
                torch.empty((rows, width), device=inputs.device, dtype=torch.int8),
                torch.empty((rows, 1), device=inputs.device, dtype=torch.float32),
            )
            self._activation[activation_key] = buffers
        quantized, activation_scale = buffers[0][:rows], buffers[1][:rows]

        output_width = int(weight_t.shape[1])
        output_key = (device_index, stream, output_width)
        output = self._outputs.get(output_key)
        if output is None or int(output.shape[0]) < rows:
            output = torch.empty((rows, output_width), device=inputs.device, dtype=torch.bfloat16)
            self._outputs[output_key] = output
        output = output[:rows]
        self._dynamic_quant(quantized, inputs, activation_scale, None)
        self._scaled_mm(
            output,
            quantized,
            weight_t,
            activation_scale,
            weight_scale_t,
            None,
        )
        return output


def cutlass_w8a8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    workspace: CutlassW8A8Workspace | None,
    weight_t: torch.Tensor | None,
    weight_scale_t: torch.Tensor | None,
) -> torch.Tensor:
    if bias is not None:
        raise RuntimeError("H3 W8A8 supports bias-free linears only")
    shape = (*inputs.shape[:-1], int(weight.shape[0]))
    flattened = inputs.reshape(-1, int(inputs.shape[-1])).contiguous()
    if workspace is not None:
        output = workspace.linear(
            flattened,
            weight.t() if weight_t is None else weight_t,
            weight_scale.t() if weight_scale_t is None else weight_scale_t,
        )
        return output.reshape(shape)
    backend = require_vllm_cutlass_w8a8()
    quantized, activation_scale, _ = backend.scaled_int8_quant(flattened)
    output = backend.cutlass_scaled_mm(
        quantized,
        weight.t(),
        activation_scale,
        weight_scale.t(),
        torch.bfloat16,
    )
    return output.reshape(shape)


def quantize_h3_interior_qkv_fc1_(transformer: Any, *, policy: str) -> dict[str, Any]:
    """Validate and summarize streamed base W8A8 while leaving LoRA BF16."""

    require_vllm_cutlass_w8a8()
    from taomate_h3.distributed.linear import ColumnParallelLinear, RowParallelLinear

    candidates: list[tuple[str, Any]] = []
    for name, module in transformer.named_modules():
        if not name.startswith("blocks.") or not name.endswith(
            (".attn.qkv_proj", ".mlp.fc1", ".mlp.fc2")
        ):
            continue
        base = getattr(module, "base", module)
        if isinstance(base, (ColumnParallelLinear, RowParallelLinear)):
            candidates.append((name, base))
    selection = resolve_h3_cutlass_w8a8_policy([name for name, _ in candidates], policy=policy)
    selected = set(selection["quantized_linear_names"])
    selected_candidates = [(name, module) for name, module in candidates if name in selected]
    if not selected_candidates:
        raise RuntimeError("H3 W8A8 policy selected no linears")

    source_bytes = 0
    quantized_bytes = 0
    for name, module in selected_candidates:
        scale = getattr(module, "_cutlass_w8a8_weight_scale", None)
        if (
            scale is None
            or module.weight.dtype != torch.int8
            or getattr(module, "_cutlass_w8a8_stream_load_pending", None) is not False
            or getattr(module, "_cutlass_w8a8_stream_loaded", None) is not True
        ):
            raise RuntimeError(f"{name} was not loaded directly as CUTLASS W8A8")
        if module.bias is not None:
            raise RuntimeError(f"{name} is not a bias-free projection")
        source_bytes += int(module._cutlass_w8a8_source_weight_bytes)
        quantized_bytes += int(module.weight.numel() * module.weight.element_size())
    return {
        "policy": policy,
        "format": "int8_w8a8",
        "linear_count": len(selected_candidates),
        "stream_load_linear_count": len(selected_candidates),
        "post_load_conversion_linear_count": 0,
        "source_weight_bytes": source_bytes,
        "quantized_weight_bytes": quantized_bytes,
        **selection,
    }


def enable_h3_cutlass_w8a8_workspace_(transformer: Any) -> dict[str, Any]:
    require_vllm_cutlass_w8a8()
    selected = [
        module
        for module in transformer.modules()
        if getattr(module, "_cutlass_w8a8_weight_scale", None) is not None
    ]
    if not selected:
        raise RuntimeError("W8A8 workspace requires quantized H3 linears")
    workspace = CutlassW8A8Workspace()
    for module in selected:
        module._cutlass_w8a8_workspace = workspace
        module._cutlass_w8a8_weight_t = module.weight.t()
        module._cutlass_w8a8_weight_scale_t = module._cutlass_w8a8_weight_scale.t()
    return {
        "linear_count": len(selected),
        "workspace_count_per_rank": 1,
        "allocation_policy": "grow_only_per_cuda_stream_and_shape",
    }


__all__ = [
    "INTERIOR_FC1_QKV_POLICY",
    "cutlass_w8a8_linear",
    "enable_h3_cutlass_w8a8_workspace_",
    "load_cutlass_w8a8_stream_weight_",
    "prepare_cutlass_w8a8_stream_load_slot_",
    "quantize_h3_interior_qkv_fc1_",
    "require_vllm_cutlass_w8a8",
    "resolve_h3_cutlass_w8a8_policy",
]
