# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Bit-exact inference fusions for the bundled MiniMax H3 Video VAE."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch
import torch.nn as nn

from .fused_kernels import (
    h3_video_vae_exact_fusions_supported,
    try_h3_video_vae_scaled_residual_exact,
    try_h3_video_vae_swiglu_exact,
)


@dataclass(frozen=True)
class H3VideoVAEExactOperators:
    swiglu: Callable[[torch.Tensor], torch.Tensor | None]
    scaled_residual: Callable[..., torch.Tensor | None]


def resolve_h3_video_vae_exact_operators(
    device: torch.device,
) -> H3VideoVAEExactOperators | None:
    if not h3_video_vae_exact_fusions_supported(device):
        return None
    return H3VideoVAEExactOperators(
        swiglu=try_h3_video_vae_swiglu_exact,
        scaled_residual=try_h3_video_vae_scaled_residual_exact,
    )


def _optimized_feed_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return type(self).forward(self, hidden_states)
    projected = self.w1(hidden_states)
    activated = self._h3_exact_swiglu(projected)
    if activated is None:
        gate, value = projected.chunk(2, dim=-1)
        activated = self.act_fn(gate) * value
    return self.w2(activated)


def _optimized_transformer_block(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, Any] | None = None,
) -> torch.Tensor:
    pack_info = {} if pack_info is None else pack_info
    if torch.compiler.is_compiling() or hidden_states.dtype != torch.float32:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    normalized = self.norm1(hidden_states.float()).to(hidden_states.dtype)
    attention_output = self.attn(normalized, rotary_pos_emb, pack_info)
    updated = self._h3_exact_scaled_residual(
        hidden_states,
        attention_output,
        self.scale1,
    )
    hidden_states = hidden_states + attention_output * self.scale1 if updated is None else updated

    normalized = self.norm2(hidden_states.float()).to(hidden_states.dtype)
    feed_forward_output = self.ff(normalized)
    updated = self._h3_exact_scaled_residual(
        hidden_states,
        feed_forward_output,
        self.scale2,
    )
    return hidden_states + feed_forward_output * self.scale2 if updated is None else updated


def _validated_decoder_blocks(decoder: nn.Module) -> tuple[nn.Module, ...] | None:
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return None
    for block in blocks:
        feed_forward = getattr(block, "ff", None)
        linears = (
            getattr(feed_forward, "w1", None),
            getattr(feed_forward, "w2", None),
        )
        if (
            not all(isinstance(linear, nn.Linear) for linear in linears)
            or not isinstance(getattr(block, "norm1", None), nn.RMSNorm)
            or not isinstance(getattr(block, "norm2", None), nn.RMSNorm)
            or not getattr(block, "use_scale", False)
            or not isinstance(getattr(block, "scale1", None), torch.Tensor)
            or not isinstance(getattr(block, "scale2", None), torch.Tensor)
            or not getattr(feed_forward, "use_gated", False)
            or not isinstance(getattr(feed_forward, "act_fn", None), nn.SiLU)
            or not hasattr(feed_forward, "_compile_forward_enabled")
            or bool(feed_forward._compile_forward_enabled)
        ):
            return None

        w1, w2 = linears
        hidden_size = w1.in_features
        if (
            w1.out_features != 2 * w2.in_features
            or w2.out_features != hidden_size
            or block.norm1.normalized_shape != (hidden_size,)
            or block.norm2.normalized_shape != (hidden_size,)
            or block.scale1.shape != (hidden_size,)
            or block.scale2.shape != (hidden_size,)
        ):
            return None
    return tuple(blocks)


def install_h3_video_vae_exact_fusions_(
    decoder: nn.Module,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Install the exact SwiGLU and scaled-residual fusions once."""

    existing = getattr(decoder, "_h3_video_vae_exact_fusions_receipt", None)
    if isinstance(existing, dict):
        return existing
    operators = resolve_h3_video_vae_exact_operators(device)
    if operators is None:
        raise RuntimeError("H3 Video VAE exact fusions require Triton on a validated CUDA target")
    blocks = _validated_decoder_blocks(decoder)
    if blocks is None:
        raise RuntimeError("bundled H3 Video VAE decoder contract differs")

    for block in blocks:
        block.ff._h3_exact_swiglu = operators.swiglu
        block.ff.forward = MethodType(_optimized_feed_forward, block.ff)
        block._h3_exact_scaled_residual = operators.scaled_residual
        block.forward = MethodType(_optimized_transformer_block, block)

    receipt = {
        "backend": "triton",
        "bit_exact": True,
        "transformer_block_count": len(blocks),
        "operators": ["swiglu", "scaled_residual"],
    }
    decoder._h3_video_vae_exact_fusions_receipt = receipt
    return receipt


__all__ = [
    "H3VideoVAEExactOperators",
    "install_h3_video_vae_exact_fusions_",
    "resolve_h3_video_vae_exact_operators",
]
