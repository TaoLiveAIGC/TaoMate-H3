# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Dependency-free MiniMax H3 architecture contract.

Values are ported from the pinned MiniMax H3 configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64
MINIMAX_H3_ADALN_MODALITY_NUM = 3


@dataclass(frozen=True)
class MiniMaxH3Architecture:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    video_latent_channels: int = 24
    audio_latent_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.attention_head_dim != 128:
            raise ValueError("MiniMax H3 attention head dimension must be 128")
        if self.patch_size != (1, 2, 2):
            raise ValueError("MiniMax H3 video patch size must be (1, 2, 2)")

    @property
    def video_row_width(self) -> int:
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        return self.video_latent_channels * patch_volume


__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3Architecture",
]
