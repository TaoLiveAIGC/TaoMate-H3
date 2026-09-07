# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Native MiniMax H3 T2VA packed-sequence builders.

The row layout and fp64 position-grid math are kept byte-for-byte compatible
with the pinned MiniMax H3 reference behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .architecture import MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
) -> dict[str, Any]:
    """Build one positive-branch T2VA layout: text/audio/video/pad."""

    if min(text_len, latent_t, latent_h, latent_w, audio_t, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 packed-sequence dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")
    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + audio_rows + video_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )

    text_sl = slice(0, text_len)
    audio_sl = slice(text_len, text_len + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    img_pos = torch.arange(video_sl.start, video_sl.stop)
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    text_pos = torch.arange(0, text_len)

    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    t_grid = _video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    video_grid = grid[video_sl].view(latent_t, frame_rows, 3)
    video_grid[:, :, 0] = t_grid[:, None]
    video_grid[:, :, 1:] = frame[None]
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    grid[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    grid[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    grid[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_sl] = 2
    token_tags[img_pos] = 0
    cu_seqlens = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": cu_seqlens,
    }


def minimax_h3_audio_only_packed_sequence(
    *,
    text_len: int,
    audio_t: int,
    latent_h: int,
    latent_w: int,
    audio_channel: int = 2,
) -> dict[str, Any]:
    """Build the Base teacher's text/audio sequence without video tokens."""

    if min(text_len, audio_t, latent_h, latent_w, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 audio-only dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")

    audio_rows = audio_t * audio_channel
    used = text_len + audio_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )
    text_sl = slice(0, text_len)
    audio_sl = slice(text_len, text_len + audio_rows)

    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    grid[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    sqrt_area = np.sqrt(latent_h * latent_w)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    grid[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    grid[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_sl] = 2
    return {
        "seq_len": seq_len,
        "img_pos": torch.empty(0, dtype=torch.long),
        "audio_pos": torch.arange(audio_sl.start, audio_sl.stop),
        "text_pos": torch.arange(text_len),
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def minimax_h3_audio_only_frozen_prefix_packed_sequence(
    *,
    text_len: int,
    ref_audio_t: int,
    audio_t: int,
    latent_h: int,
    latent_w: int,
    reference_time_start: int,
    target_time_start: int,
    audio_channel: int = 2,
) -> dict[str, Any]:
    """Build an audio-only request with a clean, read-only reference tail."""

    if min(text_len, ref_audio_t, audio_t, latent_h, latent_w, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 audio-reference dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")

    ref_rows = ref_audio_t * audio_channel
    target_rows = audio_t * audio_channel
    used = text_len + ref_rows + target_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )
    text_sl = slice(0, text_len)
    ref_sl = slice(text_sl.stop, text_sl.stop + ref_rows)
    target_sl = slice(ref_sl.stop, ref_sl.stop + target_rows)

    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    grid[ref_sl, 0] = (
        float(reference_time_start) + torch.arange(ref_audio_t, dtype=torch.float64)
    ).repeat(audio_channel)
    grid[target_sl, 0] = (
        float(target_time_start) + torch.arange(audio_t, dtype=torch.float64)
    ).repeat(audio_channel)

    sqrt_area = np.sqrt(latent_h * latent_w)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    for audio_sl, temporal_rows in ((ref_sl, ref_audio_t), (target_sl, audio_t)):
        grid[audio_sl.start : audio_sl.start + temporal_rows, 2] = float(w_grid[0])
        grid[audio_sl.start + temporal_rows : audio_sl.stop, 2] = float(w_grid[-1])

    audio_pos = torch.arange(ref_sl.start, target_sl.stop)
    audio_update_mask = torch.zeros(ref_rows + target_rows, dtype=torch.bool)
    audio_update_mask[ref_rows:] = True
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_pos] = 2
    return {
        "seq_len": seq_len,
        "img_pos": torch.empty(0, dtype=torch.long),
        "audio_pos": audio_pos,
        "audio_update_mask": audio_update_mask,
        "text_pos": torch.arange(text_len),
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


__all__ = [
    "minimax_h3_audio_only_frozen_prefix_packed_sequence",
    "minimax_h3_audio_only_packed_sequence",
    "minimax_h3_packed_sequence",
]
