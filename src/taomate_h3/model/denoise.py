# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Native MiniMax H3 cfg-distilled audio-video denoise loop.

The released algorithm is implemented directly around TaoMate-H3's explicit
local ``ParallelContext``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from taomate_h3.distributed import ParallelContext

from .architecture import MINIMAX_H3_ADALN_MODALITY_NUM

_VIDEO_ROW_WIDTH = 96
_AUDIO_ROW_WIDTH = 32


@torch.no_grad()
def _minimax_h3_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    sigma_ratio: torch.Tensor,
    one_minus_sigma_ratio: torch.Tensor,
    denoised_scratch: torch.Tensor,
) -> None:
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


def _build_local_embedding_layout(
    *,
    seq_len: int,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if seq_len % world_size:
        raise ValueError(
            f"packed seq_len {seq_len} not divisible by Ulysses world size {world_size}"
        )
    local_seq_len = seq_len // world_size
    row_start = rank * local_seq_len
    row_stop = row_start + local_seq_len

    def local_ids(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = torch.nonzero(
            (pos >= row_start) & (pos < row_stop),
            as_tuple=False,
        ).view(-1)
        return source_ids.to(device), pos.index_select(0, source_ids).to(device)

    text_source_start = min(row_start, int(text_pos.shape[0]))
    text_source_stop = min(row_stop, int(text_pos.shape[0]))
    _, img_global_ids = local_ids(img_pos)
    _, audio_global_ids = local_ids(audio_pos)
    return {
        "text_source_start": text_source_start,
        "text_source_stop": text_source_stop,
        "img_global_ids": img_global_ids,
        "img_row_ids": img_global_ids - row_start,
        "audio_global_ids": audio_global_ids,
        "audio_row_ids": audio_global_ids - row_start,
    }


class MiniMaxH3DenoiseBranch:
    """Static positive-branch packed state and step-local forward inputs."""

    def __init__(
        self,
        *,
        packed: dict[str, Any],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
        parallel_context: ParallelContext,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.audio_update_mask = packed.get(
            "audio_update_mask",
            torch.ones(self.audio_pos.shape[0], dtype=torch.bool),
        ).view(-1).to(torch.bool)
        if self.audio_update_mask.shape[0] != self.audio_pos.shape[0]:
            raise ValueError("audio update mask differs from packed audio rows")
        self.audio_target_start = int((~self.audio_update_mask).sum())
        if not torch.all(self.audio_update_mask[self.audio_target_start :]):
            raise ValueError("audio reference rows must form one frozen prefix")
        self.audio_target_slice = slice(self.audio_target_start, None)

        text_pos = packed["text_pos"].view(-1).to(torch.long)
        text_len = int(text_pos.shape[0])
        if int(text_embeddings.shape[0]) != text_len:
            raise ValueError(
                f"text_embeddings rows {list(text_embeddings.shape)} != packed text_len {text_len}"
            )
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(
                f"token_tags length {int(token_tags.view(-1).shape[0])} != seq_len {seq_len}"
            )
        cu = packed["cu_seqlens"].to(torch.int32)
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        self.img_target_seq_idx = self.img_pos_dev
        self.audio_target_seq_idx = self.audio_pos_dev[self.audio_update_mask_dev]
        self.audio_ref_seq_idx = self.audio_pos_dev[~self.audio_update_mask_dev]
        self.x_buffer = torch.zeros(
            1,
            seq_len,
            _VIDEO_ROW_WIDTH,
            dtype=torch.float32,
            device=device,
        )
        self.audio_x_buffer = torch.zeros(
            1,
            seq_len,
            _AUDIO_ROW_WIDTH,
            dtype=torch.float32,
            device=device,
        )
        self.parallel_context = parallel_context
        ulysses_world_size = parallel_context.ulysses_world_size
        ulysses_rank = parallel_context.ulysses_rank
        if seq_len % ulysses_world_size:
            raise ValueError(
                f"packed seq_len {seq_len} not divisible by Ulysses world size {ulysses_world_size}"
            )
        token_tags_host = token_tags.view(-1).to(dtype=torch.long)
        # Native streaming attention needs the complete tag vector before the
        # Ulysses-local slice is moved to the device.
        local_seq_len = seq_len // ulysses_world_size
        local_row_start = ulysses_rank * local_seq_len
        local_row_stop = local_row_start + local_seq_len
        self.local_row_slice = slice(local_row_start, local_row_stop)
        self.block_token_tags = (
            token_tags_host[local_row_start:local_row_stop].clamp(min=0).to(device)
        )
        self.img_position_ids = packed["img_position_ids"][None].to(
            device=device,
            dtype=torch.float32,
        )
        self.static_kwargs: dict[str, Any] = {
            "block_token_tags": self.block_token_tags,
            "prompt_embeds": text_embeddings.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_target_seq_idx},
            "local_embedding_layout": _build_local_embedding_layout(
                seq_len=seq_len,
                text_pos=text_pos,
                img_pos=self.img_pos,
                audio_pos=self.audio_pos,
                world_size=ulysses_world_size,
                rank=ulysses_rank,
                device=device,
            ),
            "packed_seq_params": {
                "cu_seqlens_q_host": tuple(int(value) for value in cu.tolist()),
                "max_seqlen_q": int(cu[1]),
            },
        }
        self.refiner_cu_seqlens = torch.tensor(
            [0, text_len, text_len], dtype=torch.int32, device=device
        )

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        step_timesteps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, Any]:
        x = self.x_buffer
        audio_x = self.audio_x_buffer
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        unique_timesteps, inverse_indices, block_combined_indices = step_timesteps
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "block_combined_indices": block_combined_indices,
        }

    def _expand_step_timesteps(
        self,
        *,
        t_video: float,
        t_audio: float,
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor],
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Text, padding and video rows use the video clock; all audio rows use
        # the audio clock.  Keeping torch.unique here preserves the reference
        # float32 ordering and inverse-index construction exactly.
        candidates = [float(t_video), float(t_audio)]
        audio_ref_slot: int | None = None
        if self.audio_ref_seq_idx.numel() > 0:
            audio_ref_slot = len(candidates)
            candidates.append(1.0)
        unique_cpu, slot_to_unique = torch.unique(
            torch.tensor(candidates, dtype=torch.float32),
            sorted=True,
            return_inverse=True,
        )
        device = self.img_pos_dev.device
        base_index = int(slot_to_unique[0])
        pattern = tuple(slot_to_unique.tolist())
        inverse_indices = inverse_indices_by_pattern.get(pattern)
        if inverse_indices is None:
            inverse_indices = torch.full(
                (self.seq_len,), base_index, dtype=torch.long, device=device
            )
            inverse_indices.index_fill_(0, self.audio_target_seq_idx, int(slot_to_unique[1]))
            if audio_ref_slot is not None:
                inverse_indices.index_fill_(
                    0,
                    self.audio_ref_seq_idx,
                    int(slot_to_unique[audio_ref_slot]),
                )
            inverse_indices_by_pattern[pattern] = inverse_indices
        block_combined = block_combined_by_pattern.get(pattern)
        if block_combined is None:
            block_combined = torch.add(
                self.block_token_tags,
                inverse_indices[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            block_combined_by_pattern[pattern] = block_combined
        return unique_cpu.to(device), inverse_indices, block_combined

    def prepare_timestep_plan(
        self,
        *,
        video_timesteps: list[float],
        audio_timesteps: list[float],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("video/audio timestep plans must have equal length")
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        return [
            self._expand_step_timesteps(
                t_video=t_video,
                t_audio=t_audio,
                inverse_indices_by_pattern=inverse_indices_by_pattern,
                block_combined_by_pattern=block_combined_by_pattern,
            )
            for t_video, t_audio in zip(video_timesteps, audio_timesteps)
        ]


def minimax_h3_denoise_loop(
    *,
    model: Any,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    model_forward: (
        Callable[[Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run positive-only MiniMax H3 Euler-eta0 denoising for video and audio."""

    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32, copy=True)
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32, copy=True)
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout "
            f"rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout "
            f"rows {int(positive.audio_pos.shape[0])}"
        )
    video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
    video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
    audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)
    timestep_plan = positive.prepare_timestep_plan(
        video_timesteps=video_timesteps,
        audio_timesteps=audio_timesteps,
    )
    video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
    audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
    video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
    audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
    video_sigma_t = 1.0 - video_step_t
    audio_sigma_t = 1.0 - audio_step_t
    video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
    audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios
    video_denoised_scratch = torch.empty_like(video_rows)
    audio_target_slice = positive.audio_target_slice
    audio_denoised_scratch = torch.empty_like(audio_rows[audio_target_slice])
    for step in range(len(sigmas_video) - 1):
        forward_kwargs = positive.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            step_timesteps=timestep_plan[step],
        )
        if model_forward is None:
            velocity_video, velocity_audio = model(**forward_kwargs)
        else:
            velocity_video, velocity_audio = model_forward(model, forward_kwargs, step)
        velocity_video = velocity_video.float()
        velocity_audio = velocity_audio[audio_target_slice].float()
        _minimax_h3_update_target_rows_(
            video_rows,
            velocity_video,
            sigma_t=video_sigma_t[step],
            sigma_ratio=video_sigma_ratios[step],
            one_minus_sigma_ratio=video_one_minus_sigma_ratios[step],
            denoised_scratch=video_denoised_scratch,
        )
        _minimax_h3_update_target_rows_(
            audio_rows[audio_target_slice],
            velocity_audio,
            sigma_t=audio_sigma_t[step],
            sigma_ratio=audio_sigma_ratios[step],
            one_minus_sigma_ratio=audio_one_minus_sigma_ratios[step],
            denoised_scratch=audio_denoised_scratch,
        )
        if on_step is not None:
            on_step(step, video_rows, audio_rows)
    return video_rows, audio_rows


__all__ = [
    "MiniMaxH3DenoiseBranch",
    "minimax_h3_denoise_loop",
]
