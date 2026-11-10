"""Three-step Ref2AV generation from freshly encoded references and text."""
from __future__ import annotations

from pathlib import Path

import torch

from digitalhuman_h3.model.denoise import MiniMaxH3DenoiseBranch
from digitalhuman_h3.model.geometry import select_time_shift_sigmas
from digitalhuman_h3.model.layers import set_native_stream_attention_hook
from digitalhuman_h3.model.packed_tokens import (
    minimax_h3_unpatchify_video_tokens,
    minimax_h3_unpack_audio_tokens,
)

from .streaming import CleanHistoryAttention, chunk_slices, compact_layout


@torch.no_grad()
def generate(pipeline, payload, adapters, output: Path, seed: int):
    transformer, device = pipeline.transformer, pipeline.device
    geometry = payload["metadata"]["geometry"]
    pixels = (geometry["video_latent_h"] // 2) * (geometry["video_latent_w"] // 2)
    full_video = torch.empty((geometry["video_latent_t"] * pixels, 96), device=device, dtype=torch.float32)
    full_audio = torch.empty((2 * geometry["audio_latent_t"], 32), device=device, dtype=torch.float32)
    anchors = payload["tensors"]["visual_condition_anchor"].to(device)
    sigmas_video = select_time_shift_sigmas(num_steps=50, shift_scale=12)
    sigmas_audio = select_time_shift_sigmas(num_steps=50, shift_scale=3)
    chunks = list(chunk_slices(geometry))
    history = CleanHistoryAttention()

    def refine(current_payload):
        hidden = current_payload["tensors"]["text_hidden"]
        cu = torch.tensor([0, len(hidden), len(hidden)], dtype=torch.int32)
        return transformer.refine_prompt_embeds(hidden, cu, device=device)

    def forward(branch, video, audio, sigma_video, sigma_audio, packed, *, clean=False):
        timestep = branch.prepare_timestep_plan(
            video_timesteps=[1 - sigma_video], audio_timesteps=[1 - sigma_audio],
            imgvid_cond_noise_aug=0.999, audio_ref_cond_noise_aug=1.0,
        )[0]
        kwargs = branch.forward_kwargs(
            video_rows=torch.cat((anchors, video)), audio_rows=audio, step_timesteps=timestep)
        with history.execution(packed, clean=clean, device=device):
            return transformer(**kwargs)

    set_native_stream_attention_hook(history)
    try:
        for index, (video_slice, audio_ids) in enumerate(chunks):
            with adapters.use_chunk(index):
                hidden = adapters.refined_text(payload, index, refine)
                packed = compact_layout(payload, video_slice, audio_ids)
                audio_ids = audio_ids.to(device)
                branch = MiniMaxH3DenoiseBranch(
                    packed=packed, text_embeddings=hidden, token_tags=packed["token_tags"],
                    device=device, parallel_context=pipeline.parallel_context)
                branch.static_kwargs["refined_prompt_embeds_length"] = len(hidden)
                branch.static_kwargs["rope_cache"] = transformer.build_rope_cache(
                    branch.static_kwargs["img_position_ids"], device=device)
                rng = torch.Generator(device=device).manual_seed(seed + index * 97)
                video = torch.randn(full_video[video_slice].shape, device=device, dtype=torch.float32, generator=rng)
                audio = torch.randn((len(audio_ids), 32), device=device, dtype=torch.float32, generator=rng)
                for left, right in ((0, 16), (16, 33), (33, 49)):
                    velocity = forward(branch, video, audio, sigmas_video[left], sigmas_audio[left], packed)
                    video = video + (sigmas_video[left] - sigmas_video[right]) * velocity[0]
                    audio = audio + (sigmas_audio[left] - sigmas_audio[right]) * velocity[1][branch.audio_target_slice]
                full_video[video_slice] = video
                full_audio[audio_ids] = audio
                if index + 1 < len(chunks):
                    forward(branch, video, audio, 0.0, 0.0, packed, clean=True)
            if pipeline.rank == 0:
                print(f"Generated chunk {index + 1}/{len(chunks)}", flush=True)
    finally:
        set_native_stream_attention_hook(None)
        history.clear()

    video_latents = minimax_h3_unpatchify_video_tokens(
        full_video, latent_shape=(geometry["video_latent_t"], geometry["video_latent_h"] // 2,
                                 geometry["video_latent_w"] // 2, 24), patch_size=(1, 2, 2))
    if pipeline.rank == 0:
        audio_latents = minimax_h3_unpack_audio_tokens(
            full_audio, audio_t=geometry["audio_latent_t"] * 2, audio_channel=2)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"video": video_latents.cpu(), "audio": audio_latents.cpu(),
                    "fps": geometry["fps"]}, output / "latents.pt")
        print("Saved audio-video latents", flush=True)
    torch.distributed.barrier()
