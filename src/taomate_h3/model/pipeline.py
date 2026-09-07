# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Native MiniMax H3 text-to-audio-video inference pipeline."""

from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from taomate_h3.config import DIRECT_W8A8_POLICY
from taomate_h3.distributed import ParallelContext

from ..denoise_schedule import DISTILLED_STATE_INDICES, select_time_shift_sigmas
from .components import partition_root
from .denoise import MiniMaxH3DenoiseBranch
from .dit import MiniMaxH3DiT
from .geometry import MiniMaxH3Geometry, direct_5s_geometry
from .output import (
    H3LatentDecoder,
    save_h3_video_mp4,
    save_h3_video_mp4_from_rgb24_shards,
    write_h3_rgb24_shard,
)
from .packed_sequence import minimax_h3_packed_sequence
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .presentation import minimax_h3_text_only_ids
from .text_encoder import MiniMaxH3TextEncoder, load_minimax_h3_text_weights
from .weight_loading import load_minimax_h3_dit_weights


@dataclass
class H3NativeGenerationResult:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor


class MiniMaxH3NativePipeline:
    """One positive CFG-distilled T2VA branch over TP/Ulysses groups."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        device: torch.device,
        parallel_context: ParallelContext,
        rank: int,
        world_size: int,
        transformer_load_w8a8_policy: str | None = DIRECT_W8A8_POLICY,
    ) -> None:
        self.partition_root = partition_root(Path(model_root).resolve(strict=True))
        self.device = device
        self.parallel_context = parallel_context
        self.rank = rank
        self.world_size = world_size
        self.transformer_load_w8a8_policy = transformer_load_w8a8_policy
        self.tokenizer = self._load_tokenizer()

        self.text_encoder = MiniMaxH3TextEncoder.allocate(
            self.partition_root / "text_encoder",
            device=self.device,
            context=self.parallel_context,
        )
        load_minimax_h3_text_weights(self.text_encoder, self.partition_root / "text_encoder")
        self.transformer = MiniMaxH3DiT.allocate(
            self.device,
            parallel_context=self.parallel_context,
            load_time_w8a8_policy=self.transformer_load_w8a8_policy,
        )
        self.dit_weight_receipt = load_minimax_h3_dit_weights(
            self.transformer,
            self.partition_root / "transformer",
        )
        self.text_encoder.eval()
        self.transformer.eval()

        self.output_decoder: H3LatentDecoder | None = None
        self.last_video_decode_receipt: dict[str, Any] | None = None

    def _load_tokenizer(self) -> Any:
        transformers = __import__("transformers", fromlist=["AutoTokenizer"])
        return transformers.AutoTokenizer.from_pretrained(
            str(self.partition_root / "tokenizer"), local_files_only=True
        )

    def _encode_text(self, *, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one text-only T2VA prompt."""

        ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
        tags = torch.ones(int(ids.shape[0]), dtype=torch.long)
        return self.text_encoder.encode_ids(ids), tags

    def _offload_text(self) -> None:
        self.text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _initial_video_noise(
        geometry: MiniMaxH3Geometry,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        raw_video = torch.randn(
            1,
            24,
            geometry.video_latent_t,
            geometry.video_latent_h,
            geometry.video_latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        return minimax_h3_patchify_video_latent(raw_video, patch_size=(1, 2, 2))

    @staticmethod
    def _initial_audio_noise(
        geometry: MiniMaxH3Geometry,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(
            geometry.audio_latent_t * 2,
            32,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

    def _build_denoise_branch(
        self,
        *,
        geometry: MiniMaxH3Geometry,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
    ) -> MiniMaxH3DenoiseBranch:
        packed = minimax_h3_packed_sequence(
            text_len=int(text_embeddings.shape[0]),
            latent_t=geometry.video_latent_t,
            latent_h=geometry.video_latent_h,
            latent_w=geometry.video_latent_w,
            audio_t=geometry.audio_latent_t,
        )
        token_tags = packed["token_tags"]
        token_tags[packed["text_pos"].view(-1)] = text_tags.view(-1)
        positive = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=token_tags,
            device=self.device,
            parallel_context=self.parallel_context,
        )
        refined = self.transformer.refine_prompt_embeds(
            positive.static_kwargs["prompt_embeds"],
            positive.refiner_cu_seqlens,
            device=self.device,
        )
        positive.static_kwargs["prompt_embeds"] = refined
        positive.static_kwargs["rope_cache"] = self.transformer.build_rope_cache(
            positive.img_position_ids, device=self.device
        )
        return positive

    def _collective_decode_error(self, error: str | None) -> str | None:
        errors: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(errors, error)
        return next((str(item) for item in errors if item is not None), None)

    def _load_output_decoder(self) -> H3LatentDecoder:
        if self.output_decoder is None:
            self.output_decoder = H3LatentDecoder(
                video_vae_root=self.partition_root / "video_vae",
                audio_vae_root=self.partition_root / "audio_vae",
                video_parallel_group=torch.distributed.group.WORLD,
                video_parallel_rank=self.rank,
                video_parallel_world_size=self.world_size,
                load_audio=self.rank == 0,
            )
            self.output_decoder.enable_video_vae_batched_tiles()
            self.output_decoder.enable_video_vae_exact_fusions()
            self.output_decoder.enable_video_vae_temporal_window_dp(
                microbatch_size=2,
                gpu_rgb24_output=True,
            )
            self.output_decoder.enable_video_vae_keep_on_device_after_decode()
            self.output_decoder.enable_video_vae_cpu_temporal_output()
        return self.output_decoder

    def _decode_and_save_parallel(
        self,
        *,
        video_latents: torch.Tensor,
        output_path: Path,
        video_filter: str | None = None,
        published_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        """Collectively decode one video timeline and publish it on rank zero."""

        decoder: H3LatentDecoder | None = None
        local_error: str | None = None
        try:
            decoder = self._load_output_decoder()
        except Exception as exc:
            local_error = f"rank {self.rank} VAE initialization: {type(exc).__name__}: {exc}"
        decode_error = self._collective_decode_error(local_error)
        if decode_error is not None:
            raise RuntimeError(f"video decoder initialization failed: {decode_error}")
        assert decoder is not None

        frames: torch.Tensor | None = None
        local_error = None
        try:
            frames = decoder.decode_video(
                video_latents=video_latents,
                device=self.device,
            )
        except Exception as exc:
            local_error = f"rank {self.rank} video VAE: {type(exc).__name__}: {exc}"
        decode_error = self._collective_decode_error(local_error)
        if decode_error is not None:
            raise RuntimeError(f"video VAE decode failed: {decode_error}")
        assert frames is not None

        decode_receipt = decoder.last_video_decode_receipt
        if not isinstance(decode_receipt, dict):
            raise RuntimeError("temporal-window Video VAE returned no decode receipt")
        if not bool(decode_receipt.get("active", False)):
            saved: Path | None = None
            local_error = None
            if self.rank == 0:
                try:
                    saved = save_h3_video_mp4(
                        output_path,
                        frames=frames,
                        video_filter=video_filter,
                        published_frames=published_frames,
                        crf=video_crf,
                        verify_decode=verify_video_decode,
                    )
                    self.last_video_decode_receipt = decode_receipt
                except Exception as exc:
                    local_error = f"rank 0 video publication: {type(exc).__name__}: {exc}"
            del frames
            payload: list[Any] = [
                {
                    "error": local_error,
                    "path": str(saved) if saved else None,
                    "receipt": self.last_video_decode_receipt,
                }
                if self.rank == 0
                else None
            ]
            torch.distributed.broadcast_object_list(payload, src=0)
            if payload[0]["error"] is not None:
                raise RuntimeError(f"video publication failed: {payload[0]['error']}")
            self.last_video_decode_receipt = payload[0]["receipt"]
            return Path(payload[0]["path"])

        temporary_payload: list[Any] = [None]
        if self.rank == 0:
            try:
                temporary_payload[0] = {
                    "path": tempfile.mkdtemp(
                        prefix="taomate-h3-vae-",
                        dir="/dev/shm",
                    ),
                    "error": None,
                }
            except Exception as exc:
                temporary_payload[0] = {
                    "path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        torch.distributed.broadcast_object_list(temporary_payload, src=0)
        temporary_result = temporary_payload[0]
        if not isinstance(temporary_result, dict) or temporary_result.get("error"):
            detail = (
                temporary_result.get("error")
                if isinstance(temporary_result, dict)
                else "invalid shared Video VAE temporary directory result"
            )
            raise RuntimeError(f"Video VAE temporary directory failed: {detail}")
        temporary_root = Path(str(temporary_result["path"]))
        local_shard_path = temporary_root / f"rank-{self.rank:02d}.rgb24"
        shard: dict[str, Any] | None = None
        local_error = None
        try:
            shard = write_h3_rgb24_shard(
                local_shard_path,
                frames=frames,
                frame_start=int(decode_receipt["retained_frame_start"]),
            )
            shard["rank"] = self.rank
            shard["decode"] = decode_receipt
        except Exception as exc:
            local_error = f"rank {self.rank} RGB24 shard: {type(exc).__name__}: {exc}"
        del frames

        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(
            gathered,
            {"error": local_error, "shard": shard},
        )
        shard_error = next((item["error"] for item in gathered if item["error"]), None)
        saved: Path | None = None
        publication_error: str | None = shard_error
        if self.rank == 0 and publication_error is None:
            try:
                shards = [item["shard"] for item in gathered]
                saved = save_h3_video_mp4_from_rgb24_shards(
                    output_path,
                    shards=shards,
                    video_filter=video_filter,
                    published_frames=published_frames,
                    crf=video_crf,
                    verify_decode=verify_video_decode,
                )
                self.last_video_decode_receipt = {
                    "mode": "temporal_window_dp",
                    "rank_receipts": [item["shard"]["decode"] for item in gathered],
                    "frame_count": sum(int(item["shard"]["frame_count"]) for item in gathered),
                    "balanced_cpu_output": True,
                    "gpu_rgb24_output": True,
                    "temporary_shards_removed": True,
                }
            except Exception as exc:
                publication_error = f"rank 0 video publication: {type(exc).__name__}: {exc}"
        payload: list[Any] = [
            {
                "error": publication_error,
                "path": str(saved) if saved else None,
                "receipt": self.last_video_decode_receipt,
            }
            if self.rank == 0
            else None
        ]
        torch.distributed.broadcast_object_list(payload, src=0)
        local_shard_path.unlink(missing_ok=True)
        torch.distributed.barrier()
        if self.rank == 0:
            temporary_root.rmdir()
        if payload[0]["error"] is not None:
            raise RuntimeError(f"video publication failed: {payload[0]['error']}")
        self.last_video_decode_receipt = payload[0]["receipt"]
        return Path(payload[0]["path"])

    @torch.no_grad()
    def generate(
        self,
        *,
        video_noise_seed: int,
        audio_noise_seed: int,
        width: int,
        height: int,
        precomputed_text: tuple[torch.Tensor, torch.Tensor],
        denoise_loop: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> H3NativeGenerationResult:
        """Generate one five-second native T2VA request as clean latents."""

        geometry = direct_5s_geometry(width=width, height=height)

        text_embeddings, text_tags = precomputed_text
        if (
            text_embeddings.ndim != 2
            or int(text_embeddings.shape[1]) != 5120
            or text_tags.ndim != 1
            or int(text_tags.shape[0]) != int(text_embeddings.shape[0])
        ):
            raise ValueError("precomputed H3 text feature shape differs")
        positive = self._build_denoise_branch(
            geometry=geometry,
            text_embeddings=text_embeddings,
            text_tags=text_tags,
        )

        # Preserve the official audio substream: advance past the discarded
        # full-video draw before sampling audio. Video uses its own request
        # substream so a prompt boundary cannot replay prior video noise.
        audio_generator = torch.Generator(device="cpu").manual_seed(audio_noise_seed)
        self._initial_video_noise(geometry, generator=audio_generator)
        initial_audio = self._initial_audio_noise(geometry, generator=audio_generator)
        video_generator = torch.Generator(device="cpu").manual_seed(video_noise_seed)
        initial_video = self._initial_video_noise(geometry, generator=video_generator)
        sigmas_video = select_time_shift_sigmas(
            num_steps=50,
            shift_scale=12.0,
            state_indices=DISTILLED_STATE_INDICES,
        )
        sigmas_audio = select_time_shift_sigmas(
            num_steps=50,
            shift_scale=3.0,
            state_indices=DISTILLED_STATE_INDICES,
        )
        video_rows, audio_rows = denoise_loop(
            model=self.transformer,
            positive=positive,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio,
            device=self.device,
            on_step=None,
        )
        video_latents = minimax_h3_unpatchify_video_tokens(
            video_rows,
            latent_shape=(
                geometry.video_latent_t,
                geometry.video_latent_h // 2,
                geometry.video_latent_w // 2,
                24,
            ),
            patch_size=(1, 2, 2),
        )
        audio_latents = minimax_h3_unpack_audio_tokens(
            audio_rows,
            audio_t=geometry.audio_latent_t * 2,
            audio_channel=2,
        )

        return H3NativeGenerationResult(
            video_latents=video_latents,
            audio_latents=audio_latents,
        )


__all__ = [
    "H3NativeGenerationResult",
    "MiniMaxH3NativePipeline",
]
