# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Distributed native H3 runtime used by the public direct entrypoint."""

from __future__ import annotations

import importlib.metadata
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ..config import DIRECT_POLICY

_TP_SIZE = 2
_SUPPORTED_ULYSSES_DEGREES = (2, 4)


class H3RuntimeError(RuntimeError):
    pass


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class H3Runtime:
    """MiniMax H3 FL2VA partition in T2VA mode over TP2 x Ulysses2/4."""

    def __init__(
        self,
        *,
        model_root: Path,
        adapter_root: Path,
        output_root: Path,
        ulysses_degree: int,
    ) -> None:
        self._worker_started_at = time.perf_counter()
        self._timings: dict[str, float] = {}
        if ulysses_degree not in _SUPPORTED_ULYSSES_DEGREES:
            raise ValueError("Ulysses degree must be 2 or 4")
        self.tp_size = _TP_SIZE
        self.ulysses_degree = ulysses_degree
        self.expected_world_size = self.tp_size * self.ulysses_degree
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        if self.world_size != self.expected_world_size:
            raise ValueError(
                f"torchrun world size must be {self.expected_world_size}, got {self.world_size}"
            )

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if not torch.cuda.is_available() or self.local_rank >= torch.cuda.device_count():
            raise H3RuntimeError(f"CUDA device for local rank {self.local_rank} is unavailable")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(hours=3)
            )

        self.parallel_context = self._build_parallel_context()

        self.model_root = model_root.expanduser().resolve(strict=True)
        self.output_root = output_root.expanduser().resolve()
        if not (self.model_root / "FL2VA").is_dir():
            raise H3RuntimeError("model root does not contain the FL2VA partition")
        if self.is_main_process:
            self.output_root.mkdir(parents=True, exist_ok=True)
        self.barrier()

        pipeline_error: str | None = None
        self.pipeline: Any | None = None
        self._synchronize()
        pipeline_started = time.perf_counter()
        try:
            from ..model.pipeline import MiniMaxH3NativePipeline

            self.pipeline = MiniMaxH3NativePipeline(
                model_root=self.model_root,
                device=self.device,
                parallel_context=self.parallel_context,
                rank=self.rank,
                world_size=self.world_size,
            )
        except Exception as exc:
            pipeline_error = _exception_text(exc)
        pipeline_error = self._collective_error("H3 pipeline load", pipeline_error)
        if pipeline_error:
            raise H3RuntimeError(pipeline_error)
        assert self.pipeline is not None
        self._synchronize()
        self._timings["model_load_seconds"] = time.perf_counter() - pipeline_started

        adapter_error: str | None = None
        adapter_receipt: Mapping[str, Any] | None = None
        self._synchronize()
        adapter_started = time.perf_counter()
        try:
            from ..inference.lora_checkpoint import apply_h3_lora_checkpoint

            adapter_receipt = apply_h3_lora_checkpoint(
                self.pipeline.transformer,
                adapter_root.expanduser().resolve(strict=True),
            )
        except Exception as exc:
            adapter_error = _exception_text(exc)
        adapter_error = self._collective_error("Stage3 LoRA load", adapter_error)
        if adapter_error:
            raise H3RuntimeError(adapter_error)
        self._synchronize()
        self._timings["adapter_load_seconds"] = time.perf_counter() - adapter_started

        acceleration_error: str | None = None
        acceleration_receipt: Mapping[str, Any] | None = None
        self._synchronize()
        acceleration_started = time.perf_counter()
        try:
            from ..inference.acceleration import apply_direct_acceleration

            acceleration_receipt = apply_direct_acceleration(self.pipeline.transformer)
        except Exception as exc:
            acceleration_error = _exception_text(exc)
        acceleration_error = self._collective_error(
            "direct inference acceleration", acceleration_error
        )
        if acceleration_error:
            raise H3RuntimeError(acceleration_error)
        self._synchronize()
        self._timings["acceleration_prepare_seconds"] = time.perf_counter() - acceleration_started

        self._text_precompute_receipt: dict[str, Any] | None = None
        self.model_resolution = {
            "backend": "native_torch",
            "partition": "FL2VA",
            "model_root": str(self.model_root),
            "adapter": dict(adapter_receipt or {}),
            "acceleration": dict(acceleration_receipt or {}),
            "parallelism": {
                "tensor_parallel_size": self.tp_size,
                "ulysses_degree": self.ulysses_degree,
                "world_size": self.world_size,
            },
            "precision": "bfloat16",
            "attention": DIRECT_POLICY.attention_backend,
            "dependencies": {
                name: _distribution_version(name)
                for name in ("diffusers", "safetensors", "transformers", "vllm")
            },
        }

    def _synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def _build_parallel_context(self) -> Any:
        from ..distributed import ParallelContext

        tp_groups: list[Any] = []
        for ulysses_rank in range(self.ulysses_degree):
            ranks = [ulysses_rank * self.tp_size + tp_rank for tp_rank in range(self.tp_size)]
            tp_groups.append(torch.distributed.new_group(ranks=ranks, backend="nccl"))
        ulysses_groups: list[Any] = []
        for tp_rank in range(self.tp_size):
            ranks = [
                ulysses_rank * self.tp_size + tp_rank for ulysses_rank in range(self.ulysses_degree)
            ]
            ulysses_groups.append(torch.distributed.new_group(ranks=ranks, backend="nccl"))
        return ParallelContext(
            tp_world_size=self.tp_size,
            tp_rank=self.rank % self.tp_size,
            ulysses_world_size=self.ulysses_degree,
            ulysses_rank=self.rank // self.tp_size,
            tp_group=tp_groups[self.rank // self.tp_size],
            ulysses_group=ulysses_groups[self.rank % self.tp_size],
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        torch.distributed.barrier()

    def broadcast(self, value: Any) -> Any:
        payload = [value if self.is_main_process else None]
        torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    def record_timing(self, name: str, seconds: float) -> None:
        self._timings[name] = float(seconds)

    def reset_dit_peak_memory(self) -> None:
        self._synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)

    def dit_peak_memory_receipt(self) -> dict[str, Any]:
        """Collect the peak CUDA memory observed during the DiT interval."""

        self._synchronize()
        local = {
            "rank": self.rank,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        return {
            "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in gathered),
            "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in gathered),
            "by_rank": gathered,
        }

    def _collective_error(self, label: str, local_error: str | None) -> str | None:
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, {"rank": self.rank, "error": local_error})
        failures = [item for item in gathered if item.get("error")]
        if not failures:
            return None
        return (
            f"{label} failed collectively ("
            + "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
            + ")"
        )

    def precompute_text_prompts(self, prompts: Sequence[str]) -> tuple[tuple[Any, Any], ...]:
        normalized = tuple(prompt.strip() for prompt in prompts)
        if not normalized or any(not prompt for prompt in normalized):
            raise ValueError("prompts must be non-empty")
        assert self.pipeline is not None
        self._synchronize()
        started = time.perf_counter()
        encoded: list[tuple[Any, Any]] = []
        for prompt in normalized:
            hidden, tags = self.pipeline._encode_text(prompt=prompt)
            encoded.append(
                (
                    hidden.detach().to(device="cpu").contiguous(),
                    tags.detach().to(device="cpu").contiguous(),
                )
            )
        self.pipeline._offload_text()
        self._synchronize()
        self._timings["text_precompute_seconds"] = time.perf_counter() - started
        self._text_precompute_receipt = {
            "mode": "replicated",
            "prompt_count": len(encoded),
        }
        return tuple(encoded)

    def text_precompute_receipt(self) -> dict[str, Any] | None:
        return (
            None if self._text_precompute_receipt is None else dict(self._text_precompute_receipt)
        )

    def generate_latents(
        self,
        *,
        video_noise_seed: int,
        audio_noise_seed: int,
        width: int,
        height: int,
        precomputed_text: tuple[Any, Any],
        denoise_loop: Callable[..., tuple[Any, Any]],
    ) -> tuple[Any, Any]:
        """Generate one five-second T2VA request without per-request VAE decode."""

        assert self.pipeline is not None
        self._synchronize()
        started = time.perf_counter()
        result = self.pipeline.generate(
            video_noise_seed=video_noise_seed,
            audio_noise_seed=audio_noise_seed,
            width=width,
            height=height,
            precomputed_text=precomputed_text,
            denoise_loop=denoise_loop,
        )
        video_latents = (
            result.video_latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        audio_latents = (
            result.audio_latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        self._synchronize()
        self._timings["stage3_generation_seconds"] = self._timings.get(
            "stage3_generation_seconds", 0.0
        ) + (time.perf_counter() - started)
        return video_latents, audio_latents

    def decode_video_latents(
        self,
        video_latents: Any,
        *,
        output_path: Path,
        video_filter: str | None = None,
        published_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        """Collectively decode the complete video-latent timeline once."""

        assert self.pipeline is not None
        torch.cuda.empty_cache()
        output_path = output_path.expanduser().resolve()
        self._synchronize()
        started = time.perf_counter()
        result = self.pipeline._decode_and_save_parallel(
            video_latents=video_latents,
            output_path=output_path,
            video_filter=video_filter,
            published_frames=published_frames,
            video_crf=video_crf,
            verify_video_decode=verify_video_decode,
        )
        self._synchronize()
        self._timings["video_decode_shard_one_shot_encode_seconds"] = time.perf_counter() - started
        self._video_vae_receipt = self.pipeline.last_video_decode_receipt
        return result

    def video_vae_receipt(self) -> dict[str, Any] | None:
        value = getattr(self, "_video_vae_receipt", None)
        return None if value is None else dict(value)

    def decode_audio_latents(self, audio_latents: Any) -> tuple[Any, int]:
        """Decode the complete audio-latent timeline once on rank zero."""

        if not self.is_main_process:
            raise H3RuntimeError("only rank zero decodes audio")
        assert self.pipeline is not None
        torch.cuda.empty_cache()
        decoder = self.pipeline._load_output_decoder()
        self._synchronize()
        started = time.perf_counter()
        result = decoder.decode_audio(
            audio_latents=audio_latents,
            device=self.device,
        )
        self._synchronize()
        self._timings["audio_vae_seconds"] = time.perf_counter() - started
        return result

    def performance_receipt(self) -> dict[str, Any]:
        """Collect synchronized timing and CUDA peak-memory telemetry."""

        self._synchronize()
        local = {
            "rank": self.rank,
            "timings": dict(self._timings),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
        }
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        worker_seconds = time.perf_counter() - self._worker_started_at
        worker_values: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(worker_values, worker_seconds)
        static_cache = self.pipeline.transformer.inference_static_timestep_cache_stats()
        video_vae = getattr(self, "_video_vae_receipt", None)
        rank_vae_receipts = (
            video_vae.get("rank_receipts", []) if isinstance(video_vae, dict) else []
        )
        return {
            "timings_seconds": {
                key: max(float(item["timings"].get(key, 0.0)) for item in gathered)
                for key in sorted({key for item in gathered for key in item["timings"]})
            },
            "worker_cold_e2e_seconds": max(float(value) for value in worker_values),
            "cuda_memory_by_rank": gathered,
            "peak_memory_allocated_bytes": max(
                int(item["peak_memory_allocated_bytes"]) for item in gathered
            ),
            "peak_memory_reserved_bytes": max(
                int(item["peak_memory_reserved_bytes"]) for item in gathered
            ),
            "static_timestep_adaln_cache": static_cache,
            "video_vae_active_wall_seconds": (
                max(float(item["active_wall_seconds"]) for item in rank_vae_receipts)
                if rank_vae_receipts
                else None
            ),
            "video_vae_active_cuda_seconds": (
                max(float(item["active_cuda_seconds"]) for item in rank_vae_receipts)
                if rank_vae_receipts
                else None
            ),
        }


__all__ = ["H3Runtime", "H3RuntimeError"]
