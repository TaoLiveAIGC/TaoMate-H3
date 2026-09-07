# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Generate internal audio guidance for streaming inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

from .config import DirectRunConfig, parse_devices, parse_resolution, resolve_prompts
from .inference.base10_teacher import (
    BASE10_TEACHER_STATE_NUMBERS as _TEACHER_STATE_NUMBERS,
)
from .inference.base10_teacher import (
    STAGE3_TARGET_STATE_INDICES as _STAGE3_TARGET_STATE_INDICES,
)

_TP_SIZE = 2
_ROLLOVER_LATENTS_PER_CHANNEL = 40


@dataclass(frozen=True)
class TeacherConfig:
    model_root: Path
    output_dir: Path
    prompts: tuple[str, ...]
    request_seeds: tuple[int, ...]
    duration_seconds: int
    width: int
    height: int
    resolution: str
    devices: tuple[str, ...]
    audio_seed_base: int
    gpu_count: int
    ulysses_degree: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m taomate_h3.teacher",
        description="Generate internal audio guidance for TaoMate-H3 inference.",
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="one prompt; reused for every five-second request")
    prompts.add_argument(
        "--prompt-json",
        type=Path,
        help="JSON containing prompts and optional per-request seeds",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="total seconds; must be a positive multiple of five",
    )
    parser.add_argument(
        "--resolution",
        default="768x1376",
        help="480p, 768p, or aligned 1080p consumer canvas WIDTHxHEIGHT (default: 768x1376)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        choices=(4, 8),
        default=8,
        help="local GPU count: 4 or 8 (default: 8)",
    )
    parser.add_argument(
        "--devices",
        help="comma-separated CUDA devices; defaults to 0..gpus-1",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="local MiniMax-H3 root containing FL2VA/",
    )
    parser.add_argument("--seed", type=int, default=8301)
    parser.add_argument(
        "--audio-seed-base",
        type=int,
        default=8301,
        help="first audio-noise seed; request index is added",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty audio-guidance directory",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _config_from_args(args: argparse.Namespace) -> TeacherConfig:
    prompts, duration_seconds, prompt_seeds = resolve_prompts(
        prompt=args.prompt,
        prompt_json=args.prompt_json,
        duration_seconds=args.duration,
    )
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    if type(args.audio_seed_base) is not int or args.audio_seed_base < 0:
        raise ValueError("--audio-seed-base must be a non-negative integer")
    width, height, resolution = parse_resolution(args.resolution)
    model_root = args.model_root.expanduser().resolve()
    if not (model_root / "FL2VA").is_dir():
        raise ValueError("--model-root must contain the official FL2VA directory")
    return TeacherConfig(
        model_root=model_root,
        output_dir=args.output.expanduser().resolve(),
        prompts=prompts,
        request_seeds=(prompt_seeds if prompt_seeds is not None else (args.seed,) * len(prompts)),
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        resolution=resolution,
        devices=parse_devices(args.devices, gpu_count=args.gpus),
        audio_seed_base=args.audio_seed_base,
        gpu_count=args.gpus,
        ulysses_degree={4: 2, 8: 4}[args.gpus],
    )


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("--output must be a new or empty directory")
    else:
        path.mkdir(parents=True)


def _worker_arguments(config: TeacherConfig, *, prompt_path: Path) -> list[str]:
    args = [
        "--worker",
        "--model-root",
        str(config.model_root),
        "--duration",
        str(config.duration_seconds),
        "--resolution",
        config.resolution,
        "--audio-seed-base",
        str(config.audio_seed_base),
        "--devices",
        ",".join(config.devices),
        "--gpus",
        str(config.gpu_count),
        "--output",
        str(config.output_dir),
    ]
    prompt_path.write_text(
        json.dumps(
            {"prompts": list(config.prompts), "seeds": list(config.request_seeds)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args.extend(("--prompt-json", str(prompt_path)))
    return args


def _launch(config: TeacherConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="taomate-h3-teacher-") as temporary:
        prompt_path = Path(temporary) / "prompts.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={config.gpu_count}",
            "--module",
            "taomate_h3.teacher",
            *_worker_arguments(config, prompt_path=prompt_path),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(config.devices)
        environment.setdefault("OMP_NUM_THREADS", "1")
        subprocess.run(command, check=True, env=environment)


def _parallel_context(torch: Any, rank: int, *, ulysses_degree: int) -> Any:
    from .distributed import ParallelContext

    tp_groups = [
        torch.distributed.new_group(
            ranks=[ulysses_rank * _TP_SIZE + tp_rank for tp_rank in range(_TP_SIZE)],
            backend="nccl",
        )
        for ulysses_rank in range(ulysses_degree)
    ]
    ulysses_groups = [
        torch.distributed.new_group(
            ranks=[ulysses_rank * _TP_SIZE + tp_rank for ulysses_rank in range(ulysses_degree)],
            backend="nccl",
        )
        for tp_rank in range(_TP_SIZE)
    ]
    return ParallelContext(
        tp_world_size=_TP_SIZE,
        tp_rank=rank % _TP_SIZE,
        ulysses_world_size=ulysses_degree,
        ulysses_rank=rank // _TP_SIZE,
        tp_group=tp_groups[rank // _TP_SIZE],
        ulysses_group=ulysses_groups[rank % _TP_SIZE],
    )


def _official_audio_noise(torch: Any, *, geometry: Any, seed: int) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # H3 draws full-AV video noise before audio. The teacher discards these
    # values, but advancing this exact CPU generator is part of audio identity.
    torch.randn(
        1,
        24,
        geometry.video_latent_t,
        geometry.video_latent_h,
        geometry.video_latent_w,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return torch.randn(
        2 * geometry.audio_latent_t,
        32,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


def _audio_branch(
    torch: Any,
    *,
    pipeline: Any,
    geometry: Any,
    text_features: tuple[Any, Any],
    audio_latent_count: int,
    previous_audio_latent_count: int | None,
) -> Any:
    from .model.denoise import MiniMaxH3DenoiseBranch
    from .model.packed_sequence import (
        minimax_h3_audio_only_frozen_prefix_packed_sequence,
        minimax_h3_audio_only_packed_sequence,
    )

    text_embeddings, text_tags = text_features
    text_len = int(text_embeddings.shape[0])
    if previous_audio_latent_count is None:
        packed = minimax_h3_audio_only_packed_sequence(
            text_len=text_len,
            audio_t=audio_latent_count,
            latent_h=geometry.video_latent_h,
            latent_w=geometry.video_latent_w,
        )
    else:
        packed = minimax_h3_audio_only_frozen_prefix_packed_sequence(
            text_len=text_len,
            ref_audio_t=_ROLLOVER_LATENTS_PER_CHANNEL,
            audio_t=audio_latent_count,
            latent_h=geometry.video_latent_h,
            latent_w=geometry.video_latent_w,
            reference_time_start=(
                text_len + previous_audio_latent_count - _ROLLOVER_LATENTS_PER_CHANNEL
            ),
            target_time_start=text_len + previous_audio_latent_count,
        )
    packed["token_tags"][packed["text_pos"].view(-1)] = text_tags.view(-1)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text_embeddings,
        token_tags=packed["token_tags"],
        device=pipeline.device,
        parallel_context=pipeline.parallel_context,
    )
    refined = pipeline.transformer.refine_prompt_embeds(
        branch.static_kwargs["prompt_embeds"],
        branch.refiner_cu_seqlens,
        device=pipeline.device,
    )
    branch.static_kwargs["prompt_embeds"] = refined
    branch.static_kwargs["rope_cache"] = pipeline.transformer.build_rope_cache(
        branch.img_position_ids,
        device=pipeline.device,
    )
    return branch


def _previous_clean_tail(clean: Any, *, audio_latent_count: int) -> Any:
    return (
        clean.view(2, audio_latent_count, 32)[:, -_ROLLOVER_LATENTS_PER_CHANNEL:]
        .contiguous()
        .view(2 * _ROLLOVER_LATENTS_PER_CHANNEL, 32)
    )


def _generate_request(
    torch: Any,
    *,
    pipeline: Any,
    geometry: Any,
    text_features: tuple[Any, Any],
    audio_seed: int,
    request_index: int,
    previous_clean: Any | None,
    previous_audio_latent_count: int | None,
) -> tuple[list[Any], Any, int, dict[str, Any]]:
    from .denoise_schedule import select_time_shift_sigmas
    from .model.denoise import minimax_h3_denoise_loop
    from .streaming.geometry import canonical_continuation_plan, direct_5s_plan

    official_plan = direct_5s_plan()
    active_plan = (
        official_plan
        if request_index == 0
        else canonical_continuation_plan(official_plan, request_index=request_index)
    )
    official_audio_latents = geometry.audio_latent_t
    active_audio_latents = active_plan.phases[-1].audio_latent_stop
    transport_prefix = official_audio_latents - active_audio_latents
    official_audio = _official_audio_noise(torch, geometry=geometry, seed=audio_seed)
    target_noise = (
        official_audio.view(2, official_audio_latents, 32)[:, transport_prefix:]
        .contiguous()
        .view(2 * active_audio_latents, 32)
    )
    previous_tail = (
        None
        if previous_clean is None
        else _previous_clean_tail(
            previous_clean,
            audio_latent_count=int(previous_audio_latent_count),
        )
    )
    initial_audio = (
        target_noise if previous_tail is None else torch.cat((previous_tail, target_noise), dim=0)
    )
    branch = _audio_branch(
        torch,
        pipeline=pipeline,
        geometry=geometry,
        text_features=text_features,
        audio_latent_count=active_audio_latents,
        previous_audio_latent_count=previous_audio_latent_count,
    )
    video_sigmas = select_time_shift_sigmas(num_steps=10, shift_scale=12.0)
    audio_sigmas = select_time_shift_sigmas(num_steps=10, shift_scale=3.0)
    captured: dict[int, Any] = {}

    def capture(step: int, _video: Any, audio: Any) -> None:
        state_number = step + 1
        if state_number in _TEACHER_STATE_NUMBERS:
            captured[state_number] = (
                audio[branch.audio_target_slice]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )

    empty_video = torch.empty(
        (0, int(pipeline.transformer.arch.video_row_width)),
        dtype=torch.float32,
        device="cpu",
    )
    with torch.no_grad():
        minimax_h3_denoise_loop(
            model=pipeline.transformer,
            positive=branch,
            initial_video_rows=empty_video,
            initial_audio_rows=initial_audio,
            sigmas_video=video_sigmas,
            sigmas_audio=audio_sigmas,
            device=pipeline.device,
            on_step=capture,
        )
    milestones = [captured[state_number] for state_number in _TEACHER_STATE_NUMBERS]
    target_clean = milestones[-1]
    return (
        milestones,
        target_clean,
        active_audio_latents,
        {
            "transport_prefix_audio_latents_per_channel": transport_prefix,
            "packed_text_rows": int(text_features[0].shape[0]),
            "packed_audio_rows": int(branch.audio_pos.shape[0]),
            "reference_latents_per_channel": (
                0 if previous_tail is None else _ROLLOVER_LATENTS_PER_CHANNEL
            ),
            "reference_duration_seconds": 0.0 if previous_tail is None else 1.0,
            "audio_noise_seed": audio_seed,
        },
    )


def _run_worker(config: TeacherConfig) -> None:
    import torch

    from .model.geometry import direct_5s_geometry
    from .model.pipeline import MiniMaxH3NativePipeline

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != config.gpu_count:
        raise ValueError(
            f"audio-guidance torchrun world size must be {config.gpu_count}, got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=3)
    )
    context = _parallel_context(torch, rank, ulysses_degree=config.ulysses_degree)
    pipeline = MiniMaxH3NativePipeline(
        model_root=config.model_root,
        device=device,
        parallel_context=context,
        rank=rank,
        world_size=world_size,
        transformer_load_w8a8_policy=None,
    )
    geometry = direct_5s_geometry(width=config.width, height=config.height)
    text_features = tuple(
        (
            hidden.detach().to(device="cpu").contiguous(),
            tags.detach().to(device="cpu").contiguous(),
        )
        for hidden, tags in (pipeline._encode_text(prompt=prompt) for prompt in config.prompts)
    )
    pipeline._offload_text()

    active_counts: list[int] = []
    request_receipts: list[dict[str, Any]] = []
    audio_seed_sequence = [config.audio_seed_base + index for index in range(len(config.prompts))]
    previous_clean = None
    previous_audio_latent_count: int | None = None
    for request_index, features in enumerate(text_features):
        milestones, current_clean, active_audio_latents, request_receipt = _generate_request(
            torch,
            pipeline=pipeline,
            geometry=geometry,
            text_features=features,
            audio_seed=audio_seed_sequence[request_index],
            request_index=request_index,
            previous_clean=previous_clean,
            previous_audio_latent_count=previous_audio_latent_count,
        )
        active_counts.append(active_audio_latents)
        request_receipt["prefix_source_request"] = (
            None if request_index == 0 else request_index - 1
        )
        request_receipts.append(request_receipt)
        previous_clean = current_clean
        previous_audio_latent_count = active_audio_latents
        if rank == 0:
            torch.save(
                {
                    "prompt": config.prompts[request_index],
                    "seed": config.request_seeds[request_index],
                    "audio_latent_count": active_audio_latents,
                    "teacher_state_numbers": _TEACHER_STATE_NUMBERS,
                    "stage3_target_state_indices": _STAGE3_TARGET_STATE_INDICES,
                    "milestones": milestones,
                },
                config.output_dir / f"request_{request_index:02d}.pt",
            )
        torch.distributed.barrier()

    if rank == 0:
        completion = {
            "mode": "base10_milestones",
            "strategy": "previous_clean_audio_tail_reference_then_new_noise",
            "partition": "fl2va",
            "base_precision": "bf16",
            "parallelism": {
                "tensor_parallel_size": _TP_SIZE,
                "ulysses_degree": config.ulysses_degree,
            },
            "request_count": len(config.prompts),
            "request_seconds": 5,
            "request_seeds": list(config.request_seeds),
            "audio_noise_seed_sequence": audio_seed_sequence,
            "producer_geometry": {
                "width": config.width,
                "height": config.height,
                "video_latent_h": geometry.video_latent_h,
                "video_latent_w": geometry.video_latent_w,
            },
            "official_audio_latents_per_channel": geometry.audio_latent_t,
            "active_audio_latents_per_channel": active_counts,
            "request_receipts": request_receipts,
            "full_state_count": 10,
            "executed_forwards_per_request": 9,
            "teacher_state_numbers": list(_TEACHER_STATE_NUMBERS),
            "stage3_target_state_indices": list(_STAGE3_TARGET_STATE_INDICES),
            "artifact_storage_dtype": "float32",
            "video_rows": 0,
            "noise_order": "draw_and_discard_full_video_then_draw_audio",
            "reference_latents_per_channel": _ROLLOVER_LATENTS_PER_CHANNEL,
            "reference_duration_seconds": 1.0,
            "persistent_kv": False,
            "waveform_crossfade": False,
            "audio_vae_decode_count": 0,
            "adapter_loaded": False,
        }
        (config.output_dir / "complete.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    torch.distributed.barrier()


def launch_audio_guidance(
    run_config: DirectRunConfig,
    *,
    output_dir: Path,
) -> None:
    """Generate the audio guidance consumed by one inference run."""

    config = TeacherConfig(
        model_root=run_config.model_root,
        output_dir=output_dir,
        prompts=run_config.prompts,
        request_seeds=run_config.request_seeds,
        duration_seconds=run_config.duration_seconds,
        width=run_config.width,
        height=run_config.height,
        resolution=run_config.resolution,
        devices=run_config.devices,
        audio_seed_base=run_config.seed,
        gpu_count=run_config.gpu_count,
        ulysses_degree=run_config.ulysses_degree,
    )
    _prepare_output(config.output_dir)
    _launch(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.worker:
            _run_worker(config)
        else:
            _prepare_output(config.output_dir)
            _launch(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TeacherConfig", "build_parser", "launch_audio_guidance", "main"]
