# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from .config import DirectRunConfig, build_run_config, write_run_config
from .teacher import launch_audio_guidance

DEFAULT_ADAPTER_DIR = Path("models/TaoMate-H3")
DEFAULT_ADAPTER_REPO = "TaoLiveAIGC/TaoMate-H3"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m taomate_h3",
        description="Generate direct 480p, 768p, or aligned 1080p T2VA with MiniMax H3.",
        epilog="Powered by MiniMax H3.",
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument(
        "--prompt",
        help="one prompt; reused for every five-second request",
    )
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
        help="480p, 768p, or aligned 1080p canvas WIDTHxHEIGHT (default: 768x1376)",
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
    parser.add_argument(
        "--adapter",
        type=Path,
        help="local LoRA directory; defaults to downloading TaoLiveAIGC/TaoMate-H3",
    )
    parser.add_argument("--seed", type=int, default=8301)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty output directory",
    )
    return parser


def prepare_default_adapter(directory: Path) -> None:
    if all((directory / name).is_file() for name in ADAPTER_FILES):
        return
    from huggingface_hub import snapshot_download

    print(f"Downloading LoRA from {DEFAULT_ADAPTER_REPO}...", flush=True)
    snapshot_download(
        repo_id=DEFAULT_ADAPTER_REPO,
        local_dir=str(directory),
        allow_patterns=["config.json", *ADAPTER_FILES],
    )


def _validate_input_paths(config: DirectRunConfig) -> None:
    if not config.model_root.is_dir() or not (config.model_root / "FL2VA").is_dir():
        raise ValueError("--model-root must contain the official FL2VA directory")
    if not config.adapter.is_dir():
        raise ValueError("--adapter must be a directory")
    for name in ADAPTER_FILES:
        if not (config.adapter / name).is_file():
            raise ValueError(f"--adapter must contain {name}")


def _validate_media_tools() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise ValueError("ffmpeg and ffprobe must be available on PATH")
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("cannot inspect ffmpeg encoders") from exc
    encoders = set(completed.stdout.split())
    missing = [name for name in ("libx264", "aac") if name not in encoders]
    if missing:
        raise ValueError(f"ffmpeg is missing required encoders: {', '.join(missing)}")


def _validate_acceleration_runtime() -> None:
    """Fail before torchrun when a required GPU acceleration module is absent."""

    try:
        vllm_ops = importlib.import_module("vllm._custom_ops")
        flash_attn = importlib.import_module("flash_attn_interface")
        importlib.import_module("triton")
    except ImportError as exc:
        raise ValueError(f"accelerated runtime dependency is unavailable: {exc}") from exc
    if not all(
        callable(getattr(vllm_ops, name, None))
        for name in ("scaled_int8_quant", "cutlass_scaled_mm")
    ):
        raise ValueError("vLLM CUTLASS W8A8 operations are unavailable")
    if not callable(getattr(flash_attn, "flash_attn_func", None)):
        raise ValueError("FlashAttention-3 interface is unavailable")


def _validate_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("--output must be a new or empty directory")


def _prepare_output(config: DirectRunConfig) -> Path:
    output = config.output_dir
    _validate_output(output)
    if not output.exists():
        output.mkdir(parents=True)
    config_path = output / "run_config.json"
    write_run_config(config_path, config)
    return config_path


def torchrun_command(
    config: DirectRunConfig,
    config_path: Path,
    audio_guidance_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={config.gpu_count}",
        "--module",
        "taomate_h3.worker",
        "--config",
        str(config_path),
        "--audio-guidance-dir",
        str(audio_guidance_dir),
    ]


def launch(
    config: DirectRunConfig,
    config_path: Path,
    audio_guidance_dir: Path,
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(config.devices)
    environment.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(
        torchrun_command(config, config_path, audio_guidance_dir),
        check=True,
        env=environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_run_config(
            model_root=args.model_root,
            adapter=args.adapter or DEFAULT_ADAPTER_DIR,
            output_dir=args.output,
            prompt=args.prompt,
            prompt_json=args.prompt_json,
            duration_seconds=args.duration,
            resolution=args.resolution,
            seed=args.seed,
            gpu_count=args.gpus,
            devices=args.devices,
        )
        _validate_media_tools()
        _validate_acceleration_runtime()
        _validate_output(config.output_dir)
        if args.adapter is None:
            prepare_default_adapter(config.adapter)
        _validate_input_paths(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Powered by MiniMax H3.", flush=True)
    with TemporaryDirectory(prefix="taomate-h3-audio-", dir="/dev/shm") as temporary:
        temporary_root = Path(temporary)
        audio_guidance_dir = temporary_root / "guidance"
        print("Preparing audio guidance...", flush=True)
        launch_audio_guidance(
            config,
            output_dir=audio_guidance_dir,
        )
        config_path = _prepare_output(config)
        print("Generating audio and video...", flush=True)
        launch(config, config_path, audio_guidance_dir)
    return 0


__all__ = ["build_parser", "launch", "main", "torchrun_command"]
