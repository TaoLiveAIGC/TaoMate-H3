# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SEGMENT_SECONDS = 5
SUPPORTED_SHORT_EDGES = (480, 768, 1088)


@dataclass(frozen=True)
class DirectPolicy:
    """The validated direct streaming contract.

    These values are deliberately not command-line options. Changing one of
    them creates a different inference method and must be evaluated separately.
    """

    request_seconds: int = 5
    chunk_group_counts: tuple[int, ...] = (2, 2, 2, 1)
    precision: str = "bf16_with_interior_fc1_qkv_w8a8"
    attention_backend: str = "flash_attn_3"
    persistent_kv: str = "clean_audio_video"
    video_kv_retention: str = "first_video_sink_plus_recent_2_chunks"
    audio_kv_retention: str = "match_video"
    audio_kv_reset_window_requests: int = 12
    prompt_scope: str = "current_request"
    rope: str = "split_temporal_spatial"
    audio_guidance: str = "fl2va_tail40_rollover"
    audio_guidance_state_numbers: tuple[int, ...] = (3, 6, 9)
    stage3_target_state_indices: tuple[int, ...] = (16, 33, 49)
    prefix_normalization: str = "affine_to_first_stream_chunk"
    canonical_continuation_geometry: bool = True
    publication: str = "one_shot_audio_video_vae"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIRECT_POLICY = DirectPolicy()


DIRECT_W8A8_POLICY = "interior_fc1_qkv_v1"


def parse_resolution(value: str) -> tuple[int, int, str]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("resolution must use WIDTHxHEIGHT, for example 480x864")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("resolution width and height must be integers") from exc
    if min(width, height) not in SUPPORTED_SHORT_EDGES or width % 32 or height % 32:
        raise ValueError(
            "direct inference requires a 480-, 768-, or 1088-pixel short edge and "
            "32-pixel alignment"
        )
    return width, height, f"{width}x{height}"


def parse_devices(value: str | None, *, gpu_count: int) -> tuple[str, ...]:
    if gpu_count not in (4, 8):
        raise ValueError("GPU count must be 4 or 8")
    devices = (
        tuple(str(index) for index in range(gpu_count))
        if value is None
        else tuple(part.strip() for part in value.split(","))
    )
    if len(devices) != gpu_count or any(not item for item in devices):
        raise ValueError("--devices must list exactly --gpus entries")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return devices


def load_prompt_json(path: Path) -> tuple[tuple[str, ...], tuple[int, ...] | None]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prompt JSON: {exc}") from exc
    prompts = payload.get("prompts") if isinstance(payload, Mapping) else None
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('prompt JSON must contain {"prompts": ["...", "..."]}')
    normalized: list[str] = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt JSON item {index} must be a non-empty string")
        normalized.append(prompt.strip())
    raw_seeds = payload.get("seeds")
    if raw_seeds is None:
        seeds = None
    elif (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) != len(normalized)
        or any(type(seed) is not int or seed < 0 for seed in raw_seeds)
    ):
        raise ValueError("prompt JSON seeds must be one non-negative integer per prompt")
    else:
        seeds = tuple(raw_seeds)
    return tuple(normalized), seeds


def resolve_prompts(
    *,
    prompt: str | None,
    prompt_json: Path | None,
    duration_seconds: int | None,
) -> tuple[tuple[str, ...], int, tuple[int, ...] | None]:
    if (prompt is None) == (prompt_json is None):
        raise ValueError("choose exactly one of --prompt or --prompt-json")
    if prompt_json is not None:
        supplied, seeds = load_prompt_json(prompt_json)
    else:
        supplied, seeds = (str(prompt).strip(),), None
    if any(not item for item in supplied):
        raise ValueError("prompt must not be empty")
    if duration_seconds is None:
        duration_seconds = (
            len(supplied) * SEGMENT_SECONDS if prompt_json is not None else SEGMENT_SECONDS
        )
    if duration_seconds <= 0 or duration_seconds % SEGMENT_SECONDS:
        raise ValueError("duration must be a positive multiple of 5 seconds")
    request_count = duration_seconds // SEGMENT_SECONDS
    if prompt_json is not None:
        if len(supplied) != request_count:
            raise ValueError("prompt JSON count must equal duration divided by 5 seconds")
        prompts = supplied
    else:
        prompts = supplied * request_count
    return prompts, duration_seconds, seeds


@dataclass(frozen=True)
class DirectRunConfig:
    model_root: Path
    adapter: Path
    output_dir: Path
    prompts: tuple[str, ...]
    duration_seconds: int
    resolution: str
    width: int
    height: int
    seed: int
    request_seeds: tuple[int, ...]
    gpu_count: int
    devices: tuple[str, ...]
    tensor_parallel_size: int = 2
    ulysses_degree: int = 4
    segment_seconds: int = SEGMENT_SECONDS

    def __post_init__(self) -> None:
        expected_width, expected_height, normalized = parse_resolution(self.resolution)
        if self.segment_seconds != SEGMENT_SECONDS:
            raise ValueError("the direct streaming request size must remain 5 seconds")
        if self.duration_seconds <= 0 or self.duration_seconds % SEGMENT_SECONDS:
            raise ValueError("duration must be a positive multiple of 5 seconds")
        if len(self.prompts) != self.duration_seconds // SEGMENT_SECONDS:
            raise ValueError("one prompt is required for every 5-second request")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in self.prompts):
            raise ValueError("prompts must be non-empty strings")
        if self.resolution != normalized:
            raise ValueError("resolution must be normalized")
        if (self.width, self.height) != (expected_width, expected_height):
            raise ValueError("output canvas differs from the requested resolution")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(self.request_seeds) != len(self.prompts) or any(
            type(seed) is not int or seed < 0 for seed in self.request_seeds
        ):
            raise ValueError("one non-negative seed is required per request")
        expected_ulysses = {4: 2, 8: 4}.get(self.gpu_count)
        if (
            self.tensor_parallel_size != 2
            or self.ulysses_degree != expected_ulysses
            or len(self.devices) != self.gpu_count
            or len(set(self.devices)) != self.gpu_count
        ):
            raise ValueError("distributed topology differs from TP2 x Ulysses2/4")

    @property
    def request_count(self) -> int:
        return len(self.prompts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_root": str(self.model_root),
            "adapter": str(self.adapter),
            "output_dir": str(self.output_dir),
            "prompts": list(self.prompts),
            "duration_seconds": self.duration_seconds,
            "segment_seconds": self.segment_seconds,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "request_seeds": list(self.request_seeds),
            "gpu_count": self.gpu_count,
            "devices": list(self.devices),
            "parallelism": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "ulysses_degree": self.ulysses_degree,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DirectRunConfig":
        parallelism = payload.get("parallelism")
        if not isinstance(parallelism, Mapping):
            raise ValueError("run config parallelism is absent")
        prompts = payload.get("prompts")
        devices = payload.get("devices")
        if not isinstance(prompts, list) or not isinstance(devices, list):
            raise ValueError("run config prompts or devices are invalid")
        config = cls(
            model_root=Path(str(payload.get("model_root"))),
            adapter=Path(str(payload.get("adapter"))),
            output_dir=Path(str(payload.get("output_dir"))),
            prompts=tuple(prompts),
            duration_seconds=int(payload.get("duration_seconds")),
            segment_seconds=int(payload.get("segment_seconds")),
            resolution=str(payload.get("resolution")),
            width=int(payload.get("width")),
            height=int(payload.get("height")),
            seed=int(payload.get("seed")),
            request_seeds=tuple(int(item) for item in payload.get("request_seeds", [])),
            gpu_count=int(payload.get("gpu_count")),
            devices=tuple(str(item) for item in devices),
            tensor_parallel_size=int(parallelism.get("tensor_parallel_size")),
            ulysses_degree=int(parallelism.get("ulysses_degree")),
        )
        return config


def build_run_config(
    *,
    model_root: Path,
    adapter: Path,
    output_dir: Path,
    prompt: str | None,
    prompt_json: Path | None,
    duration_seconds: int | None,
    resolution: str,
    seed: int,
    gpu_count: int,
    devices: str | None,
) -> DirectRunConfig:
    prompts, resolved_duration, supplied_seeds = resolve_prompts(
        prompt=prompt,
        prompt_json=prompt_json,
        duration_seconds=duration_seconds,
    )
    width, height, normalized_resolution = parse_resolution(resolution)
    resolved_devices = parse_devices(devices, gpu_count=gpu_count)
    return DirectRunConfig(
        model_root=model_root.expanduser().resolve(),
        adapter=adapter.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        prompts=prompts,
        duration_seconds=resolved_duration,
        resolution=normalized_resolution,
        width=width,
        height=height,
        seed=seed,
        request_seeds=(supplied_seeds if supplied_seeds is not None else (seed,) * len(prompts)),
        gpu_count=gpu_count,
        devices=resolved_devices,
        ulysses_degree={4: 2, 8: 4}[gpu_count],
    )


def load_run_config(path: Path) -> DirectRunConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run config: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("run config must be a JSON object")
    return DirectRunConfig.from_dict(payload)


def write_run_config(path: Path, config: DirectRunConfig) -> None:
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DIRECT_POLICY",
    "DIRECT_W8A8_POLICY",
    "DirectPolicy",
    "DirectRunConfig",
    "SEGMENT_SECONDS",
    "SUPPORTED_SHORT_EDGES",
    "build_run_config",
    "load_prompt_json",
    "load_run_config",
    "parse_devices",
    "parse_resolution",
    "resolve_prompts",
    "write_run_config",
]
