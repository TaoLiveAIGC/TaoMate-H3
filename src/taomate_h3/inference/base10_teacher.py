# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Load request-level Base10 audio milestones produced ahead of inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE10_TEACHER_STATE_NUMBERS = (3, 6, 9)
STAGE3_TARGET_STATE_INDICES = (16, 33, 49)
BASE10_ROLLOVER_REFERENCE_LATENTS = 40
_GEOMETRY_KEYS = ("width", "height", "video_latent_h", "video_latent_w")


class Base10TeacherArtifactError(RuntimeError):
    """The external teacher does not match the current inference request."""


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Base10TeacherArtifactError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise Base10TeacherArtifactError(f"{label} must be a JSON object")
    return value


def _geometry(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise Base10TeacherArtifactError(f"{label} is absent")
    result: dict[str, int] = {}
    for key in _GEOMETRY_KEYS:
        item = value.get(key)
        if type(item) is not int or item <= 0:
            raise Base10TeacherArtifactError(f"{label}.{key} must be a positive integer")
        result[key] = item
    return result


@dataclass(frozen=True)
class ExternalBase10TeacherArtifact:
    """A completed BF16 Base10 teacher set consumed request by request.

    A producer may contain more requests than the current run. Only the ordered
    prompt/seed prefix passed to :meth:`open` is consumed. Spatial geometry is
    still exact because H3 samples audio noise after advancing the same random
    generator through resolution-dependent video noise.
    """

    root: Path
    prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    producer_geometry: Mapping[str, int]
    artifact_request_count: int
    audio_noise_seeds: tuple[int, ...]

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        prompts: Sequence[str],
        seeds: Sequence[int],
        consumer_geometry: Mapping[str, Any],
    ) -> "ExternalBase10TeacherArtifact":
        normalized_prompts = tuple(prompts)
        normalized_seeds = tuple(seeds)
        if not normalized_prompts or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in normalized_prompts
        ):
            raise Base10TeacherArtifactError("teacher prompts must be non-empty strings")
        if len(normalized_seeds) != len(normalized_prompts) or any(
            type(seed) is not int or seed < 0 for seed in normalized_seeds
        ):
            raise Base10TeacherArtifactError("one non-negative teacher seed is required per prompt")

        resolved = root.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise Base10TeacherArtifactError("Base10 teacher root must be a directory")
        completion = _read_json(resolved / "complete.json", label="Base10 completion marker")

        request_count = completion.get("request_count")
        if type(request_count) is not int or request_count < len(normalized_prompts):
            raise Base10TeacherArtifactError(
                "Base10 artifact has fewer requests than the current run"
            )
        if completion.get("base_precision") != "bf16":
            raise Base10TeacherArtifactError("external Base10 teacher must use BF16 base weights")
        if (
            completion.get("strategy")
            != "previous_clean_audio_tail_reference_then_new_noise"
            or completion.get("partition") != "fl2va"
        ):
            raise Base10TeacherArtifactError(
                "Base10 teacher must use the FL2VA tail40 rollover contract"
            )
        audio_noise_seed_sequence = completion.get("audio_noise_seed_sequence")
        consumed_audio_noise_seeds = (
            audio_noise_seed_sequence[: len(normalized_prompts)]
            if isinstance(audio_noise_seed_sequence, list)
            else ()
        )
        if (
            not isinstance(audio_noise_seed_sequence, list)
            or len(audio_noise_seed_sequence) < len(normalized_prompts)
            or any(type(value) is not int or value < 0 for value in consumed_audio_noise_seeds)
        ):
            raise Base10TeacherArtifactError("Base10 audio-noise seed sequence is invalid")

        producer_geometry = _geometry(
            completion.get("producer_geometry"), label="producer geometry"
        )
        expected_geometry = _geometry(consumer_geometry, label="consumer geometry")
        if producer_geometry != expected_geometry:
            raise Base10TeacherArtifactError(
                "Base10 producer geometry differs from current inference geometry"
            )

        return cls(
            root=resolved,
            prompts=normalized_prompts,
            seeds=normalized_seeds,
            producer_geometry=producer_geometry,
            artifact_request_count=request_count,
            audio_noise_seeds=tuple(consumed_audio_noise_seeds),
        )

    @property
    def request_count(self) -> int:
        return len(self.prompts)

    def audio_noise_seed(self, request_index: int) -> int:
        return self.audio_noise_seeds[request_index]

    def load_request(
        self,
        torch: Any,
        *,
        request_index: int,
        audio_latent_count: int,
        device: Any,
    ) -> dict[str, Any]:
        """Load one request's three float32 milestone tensors onto ``device``."""

        if not 0 <= request_index < self.request_count:
            raise Base10TeacherArtifactError("teacher request index is outside the current run")
        if type(audio_latent_count) is not int or audio_latent_count <= 0:
            raise Base10TeacherArtifactError("audio latent count must be a positive integer")

        prompt = self.prompts[request_index]
        seed = self.seeds[request_index]
        tensor_path = self.root / f"request_{request_index:02d}.pt"

        expected_request_values = {
            "prompt": prompt,
            "seed": seed,
            "audio_latent_count": audio_latent_count,
        }
        if not tensor_path.is_file():
            raise Base10TeacherArtifactError(
                f"Base10 request {request_index} tensor file is absent: {tensor_path}"
            )
        try:
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise Base10TeacherArtifactError(
                f"cannot load Base10 request {request_index} tensors: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise Base10TeacherArtifactError(
                f"Base10 request {request_index} tensor payload must be a mapping"
            )
        if any(payload.get(key) != value for key, value in expected_request_values.items()):
            raise Base10TeacherArtifactError(
                f"Base10 request {request_index} tensor metadata differs from current request"
            )
        if tuple(payload.get("teacher_state_numbers", ())) != BASE10_TEACHER_STATE_NUMBERS:
            raise Base10TeacherArtifactError(
                f"Base10 request {request_index} must contain states {BASE10_TEACHER_STATE_NUMBERS}"
            )
        if tuple(payload.get("stage3_target_state_indices", ())) != STAGE3_TARGET_STATE_INDICES:
            raise Base10TeacherArtifactError(
                f"Base10 teacher states do not target Stage3 states {STAGE3_TARGET_STATE_INDICES}"
            )

        values = payload.get("milestones")
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise Base10TeacherArtifactError(
                f"Base10 request {request_index} must contain three milestones"
            )
        expected_shape = (2 * audio_latent_count, 32)
        for stage_index, value in enumerate(values):
            if not isinstance(value, torch.Tensor):
                raise Base10TeacherArtifactError(
                    f"Base10 request {request_index} milestone {stage_index} is not a tensor"
                )
            if tuple(value.shape) != expected_shape or value.dtype != torch.float32:
                raise Base10TeacherArtifactError(
                    f"Base10 request {request_index} milestone {stage_index} must be "
                    f"float32 with shape {expected_shape}"
                )

        milestones = {
            index: value.to(device=device, dtype=torch.float32)
            for index, value in enumerate(values)
        }
        receipt = {
            "mode": "base10_fl2va_tail40_rollover",
            "model": "external_base_h3_bf16",
            "source": "external_bf16_artifact",
            "base_precision": "bf16",
            "artifact_storage_dtype": "float32",
            "executed_forwards": 0,
            "producer_executed_forwards": 9,
            "full_state_count": 10,
            "captured_base_state_numbers": list(BASE10_TEACHER_STATE_NUMBERS),
            "stage3_target_state_indices": list(STAGE3_TARGET_STATE_INDICES),
            "packed_video_rows": 0,
            "packed_audio_rows": 2 * (
                audio_latent_count
                + (0 if request_index == 0 else BASE10_ROLLOVER_REFERENCE_LATENTS)
            ),
            "audio_latents_per_channel": audio_latent_count,
            "milestone_shape": list(expected_shape),
            "reused_current_request_initial_audio_noise": True,
            "reused_by_every_streaming_chunk": True,
            "video_projection_executed": False,
            "video_output_head_executed": False,
            "consumer_teacher_model_resident": False,
            "continuation_partition": "fl2va",
            "audio_noise_seed": self.audio_noise_seed(request_index),
            "reference_latents_per_channel": (
                0 if request_index == 0 else BASE10_ROLLOVER_REFERENCE_LATENTS
            ),
            "reference_duration_seconds": 0.0 if request_index == 0 else 1.0,
            "prefix_source_request": None if request_index == 0 else request_index - 1,
            "producer_geometry": dict(self.producer_geometry),
            "artifact_request_count": self.artifact_request_count,
            "consumed_request_count": self.request_count,
        }
        return {"milestones": milestones, "receipt": receipt}


__all__ = [
    "BASE10_ROLLOVER_REFERENCE_LATENTS",
    "BASE10_TEACHER_STATE_NUMBERS",
    "STAGE3_TARGET_STATE_INDICES",
    "Base10TeacherArtifactError",
    "ExternalBase10TeacherArtifact",
]
