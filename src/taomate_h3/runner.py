# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Direct long-video generation and one-shot publication."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .config import DIRECT_POLICY, DirectRunConfig
from .inference import ExternalBase10TeacherArtifact
from .inference.media_timing import (
    endpoint_preserving_video_filter,
    exact_audio_delivery_filter,
)
from .model.geometry import direct_5s_geometry
from .runtime import H3Runtime
from .streaming import direct_5s_plan
from .streaming.runtime import H3StreamingRuntime


_VIDEO_NOISE_REQUEST_STRIDE = 1_000_003


def _media_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required media tool is not installed: {name}")
    return path


def _run_media(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            _media_binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:stream=codec_type,codec_name,pix_fmt,width,height,"
                "r_frame_rate,nb_frames,duration_ts,time_base,sample_rate,channels"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("ffprobe returned an invalid result")
    return value


def _strict_decode(path: Path) -> None:
    _run_media(
        [
            _media_binary("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )


def _video_codec_frames(temporal_latents: int) -> int:
    if temporal_latents < 2 or (temporal_latents - 2) % 5:
        raise RuntimeError("video latents do not form one native H3 codec stream")
    return 5 + 17 * ((temporal_latents - 2) // 5)


def _validate_final(path: Path, *, config: DirectRunConfig) -> dict[str, Any]:
    probe = _probe(path)
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError("output has no streams")
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, Mapping) or not isinstance(audio, Mapping):
        raise RuntimeError("output must contain video and audio")
    if (
        int(video.get("width", 0)) != config.width
        or int(video.get("height", 0)) != config.height
        or int(video.get("nb_frames", 0)) != config.duration_seconds * 24
        or video.get("r_frame_rate") != "24/1"
        or video.get("codec_name") != "h264"
        or video.get("pix_fmt") != "yuv420p"
    ):
        raise RuntimeError("output video differs from the requested direct canvas")
    if (
        audio.get("codec_name") != "aac"
        or int(audio.get("sample_rate", 0)) != 32_000
        or int(audio.get("channels", 0)) != 2
    ):
        raise RuntimeError("output audio differs from the H3 delivery contract")
    duration = float(probe.get("format", {}).get("duration", -1))
    if abs(duration - config.duration_seconds) > 0.05:
        raise RuntimeError("output duration differs from the requested duration")
    _strict_decode(path)
    return probe


def _publish(
    *,
    runtime: H3Runtime,
    config: DirectRunConfig,
    streaming: H3StreamingRuntime,
    video_segments: list[Any],
    audio_segments: list[Any],
) -> dict[str, Any]:
    import torch

    if len(streaming.executions) != config.request_count:
        raise RuntimeError("not every request entered the streaming runtime")
    cropped_video: list[Any] = []
    cropped_audio: list[Any] = []
    transport_video: list[int] = []
    transport_audio: list[int] = []
    for index, (video, audio, execution) in enumerate(
        zip(video_segments, audio_segments, streaming.executions, strict=True)
    ):
        if execution.request_index != index:
            raise RuntimeError("streaming execution order differs from prompts")
        video_prefix = int(execution.video_transport_prefix_latents)
        audio_prefix = int(execution.audio_transport_prefix_latents_per_channel)
        if video_prefix + int(execution.published_video_latents) != video.shape[2]:
            raise RuntimeError("video transport prefix differs from runtime output")
        if audio_prefix + int(execution.published_audio_latents_per_channel) != audio.shape[2]:
            raise RuntimeError("audio transport prefix differs from runtime output")
        cropped_video.append(video[:, :, video_prefix:].contiguous())
        cropped_audio.append(audio[:, :, audio_prefix:].contiguous())
        transport_video.append(video_prefix)
        transport_audio.append(audio_prefix)

    full_video = torch.cat(cropped_video, dim=2).contiguous()
    full_audio = torch.cat(cropped_audio, dim=2).contiguous()
    native_frames = _video_codec_frames(int(full_video.shape[2]))
    published_frames = config.duration_seconds * 24
    published_samples = config.duration_seconds * 32_000
    video_only = config.output_dir / "video.only.mp4"
    final_video = config.output_dir / "video.mp4"

    runtime.decode_video_latents(
        full_video,
        output_path=video_only,
        video_filter=endpoint_preserving_video_filter(
            native_frames=native_frames,
            published_frames=published_frames,
        ),
        published_frames=published_frames,
        video_crf=18,
        verify_video_decode=False,
    )
    publication_error: str | None = None
    publication: dict[str, Any] | None = None
    if runtime.is_main_process:
        try:
            waveform, sample_rate = runtime.decode_audio_latents(full_audio)
            if sample_rate != 32_000 or tuple(waveform.shape[:2]) != (1, 2):
                raise RuntimeError("Audio VAE returned an invalid stereo waveform")
            native_samples = int(waveform.shape[2])
            pcm = waveform[0].detach().float().transpose(0, 1).contiguous().cpu().numpy()
            with tempfile.TemporaryDirectory(prefix="taomate-h3-audio-") as temporary:
                pcm_path = Path(temporary) / "audio.f32le"
                pcm_path.write_bytes(pcm.tobytes())
                _run_media(
                    [
                        _media_binary("ffmpeg"),
                        "-nostdin",
                        "-v",
                        "error",
                        "-xerror",
                        "-i",
                        str(video_only),
                        "-f",
                        "f32le",
                        "-ar",
                        str(sample_rate),
                        "-ac",
                        "2",
                        "-i",
                        str(pcm_path),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-af",
                        exact_audio_delivery_filter(
                            native_samples=native_samples,
                            published_samples=published_samples,
                            sample_rate=sample_rate,
                        ),
                        "-frames:v",
                        str(published_frames),
                        "-t",
                        str(config.duration_seconds),
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-profile:a",
                        "aac_low",
                        "-b:a",
                        "192k",
                        "-ar",
                        "32000",
                        "-ac",
                        "2",
                        "-movflags",
                        "+faststart",
                        "-n",
                        str(final_video),
                    ]
                )
            probe = _validate_final(final_video, config=config)
            video_only.unlink()
            publication = {
                "video": str(final_video),
                "probe": probe,
                "strict_decode": True,
                "video_latent_shape": list(full_video.shape),
                "audio_latent_shape": list(full_audio.shape),
                "video_transport_prefix_latents": transport_video,
                "audio_transport_prefix_latents": transport_audio,
                "native_video_frames": native_frames,
                "published_video_frames": published_frames,
                "published_audio_samples": published_samples,
                "video_vae_decodes": 1,
                "h264_encode_count": 1,
                "video_vae": runtime.video_vae_receipt(),
                "audio_vae_decodes": 1,
                "temporary_media_removed": True,
            }
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = runtime.broadcast(publication_error)
    if publication_error:
        raise RuntimeError(f"media publication failed: {publication_error}")
    publication = runtime.broadcast(publication)
    if not isinstance(publication, dict):
        raise RuntimeError("rank zero returned no publication receipt")
    return publication


def run_direct(
    config: DirectRunConfig,
    *,
    audio_guidance_dir: Path,
) -> dict[str, Any]:
    """Generate all five-second requests and publish one continuous AV file."""

    started = time.perf_counter()
    stream_plan = direct_5s_plan()
    geometry = direct_5s_geometry(width=config.width, height=config.height)
    teacher = ExternalBase10TeacherArtifact.open(
        audio_guidance_dir,
        prompts=config.prompts,
        seeds=config.request_seeds,
        consumer_geometry={
            "width": config.width,
            "height": config.height,
            "video_latent_h": geometry.video_latent_h,
            "video_latent_w": geometry.video_latent_w,
        },
    )
    runtime = H3Runtime(
        model_root=config.model_root,
        adapter_root=config.adapter,
        output_root=config.output_dir,
        ulysses_degree=config.ulysses_degree,
    )
    precomputed = runtime.precompute_text_prompts(config.prompts)
    video_segments: list[Any] = []
    audio_segments: list[Any] = []
    streaming = H3StreamingRuntime(base10_teacher=teacher)
    runtime.reset_dit_peak_memory()
    noise_contract: list[dict[str, int]] = []
    for request_index, (text_features, request_seed) in enumerate(
        zip(precomputed, config.request_seeds, strict=True)
    ):
        video_noise_seed = (
            request_seed + request_index * _VIDEO_NOISE_REQUEST_STRIDE
        ) % (2**63)
        audio_noise_seed = teacher.audio_noise_seed(request_index)
        video_latents, audio_latents = runtime.generate_latents(
            video_noise_seed=video_noise_seed,
            audio_noise_seed=audio_noise_seed,
            width=config.width,
            height=config.height,
            precomputed_text=text_features,
            denoise_loop=streaming.run,
        )
        video_segments.append(video_latents)
        audio_segments.append(audio_latents)
        noise_contract.append(
            {
                "request_index": request_index,
                "authored_seed": request_seed,
                "video_noise_seed": video_noise_seed,
                "audio_noise_seed": audio_noise_seed,
            }
        )
    dit_timing = streaming.dit_timing_receipt()
    dit_memory = runtime.dit_peak_memory_receipt()
    streaming.release_retained_state()
    publication_started = time.perf_counter()
    publication = _publish(
        runtime=runtime,
        config=config,
        streaming=streaming,
        video_segments=video_segments,
        audio_segments=audio_segments,
    )
    runtime.record_timing(
        "one_shot_publication_seconds",
        time.perf_counter() - publication_started,
    )
    performance = runtime.performance_receipt()
    dit_timing["frames_per_second"] = (
        config.duration_seconds * 24 / float(dit_timing["cuda_seconds"])
    )
    performance["dit"] = {**dit_timing, **dit_memory}
    performance["run_direct_wall_seconds"] = time.perf_counter() - started
    performance["dit_frames_per_second"] = dit_timing["frames_per_second"]
    performance["real_time_factor"] = (
        config.duration_seconds / performance["worker_cold_e2e_seconds"]
    )
    streaming_receipt = {
        "policy": DIRECT_POLICY.to_dict(),
        "stream_plan": stream_plan.to_dict(),
        "request_count": config.request_count,
        "executions": [execution.to_dict() for execution in streaming.executions],
    }

    result = {
        "config": config.to_dict(),
        "model": runtime.model_resolution,
        "text_precompute": runtime.text_precompute_receipt(),
        "streaming": streaming_receipt,
        "noise_contract": noise_contract,
        "publication": publication,
        "performance": performance,
    }
    if runtime.is_main_process:
        (config.output_dir / "latent_streaming.json").write_text(
            json.dumps(streaming_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (config.output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    runtime.barrier()
    return result


__all__ = ["run_direct"]
