# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

from ..config import DIRECT_POLICY
from ..inference import ExternalBase10TeacherArtifact
from .cache import CleanAVKVCache, kv_contract_from_model
from .geometry import (
    StreamPhase,
    canonical_continuation_plan,
    direct_5s_plan,
    video_temporal_position,
    video_temporal_positions,
)
from .session import H3StreamingSession


@dataclass(frozen=True)
class StreamingPhaseExecution:
    phase_index: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int
    denoise_forwards: int
    clean_forwards: int
    committed_history_tokens: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StreamingExecution:
    phase_executions: tuple[StreamingPhaseExecution, ...]
    video_latent_offset: int = 0
    audio_latent_offset: int = 0
    starting_history_tokens: int = 0
    retained_history_tokens: int = 0
    cross_request_kv_reused: bool = False
    request_index: int = 0
    audio_kv_reset_applied: bool = False
    audio_kv_reset_tokens: int = 0
    starting_history_audio_tokens: int = 0
    starting_history_video_tokens: int = 0
    retained_history_audio_tokens: int = 0
    retained_history_video_tokens: int = 0
    native_frame_offset: int = 0
    published_native_frames: int = 0
    video_transport_prefix_latents: int = 0
    audio_transport_prefix_latents_per_channel: int = 0
    published_video_latents: int = 0
    published_audio_latents_per_channel: int = 0
    current_text_token_count: int = 0
    text_time_start: float = 0.0
    fixed_media_time_origin: int = 0
    position_capture: dict[str, object] | None = None
    audio_teacher: dict[str, object] | None = None
    attention_backend: dict[str, object] | None = None

    @property
    def phase_count(self) -> int:
        return len(self.phase_executions)

    @property
    def denoise_forwards(self) -> int:
        return sum(item.denoise_forwards for item in self.phase_executions)

    @property
    def clean_forwards(self) -> int:
        return sum(item.clean_forwards for item in self.phase_executions)

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_count": self.phase_count,
            "denoise_forwards": self.denoise_forwards,
            "clean_forwards": self.clean_forwards,
            "video_latent_offset": self.video_latent_offset,
            "audio_latent_offset": self.audio_latent_offset,
            "starting_history_tokens": self.starting_history_tokens,
            "retained_history_tokens": self.retained_history_tokens,
            "cross_request_kv_reused": self.cross_request_kv_reused,
            "request_index": self.request_index,
            "audio_kv_reset_applied": self.audio_kv_reset_applied,
            "audio_kv_reset_tokens": self.audio_kv_reset_tokens,
            "starting_history_audio_tokens": self.starting_history_audio_tokens,
            "starting_history_video_tokens": self.starting_history_video_tokens,
            "retained_history_audio_tokens": self.retained_history_audio_tokens,
            "retained_history_video_tokens": self.retained_history_video_tokens,
            "native_frame_offset": self.native_frame_offset,
            "published_native_frames": self.published_native_frames,
            "video_transport_prefix_latents": (self.video_transport_prefix_latents),
            "audio_transport_prefix_latents_per_channel": (
                self.audio_transport_prefix_latents_per_channel
            ),
            "published_video_latents": self.published_video_latents,
            "published_audio_latents_per_channel": (self.published_audio_latents_per_channel),
            "current_text_token_count": self.current_text_token_count,
            "text_time_start": self.text_time_start,
            "fixed_media_time_origin": self.fixed_media_time_origin,
            "position_capture": self.position_capture,
            "audio_teacher": self.audio_teacher,
            "attention_backend": self.attention_backend,
            "phases": [item.to_dict() for item in self.phase_executions],
        }


def _required_tensor(value: Any, name: str) -> Any:
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _resolve_spatial_latent_shape(
    positive: Any,
    *,
    frame_rows: int,
) -> tuple[int, int]:
    """Recover the exact target latent canvas from official absolute positions."""

    import torch

    position_ids = _required_tensor(
        positive.img_position_ids,
        "positive.img_position_ids",
    )
    target_info = positive.static_kwargs.get("img_pos_for_infer_output_info")
    if not isinstance(target_info, dict):
        raise TypeError("img_pos_for_infer_output_info must be a mapping")
    target_ids = _required_tensor(
        target_info.get("position_ids"),
        "img_pos_for_infer_output_info.position_ids",
    ).view(-1)
    if position_ids.ndim != 3 or int(position_ids.shape[0]) != 1:
        raise ValueError("official H3 img_position_ids must be [1, sequence, 3]")
    with torch.no_grad():
        target_grid = position_ids[0].index_select(
            0, target_ids.to(device=position_ids.device, dtype=torch.long)
        )
        latent_h = int(torch.unique(target_grid[:, 1]).numel()) * 2
        latent_w = int(torch.unique(target_grid[:, 2]).numel()) * 2
    if (latent_h // 2) * (latent_w // 2) != frame_rows:
        raise RuntimeError("official H3 spatial RoPE grid and target packed-row count disagree")
    return latent_h, latent_w


def _current_prompt_rope_start(
    *,
    text_len: int,
    media_time_origin: int,
    video_latent_offset: int,
) -> float:
    """Right-align the current prompt to the global request media origin."""

    return float(media_time_origin + video_temporal_position(video_latent_offset) - text_len)


def build_streaming_phase_packed_layout(
    *,
    positive: Any,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    text_token_tags: Any,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
) -> dict[str, Any]:
    """Build one future-free T2VA chunk on the global canonical timeline."""

    import torch

    from taomate_h3.model.packed_sequence import minimax_h3_packed_sequence

    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    text_len = int(prompt_embeds.shape[0])
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=phase.video_latent_count,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=phase.audio_latent_count,
        audio_channel=2,
    )
    text_pos = _required_tensor(packed.get("text_pos"), "text_pos").view(-1)
    tags = _required_tensor(packed.get("token_tags"), "token_tags")
    if list(text_token_tags.shape) != [text_len]:
        raise ValueError("text token tags do not match prompt embeddings")
    tags[text_pos] = text_token_tags.to(device=tags.device, dtype=tags.dtype)

    grid = _required_tensor(packed.get("img_position_ids"), "img_position_ids")
    img_pos = _required_tensor(packed.get("img_pos"), "img_pos").view(-1)
    target_pos = img_pos
    frame_rows = (latent_h // 2) * (latent_w // 2)
    if int(target_pos.numel()) != phase.video_latent_count * frame_rows:
        raise RuntimeError("chunk video rows do not match its latent geometry")

    prompt_start = _current_prompt_rope_start(
        text_len=text_len,
        media_time_origin=media_time_origin,
        video_latent_offset=video_latent_offset,
    )
    grid[text_pos, 0] = prompt_start + torch.arange(text_len, dtype=grid.dtype, device=grid.device)
    video_positions = video_temporal_positions(
        video_latent_offset + phase.video_latent_start,
        phase.video_latent_count,
    )
    video_times = torch.tensor(
        [float(media_time_origin + position) for position in video_positions],
        dtype=grid.dtype,
    )
    grid[target_pos, 0] = video_times.repeat_interleave(frame_rows)

    audio_pos = _required_tensor(packed.get("audio_pos"), "audio_pos").view(
        2, phase.audio_latent_count
    )
    audio_times = media_time_origin + torch.arange(
        audio_latent_offset + phase.audio_latent_start,
        audio_latent_offset + phase.audio_latent_stop,
        dtype=grid.dtype,
    )
    grid[audio_pos[0], 0] = audio_times
    grid[audio_pos[1], 0] = audio_times
    packed["_taomate_h3_position_capture"] = {
        "text_count": text_len,
        "text_time_start": prompt_start,
        "text_time_stop": prompt_start + text_len,
        "request_media_time_start": float(
            media_time_origin + video_temporal_position(video_latent_offset)
        ),
        "phase_video_time_start": float(video_times[0]),
        "phase_audio_time_start": float(audio_times[0]),
    }
    return packed


def _build_phase_branch(
    *,
    positive: Any,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    model: Any,
    device: Any,
    text_token_tags: Any,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
) -> Any:
    import torch

    packed = build_streaming_phase_packed_layout(
        positive=positive,
        phase=phase,
        latent_h=latent_h,
        latent_w=latent_w,
        text_token_tags=text_token_tags,
        media_time_origin=media_time_origin,
        video_latent_offset=video_latent_offset,
        audio_latent_offset=audio_latent_offset,
    )
    position_capture = packed.pop("_taomate_h3_position_capture")
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    branch = type(positive)(
        packed=packed,
        text_embeddings=prompt_embeds,
        token_tags=packed["token_tags"],
        device=device,
        parallel_context=model.parallel_context,
    )
    with torch.no_grad():
        branch.static_kwargs["rope_cache"] = model.build_rope_cache(
            branch.img_position_ids, device=device
        )
    branch._taomate_h3_position_capture = position_capture
    return branch


def _resolve_global_text_token_tags(positive: Any) -> Any:
    """Recover official text/vision AdaLN tags before rebuilding phase layouts."""

    import torch

    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    text_len = int(prompt_embeds.shape[0])
    local_tags = _required_tensor(
        positive.static_kwargs.get("block_token_tags"),
        "positive.static_kwargs['block_token_tags']",
    ).view(-1)
    seq_len = int(getattr(positive, "seq_len", 0))
    if seq_len <= 0:
        raise ValueError("official H3 branch sequence length is invalid")
    if int(local_tags.numel()) == seq_len:
        global_tags = local_tags
    else:
        from taomate_h3.distributed import ulysses_all_gather_rows

        global_tags = ulysses_all_gather_rows(
            local_tags.view(-1, 1), context=positive.parallel_context
        ).view(-1)
    if int(global_tags.numel()) != seq_len or text_len > seq_len:
        raise RuntimeError("official H3 global token-tag reconstruction differs")
    tags = global_tags[:text_len].detach().to(device="cpu", dtype=torch.long)
    if bool(((tags != 0) & (tags != 1)).any().item()):
        raise ValueError("H3 prompt rows contain unsupported streaming token tags")
    return tags


def _teacher_phase_audio_rows(
    rows: Any,
    *,
    phase: StreamPhase,
    total_audio_latents: int,
) -> Any:
    return (
        rows.view(2, total_audio_latents, -1)[:, phase.audio_latent_start : phase.audio_latent_stop]
        .contiguous()
        .view(-1, int(rows.shape[-1]))
    )


def _base_audio_teacher_step_callback(
    teacher: dict[str, Any],
    *,
    phase: StreamPhase,
    total_audio_latents: int,
) -> Callable[[int, Any, Any], None]:
    """Inject the request-wide Base10 milestone used by one Stage3 loop."""

    milestones = teacher["milestones"]

    def on_step(step_index: int, video_rows: Any, audio_rows: Any) -> None:
        milestone = milestones.get(step_index)
        if milestone is not None:
            teacher_rows = _teacher_phase_audio_rows(
                milestone,
                phase=phase,
                total_audio_latents=total_audio_latents,
            )
            audio_rows.copy_(teacher_rows)

    return on_step


@contextmanager
def _native_attention_metadata(
    branch: Any,
) -> Iterator[None]:
    """Bind full-sequence modality tags and target-media commit rows."""

    import torch

    from taomate_h3.distributed import ulysses_all_gather_rows
    from taomate_h3.model.layers import native_stream_attention_metadata

    tags = _required_tensor(branch.static_kwargs.get("block_token_tags"), "block_token_tags").view(
        -1
    )
    if int(tags.numel()) != int(branch.seq_len):
        tags = ulysses_all_gather_rows(tags.view(-1, 1), context=branch.parallel_context).view(-1)
    if int(tags.numel()) != int(branch.seq_len):
        raise RuntimeError("native H3 streaming token tags do not cover the sequence")
    mask = torch.zeros(int(branch.seq_len), dtype=torch.bool, device=tags.device)
    mask.index_fill_(0, branch.img_target_seq_idx.to(tags.device), True)
    mask.index_fill_(0, branch.audio_target_seq_idx.to(tags.device), True)
    with native_stream_attention_metadata(tags, mask):
        yield


def _phase_initial_rows(
    *,
    initial_video_rows: Any,
    initial_audio_rows: Any,
    phase: StreamPhase,
    frame_rows: int,
    total_audio_latents: int,
) -> tuple[Any, Any]:
    video = initial_video_rows[
        phase.video_latent_start * frame_rows : phase.video_latent_stop * frame_rows
    ].contiguous()
    audio = initial_audio_rows.view(2, total_audio_latents, -1)[
        :, phase.audio_latent_start : phase.audio_latent_stop
    ].contiguous()
    return video, audio.view(-1, int(audio.shape[-1]))


def _join_phase_audio(rows: list[Any], phases: tuple[StreamPhase, ...]) -> Any:
    import torch

    by_channel: list[list[Any]] = [[], []]
    for value, phase in zip(rows, phases):
        current = value.view(2, phase.audio_latent_count, -1)
        by_channel[0].append(current[0])
        by_channel[1].append(current[1])
    return torch.cat(
        (torch.cat(by_channel[0], dim=0), torch.cat(by_channel[1], dim=0)),
        dim=0,
    )


class H3StreamingRuntime:
    """Split each T2VA request into small phases with persistent clean AV-KV."""

    def __init__(self, *, base10_teacher: ExternalBase10TeacherArtifact) -> None:
        self.plan = direct_5s_plan()
        self.executions: list[StreamingExecution] = []
        self._running = False
        self._cache: CleanAVKVCache | None = None
        self._media_time_origin: int | None = None
        self._request_index = 0
        self._native_frame_offset = 0
        self._video_latent_offset = 0
        self._audio_latent_offset = 0
        self._canonical_transport_video_rows: Any | None = None
        self._canonical_transport_audio_rows: Any | None = None
        self._prefix_renorm_video_anchor: tuple[Any, Any] | None = None
        self._base10_teacher = base10_teacher
        self._dit_cuda_events: list[tuple[Any, Any]] = []
        self._dit_noisy_calls = 0
        self._dit_clean_calls = 0

    def release_retained_state(self) -> None:
        """Release cross-request CUDA state after the final generated chunk."""

        if self._running:
            raise RuntimeError("cannot release streaming state during generation")
        if self._cache is not None:
            self._cache.clear()
            self._cache = None
        self._canonical_transport_video_rows = None
        self._canonical_transport_audio_rows = None
        self._prefix_renorm_video_anchor = None

    def _renorm_clean_video_rows(self, rows: Any) -> Any:
        """Match generated blocks to persistent block-zero statistics."""

        current = rows.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self._prefix_renorm_video_anchor is None:
            self._prefix_renorm_video_anchor = (mean, std)
            return rows
        anchor_mean, anchor_std = self._prefix_renorm_video_anchor
        normalized = (current - mean).div(std).mul(anchor_std).add(anchor_mean)
        return normalized.to(dtype=rows.dtype)

    def dit_timing_receipt(self) -> dict[str, float | int]:
        """Return synchronized Transformer-only CUDA time across all ranks."""

        import torch

        torch.cuda.synchronize()
        local_seconds = sum(
            float(start.elapsed_time(stop)) for start, stop in self._dit_cuda_events
        ) / 1000.0
        maximum = torch.tensor(local_seconds, dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        return {
            "noisy_calls": self._dit_noisy_calls,
            "clean_kv_calls": self._dit_clean_calls,
            "total_calls": self._dit_noisy_calls + self._dit_clean_calls,
            "cuda_seconds": float(maximum.item()),
        }

    def run(self, **kwargs: Any) -> Any:
        import torch

        from taomate_h3.model.denoise import minimax_h3_denoise_loop

        if self._running:
            raise RuntimeError("one H3 streaming runtime cannot run concurrent requests")
        self._running = True
        try:
            model = kwargs["model"]
            positive = kwargs["positive"]
            initial_video = _required_tensor(kwargs["initial_video_rows"], "initial_video_rows")
            initial_audio = _required_tensor(kwargs["initial_audio_rows"], "initial_audio_rows")
            request_index = self._request_index
            canonical_continuation = request_index > 0
            active_plan = (
                canonical_continuation_plan(
                    self.plan,
                    request_index=request_index,
                )
                if canonical_continuation
                else self.plan
            )
            official_video_latents = self.plan.phases[-1].video_latent_stop
            official_audio_latents = self.plan.phases[-1].audio_latent_stop
            total_video_latents = active_plan.phases[-1].video_latent_stop
            total_audio_latents = active_plan.phases[-1].audio_latent_stop
            video_transport_prefix_latents = official_video_latents - total_video_latents
            audio_transport_prefix_latents = official_audio_latents - total_audio_latents
            target_video_rows = int(initial_video.shape[0])
            if target_video_rows <= 0 or target_video_rows % official_video_latents:
                raise ValueError("request target video rows do not match the stream plan")
            frame_rows = target_video_rows // official_video_latents
            if list(initial_audio.shape) != [2 * official_audio_latents, 32]:
                raise ValueError(
                    "request audio rows do not match cumulative 40 Hz phase boundaries"
                )
            latent_h, latent_w = _resolve_spatial_latent_shape(
                positive,
                frame_rows=frame_rows,
            )
            if list(initial_video.shape) != [official_video_latents * frame_rows, 96]:
                raise ValueError("direct streaming accepts T2VA target rows only")
            request_video_rows = initial_video
            transport_video_rows = None
            transport_audio_rows = None
            if canonical_continuation:
                transport_video_rows = self._canonical_transport_video_rows
                transport_audio_rows = self._canonical_transport_audio_rows
                if transport_video_rows is None or list(transport_video_rows.shape) != [
                    video_transport_prefix_latents * frame_rows,
                    96,
                ]:
                    raise RuntimeError(
                        "canonical continuation has no exact prior video transport prefix"
                    )
                if transport_audio_rows is None:
                    raise RuntimeError("canonical continuation has no prior audio transport prefix")
                previous_audio_t = int(transport_audio_rows.shape[0]) // 2
                if list(transport_audio_rows.shape) != [2 * previous_audio_t, 32] or (
                    previous_audio_t < audio_transport_prefix_latents
                ):
                    raise RuntimeError(
                        "canonical continuation has no exact prior audio transport prefix"
                    )
                request_video_rows = request_video_rows[
                    video_transport_prefix_latents * frame_rows :
                ].contiguous()
                initial_audio_by_channel = initial_audio.view(2, official_audio_latents, -1)
                initial_audio = (
                    initial_audio_by_channel[:, audio_transport_prefix_latents:]
                    .contiguous()
                    .view(-1, int(initial_audio.shape[-1]))
                )
            text_token_tags = _resolve_global_text_token_tags(positive)
            text_len = int(
                _required_tensor(
                    positive.static_kwargs.get("prompt_embeds"),
                    "positive.static_kwargs['prompt_embeds']",
                ).shape[0]
            )
            media_time_origin = (
                text_len if self._media_time_origin is None else self._media_time_origin
            )
            device = initial_video.device
            if device.type != "cpu":
                raise ValueError("official H3 initial noise must enter denoise from CPU")
            cuda_device = torch.device("cuda")

            sigmas_video = list(kwargs["sigmas_video"])
            sigmas_audio = list(kwargs["sigmas_audio"])
            if len(sigmas_video) != len(sigmas_audio) or len(sigmas_video) < 2:
                raise ValueError("H3 video/audio sigma schedules must match")
            phase_video_outputs: list[Any] = []
            phase_audio_outputs: list[Any] = []
            phase_records: list[StreamingPhaseExecution] = []
            position_capture: dict[str, object] = {"phases": []}
            phases = active_plan.phases
            cache = self._cache
            if cache is None:
                cache = CleanAVKVCache(kv_contract_from_model(model))
                self._cache = cache
            audio_kv_reset_applied = (
                request_index > 0
                and request_index % DIRECT_POLICY.audio_kv_reset_window_requests == 0
            )
            audio_kv_reset_tokens = cache.drop_audio_history() if audio_kv_reset_applied else 0
            starting_history_tokens = cache.history_tokens
            starting_history_audio_tokens = cache.history_audio_tokens
            starting_history_video_tokens = cache.history_video_tokens
            reused_kv = starting_history_tokens > 0
            current_prompt_rope_start = _current_prompt_rope_start(
                text_len=text_len,
                media_time_origin=media_time_origin,
                video_latent_offset=self._video_latent_offset,
            )
            audio_teacher = self._base10_teacher.load_request(
                torch,
                request_index=request_index,
                audio_latent_count=total_audio_latents,
                device=cuda_device,
            )

            with H3StreamingSession(
                active_plan,
                cache=cache,
            ) as session:
                for phase in phases:
                    branch = _build_phase_branch(
                        positive=positive,
                        phase=phase,
                        latent_h=latent_h,
                        latent_w=latent_w,
                        model=model,
                        device=cuda_device,
                        text_token_tags=text_token_tags,
                        media_time_origin=media_time_origin,
                        video_latent_offset=self._video_latent_offset,
                        audio_latent_offset=self._audio_latent_offset,
                    )
                    position_capture["phases"].append(
                        {
                            "phase_index": phase.index,
                            **dict(branch._taomate_h3_position_capture),
                        }
                    )
                    phase_video, phase_audio = _phase_initial_rows(
                        initial_video_rows=request_video_rows,
                        initial_audio_rows=initial_audio,
                        phase=phase,
                        frame_rows=frame_rows,
                        total_audio_latents=total_audio_latents,
                    )

                    def noisy_forward(
                        active_model: Any,
                        call_kwargs: dict[str, Any],
                        step_index: int,
                        active_phase_index: int = phase.index,
                        active_branch: Any = branch,
                    ) -> tuple[Any, Any]:
                        del step_index
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_stop = torch.cuda.Event(enable_timing=True)
                        with (
                            session.noisy_step(active_phase_index),
                            _native_attention_metadata(active_branch),
                        ):
                            cuda_start.record()
                            result = active_model(**call_kwargs)
                            cuda_stop.record()
                        self._dit_cuda_events.append((cuda_start, cuda_stop))
                        self._dit_noisy_calls += 1
                        return result

                    phase_on_step = _base_audio_teacher_step_callback(
                        audio_teacher,
                        phase=phase,
                        total_audio_latents=total_audio_latents,
                    )

                    generated_video, generated_audio = minimax_h3_denoise_loop(
                        model=model,
                        model_forward=noisy_forward,
                        positive=branch,
                        initial_video_rows=phase_video,
                        initial_audio_rows=phase_audio,
                        sigmas_video=sigmas_video,
                        sigmas_audio=sigmas_audio,
                        device=cuda_device,
                        on_step=phase_on_step,
                    )

                    clean_phase_video = self._renorm_clean_video_rows(
                        generated_video,
                    )
                    generated_video.copy_(clean_phase_video)
                    history_before_commit = session.cache.history_tokens

                    clean_timestep = branch.prepare_timestep_plan(
                        video_timesteps=[1.0],
                        audio_timesteps=[1.0],
                    )[0]
                    clean_kwargs = branch.forward_kwargs(
                        video_rows=generated_video,
                        audio_rows=generated_audio,
                        step_timesteps=clean_timestep,
                    )
                    with (
                        session.clean_commit(phase.index),
                        _native_attention_metadata(branch),
                        torch.no_grad(),
                    ):
                        clean_kwargs["streaming_cache_only"] = True
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_stop = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                        model(**clean_kwargs)
                        cuda_stop.record()
                        self._dit_cuda_events.append((cuda_start, cuda_stop))
                        self._dit_clean_calls += 1

                    expected_history_tokens = history_before_commit + (
                        phase.video_latent_count * frame_rows + 2 * phase.audio_latent_count
                    )
                    if session.cache.history_tokens != expected_history_tokens:
                        raise RuntimeError(
                            "clean H3 KV history does not match the committed "
                            f"AV phase boundary: {session.cache.history_tokens} != "
                            f"{expected_history_tokens}"
                        )

                    session.cache.retain_sink_and_recent_commits()
                    phase_video_outputs.append(clean_phase_video)
                    phase_audio_outputs.append(generated_audio)
                    phase_records.append(
                        StreamingPhaseExecution(
                            phase_index=phase.index,
                            frame_start=phase.frame_start,
                            frame_stop=phase.frame_stop,
                            video_latent_start=phase.video_latent_start,
                            video_latent_stop=phase.video_latent_stop,
                            audio_latent_start=phase.audio_latent_start,
                            audio_latent_stop=phase.audio_latent_stop,
                            denoise_forwards=len(sigmas_video) - 1,
                            clean_forwards=1,
                            committed_history_tokens=session.cache.history_tokens,
                        )
                    )

            attention_receipt = session.hook.receipt()
            joined_video = torch.cat(phase_video_outputs, dim=0)
            joined_audio = _join_phase_audio(phase_audio_outputs, phases)
            if joined_video.shape != request_video_rows.shape:
                raise RuntimeError("reassembled streaming video rows changed request shape")
            if joined_audio.shape != initial_audio.shape:
                raise RuntimeError("reassembled streaming audio rows changed request shape")
            final_teacher = audio_teacher["milestones"][2]
            if not torch.equal(joined_audio, final_teacher):
                raise RuntimeError(
                    "published streaming audio differs from Base teacher clean latent"
                )
            audio_teacher_receipt = dict(audio_teacher["receipt"])
            audio_teacher_receipt["published_clean_audio_exact_match"] = True
            returned_video = joined_video
            returned_audio = joined_audio
            if canonical_continuation:
                returned_video = torch.cat(
                    (
                        transport_video_rows.to(
                            device=joined_video.device,
                            dtype=joined_video.dtype,
                        ),
                        joined_video,
                    ),
                    dim=0,
                )
                previous_audio = transport_audio_rows.view(2, -1, int(joined_audio.shape[-1]))[
                    :, -audio_transport_prefix_latents:
                ]
                returned_audio = (
                    torch.cat(
                        (
                            previous_audio.to(
                                device=joined_audio.device,
                                dtype=joined_audio.dtype,
                            ),
                            joined_audio.view(2, total_audio_latents, int(joined_audio.shape[-1])),
                        ),
                        dim=1,
                    )
                    .contiguous()
                    .view(-1, int(joined_audio.shape[-1]))
                )
            self._canonical_transport_video_rows = (
                joined_video[-2 * frame_rows :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            audio_by_channel = joined_audio.view(
                2, total_audio_latents, int(joined_audio.shape[-1])
            )
            self._canonical_transport_audio_rows = (
                audio_by_channel[:, -min(9, total_audio_latents) :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .view(-1, int(joined_audio.shape[-1]))
            )
            retained_history_tokens = cache.history_tokens
            retained_history_audio_tokens = cache.history_audio_tokens
            retained_history_video_tokens = cache.history_video_tokens
            self._media_time_origin = media_time_origin
            execution = StreamingExecution(
                phase_executions=tuple(phase_records),
                video_latent_offset=self._video_latent_offset,
                audio_latent_offset=self._audio_latent_offset,
                starting_history_tokens=starting_history_tokens,
                retained_history_tokens=retained_history_tokens,
                cross_request_kv_reused=reused_kv,
                request_index=request_index,
                audio_kv_reset_applied=audio_kv_reset_applied,
                audio_kv_reset_tokens=audio_kv_reset_tokens,
                starting_history_audio_tokens=starting_history_audio_tokens,
                starting_history_video_tokens=starting_history_video_tokens,
                retained_history_audio_tokens=retained_history_audio_tokens,
                retained_history_video_tokens=retained_history_video_tokens,
                native_frame_offset=self._native_frame_offset,
                published_native_frames=active_plan.native_frame_count,
                video_transport_prefix_latents=(video_transport_prefix_latents),
                audio_transport_prefix_latents_per_channel=(audio_transport_prefix_latents),
                published_video_latents=total_video_latents,
                published_audio_latents_per_channel=total_audio_latents,
                current_text_token_count=text_len,
                text_time_start=current_prompt_rope_start,
                fixed_media_time_origin=media_time_origin,
                position_capture=position_capture,
                audio_teacher=audio_teacher_receipt,
                attention_backend=attention_receipt,
            )
            self.executions.append(execution)
            self._video_latent_offset += total_video_latents
            self._audio_latent_offset += total_audio_latents
            self._native_frame_offset += active_plan.native_frame_count
            self._request_index += 1
            if returned_video.shape != initial_video.shape:
                raise RuntimeError("streaming output no longer matches official request rows")
            if list(returned_audio.shape) != [2 * official_audio_latents, 32]:
                raise RuntimeError("streaming audio output no longer matches official request rows")
            return returned_video, returned_audio
        finally:
            self._running = False


__all__ = [
    "H3StreamingRuntime",
]
