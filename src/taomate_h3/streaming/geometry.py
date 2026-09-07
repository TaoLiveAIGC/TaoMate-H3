# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction

VIDEO_FPS = 24
AUDIO_LATENT_RATE = 40
VIDEO_PREFIX_FRAMES = 5
VIDEO_PREFIX_LATENTS = 2
VIDEO_GROUP_FRAMES = 17
VIDEO_GROUP_LATENTS = 5


def _round_half_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def _audio_latent_boundary(frame: int) -> int:
    return _round_half_even(Fraction(frame * AUDIO_LATENT_RATE, VIDEO_FPS))


def canonical_continuation_plan(
    base_plan: StreamPlan,
    *,
    request_index: int,
) -> StreamPlan:
    """Build the true steady-state target geometry after request zero.

    The standalone H3 request owns a five-frame/two-latent affine prefix.  A
    continuation does not generate that prefix again: it advances seven native
    17-frame groups (35 video latents), while audio boundaries are rounded on
    the global frame timeline.  Consequently the per-request audio target is
    198 or 199 latents rather than a repeated request-local 207.
    """

    if not isinstance(base_plan, StreamPlan):
        raise TypeError("base_plan must be a StreamPlan")
    if isinstance(request_index, bool) or not isinstance(request_index, int):
        raise TypeError("request_index must be an integer")
    if request_index <= 0:
        raise ValueError("canonical continuation requires request_index > 0")
    if base_plan.native_frame_count <= VIDEO_PREFIX_FRAMES:
        raise ValueError("base stream has no steady continuation geometry")

    steady_frame_count = base_plan.native_frame_count - VIDEO_PREFIX_FRAMES
    global_frame_start = base_plan.native_frame_count + (request_index - 1) * steady_frame_count
    global_audio_start = _audio_latent_boundary(global_frame_start)

    phases: list[StreamPhase] = []
    frame_stop = 0
    video_stop = 0
    for index, base_phase in enumerate(base_plan.phases):
        frame_start = frame_stop
        video_start = video_stop
        frame_stop += base_phase.group_count * VIDEO_GROUP_FRAMES
        video_stop += base_phase.group_count * VIDEO_GROUP_LATENTS
        phases.append(
            StreamPhase(
                index=index,
                group_count=base_phase.group_count,
                frame_start=frame_start,
                frame_stop=frame_stop,
                video_latent_start=video_start,
                video_latent_stop=video_stop,
                audio_latent_start=(
                    _audio_latent_boundary(global_frame_start + frame_start) - global_audio_start
                ),
                audio_latent_stop=(
                    _audio_latent_boundary(global_frame_start + frame_stop) - global_audio_start
                ),
            )
        )
    return StreamPlan(
        native_frame_count=steady_frame_count,
        phases=tuple(phases),
    )


def video_temporal_position(latent_index: int) -> Fraction:
    """Absolute H3 RoPE time coordinate before the request text-origin offset."""

    if isinstance(latent_index, bool) or not isinstance(latent_index, int):
        raise TypeError("latent_index must be an integer")
    if latent_index < 0:
        raise ValueError("latent_index must be non-negative")
    weights = (1, 4, 4, 4, 4)
    return sum(
        (Fraction(5, 3) * weights[index % len(weights)] for index in range(latent_index)),
        start=Fraction(0),
    )


def video_temporal_positions(start: int, count: int) -> tuple[Fraction, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    return tuple(video_temporal_position(index) for index in range(start, start + count))


@dataclass(frozen=True)
class StreamPhase:
    index: int
    group_count: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def duration(self) -> Fraction:
        return Fraction(self.frame_count, VIDEO_FPS)

    @property
    def video_latent_count(self) -> int:
        return self.video_latent_stop - self.video_latent_start

    @property
    def audio_latent_count(self) -> int:
        return self.audio_latent_stop - self.audio_latent_start

    @property
    def video_rope_start(self) -> Fraction:
        return video_temporal_position(self.video_latent_start)

    @property
    def audio_rope_start(self) -> int:
        return self.audio_latent_start

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.update(
            duration_numerator=self.duration.numerator,
            duration_denominator=self.duration.denominator,
            duration_seconds=float(self.duration),
            video_rope_start_numerator=self.video_rope_start.numerator,
            video_rope_start_denominator=self.video_rope_start.denominator,
            audio_rope_start=self.audio_rope_start,
        )
        return value


@dataclass(frozen=True)
class StreamPlan:
    native_frame_count: int
    phases: tuple[StreamPhase, ...]

    @property
    def duration(self) -> Fraction:
        return Fraction(self.native_frame_count, VIDEO_FPS)

    def to_dict(self) -> dict[str, object]:
        return {
            "native_frame_count": self.native_frame_count,
            "duration_seconds": float(self.duration),
            "phase_count": len(self.phases),
            "phases": [phase.to_dict() for phase in self.phases],
        }


def direct_5s_plan() -> StreamPlan:
    """Return the validated four-chunk plan for one five-second request."""

    native_frame_count = 124
    phases: list[StreamPhase] = []
    frame_stop = 0
    video_stop = 0
    for index, groups in enumerate((2, 2, 2, 1)):
        frame_start = frame_stop
        video_start = video_stop
        frame_stop += groups * VIDEO_GROUP_FRAMES
        video_stop += groups * VIDEO_GROUP_LATENTS
        if index == 0:
            frame_stop += VIDEO_PREFIX_FRAMES
            video_stop += VIDEO_PREFIX_LATENTS
        phases.append(
            StreamPhase(
                index=index,
                group_count=groups,
                frame_start=frame_start,
                frame_stop=frame_stop,
                video_latent_start=video_start,
                video_latent_stop=video_stop,
                audio_latent_start=_audio_latent_boundary(frame_start),
                audio_latent_stop=_audio_latent_boundary(frame_stop),
            )
        )
    return StreamPlan(native_frame_count=native_frame_count, phases=tuple(phases))


__all__ = [
    "StreamPhase",
    "StreamPlan",
    "canonical_continuation_plan",
    "direct_5s_plan",
    "video_temporal_position",
    "video_temporal_positions",
]
