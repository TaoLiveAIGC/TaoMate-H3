# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Exact video/audio timing transforms for direct H3 publication."""

from __future__ import annotations


def endpoint_preserving_video_filter(
    *,
    native_frames: int,
    published_frames: int,
    fps: int = 24,
) -> str:
    """Compress the complete native request onto the published frame timeline."""

    if native_frames <= 1 or published_frames <= 1:
        raise ValueError("video timing requires at least two native and published frames")
    return (
        f"trim=end_frame={native_frames},"
        f"setpts=(PTS-STARTPTS)*{published_frames}/{native_frames},"
        f"fps={fps}:round=near"
    )


def exact_audio_delivery_filter(
    *,
    native_samples: int,
    published_samples: int,
    sample_rate: int = 32_000,
) -> str:
    """Retimestamp the complete native waveform to an exact delivery sample count."""

    if native_samples <= 0 or published_samples <= 0:
        raise ValueError("audio timing requires positive sample counts")
    return (
        f"atempo={native_samples}/{published_samples},"
        f"aresample={sample_rate},"
        f"apad=whole_len={published_samples},"
        f"atrim=end_sample={published_samples},"
        "asetpts=N/SR/TB"
    )


__all__ = [
    "endpoint_preserving_video_filter",
    "exact_audio_delivery_filter",
]
