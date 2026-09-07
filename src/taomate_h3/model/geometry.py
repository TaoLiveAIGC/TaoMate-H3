# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Native geometry for direct MiniMax H3 requests."""

from __future__ import annotations

from dataclasses import dataclass

_CANVAS_MULTIPLE = 32
_SUPPORTED_SHORT_EDGES = (480, 768, 1088)


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    video_latent_t: int
    video_latent_h: int
    video_latent_w: int
    audio_latent_t: int


def direct_5s_geometry(*, width: int, height: int) -> MiniMaxH3Geometry:
    """Resolve the fixed five-second direct request geometry."""

    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("width and height must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0 or width % _CANVAS_MULTIPLE or height % _CANVAS_MULTIPLE:
        raise ValueError("width and height must be positive multiples of 32")
    if min(width, height) not in _SUPPORTED_SHORT_EDGES:
        raise ValueError("direct inference requires a 480-, 768-, or 1088-pixel short edge")

    return MiniMaxH3Geometry(
        video_latent_t=37,
        video_latent_h=height // 16,
        video_latent_w=width // 16,
        audio_latent_t=207,
    )


__all__ = ["MiniMaxH3Geometry", "direct_5s_geometry"]
