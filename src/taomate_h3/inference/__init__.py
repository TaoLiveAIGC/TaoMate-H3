# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Inference helpers for direct generation."""

from .base10_teacher import (
    BASE10_TEACHER_STATE_NUMBERS,
    STAGE3_TARGET_STATE_INDICES,
    Base10TeacherArtifactError,
    ExternalBase10TeacherArtifact,
)

__all__ = [
    "BASE10_TEACHER_STATE_NUMBERS",
    "STAGE3_TARGET_STATE_INDICES",
    "Base10TeacherArtifactError",
    "ExternalBase10TeacherArtifact",
]
