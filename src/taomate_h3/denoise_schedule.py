# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

import math

DISTILLED_STATE_INDICES = (0, 16, 33, 49)


def time_shift_sigmas(*, num_steps: int = 50, shift_scale: float) -> list[float]:
    if num_steps <= 1 or not math.isfinite(shift_scale) or shift_scale <= 0:
        raise ValueError("sigma schedule requires num_steps > 1 and shift_scale > 0")
    import torch

    base = torch.linspace(1.0, 0.0, num_steps, dtype=torch.float32, device="cpu")
    shifted = shift_scale * base / (1.0 + (shift_scale - 1.0) * base)
    shifted = torch.unique_consecutive(shifted)
    if int(shifted.numel()) != num_steps:
        raise ValueError("shifted sigma schedule changed cardinality")
    return [float(item) for item in shifted.tolist()]


def select_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float,
    state_indices: tuple[int, ...] | None = None,
) -> list[float]:
    schedule = time_shift_sigmas(num_steps=num_steps, shift_scale=shift_scale)
    if state_indices is None:
        return schedule
    indices = tuple(state_indices)
    if (
        len(indices) < 2
        or indices[0] != 0
        or indices[-1] != num_steps - 1
        or tuple(sorted(set(indices))) != indices
    ):
        raise ValueError("retained sigma indices must be sorted, unique, and span the schedule")
    return [schedule[index] for index in indices]


__all__ = [
    "DISTILLED_STATE_INDICES",
    "select_time_shift_sigmas",
    "time_shift_sigmas",
]
