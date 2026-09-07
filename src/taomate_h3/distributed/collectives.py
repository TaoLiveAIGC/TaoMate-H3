# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

from typing import Any

import torch

from .parallel import ParallelContext


def _require_distributed(group: Any, name: str) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(f"{name} collective requires initialized torch.distributed")
    if group is None:
        raise RuntimeError(f"{name} process group is absent")


def all_reduce_sum(
    value: torch.Tensor,
    *,
    group: Any,
) -> torch.Tensor:
    _require_distributed(group, "all-reduce")
    output = value.clone()
    torch.distributed.all_reduce(output, group=group)
    return output


def all_reduce_sum_inplace_(value: torch.Tensor, *, group: Any) -> torch.Tensor:
    """Sum an inference-owned temporary without allocating a clone."""

    _require_distributed(group, "all-reduce")
    torch.distributed.all_reduce(value, group=group)
    return value


def all_gather_concat(
    value: torch.Tensor,
    *,
    dim: int,
    group: Any,
    world_size: int,
) -> torch.Tensor:
    _require_distributed(group, "all-gather")
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, value.contiguous(), group=group)
    return torch.cat(tuple(gathered), dim=dim)


def tp_all_reduce(value: torch.Tensor, *, context: ParallelContext) -> torch.Tensor:
    return all_reduce_sum(
        value,
        group=context.tp_group,
    )


def tp_all_reduce_inplace_(value: torch.Tensor, *, context: ParallelContext) -> torch.Tensor:
    return all_reduce_sum_inplace_(value, group=context.tp_group)


def tp_all_gather(
    value: torch.Tensor,
    *,
    dim: int = -1,
    context: ParallelContext,
) -> torch.Tensor:
    return all_gather_concat(
        value,
        dim=dim,
        group=context.tp_group,
        world_size=context.tp_world_size,
    )


def ulysses_all_gather_rows(
    value: torch.Tensor,
    *,
    context: ParallelContext,
) -> torch.Tensor:
    return all_gather_concat(
        value,
        dim=0,
        group=context.ulysses_group,
        world_size=context.ulysses_world_size,
    )


__all__ = [
    "all_gather_concat",
    "all_reduce_sum",
    "all_reduce_sum_inplace_",
    "tp_all_gather",
    "tp_all_reduce",
    "tp_all_reduce_inplace_",
    "ulysses_all_gather_rows",
]
