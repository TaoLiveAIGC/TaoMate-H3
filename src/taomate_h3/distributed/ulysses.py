# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

import torch

from .parallel import ParallelContext


def _all_to_all_swap(
    value: torch.Tensor,
    *,
    world_size: int,
    group: object,
    sequence_to_heads: bool,
) -> torch.Tensor:
    """Swap sequence and head shards with one contiguous all-to-all."""

    if sequence_to_heads:
        if value.ndim < 3 or int(value.shape[-2]) % world_size:
            raise ValueError("Ulysses input heads must be divisible by its world size")
        prefix_ndim = value.ndim - 3
        local_heads = int(value.shape[-2]) // world_size
        split_shape = (*value.shape[:-2], world_size, local_heads, value.shape[-1])
        # Put the destination/head-shard axis first for all_to_all_single.
        inputs = value.contiguous().view(split_shape).movedim(prefix_ndim + 1, 0).contiguous()
        outputs = torch.empty_like(inputs)
        torch.distributed.all_to_all_single(outputs, inputs, group=group)
        # Incoming chunks are consecutive source sequence shards.
        outputs = outputs.movedim(0, prefix_ndim).contiguous()
        result_shape = (
            *value.shape[:-3],
            value.shape[-3] * world_size,
            local_heads,
            value.shape[-1],
        )
        result = outputs.view(result_shape)
    else:
        if value.ndim < 3 or int(value.shape[-3]) % world_size:
            raise ValueError("Ulysses global rows must be divisible by its world size")
        local_rows = int(value.shape[-3]) // world_size
        split_shape = (*value.shape[:-3], world_size, local_rows, *value.shape[-2:])
        inputs = value.contiguous().view(split_shape).movedim(-4, 0).contiguous()
        outputs = torch.empty_like(inputs)
        torch.distributed.all_to_all_single(outputs, inputs, group=group)
        # Incoming chunks are different head shards for this local sequence.
        outputs = outputs.movedim(0, -3).contiguous()
        result_shape = (
            *value.shape[:-3],
            local_rows,
            value.shape[-2] * world_size,
            value.shape[-1],
        )
        result = outputs.view(result_shape)
    return result


def _swap(
    value: torch.Tensor,
    *,
    context: ParallelContext,
    sequence_to_heads: bool,
) -> torch.Tensor:
    world_size = context.ulysses_world_size
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Ulysses all-to-all requires initialized torch.distributed")
    if context.ulysses_group is None:
        raise RuntimeError("Ulysses process group is absent")
    return _all_to_all_swap(
        value,
        world_size=world_size,
        group=context.ulysses_group,
        sequence_to_heads=sequence_to_heads,
    )


def heads_to_sequence(
    value: torch.Tensor,
    *,
    context: ParallelContext,
) -> torch.Tensor:
    """Inverse of :func:`sequence_to_heads`."""

    return _swap(value, context=context, sequence_to_heads=False)


def packed_qkv_sequence_to_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    context: ParallelContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/K/V into one Ulysses collective, then restore three views."""

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Ulysses packed Q/K/V shapes differ")
    packed = torch.stack((q, k, v), dim=0)
    exchanged = _swap(packed, context=context, sequence_to_heads=True)
    return tuple(exchanged.unbind(dim=0))  # type: ignore[return-value]


def merged_qkv_sequence_to_heads(
    value: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    context: ParallelContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Send an already merged QKV allocation directly through Ulysses."""

    world_size = context.ulysses_world_size
    tokens = int(value.shape[0])
    if int(value.shape[-1]) != 3 * num_heads * head_dim:
        raise ValueError("merged QKV width differs")
    packed = value.view(tokens, 3, num_heads, head_dim)
    if world_size == 1:
        return tuple(packed.unbind(dim=1))  # type: ignore[return-value]
    if num_heads % world_size:
        raise ValueError("merged QKV heads are not divisible by Ulysses")
    local_heads = num_heads // world_size
    inputs = (
        packed.view(tokens, 3, world_size, local_heads, head_dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    outputs = torch.empty_like(inputs)
    torch.distributed.all_to_all_single(outputs, inputs, group=context.ulysses_group)
    exchanged = (
        outputs.movedim(0, 1).contiguous().view(3, tokens * world_size, local_heads, head_dim)
    )
    return tuple(exchanged.unbind(dim=0))  # type: ignore[return-value]


__all__ = [
    "heads_to_sequence",
    "merged_qkv_sequence_to_heads",
    "packed_qkv_sequence_to_heads",
]
