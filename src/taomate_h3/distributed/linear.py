# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collectives import tp_all_gather, tp_all_reduce, tp_all_reduce_inplace_
from .parallel import ParallelContext


def _module_linear(
    module: nn.Module,
    inputs: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    scale = getattr(module, "_cutlass_w8a8_weight_scale", None)
    if scale is None:
        return F.linear(inputs, module.weight, bias)
    from taomate_h3.inference.w8a8 import cutlass_w8a8_linear

    return cutlass_w8a8_linear(
        inputs,
        module.weight,
        scale,
        bias=bias,
        workspace=getattr(module, "_cutlass_w8a8_workspace", None),
        weight_t=getattr(module, "_cutlass_w8a8_weight_t", None),
        weight_scale_t=getattr(module, "_cutlass_w8a8_weight_scale_t", None),
    )


class ColumnParallelLinear(nn.Module):
    """Output-column sharded linear layer with checkpoint-compatible metadata."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool,
        gather_output: bool,
        dtype: torch.dtype,
        context: ParallelContext,
    ) -> None:
        super().__init__()
        self.parallel_context = context
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.tp_size = self.parallel_context.tp_world_size
        self.tp_rank = self.parallel_context.tp_rank
        self.gather_output = bool(gather_output)
        if self.output_size % self.tp_size:
            raise ValueError(
                f"column output {self.output_size} is not divisible by TP={self.tp_size}"
            )
        self.local_output_size = self.output_size // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.local_output_size, self.input_size, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.empty(self.local_output_size, dtype=dtype)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = _module_linear(self, inputs, self.bias)
        if self.gather_output:
            output = tp_all_gather(output, context=self.parallel_context)
        return output

    def full_weight_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size, self.input_size):
            raise ValueError("full column-parallel weight shape differs")
        return value.narrow(
            0,
            self.tp_rank * self.local_output_size,
            self.local_output_size,
        ).contiguous()

    def full_bias_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size,):
            raise ValueError("full column-parallel bias shape differs")
        return value.narrow(
            0,
            self.tp_rank * self.local_output_size,
            self.local_output_size,
        ).contiguous()


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Column sharding applied independently to fused logical projections."""

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        *,
        bias: bool,
        gather_output: bool,
        dtype: torch.dtype,
        context: ParallelContext,
    ) -> None:
        self.output_sizes = tuple(int(value) for value in output_sizes)
        if not self.output_sizes or any(value <= 0 for value in self.output_sizes):
            raise ValueError("merged column output sizes must be positive")
        if any(value % context.tp_world_size for value in self.output_sizes):
            raise ValueError("every merged output must be divisible by TP")
        super().__init__(
            input_size,
            sum(self.output_sizes),
            bias=bias,
            gather_output=gather_output,
            dtype=dtype,
            context=context,
        )
        self.local_output_sizes = tuple(value // self.tp_size for value in self.output_sizes)

    def full_weight_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size, self.input_size):
            raise ValueError("full merged-column weight shape differs")
        pieces: list[torch.Tensor] = []
        offset = 0
        for size, local_size in zip(self.output_sizes, self.local_output_sizes):
            pieces.append(value.narrow(0, offset + self.tp_rank * local_size, local_size))
            offset += size
        return torch.cat(pieces, dim=0).contiguous()

    def full_bias_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size,):
            raise ValueError("full merged-column bias shape differs")
        pieces: list[torch.Tensor] = []
        offset = 0
        for size, local_size in zip(self.output_sizes, self.local_output_sizes):
            pieces.append(value.narrow(0, offset + self.tp_rank * local_size, local_size))
            offset += size
        return torch.cat(pieces, dim=0).contiguous()


class RowParallelLinear(nn.Module):
    """Input-row sharded linear layer with summed TP partial outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool,
        dtype: torch.dtype,
        context: ParallelContext,
    ) -> None:
        super().__init__()
        self.parallel_context = context
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.tp_size = self.parallel_context.tp_world_size
        self.tp_rank = self.parallel_context.tp_rank
        self.inference_inplace_reduce_enabled = False
        if self.input_size % self.tp_size:
            raise ValueError(f"row input {self.input_size} is not divisible by TP={self.tp_size}")
        self.local_input_size = self.input_size // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.local_input_size, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.empty(self.output_size, dtype=dtype)) if bias else None

    def forward_local(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return this TP rank's partial output before reduction and bias."""

        return _module_linear(self, inputs, None)

    def reduce_local_output(self, value: torch.Tensor) -> torch.Tensor:
        if self.inference_inplace_reduce_enabled:
            return tp_all_reduce_inplace_(value, context=self.parallel_context)
        return tp_all_reduce(value, context=self.parallel_context)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.forward_local(inputs)
        output = self.reduce_local_output(output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def full_weight_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size, self.input_size):
            raise ValueError("full row-parallel weight shape differs")
        return value.narrow(
            1,
            self.tp_rank * self.local_input_size,
            self.local_input_size,
        ).contiguous()


__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "RowParallelLinear",
]
