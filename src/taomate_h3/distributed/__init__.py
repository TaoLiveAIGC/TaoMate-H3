# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Model-parallel primitives used by TaoMate-H3 inference."""

from .collectives import tp_all_gather, ulysses_all_gather_rows
from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from .parallel import ParallelContext
from .ulysses import (
    heads_to_sequence,
    merged_qkv_sequence_to_heads,
    packed_qkv_sequence_to_heads,
)

__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "ParallelContext",
    "RowParallelLinear",
    "heads_to_sequence",
    "merged_qkv_sequence_to_heads",
    "packed_qkv_sequence_to_heads",
    "tp_all_gather",
    "ulysses_all_gather_rows",
]
