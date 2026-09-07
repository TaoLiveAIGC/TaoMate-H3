# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParallelContext:
    """Explicit rank topology for native TP/Ulysses execution.

    Process groups are deliberately not inferred here. The runtime owns group
    construction and installs the rank view for the current worker.
    """

    tp_world_size: int
    tp_rank: int
    ulysses_world_size: int
    ulysses_rank: int
    tp_group: Any
    ulysses_group: Any

    def __post_init__(self) -> None:
        for name in ("tp", "ulysses"):
            size = getattr(self, f"{name}_world_size")
            rank = getattr(self, f"{name}_rank")
            if size <= 0 or not 0 <= rank < size:
                raise ValueError(f"invalid {name} topology: world_size={size}, rank={rank}")
        if self.tp_group is None or self.ulysses_group is None:
            raise ValueError("TP and Ulysses process groups are required")


__all__ = ["ParallelContext"]
