# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""MiniMax H3 Qwen presentation construction without a serving framework."""

from __future__ import annotations

from typing import Any

import torch


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def minimax_h3_text_only_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    return torch.tensor(_text_ids(tokenizer, prompt), dtype=torch.long)


__all__ = [
    "minimax_h3_text_only_ids",
]
