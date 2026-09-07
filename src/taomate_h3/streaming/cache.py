# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Persistent clean audio/video KV cache for direct Stage3 streaming."""

from __future__ import annotations

from dataclasses import dataclass

import torch

MAIN_LAYER_NAMES = tuple(f"blocks.{index}.attn" for index in range(50))
VIDEO_TOKEN_TAG = 0
AUDIO_TOKEN_TAG = 2


@dataclass(frozen=True)
class KVContract:
    """Shape contract for the local attention shard."""

    local_heads: int = 14
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16
    device_type: str = "cuda"


def kv_contract_from_model(model: object) -> KVContract:
    """Build the KV shape contract from an installed MiniMax-H3 model."""

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("model does not expose transformer blocks")
    layers = len(blocks)
    if layers != len(MAIN_LAYER_NAMES):
        raise RuntimeError(f"expected {len(MAIN_LAYER_NAMES)} transformer layers, got {layers}")

    architecture = getattr(model, "arch", None)
    context = getattr(model, "parallel_context", None)
    total_heads = int(getattr(architecture, "num_attention_heads", 0))
    head_dim = int(getattr(architecture, "attention_head_dim", 0))
    tp_size = int(getattr(context, "tp_world_size", 0))
    ulysses_size = int(getattr(context, "ulysses_world_size", 0))
    parallel_size = tp_size * ulysses_size
    if total_heads <= 0 or head_dim <= 0 or parallel_size <= 0 or total_heads % parallel_size:
        raise RuntimeError("model topology cannot resolve post-Ulysses local KV heads")
    return KVContract(
        local_heads=total_heads // parallel_size,
        head_dim=head_dim,
    )


@dataclass(frozen=True)
class AVKV:
    """Dense BF16 clean audio/video keys and values for one attention layer."""

    key: torch.Tensor
    value: torch.Tensor


def _validate_av_pair(pair: AVKV, contract: KVContract, *, layer_name: str) -> None:
    expected_tail = (contract.local_heads, contract.head_dim)
    if pair.key.ndim != 3 or tuple(pair.key.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"{layer_name}: expected key shape [tokens, {expected_tail[0]}, {expected_tail[1]}], "
            f"got {tuple(pair.key.shape)}"
        )
    if pair.value.shape != pair.key.shape:
        raise RuntimeError(f"{layer_name}: key/value shape mismatch")
    if pair.key.dtype != contract.dtype or pair.value.dtype != contract.dtype:
        raise RuntimeError(f"{layer_name}: persistent KV must be {contract.dtype}")
    if (
        pair.key.device.type != contract.device_type
        or pair.value.device.type != contract.device_type
    ):
        raise RuntimeError(f"{layer_name}: persistent KV must reside on {contract.device_type}")


class CleanAVKVCache:
    """Transactional persistent cache containing clean media KV only.

    Clean commits append one small streaming chunk. Retention preserves the first
    video's rows as a long-term sink and the two most recent clean chunks.
    """

    def __init__(self, contract: KVContract) -> None:
        self.contract = contract
        self._history: dict[str, AVKV] = {}
        self._staged: dict[str, AVKV] = {}
        self._staged_block: int | None = None
        self._staged_tags: tuple[int, ...] | None = None
        self._staged_indices: torch.Tensor | None = None
        self._block_index = 0
        self._commit_token_counts: list[int] = []
        self._commit_token_tags: list[tuple[int, ...]] = []

    @property
    def committed_blocks(self) -> int:
        """Logical number of clean chunks committed since the last clear."""

        return self._block_index

    @property
    def history_tokens(self) -> int:
        if not self._history:
            return 0
        return int(next(iter(self._history.values())).key.shape[0])

    @property
    def history_audio_tokens(self) -> int:
        return sum(tag == AUDIO_TOKEN_TAG for tags in self._commit_token_tags for tag in tags)

    @property
    def history_video_tokens(self) -> int:
        return sum(tag == VIDEO_TOKEN_TAG for tags in self._commit_token_tags for tag in tags)

    @property
    def clean_commit_active(self) -> bool:
        return self._staged_block is not None

    def history(self, layer_name: str) -> AVKV | None:
        self._validate_layer_name(layer_name)
        return self._history.get(layer_name)

    def begin_clean_commit(self, block_index: int) -> None:
        if self.clean_commit_active:
            raise RuntimeError("a clean KV commit is already active")
        if block_index != self._block_index:
            raise RuntimeError(f"expected clean chunk {self._block_index}, got {block_index}")
        self._staged_block = block_index
        self._staged.clear()
        self._staged_tags = None
        self._staged_indices = None

    def stage(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        token_tags: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> None:
        """Stage clean audio/video rows produced by one transformer layer."""

        if not self.clean_commit_active:
            raise RuntimeError("begin_clean_commit() must be called before stage()")
        self._validate_layer_name(layer_name)
        if layer_name in self._staged:
            raise RuntimeError(f"{layer_name}: KV was staged more than once")
        if key.shape != value.shape or key.ndim != 3:
            raise RuntimeError(f"{layer_name}: key/value must have matching rank-3 shapes")
        if token_tags.ndim != 1 or commit_mask.ndim != 1:
            raise RuntimeError("token_tags and commit_mask must be rank-1")
        if token_tags.shape[0] != key.shape[0] or commit_mask.shape[0] != key.shape[0]:
            raise RuntimeError(f"{layer_name}: packed metadata length does not match KV rows")

        if self._staged_indices is None:
            indices = torch.nonzero(commit_mask.to(torch.bool), as_tuple=False).flatten()
            if indices.numel() == 0:
                raise RuntimeError("clean commit did not contain audio/video rows")
            selected_tags = token_tags.index_select(0, indices)
            tags = tuple(int(tag) for tag in selected_tags.detach().cpu().tolist())
            if frozenset(tags) != frozenset((VIDEO_TOKEN_TAG, AUDIO_TOKEN_TAG)):
                raise RuntimeError("each clean chunk must contain both video and audio KV")
            self._staged_indices = indices
            self._staged_tags = tags

        indices = self._staged_indices
        assert indices is not None
        pair = AVKV(
            key=key.index_select(0, indices).detach(),
            value=value.index_select(0, indices).detach(),
        )
        _validate_av_pair(pair, self.contract, layer_name=layer_name)
        self._staged[layer_name] = pair

    def commit(self) -> None:
        """Atomically append all staged layers to persistent history."""

        if not self.clean_commit_active:
            raise RuntimeError("no clean KV commit is active")
        missing = [name for name in MAIN_LAYER_NAMES if name not in self._staged]
        if missing:
            raise RuntimeError(f"clean KV commit is missing {len(missing)} transformer layers")
        assert self._staged_tags is not None
        token_count = len(self._staged_tags)

        # Update one layer at a time so old and new 50-layer caches are never
        # resident together at the direct-inference steady-state memory peak.
        for layer_name in MAIN_LAYER_NAMES:
            current = self._staged.pop(layer_name)
            previous = self._history.pop(layer_name, None)
            if previous is None:
                combined = current
            else:
                combined = AVKV(
                    key=torch.cat((previous.key, current.key), dim=0),
                    value=torch.cat((previous.value, current.value), dim=0),
                )
            _validate_av_pair(combined, self.contract, layer_name=layer_name)
            self._history[layer_name] = combined

        self._commit_token_counts.append(token_count)
        self._commit_token_tags.append(self._staged_tags)
        self._block_index += 1
        self._clear_staging()

    def retain_sink_and_recent_commits(self) -> None:
        """Keep the first chunk's video sink and the most recent clean chunks."""

        if self.clean_commit_active:
            raise RuntimeError("cannot trim persistent KV during an active clean commit")
        block_count = len(self._commit_token_counts)
        # As soon as three clean chunks exist, age chunk zero to a video-only
        # sink and keep chunks one and two as the two complete AV recents.
        if block_count <= 2:
            return

        recent_start = max(1, block_count - 2)
        selection = [(0, True)] + [(index, False) for index in range(recent_start, block_count)]
        self._retain_commit_rows(selection)

    def drop_audio_history(self) -> int:
        """Remove audio rows while preserving every retained video row."""

        if self.clean_commit_active:
            raise RuntimeError("cannot drop audio history during an active clean commit")
        removed_tokens = self.history_audio_tokens
        if not self._commit_token_counts:
            return 0
        self._retain_commit_rows([(index, True) for index in range(len(self._commit_token_counts))])
        return removed_tokens

    def rollback(self) -> None:
        """Discard the currently staged clean chunk."""

        self._clear_staging()

    def clear(self) -> None:
        """Clear persistent and staged KV and reset the logical chunk count."""

        self._history.clear()
        self._commit_token_counts.clear()
        self._commit_token_tags.clear()
        self._block_index = 0
        self._clear_staging()

    def _retain_commit_rows(self, selection: list[tuple[int, bool]]) -> None:
        offsets = [0]
        for count in self._commit_token_counts:
            offsets.append(offsets[-1] + count)

        selected_rows: list[int] = []
        selected_counts: list[int] = []
        selected_tags: list[tuple[int, ...]] = []
        for block_index, video_only in selection:
            tags = self._commit_token_tags[block_index]
            start = offsets[block_index]
            local_rows = [
                i for i, tag in enumerate(tags) if not video_only or tag == VIDEO_TOKEN_TAG
            ]
            if not local_rows:
                continue
            selected_rows.extend(start + row for row in local_rows)
            kept_tags = tuple(tags[row] for row in local_rows)
            selected_counts.append(len(kept_tags))
            selected_tags.append(kept_tags)

        if not selected_rows:
            self._history.clear()
            self._commit_token_counts.clear()
            self._commit_token_tags.clear()
            return

        next_history: dict[str, AVKV] = {}
        for layer_name, pair in self._history.items():
            indices = torch.tensor(selected_rows, dtype=torch.long, device=pair.key.device)
            retained = AVKV(
                key=pair.key.index_select(0, indices),
                value=pair.value.index_select(0, indices),
            )
            _validate_av_pair(retained, self.contract, layer_name=layer_name)
            next_history[layer_name] = retained
        self._history = next_history
        self._commit_token_counts = selected_counts
        self._commit_token_tags = selected_tags

    def _clear_staging(self) -> None:
        self._staged.clear()
        self._staged_block = None
        self._staged_tags = None
        self._staged_indices = None

    def _validate_layer_name(self, layer_name: str) -> None:
        if layer_name not in MAIN_LAYER_NAMES:
            raise KeyError(f"unexpected transformer layer: {layer_name}")
