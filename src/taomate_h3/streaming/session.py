# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Streaming session state machine for noisy forwards and clean KV commits."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ..config import DIRECT_POLICY
from ..model.layers import set_native_stream_attention_hook
from .attention_hook import H3StreamingAttentionHook, HookMode
from .cache import CleanAVKVCache
from .geometry import StreamPhase, StreamPlan


class H3StreamingSession:
    """Install direct streaming attention around real H3 transformer forwards."""

    def __init__(
        self,
        plan: StreamPlan,
        *,
        cache: CleanAVKVCache,
    ) -> None:
        if not isinstance(plan, StreamPlan):
            raise TypeError("plan must be a StreamPlan")
        self.plan = plan
        self.cache = cache
        self._block_index_base = self.cache.committed_blocks
        self.hook = H3StreamingAttentionHook(
            self.cache,
            backend=DIRECT_POLICY.attention_backend,
        )
        self._installed = False
        self._noisy_blocks: set[int] = set()

    def __enter__(self) -> H3StreamingSession:
        if self._installed:
            raise RuntimeError("H3 streaming session is already installed")
        set_native_stream_attention_hook(self.hook)
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.cache.clean_commit_active:
            self.cache.rollback()
        self.hook.deactivate()
        if self._installed:
            set_native_stream_attention_hook(None)
            self._installed = False

    def phase(self, block_index: int) -> StreamPhase:
        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise TypeError("block_index must be an integer")
        if block_index < 0 or block_index >= len(self.plan.phases):
            raise ValueError(f"unknown streaming block {block_index}")
        return self.plan.phases[block_index]

    def _require_installed(self, block_index: int) -> StreamPhase:
        if not self._installed:
            raise RuntimeError("use H3StreamingSession as a context manager")
        phase = self.phase(block_index)
        if self._block_index_base + block_index != self.cache.committed_blocks:
            raise ValueError("streaming blocks must be processed and committed in order")
        return phase

    @contextmanager
    def noisy_step(self, block_index: int) -> Iterator[StreamPhase]:
        """Wrap one noisy DiT forward; it reads but never mutates clean KV."""

        phase = self._require_installed(block_index)
        global_block_index = self._block_index_base + block_index
        self.hook.activate(HookMode.NOISY)
        try:
            yield phase
            self._noisy_blocks.add(global_block_index)
        finally:
            self.hook.deactivate()

    @contextmanager
    def clean_commit(self, block_index: int) -> Iterator[StreamPhase]:
        """Wrap the dedicated sigma-zero forward and atomically commit clean KV."""

        phase = self._require_installed(block_index)
        global_block_index = self._block_index_base + block_index
        if global_block_index not in self._noisy_blocks:
            raise RuntimeError("a clean commit requires at least one noisy denoise forward")

        self.cache.begin_clean_commit(global_block_index)
        try:
            self.hook.activate(HookMode.CLEAN_COMMIT)
        except BaseException:
            self.cache.rollback()
            raise
        try:
            yield phase
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit()
            except BaseException:
                self.cache.rollback()
                raise
            self._noisy_blocks.remove(global_block_index)
        finally:
            self.hook.deactivate()


__all__ = ["H3StreamingSession"]
