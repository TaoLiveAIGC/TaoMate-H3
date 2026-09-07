# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Native in-process MiniMax H3 audio-video DiT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from taomate_h3.distributed import (
    ColumnParallelLinear,
    ParallelContext,
    tp_all_gather,
    ulysses_all_gather_rows,
)

from .architecture import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3Architecture,
)
from .layers import (
    BF16,
    FP32,
    MiniMaxH3DiTBlock,
    MiniMaxH3FinalLayer,
    MiniMaxH3Rope,
    MiniMaxH3TimeEmbedder,
    MiniMaxH3TokenRefiner,
    rope_cos_sin_cache,
)

MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})

_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "block_token_tags",
        "block_combined_indices",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "streaming_cache_only",
    }
)


def _required(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiT.forward requires kwarg {key!r}")
    return kwargs[key]


class MiniMaxH3DiT(nn.Module):
    """Checkpoint-key-compatible MiniMax H3 DiT with native PyTorch kernels."""

    def __init__(
        self,
        *,
        parallel_context: ParallelContext,
        load_time_w8a8_policy: str | None = None,
    ) -> None:
        super().__init__()
        self.arch = MiniMaxH3Architecture()
        self.parallel_context = parallel_context
        self.hidden_size = self.arch.hidden_size
        self._validate_topology()
        context = self.parallel_context
        self.video_patch_proj = ColumnParallelLinear(
            self.arch.video_row_width,
            self.arch.hidden_size,
            bias=True,
            gather_output=True,
            dtype=FP32,
            context=context,
        )
        self.audio_patch_proj = ColumnParallelLinear(
            self.arch.audio_latent_channels,
            self.arch.hidden_size,
            bias=True,
            gather_output=True,
            dtype=FP32,
            context=context,
        )
        self.condition_proj = ColumnParallelLinear(
            self.arch.text_dim,
            self.arch.hidden_size,
            bias=True,
            gather_output=True,
            dtype=BF16,
            context=context,
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(self.arch, context=context)
        self.rope = MiniMaxH3Rope(self.arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(self.arch, context=context)
        load_time_w8a8_names: frozenset[str] = frozenset()
        prepare_w8a8_slot: Any | None = None
        if load_time_w8a8_policy is not None:
            from taomate_h3.inference.w8a8 import (
                prepare_cutlass_w8a8_stream_load_slot_,
                require_vllm_cutlass_w8a8,
                resolve_h3_cutlass_w8a8_policy,
            )

            require_vllm_cutlass_w8a8()
            candidate_names = [
                f"blocks.{index}.{projection}"
                for index in range(self.arch.num_layers)
                for projection in ("attn.qkv_proj", "mlp.fc1", "mlp.fc2")
            ]
            selection = resolve_h3_cutlass_w8a8_policy(
                candidate_names,
                policy=load_time_w8a8_policy,
            )
            load_time_w8a8_names = frozenset(selection["quantized_linear_names"])
            prepare_w8a8_slot = prepare_cutlass_w8a8_stream_load_slot_

        self.blocks = nn.ModuleList()
        for index in range(self.arch.num_layers):
            if prepare_w8a8_slot is None:
                block = MiniMaxH3DiTBlock(self.arch, context=context)
            else:
                # Build on meta first so selected projections never own a
                # full-size BF16 CUDA allocation. ``to_empty`` then allocates
                # every parameter directly in its final dtype on this rank.
                with torch.device("meta"):
                    block = MiniMaxH3DiTBlock(self.arch, context=context)
                for suffix, module in (
                    ("attn.qkv_proj", block.attn.qkv_proj),
                    ("mlp.fc1", block.mlp.fc1),
                ):
                    name = f"blocks.{index}.{suffix}"
                    if name in load_time_w8a8_names:
                        prepare_w8a8_slot(module, name=name)
                block.to_empty(device=self.condition_proj.weight.device)
            block.attn.layer_name = f"blocks.{index}"
            self.blocks.append(block)
        self._load_time_w8a8_policy = load_time_w8a8_policy
        self._load_time_w8a8_linear_names = load_time_w8a8_names
        self.final_layer = MiniMaxH3FinalLayer(self.arch, context=context)
        self._inference_static_timestep_cache_enabled = False
        self._inference_static_adaln_input_cache: dict[int, torch.Tensor] = {}
        self._inference_static_adaln_input_hits = 0
        self._inference_static_adaln_input_misses = 0

    def enable_inference_static_timestep_cache(self) -> None:
        """Reuse timestep and AdaLN projections across the fixed denoise schedule."""

        if self.training:
            raise RuntimeError("static timestep cache is inference-only")
        self._inference_static_timestep_cache_enabled = True
        self._inference_static_adaln_input_cache.clear()
        self._inference_static_adaln_input_hits = 0
        self._inference_static_adaln_input_misses = 0
        self.time_embedder.enable_inference_static_cache()
        for block in self.blocks:
            block.adaln_proj.enable_inference_static_cache()
        self.final_layer.adaln_proj.enable_inference_static_cache()

    def inference_static_timestep_cache_stats(self) -> dict[str, int]:
        time_hits, time_misses = self.time_embedder.inference_static_cache_stats()
        projection_hits = 0
        projection_misses = 0
        for module in [
            *(block.adaln_proj for block in self.blocks),
            self.final_layer.adaln_proj,
        ]:
            hits, misses = module.inference_static_cache_stats()
            projection_hits += hits
            projection_misses += misses
        return {
            "time_embedding_hits": time_hits,
            "time_embedding_misses": time_misses,
            "adaln_input_hits": self._inference_static_adaln_input_hits,
            "adaln_input_misses": self._inference_static_adaln_input_misses,
            "adaln_projection_hits": projection_hits,
            "adaln_projection_misses": projection_misses,
        }

    @classmethod
    def allocate(
        cls,
        device: str | torch.device,
        *,
        parallel_context: ParallelContext,
        load_time_w8a8_policy: str | None = None,
    ) -> "MiniMaxH3DiT":
        """Allocate TP-local parameter storage directly on the target device."""

        with torch.device(device):
            return cls(
                parallel_context=parallel_context,
                load_time_w8a8_policy=load_time_w8a8_policy,
            )

    def _validate_topology(self) -> None:
        context = self.parallel_context
        for name, value in (
            ("num_attention_heads", self.arch.num_attention_heads),
            ("hidden_size", self.arch.hidden_size),
            ("ffn_hidden_size", self.arch.ffn_hidden_size),
            ("time_embed_hidden_size", self.arch.time_embed_hidden_size),
            ("adaln_out_features", self.arch.adaln_out_features),
            ("final_adaln_out_features", self.arch.final_adaln_out_features),
            ("video_row_width", self.arch.video_row_width),
            ("audio_latent_channels", self.arch.audio_latent_channels),
        ):
            if value % context.tp_world_size:
                raise ValueError(f"MiniMax H3 {name}={value} is not divisible by TP")
        local_heads = self.arch.num_attention_heads // context.tp_world_size
        if local_heads % context.ulysses_world_size:
            raise ValueError("TP-local H3 heads are not divisible by Ulysses")
        if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % context.ulysses_world_size:
            raise ValueError("H3 packed alignment is not divisible by Ulysses")

    @staticmethod
    def _position_ids(value: Any, key: str) -> torch.Tensor:
        ids = value.get("position_ids")
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        text_len = int(refiner_cu_seqlens[1].item())
        if text_len <= 0 or text_len > int(prompt_embeds.shape[0]):
            raise ValueError("refiner live text length differs")
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=BF16)
        return self.token_refiner(
            self.condition_proj(text_rows),
            cu_seqlens_host=(0, text_len, text_len),
            max_seqlen=text_len,
        )

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError("img_position_ids must be [1, S, 3]")
        sequence_length = int(img_position_ids.shape[1])
        world_size = self.parallel_context.ulysses_world_size
        rank = self.parallel_context.ulysses_rank
        if sequence_length % world_size:
            raise ValueError("packed sequence is not divisible by Ulysses")
        local_length = sequence_length // world_size
        row_start = rank * local_length
        frequencies = self.rope(img_position_ids[:, row_start : row_start + local_length]).to(
            device
        )
        return (
            rope_cos_sin_cache(frequencies, dtype=BF16),
            torch.arange(local_length, device=device, dtype=torch.long),
        )

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        row_start: int,
        row_stop: int,
        device: torch.device,
        local_embedding_layout: dict[str, torch.Tensor | int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embed = text_embeddings_selected.to(device=device, dtype=BF16)
        text_len = int(text_embed.shape[0])
        if text_len <= 0 or int(text_embed.shape[-1]) != self.hidden_size:
            raise ValueError("refined prompt embedding shape differs")

        local_seq_len = row_stop - row_start
        used_len = text_len + int(img_pos.numel()) + int(audio_pos.numel())
        local_live_rows = min(max(used_len - row_start, 0), local_seq_len)
        embeddings = torch.empty((local_seq_len, self.hidden_size), device=device, dtype=BF16)
        if local_live_rows < local_seq_len:
            embeddings[local_live_rows:].zero_()
        text_source_start = int(local_embedding_layout["text_source_start"])
        text_source_stop = int(local_embedding_layout["text_source_stop"])
        img_global_ids = local_embedding_layout["img_global_ids"]
        img_row_ids = local_embedding_layout["img_row_ids"]
        audio_global_ids = local_embedding_layout["audio_global_ids"]
        audio_row_ids = local_embedding_layout["audio_row_ids"]
        text_rows = text_source_stop - text_source_start
        if text_rows:
            embeddings[:text_rows].copy_(text_embed[text_source_start:text_source_stop])

        video_rows = (
            x.view(-1, x.shape[-1]).index_select(0, img_global_ids)
            if img_row_ids.numel()
            else x.new_empty((0, x.shape[-1]))
        ).to(FP32)
        if int(img_pos.numel()) == 0:
            video_embeddings = x.new_empty((0, self.hidden_size), dtype=BF16)
        else:
            video_embeddings = self.video_patch_proj(video_rows).to(BF16)
        if img_row_ids.numel():
            embeddings.index_copy_(0, img_row_ids, video_embeddings)

        audio_rows = (
            audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_global_ids)
            if audio_row_ids.numel()
            else audio_x.new_empty((0, audio_x.shape[-1]))
        ).to(FP32)
        audio_embeddings = self.audio_patch_proj(audio_rows).to(BF16)
        if audio_row_ids.numel():
            embeddings.index_copy_(0, audio_row_ids, audio_embeddings)
        return embeddings, self.time_embedder(unique_timesteps)

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(f"MiniMaxH3DiT received unexpected kwargs: {unexpected}")
        streaming_cache_only = kwargs.get("streaming_cache_only", False)
        if type(streaming_cache_only) is not bool:
            raise TypeError("streaming_cache_only must be a boolean")
        x = _required(kwargs, "x")
        audio_x = _required(kwargs, "audio_x")
        unique_timesteps = _required(kwargs, "unique_timesteps")
        inverse_indices = _required(kwargs, "inverse_indices").view(-1).to(torch.long)
        block_token_tags = _required(kwargs, "block_token_tags").view(-1).to(torch.long)
        prompt_embeds = _required(kwargs, "prompt_embeds")
        img_pos = self._position_ids(_required(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._position_ids(_required(kwargs, "audio_pos_info"), "audio_pos_info")
        infer_out_pos = self._position_ids(
            _required(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )
        packed = _required(kwargs, "packed_seq_params")
        cu_seqlens_host = tuple(int(value) for value in packed["cu_seqlens_q_host"])
        max_seqlen = int(packed["max_seqlen_q"])
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError("x must be [1, S, C]")
        seq_len = int(x.shape[1])
        if int(inverse_indices.shape[0]) != seq_len:
            raise ValueError("inverse_indices must cover the packed sequence")
        context = self.parallel_context
        if seq_len % context.ulysses_world_size:
            raise ValueError("packed sequence is not divisible by Ulysses")
        local_seq_len = seq_len // context.ulysses_world_size
        row_start = context.ulysses_rank * local_seq_len
        row_stop = row_start + local_seq_len
        device = x.device
        rope_cache = _required(kwargs, "rope_cache")
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        hidden, timestep_embedding = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=prompt_embeds,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            row_start=row_start,
            row_stop=row_stop,
            device=device,
            local_embedding_layout=_required(kwargs, "local_embedding_layout"),
        )
        if self._inference_static_timestep_cache_enabled:
            cache_key = id(timestep_embedding)
            adaln_input = self._inference_static_adaln_input_cache.get(cache_key)
            if adaln_input is None:
                adaln_input = F.silu(timestep_embedding).to(BF16)
                self._inference_static_adaln_input_cache[cache_key] = adaln_input
                self._inference_static_adaln_input_misses += 1
            else:
                self._inference_static_adaln_input_hits += 1
        else:
            adaln_input = F.silu(timestep_embedding).to(BF16)
        inverse_indices = inverse_indices.to(device)
        block_inverse = inverse_indices[row_start:row_stop]
        block_token_tags = block_token_tags.to(device)
        if int(block_token_tags.shape[0]) != local_seq_len:
            raise ValueError("block_token_tags must cover the local sequence")
        combined = _required(kwargs, "block_combined_indices")
        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined,
                rope_cache=rope_cache,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
            )
        # Sigma=0 clean commit consumes only the K/V staged by the 50 attention
        # hooks above. Its caller explicitly sets
        # this private flag so the unused final logits are never constructed or
        # gathered; ordinary denoise forwards retain the original path below.
        if streaming_cache_only:
            return None
        if int(infer_out_pos.numel()) == 0:
            audio_logits = self.final_layer.forward_audio_only(
                hidden, adaln_input=adaln_input, inverse_indices=block_inverse
            )
            audio_logits = ulysses_all_gather_rows(audio_logits, context=context)
            audio_logits = audio_logits.index_select(0, audio_pos.to(device))
            audio_logits = tp_all_gather(audio_logits, context=context)
            video_logits = audio_logits.new_empty((0, self.arch.video_row_width))
            return video_logits, audio_logits
        video_logits = self.final_layer.forward_video_only(
            hidden, adaln_input=adaln_input, inverse_indices=block_inverse
        )
        video_logits = ulysses_all_gather_rows(video_logits, context=context)
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        video_logits = tp_all_gather(video_logits, context=context)
        audio_logits = torch.zeros(
            (int(audio_pos.numel()), self.arch.audio_latent_channels),
            device=device,
            dtype=video_logits.dtype,
        )
        return video_logits, audio_logits


__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiT",
]
