# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""TP2 Qwen3-VL layer-50 feature encoder used by MiniMax H3."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from taomate_h3.distributed import (
    ColumnParallelLinear,
    ParallelContext,
    RowParallelLinear,
)

SELECTED_LANGUAGE_LAYERS = 50
HIDDEN_SIZE = 5120
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class H3TextEncoderError(RuntimeError):
    pass


def _transformers_qwen_symbols() -> tuple[Any, Any, Any]:
    try:
        config_mod = importlib.import_module("transformers.models.qwen3_vl.configuration_qwen3_vl")
        model_mod = importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
    except ImportError as exc:
        raise H3TextEncoderError("MiniMax H3 requires a Transformers build with Qwen3-VL") from exc
    return (
        config_mod.Qwen3VLConfig,
        model_mod.Qwen3VLTextRMSNorm,
        model_mod.Qwen3VLTextRotaryEmbedding,
    )


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    model_mod = importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
    return model_mod.apply_rotary_pos_emb(q, k, cos, sin)


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, heads, sequence, width = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, width)
        .reshape(batch, heads * repeats, sequence, width)
    )


class _TextAttention(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        if config.num_attention_heads % context.tp_world_size:
            raise ValueError("Qwen3-VL query heads must be divisible by TP")
        if config.num_key_value_heads % context.tp_world_size:
            raise ValueError("Qwen3-VL key/value heads must be divisible by TP")
        self.head_dim = int(config.head_dim)
        self.query_heads = int(config.num_attention_heads) // context.tp_world_size
        self.kv_heads = int(config.num_key_value_heads) // context.tp_world_size
        self.kv_repeats = self.query_heads // self.kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=bool(config.attention_bias),
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        _, rms_norm, _ = _transformers_qwen_symbols()
        self.q_norm = rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
            context=context,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        q = self.q_norm(
            self.q_proj(hidden).view(batch, sequence, self.query_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden).view(batch, sequence, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, sequence, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)
        k = _repeat_kv(k, self.kv_repeats)
        v = _repeat_kv(v, self.kv_repeats)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scaling,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, sequence, -1))


class _TextMLP(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
            context=context,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _TextLayer(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        _, rms_norm, _ = _transformers_qwen_symbols()
        self.self_attn = _TextAttention(config, context)
        self.mlp = _TextMLP(config, context)
        self.input_layernorm = rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), position_embeddings=position_embeddings
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _LanguageModel(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
            dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleList(
            _TextLayer(config, context) for _ in range(SELECTED_LANGUAGE_LAYERS)
        )
        _, _, rotary = _transformers_qwen_symbols()
        self.rotary_emb = rotary(config=config)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        return hidden


class _Backbone(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        self.language_model = _LanguageModel(config.text_config, context)


class MiniMaxH3TextEncoder(nn.Module):
    """Text-only Qwen3-VL backbone ending at unnormalized layer 50."""

    def __init__(self, config: Any, *, context: ParallelContext) -> None:
        super().__init__()
        self.parallel_context = context
        if int(config.text_config.hidden_size) != HIDDEN_SIZE:
            raise ValueError("MiniMax H3 Qwen3-VL hidden size must be 5120")
        if int(config.text_config.num_hidden_layers) < SELECTED_LANGUAGE_LAYERS:
            raise ValueError("Qwen3-VL checkpoint has fewer than 50 language layers")
        self.model = _Backbone(config, self.parallel_context)

    @classmethod
    def allocate(
        cls,
        text_encoder_root: str | Path,
        *,
        device: str | torch.device,
        context: ParallelContext,
    ) -> "MiniMaxH3TextEncoder":
        root = Path(text_encoder_root).resolve(strict=True)
        config_type, _, _ = _transformers_qwen_symbols()
        config = config_type.from_pretrained(str(root), local_files_only=True)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(device):
                return cls(config, context=context)
        finally:
            torch.set_default_dtype(previous_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _position_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.arange(input_ids.shape[1], dtype=torch.long)[None, None].expand(
            3, input_ids.shape[0], -1
        )

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be one-dimensional")
        host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
        position_ids = self._position_ids(host_ids).to(self.device)
        ids = host_ids.to(self.device)
        embeddings = self.model.language_model.embed_tokens(ids)
        hidden = self.model.language_model(
            inputs_embeds=embeddings,
            position_ids=position_ids,
        )[0]
        if tuple(hidden.shape) != (int(ids.shape[1]), HIDDEN_SIZE):
            raise ValueError(f"unexpected Qwen3-VL layer-50 shape {tuple(hidden.shape)}")
        return hidden.to(torch.bfloat16)


def _index(root: Path) -> Mapping[str, str]:
    path = root / "model.safetensors.index.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3TextEncoderError(f"cannot read Qwen3-VL weight index: {exc}") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise H3TextEncoderError("Qwen3-VL weight index has no weight_map")
    return dict(weight_map)


def _selected_weight(name: str) -> bool:
    if not name.startswith("model.language_model."):
        return False
    if name.startswith("model.language_model.norm."):
        return False
    match = _LAYER_RE.match(name)
    if match:
        return int(match.group(1)) < SELECTED_LANGUAGE_LAYERS
    return name == "model.language_model.embed_tokens.weight"


def load_minimax_h3_text_weights(
    model: MiniMaxH3TextEncoder,
    text_encoder_root: str | Path,
) -> None:
    root = Path(text_encoder_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise H3TextEncoderError("Qwen3-VL weight root must be a directory")
    full_map = _index(root)
    weight_map = {name: file for name, file in full_map.items() if _selected_weight(name)}
    state = model.state_dict(keep_vars=True)
    expected, present = set(state), set(weight_map)
    if expected != present:
        raise H3TextEncoderError(
            "Qwen3-VL selected tensor inventory differs: "
            f"missing={sorted(expected - present)[:8]}, "
            f"unexpected={sorted(present - expected)[:8]}"
        )
    modules = dict(model.named_modules())
    shards: dict[Path, list[str]] = {}
    for name, filename in weight_map.items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise H3TextEncoderError(f"Qwen3-VL shard path escapes its index: {filename}")
        shard = (root / relative).resolve(strict=True)
        if not shard.is_file():
            raise H3TextEncoderError(f"Qwen3-VL shard is unavailable: {filename}")
        shards.setdefault(shard, []).append(name)
    safe_open = importlib.import_module("safetensors").safe_open
    with torch.no_grad():
        for shard, names in sorted(shards.items(), key=lambda item: item[0].name):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    value = handle.get_tensor(name)
                    owner_name, _, leaf = name.rpartition(".")
                    owner = modules.get(owner_name)
                    if owner is None:
                        raise H3TextEncoderError(f"Qwen3-VL state owner is absent: {name}")
                    if isinstance(owner, ColumnParallelLinear):
                        value = (
                            owner.full_weight_slice(value)
                            if leaf == "weight"
                            else owner.full_bias_slice(value)
                        )
                    elif isinstance(owner, RowParallelLinear) and leaf == "weight":
                        value = owner.full_weight_slice(value)
                    destination = state[name]
                    if tuple(value.shape) != tuple(destination.shape):
                        raise H3TextEncoderError(
                            f"Qwen3-VL local tensor shape differs for {name}: "
                            f"{tuple(value.shape)} != {tuple(destination.shape)}"
                        )
                    destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
    model.eval()


__all__ = [
    "H3TextEncoderError",
    "MiniMaxH3TextEncoder",
    "SELECTED_LANGUAGE_LAYERS",
    "load_minimax_h3_text_weights",
]
