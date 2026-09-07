# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from taomate_h3.distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from .dit import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiT,
)


class H3NativeWeightError(RuntimeError):
    pass


def reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if int(weight.shape[0]) != expected_out:
        raise H3NativeWeightError("qkv checkpoint rows differ from grouped MiniMax H3 layout")
    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        (
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ),
        dim=0,
    )


def _safe_regular_root(path: Path) -> Path:
    try:
        root = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise H3NativeWeightError(f"cannot resolve H3 transformer root: {exc}") from exc
    if not root.is_dir():
        raise H3NativeWeightError("H3 transformer root must be a directory")
    return root


def _index_weight_map(root: Path) -> Mapping[str, str] | None:
    candidates = (
        root / "diffusion_pytorch_model.safetensors.index.json",
        root / "model.safetensors.index.json",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise H3NativeWeightError("multiple H3 safetensors index files are present")
    if not existing:
        return None
    try:
        payload = json.loads(existing[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3NativeWeightError(f"cannot read H3 safetensors index: {exc}") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise H3NativeWeightError("H3 safetensors index weight_map is absent")
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()
    ):
        raise H3NativeWeightError("H3 safetensors index contains non-string entries")
    return dict(weight_map)


def _resolve_shards(root: Path) -> tuple[dict[str, Path], tuple[Path, ...]]:
    weight_map = _index_weight_map(root)
    if weight_map is None:
        shards = tuple(sorted(root.glob("*.safetensors")))
        if not shards:
            raise H3NativeWeightError("H3 transformer safetensors are absent")
        safe_open = importlib.import_module("safetensors").safe_open
        resolved: dict[str, Path] = {}
        canonical_shards = tuple(shard.resolve(strict=True) for shard in shards)
        for shard in canonical_shards:
            if not shard.is_file():
                raise H3NativeWeightError("H3 safetensors shard must be a file")
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in resolved:
                        raise H3NativeWeightError(f"duplicate H3 tensor key: {key}")
                    resolved[key] = shard
        return resolved, canonical_shards

    filenames = tuple(sorted(set(weight_map.values())))
    shards: list[Path] = []
    resolved_by_filename: dict[str, Path] = {}
    for filename in filenames:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise H3NativeWeightError("H3 safetensors index escapes transformer root")
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_file():
            raise H3NativeWeightError(f"H3 safetensors shard is absent: {filename}")
        shards.append(candidate)
        resolved_by_filename[filename] = candidate
    return {key: resolved_by_filename[value] for key, value in weight_map.items()}, tuple(shards)


def _local_tensor(
    name: str,
    value: torch.Tensor,
    *,
    owner: torch.nn.Module,
    leaf: str,
    model: MiniMaxH3DiT,
) -> torch.Tensor:
    if isinstance(owner, MergedColumnParallelLinear):
        if leaf not in {"weight", "bias"}:
            return value
        if name.endswith("attn.qkv_proj.weight"):
            value = reorder_grouped_qkv_to_qkv(
                value,
                num_query_groups=model.arch.num_attention_heads,
                heads_per_group=1,
                head_dim=model.arch.attention_head_dim,
            )
        return owner.full_weight_slice(value) if leaf == "weight" else owner.full_bias_slice(value)
    if isinstance(owner, ColumnParallelLinear):
        if leaf == "weight":
            return owner.full_weight_slice(value)
        if leaf == "bias":
            return owner.full_bias_slice(value)
    if isinstance(owner, RowParallelLinear) and leaf == "weight":
        return owner.full_weight_slice(value)
    return value


def load_minimax_h3_dit_weights(
    model: MiniMaxH3DiT,
    transformer_root: str | Path,
) -> dict[str, Any]:
    """Stream official safetensors one tensor at a time into TP-local parameters."""

    if not isinstance(model, MiniMaxH3DiT):
        raise TypeError("model must be MiniMaxH3DiT")
    root = _safe_regular_root(Path(transformer_root))
    key_to_shard, shards = _resolve_shards(root)
    safe_open = importlib.import_module("safetensors").safe_open
    state = model.state_dict(keep_vars=True)
    expected = set(state)
    present = set(key_to_shard)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing or unexpected:
        raise H3NativeWeightError(
            "H3 transformer tensor inventory differs: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    modules = dict(model.named_modules())
    by_shard: dict[Path, list[str]] = {path: [] for path in shards}
    for name, path in key_to_shard.items():
        by_shard[path].append(name)
    streamed_names: list[str] = []
    streamed_source_bytes = 0
    streamed_quantized_bytes = 0
    streamed_scale_bytes = 0
    streamed_temporary_bytes_limit = 0
    bf16_copy_count = 0
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in sorted(by_shard[shard]):
                    destination = state[name]
                    owner_name, separator, leaf = name.rpartition(".")
                    if not separator:
                        owner_name, leaf = "", name
                    owner = modules.get(owner_name)
                    if owner is None:
                        raise H3NativeWeightError(f"H3 state owner is absent: {name}")
                    value = _local_tensor(
                        name,
                        handle.get_tensor(name),
                        owner=owner,
                        leaf=leaf,
                        model=model,
                    )
                    if tuple(value.shape) != tuple(destination.shape):
                        raise H3NativeWeightError(
                            f"H3 local tensor shape differs for {name}: "
                            f"{tuple(value.shape)} != {tuple(destination.shape)}"
                        )
                    if (
                        leaf == "weight"
                        and getattr(owner, "_cutlass_w8a8_stream_load_pending", None) is True
                    ):
                        from taomate_h3.inference.w8a8 import (
                            load_cutlass_w8a8_stream_weight_,
                        )

                        load_cutlass_w8a8_stream_weight_(owner, value, name=name)
                        streamed_names.append(owner_name)
                        streamed_source_bytes += int(owner._cutlass_w8a8_source_weight_bytes)
                        streamed_quantized_bytes += int(
                            owner.weight.numel() * owner.weight.element_size()
                        )
                        streamed_scale_bytes += int(
                            owner._cutlass_w8a8_weight_scale.numel()
                            * owner._cutlass_w8a8_weight_scale.element_size()
                        )
                        streamed_temporary_bytes_limit = max(
                            streamed_temporary_bytes_limit,
                            int(owner._cutlass_w8a8_stream_load_temporary_bytes_limit),
                        )
                        continue
                    if value.dtype != destination.dtype:
                        raise H3NativeWeightError(
                            f"H3 tensor dtype differs for {name}: "
                            f"{value.dtype} != {destination.dtype}"
                        )
                    destination.copy_(value)
                    if leaf == "weight" and value.dtype == torch.bfloat16:
                        bf16_copy_count += 1
    pending_w8a8 = [
        name
        for name, module in model.named_modules()
        if getattr(module, "_cutlass_w8a8_stream_load_pending", None) is True
    ]
    if pending_w8a8:
        raise H3NativeWeightError(f"H3 streamed W8A8 weights were not loaded: {pending_w8a8[:8]}")
    for name in MINIMAX_H3_FP32_PARAM_NAMES:
        if model.get_parameter(name).dtype != torch.float32:
            raise H3NativeWeightError(f"H3 parameter must stay fp32: {name}")
    for name in MINIMAX_H3_FP32_BUFFER_NAMES:
        if model.get_buffer(name).dtype != torch.float32:
            raise H3NativeWeightError(f"H3 buffer must stay fp32: {name}")
    receipt = {
        "shard_count": len(shards),
        "tensor_count": len(state),
        "stream_load_policy": model._load_time_w8a8_policy,
        "streamed_w8a8_linear_count": len(streamed_names),
        "streamed_w8a8_source_weight_bytes": streamed_source_bytes,
        "streamed_w8a8_quantized_weight_bytes": streamed_quantized_bytes,
        "streamed_w8a8_scale_bytes": streamed_scale_bytes,
        "stream_load_bf16_staging_bytes_limit": streamed_temporary_bytes_limit,
        "bf16_weight_copy_count": bf16_copy_count,
    }
    model._native_weight_load_receipt = receipt
    return receipt


__all__ = [
    "H3NativeWeightError",
    "load_minimax_h3_dit_weights",
    "reorder_grouped_qkv_to_qkv",
]
