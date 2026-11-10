"""Resident opening and continuation LoRA banks."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from safetensors import safe_open

from digitalhuman_h3.distributed.collectives import tp_all_reduce
from digitalhuman_h3.distributed.linear import MergedColumnParallelLinear
from taomate_h3.inference.lora_checkpoint import (
    _column_local_b,
    _row_local_a,
    canonical_h3_lora_targets,
)


class BankedLinear(torch.nn.Module):
    def __init__(self, base, banks, scale):
        super().__init__()
        self.base = base
        self.scale = scale
        self.active = "warm"
        for role, (a, b) in banks.items():
            self.register_buffer(f"{role}_a", a, persistent=False)
            self.register_buffer(f"{role}_b", b, persistent=False)

    def forward(self, inputs):
        output = self.base(inputs)
        a = getattr(self, f"{self.active}_a").to(output.dtype)
        b = getattr(self, f"{self.active}_b").to(output.dtype)
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs.to(output.dtype), a), b)
        if not isinstance(self.base, MergedColumnParallelLinear) and self.base.tp_size > 1:
            update = tp_all_reduce(update, context=self.base.parallel_context)
        return output + update * self.scale


class Adapters:
    def __init__(self, transformer, directory: Path):
        config = json.loads((directory / "config.json").read_text())
        self.modules = []
        self.text = {}
        handles = {role: safe_open(str(directory / config["banks"][role]),
                                   framework="pt", device="cpu") for role in ("cold", "warm")}
        targets = canonical_h3_lora_targets()
        expected = {f"{name}.{leaf}" for name in targets for leaf in ("lora_a", "lora_b")}
        for role, handle in handles.items():
            if set(handle.keys()) != expected:
                raise ValueError(f"{role} adapter does not contain the H3 LoRA projections")
        for name in targets:
            parent_name, _, leaf = name.rpartition(".")
            parent = transformer.get_submodule(parent_name)
            base = getattr(parent, leaf)
            column = isinstance(base, MergedColumnParallelLinear)
            banks = {}
            for role, handle in handles.items():
                a, b = (handle.get_tensor(f"{name}.{key}") for key in ("lora_a", "lora_b"))
                if a.shape != (config["rank"], base.input_size) or b.shape != (
                    base.output_size, config["rank"]
                ):
                    raise ValueError(f"{role} LoRA shape differs: {name}")
                a = a if column else _row_local_a(a, base)
                b = _column_local_b(b, base) if column else b
                banks[role] = (a.to(base.weight.device), b.to(base.weight.device))
            wrapped = BankedLinear(base, banks, config["alpha"] / config["rank"])
            setattr(parent, leaf, wrapped)
            self.modules.append(wrapped)

    @contextmanager
    def use_chunk(self, index):
        role = "cold" if index == 0 else "warm"
        previous = self.modules[0].active
        for module in self.modules:
            module.active = role
        try:
            yield
        finally:
            for module in self.modules:
                module.active = previous

    def refined_text(self, payload, index, refine):
        role = "cold" if index == 0 else "warm"
        if role not in self.text:
            with self.use_chunk(index):
                self.text[role] = refine(payload)
        return self.text[role]
