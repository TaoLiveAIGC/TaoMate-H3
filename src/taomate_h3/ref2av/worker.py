"""Separate input encoding and DiT workers to release encoder memory."""
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch

from digitalhuman_h3.distributed import ParallelContext, use_parallel_context
from digitalhuman_h3.model.pipeline import MiniMaxH3NativePipeline


def parallel_context(rank, world):
    tp = [torch.distributed.new_group([i, i + 1]) for i in range(0, world, 2)]
    sp = [torch.distributed.new_group(list(range(i, world, 2))) for i in range(2)]
    return ParallelContext(tp_world_size=2, tp_rank=rank % 2, tp_group=tp[rank // 2],
                           ulysses_world_size=world // 2, ulysses_rank=rank // 2,
                           ulysses_group=sp[rank % 2])


def main():
    stage, root = sys.argv[1], Path(sys.argv[2])
    config = json.loads((root / "request.json").read_text())
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=45))
    context = parallel_context(rank, world)
    torch.set_grad_enabled(False)
    try:
        with use_parallel_context(context):
            pipeline = MiniMaxH3NativePipeline(
                model_root=config["model_root"], variant="ref2va", device=device,
                parallel_context=context, rank=rank, world_size=world,
                defer_transformer=stage == "prepare", defer_text_encoder=stage == "generate")
            if stage == "prepare":
                from .conditioning import prepare

                payload = prepare(pipeline, config["prompt"], config["references"],
                                  config["duration"], tuple(config["resolution"]), config["seed"])
                if rank == 0:
                    torch.save(payload, root / "conditions.pt")
                torch.distributed.barrier()
            else:
                from .adapters import Adapters
                from .generation import generate

                adapters = Adapters(pipeline.transformer, Path(config["adapter"]))
                payload = torch.load(root / "conditions.pt", map_location="cpu", weights_only=True)
                generate(pipeline, payload, adapters, Path(config["output"]), config["seed"])
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
