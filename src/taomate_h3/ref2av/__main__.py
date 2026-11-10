"""Command-line entry for Ref2AV inference."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


class Reference(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        items = list(namespace.references or [])
        kind = "image" if option_string == "--reference-image" else "video"
        items.append({"type": kind, "uri": str(Path(value).expanduser().resolve(strict=True))})
        namespace.references = items


def resolution(value):
    try:
        width, height = map(int, value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, for example 768x1344") from None
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise argparse.ArgumentTypeError("width and height must be positive multiples of 32")
    return width, height


def parser():
    result = argparse.ArgumentParser(description="Generate audio-video from visual references.")
    prompt = result.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    result.add_argument("--reference-image", dest="references", action=Reference, metavar="PATH")
    result.add_argument("--reference-video", dest="references", action=Reference, metavar="PATH")
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--duration", type=float, default=10)
    result.add_argument("--resolution", type=resolution, default=(768, 1344))
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--gpus", type=int, default=8)
    result.add_argument("--devices", help="CUDA device IDs, in rank order")
    result.add_argument("--output", type=Path, required=True)
    return result


def main():
    import json

    command = parser()
    args = command.parse_args()
    if not args.references:
        command.error("provide --reference-image or --reference-video")
    if args.gpus < 2 or args.gpus % 2:
        command.error("TP2 requires a positive even GPU count")
    if args.duration <= 0:
        command.error("--duration must be positive")
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    if not prompt.strip():
        command.error("prompt must not be empty")
    devices = args.devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = devices or ",".join(map(str, range(args.gpus)))
    if len(devices.split(",")) != args.gpus:
        command.error("--devices must contain --gpus entries")
    if (args.output / "video.mp4").exists():
        command.error("output video already exists; choose another --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": devices}
    environment.setdefault("OMP_NUM_THREADS", "1")
    config = {
        "prompt": prompt, "references": args.references,
        "model_root": str(args.model_root.expanduser().resolve(strict=True)),
        "adapter": str(args.adapter.expanduser().resolve(strict=True)),
        "duration": args.duration, "resolution": args.resolution, "seed": args.seed,
        "output": str(args.output.resolve()),
    }
    with TemporaryDirectory(prefix=".ref2av-", dir=args.output) as temporary:
        root = Path(temporary)
        (root / "request.json").write_text(json.dumps(config, ensure_ascii=False))
        for stage in ("prepare", "generate"):
            print(f"Ref2AV: {stage}", flush=True)
            subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc-per-node={args.gpus}", "--module", "taomate_h3.ref2av.worker",
                stage, str(root),
            ], check=True, env=environment)
    print("Ref2AV: decode", flush=True)
    subprocess.run([sys.executable, "-m", "taomate_h3.ref2av.decoding",
                    config["model_root"], config["output"]], check=True, env=environment)
    print(args.output / "video.mp4", flush=True)


if __name__ == "__main__":
    main()
