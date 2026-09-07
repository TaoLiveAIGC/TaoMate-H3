# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""torchrun worker for TaoMate-H3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_run_config
from .runner import run_direct


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--audio-guidance-dir",
        type=Path,
        required=True,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    run_direct(
        load_run_config(args.config.resolve(strict=True)),
        audio_guidance_dir=args.audio_guidance_dir.resolve(strict=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
