"""Decode after the distributed encoders and generator have exited."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

from digitalhuman_h3.model.components import partition_root
from digitalhuman_h3.model.output import H3LatentDecoder, save_h3_mp4


@torch.no_grad()
def decode(model_root: Path, output: Path):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    root = partition_root(model_root, "ref2va")
    latents = torch.load(output / "latents.pt", map_location="cpu", weights_only=True)
    decoder = H3LatentDecoder(video_vae_root=root / "video_vae",
                              audio_vae_root=root / "audio_vae")
    decoder.enable_video_vae_cpu_temporal_output()
    frames = decoder.decode_video(video_latents=latents["video"], device=device)
    waveform, sample_rate = decoder.decode_audio(audio_latents=latents["audio"], device=device)
    save_h3_mp4(output / "video.mp4", frames=frames, audio=waveform,
                fps=latents["fps"], sample_rate=sample_rate)
    (output / "latents.pt").unlink()


if __name__ == "__main__":
    decode(Path(sys.argv[1]), Path(sys.argv[2]))
