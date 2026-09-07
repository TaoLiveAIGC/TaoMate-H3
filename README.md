# TaoMate-H3

## Demo Video

<!-- Add the demo video here. -->

<br>

TaoMate-H3 is a low-latency streaming audio-video generation runtime built on
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3). It generates synchronized
audio and video in small chunks and supports continuous long-form generation at
480p/768p/1080p resolutions.

Developed by the **[Alibaba TaoLive AIGC Team](https://github.com/TaoLiveAIGC)**. Powered by [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3).

## Features

- **Three-step LoRA streaming generation** — each small chunk uses three
  Stage3 denoising intervals.
- **Audio-video joint generation** — speech, sound, and video are generated on
  one synchronized timeline.
- **Low chunk latency** — each small chunk reaches its final latent state well
  before a full MiniMax H3 request completes.
- **Long-form continuity** — clean KV cache and integrated audio guidance preserve
  visual identity, voice, and motion across prompt boundaries.
- **Multiple resolutions** — portrait and landscape generation at 480p, 768p,
  and aligned 1080p.
- **Single-node inference** — supports 4 GPUs or 8 GPUs with TP2 and Ulysses
  sequence parallelism.

## Installation

### Requirements

- Linux
- Python 3.10 or 3.11
- NVIDIA Hopper/SM90 GPUs; 8 × H20 96 GB is the validated configuration
- CUDA 12.8 and PyTorch 2.8
- FFmpeg with H.264 and AAC support

Create the environment:

```bash
conda create -n taomate-h3 python=3.10 -y
conda activate taomate-h3

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install triton==3.4.0 vllm==0.11.1

git clone https://github.com/Dao-AILab/flash-attention.git
pip install --no-build-isolation ./flash-attention/hopper

git clone https://github.com/TaoLiveAIGC/TaoMate-H3.git
cd TaoMate-H3
pip install -e .
```

Install FFmpeg on Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Download the MiniMax H3 FL2VA model:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" "FL2VA/*" \
  --local-dir models/MiniMax-H3
```

TaoMate-H3 LoRA weights are available on
[Hugging Face](https://huggingface.co/TaoLiveAIGC/TaoMate-H3). The current
release is the step-3000 generator EMA adapter (rank 128, alpha 128).
Inference downloads it automatically to `models/TaoMate-H3` on first use.
To download it in advance:

```bash
hf download TaoLiveAIGC/TaoMate-H3 \
  --include "config.json" "adapter_config.json" "adapter_model.safetensors" \
  --local-dir models/TaoMate-H3
```

For private or gated models, run `hf auth login` with an account that has access.
Downloads also accept the standard `HF_TOKEN` environment variable.

The LoRA directory contains:

```text
models/TaoMate-H3/
├── config.json
├── adapter_config.json
└── adapter_model.safetensors
```

## Inference

TaoMate-H3 accepts either one prompt through `--prompt` or one prompt per
five-second block through `--prompt-json`.

Example prompt file:

```json
{
  "prompts": [
    "prompt for seconds 0-5",
    "prompt for seconds 5-10"
  ],
  "seeds": [8301, 8301]
}
```

Run TaoMate-H3:

```bash
python -m taomate_h3 \
  --model-root models/MiniMax-H3 \
  --prompt-json examples/prompts_10s.json \
  --duration 10 \
  --resolution 768x1376 \
  --gpus 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --seed 8301 \
  --output outputs/demo_10s
```

The command runs the complete pipeline and writes the final video to
`outputs/demo_10s/video.mp4`. It starts its own local distributed workers, so
no external `torchrun` command is needed.

To use a different local LoRA, add `--adapter /path/to/adapter`.

Common resolutions:

| Format | Portrait | Landscape |
|---|---|---|
| 480p | `480x864` | `864x480` |
| 768p | `768x1376` | `1376x768` |
| 1080p | `1088x1920` | `1920x1088` |

For an exact 1080-pixel delivery edge, crop the generated 1088-pixel edge after
inference.

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `--model-root` | MiniMax H3 directory containing `FL2VA/` | Required |
| `--adapter` | Local LoRA directory; omit to download and reuse the official TaoMate-H3 adapter | `models/TaoMate-H3` |
| `--prompt` | One prompt reused for every five-second block | — |
| `--prompt-json` | JSON file with one prompt per five-second block | — |
| `--duration` | Total duration in seconds; must be a multiple of 5 | `5`, or inferred from JSON |
| `--resolution` | `WIDTHxHEIGHT`; short edge 480, 768, or 1088; both edges divisible by 32 | `768x1376` |
| `--gpus` | Local inference GPU count: `4` or `8` | `8` |
| `--devices` | Comma-separated CUDA device IDs | `0` to `gpus-1` |
| `--seed` | Authored request seed | `8301` |
| `--output` | New or empty output directory | Required |

`--prompt` and `--prompt-json` are mutually exclusive.

## Performance and Advantages

The following results were measured on one 8 × NVIDIA H20 96 GB node with
TP2 × Ulysses4, a `480x864` canvas, a 10-second output, and seed `8301`.

| Metric | TaoMate-H3 | MiniMax H3 | Improvement |
|---|---:|---:|---:|
| Pure DiT time | 14.810 s | 169.572 s | **11.45× faster** |
| First final chunk latent | 6.148 s | 170.052 s | **27.66× faster** |
| Benchmark first playable video | 17.287 s | 183.313 s | **10.60× faster** |
| Peak DiT memory allocated | 31.37 GiB | 32.03 GiB | - |

Pure DiT time excludes model loading, text encoding, VAE decoding, and media
encoding. First playable video includes Video VAE decoding and H.264
publication in the matched first-chunk publication benchmark. The table covers
the Stage3 generation path and excludes the command's internal audio preparation.
A 10-second TaoMate-H3 run contains 24 generation forwards and eight clean-KV
updates.

## License

TaoMate-H3 is released under the
[MiniMax H3 Community License Agreement](LICENSE). Use and distribution must
follow the terms of that license.

## Acknowledgements

We thank the teams and contributors behind:

- [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Triton](https://github.com/triton-lang/triton)
- [vLLM](https://github.com/vllm-project/vllm)
