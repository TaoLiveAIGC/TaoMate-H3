# TaoMate-H3

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)  [![ModelScope](https://img.shields.io/badge/ModelScope-Model-white?labelColor=%23EF3D5D)](https://modelscope.ai/models/TaoLiveAIGC/TaoMate-H3)

## 🎥 Demo Video

https://github.com/user-attachments/assets/15d276f9-02c0-49c6-b98f-98656c36164a


<br>

TaoMate-H3 brings three-step streaming generation to
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3). It supports text-to-audio-video
(T2AV), first/last-frame control (FL2AV), and reference-conditioned audio-video
generation (Ref2AV) with [TaoMate-R1](https://huggingface.co/TaoLiveAIGC/TaoMate-R1).

Developed by the **[Alibaba TaoLive AIGC Team](https://github.com/TaoLiveAIGC)**. Powered by [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3).

## 🔥 News

- **2026/11/10:** Introducing [TaoMate-R1](https://huggingface.co/TaoLiveAIGC/TaoMate-R1)
  for Ref2AV, with inference code and opening/continuation LoRA weights. Image
  and video references guide three-step generation of video and sound together. Our technical report is released on [arxiv](https://arxiv.org/abs/2611.14307). 
- **2026/11/03:** We release [TaoMate-HD](https://taoliveaigc.github.io/TaoMate-HD/),
  which uses coarse-to-fine generation to retain visual detail with less
  high-resolution computation.
- **2026/10/15:** We release [TaoMate-H3-FL2VA](https://huggingface.co/TaoLiveAIGC/TaoMate-H3-FL2VA)
  for streaming first/last-frame control with three-step generation and continuous
  audiovisual KV caching.
- **2026/09/08:** We introduce streaming T2AV generation with TaoMate-H3,
  building on [TaoMate](https://github.com/TaoLiveAIGC/TaoMate).


## 🚀 Features

- **Reference-to-audio-video generation** — image and video references guide
  three-step chunkwise generation, with separate opening and continuation LoRAs.
- **Streaming FL2AV** — first/last-frame control with three-step
  blockwise generation, persistent clean AV-KV, and a global endpoint deadline.
- **Three-step LoRA streaming generation** — each chunk uses three denoising steps.
- **Audio-video generation** — speech, sound, and video share one timeline.
  Ref2AV generates audio and video jointly; T2AV and FL2AV prepare base-model
  audio guidance automatically.
- **Low chunk latency** — each small chunk reaches its final latent state well
  before a full MiniMax H3 request completes.
- **Long-form continuity** — clean KV cache and integrated audio guidance preserve
  visual identity, voice, and motion across prompt boundaries.
- **Multiple resolutions** — portrait and landscape generation at 480p, 768p,
  and aligned 1080p.
- **Single-node inference** — supports 4 GPUs or 8 GPUs with TP2 and Ulysses
  sequence parallelism.

## 🧱 Model Downloads

| Models                |                       Download Link                                           |    Notes                      |
|-----------------------|-------------------------------------------------------------------------------|-------------------------------|
| [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)        |      🤗 [Huggingface](https://huggingface.co/MiniMaxAI/MiniMax-H3)     | Base model
| TaoMate-H3 T2AV  |      🤗 [Huggingface](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)         | Our 3-step LoRA weights for T2AV
| TaoMate-H3 FL2AV       |      🤗 [Hugging Face](https://huggingface.co/TaoLiveAIGC/TaoMate-H3-FL2VA)     | Our 3-step LoRA weights for FL2AV
| TaoMate-R1 Ref2AV |      🤗 [Hugging Face](https://huggingface.co/TaoLiveAIGC/TaoMate-R1)         | Opening and continuation LoRA weights for Ref2AV

T2AV and FL2AV each use one adapter. TaoMate-R1 uses two adapters: an opening
bank for the first chunk and a continuation bank for the remaining chunks.


## 🛠️ Installation

### Requirements

- Linux
- Python 3.10 or 3.11
- NVIDIA GPUs with BF16 support; accelerated T2AV uses Hopper/SM90
- CUDA 12.8 and PyTorch 2.8
- FFmpeg with H.264 and AAC support

Create the environment:

```bash
conda create -n taomate-h3 python=3.10 -y
conda activate taomate-h3

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/TaoLiveAIGC/TaoMate-H3.git
cd TaoMate-H3
pip install -e .
```

Accelerated T2AV additionally uses the following kernels. Ref2AV uses BF16 and
PyTorch SDPA and does not need this step:

```bash
pip install triton==3.4.0 vllm==0.11.1
git clone https://github.com/Dao-AILab/flash-attention.git
pip install --no-build-isolation ./flash-attention/hopper
```

Install FFmpeg on Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### Base model and T2AV weights

T2AV and FL2AV use the MiniMax H3 FL2VA partition below. Ref2AV uses only the
Ref2VA partition; its download command is in the Ref2AV section below.

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" "FL2VA/*" \
  --local-dir models/MiniMax-H3
```

TaoMate-H3 T2AV LoRA weights are available on
[Hugging Face](https://huggingface.co/TaoLiveAIGC/TaoMate-H3), with LoRA rank 128
and alpha 128.
T2AV inference downloads it automatically to `models/TaoMate-H3` on first use.
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

## ⭐️ T2AV Inference

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

The following **T2AV** results were measured on one 8 × NVIDIA H20 96 GB node with
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

## 🎬 FL2AV: streaming first/last-frame control

FL2AV generates audio and video from a prompt and first/last-frame constraints.
Its inference subproject lives in `src/taomate-fl2av`, with a seated presentation
example about living-room organization.

https://github.com/user-attachments/assets/caab7175-219d-4ac0-80d9-e614ce29988e

**30s mode · 770 frames / 32.083333s · 3 requests (7+7+7 blocks)**

Three-step generation with an 85-frame final block. The inference command
prepares the base-model audio used for guidance and the final soundtrack automatically.

[Preview image](assets/fl2va_seated_001_30s_comparison.jpg) · [Download video](assets/fl2va_seated_001_30s_comparison.mp4?raw=true)

| Option | Native block frames | Native output | Requests |
|---|---|---|---|
| `--time 30` | `[39] + [34] * 19 + [85]` | 770 frames / 32.083333s | Three continuous requests, seven blocks each |
| `--time 15` | `[39] + [34] * 8 + [85]` | 396 frames / 16.5s | Two continuous requests |
| `--time 10` | `[39] + [34] * 4 + [85]` | 260 frames / 10.833333s | One request, six blocks |
| `--time 5` | `[39,85]` | 124 frames / 5.166667s | One request |

All modes use 24fps, 768×1376, TP2/SP4 on eight GPUs, three-step denoising,
recent-seven clean AV-KV and one-shot whole-clip decoding.

```bash
# Download FL2AV weights
hf download TaoLiveAIGC/TaoMate-H3-FL2VA \
  --include "config.json" "adapter_config.json" \
            "adapter_model.safetensors" "LICENSE" "NOTICE" \
  --local-dir models/TaoMate-H3-FL2VA

# Run FL2AV
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/taomate-fl2av/infer.py \
  --time 30 --model-root models/MiniMax-H3 \
  --checkpoint models/TaoMate-H3-FL2VA \
  --output outputs/seated_001_30
```

To use your own first/last frames and prompts, prepare a case JSON using
`src/taomate-fl2av/examples/seated_001/case.json` as the template and add
`--case /path/to/case.json`. The result is written to
`outputs/seated_001_30/result/video.mp4`.

## 🎞️ Ref2AV: TaoMate-R1

TaoMate-R1 accepts a prompt and one or more image or video references. It uses the
MiniMax H3 **Ref2VA** base model, three denoising steps per chunk, and clean
audio-video history. The opening chunk uses the cold LoRA; subsequent chunks
use the warm LoRA. Audio is generated jointly with video.

### Example

The presenter and scene come from the first reference; the mug comes from the
second. The prompt describes the presentation and its Mandarin dialogue.

| Person and scene reference | Product reference | Generated audio-video |
|:---:|:---:|:---:|
| <img src="examples/ref2av/presenter.png" width="220" alt="Person and scene reference"> | <img src="examples/ref2av/product.png" width="220" alt="Mist-blue mug reference"> | https://github.com/user-attachments/assets/8a95695b-3e6a-410f-a82d-e74169b06173|

[Prompt](examples/ref2av_prompt.txt) · 10-second request · 768×1344 · Seed 9111.
Click the video link to play it with sound.

### Run

Use the environment installed above. Download the Ref2VA base-model partition:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "Ref2VA/*" \
  --local-dir models/MiniMax-H3

hf download TaoLiveAIGC/TaoMate-R1 --local-dir models/TaoMate-R1
```

The downloaded adapter layout is:

```text
models/TaoMate-R1/
├── config.json
├── cold/
│   ├── config.json
│   └── adapter_model.safetensors
└── warm/
    ├── config.json
    └── adapter_model.safetensors
```

```bash
python -m taomate_h3.ref2av \
  --model-root models/MiniMax-H3 \
  --adapter models/TaoMate-R1 \
  --prompt-file examples/ref2av_prompt.txt \
  --reference-image examples/ref2av/presenter.png \
  --reference-image examples/ref2av/product.png \
  --duration 10 --resolution 768x1344 --seed 9111 \
  --gpus 8 --output outputs/ref2av
```

The command encodes the references and prompt, then generates and decodes the
complete audio-video sequence to `outputs/ref2av/video.mp4`. Distributed workers
are started automatically. Repeat `--reference-image` or `--reference-video`
to provide multiple references; their command-line order is preserved. Prompt
labels use `<Picture 1>`, `<Picture 2>` and `<Video 1>`. Reference-video audio
is ignored; these inputs provide visual conditioning only.

Shared options such as `--model-root`, `--adapter`, `--gpus`, `--devices`,
`--seed`, `--resolution` and `--output` have the same meanings as above. Ref2AV
uses `--prompt` or `--prompt-file` for a single prompt, with a default seed of
42 and duration of 10 seconds. Set duration in seconds and resolution as
`WIDTHxHEIGHT`, with both dimensions divisible by 32.
The output follows H3's native frame alignment: a 10-second request produces
243 frames at 24 FPS (10.125 seconds).

## License

TaoMate-H3 is released under the
[MiniMax H3 Community License Agreement](LICENSE). Use and distribution must
follow the terms of that license.

## Citation

If you find our work helpful, feel free to cite us.

```bibtex
@misc{gan2026taomater1temporalrolespecialization,
      title={TaoMate-R1: Temporal Role Specialization for Streaming Multi-Reference Video Generation}, 
      author={Qijun Gan and Chenwei Zhang and Xinyang Li and Meiguang Jin and Junfeng Ma},
      year={2026},
      eprint={2611.14307},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.14307}, 
}

@misc{taomate2026,
      title={TaoMate: Anchor-Guided Memory Bridging Evolving and Reference States for Real-Time Audio-Video Digital Human Generation}, 
      author={Qijun Gan and Chenwei Zhang and Meiguang Jin and Junfeng Ma and Qiu Shen},
      year={2026},
      eprint={2607.24359},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.24359}, 
}
```

## Acknowledgements

We thank the teams and contributors behind:

- [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Triton](https://github.com/triton-lang/triton)
- [vLLM](https://github.com/vllm-project/vllm)
- [Self Forcing](https://github.com/guandeh17/Self-Forcing)
