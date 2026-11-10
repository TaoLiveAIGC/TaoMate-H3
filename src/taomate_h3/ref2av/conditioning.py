# Modified for TaoMate-H3 Ref2AV inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Encode a prompt and ordered visual references with native H3 components."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from digitalhuman_h3.distributed import use_parallel_context
from digitalhuman_h3.model.components import (
    configure_bundled_video_vae_tile_parallel,
    read_component_config,
)
from digitalhuman_h3.model.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
from digitalhuman_h3.model.geometry import resolve_geometry
from digitalhuman_h3.model.packed_sequence import minimax_h3_packed_sequence_ref2va
from digitalhuman_h3.model.pipeline import _normalize_keyframe_latent
from digitalhuman_h3.model.presentation import TEXT_TAG, VIDEO_TAG, _text_ids, _vision_ids


def reference_size(width: int, height: int, short_edge: int) -> tuple[int, int]:
    scale = short_edge / min(width, height)
    return tuple(max(32, round(value * scale / 32) * 32) for value in (width, height))


def load_materials(references: list[dict], target_frames: int) -> list[dict]:
    materials = []
    for reference in references:
        kind, path = reference["type"], Path(reference["uri"])
        if kind == "image":
            with Image.open(path) as source:
                image = source.convert("RGB")
                image = image.resize(reference_size(*image.size, 2048), Image.Resampling.LANCZOS)
            materials.append({"kind": kind, "image": image})
        elif kind == "video":
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path),
            ]))
            stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
            width, height = int(stream["width"]), int(stream["height"])
            sar = stream.get("sample_aspect_ratio", "1:1")
            if sar not in {"N/A", "0:1"}:
                numerator, denominator = (int(value) for value in sar.split(":"))
                width = round(width * numerator / denominator)
            rotation = next((int(s["rotation"]) for s in stream.get("side_data_list", [])
                             if "rotation" in s), 0)
            if abs(rotation) % 180 == 90:
                width, height = height, width
            shape = resolve_geometry(duration_seconds=target_frames / 24,
                                     aspect_ratio=f"{width}:{height}", short_edge=768)
            # Only the visual stream is a reference; any source audio is ignored.
            raw = subprocess.check_output([
                "ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
                "-map", "0:v:0", "-an", "-vf",
                f"fps=24,scale={shape.width}:{shape.height}:flags=lanczos,setsar=1",
                "-frames:v", str(target_frames), "-threads", "2",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ])
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, shape.height, shape.width, 3)
            if not len(frames):
                raise ValueError(f"reference video has no frames: {path}")
            materials.append({"kind": kind, "frames": frames})
        else:
            raise ValueError(f"unsupported visual reference type: {kind}")
    if not materials:
        raise ValueError("provide at least one image or video reference")
    return materials


def sample_qwen_video(frames: np.ndarray) -> tuple[torch.Tensor, list[float]]:
    sampled = frames[::12]
    timestamps = [index / 2 for index in range(len(sampled))]
    if len(timestamps) % 2:
        timestamps.append(timestamps[-1])
    return (
        torch.from_numpy(sampled.copy()).permute(0, 3, 1, 2),
        [(timestamps[i] + timestamps[i + 1]) / 2 for i in range(0, len(timestamps), 2)],
    )


def presentation(tokenizer, prompt, materials, image_counts, video_counts, timestamps):
    ids, tags = [], []
    counters = {"image": 0, "video": 0}

    def text(value):
        part = _text_ids(tokenizer, value)
        ids.extend(part)
        tags.extend([TEXT_TAG] * len(part))

    def vision(count, video=False):
        part = _vision_ids(tokenizer, count)
        if video:
            part[1:-1] = [tokenizer.convert_tokens_to_ids("<|video_pad|>")] * count
        ids.extend(part)
        tags.extend([VIDEO_TAG] * len(part))

    for material in materials:
        kind = material["kind"]
        index = counters[kind]
        counters[kind] += 1
        if kind == "image":
            text(f"<Picture {index + 1}>: ")
            vision(image_counts[index])
        else:
            text(f"<Video {index + 1}>: ")
            if len(video_counts[index]) != len(timestamps[index]):
                raise ValueError("Qwen video blocks and timestamps differ")
            for count, timestamp in zip(video_counts[index], timestamps[index]):
                text(f"<{timestamp:.1f} seconds>")
                vision(count, video=True)
    text(prompt)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(tags, dtype=torch.long)


def encode_text(pipeline, prompt, materials):
    images = [m["image"] for m in materials if m["kind"] == "image"]
    sampled = [sample_qwen_video(m["frames"]) for m in materials if m["kind"] == "video"]
    kwargs, image_counts, video_counts = {}, [], []
    processor = pipeline.processor
    merge = int(processor.image_processor.merge_size) ** 2
    if images:
        out = processor.image_processor(images=images, return_tensors="pt")
        kwargs.update(pixel_values=out["pixel_values"], image_grid_thw=out["image_grid_thw"])
        image_counts = [int(grid.prod()) // merge for grid in out["image_grid_thw"]]
    if sampled:
        out = processor.video_processor(videos=[item[0] for item in sampled],
                                        do_sample_frames=False,
                                        input_data_format="channels_first", return_tensors="pt")
        kwargs.update(pixel_values_videos=out["pixel_values_videos"], video_grid_thw=out["video_grid_thw"])
        video_counts = [[int(grid[1] * grid[2]) // merge] * int(grid[0])
                        for grid in out["video_grid_thw"]]
    ids, tags = presentation(pipeline.tokenizer, prompt, materials,
                             image_counts, video_counts, [item[1] for item in sampled])
    pipeline._move_text_encoder(pipeline.device)
    return pipeline.text_encoder.encode_ids(ids, **kwargs), tags


def encode_visual(pipeline, materials, geometry, seed):
    decoder = pipeline._load_output_decoder()
    vae = pipeline.video_vae = decoder.video_vae
    configure_bundled_video_vae_tile_parallel(
        vae, group=torch.distributed.group.WORLD if pipeline.world_size > 1 else None,
        rank=pipeline.rank, world_size=pipeline.world_size,
    )
    vae.to(pipeline.device, dtype=torch.float32).eval()
    config = read_component_config(pipeline.partition_root / "video_vae")
    anchors, blocks = [], []
    devices = [pipeline.device] if pipeline.device.type == "cuda" else []
    try:
        for material in materials:
            with torch.random.fork_rng(devices=devices):
                torch.default_generator.manual_seed(42)
                if devices:
                    torch.cuda.manual_seed(42)
                if material["kind"] == "image":
                    latent = vae.encode_images(material["image"], use_fp16_latent=True)[0]
                else:
                    latent = vae.encode_videos(material["frames"], use_fp16_latent=True)[0]
            latent_t, latent_h, latent_w = map(int, latent.shape[-3:])
            block = {"kind": material["kind"], "latent_h": latent_h, "latent_w": latent_w}
            if material["kind"] == "video":
                block.update(latent_t=latent_t, ref_audio_t=0)
            blocks.append(block)
            clean = _normalize_keyframe_latent(latent, config=config).to(pipeline.device)
            if pipeline.world_size > 1:
                torch.distributed.broadcast(clean, src=0)
            anchors.append(clean.cpu())
    finally:
        vae.to("cpu")
        if devices:
            torch.cuda.empty_cache()
    # H3 restarts the same conditioning-noise seed for every reference entry.
    return torch.cat([
        minimax_h3_imgvid_cond_noise_aug_rows(
            clean, condition_shapes=[(block.get("latent_t", 1), block["latent_h"], block["latent_w"])],
            target_latent_t=geometry.video_latent_t, imgvid_cond_num_frames=len(blocks),
            generator=torch.Generator(device="cpu").manual_seed(seed), noise_aug=0.999,
        ) for clean, block in zip(anchors, blocks)
    ]), blocks


@torch.no_grad()
def prepare(pipeline, prompt: str, references: list[dict], duration: float,
            resolution: tuple[int, int], seed: int) -> dict:
    """Encode raw inputs; the returned payload contains no target or teacher data."""
    geometry = resolve_geometry(duration_seconds=duration, width=resolution[0], height=resolution[1])
    with use_parallel_context(pipeline.parallel_context):
        materials = load_materials(references, geometry.frame_count)
        hidden, tags = encode_text(pipeline, prompt, materials)
        anchors, blocks = encode_visual(pipeline, materials, geometry, seed)
        packed = minimax_h3_packed_sequence_ref2va(
            text_len=len(hidden), latent_t=geometry.video_latent_t,
            latent_h=geometry.video_latent_h, latent_w=geometry.video_latent_w,
            audio_t=geometry.audio_latent_t, ref_blocks=blocks,
        )
        packed["token_tags"][packed["text_pos"].view(-1)] = tags.cpu()
    return {
        "metadata": {"geometry": geometry.to_dict(), "reference_blocks": blocks,
                     "video_target_start": len(anchors), "audio_target_start": 0},
        "tensors": {"text_hidden": hidden.detach().cpu(), "text_tags": tags.cpu(),
                    "visual_condition_anchor": anchors.cpu()},
        "packed_sequence": packed,
    }
