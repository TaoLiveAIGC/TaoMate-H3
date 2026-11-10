"""Chunk layouts and clean-history attention for Ref2AV."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from digitalhuman_h3.model.architecture import MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
from digitalhuman_h3.model.geometry import video_latent_t
from digitalhuman_h3.model.layers import native_stream_attention_metadata


def chunk_slices(geometry):
    """Use the native 39-frame opening and 34-frame continuation chunks."""
    frames = geometry["frame_count"]
    pixels = (geometry["video_latent_h"] // 2) * (geometry["video_latent_w"] // 2)
    audio_t = geometry["audio_latent_t"]
    start_frame = start_video = start_audio = 0
    while start_frame < frames:
        end_frame = min(frames, 39 if start_frame == 0 else start_frame + 34)
        end_video = video_latent_t(end_frame)
        end_audio = round(end_frame * 40 / 24)
        audio_ids = torch.cat([
            torch.arange(channel * audio_t + start_audio, channel * audio_t + end_audio)
            for channel in range(2)
        ])
        yield slice(start_video * pixels, end_video * pixels), audio_ids
        start_frame, start_video, start_audio = end_frame, end_video, end_audio


def compact_layout(payload, video_slice, audio_ids):
    """Gather a chunk without resetting text, reference or target RoPE."""
    original = payload["packed_sequence"]
    reference_count = payload["metadata"]["video_target_start"]
    text = original["text_pos"].view(-1).long()
    references = original["img_pos"].view(-1).long()[:reference_count]
    audio = original["audio_pos"].view(-1).long()[audio_ids]
    video = original["img_pos"].view(-1).long()[reference_count:][video_slice]
    selected = torch.cat((text, references, audio, video))
    used = selected.numel()
    alignment = MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    length = (used + alignment - 1) // alignment * alignment
    grid = torch.zeros(length, 3, dtype=original["img_position_ids"].dtype)
    grid[:used] = original["img_position_ids"][selected]
    tags = torch.full((length,), -1, dtype=torch.long)
    tags[:used] = original["token_tags"][selected]
    nt, nr, na = len(text), len(references), len(audio)
    image_positions = torch.cat((torch.arange(nt, nt + nr), torch.arange(nt + nr + na, used)))
    return dict(
        seq_len=length, text_pos=torch.arange(nt), img_pos=image_positions,
        audio_pos=torch.arange(nt + nr, nt + nr + na),
        update_mask=torch.arange(len(image_positions)) >= nr,
        audio_update_mask=torch.ones(na, dtype=torch.bool),
        img_position_ids=grid, token_tags=tags,
        cu_seqlens=torch.tensor([0, used, length], dtype=torch.int32),
    )


def _sdpa(query, key, value, scale):
    backends = [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    kernel = torch.nn.attention.sdpa_kernel(backends) if query.is_cuda else nullcontext()
    with kernel:
        result = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=scale,
        )
    return result[0].transpose(0, 1)


class CleanHistoryAttention:
    """Keep the opening video sink and the two latest full AV chunks."""

    def __init__(self):
        self.sink = {}
        self.recent = []
        self.commits = 0
        self.capture = False
        self.captured = {}

    @contextmanager
    def execution(self, packed, *, clean=False, device):
        used = int(packed["cu_seqlens"][1])
        commit_mask = torch.zeros(packed["seq_len"], dtype=torch.bool, device=device)
        commit_mask[packed["audio_pos"].to(device)] = True
        commit_mask[packed["img_pos"][packed["update_mask"]].to(device)] = True
        self.current = torch.nonzero(commit_mask, as_tuple=False).flatten()
        self.condition = torch.nonzero(~commit_mask[:used], as_tuple=False).flatten()
        self.video_offset = len(packed["audio_pos"])
        self.capture = clean
        self.captured = {}
        try:
            with native_stream_attention_metadata(packed["token_tags"].to(device), commit_mask):
                yield
            if clean:
                if self.commits == 0:
                    self.sink = {
                        name: (key[self.video_offset:], value[self.video_offset:])
                        for name, (key, value) in self.captured.items()
                    }
                self.recent.append(self.captured)
                self.recent = self.recent[-2:]
                self.commits += 1
        finally:
            self.capture = False
            self.captured = {}

    def __call__(self, *, attention: object, layer_name, query, key, value, **kwargs):
        condition_k = key.index_select(0, self.condition)
        condition_v = value.index_select(0, self.condition)
        current_k = key.index_select(0, self.current)
        current_v = value.index_select(0, self.current)
        history = ([self.sink] if self.commits > 2 else []) + self.recent
        visible_k = torch.cat((condition_k, *(block[layer_name][0] for block in history), current_k))
        visible_v = torch.cat((condition_v, *(block[layer_name][1] for block in history), current_v))
        output = torch.zeros_like(query)
        scale = float(attention.softmax_scale)
        output.index_copy_(0, self.condition, _sdpa(
            query.index_select(0, self.condition), condition_k, condition_v, scale))
        output.index_copy_(0, self.current, _sdpa(
            query.index_select(0, self.current), visible_k, visible_v, scale))
        if self.capture:
            self.captured[layer_name] = (current_k.detach(), current_v.detach())
        return output

    def clear(self):
        self.sink.clear()
        self.recent.clear()
        self.captured.clear()
