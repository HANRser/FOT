"""Trainable motion and uncertainty estimator used by Flow of Truth."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from motion_capture import confidence_map


def _groups(channels: int, maximum: int = 8) -> int:
    groups = min(channels, maximum)
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MotionPrediction(NamedTuple):
    flow: Tensor
    alpha_logits: Tensor
    beta2: Tensor
    confidence: Tensor


class VideoMotionPrediction(NamedTuple):
    flow: Tensor
    alpha_logits: Tensor
    beta2: Tensor
    confidence: Tensor


class MotionCaptureNet(nn.Module):
    """U-Net that predicts flow and the two Laplace-mixture parameters.

    The input is a protected/reconstructed reference and one generated frame.
    The output follows the conventions in :mod:`motion_capture`: ``flow`` is in
    pixels, ``alpha_logits`` is the first mixture weight before sigmoid, and
    ``beta2`` is the second component's log-scale. ``beta1`` stays fixed at 0.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        beta_bounds: tuple[float, float] = (-6.0, 6.0),
        video_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        if image_channels <= 0 or base_channels <= 0:
            raise ValueError("image_channels 与 base_channels 必须为正数")
        if beta_bounds[0] >= beta_bounds[1]:
            raise ValueError("beta_bounds 必须满足 min < max")
        if video_chunk_size <= 0:
            raise ValueError("video_chunk_size 必须大于 0")
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.beta_bounds = beta_bounds
        self.video_chunk_size = video_chunk_size

        b = base_channels
        self.enc0 = ConvBlock(2 * image_channels, b)
        self.enc1 = ConvBlock(b, 2 * b, stride=2)
        self.enc2 = ConvBlock(2 * b, 4 * b, stride=2)
        self.bottleneck = ConvBlock(4 * b, 4 * b)
        self.dec1 = ConvBlock(6 * b, 2 * b)
        self.dec0 = ConvBlock(3 * b, b)
        self.flow_head = nn.Conv2d(b, 2, 3, padding=1)
        self.alpha_head = nn.Conv2d(b, 1, 3, padding=1)
        self.beta2_head = nn.Conv2d(b, 1, 3, padding=1)

        # Start with zero flow and a broad second Laplace component.
        for head in (self.flow_head, self.alpha_head, self.beta2_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.beta2_head.bias, 1.0)

    def forward(self, reference: Tensor, frame: Tensor) -> MotionPrediction:
        if reference.shape != frame.shape or reference.ndim != 4:
            raise ValueError("reference 与 frame 必须同为 [B,C,H,W] 且形状一致")
        if reference.shape[1] != self.image_channels:
            raise ValueError(f"输入通道数应为 {self.image_channels}")
        if reference.device != frame.device:
            raise ValueError("reference 与 frame 必须位于同一设备")

        x0 = self.enc0(torch.cat((reference, frame), dim=1))
        x1 = self.enc1(x0)
        x2 = self.bottleneck(self.enc2(x1))
        x = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((x, x1), dim=1))
        x = F.interpolate(x, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec0(torch.cat((x, x0), dim=1))

        flow = self.flow_head(x)
        alpha_logits = self.alpha_head(x)
        beta2 = self.beta2_head(x).clamp(*self.beta_bounds)
        confidence = confidence_map(
            alpha_logits,
            beta1=0.0,
            beta2=beta2,
            alpha_is_logits=True,
            normalization="minmax",
            beta_bounds=self.beta_bounds,
        )
        return MotionPrediction(flow, alpha_logits, beta2, confidence)

    def forward_video(self, reference: Tensor, video: Tensor) -> VideoMotionPrediction:
        if video.ndim != 5:
            raise ValueError("video 必须为 [B,T,C,H,W]")
        if (
            reference.ndim != 4
            or reference.shape[0] != video.shape[0]
            or reference.shape[1:] != video.shape[2:]
        ):
            raise ValueError("reference 与 video 的 B、C、H、W 必须一致")

        batch, frames, channels, height, width = video.shape
        chunks: list[MotionPrediction] = []
        for start in range(0, frames, self.video_chunk_size):
            end = min(start + self.video_chunk_size, frames)
            count = end - start
            frame_chunk = video[:, start:end]
            reference_chunk = reference.unsqueeze(1).expand(-1, count, -1, -1, -1)
            prediction = self.forward(
                reference_chunk.reshape(batch * count, channels, height, width),
                frame_chunk.reshape(batch * count, channels, height, width),
            )

            def restore(tensor: Tensor) -> Tensor:
                return tensor.reshape(batch, count, tensor.shape[1], height, width)

            chunks.append(MotionPrediction(*(restore(tensor) for tensor in prediction)))

        return VideoMotionPrediction(
            *(torch.cat([chunk[index] for chunk in chunks], dim=1) for index in range(4))
        )
