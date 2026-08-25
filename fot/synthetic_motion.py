"""Synthetic affine motion supervision for differentiable FoT training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from scatter_operator import scatter_operator


@dataclass(frozen=True)
class AffineMotionConfig:
    max_translation: float = 12.0
    max_rotation_degrees: float = 5.0
    min_scale: float = 0.96
    max_scale: float = 1.04
    include_identity: bool = True

    def validate(self) -> None:
        if self.max_translation < 0 or self.max_rotation_degrees < 0:
            raise ValueError("平移和旋转范围不能为负")
        if self.min_scale <= 0 or self.min_scale > self.max_scale:
            raise ValueError("缩放范围必须满足 0 < min_scale <= max_scale")


def sample_affine_flows(
    image: Tensor,
    num_frames: int,
    config: AffineMotionConfig = AffineMotionConfig(),
) -> Tensor:
    """Return dense source-to-target affine flows as [B,T,2,H,W]."""
    if image.ndim != 4 or not image.is_floating_point():
        raise ValueError("image 必须是浮点 [B,C,H,W]")
    if num_frames <= 0:
        raise ValueError("num_frames 必须大于 0")
    config.validate()

    batch, _, height, width = image.shape
    parameter_shape = (batch, num_frames, 1, 1)
    translation_x = image.new_empty(parameter_shape).uniform_(
        -config.max_translation, config.max_translation
    )
    translation_y = image.new_empty(parameter_shape).uniform_(
        -config.max_translation, config.max_translation
    )
    angle = image.new_empty(parameter_shape).uniform_(
        -config.max_rotation_degrees, config.max_rotation_degrees
    )
    angle = torch.deg2rad(angle)
    scale = image.new_empty(parameter_shape).uniform_(config.min_scale, config.max_scale)

    if config.include_identity:
        translation_x[:, 0] = 0
        translation_y[:, 0] = 0
        angle[:, 0] = 0
        scale[:, 0] = 1

    y, x = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    centered_x = x - (width - 1) / 2
    centered_y = y - (height - 1) / 2
    cosine, sine = torch.cos(angle), torch.sin(angle)
    target_x = scale * (cosine * centered_x - sine * centered_y)
    target_y = scale * (sine * centered_x + cosine * centered_y)
    target_x = target_x + (width - 1) / 2 + translation_x
    target_y = target_y + (height - 1) / 2 + translation_y
    return torch.stack((target_x - x, target_y - y), dim=2)


def render_synthetic_video(
    image: Tensor,
    flows: Tensor,
) -> tuple[Tensor, Tensor]:
    """Forward-splat an image with known flows and return video and masks."""
    if flows.ndim != 5 or flows.shape[0] != image.shape[0] or flows.shape[2] != 2:
        raise ValueError("flows 必须为与 image 匹配的 [B,T,2,H,W]")
    if flows.shape[-2:] != image.shape[-2:]:
        raise ValueError("flows 与 image 的空间尺寸必须一致")

    frames = []
    masks = []
    for flow in flows.unbind(1):
        result = scatter_operator(image, flow, reduction="mean", return_aux=True)
        frames.append(result.image)
        masks.append(result.mask.to(dtype=image.dtype))
    return torch.stack(frames, dim=1), torch.stack(masks, dim=1)


def make_synthetic_video(
    image: Tensor,
    num_frames: int,
    config: AffineMotionConfig = AffineMotionConfig(),
) -> tuple[Tensor, Tensor, Tensor]:
    flows = sample_affine_flows(image, num_frames, config)
    video, masks = render_synthetic_video(image, flows)
    return video, flows, masks

