"""Flow of Truth 第 3.6 节：光流回溯与置信度引导的多帧融合。"""

from __future__ import annotations

from typing import List, Literal, NamedTuple, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


PaddingMode = Literal["zeros", "border", "reflection"]
Fallback = Literal["zeros", "mean"]


class BackwardWarpResult(NamedTuple):
    image: Tensor
    valid_mask: Tensor


class FusionResult(NamedTuple):
    image: Tensor
    weight_sum: Tensor


def _normalize_pixel_grid(
    x: Tensor,
    y: Tensor,
    height: int,
    width: int,
    align_corners: bool,
) -> Tensor:
    """将像素坐标转换为 grid_sample 使用的 [-1,1] 坐标。"""
    if align_corners:
        # 单像素维度没有左右两个角点，grid_sample 规定中心坐标为 0。
        grid_x = 2.0 * x / (width - 1) - 1.0 if width > 1 else torch.zeros_like(x)
        grid_y = 2.0 * y / (height - 1) - 1.0 if height > 1 else torch.zeros_like(y)
    else:
        grid_x = 2.0 * (x + 0.5) / width - 1.0
        grid_y = 2.0 * (y + 0.5) / height - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def backward_warp(
    frame: Tensor,
    flow_0_to_t: Tensor,
    *,
    padding_mode: PaddingMode = "zeros",
    align_corners: bool = True,
    return_valid_mask: bool = False,
) -> Union[Tensor, BackwardWarpResult]:
    """按公式 (10) 将第 t 帧反向采样到原图坐标系。

    对原图坐标 ``(x,y)``，从视频帧的
    ``(x + flow_x(x,y), y + flow_y(x,y))`` 位置进行双线性采样：

    ``I_t_to_0(x,y) = I_t(x + F_x(x,y), y + F_y(x,y))``。

    Args:
        frame: 第 t 帧 ``I_t``，[B,C,H,W]。
        flow_0_to_t: 预测前向光流 ``F_0_to_t``，[B,2,H,W]，单位为像素；
            第 0 通道为 dx（向右为正），第 1 通道为 dy（向下为正）。
        padding_mode: 越界采样方式，语义与 ``grid_sample`` 一致。
        align_corners: 坐标归一化方式；默认 True，使整数像素坐标精确对齐角点。
        return_valid_mask: 是否同时返回采样中心位于图像范围内的 mask。

    Returns:
        默认返回回溯图像 [B,C,H,W]；启用 ``return_valid_mask`` 时返回
        ``BackwardWarpResult(image, valid_mask)``，其中 mask 为 [B,1,H,W]。

    该操作对 frame 和亚像素 flow 几乎处处可微。
    """
    if frame.ndim != 4:
        raise ValueError(f"frame 应为 [B,C,H,W]，实际为 {tuple(frame.shape)}")
    if flow_0_to_t.ndim != 4 or flow_0_to_t.shape[1] != 2:
        raise ValueError(
            f"flow_0_to_t 应为 [B,2,H,W]，实际为 {tuple(flow_0_to_t.shape)}"
        )
    if frame.shape[0] != flow_0_to_t.shape[0] or frame.shape[2:] != flow_0_to_t.shape[2:]:
        raise ValueError("frame 与 flow_0_to_t 的 B、H、W 必须一致")
    if frame.device != flow_0_to_t.device:
        raise ValueError("frame 与 flow_0_to_t 必须位于同一设备")
    if not frame.is_floating_point() or not flow_0_to_t.is_floating_point():
        raise TypeError("frame 与 flow_0_to_t 必须是浮点张量")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("padding_mode 只能是 'zeros'、'border' 或 'reflection'")

    batch, _, height, width = frame.shape
    # grid_sample 要求输入与 grid dtype 一致；to() 不会切断 flow 的梯度。
    flow = flow_0_to_t.to(dtype=frame.dtype)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=frame.dtype, device=frame.device),
        torch.arange(width, dtype=frame.dtype, device=frame.device),
        indexing="ij",
    )
    sample_x = x.unsqueeze(0) + flow[:, 0]
    sample_y = y.unsqueeze(0) + flow[:, 1]
    sampling_grid = _normalize_pixel_grid(
        sample_x, sample_y, height, width, align_corners
    )

    reversed_frame = F.grid_sample(
        frame,
        sampling_grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if not return_valid_mask:
        return reversed_frame

    valid = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    ).unsqueeze(1)
    return BackwardWarpResult(reversed_frame, valid)


def _stack_images(images: Union[Sequence[Tensor], Tensor]) -> Tensor:
    if isinstance(images, Tensor):
        if images.ndim != 5:
            raise ValueError("堆叠图像应为 [B,T,C,H,W]")
        return images
    if len(images) == 0:
        raise ValueError("warped_images 不能为空")
    if any(image.ndim != 4 for image in images):
        raise ValueError("列表中的每幅图像都必须是 [B,C,H,W]")
    try:
        return torch.stack(tuple(images), dim=1)
    except RuntimeError as exc:
        raise ValueError("所有回溯图像的形状、设备和 dtype 必须一致") from exc


def _stack_confidences(
    confidences: Union[Sequence[Tensor], Tensor],
    image_stack: Tensor,
) -> Tensor:
    if isinstance(confidences, Tensor):
        confidence_stack = confidences
        # 允许 [B,T,H,W]，自动补上单通道维。
        if confidence_stack.ndim == 4:
            confidence_stack = confidence_stack.unsqueeze(2)
        if confidence_stack.ndim != 5:
            raise ValueError("堆叠置信图应为 [B,T,1,H,W] 或 [B,T,H,W]")
    else:
        if len(confidences) == 0:
            raise ValueError("confidence_maps 不能为空")
        canonical: List[Tensor] = []
        for confidence in confidences:
            if confidence.ndim == 3:
                confidence = confidence.unsqueeze(1)
            if confidence.ndim != 4 or confidence.shape[1] != 1:
                raise ValueError("每幅置信图应为 [B,1,H,W] 或 [B,H,W]")
            canonical.append(confidence)
        try:
            confidence_stack = torch.stack(tuple(canonical), dim=1)
        except RuntimeError as exc:
            raise ValueError("所有置信图的形状、设备和 dtype 必须一致") from exc

    expected = (
        image_stack.shape[0],
        image_stack.shape[1],
        1,
        image_stack.shape[3],
        image_stack.shape[4],
    )
    if confidence_stack.shape != expected:
        raise ValueError(
            f"置信图堆叠形状应为 {expected}，实际为 {tuple(confidence_stack.shape)}"
        )
    if confidence_stack.device != image_stack.device:
        raise ValueError("图像与置信图必须位于同一设备")
    if not confidence_stack.is_floating_point():
        raise TypeError("置信图必须是浮点张量")
    return confidence_stack


def confidence_weighted_fusion(
    warped_images: Union[Sequence[Tensor], Tensor],
    confidence_maps: Union[Sequence[Tensor], Tensor],
    *,
    eps: float = 1e-6,
    fallback: Fallback = "zeros",
    return_weight_sum: bool = False,
) -> Union[Tensor, FusionResult]:
    """按公式 (11) 对多幅回溯图像做逐像素置信度加权融合。

    输入既可为长度 T 的列表（每项 [B,C,H,W]/[B,1,H,W]），也可直接为
    ``images [B,T,C,H,W]`` 和 ``confidences [B,T,1,H,W]``。

    负置信度没有概率意义，会被截为 0。若某像素的总权重为 0，默认填 0；
    ``fallback="mean"`` 可改为使用所有回溯帧的非加权均值。
    """
    if eps <= 0:
        raise ValueError("eps 必须大于 0")
    if fallback not in ("zeros", "mean"):
        raise ValueError("fallback 只能是 'zeros' 或 'mean'")

    images = _stack_images(warped_images)
    if not images.is_floating_point():
        raise TypeError("回溯图像必须是浮点张量")
    confidences = _stack_confidences(confidence_maps, images)
    weights = confidences.to(dtype=images.dtype).clamp_min(0.0)

    numerator = (images * weights).sum(dim=1)
    denominator = weights.sum(dim=1)  # [B,1,H,W]，自动广播到 C 通道。
    fused = numerator / denominator.clamp_min(eps)

    if fallback == "mean":
        fallback_image = images.mean(dim=1)
    else:
        fallback_image = torch.zeros_like(fused)
    fused = torch.where(denominator > eps, fused, fallback_image)

    if return_weight_sum:
        return FusionResult(fused, denominator)
    return fused


def reverse_and_fuse(
    frames: Sequence[Tensor],
    flows_0_to_t: Sequence[Tensor],
    confidence_maps: Sequence[Tensor],
    *,
    padding_mode: PaddingMode = "zeros",
    align_corners: bool = True,
    eps: float = 1e-6,
    fallback: Fallback = "zeros",
) -> Tensor:
    """便捷接口：依次执行公式 (10)，再执行公式 (11)。

    回溯越界位置的有效 mask 会自动乘入置信度，避免 padding 值参与融合。
    """
    if not (len(frames) == len(flows_0_to_t) == len(confidence_maps)):
        raise ValueError("frames、flows_0_to_t、confidence_maps 的长度必须相同")
    if len(frames) == 0:
        raise ValueError("至少需要一帧")

    warped: List[Tensor] = []
    effective_confidences: List[Tensor] = []
    for frame, flow, confidence in zip(frames, flows_0_to_t, confidence_maps):
        result = backward_warp(
            frame,
            flow,
            padding_mode=padding_mode,
            align_corners=align_corners,
            return_valid_mask=True,
        )
        if confidence.ndim == 3:
            confidence = confidence.unsqueeze(1)
        if confidence.ndim != 4 or confidence.shape[1] != 1:
            raise ValueError("每幅置信图应为 [B,1,H,W] 或 [B,H,W]")
        warped.append(result.image)
        effective_confidences.append(confidence * result.valid_mask.to(confidence.dtype))

    return confidence_weighted_fusion(
        warped,
        effective_confidences,
        eps=eps,
        fallback=fallback,
    )

