"""Flow of Truth 第 3.4 节中的可微像素散射算子。

论文仅说明算子 S 按二维运动场 M 对像素进行可微散射，并未给出碰撞与
边界处理的实现细节。本模块采用标准的 bilinear splatting（双线性前向映射）：
每个源像素被分配到目标连续坐标周围的四个整数像素。
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn


Reduction = Literal["sum", "mean"]


class ScatterResult(NamedTuple):
    """散射结果、累计权重以及目标位置是否被覆盖。"""

    image: Tensor
    weight: Tensor
    mask: Tensor


def _validate_inputs(image: Tensor, motion: Tensor) -> None:
    if image.ndim != 4:
        raise ValueError(f"image 应为 [B,C,H,W]，实际为 {tuple(image.shape)}")
    if motion.ndim != 4 or motion.shape[1] != 2:
        raise ValueError(f"motion 应为 [B,2,H,W]，实际为 {tuple(motion.shape)}")
    if image.shape[0] != motion.shape[0] or image.shape[2:] != motion.shape[2:]:
        raise ValueError("image 与 motion 的 B、H、W 必须一致")
    if image.device != motion.device:
        raise ValueError("image 与 motion 必须位于同一设备")
    if not image.is_floating_point() or not motion.is_floating_point():
        raise TypeError("image 与 motion 必须是浮点张量")


def scatter_operator(
    image: Tensor,
    motion: Tensor,
    *,
    reduction: Reduction = "mean",
    fill_value: float = 0.0,
    eps: float = 1e-6,
    return_aux: bool = False,
) -> Tensor | ScatterResult:
    """用像素单位的运动场对图像执行可微前向映射。

    坐标约定：``motion[:, 0]`` 是 dx（向右为正），``motion[:, 1]`` 是
    dy（向下为正）。源位置 (x, y) 被移动到 (x + dx, y + dy)。运动场以
    像素为单位，无需转换为 grid_sample 使用的 [-1, 1] 归一化坐标。

    对非整数目标坐标 (x', y')，将源像素按双线性权重散射到
    (floor/ceil(x'), floor/ceil(y')) 四个邻居。超出图像边界的贡献被丢弃。

    Args:
        image: [B,C,H,W] 输入图像。
        motion: [B,2,H,W] 前向运动场 (dx, dy)，单位为像素。
        reduction: ``"sum"`` 为标准加权 splat；``"mean"`` 会除以目标点
            的累计权重，避免多个源像素碰撞时亮度随数量增加。
        fill_value: 完全没有源像素覆盖的目标位置填充值。
        eps: ``mean`` 模式分母的数值稳定项。
        return_aux: 为 True 时同时返回累计权重 [B,1,H,W] 与覆盖 mask。

    Returns:
        默认返回 [B,C,H,W]；``return_aux=True`` 时返回 ScatterResult。

    可微性说明：scatter_add 的 value 对 image 和双线性权重可微，因此算子
    对 image 完全可微、对 motion 几乎处处可微。floor 与离散索引跨越整数
    边界时不可微，这与 grid_sample 的分段线性插值性质相同。
    """
    _validate_inputs(image, motion)
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction 只能是 'sum' 或 'mean'")
    if eps <= 0:
        raise ValueError("eps 必须大于 0")

    batch, channels, height, width = image.shape
    # 使用 image dtype 计算权重，避免混合精度时 scatter_add 的 dtype 不一致。
    flow = motion.to(dtype=image.dtype)
    coord_dtype = image.dtype

    y, x = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=coord_dtype),
        torch.arange(width, device=image.device, dtype=coord_dtype),
        indexing="ij",
    )
    target_x = x.unsqueeze(0) + flow[:, 0]
    target_y = y.unsqueeze(0) + flow[:, 1]

    x0 = torch.floor(target_x)
    y0 = torch.floor(target_y)
    x1 = x0 + 1
    y1 = y0 + 1

    # 四邻域权重。即使目标恰好落在整数坐标，另三个权重也自然为 0。
    neighbors = (
        (x0, y0, (x1 - target_x) * (y1 - target_y)),
        (x1, y0, (target_x - x0) * (y1 - target_y)),
        (x0, y1, (x1 - target_x) * (target_y - y0)),
        (x1, y1, (target_x - x0) * (target_y - y0)),
    )

    source = image.reshape(batch, channels, height * width)
    output = image.new_zeros(batch, channels, height * width)
    weight_sum = image.new_zeros(batch, 1, height * width)

    for neighbor_x, neighbor_y, weight in neighbors:
        valid = (
            (neighbor_x >= 0)
            & (neighbor_x < width)
            & (neighbor_y >= 0)
            & (neighbor_y < height)
        )
        # 无效坐标先截断成合法索引，再用 valid 将其贡献严格置零。
        linear_index = (
            neighbor_y.clamp(0, height - 1).long() * width
            + neighbor_x.clamp(0, width - 1).long()
        ).reshape(batch, 1, -1)
        safe_weight = (weight * valid.to(weight.dtype)).reshape(batch, 1, -1)

        output = output.scatter_add(
            2, linear_index.expand(-1, channels, -1), source * safe_weight
        )
        weight_sum = weight_sum.scatter_add(2, linear_index, safe_weight)

    covered = weight_sum > eps
    if reduction == "mean":
        normalized = output / weight_sum.clamp_min(eps)
        fill = torch.as_tensor(fill_value, dtype=image.dtype, device=image.device)
        output = torch.where(covered.expand(-1, channels, -1), normalized, fill)
    elif fill_value != 0.0:
        fill = torch.as_tensor(fill_value, dtype=image.dtype, device=image.device)
        output = torch.where(covered.expand(-1, channels, -1), output, fill)

    warped = output.reshape(batch, channels, height, width)
    weights = weight_sum.reshape(batch, 1, height, width)
    mask = covered.reshape(batch, 1, height, width)
    if return_aux:
        return ScatterResult(warped, weights, mask)
    return warped


class ScatterOperator(nn.Module):
    """便于插入 ``nn.Module`` 训练管线的封装。"""

    def __init__(
        self,
        reduction: Reduction = "mean",
        fill_value: float = 0.0,
        eps: float = 1e-6,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.fill_value = fill_value
        self.eps = eps
        self.return_aux = return_aux

    def forward(self, image: Tensor, motion: Tensor) -> Tensor | ScatterResult:
        return scatter_operator(
            image,
            motion,
            reduction=self.reduction,
            fill_value=self.fill_value,
            eps=self.eps,
            return_aux=self.return_aux,
        )

