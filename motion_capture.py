"""Flow of Truth 第 3.5 节中的 Mixture-of-Laplace 运动损失。

公式 (6) 对光流的 x/y 坐标分量分别计算双拉普拉斯混合负对数似然；公式 (8)
使用两个分量在其中心位置的混合概率密度构造像素级置信度。
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Reduction = Literal["none", "mean", "sum"]
Normalization = Literal["none", "max", "minmax"]
ScalarOrTensor = Union[float, Tensor]


def _compute_dtype(tensor: Tensor) -> torch.dtype:
    # exp/log 在半精度下容易溢出；AMP 时在 float32 中计算损失更稳健。
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def _parameter_map(value: ScalarOrTensor, reference: Tensor, name: str) -> Tensor:
    """将标量、[B,H,W] 或 [B,1,H,W] 参数广播到 [B,1,H,W]。"""
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    target_shape = (reference.shape[0], 1, reference.shape[2], reference.shape[3])
    try:
        return torch.broadcast_to(value, target_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} 必须可广播到 [B,1,H,W]={target_shape}，实际为 {tuple(value.shape)}"
        ) from exc


def _log_mixture_weights(
    alpha: Tensor, alpha_is_logits: bool, eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """返回 log(alpha)、log(1-alpha) 和概率形式的 alpha。"""
    if alpha_is_logits:
        return F.logsigmoid(alpha), F.logsigmoid(-alpha), torch.sigmoid(alpha)

    # 概率形式要避开 log(0)。使用计算 dtype 的 eps，兼容 AMP。
    safe_eps = max(eps, torch.finfo(alpha.dtype).eps)
    probability = alpha.clamp(safe_eps, 1.0 - safe_eps)
    return probability.log(), torch.log1p(-probability), probability


def mixture_of_laplace_nll(
    v_pred: Tensor,
    v_gt: Tensor,
    alpha: ScalarOrTensor,
    beta1: ScalarOrTensor,
    beta2: ScalarOrTensor,
    *,
    reduction: Reduction = "mean",
    valid_mask: Optional[Tensor] = None,
    alpha_is_logits: bool = False,
    beta_bounds: Optional[Tuple[float, float]] = (-20.0, 20.0),
    eps: float = 1e-6,
) -> Tensor:
    """计算论文公式 (6) 的 Mixture-of-Laplace 负对数似然。

    对第 k 个拉普拉斯分量，尺度 ``b_k = exp(beta_k)``，其单坐标密度为：

    ``p_k(e) = exp(-|e| / b_k) / (2 * b_k)``。

    两个运动坐标 dx/dy 分别计算 NLL；默认对 batch、坐标和有效像素求平均，
    与概率光流文献中的 ``1/(2HW)`` 归约一致。

    Args:
        v_pred: 预测运动向量 [B,2,H,W]。
        v_gt: 真实运动向量 [B,2,H,W]。
        alpha: 第一分量权重，可为标量、[B,H,W] 或 [B,1,H,W]。
        beta1: 第一分量的 log-scale。按 FoT 论文应固定为 0。
        beta2: 第二分量的 log-scale，通常由网络预测。
        reduction: ``none`` 返回 [B,2,H,W]；也支持 ``mean``/``sum``。
        valid_mask: 可选的有效像素 mask，[B,H,W] 或 [B,1,H,W]。
        alpha_is_logits: 为 True 时 alpha 是未过 sigmoid 的 logit，训练更稳定。
        beta_bounds: 对 log-scale 的数值安全截断；设为 None 可关闭。
        eps: 概率 alpha 的截断下界。

    注意：论文文字明确 beta1 固定为 0。保留 beta1 输入是为了忠实表达公式并方便
    消融实验；正式复现可直接传 ``beta1=0.0``。
    """
    if v_pred.ndim != 4 or v_pred.shape[1] != 2:
        raise ValueError(f"v_pred 应为 [B,2,H,W]，实际为 {tuple(v_pred.shape)}")
    if v_gt.shape != v_pred.shape:
        raise ValueError("v_gt 必须与 v_pred 形状相同")
    if v_pred.device != v_gt.device:
        raise ValueError("v_pred 与 v_gt 必须位于同一设备")
    if not v_pred.is_floating_point() or not v_gt.is_floating_point():
        raise TypeError("v_pred 与 v_gt 必须是浮点张量")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction 只能是 'none'、'mean' 或 'sum'")
    if beta_bounds is not None and beta_bounds[0] >= beta_bounds[1]:
        raise ValueError("beta_bounds 必须满足 min < max")
    if eps <= 0:
        raise ValueError("eps 必须大于 0")

    dtype = _compute_dtype(v_pred)
    pred = v_pred.to(dtype)
    target = v_gt.to(dtype)
    alpha_map = _parameter_map(alpha, pred, "alpha")
    beta1_map = _parameter_map(beta1, pred, "beta1")
    beta2_map = _parameter_map(beta2, pred, "beta2")

    if beta_bounds is not None:
        beta1_map = beta1_map.clamp(*beta_bounds)
        beta2_map = beta2_map.clamp(*beta_bounds)

    log_alpha, log_one_minus_alpha, _ = _log_mixture_weights(
        alpha_map, alpha_is_logits, eps
    )
    error = (pred - target).abs()
    log_two = math.log(2.0)

    # 在 log 域中计算每个分量，避免先 exp、相加、再 log 导致数值下溢。
    log_prob1 = (
        log_alpha - log_two - beta1_map - error * torch.exp(-beta1_map)
    )
    log_prob2 = (
        log_one_minus_alpha
        - log_two
        - beta2_map
        - error * torch.exp(-beta2_map)
    )
    nll = -torch.logsumexp(torch.stack((log_prob1, log_prob2), dim=0), dim=0)

    mask = None
    if valid_mask is not None:
        mask = _parameter_map(valid_mask, pred, "valid_mask").bool()
        mask = mask.expand_as(nll)
        nll = torch.where(mask, nll, torch.zeros_like(nll))

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if mask is None:
        return nll.mean()

    count = mask.sum()
    # 全无效 batch 返回带计算图的 0，而不是 NaN。
    return torch.where(count > 0, nll.sum() / count.clamp_min(1), nll.sum() * 0.0)


class MixtureOfLaplaceLoss(nn.Module):
    """可直接放入训练管线的公式 (6) 模块封装。"""

    def __init__(
        self,
        reduction: Reduction = "mean",
        *,
        alpha_is_logits: bool = False,
        beta_bounds: Optional[Tuple[float, float]] = (-20.0, 20.0),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha_is_logits = alpha_is_logits
        self.beta_bounds = beta_bounds
        self.eps = eps

    def forward(
        self,
        v_pred: Tensor,
        v_gt: Tensor,
        alpha: ScalarOrTensor,
        beta1: ScalarOrTensor,
        beta2: ScalarOrTensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return mixture_of_laplace_nll(
            v_pred,
            v_gt,
            alpha,
            beta1,
            beta2,
            reduction=self.reduction,
            valid_mask=valid_mask,
            alpha_is_logits=self.alpha_is_logits,
            beta_bounds=self.beta_bounds,
            eps=self.eps,
        )


def confidence_map(
    alpha: Tensor,
    beta1: ScalarOrTensor,
    beta2: ScalarOrTensor,
    *,
    alpha_is_logits: bool = False,
    normalization: Normalization = "minmax",
    beta_bounds: Optional[Tuple[float, float]] = (-20.0, 20.0),
    eps: float = 1e-6,
) -> Tensor:
    """计算公式 (8) 的像素级置信度图 [B,1,H,W]。

    未归一化值是两个拉普拉斯分量在误差为零处的混合密度：
    ``alpha/(2*exp(beta1)) + (1-alpha)/(2*exp(beta2))``。

    论文没有规定 ``Norm`` 的具体形式；默认采用每个样本空间维度上的 min-max
    归一化。也可使用最大值归一化，或用 ``none`` 返回原始中心密度。
    """
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(1)
    if alpha.ndim != 4 or alpha.shape[1] != 1:
        raise ValueError("alpha 应为 [B,1,H,W] 或 [B,H,W]")
    if not alpha.is_floating_point():
        raise TypeError("alpha 必须是浮点张量")
    if normalization not in ("none", "max", "minmax"):
        raise ValueError("normalization 只能是 'none'、'max' 或 'minmax'")
    if beta_bounds is not None and beta_bounds[0] >= beta_bounds[1]:
        raise ValueError("beta_bounds 必须满足 min < max")
    if eps <= 0:
        raise ValueError("eps 必须大于 0")

    dtype = _compute_dtype(alpha)
    alpha_map = alpha.to(dtype)
    beta1_map = _parameter_map(beta1, alpha_map, "beta1")
    beta2_map = _parameter_map(beta2, alpha_map, "beta2")
    if beta_bounds is not None:
        beta1_map = beta1_map.clamp(*beta_bounds)
        beta2_map = beta2_map.clamp(*beta_bounds)

    log_alpha, log_one_minus_alpha, _ = _log_mixture_weights(
        alpha_map, alpha_is_logits, eps
    )
    log_two = math.log(2.0)
    log_peak = torch.logsumexp(
        torch.stack(
            (
                log_alpha - log_two - beta1_map,
                log_one_minus_alpha - log_two - beta2_map,
            ),
            dim=0,
        ),
        dim=0,
    )

    if normalization == "none":
        max_log = math.log(torch.finfo(log_peak.dtype).max)
        return torch.exp(log_peak.clamp(max=max_log))

    # 减去每张图的最大 log-density 只引入公共正比例因子，不改变归一化结果。
    shifted = torch.exp(log_peak - log_peak.amax(dim=(-2, -1), keepdim=True))
    if normalization == "max":
        return shifted

    minimum = shifted.amin(dim=(-2, -1), keepdim=True)
    span = shifted.amax(dim=(-2, -1), keepdim=True) - minimum
    normalized = (shifted - minimum) / span.clamp_min(eps)
    # 常量置信图没有相对高低之分；视为所有像素同等且完全可信。
    return torch.where(span > eps, normalized, torch.ones_like(normalized))

