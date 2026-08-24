from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(a: Tensor, b: Tensor, data_range: float = 1.0) -> Tensor:
    mse = F.mse_loss(a, b)
    return 10 * torch.log10(a.new_tensor(data_range ** 2) / mse.clamp_min(1e-12))


def ssim(a: Tensor, b: Tensor, data_range: float = 1.0) -> Tensor:
    # Global SSIM is dependency-free and deterministic; evaluation CLI reports its mean.
    dims = (-2, -1)
    ux, uy = a.mean(dims, keepdim=True), b.mean(dims, keepdim=True)
    vx, vy = ((a-ux)**2).mean(dims), ((b-uy)**2).mean(dims)
    cov = ((a-ux)*(b-uy)).mean(dims)
    c1, c2 = (0.01*data_range)**2, (0.03*data_range)**2
    return (((2*ux.squeeze(-1).squeeze(-1)*uy.squeeze(-1).squeeze(-1)+c1)*(2*cov+c2)) /
            ((ux.squeeze(-1).squeeze(-1)**2+uy.squeeze(-1).squeeze(-1)**2+c1)*(vx+vy+c2))).mean()


def evaluate(reference: Tensor, recovered: Tensor, lpips_model=None) -> dict[str, float]:
    out = {"psnr": float(psnr(reference, recovered)), "ssim": float(ssim(reference, recovered))}
    if lpips_model is not None:
        out["lpips"] = float(lpips_model(reference*2-1, recovered*2-1).mean())
    return out
