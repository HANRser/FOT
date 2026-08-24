from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RAFTEstimator(nn.Module):
    """Torchvision RAFT wrapper returning flow from the reference to every frame."""

    def __init__(self, small: bool = True, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models.optical_flow import (
            Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
        )
        weights = (Raft_Small_Weights.DEFAULT if small else Raft_Large_Weights.DEFAULT) if pretrained else None
        self.weights = weights
        self.model = (raft_small if small else raft_large)(weights=weights, progress=True).eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _pad(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        return F.pad(x, (0, pw, 0, ph), mode="replicate"), (h, w)

    def pair(self, reference: Tensor, frame: Tensor) -> Tensor:
        if reference.shape != frame.shape or reference.ndim != 4:
            raise ValueError("reference/frame 必须是相同形状 [B,3,H,W]")
        a, (h, w) = self._pad(reference)
        b, _ = self._pad(frame)
        # torchvision RAFT expects [-1, 1].
        return self.model(a * 2 - 1, b * 2 - 1)[-1][..., :h, :w]

    def forward(self, reference: Tensor, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError("video 必须为 [B,T,3,H,W]")
        return torch.stack([self.pair(reference, video[:, t]) for t in range(video.shape[1])], dim=1)
