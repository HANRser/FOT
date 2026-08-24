from __future__ import annotations

import torch
from torch import Tensor

from flow_reversal import reverse_and_fuse


def photometric_confidence(reference: Tensor, frames: Tensor, flows: Tensor,
                           temperature: float = 0.08) -> Tensor:
    """A model-free confidence fallback based on flow-aligned photometric error."""
    from flow_reversal import backward_warp
    maps = []
    for t in range(frames.shape[1]):
        warped = backward_warp(frames[:, t], flows[:, t])
        error = (warped - reference).abs().mean(1, keepdim=True)
        maps.append(torch.exp(-error / temperature))
    return torch.stack(maps, dim=1)


def recover(frames: Tensor, flows: Tensor, confidences: Tensor) -> Tensor:
    if frames.ndim != 5 or flows.ndim != 5 or confidences.ndim != 5:
        raise ValueError("frames/flows/confidences 必须含 [B,T,...] 维度")
    return reverse_and_fuse(list(frames.unbind(1)), list(flows.unbind(1)), list(confidences.unbind(1)), fallback="mean")
