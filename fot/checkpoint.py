"""Versioned and atomic training checkpoints for FoT."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    template: nn.Module,
    motion_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = float("inf"),
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a complete checkpoint atomically, avoiding half-written files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "template": template.state_dict(),
        "motion_model": motion_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "config": dict(config or {}),
    }

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    template: nn.Module,
    motion_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore modules and optional training state from a trusted checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"不支持的 checkpoint 版本 {version!r}，期望 {CHECKPOINT_VERSION}"
        )
    template.load_state_dict(payload["template"], strict=strict)
    motion_model.load_state_dict(payload["motion_model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "best_metric": float(payload.get("best_metric", float("inf"))),
        "config": dict(payload.get("config") or {}),
    }

