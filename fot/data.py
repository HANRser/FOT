"""Image and optical-flow datasets used by FoT training."""

from __future__ import annotations

import random
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FLOW_SUFFIXES = {".flo"}
FLO_MAGIC = 202021.25


def _paths_from_source(source: str | Path, suffixes: set[str]) -> list[Path]:
    """Resolve a directory tree or a newline-delimited manifest into paths."""
    source = Path(source)
    if source.is_dir():
        files = sorted(
            path for path in source.rglob("*") if path.suffix.lower() in suffixes
        )
    elif source.is_file():
        base = source.parent
        files = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            files.append(path if path.is_absolute() else base / path)
    else:
        raise ValueError(f"数据源不存在：{source}")
    files = [path for path in files if path.suffix.lower() in suffixes]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"清单包含不存在的文件：{missing[0]}")
    if not files:
        raise ValueError(f"未在 {source} 找到支持的数据文件")
    return files


class ImageFolderDataset(Dataset[Tensor]):
    """Load RGB images from a directory or a text manifest."""

    def __init__(self, root: str | Path, size: int = 256):
        self.files = _paths_from_source(root, IMAGE_SUFFIXES)
        self.transform = v2.Compose(
            [
                v2.Resize((size, size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def read_flo(path: str | Path) -> Tensor:
    """Read a Middlebury ``.flo`` file as float32 ``[2,H,W]``."""
    path = Path(path)
    with path.open("rb") as stream:
        magic_bytes = stream.read(4)
        shape_bytes = stream.read(8)
        if len(magic_bytes) != 4 or len(shape_bytes) != 8:
            raise ValueError(f"无效或截断的 .flo 文件：{path}")
        magic = struct.unpack("<f", magic_bytes)[0]
        width, height = struct.unpack("<ii", shape_bytes)
        if magic != FLO_MAGIC or width <= 0 or height <= 0:
            raise ValueError(f"无效的 .flo 头：{path}")
        values = np.fromfile(stream, dtype="<f4", count=height * width * 2)
        if values.size != height * width * 2 or stream.read(1):
            raise ValueError(f".flo 数据长度不正确：{path}")
    flow = values.reshape(height, width, 2).copy()
    return torch.from_numpy(flow).permute(2, 0, 1)


def resize_flow(flow: Tensor, size: int | tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Resize dense flow and vector magnitudes; also return a valid mask."""
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError("flow 必须为 [2,H,W]")
    target_height, target_width = (size, size) if isinstance(size, int) else size
    source_height, source_width = flow.shape[-2:]
    valid = torch.isfinite(flow).all(dim=0, keepdim=True)
    safe_flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    resized = F.interpolate(
        safe_flow.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    resized[0].mul_(target_width / source_width)
    resized[1].mul_(target_height / source_height)
    resized_valid = F.interpolate(
        valid.float().unsqueeze(0),
        size=(target_height, target_width),
        mode="nearest",
    ).squeeze(0)
    return resized, resized_valid


class ImageMotionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Pair each cover image with flows sampled from a ground-truth flow bank."""

    def __init__(
        self,
        images: str | Path,
        flows: str | Path,
        *,
        size: int = 512,
        num_frames: int = 4,
        include_identity: bool = True,
        randomize_flows: bool = True,
    ):
        if num_frames <= 0:
            raise ValueError("num_frames 必须大于 0")
        self.images = ImageFolderDataset(images, size)
        self.flow_files = _paths_from_source(flows, FLOW_SUFFIXES)
        self.size = size
        self.num_frames = num_frames
        self.include_identity = include_identity
        self.randomize_flows = randomize_flows

    def __len__(self) -> int:
        return len(self.images)

    def _flow_indices(self, index: int, count: int) -> Iterable[int]:
        if self.randomize_flows:
            return (random.randrange(len(self.flow_files)) for _ in range(count))
        start = index * max(count, 1)
        return ((start + offset) % len(self.flow_files) for offset in range(count))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.images[index]
        real_count = self.num_frames - int(self.include_identity)
        flows: list[Tensor] = []
        masks: list[Tensor] = []
        if self.include_identity:
            flows.append(torch.zeros(2, self.size, self.size))
            masks.append(torch.ones(1, self.size, self.size))
        for flow_index in self._flow_indices(index, real_count):
            flow, valid = resize_flow(read_flo(self.flow_files[flow_index]), self.size)
            flows.append(flow)
            masks.append(valid)
        return image, torch.stack(flows), torch.stack(masks)
