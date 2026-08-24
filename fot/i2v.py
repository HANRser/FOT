from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class I2VGenerator:
    """Stable Video Diffusion adapter with a deterministic offline fallback."""

    def __init__(self, model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt", *,
                 device: Optional[str] = None, dtype: torch.dtype = torch.float16,
                 local_files_only: bool = False, mock: bool = False) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.mock = mock
        self.pipe = None
        if not mock:
            try:
                from diffusers import StableVideoDiffusionPipeline
                self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=self.dtype, local_files_only=local_files_only,
                ).to(self.device)
                self.pipe.enable_model_cpu_offload() if self.device.type == "cuda" else None
            except Exception as exc:
                raise RuntimeError("SVD 加载失败；离线自检可使用 mock=True。") from exc

    def generate(self, image: Tensor, *, num_frames: int = 14, fps: int = 7,
                 motion_bucket_id: int = 127, noise_aug_strength: float = 0.02,
                 seed: int = 0) -> Tensor:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("image 必须为 [1,3,H,W]")
        image = image.clamp(0, 1)
        if self.mock:
            # Differentiable synthetic motion, useful for CI and pipeline debugging.
            frames = []
            for t in range(num_frames):
                dx = 3.0 * t / max(num_frames - 1, 1)
                theta = image.new_tensor([[[1, 0, 2 * dx / image.shape[-1]], [0, 1, 0]]])
                grid = F.affine_grid(theta, image.shape, align_corners=False)
                frames.append(F.grid_sample(image, grid, align_corners=False, padding_mode="border"))
            return torch.stack(frames, dim=1)
        from diffusers.utils import export_to_video  # noqa: F401 (ensures image deps)
        from torchvision.transforms.functional import to_pil_image
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(to_pil_image(image[0].cpu()), num_frames=num_frames, fps=fps,
                           motion_bucket_id=motion_bucket_id,
                           noise_aug_strength=noise_aug_strength, generator=generator)
        frames = [torch.from_numpy(__import__("numpy").array(x)).permute(2, 0, 1).float() / 255 for x in result.frames[0]]
        return torch.stack(frames, dim=0).unsqueeze(0).to(self.device)
