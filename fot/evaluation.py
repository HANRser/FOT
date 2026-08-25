"""Single-pair and directory-level image quality evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch import Tensor, nn
from torchvision.transforms.functional import pil_to_tensor

from .metrics import psnr


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _tensor(image: Image.Image, device: torch.device) -> Tensor:
    return pil_to_tensor(image.convert("RGB")).float().div(255).unsqueeze(0).to(device)


class ImageEvaluator:
    def __init__(
        self,
        *,
        device: Optional[str] = None,
        use_lpips: bool = False,
        use_clip: bool = False,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = False,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.lpips_model: Optional[nn.Module] = None
        self.clip_model = None
        self.clip_processor = None
        if use_lpips:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
            self.lpips_model.requires_grad_(False)
        if use_clip:
            from transformers import CLIPImageProcessor, CLIPModel

            self.clip_processor = CLIPImageProcessor.from_pretrained(
                clip_model_id, local_files_only=local_files_only
            )
            self.clip_model = CLIPModel.from_pretrained(
                clip_model_id, local_files_only=local_files_only
            ).to(self.device).eval()
            self.clip_model.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, reference: Image.Image, recovered: Image.Image) -> dict[str, float]:
        reference = reference.convert("RGB")
        recovered = recovered.convert("RGB")
        if reference.size != recovered.size:
            raise ValueError(
                f"reference 与 recovered 尺寸不一致：{reference.size} vs {recovered.size}"
            )
        first = _tensor(reference, self.device)
        second = _tensor(recovered, self.device)
        first_array = np.asarray(reference, dtype=np.float32) / 255.0
        second_array = np.asarray(recovered, dtype=np.float32) / 255.0
        result = {
            "psnr": float(psnr(first, second)),
            "ssim": float(
                structural_similarity(
                    first_array,
                    second_array,
                    channel_axis=2,
                    data_range=1.0,
                )
            ),
        }
        if self.lpips_model is not None:
            result["lpips"] = float(
                self.lpips_model(first * 2 - 1, second * 2 - 1).mean()
            )
        if self.clip_model is not None and self.clip_processor is not None:
            pixels = self.clip_processor(
                images=[reference, recovered], return_tensors="pt"
            ).pixel_values.to(self.device)
            features = self.clip_model.get_image_features(pixel_values=pixels)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            result["clip_similarity"] = float((features[0] * features[1]).sum())
        return result


def pair_directories(
    reference_dir: str | Path, recovered_dir: str | Path
) -> list[tuple[str, Path, Path]]:
    reference_root = Path(reference_dir)
    recovered_root = Path(recovered_dir)
    references = {
        path.relative_to(reference_root).as_posix(): path
        for path in reference_root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    recovered = {
        path.relative_to(recovered_root).as_posix(): path
        for path in recovered_root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = sorted(set(references) - set(recovered))
    extra = sorted(set(recovered) - set(references))
    if missing or extra:
        raise ValueError(
            f"目录文件不匹配；缺少 recovered={missing[:5]}，多余 recovered={extra[:5]}"
        )
    if not references:
        raise ValueError("没有找到可测评图片")
    return [(name, references[name], recovered[name]) for name in sorted(references)]


def aggregate(rows: Iterable[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    materialized = list(rows)
    metric_names = [key for key in materialized[0] if key != "name"]
    summary = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in materialized])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def save_report(
    output_json: str | Path,
    rows: list[dict[str, float | str]],
    summary: dict[str, dict[str, float]],
) -> tuple[Path, Path]:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = json_path.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {"count": len(rows), "summary": summary, "per_image": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path

