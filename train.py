"""Joint training for Template Embedding and Motion Capture.

The frozen VAE and differentiable scatter operator approximate the I2V channel
during training. Known synthetic affine flows supervise Motion Capture, and the
predicted flows/confidences are used to recover the source image.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from VAE import FrozenVAEReconstructor
from fot.checkpoint import load_checkpoint, save_checkpoint
from fot.data import ImageFolderDataset
from fot.motion_model import MotionCaptureNet
from fot.recovery import recover
from fot.synthetic_motion import AffineMotionConfig, make_synthetic_video
from motion_capture import mixture_of_laplace_nll
from template_embedding import TemplateEmbedding


@dataclass(frozen=True)
class LossWeights:
    fidelity_mse: float = 1.0
    fidelity_lpips: float = 0.1
    motion: float = 1.0
    recovery_l1: float = 1.0
    recovery_lpips: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Training image directory")
    parser.add_argument("--val-data", help="Optional validation image directory")
    parser.add_argument("--output-dir", default="checkpoints/default")
    parser.add_argument("--resume", help="Path to a last.pt/best.pt checkpoint")
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--template-channels", type=int, default=32)
    parser.add_argument("--motion-channels", type=int, default=32)
    parser.add_argument("--motion-chunk-size", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, help="Stop early for smoke tests")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lambda-fidelity-mse", type=float, default=1.0)
    parser.add_argument("--lambda-fidelity-lpips", type=float, default=0.1)
    parser.add_argument("--lambda-motion", type=float, default=1.0)
    parser.add_argument("--lambda-recovery-l1", type=float, default=1.0)
    parser.add_argument("--lambda-recovery-lpips", type=float, default=0.1)
    parser.add_argument("--max-translation", type=float, default=12.0)
    parser.add_argument("--max-rotation", type=float, default=5.0)
    parser.add_argument("--min-scale", type=float, default=0.96)
    parser.add_argument("--max-scale", type=float, default=1.04)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lpips(device: torch.device, enabled: bool) -> Optional[nn.Module]:
    if not enabled:
        return None
    import lpips

    model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    return model


def perceptual_loss(
    model: Optional[nn.Module], first: Tensor, second: Tensor
) -> Tensor:
    if model is None:
        return first.new_zeros(())
    model.eval()
    return model(first.float() * 2 - 1, second.float() * 2 - 1).mean()


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def compute_batch(
    image: Tensor,
    *,
    template: TemplateEmbedding,
    motion_model: MotionCaptureNet,
    vae: FrozenVAEReconstructor,
    lpips_model: Optional[nn.Module],
    weights: LossWeights,
    motion_config: AffineMotionConfig,
    num_frames: int,
    amp_dtype: str,
) -> tuple[Tensor, dict[str, float]]:
    with autocast_context(image.device, amp_dtype):
        protected = template(image)
        reconstructed = vae(protected)
        video, ground_truth_flow, valid_masks = make_synthetic_video(
            reconstructed, num_frames, motion_config
        )
        prediction = motion_model.forward_video(reconstructed, video)

        batch, frames = ground_truth_flow.shape[:2]
        height, width = image.shape[-2:]
        motion_loss = mixture_of_laplace_nll(
            prediction.flow.reshape(batch * frames, 2, height, width),
            ground_truth_flow.reshape(batch * frames, 2, height, width),
            prediction.alpha_logits.reshape(batch * frames, 1, height, width),
            beta1=0.0,
            beta2=prediction.beta2.reshape(batch * frames, 1, height, width),
            valid_mask=valid_masks.reshape(batch * frames, 1, height, width),
            alpha_is_logits=True,
        )
        recovered = recover(video, prediction.flow, prediction.confidence)
        fidelity_mse = F.mse_loss(protected, image)
        fidelity_lpips = perceptual_loss(lpips_model, protected, image)
        recovery_l1 = F.l1_loss(recovered, image)
        recovery_lpips = perceptual_loss(lpips_model, recovered, image)
        total = (
            weights.fidelity_mse * fidelity_mse
            + weights.fidelity_lpips * fidelity_lpips
            + weights.motion * motion_loss
            + weights.recovery_l1 * recovery_l1
            + weights.recovery_lpips * recovery_lpips
        )

    metrics = {
        "total": float(total.detach()),
        "fidelity_mse": float(fidelity_mse.detach()),
        "fidelity_lpips": float(fidelity_lpips.detach()),
        "motion": float(motion_loss.detach()),
        "recovery_l1": float(recovery_l1.detach()),
        "recovery_lpips": float(recovery_lpips.detach()),
    }
    return total, metrics


def make_loader(path: str, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = ImageFolderDataset(path, args.size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=shuffle and len(dataset) >= args.batch_size,
    )


def run_validation(
    loader: DataLoader,
    *,
    template: TemplateEmbedding,
    motion_model: MotionCaptureNet,
    vae: FrozenVAEReconstructor,
    lpips_model: Optional[nn.Module],
    weights: LossWeights,
    motion_config: AffineMotionConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    template.eval()
    motion_model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for image in loader:
            _, metrics = compute_batch(
                image.to(device, non_blocking=True),
                template=template,
                motion_model=motion_model,
                vae=vae,
                lpips_model=lpips_model,
                weights=weights,
                motion_config=motion_config,
                num_frames=args.num_frames,
                amp_dtype=args.amp_dtype,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_frames <= 0:
        raise SystemExit("epochs、batch-size 与 num-frames 必须大于 0")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_loader = make_loader(args.data, args, shuffle=True)
    val_loader = (
        make_loader(args.val_data, args, shuffle=False) if args.val_data else None
    )
    template = TemplateEmbedding(
        3,
        args.size,
        args.size,
        base_channels=args.template_channels,
        residual_scale=args.residual_scale,
    ).to(device)
    motion_model = MotionCaptureNet(
        base_channels=args.motion_channels,
        video_chunk_size=args.motion_chunk_size,
    ).to(device)
    vae = FrozenVAEReconstructor(
        model_id=args.vae_model,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
        enable_slicing=True,
    ).to(device)
    weights = LossWeights(
        fidelity_mse=args.lambda_fidelity_mse,
        fidelity_lpips=args.lambda_fidelity_lpips,
        motion=args.lambda_motion,
        recovery_l1=args.lambda_recovery_l1,
        recovery_lpips=args.lambda_recovery_lpips,
    )
    lpips_model = build_lpips(
        device, weights.fidelity_lpips > 0 or weights.recovery_lpips > 0
    )
    motion_config = AffineMotionConfig(
        max_translation=args.max_translation,
        max_rotation_degrees=args.max_rotation,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    parameters = list(template.parameters()) + list(motion_model.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16"
    )

    start_epoch, global_step, best_metric = 0, 0, float("inf")
    if args.resume:
        metadata = load_checkpoint(
            args.resume,
            template=template,
            motion_model=motion_model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = metadata["epoch"] + 1
        global_step = metadata["global_step"]
        best_metric = metadata["best_metric"]
        print(f"resumed={args.resume} epoch={start_epoch} step={global_step}")

    for epoch in range(start_epoch, args.epochs):
        template.train()
        motion_model.train()
        running: dict[str, float] = {}
        batches = 0
        for image in train_loader:
            image = image.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, metrics = compute_batch(
                image,
                template=template,
                motion_model=motion_model,
                vae=vae,
                lpips_model=lpips_model,
                weights=weights,
                motion_config=motion_config,
                num_frames=args.num_frames,
                amp_dtype=args.amp_dtype,
            )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            batches += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            if global_step % args.log_every == 0 or global_step == 1:
                summary = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                print(f"epoch={epoch + 1} step={global_step} {summary}", flush=True)
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        scheduler.step()
        train_metrics = {
            key: value / max(batches, 1) for key, value in running.items()
        }
        validation_metrics = (
            run_validation(
                val_loader,
                template=template,
                motion_model=motion_model,
                vae=vae,
                lpips_model=lpips_model,
                weights=weights,
                motion_config=motion_config,
                args=args,
                device=device,
            )
            if val_loader is not None
            else train_metrics
        )
        monitored = validation_metrics.get("total", float("inf"))
        config = {**vars(args), "loss_weights": asdict(weights)}
        save_checkpoint(
            output_dir / "last.pt",
            template=template,
            motion_model=motion_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=min(best_metric, monitored),
            config=config,
        )
        if monitored < best_metric:
            best_metric = monitored
            save_checkpoint(
                output_dir / "best.pt",
                template=template,
                motion_model=motion_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "best": best_metric,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
