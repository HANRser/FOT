"""Run checkpoint-backed FoT inference over a frozen image directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from run_demo import build


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUTPUT_NAMES = ("protected", "forged", "recovered", "confidence")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--template-channels", type=int, default=32)
    parser.add_argument("--motion-channels", type=int, default=32)
    parser.add_argument("--motion-chunk-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    images = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "run.json"
    existing: dict[str, object] = {}
    if args.resume and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
    prior_rows = {
        str(row["name"]): row for row in existing.get("images", [])  # type: ignore[union-attr]
    }
    report: dict[str, object] = {
        "started_at": existing.get("started_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "git_revision": git_revision(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "dataset_manifest": (
            str(args.dataset_manifest.resolve()) if args.dataset_manifest else None
        ),
        "dataset_manifest_sha256": (
            digest(args.dataset_manifest) if args.dataset_manifest else None
        ),
        "settings": {
            "size": args.size,
            "template_channels": args.template_channels,
            "motion_channels": args.motion_channels,
            "motion_chunk_size": args.motion_chunk_size,
            "mock": args.mock,
            "local_files_only": args.local_files_only,
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "images": list(prior_rows.values()),
    }
    save_json(report_path, report)

    runner = build(
        args.mock,
        args.size,
        checkpoint=str(args.checkpoint),
        template_channels=args.template_channels,
        motion_channels=args.motion_channels,
        motion_chunk_size=args.motion_chunk_size,
        local_files_only=args.local_files_only,
    )
    for index, input_path in enumerate(images, start=1):
        relative = input_path.relative_to(args.input_dir).with_suffix(".png")
        outputs = {
            name: args.output_dir / name / relative for name in OUTPUT_NAMES
        }
        if args.resume and all(path.is_file() for path in outputs.values()):
            print(f"[{index}/{len(images)}] skip {relative.as_posix()}", flush=True)
            continue

        started = time.perf_counter()
        try:
            with Image.open(input_path) as source:
                generated = runner(source.convert("RGB"))
            for name, image in zip(OUTPUT_NAMES, generated):
                outputs[name].parent.mkdir(parents=True, exist_ok=True)
                image.save(outputs[name], format="PNG")
            row: dict[str, object] = {
                "name": relative.as_posix(),
                "status": "ok",
                "seconds": time.perf_counter() - started,
            }
            if torch.cuda.is_available():
                row["max_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            prior_rows[relative.as_posix()] = row
            print(
                f"[{index}/{len(images)}] ok {relative.as_posix()} "
                f"{row['seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            prior_rows[relative.as_posix()] = {
                "name": relative.as_posix(),
                "status": "error",
                "seconds": time.perf_counter() - started,
                "error": repr(exc),
            }
            report["images"] = list(prior_rows.values())
            report["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_json(report_path, report)
            raise

        report["images"] = list(prior_rows.values())
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(report_path, report)

    report["completed"] = len(prior_rows) >= len(images) and all(
        row.get("status") == "ok" for row in prior_rows.values()
    )
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
