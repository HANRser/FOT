"""Prepare the compact FoT reproduction dataset from official archives."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-zip", required=True)
    parser.add_argument("--sintel-zip", required=True)
    parser.add_argument("--output", default="data/processed/fot-mini")
    parser.add_argument("--train-images", type=int, default=4500)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--val-flow-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-zip-test", action="store_true")
    return parser.parse_args()


def validate_member(member: str) -> PurePosixPath:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"压缩包包含不安全路径：{member}")
    return path


def extract_selected(
    archive_path: Path,
    destination: Path,
    *,
    suffix: str,
    strip_prefix: tuple[str, ...] = (),
    test_archive: bool = True,
) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        if test_archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP 校验失败：{archive_path}，损坏成员：{bad}")
        for info in archive.infolist():
            path = validate_member(info.filename)
            if info.is_dir() or path.suffix.lower() != suffix:
                continue
            parts = path.parts
            if strip_prefix and parts[: len(strip_prefix)] != strip_prefix:
                continue
            relative = Path(*parts[len(strip_prefix) :])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != info.file_size:
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)
    return sorted(extracted)


def write_manifest(path: Path, root: Path, files: list[Path]) -> None:
    lines = [file.relative_to(root).as_posix() for file in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.train_images <= 0 or args.val_images <= 0:
        raise SystemExit("训练与验证图像数量必须大于 0")
    if not 0 < args.val_flow_ratio < 1:
        raise SystemExit("--val-flow-ratio 必须位于 (0,1)")

    coco_zip = Path(args.coco_zip).resolve()
    sintel_zip = Path(args.sintel_zip).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    images = extract_selected(
        coco_zip,
        output / "images",
        suffix=".jpg",
        strip_prefix=("val2017",),
        test_archive=not args.skip_zip_test,
    )
    flows = extract_selected(
        sintel_zip,
        output / "flows" / "sintel",
        suffix=".flo",
        strip_prefix=("training", "flow"),
        test_archive=not args.skip_zip_test,
    )
    requested = args.train_images + args.val_images
    if len(images) < requested:
        raise ValueError(f"COCO 图像不足：需要 {requested}，实际 {len(images)}")
    if len(flows) < 2:
        raise ValueError("Sintel 光流数量不足，无法划分训练/验证集")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    rng.shuffle(flows)
    train_images = sorted(images[: args.train_images])
    val_images = sorted(images[args.train_images : requested])
    val_flow_count = max(1, round(len(flows) * args.val_flow_ratio))
    val_flows = sorted(flows[:val_flow_count])
    train_flows = sorted(flows[val_flow_count:])

    manifests = {
        "train_images": train_images,
        "val_images": val_images,
        "train_flows": train_flows,
        "val_flows": val_flows,
    }
    for name, files in manifests.items():
        write_manifest(output / f"{name}.txt", output, files)

    metadata = {
        "name": "fot-mini",
        "purpose": "basic Flow of Truth reproduction, not paper-scale training",
        "seed": args.seed,
        "target_resolution": [512, 512],
        "sources": {
            "images": "COCO 2017 validation images",
            "flows": "MPI Sintel training ground-truth optical flow",
        },
        "counts": {name: len(files) for name, files in manifests.items()},
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
