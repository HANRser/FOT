"""Build a compact, auditable Original/FoT output comparison gallery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


KINDS = ("original", "protected", "forged", "confidence", "recovered")
CATEGORY_ORDER = ("animal", "camera", "face", "human_environment", "multi_human")
CATEGORY_LABELS = {
    "animal": "Animal",
    "camera": "Camera",
    "face": "Face",
    "human_environment": "Human-Environment",
    "multi_human": "Multi-Human",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def image_names(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*.png")}


def make_thumbnail(source: Path, target: Path, size: int, quality: int) -> dict[str, object]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        contained = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (24, 24, 24))
        offset = ((size - contained.width) // 2, (size - contained.height) // 2)
        canvas.paste(contained, offset)
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="WEBP", quality=quality, method=6)
        with Image.open(target) as verification:
            verification.verify()
        return {
            "source_width": image.width,
            "source_height": image.height,
            "source_mean": float(array.mean()),
            "source_sha256": sha256(source),
            "thumbnail_sha256": sha256(target),
            "thumbnail_bytes": target.stat().st_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=86)
    parser.add_argument("--expected-count", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = {"original": args.original_dir}
    roots.update({kind: args.formal_dir / kind for kind in KINDS[1:]})
    expected = image_names(args.original_dir)
    if len(expected) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} originals, found {len(expected)}")
    for kind, root in roots.items():
        actual = image_names(root)
        if actual != expected:
            raise SystemExit(
                f"{kind} mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, object]] = []
    log_lines = [
        f"generated_at={generated_at}",
        f"git_revision={git_revision()}",
        f"checkpoint={args.checkpoint.resolve()}",
        f"checkpoint_sha256={sha256(args.checkpoint)}",
        f"dataset_manifest={args.dataset_manifest.resolve()}",
        f"dataset_manifest_sha256={sha256(args.dataset_manifest)}",
        f"thumbnail_size={args.size}",
        f"thumbnail_quality={args.quality}",
        f"expected_count={args.expected_count}",
    ]

    for relative_name in sorted(expected):
        relative = Path(relative_name)
        category = relative.parts[0]
        record: dict[str, object] = {
            "id": relative.stem,
            "category": category,
            "relative_path": relative_name,
            "artifacts": {},
        }
        for kind in KINDS:
            source = roots[kind] / relative
            thumbnail_relative = Path("thumbnails") / kind / relative.with_suffix(".webp")
            thumbnail = args.output_dir / thumbnail_relative
            metadata = make_thumbnail(source, thumbnail, args.size, args.quality)
            record["artifacts"][kind] = {  # type: ignore[index]
                "source": str(source.resolve()),
                "thumbnail": thumbnail_relative.as_posix(),
                **metadata,
            }
        records.append(record)
        log_lines.append(
            f"OK\t{category}\t{relative.stem}\t"
            + "\t".join(
                f"{kind}={record['artifacts'][kind]['source_sha256']}"  # type: ignore[index]
                for kind in KINDS
            )
        )

    manifest = {
        "generated_at": generated_at,
        "git_revision": git_revision(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "count": len(records),
        "columns": list(KINDS),
        "thumbnail": {"format": "WEBP", "size": args.size, "quality": args.quality},
        "records": records,
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_fields = ["id", "category", "relative_path"] + [
        field
        for kind in KINDS
        for field in (f"{kind}_thumbnail", f"{kind}_source_sha256")
    ]
    with (args.output_dir / "comparison_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ("id", "category", "relative_path")}
            for kind in KINDS:
                artifact = record["artifacts"][kind]  # type: ignore[index]
                row[f"{kind}_thumbnail"] = artifact["thumbnail"]
                row[f"{kind}_source_sha256"] = artifact["source_sha256"]
            writer.writerow(row)

    log_lines.append(f"completed={len(records)}")
    log_lines.append("status=PASS")
    (args.output_dir / "comparison_generation.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    markdown = [
        "# FoT real50 图片对比表",
        "",
        "每行对应一张冻结测试图片。列顺序为原图、保护图、SVD 最后一帧、原始置信度图和恢复图。",
        "缩略图只用于 GitHub 浏览；正式指标使用 LDS 上的全分辨率文件计算。",
        "Confidence 保留原始显示范围，没有为了变亮而重新归一化，因此低置信度图会接近黑色。",
        "",
        f"- 样本数：{len(records)}；",
        f"- 生成时间（UTC）：`{generated_at}`；",
        f"- checkpoint SHA-256：`{sha256(args.checkpoint)}`；",
        f"- 数据清单 SHA-256：`{sha256(args.dataset_manifest)}`；",
        "- 完整性状态：PASS。",
        "",
    ]
    for category in CATEGORY_ORDER:
        markdown.extend(
            [
                f"## {CATEGORY_LABELS[category]}",
                "",
                "| ID | Original | Protected | Forged | Confidence | Recovered |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for record in records:
            if record["category"] != category:
                continue
            cells = []
            for kind in KINDS:
                path = record["artifacts"][kind]["thumbnail"]  # type: ignore[index]
                cells.append(f'<img src="{path}" width="160" alt="{record["id"]} {kind}">')
            markdown.append(f"| `{record['id']}` | " + " | ".join(cells) + " |")
        markdown.append("")
    markdown.extend(
        [
            "## 审计文件",
            "",
            "- [`comparison_manifest.csv`](comparison_manifest.csv)：便于表格软件查看；",
            "- [`comparison_manifest.json`](comparison_manifest.json)：尺寸、亮度、来源及缩略图哈希；",
            "- [`comparison_generation.log`](comparison_generation.log)：逐张生成日志和源文件 SHA-256。",
            "",
        ]
    )
    (args.output_dir / "README_zh.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
