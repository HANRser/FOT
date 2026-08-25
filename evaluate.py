"""Evaluate one image pair or two matching directory trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from fot.evaluation import ImageEvaluator, aggregate, pair_directories, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", nargs="?")
    parser.add_argument("recovered", nargs="?")
    parser.add_argument("--reference-dir")
    parser.add_argument("--recovered-dir")
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory_mode = args.reference_dir is not None or args.recovered_dir is not None
    if directory_mode:
        if not args.reference_dir or not args.recovered_dir:
            raise SystemExit("目录模式必须同时传 --reference-dir 和 --recovered-dir")
    elif not args.reference or not args.recovered:
        raise SystemExit("请提供 reference/recovered，或使用两个目录参数")

    evaluator = ImageEvaluator(
        device=args.device,
        use_lpips=args.lpips,
        use_clip=args.clip,
        clip_model_id=args.clip_model,
        local_files_only=args.local_files_only,
    )
    if not directory_mode:
        result = evaluator(Image.open(args.reference), Image.open(args.recovered))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    rows = []
    for name, reference_path, recovered_path in pair_directories(
        args.reference_dir, args.recovered_dir
    ):
        metrics = evaluator(Image.open(reference_path), Image.open(recovered_path))
        rows.append({"name": name, **metrics})
    summary = aggregate(rows)
    json_path, csv_path = save_report(args.output, rows, summary)
    print(
        json.dumps(
            {
                "count": len(rows),
                "summary": summary,
                "json": str(json_path),
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
