"""Audit FoT real50 outputs for missing, extra, corrupt, or skipped images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


KINDS = ("protected", "forged", "recovered", "confidence")


def names(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*.png")}


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--chunk-count", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    expected = names(args.input_dir)
    errors: list[str] = []
    output_checks: dict[str, object] = {}
    if len(expected) != args.expected_count:
        errors.append(f"input count is {len(expected)}, expected {args.expected_count}")

    for kind in KINDS:
        root = args.formal_dir / kind
        actual = names(root)
        unreadable: list[str] = []
        for relative in sorted(actual):
            try:
                with Image.open(root / relative) as image:
                    image.verify()
            except Exception as exc:
                unreadable.append(f"{relative}: {exc}")
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        output_checks[kind] = {
            "count": len(actual),
            "missing": missing,
            "extra": extra,
            "unreadable": unreadable,
        }
        if missing or extra or unreadable:
            errors.append(f"{kind} output set is invalid")

    run = load(args.formal_dir / "run.json")
    run_rows = list(run.get("images", []))
    run_names = [str(row.get("name")) for row in run_rows]
    run_errors = [row for row in run_rows if row.get("status") != "ok"]
    if set(run_names) != expected or len(run_names) != len(expected) or run_errors:
        errors.append("run.json does not contain exactly one successful row per input")

    chunk_checks: list[dict[str, object]] = []
    chunk_names: list[str] = []
    for number in range(1, args.chunk_count + 1):
        name = f"chunk_{number:03d}"
        root = args.chunks_dir / name
        protected_path = root / "metrics_protected.json"
        recovered_path = root / "metrics_recovered.json"
        check: dict[str, object] = {
            "name": name,
            "complete_marker": (root / "COMPLETE").is_file(),
        }
        if not protected_path.is_file() or not recovered_path.is_file():
            check["error"] = "missing metric report"
            errors.append(f"{name} is missing a metric report")
            chunk_checks.append(check)
            continue
        protected = load(protected_path)
        recovered = load(recovered_path)
        protected_names = [str(row["name"]) for row in protected.get("per_image", [])]
        recovered_names = [str(row["name"]) for row in recovered.get("per_image", [])]
        check["protected_count"] = protected.get("count")
        check["recovered_count"] = recovered.get("count")
        check["images"] = protected_names
        if (
            protected.get("count") != args.chunk_size
            or recovered.get("count") != args.chunk_size
            or protected_names != recovered_names
            or not check["complete_marker"]
        ):
            errors.append(f"{name} is incomplete or inconsistent")
        chunk_names.extend(protected_names)
        chunk_checks.append(check)

    frequencies = Counter(chunk_names)
    chunk_missing = sorted(expected - set(chunk_names))
    chunk_extra = sorted(set(chunk_names) - expected)
    chunk_duplicates = sorted(name for name, count in frequencies.items() if count != 1)
    if chunk_missing or chunk_extra or chunk_duplicates:
        errors.append("chunk membership does not partition the input set")

    full_metrics: dict[str, object] = {}
    for kind in ("protected", "recovered"):
        path = args.formal_dir / f"metrics_{kind}.json"
        report = load(path)
        full_metrics[kind] = {
            "count": report.get("count"),
            "summary": report.get("summary"),
        }
        if report.get("count") != args.expected_count:
            errors.append(f"full {kind} metrics count is not {args.expected_count}")

    result = {
        "passed": not errors,
        "expected_count": len(expected),
        "formal_complete_marker": (args.formal_dir / "COMPLETE").is_file(),
        "errors": errors,
        "outputs": output_checks,
        "run": {
            "count": len(run_rows),
            "ok": len(run_rows) - len(run_errors),
            "errors": run_errors,
        },
        "chunks": chunk_checks,
        "chunk_partition": {
            "count": len(chunk_names),
            "missing": chunk_missing,
            "extra": chunk_extra,
            "duplicates": chunk_duplicates,
        },
        "full_metrics": full_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
