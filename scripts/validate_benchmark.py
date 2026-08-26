"""Validate a frozen image benchmark and check it against training images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dct_matrix(size: int) -> np.ndarray:
    rows = np.arange(size, dtype=np.float32)[:, None]
    cols = np.arange(size, dtype=np.float32)[None, :]
    matrix = np.cos(math.pi * (2 * cols + 1) * rows / (2 * size))
    matrix[0] *= 1 / math.sqrt(2)
    return matrix * math.sqrt(2 / size)


DCT_32 = dct_matrix(32)


def hashes(image: Image.Image) -> tuple[int, int, int]:
    gray = image.convert("L")

    dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = pack_bits(dh[:, 1:] > dh[:, :-1])

    ah = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    ahash = pack_bits(ah > np.median(ah))

    ph = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    frequencies = DCT_32 @ ph @ DCT_32.T
    low = frequencies[:8, :8].copy()
    median = np.median(low.reshape(-1)[1:])
    phash = pack_bits(low > median)
    return dhash, ahash, phash


def pack_bits(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def inspect(path: Path, root: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        dhash, ahash, phash = hashes(image)
        return {
            "path": path.relative_to(root).as_posix(),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "sha256": sha256(path),
            "dhash": f"{dhash:016x}",
            "ahash": f"{ahash:016x}",
            "phash": f"{phash:016x}",
            "_hashes": (dhash, ahash, phash),
        }


def public_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "_hashes"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--expected-size", type=int, default=512)
    parser.add_argument("--near-threshold", type=int, default=8)
    args = parser.parse_args()

    inputs = image_paths(args.input)
    references = image_paths(args.reference) if args.reference else []
    errors: list[str] = []
    input_records: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = []

    for path in inputs:
        try:
            record = inspect(path, args.input)
            input_records.append(record)
            if (record["width"], record["height"]) != (args.expected_size, args.expected_size):
                errors.append(f"unexpected size: {record['path']}")
            if record["mode"] != "RGB":
                errors.append(f"unexpected mode: {record['path']} ({record['mode']})")
        except Exception as exc:  # keep auditing the remaining files
            errors.append(f"unreadable: {path} ({exc})")

    if len(inputs) != args.expected_count:
        errors.append(f"expected {args.expected_count} images, found {len(inputs)}")

    for path in references:
        try:
            reference_records.append(inspect(path, args.reference))
        except Exception as exc:
            errors.append(f"unreadable reference: {path} ({exc})")

    exact_internal: list[list[str]] = []
    sha_groups: dict[str, list[str]] = {}
    for record in input_records:
        sha_groups.setdefault(str(record["sha256"]), []).append(str(record["path"]))
    exact_internal.extend(paths for paths in sha_groups.values() if len(paths) > 1)

    near_internal: list[dict[str, object]] = []
    for index, left in enumerate(input_records):
        for right in input_records[index + 1 :]:
            distances = [hamming(a, b) for a, b in zip(left["_hashes"], right["_hashes"])]
            if max(distances) <= args.near_threshold:
                near_internal.append(
                    {"left": left["path"], "right": right["path"], "distances": distances}
                )

    reference_matches: list[dict[str, object]] = []
    reference_sha = {str(record["sha256"]): record for record in reference_records}
    for candidate in input_records:
        if str(candidate["sha256"]) in reference_sha:
            reference_matches.append(
                {
                    "input": candidate["path"],
                    "reference": reference_sha[str(candidate["sha256"])]["path"],
                    "kind": "exact_bytes",
                    "distances": [0, 0, 0],
                }
            )
            continue
        for reference in reference_records:
            distances = [
                hamming(a, b) for a, b in zip(candidate["_hashes"], reference["_hashes"])
            ]
            if max(distances) <= args.near_threshold:
                reference_matches.append(
                    {
                        "input": candidate["path"],
                        "reference": reference["path"],
                        "kind": "perceptual",
                        "distances": distances,
                    }
                )

    categories = Counter(Path(str(record["path"])).parts[0] for record in input_records)
    passed = not errors and not exact_internal and not near_internal and not reference_matches
    report = {
        "passed": passed,
        "input_root": str(args.input.resolve()),
        "reference_root": str(args.reference.resolve()) if args.reference else None,
        "input_count": len(input_records),
        "reference_count": len(reference_records),
        "categories": dict(sorted(categories.items())),
        "expected_size": args.expected_size,
        "near_threshold": args.near_threshold,
        "errors": errors,
        "exact_internal_duplicates": exact_internal,
        "near_internal_duplicates": near_internal,
        "reference_matches": reference_matches,
        "images": [public_record(record) for record in input_records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "images"}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
