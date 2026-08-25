#!/usr/bin/env bash
set -euo pipefail

root="${1:-/data/lvzhengshu/FOT}"
raw="$root/data/raw"
processed="$root/data/processed/fot-mini"
mkdir -p "$raw/coco2017" "$raw/sintel" "$processed"

wget -c -O "$raw/coco2017/val2017.zip" \
  "http://images.cocodataset.org/zips/val2017.zip"
wget -c -O "$raw/sintel/MPI-Sintel-training_extras.zip" \
  "https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip"

"$root/.venv/bin/python" "$root/prepare_data.py" \
  --coco-zip "$raw/coco2017/val2017.zip" \
  --sintel-zip "$raw/sintel/MPI-Sintel-training_extras.zip" \
  --output "$processed"
