#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

gpu_index="$1"
project_root="/data/lvzhengshu/FOT"
input_dir="$project_root/data/benchmarks/fot_real50/test_512"
manifest="$project_root/data/benchmarks/fot_real50/manifest.json"
checkpoint="$project_root/checkpoints/fot-mini-512/best.pt"
output_dir="$project_root/outputs/fot-real50/formal"

cd "$project_root"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$project_root/models/huggingface"
export TORCH_HOME="$project_root/models/torch"

common_args=(
  --input-dir "$input_dir"
  --output-dir "$output_dir"
  --checkpoint "$checkpoint"
  --dataset-manifest "$manifest"
  --size 512
  --template-channels 32
  --motion-channels 32
  --motion-chunk-size 2
  --local-files-only
)

# A real one-image run catches model-loading and SVD failures before the batch.
.venv/bin/python batch_test.py "${common_args[@]}" --limit 1
.venv/bin/python batch_test.py "${common_args[@]}" --resume

# Metrics are evaluated on CPU so the GPU can be released as soon as generation ends.
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python evaluate.py \
  --reference-dir "$input_dir" \
  --recovered-dir "$output_dir/protected" \
  --output "$output_dir/metrics_protected.json" \
  --lpips --clip --local-files-only --device cpu
.venv/bin/python evaluate.py \
  --reference-dir "$input_dir" \
  --recovered-dir "$output_dir/recovered" \
  --output "$output_dir/metrics_recovered.json" \
  --lpips --clip --local-files-only --device cpu

date --iso-8601=seconds > "$output_dir/COMPLETE"
