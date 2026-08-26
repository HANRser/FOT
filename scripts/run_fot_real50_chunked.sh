#!/usr/bin/env bash
set -euo pipefail

project_root="/data/lvzhengshu/FOT"
input_dir="$project_root/data/benchmarks/fot_real50/test_512"
manifest="$project_root/data/benchmarks/fot_real50/manifest.json"
checkpoint="$project_root/checkpoints/fot-mini-512/best.pt"
output_dir="$project_root/outputs/fot-real50/formal"
chunks_dir="$project_root/outputs/fot-real50/chunks"
status_file="$project_root/outputs/fot-real50/chunked_status.txt"
log_file="$project_root/outputs/fot-real50/chunked_progress.tsv"
batch_size=5
total=50

cd "$project_root"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$project_root/models/huggingface"
export TORCH_HOME="$project_root/models/torch"
mkdir -p "$output_dir" "$chunks_dir"
touch "$log_file"

completed=$(find "$output_dir/recovered" -type f -name "*.png" 2>/dev/null | wc -l)
if (( completed % batch_size != 0 || completed > total )); then
  echo "unexpected completed image count: $completed" >&2
  exit 1
fi

choose_gpu() {
  while true; do
    best_gpu=""
    best_util=101
    for gpu_index in 0 5; do
      read -r memory_free utilization < <(
        nvidia-smi --id="$gpu_index" \
          --query-gpu=memory.free,utilization.gpu \
          --format=csv,noheader,nounits | tr -d ','
      )
      if (( memory_free >= 45000 && utilization <= 20 && utilization < best_util )); then
        best_gpu="$gpu_index"
        best_util="$utilization"
      fi
    done
    if [[ -n "$best_gpu" ]]; then
      printf '%s\n' "$best_gpu"
      return
    fi
    echo "waiting_for_gpu $(date --iso-8601=seconds)" > "$status_file"
    sleep 30
  done
}

for (( limit=completed + batch_size; limit<=total; limit+=batch_size )); do
  chunk_number=$(( limit / batch_size ))
  chunk_name=$(printf 'chunk_%03d' "$chunk_number")
  chunk_dir="$chunks_dir/$chunk_name"
  gpu_index=$(choose_gpu)
  echo "running $chunk_name images=$((limit - batch_size + 1))-$limit gpu=$gpu_index $(date --iso-8601=seconds)" \
    | tee "$status_file"

  CUDA_VISIBLE_DEVICES="$gpu_index" .venv/bin/python batch_test.py \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --checkpoint "$checkpoint" \
    --dataset-manifest "$manifest" \
    --size 512 \
    --template-channels 32 \
    --motion-channels 32 \
    --motion-chunk-size 2 \
    --limit "$limit" \
    --resume \
    --local-files-only

  mkdir -p "$chunk_dir/reference" "$chunk_dir/protected" "$chunk_dir/recovered"
  start=$(( limit - batch_size + 1 ))
  find "$input_dir" -type f -name "*.png" | sort | sed -n "${start},${limit}p" |
    while IFS= read -r input_path; do
      relative=${input_path#"$input_dir"/}
      mkdir -p \
        "$chunk_dir/reference/$(dirname "$relative")" \
        "$chunk_dir/protected/$(dirname "$relative")" \
        "$chunk_dir/recovered/$(dirname "$relative")"
      [[ -e "$chunk_dir/reference/$relative" ]] || ln -s "$input_path" "$chunk_dir/reference/$relative"
      [[ -e "$chunk_dir/protected/$relative" ]] || ln -s "$output_dir/protected/$relative" "$chunk_dir/protected/$relative"
      [[ -e "$chunk_dir/recovered/$relative" ]] || ln -s "$output_dir/recovered/$relative" "$chunk_dir/recovered/$relative"
    done

  CUDA_VISIBLE_DEVICES="" .venv/bin/python evaluate.py \
    --reference-dir "$chunk_dir/reference" \
    --recovered-dir "$chunk_dir/protected" \
    --output "$chunk_dir/metrics_protected.json" \
    --lpips --clip --local-files-only --device cpu
  CUDA_VISIBLE_DEVICES="" .venv/bin/python evaluate.py \
    --reference-dir "$chunk_dir/reference" \
    --recovered-dir "$chunk_dir/recovered" \
    --output "$chunk_dir/metrics_recovered.json" \
    --lpips --clip --local-files-only --device cpu

  cp "$output_dir/run.json" "$chunk_dir/run.json"
  date --iso-8601=seconds > "$chunk_dir/COMPLETE"
  printf '%s\t%s\t%s\t%s\n' \
    "$chunk_name" "$((limit - batch_size + 1))-$limit" "$gpu_index" "$(date --iso-8601=seconds)" \
    >> "$log_file"
  echo "completed $chunk_name images=$((limit - batch_size + 1))-$limit $(date --iso-8601=seconds)" \
    | tee "$status_file"
done

CUDA_VISIBLE_DEVICES="" .venv/bin/python evaluate.py \
  --reference-dir "$input_dir" \
  --recovered-dir "$output_dir/protected" \
  --output "$output_dir/metrics_protected.json" \
  --lpips --clip --local-files-only --device cpu
CUDA_VISIBLE_DEVICES="" .venv/bin/python evaluate.py \
  --reference-dir "$input_dir" \
  --recovered-dir "$output_dir/recovered" \
  --output "$output_dir/metrics_recovered.json" \
  --lpips --clip --local-files-only --device cpu

date --iso-8601=seconds > "$output_dir/COMPLETE"
echo "all_complete $(date --iso-8601=seconds)" | tee "$status_file"
