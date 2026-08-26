#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_INDEX [GPU_INDEX ...]" >&2
  exit 2
fi

project_root="/data/lvzhengshu/FOT"
status_file="$project_root/outputs/fot-real50/launcher_status.txt"
mkdir -p "$(dirname "$status_file")"

echo "waiting $(date --iso-8601=seconds) candidates=$*" | tee "$status_file"
while true; do
  for gpu_index in "$@"; do
    read -r memory_used utilization < <(
      nvidia-smi --id="$gpu_index" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ','
    )
    if (( memory_used <= 2048 && utilization <= 5 )); then
      sleep 10
      read -r memory_used utilization < <(
        nvidia-smi --id="$gpu_index" \
          --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits | tr -d ','
      )
      if (( memory_used <= 2048 && utilization <= 5 )); then
        echo "running $(date --iso-8601=seconds) gpu=$gpu_index" | tee "$status_file"
        exec "$project_root/scripts/run_fot_real50.sh" "$gpu_index"
      fi
    fi
  done
  sleep 30
done
