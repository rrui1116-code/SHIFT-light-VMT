#!/usr/bin/env bash
set -euo pipefail

cd /root/SHIFT-main

PY=${PY:-/root/.virtualenvs/SHIFT-main/bin/python}
DATASET=${1:-/root/autodl-tmp/train_data/val_selector_small.json}
PREFIX=${2:-val_selector_small_real_selector}
THRESHOLD=${3:-0.02}
MODEL_PATH=${MODEL_PATH:-/autodl-fs/data/SHIFT-main-results/train_output_small/best_model}
PROCESSOR_PATH=${PROCESSOR_PATH:-$MODEL_PATH}
FRAMES_ROOT=${FRAMES_ROOT:-/root/autodl-tmp/frames/50frames}
BATCH_SIZE=${BATCH_SIZE:-1}
OUT=${OUT:-/root/autodl-tmp/light_vmt_outputs}

selector_out="$OUT/selector_scores/${PREFIX}"
threshold_suffix="$($PY -c 'import sys; print(str(float(sys.argv[1])).replace("-", "neg").replace(".", "p"))' "$THRESHOLD")"
gate_path="$OUT/gate_outputs/${PREFIX}_threshold_${threshold_suffix}.json"

"$PY" codes/evaluate_selector.py \
  --test_data_path "$DATASET" \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --frames_root "$FRAMES_ROOT" \
  --output_dir "$selector_out" \
  --batch_size "$BATCH_SIZE"

"$PY" -m light_vmt.visual_gate \
  --input "$selector_out/selector_scores.json" \
  --output_dir "$OUT/gate_outputs" \
  --thresholds "$THRESHOLD" \
  --prefix "$PREFIX"

"$PY" -m light_vmt.evaluate_gate \
  --gate "$gate_path" \
  --output_dir "$OUT/reports" \
  --prefix "${PREFIX}_threshold_${threshold_suffix}"

"$PY" -m light_vmt.route_dataset \
  --dataset "$DATASET" \
  --gate "$gate_path" \
  --output_dir "$OUT/routed_datasets" \
  --prefix "${PREFIX}_threshold_${threshold_suffix}"
