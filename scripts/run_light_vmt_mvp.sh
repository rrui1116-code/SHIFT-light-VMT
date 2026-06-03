#!/usr/bin/env bash
set -euo pipefail

cd /root/SHIFT-main

PY=${PY:-/root/.virtualenvs/SHIFT-main/bin/python}
DATASET=${1:-/root/autodl-tmp/train_data/val_selector_small.json}
PREFIX=${2:-val_selector_small}
THRESHOLD=${3:-0.02}
OUT=${OUT:-/root/autodl-tmp/light_vmt_outputs}

threshold_suffix="$($PY -c 'import sys; print(str(float(sys.argv[1])).replace("-", "neg").replace(".", "p"))' "$THRESHOLD")"
gate_path="$OUT/gate_outputs/${PREFIX}_threshold_${threshold_suffix}.json"

"$PY" -m light_vmt.selector_score_export \
  --input "$DATASET" \
  --output "$OUT/selector_scores/${PREFIX}_selector_scores.json"

"$PY" -m light_vmt.visual_gate \
  --input "$OUT/selector_scores/${PREFIX}_selector_scores.json" \
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

if [[ -n "${TEXT_RESULTS:-}" && -n "${VISUAL_RESULTS:-}" ]]; then
  merged_path="$OUT/merged_results/${PREFIX}_threshold_${threshold_suffix}_results.json"
  "$PY" -m light_vmt.merge_results \
    --dataset "$DATASET" \
    --gate "$gate_path" \
    --text_results "$TEXT_RESULTS" \
    --visual_results "$VISUAL_RESULTS" \
    --output "$merged_path"

  "$PY" -m light_vmt.evaluate_results \
    --results "$merged_path" \
    --metrics ${METRICS:-BLEU chrF} \
    --output_dir "$OUT/reports" \
    --prefix "${PREFIX}_threshold_${threshold_suffix}"
fi
