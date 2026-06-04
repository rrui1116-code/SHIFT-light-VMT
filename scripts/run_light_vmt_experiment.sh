#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: bash scripts/run_light_vmt_experiment.sh MODE [EXTRA_ARGS...]}"
shift || true

DATASET="${LIGHT_VMT_DATASET:-/root/autodl-tmp/train_data/val_selector_small.json}"
CLUSTER_PATH="${LIGHT_VMT_CLUSTER_PATH:-/root/autodl-tmp/train_data/clustered_val.json}"
SELECTOR_SCORES="${LIGHT_VMT_SELECTOR_SCORES:-/root/autodl-tmp/light_vmt_outputs/selector_scores/val_selector_small_full_selector/selector_scores.json}"
OUTPUT_ROOT="${LIGHT_VMT_OUTPUT_ROOT:-/root/autodl-tmp/light_vmt_outputs/experiments}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m light_vmt.experiment_runner \
  --mode "$MODE" \
  --dataset "$DATASET" \
  --cluster_path "$CLUSTER_PATH" \
  --selector_scores "$SELECTOR_SCORES" \
  --output_root "$OUTPUT_ROOT" \
  "$@"
