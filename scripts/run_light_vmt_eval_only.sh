#!/usr/bin/env bash
set -euo pipefail
cd /root/SHIFT-main
PY=/root/.virtualenvs/SHIFT-main/bin/python
INPUT=${1:-/root/autodl-tmp/train_data/val_selector_small.json}
PREFIX=${2:-val_selector_small}
OUT=/root/autodl-tmp/light_vmt_outputs
$PY -m light_vmt.visual_gate --input "$INPUT" --output_dir "$OUT/gate_outputs" --prefix "$PREFIX"
$PY -m light_vmt.evaluate_gate --gate "$OUT/gate_outputs/${PREFIX}_threshold_0p02.json" --output_dir "$OUT/reports" --prefix "${PREFIX}_threshold_0p02"
