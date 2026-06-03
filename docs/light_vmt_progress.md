# Light-VMT Project Progress

Updated: 2026-06-03

## Overall Goal

Build a lightweight VMT workflow on top of SHIFT that routes each sample to either:

- `text-only` when visual context is unlikely to help.
- `image-text` with one selected frame when visual context is needed.

The first milestone is an MVP engineering loop without retraining or downloading new data.

## Stage Status

| Stage | Status | Notes |
|---|---|---|
| 0. Path/config setup | Done | `configs/light_vmt_paths.json`, `configs/gate_thresholds.json`, and `light_vmt/paths.py` are in place. |
| 1. Selector score export | Done | `codes/evaluate_selector.py` now keeps `chooseOutput.json` and writes `selector_scores.json`; `light_vmt/selector_score_export.py` supports offline score export. |
| 2. Visual necessity gate | Done | `light_vmt/visual_gate.py` computes `visual_gain`, `need_visual`, route, and threshold summaries. |
| 3. Dataset routing | Done | `light_vmt/route_dataset.py` splits datasets into text-only and visual-needed branches. |
| 4. Result merge | Done | `light_vmt/merge_results.py` merges text and visual branch results back to the original order. |
| 5. Gate/report evaluation | In progress | `light_vmt/evaluate_gate.py` reports route, oracle-style gain, and estimated visual saving rate. Full BLEU/COMET comparison requires inference outputs. |
| 6. Lightweight reranker | Not started | MVP uses selector/gain only; no extra model or visual feature reranker yet. |
| 7. Multi-frame budget | Not started | Requires GPU validation and changes to `image_selection` behavior. |

## Current Verified Loop

Command:

```bash
./scripts/run_light_vmt_mvp.sh /root/autodl-tmp/train_data/val_selector_small.json val_selector_small_mvp 0.02
```

Verified outputs:

- `/root/autodl-tmp/light_vmt_outputs/selector_scores/val_selector_small_mvp_selector_scores.json`
- `/root/autodl-tmp/light_vmt_outputs/gate_outputs/val_selector_small_mvp_threshold_0p02.json`
- `/root/autodl-tmp/light_vmt_outputs/reports/val_selector_small_mvp_threshold_0p02.json`
- `/root/autodl-tmp/light_vmt_outputs/routed_datasets/val_selector_small_mvp_threshold_0p02_summary.json`

Latest small validation result:

- Samples: 50
- `visual_call_rate`: 0.96
- `estimated_visual_saving_rate`: 0.808
- Routing coverage: matched 50, missing 0

## Next Work

1. Run selector on a GPU instance to produce reference-free `selector_scores.json`.
2. Run text-only and image-text inference branches on routed datasets.
3. Merge branch outputs and compute translation quality metrics.
4. Sweep thresholds on validation data and choose a threshold with a quality/cost tradeoff.
5. Consider reranker and multi-frame budget only after the MVP comparison is stable.
