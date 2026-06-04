import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def is_shift_video(record):
    return record.get("selection_type") == "video"


def is_gate_video(record):
    return bool(record.get("need_visual"))


def main():
    parser = argparse.ArgumentParser(description="Write the 5-way Light-VMT experiment report.")
    parser.add_argument("--experiment_root", default="/root/autodl-tmp/light_vmt_outputs/experiments")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    output = Path(args.output) if args.output else root / "analysis" / "light_vmt_5way_experiment_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_paths = {
        "Text-only": ("exp_text_only/metrics/text_only_translation_metrics.json", "Only source sentence"),
        "Random frame": ("exp_random_seed13_top1/metrics/random_seed13_top1_translation_metrics.json", "seed=13, top_k=1"),
        "Uniform frame": ("exp_uniform_top1/metrics/uniform_top1_translation_metrics.json", "top_k=1"),
        "Original SHIFT": ("exp_shift/metrics/shift_translation_metrics.json", "Original selector"),
        "SHIFT + Gate": ("exp_shift_gate_t0/metrics/shift_gate_translation_metrics.json", "threshold=0.0"),
    }
    metric_rows = []
    for method, (relative_path, note) in metric_paths.items():
        scores = load_json(root / relative_path)
        metrics = scores["metrics"]
        metric_rows.append((method, metrics.get("BLEU"), metrics.get("chrF"), note))

    shift_config = load_json(root / "exp_shift" / "config" / "config.json")
    gate_summary = load_json(root / "exp_shift_gate_t0" / "gate_analysis" / "gate_summary.json")
    diff_summary = load_json(root / "analysis" / "shift_vs_gate_selection_summary.json")
    shift_records = load_json(root / "exp_shift" / "selected_frames" / "selected_frames.json")
    gate_records = load_json(root / "exp_shift_gate_t0" / "gate_analysis" / "gate_records.json")

    shift_by_clip = {record["clipID"]: record for record in shift_records}
    gate_by_clip = {record["clipID"]: record for record in gate_records}
    contingency = {
        "SHIFT=text-only, Gate=text-only": 0,
        "SHIFT=text-only, Gate=video": 0,
        "SHIFT=video, Gate=text-only": 0,
        "SHIFT=video, Gate=video": 0,
    }
    for clip_id in sorted(set(shift_by_clip) & set(gate_by_clip)):
        shift_video = is_shift_video(shift_by_clip[clip_id])
        gate_video = is_gate_video(gate_by_clip[clip_id])
        if not shift_video and not gate_video:
            contingency["SHIFT=text-only, Gate=text-only"] += 1
        elif not shift_video and gate_video:
            contingency["SHIFT=text-only, Gate=video"] += 1
        elif shift_video and not gate_video:
            contingency["SHIFT=video, Gate=text-only"] += 1
        else:
            contingency["SHIFT=video, Gate=video"] += 1

    lines = [
        "# Light-VMT 5-way Experiment Report",
        "",
        "Dataset: `val_selector_small_infer.json`, 50 samples. Results are from actual runs on 2026-06-04.",
        "",
        "## Main Metrics",
        "",
        "| Method | BLEU | chrF | COMET | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for method, bleu, chrf, note in metric_rows:
        lines.append(f"| {method} | {bleu:.4f} | {chrf:.4f} | N/A | {note} |")

    lines.extend([
        "",
        "COMET was not computed because the current evaluation environment cannot import `utils.computeTransMetric` successfully (`ModuleNotFoundError: jieba`). No COMET score is fabricated.",
        "",
        "## Selection Behavior",
        "",
        "| Method | text-only count | video count | visual_call_rate |",
        "|---|---:|---:|---:|",
        f"| Original SHIFT | {shift_config['text_only_count']} | {shift_config['video_count']} | {shift_config['video_count'] / shift_config['total']:.4f} |",
        f"| SHIFT + Gate | {gate_summary['text_only_count']} | {gate_summary['visual_count']} | {gate_summary['visual_call_rate']:.4f} |",
        "",
        "## SHIFT vs Gate Difference",
        "",
        "| Comparison | Count |",
        "|---|---:|",
        f"| Consistent | {diff_summary['consistent_count']} |",
        f"| Inconsistent | {diff_summary['inconsistent_count']} |",
    ])
    for key, value in contingency.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Key Files",
        "",
    ])
    for experiment in [
        "exp_text_only",
        "exp_random_seed13_top1",
        "exp_uniform_top1",
        "exp_shift",
        "exp_shift_gate_t0",
    ]:
        lines.append(f"- `{root / experiment}`")
    lines.extend([
        f"- Case file: `{root / 'analysis' / 'shift_vs_gate_cases.json'}`",
        f"- Case candidates: `{root / 'analysis' / 'case_study_candidates.json'}`",
        "",
        "## Preliminary Interpretation",
        "",
        "On this 50-sample small validation subset, Uniform frame has the highest BLEU and chrF. SHIFT + Gate slightly improves over Original SHIFT in BLEU and chrF, but the margin is very small. The current result is enough to show the pipeline and routing analysis work, but not enough to claim the gate is a strong paper-level main contribution.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
