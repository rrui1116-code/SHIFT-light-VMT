import argparse
import json
from pathlib import Path
from statistics import mean

from light_vmt.visual_gate import load_json, save_json, summarize
from light_vmt.paths import ensure_output_dirs, load_paths


def oracle_summary(records):
    # If scores are COMET-like scores, this estimates how often visual candidates beat text-only.
    visual_better = [r for r in records if r["visual_gain"] > 0]
    gate_visual = [r for r in records if r["need_visual"]]
    gate_miss = [r for r in records if (r["visual_gain"] > 0 and not r["need_visual"])]
    negative_visual = [r for r in records if r["visual_gain"] <= 0]
    return {
        "oracle_visual_better_count": len(visual_better),
        "oracle_visual_better_rate": len(visual_better) / len(records) if records else 0.0,
        "gate_visual_count": len(gate_visual),
        "gate_miss_count": len(gate_miss),
        "gate_miss_rate_among_oracle_visual": len(gate_miss) / len(visual_better) if visual_better else 0.0,
        "visual_noise_or_no_gain_count": len(negative_visual),
        "avg_gain_gate_visual": mean([r["visual_gain"] for r in gate_visual]) if gate_visual else 0.0,
        "avg_gain_gate_text_only": mean([r["visual_gain"] for r in records if not r["need_visual"]]) if records else 0.0,
    }


def efficiency_summary(records, frames_per_visual_sample=5):
    total = len(records)
    visual_count = sum(1 for r in records if r["need_visual"])
    avg_frames_per_sample = visual_count / total if total else 0.0
    baseline_frames_per_sample = float(frames_per_visual_sample)
    estimated_visual_saving_rate = (
        1.0 - (avg_frames_per_sample / baseline_frames_per_sample)
        if baseline_frames_per_sample and total
        else 0.0
    )
    return {
        "baseline_frames_per_sample": baseline_frames_per_sample,
        "avg_frames_per_sample": avg_frames_per_sample,
        "estimated_visual_saving_rate": estimated_visual_saving_rate,
    }


def write_report(summary, oracle, efficiency, output):
    lines = ["# Light-VMT Gate Evaluation", ""]
    lines.append("## Route Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Efficiency Summary")
    for key, value in efficiency.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Oracle-Style Visual Gain Summary")
    for key, value in oracle.items():
        lines.append(f"- {key}: {value}")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate gate routing statistics.")
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix", default="gate_eval")
    parser.add_argument("--frames_per_visual_sample", type=int, default=5)
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    output_dir = Path(args.output_dir) if args.output_dir else paths["output_root"] / "reports"
    records = load_json(args.gate)
    summary = summarize(records)
    oracle = oracle_summary(records)
    efficiency = efficiency_summary(records, args.frames_per_visual_sample)
    report = {"route_summary": summary, "efficiency_summary": efficiency, "oracle_summary": oracle}
    save_json(report, output_dir / f"{args.prefix}.json")
    write_report(summary, oracle, efficiency, output_dir / f"{args.prefix}.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
