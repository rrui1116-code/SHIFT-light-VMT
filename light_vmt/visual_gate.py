import argparse
import json
from pathlib import Path
from statistics import mean

from light_vmt.paths import ensure_output_dirs, load_paths


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_clip_id(item):
    if "clipID" in item:
        return item["clipID"]
    if "videoClipID" in item:
        return item["videoClipID"]
    if "videoClipId" in item:
        return item["videoClipId"]
    if "video_id" in item and "clip_id" in item:
        return f"{item['video_id']}_{item['clip_id']}"
    raise KeyError("Cannot infer clip id from item")


def get_scores(item, score_field="auto"):
    if score_field != "auto":
        return item[score_field]
    for key in ["selector_scores", "scores", "cometScores", "COMET_scores"]:
        if key in item:
            return item[key]
    raise KeyError("Cannot find selector_scores/scores/cometScores/COMET_scores")


def compute_gate_record(item, threshold, text_index=5, score_field="auto"):
    scores = [float(x) for x in get_scores(item, score_field)]
    if len(scores) <= text_index:
        raise ValueError(f"Need at least {text_index + 1} scores, got {len(scores)} for {get_clip_id(item)}")
    visual_scores = scores[:text_index]
    best_visual_score = max(visual_scores)
    best_visual_index = visual_scores.index(best_visual_score)
    text_score = scores[text_index]
    visual_gain = best_visual_score - text_score
    need_visual = visual_gain >= threshold
    chosen_index = best_visual_index if need_visual else text_index
    record = {
        "clipID": get_clip_id(item),
        "src": item.get("src") or item.get("sentence") or item.get("EN_sentence") or item.get("ZH_sentence"),
        "selector_scores": scores,
        "best_visual_index": best_visual_index,
        "text_index": text_index,
        "best_visual_score": best_visual_score,
        "text_score": text_score,
        "visual_gain": visual_gain,
        "threshold": threshold,
        "need_visual": need_visual,
        "picIDChoose": chosen_index,
        "route": "image-text" if need_visual else "text-only",
    }
    if "clusteredInfo" in item:
        record["clusteredInfo"] = item["clusteredInfo"]
        record["selected_frame_id"] = item["clusteredInfo"][best_visual_index] if need_visual else None
    return record


def summarize(records):
    total = len(records)
    visual = sum(1 for r in records if r["need_visual"])
    gains = [r["visual_gain"] for r in records]
    chosen = {}
    for r in records:
        chosen[str(r["picIDChoose"])] = chosen.get(str(r["picIDChoose"]), 0) + 1
    return {
        "total": total,
        "visual_count": visual,
        "text_only_count": total - visual,
        "visual_call_rate": visual / total if total else 0.0,
        "avg_frames_per_sample": visual / total if total else 0.0,
        "avg_visual_gain": mean(gains) if gains else 0.0,
        "min_visual_gain": min(gains) if gains else 0.0,
        "max_visual_gain": max(gains) if gains else 0.0,
        "chosen_index_distribution": chosen,
    }


def write_summary_md(summary_by_threshold, output_path):
    lines = ["# Visual Gate Summary", ""]
    lines.append("| threshold | total | visual_count | visual_call_rate | avg_visual_gain | avg_frames_per_sample |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for threshold, summary in summary_by_threshold:
        lines.append(
            f"| {threshold:.4g} | {summary['total']} | {summary['visual_count']} | "
            f"{summary['visual_call_rate']:.4f} | {summary['avg_visual_gain']:.4f} | "
            f"{summary['avg_frames_per_sample']:.4f} |"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compute visual necessity gate outputs.")
    parser.add_argument("--input", required=True, help="JSON file with selector_scores or cometScores")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[-0.05, 0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--text_index", type=int, default=5)
    parser.add_argument("--score_field", default="auto")
    parser.add_argument("--prefix", default="gate")
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    output_dir = Path(args.output_dir) if args.output_dir else paths["output_root"] / "gate_outputs"
    records_in = load_json(args.input)
    summary_by_threshold = []
    for threshold in args.thresholds:
        records = [compute_gate_record(item, threshold, args.text_index, args.score_field) for item in records_in]
        summary = summarize(records)
        suffix = str(threshold).replace("-", "neg").replace(".", "p")
        save_json(records, output_dir / f"{args.prefix}_threshold_{suffix}.json")
        save_json(summary, output_dir / f"{args.prefix}_threshold_{suffix}_summary.json")
        summary_by_threshold.append((threshold, summary))
    write_summary_md(summary_by_threshold, output_dir / f"{args.prefix}_summary.md")
    print(json.dumps({"output_dir": str(output_dir), "thresholds": args.thresholds}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
