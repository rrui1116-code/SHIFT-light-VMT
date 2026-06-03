import argparse
import json
from pathlib import Path

from light_vmt.visual_gate import get_clip_id, load_json, save_json
from light_vmt.paths import ensure_output_dirs, load_paths


def index_results(results):
    return {item["clipID"]: item for item in results}


def merge_results(original_dataset, gate_records, text_results, visual_results):
    gate_by_clip = {item["clipID"]: item for item in gate_records}
    text_by_clip = index_results(text_results)
    visual_by_clip = index_results(visual_results)
    merged = []
    missing = []
    for item in original_dataset:
        clip_id = get_clip_id(item)
        gate = gate_by_clip.get(clip_id)
        if gate is None:
            missing.append({"clipID": clip_id, "reason": "missing_gate"})
            continue
        source = visual_by_clip if gate["need_visual"] else text_by_clip
        result = source.get(clip_id)
        if result is None:
            missing.append({"clipID": clip_id, "reason": "missing_visual_result" if gate["need_visual"] else "missing_text_result"})
            continue
        merged.append({
            "clipID": clip_id,
            "src": result.get("src") or gate.get("src"),
            "preds": result.get("preds"),
            "refs": result.get("refs"),
            "route": gate["route"],
            "need_visual": gate["need_visual"],
            "visual_gain": gate["visual_gain"],
            "picIDChoose": gate["picIDChoose"],
            "selected_frame_id": gate.get("selected_frame_id"),
        })
    summary = {
        "original_count": len(original_dataset),
        "merged_count": len(merged),
        "missing_count": len(missing),
        "visual_count": sum(1 for item in merged if item["need_visual"]),
        "text_only_count": sum(1 for item in merged if not item["need_visual"]),
        "missing": missing[:50],
    }
    return merged, summary


def main():
    parser = argparse.ArgumentParser(description="Merge text-only and visual branch results.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--text_results", required=True)
    parser.add_argument("--visual_results", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    output = Path(args.output) if args.output else paths["output_root"] / "merged_results" / "merged_results.json"
    merged, summary = merge_results(
        load_json(args.dataset), load_json(args.gate), load_json(args.text_results), load_json(args.visual_results)
    )
    save_json(merged, output)
    save_json(summary, output.with_name(output.stem + "_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
