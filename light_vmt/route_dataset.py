import argparse
import json
from pathlib import Path

from light_vmt.paths import ensure_output_dirs, load_paths
from light_vmt.visual_gate import get_clip_id, load_json, save_json


def route_dataset(dataset, gate_records):
    gate_by_clip = {item["clipID"]: item for item in gate_records}
    text_only = []
    visual_needed = []
    picid_visual = []
    missing = []
    for item in dataset:
        clip_id = get_clip_id(item)
        gate = gate_by_clip.get(clip_id)
        if gate is None:
            missing.append(clip_id)
            continue
        enriched = dict(item)
        enriched["light_vmt_route"] = gate["route"]
        enriched["visual_gain"] = gate["visual_gain"]
        enriched["need_visual"] = gate["need_visual"]
        if gate["need_visual"]:
            visual_needed.append(enriched)
            picid_visual.append({"clipID": clip_id, "picIDChoose": gate["picIDChoose"]})
        else:
            text_only.append(enriched)
    summary = {
        "dataset_count": len(dataset),
        "matched_count": len(text_only) + len(visual_needed),
        "missing_count": len(missing),
        "text_only_count": len(text_only),
        "visual_needed_count": len(visual_needed),
        "visual_call_rate": len(visual_needed) / (len(text_only) + len(visual_needed)) if (text_only or visual_needed) else 0.0,
        "missing_clip_ids": missing[:50],
    }
    return text_only, visual_needed, picid_visual, summary


def main():
    parser = argparse.ArgumentParser(description="Split dataset by visual gate route.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix", default="routed")
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    output_dir = Path(args.output_dir) if args.output_dir else paths["output_root"] / "routed_datasets"
    dataset = load_json(args.dataset)
    gate_records = load_json(args.gate)
    text_only, visual_needed, picid_visual, summary = route_dataset(dataset, gate_records)
    save_json(text_only, output_dir / f"{args.prefix}_text_only.json")
    save_json(visual_needed, output_dir / f"{args.prefix}_visual_needed.json")
    save_json(picid_visual, output_dir / f"{args.prefix}_picIDChoose_visual.json")
    save_json(summary, output_dir / f"{args.prefix}_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
