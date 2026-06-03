import argparse
import json
from pathlib import Path

from light_vmt.visual_gate import get_clip_id, load_json, save_json


def normalize_clip_id(item):
    if "videoClipID" in item:
        return item["videoClipID"]
    if "videoClipId" in item:
        return item["videoClipId"]
    return get_clip_id(item)


def index_reference(reference_data):
    index = {}
    for item in reference_data:
        clip_id = normalize_clip_id(item)
        index[clip_id] = item
    return index


def to_inference_item(item, reference):
    merged = dict(reference)
    merged.update(item)
    clip_id = normalize_clip_id(merged)
    if "videoClipId" not in merged:
        merged["videoClipId"] = clip_id
    if "videoClipID" not in merged:
        merged["videoClipID"] = clip_id
    if "EN_sentence" not in merged:
        merged["EN_sentence"] = merged.get("sentence") or reference.get("sentence")
    if "ZH_sentence" not in merged:
        merged["ZH_sentence"] = merged.get("chSentence") or reference.get("chSentence")
    if "video_id" not in merged or "clip_id" not in merged:
        parts = clip_id.rsplit("_", 2)
        if len(parts) == 3:
            merged.setdefault("video_id", parts[0])
            try:
                merged.setdefault("clip_id", int(parts[2]))
            except ValueError:
                merged.setdefault("clip_id", parts[2])
    return merged


def prepare_dataset(routed_data, reference_data):
    ref_by_clip = index_reference(reference_data)
    output = []
    missing = []
    for item in routed_data:
        clip_id = normalize_clip_id(item)
        reference = ref_by_clip.get(clip_id)
        if reference is None:
            missing.append(clip_id)
            continue
        output.append(to_inference_item(item, reference))
    return output, {
        "input_count": len(routed_data),
        "output_count": len(output),
        "missing_count": len(missing),
        "missing_clip_ids": missing[:50],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare routed Light-VMT data for vmtDatasetForLLM inference.")
    parser.add_argument("--routed", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output, summary = prepare_dataset(load_json(args.routed), load_json(args.reference))
    save_json(output, args.output)
    save_json(summary, Path(args.output).with_name(Path(args.output).stem + "_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
