import argparse
import json
from pathlib import Path

from light_vmt.paths import ensure_output_dirs, load_paths
from light_vmt.visual_gate import get_clip_id, get_scores, load_json, save_json


def choose_best_index(scores, text_index):
    max_score = max(scores)
    max_indices = [idx for idx, value in enumerate(scores) if value == max_score]
    return text_index if text_index in max_indices else max_indices[0]


def export_score_record(item, text_index=5, score_field="auto"):
    scores = [float(x) for x in get_scores(item, score_field)]
    if len(scores) <= text_index:
        raise ValueError(f"Need at least {text_index + 1} scores, got {len(scores)} for {get_clip_id(item)}")
    visual_scores = scores[:text_index]
    best_visual_score = max(visual_scores)
    best_visual_index = visual_scores.index(best_visual_score)
    text_score = scores[text_index]
    best_index = choose_best_index(scores, text_index)
    record = {
        "clipID": get_clip_id(item),
        "src": item.get("src") or item.get("sentence") or item.get("EN_sentence") or item.get("ZH_sentence"),
        "selector_scores": scores,
        "best_visual_index": int(best_visual_index),
        "text_index": int(text_index),
        "best_index": int(best_index),
        "picIDChoose": int(best_index),
        "best_visual_score": float(best_visual_score),
        "text_score": float(text_score),
        "visual_gain": float(best_visual_score - text_score),
    }
    if "clusteredInfo" in item:
        record["clusteredInfo"] = item["clusteredInfo"]
    return record


def main():
    parser = argparse.ArgumentParser(description="Export unified selector score records from selector or oracle data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--text_index", type=int, default=5)
    parser.add_argument("--score_field", default="auto")
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    output = Path(args.output) if args.output else paths["output_root"] / "selector_scores" / "selector_scores.json"
    records = [
        export_score_record(item, text_index=args.text_index, score_field=args.score_field)
        for item in load_json(args.input)
    ]
    save_json(records, output)
    print(json.dumps({"output": str(output), "count": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
