import argparse
import json
from pathlib import Path

from light_vmt.visual_gate import get_clip_id, load_json, save_json


def text_of(item, source_language="en"):
    return (
        item.get("src")
        or item.get("sentence")
        or item.get(f"{source_language.upper()}_sentence")
        or item.get("EN_sentence")
        or item.get("ZH_sentence")
    )


def ref_of(item, target_language="zh"):
    return item.get("refs") or item.get(f"{target_language.upper()}_sentence") or item.get("ZH_sentence") or item.get("EN_sentence")


def selection_label(record):
    if record.get("selection_type") == "text-only":
        return "text-only"
    if record.get("need_visual") is False or record.get("route") == "text-only":
        return "text-only"
    if record.get("picIDChoose") == 5:
        return "text-only"
    if "selected_frame_id" in record and record.get("selected_frame_id") is not None:
        return f"video:{record.get('selected_frame_id')}"
    if "selected_frame_ids" in record and record.get("selected_frame_ids"):
        return "video:" + ",".join(str(x) for x in record["selected_frame_ids"])
    if "picIDChoose" in record:
        return f"video_candidate:{record['picIDChoose']}"
    return "unknown"


def is_video(record):
    return selection_label(record).startswith("video")


def index_by_clip(items):
    return {get_clip_id(item): item for item in items}


def load_results(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return {item["clipID"]: item for item in load_json(p)}


def sentence_chrf(pred, ref):
    try:
        from sacrebleu.metrics import CHRF

        return CHRF().sentence_score(pred or "", [ref or ""]).score
    except Exception:
        return None


def build_cases(dataset, shift_records, gate_records, shift_results, gate_results):
    dataset_by_clip = index_by_clip(dataset)
    shift_by_clip = {item["clipID"]: item for item in shift_records}
    gate_by_clip = {item["clipID"]: item for item in gate_records}
    rows = []
    for clip_id in sorted(set(shift_by_clip) & set(gate_by_clip)):
        data_item = dataset_by_clip.get(clip_id, {})
        shift_result = shift_results.get(clip_id, {})
        gate_result = gate_results.get(clip_id, {})
        ref = gate_result.get("refs") or shift_result.get("refs") or ref_of(data_item)
        shift_pred = shift_result.get("preds")
        gate_pred = gate_result.get("preds")
        shift_score = sentence_chrf(shift_pred, ref) if shift_pred is not None else None
        gate_score = sentence_chrf(gate_pred, ref) if gate_pred is not None else None
        rows.append({
            "sample_id": clip_id,
            "source_sentence": shift_result.get("src") or gate_result.get("src") or text_of(data_item),
            "reference_translation": ref,
            "original_SHIFT_selection": selection_label(shift_by_clip[clip_id]),
            "SHIFT_Gate_selection": selection_label(gate_by_clip[clip_id]),
            "original_SHIFT_translation": shift_pred,
            "SHIFT_Gate_translation": gate_pred,
            "sentence_chrF_original_SHIFT": shift_score,
            "sentence_chrF_SHIFT_Gate": gate_score,
            "sentence_chrF_delta_gate_minus_shift": (gate_score - shift_score) if gate_score is not None and shift_score is not None else None,
            "original_SHIFT_raw_selection": shift_by_clip[clip_id],
            "SHIFT_Gate_raw_selection": gate_by_clip[clip_id],
        })
    return rows


def summarize(shift_records, gate_records):
    shift_by_clip = {item["clipID"]: item for item in shift_records}
    gate_by_clip = {item["clipID"]: item for item in gate_records}
    common = sorted(set(shift_by_clip) & set(gate_by_clip))
    shift_video_gate_text = 0
    shift_text_gate_video = 0
    consistent = 0
    inconsistent = 0
    for clip_id in common:
        shift_is_video = is_video(shift_by_clip[clip_id])
        gate_is_video = is_video(gate_by_clip[clip_id])
        if shift_is_video == gate_is_video:
            consistent += 1
        else:
            inconsistent += 1
        if shift_is_video and not gate_is_video:
            shift_video_gate_text += 1
        if not shift_is_video and gate_is_video:
            shift_text_gate_video += 1
    return {
        "common_count": len(common),
        "original_SHIFT_video_but_SHIFT_Gate_text_count": shift_video_gate_text,
        "original_SHIFT_text_but_SHIFT_Gate_video_count": shift_text_gate_video,
        "consistent_count": consistent,
        "inconsistent_count": inconsistent,
        "inconsistent_rate": inconsistent / len(common) if common else 0.0,
    }


def pick_case_studies(rows, limit):
    scored = [row for row in rows if row["sentence_chrF_delta_gate_minus_shift"] is not None]
    gate_improves = sorted(
        [row for row in scored if row["sentence_chrF_delta_gate_minus_shift"] > 0],
        key=lambda row: row["sentence_chrF_delta_gate_minus_shift"],
        reverse=True,
    )[:limit]
    gate_worse = sorted(
        [row for row in scored if row["sentence_chrF_delta_gate_minus_shift"] < 0],
        key=lambda row: row["sentence_chrF_delta_gate_minus_shift"],
    )[:limit]
    gate_text = [row for row in rows if row["SHIFT_Gate_selection"] == "text-only"]
    gate_text_reasonable = [
        row for row in gate_text
        if row["sentence_chrF_delta_gate_minus_shift"] is None or row["sentence_chrF_delta_gate_minus_shift"] >= 0
    ][:limit]
    wrongly_filtered = [
        row for row in gate_text
        if row["original_SHIFT_selection"].startswith("video")
        and row["sentence_chrF_delta_gate_minus_shift"] is not None
        and row["sentence_chrF_delta_gate_minus_shift"] < 0
    ][:limit]
    return {
        "gate_improves_translation": gate_improves,
        "gate_worsens_translation": gate_worse,
        "gate_text_only_reasonable_or_manual_check": gate_text_reasonable,
        "gate_wrongly_filters_visual": wrongly_filtered,
    }


def write_markdown(summary, output_path):
    lines = ["# SHIFT vs SHIFT+Gate Selection Analysis", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare Original SHIFT selections with SHIFT+Gate selections.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shift_selected", required=True)
    parser.add_argument("--gate_records", required=True)
    parser.add_argument("--shift_results", default=None)
    parser.add_argument("--gate_results", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_limit", type=int, default=5)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    shift_records = load_json(args.shift_selected)
    gate_records = load_json(args.gate_records)
    shift_results = load_results(args.shift_results)
    gate_results = load_results(args.gate_results)

    output_dir = Path(args.output_dir)
    rows = build_cases(dataset, shift_records, gate_records, shift_results, gate_results)
    summary = summarize(shift_records, gate_records)
    summary["case_count"] = len(rows)
    summary["has_translation_results"] = bool(shift_results and gate_results)
    case_studies = pick_case_studies(rows, args.case_limit)

    save_json(summary, output_dir / "shift_vs_gate_selection_summary.json")
    save_json(rows, output_dir / "shift_vs_gate_cases.json")
    save_json(case_studies, output_dir / "case_study_candidates.json")
    write_markdown(summary, output_dir / "shift_vs_gate_selection_summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
