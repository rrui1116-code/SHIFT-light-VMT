import argparse
import json
import random
import shlex
from pathlib import Path

from light_vmt.paths import ensure_output_dirs, load_paths
from light_vmt.route_dataset import route_dataset
from light_vmt.visual_gate import compute_gate_record, get_clip_id, load_json, save_json, summarize


TEXT_INDEX = 5


def clip_text(item, source_language="en"):
    return (
        item.get("src")
        or item.get("sentence")
        or item.get(f"{source_language.upper()}_sentence")
        or item.get("EN_sentence")
        or item.get("ZH_sentence")
    )


def clip_ref(item, target_language="zh"):
    return item.get(f"{target_language.upper()}_sentence") or item.get("ZH_sentence") or item.get("EN_sentence")


def build_cluster_map(cluster_path):
    clusters = load_json(cluster_path)
    return {key: value for clip_cluster in clusters for key, value in clip_cluster.items()}


def make_dirs(root):
    root = Path(root)
    for name in ["config", "logs", "predictions", "selected_frames", "metrics", "gate_analysis"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def q(path):
    return shlex.quote(str(path))


def uniform_indices(count, top_k):
    if count <= 0:
        return []
    if top_k >= count:
        return list(range(count))
    if top_k <= 1:
        return [count // 2]
    indices = [round(i * (count - 1) / (top_k - 1)) for i in range(top_k)]
    deduped = []
    for idx in indices:
        if idx not in deduped:
            deduped.append(idx)
    fill = 0
    while len(deduped) < top_k and fill < count:
        if fill not in deduped:
            deduped.append(fill)
        fill += 1
    return sorted(deduped[:top_k])


def selected_record(dataset_item, cluster_map, candidate_indices, mode, seed=None):
    clip_id = get_clip_id(dataset_item)
    frame_ids = [cluster_map[clip_id][idx] for idx in candidate_indices]
    record = {
        "clipID": clip_id,
        "mode": mode,
        "src": clip_text(dataset_item),
        "ref": clip_ref(dataset_item),
        "selected_candidate_indices": candidate_indices,
        "selected_frame_ids": frame_ids,
        "selection_type": "text-only" if not candidate_indices else "video",
    }
    if seed is not None:
        record["seed"] = seed
    if len(candidate_indices) == 1:
        record["picIDChoose"] = candidate_indices[0]
        record["selected_frame_id"] = frame_ids[0]
    return record


def picid_records(selected_records):
    return [
        {"clipID": item["clipID"], "picIDChoose": item["picIDChoose"]}
        for item in selected_records
        if "picIDChoose" in item
    ]


def selector_choice_record(score_item, dataset_by_clip, cluster_map):
    clip_id = get_clip_id(score_item)
    scores = [float(x) for x in score_item["selector_scores"]]
    max_score = max(scores)
    max_indices = [idx for idx, score in enumerate(scores) if score == max_score]
    if TEXT_INDEX in max_indices:
        best_index = TEXT_INDEX
    else:
        best_index = max_indices[0]
    dataset_item = dataset_by_clip.get(clip_id, score_item)
    if best_index == TEXT_INDEX:
        return {
            "clipID": clip_id,
            "mode": "shift",
            "src": clip_text(dataset_item),
            "ref": clip_ref(dataset_item),
            "selection_type": "text-only",
            "picIDChoose": TEXT_INDEX,
            "selector_scores": scores,
            "text_score": scores[TEXT_INDEX],
            "best_visual_score": max(scores[:TEXT_INDEX]),
        }
    frame_id = cluster_map[clip_id][best_index]
    return {
        "clipID": clip_id,
        "mode": "shift",
        "src": clip_text(dataset_item),
        "ref": clip_ref(dataset_item),
        "selection_type": "video",
        "picIDChoose": best_index,
        "selected_candidate_indices": [best_index],
        "selected_frame_ids": [frame_id],
        "selected_frame_id": frame_id,
        "selector_scores": scores,
        "text_score": scores[TEXT_INDEX],
        "best_visual_score": scores[best_index],
    }


def write_command_files(exp_dir, commands):
    command_text = "set -euo pipefail\n" + "\n".join(commands) + "\n"
    (exp_dir / "config" / "commands.sh").write_text(command_text, encoding="utf-8")
    (exp_dir / "logs" / "commands.log").write_text(command_text, encoding="utf-8")


def inference_command(args, dataset_path, dataset_type, output_dir, extra=None):
    cmd = [
        args.python_bin, "codes/inference.py",
        "--dataset_path", q(dataset_path),
        "--dataset_type", dataset_type,
        "--model_type", args.model_type,
        "--model_path", q(args.model_path),
        "--model_name", args.model_name,
        "--source_language", args.source_language,
        "--target_language", args.target_language,
        "--prompt_language", args.prompt_language,
        "--batch_size", str(args.batch_size),
        "--max_src_length", str(args.max_src_length),
        "--max_tgt_length", str(args.max_tgt_length),
        "--output_dir", q(output_dir),
        "--trans_metric",
    ]
    if extra:
        cmd.extend(extra)
    if args.vatex:
        cmd.append("--vatex")
    return " ".join(cmd)


def evaluate_command(args, results_path, metrics_dir, prefix):
    return " ".join([
        args.python_bin, "-m", "light_vmt.evaluate_results",
        "--results", q(results_path),
        "--metrics", "BLEU", "chrF",
        "--output_dir", q(metrics_dir),
        "--prefix", prefix,
    ])


def build_text_only(args, exp_dir):
    save_json({"mode": "text_only", "dataset": args.dataset}, exp_dir / "config" / "config.json")
    predictions_dir = exp_dir / "predictions"
    metrics_dir = exp_dir / "metrics"
    commands = [
        inference_command(args, args.dataset, "text", predictions_dir),
        evaluate_command(args, predictions_dir / "results.json", metrics_dir, "text_only"),
    ]
    write_command_files(exp_dir, commands)


def build_random(args, exp_dir, dataset, cluster_map):
    rng = random.Random(args.seed)
    records = []
    for item in dataset:
        clip_id = get_clip_id(item)
        candidates = list(range(len(cluster_map[clip_id])))
        chosen = sorted(rng.sample(candidates, min(args.top_k, len(candidates))))
        records.append(selected_record(item, cluster_map, chosen, "random", seed=args.seed))
    save_json(records, exp_dir / "selected_frames" / "selected_frames.json")
    if args.top_k == 1:
        save_json(picid_records(records), exp_dir / "selected_frames" / "picIDChoose.json")
    save_json({"mode": "random", "seed": args.seed, "top_k": args.top_k, "sampling_scope": "SHIFT clustered candidate frames"}, exp_dir / "config" / "config.json")
    extra = ["--image_selection", "light-vmt-multi", "--selected_frames_path", q(exp_dir / "selected_frames" / "selected_frames.json")]
    if args.top_k == 1:
        extra = ["--image_selection", "select", "--cluster_path", q(args.cluster_path), "--picID_path", q(exp_dir / "selected_frames" / "picIDChoose.json")]
    commands = [
        inference_command(args, args.dataset, "image-text", exp_dir / "predictions", extra),
        evaluate_command(args, exp_dir / "predictions" / "results.json", exp_dir / "metrics", f"random_seed{args.seed}_top{args.top_k}"),
    ]
    write_command_files(exp_dir, commands)


def build_uniform(args, exp_dir, dataset, cluster_map):
    records = []
    for item in dataset:
        clip_id = get_clip_id(item)
        indices = uniform_indices(len(cluster_map[clip_id]), args.top_k)
        records.append(selected_record(item, cluster_map, indices, "uniform"))
    save_json(records, exp_dir / "selected_frames" / "selected_frames.json")
    if args.top_k == 1:
        save_json(picid_records(records), exp_dir / "selected_frames" / "picIDChoose.json")
    save_json({"mode": "uniform", "top_k": args.top_k, "sampling_scope": "SHIFT clustered candidate frames"}, exp_dir / "config" / "config.json")
    extra = ["--image_selection", "light-vmt-multi", "--selected_frames_path", q(exp_dir / "selected_frames" / "selected_frames.json")]
    if args.top_k == 1:
        extra = ["--image_selection", "select", "--cluster_path", q(args.cluster_path), "--picID_path", q(exp_dir / "selected_frames" / "picIDChoose.json")]
    commands = [
        inference_command(args, args.dataset, "image-text", exp_dir / "predictions", extra),
        evaluate_command(args, exp_dir / "predictions" / "results.json", exp_dir / "metrics", f"uniform_top{args.top_k}"),
    ]
    write_command_files(exp_dir, commands)


def build_shift(args, exp_dir, dataset, cluster_map):
    if not args.selector_scores:
        raise ValueError("--selector_scores is required for mode=shift")
    dataset_by_clip = {get_clip_id(item): item for item in dataset}
    records = [selector_choice_record(item, dataset_by_clip, cluster_map) for item in load_json(args.selector_scores)]
    save_json(records, exp_dir / "selected_frames" / "selected_frames.json")
    visual_records = [item for item in records if item["selection_type"] == "video"]
    text_clip_ids = {item["clipID"] for item in records if item["selection_type"] == "text-only"}
    visual_clip_ids = {item["clipID"] for item in visual_records}
    text_dataset = [item for item in dataset if get_clip_id(item) in text_clip_ids]
    visual_dataset = [item for item in dataset if get_clip_id(item) in visual_clip_ids]
    shift_gate_records = []
    for item in records:
        need_visual = item["selection_type"] == "video"
        shift_gate_records.append({
            "clipID": item["clipID"],
            "src": item.get("src"),
            "route": "image-text" if need_visual else "text-only",
            "need_visual": need_visual,
            "visual_gain": item.get("best_visual_score", 0.0) - item.get("text_score", 0.0),
            "picIDChoose": item["picIDChoose"],
            "selected_frame_id": item.get("selected_frame_id"),
        })
    save_json(text_dataset, exp_dir / "selected_frames" / "text_only_dataset.json")
    save_json(visual_dataset, exp_dir / "selected_frames" / "visual_needed_dataset.json")
    save_json(shift_gate_records, exp_dir / "selected_frames" / "shift_route_records.json")
    save_json(picid_records(visual_records), exp_dir / "selected_frames" / "picIDChoose_visual.json")
    save_json({
        "mode": "shift",
        "selector_scores": args.selector_scores,
        "text_only_candidate_index": TEXT_INDEX,
        "text_only_count": sum(1 for item in records if item["selection_type"] == "text-only"),
        "video_count": len(visual_records),
        "total": len(records),
    }, exp_dir / "config" / "config.json")
    text_pred = exp_dir / "predictions" / "text_branch"
    visual_pred = exp_dir / "predictions" / "visual_branch"
    merged_path = exp_dir / "predictions" / "merged_results.json"
    commands = [
        inference_command(args, exp_dir / "selected_frames" / "text_only_dataset.json", "text", text_pred),
        inference_command(args, exp_dir / "selected_frames" / "visual_needed_dataset.json", "image-text", visual_pred, ["--image_selection", "select", "--cluster_path", q(args.cluster_path), "--picID_path", q(exp_dir / "selected_frames" / "picIDChoose_visual.json")]),
        " ".join([
            args.python_bin, "-m", "light_vmt.merge_results",
            "--dataset", q(args.dataset),
            "--gate", q(exp_dir / "selected_frames" / "shift_route_records.json"),
            "--text_results", q(text_pred / "results.json"),
            "--visual_results", q(visual_pred / "results.json"),
            "--output", q(merged_path),
        ]),
        evaluate_command(args, merged_path, exp_dir / "metrics", "shift"),
    ]
    write_command_files(exp_dir, commands)


def build_shift_gate(args, exp_dir, dataset, cluster_map):
    if not args.selector_scores:
        raise ValueError("--selector_scores is required for mode=shift_gate")
    selector_items = load_json(args.selector_scores)
    gate_records = [compute_gate_record(item, args.threshold, TEXT_INDEX) for item in selector_items]
    for record in gate_records:
        if record["need_visual"] and record["clipID"] in cluster_map:
            record["selected_frame_id"] = cluster_map[record["clipID"]][record["picIDChoose"]]
    text_only, visual_needed, picid_visual, route_summary = route_dataset(dataset, gate_records)
    gate_summary = summarize(gate_records)
    save_json(gate_records, exp_dir / "gate_analysis" / "gate_records.json")
    save_json(gate_summary, exp_dir / "gate_analysis" / "gate_summary.json")
    save_json(route_summary, exp_dir / "gate_analysis" / "route_summary.json")
    save_json(text_only, exp_dir / "selected_frames" / "text_only_dataset.json")
    save_json(visual_needed, exp_dir / "selected_frames" / "visual_needed_dataset.json")
    save_json(picid_visual, exp_dir / "selected_frames" / "picIDChoose_visual.json")
    save_json({"mode": "shift_gate", "threshold": args.threshold, "selector_scores": args.selector_scores}, exp_dir / "config" / "config.json")
    text_pred = exp_dir / "predictions" / "text_branch"
    visual_pred = exp_dir / "predictions" / "visual_branch"
    merged_path = exp_dir / "predictions" / "merged_results.json"
    commands = [
        inference_command(args, exp_dir / "selected_frames" / "text_only_dataset.json", "text", text_pred),
        inference_command(args, exp_dir / "selected_frames" / "visual_needed_dataset.json", "image-text", visual_pred, ["--image_selection", "select", "--cluster_path", q(args.cluster_path), "--picID_path", q(exp_dir / "selected_frames" / "picIDChoose_visual.json")]),
        " ".join([
            args.python_bin, "-m", "light_vmt.merge_results",
            "--dataset", q(args.dataset),
            "--gate", q(exp_dir / "gate_analysis" / "gate_records.json"),
            "--text_results", q(text_pred / "results.json"),
            "--visual_results", q(visual_pred / "results.json"),
            "--output", q(merged_path),
        ]),
        evaluate_command(args, merged_path, exp_dir / "metrics", "shift_gate"),
    ]
    write_command_files(exp_dir, commands)


def main():
    parser = argparse.ArgumentParser(description="Prepare reproducible Light-VMT experiment directories.")
    parser.add_argument("--mode", required=True, choices=["text_only", "random", "uniform", "shift", "shift_gate"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster_path", required=True)
    parser.add_argument("--selector_scores", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--model_path", default="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    parser.add_argument("--model_name", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_type", default="multimodal")
    parser.add_argument("--python_bin", default="python3")
    parser.add_argument("--vatex", action="store_true", default=True)
    parser.add_argument("--source_language", default="en")
    parser.add_argument("--target_language", default="zh")
    parser.add_argument("--prompt_language", default="en")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_src_length", type=int, default=1024)
    parser.add_argument("--max_tgt_length", type=int, default=256)
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    name = args.experiment_name or f"{args.mode}_top{args.top_k}_seed{args.seed}_t{args.threshold}"
    output_root = Path(args.output_root) if args.output_root else paths["output_root"] / "experiments"
    exp_dir = make_dirs(output_root / name)
    dataset = load_json(args.dataset)
    cluster_map = build_cluster_map(args.cluster_path)

    if args.mode == "text_only":
        build_text_only(args, exp_dir)
    elif args.mode == "random":
        build_random(args, exp_dir, dataset, cluster_map)
    elif args.mode == "uniform":
        build_uniform(args, exp_dir, dataset, cluster_map)
    elif args.mode == "shift":
        build_shift(args, exp_dir, dataset, cluster_map)
    elif args.mode == "shift_gate":
        build_shift_gate(args, exp_dir, dataset, cluster_map)

    print(json.dumps({"experiment_dir": str(exp_dir), "commands": str(exp_dir / "config" / "commands.sh")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
