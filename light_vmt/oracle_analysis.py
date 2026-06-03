import argparse
import json
from pathlib import Path

import numpy as np

from light_vmt.paths import ensure_output_dirs, load_paths
from light_vmt.visual_gate import save_json


def metric_scores(srcs, preds, refs, metric, trans_direction):
    if metric == "comet":
        from utils.computeTransMetric import computeCOMET

        return np.array(computeCOMET(srcs, preds, refs)["scores"], dtype=float) * 100
    if metric == "bleu":
        from sacrebleu.metrics import BLEU

        zh_target = trans_direction.endswith("zh")
        bleu = BLEU(tokenize="zh") if zh_target else BLEU()
        return np.array(
            [bleu.corpus_score([pred], [[ref]]).score for pred, ref in zip(preds, refs)],
            dtype=float,
        )
    if metric == "bleurt":
        from utils.computeTransMetric import computeBLEURT

        return np.array(computeBLEURT(preds, refs, returnAverage=False), dtype=float) * 100
    raise ValueError(f"Unsupported metric: {metric}")


def load_result_sets(result_dirs):
    from utils.computeTransMetric import getSrcPredsRefs

    result_sets = [getSrcPredsRefs(result_dir) for result_dir in result_dirs]
    first_results_path = Path(result_dirs[0]) / "results.json"
    with first_results_path.open("r", encoding="utf-8") as f:
        first_results = json.load(f)
    clip_ids = [item["clipID"] for item in first_results]
    for result_dir in result_dirs[1:]:
        with (Path(result_dir) / "results.json").open("r", encoding="utf-8") as f:
            other_clip_ids = [item["clipID"] for item in json.load(f)]
        if other_clip_ids != clip_ids:
            raise ValueError(f"Result order mismatch in {result_dir}")
    return clip_ids, result_sets


def oracle_analysis(result_dirs, labels, metric, trans_direction):
    clip_ids, result_sets = load_result_sets(result_dirs)
    srcs, refs = result_sets[0][0], result_sets[0][2]
    score_matrix = np.vstack(
        [
            metric_scores(src, preds, ref, metric, trans_direction)
            for src, preds, ref in result_sets
        ]
    )
    best_indices = np.argmax(score_matrix, axis=0)
    records = []
    for sample_idx, clip_id in enumerate(clip_ids):
        best_branch = int(best_indices[sample_idx])
        records.append(
            {
                "clipID": clip_id,
                "best_branch_index": best_branch,
                "best_branch_label": labels[best_branch],
                f"{metric}_scores": score_matrix[:, sample_idx].astype(float).tolist(),
                "src": srcs[sample_idx],
                "ref": refs[sample_idx],
                "preds_by_branch": {
                    label: result_sets[branch_idx][1][sample_idx]
                    for branch_idx, label in enumerate(labels)
                },
            }
        )
    branch_counts = {label: 0 for label in labels}
    for record in records:
        branch_counts[record["best_branch_label"]] += 1
    return records, {
        "metric": metric,
        "trans_direction": trans_direction,
        "sample_count": len(records),
        "branch_labels": labels,
        "oracle_mean_score": float(np.mean(np.max(score_matrix, axis=0))) if len(records) else 0.0,
        "branch_best_counts": branch_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute oracle best-branch analysis for multiple VMT result dirs.")
    parser.add_argument("--result_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--metric", choices=["comet", "bleu", "bleurt"], default="comet")
    parser.add_argument("--trans_direction", default="en_zh")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix", default="oracle")
    args = parser.parse_args()

    labels = args.labels or [Path(path).name for path in args.result_dirs]
    if len(labels) != len(args.result_dirs):
        raise ValueError("--labels length must match --result_dirs length")

    paths = ensure_output_dirs(load_paths())
    output_dir = Path(args.output_dir) if args.output_dir else paths["output_root"] / "reports"
    records, summary = oracle_analysis(args.result_dirs, labels, args.metric, args.trans_direction)
    save_json(records, output_dir / f"{args.prefix}_{args.metric}_records.json")
    save_json(summary, output_dir / f"{args.prefix}_{args.metric}_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
