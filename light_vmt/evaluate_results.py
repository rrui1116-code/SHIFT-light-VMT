import argparse
import json
from pathlib import Path

from light_vmt.paths import ensure_output_dirs, load_paths
from light_vmt.visual_gate import save_json


def load_results(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    srcs, preds, refs = [], [], []
    for item in data:
        srcs.append(item["src"])
        preds.append(item["preds"])
        refs.append(item["refs"])
    return srcs, preds, refs


def looks_chinese(text):
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def compute_bleu(preds, refs, is_zh=False):
    from sacrebleu.metrics import BLEU

    metric = BLEU(tokenize="zh") if is_zh else BLEU()
    return metric.corpus_score(preds, [refs]).score


def compute_chrf(preds, refs):
    from sacrebleu.metrics import CHRF

    return CHRF().corpus_score(preds, [refs]).score


def evaluate_results(results_path, metrics):
    srcs, preds, refs = load_results(results_path)
    is_zh = looks_chinese(refs[0]) if refs else False
    scores = {"sample_count": len(preds), "metrics": {}}
    metric_names = {metric.lower(): metric for metric in metrics}

    if "bleu" in metric_names:
        scores["metrics"]["BLEU"] = compute_bleu(preds, refs, is_zh)
    if "meteor" in metric_names:
        from utils.computeTransMetric import computeMETEOR

        scores["metrics"]["METEOR"] = computeMETEOR(preds, refs, is_zh) * 100
    if "chrf" in metric_names:
        scores["metrics"]["chrF"] = compute_chrf(preds, refs)
    if "comet" in metric_names:
        from utils.computeTransMetric import computeCOMET

        scores["metrics"]["COMET"] = computeCOMET(srcs, preds, refs)["mean_score"] * 100
    if "bleurt" in metric_names:
        from utils.computeTransMetric import computeBLEURT

        scores["metrics"]["BLEURT"] = computeBLEURT(preds, refs) * 100
    return scores


def write_markdown(scores, output_path):
    lines = ["# Light-VMT Translation Evaluation", ""]
    lines.append(f"- sample_count: {scores['sample_count']}")
    for metric, score in scores["metrics"].items():
        lines.append(f"- {metric}: {score}")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Light-VMT merged results JSON file.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--metrics", nargs="+", default=["BLEU", "chrF"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    paths = ensure_output_dirs(load_paths())
    output_dir = Path(args.output_dir) if args.output_dir else paths["output_root"] / "reports"
    prefix = args.prefix or Path(args.results).stem
    scores = evaluate_results(args.results, args.metrics)
    save_json(scores, output_dir / f"{prefix}_translation_metrics.json")
    write_markdown(scores, output_dir / f"{prefix}_translation_metrics.md")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
