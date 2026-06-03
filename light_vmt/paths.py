import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "light_vmt_paths.json"


def load_paths(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {key: Path(value) for key, value in data.items()}


def ensure_output_dirs(paths=None):
    paths = paths or load_paths()
    for key in ["output_root", "final_result_root"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    for child in [
        "selector_scores",
        "gate_outputs",
        "routed_datasets",
        "inference_text",
        "inference_visual",
        "merged_results",
        "reports",
    ]:
        (paths["output_root"] / child).mkdir(parents=True, exist_ok=True)
    return paths
