import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Write a compact Markdown summary from JSON summaries.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = ["# Light-VMT Report Summary", ""]
    for input_path in args.inputs:
        path = Path(input_path)
        lines.append(f"## {path.name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append("```json")
        lines.append(json.dumps(data, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
