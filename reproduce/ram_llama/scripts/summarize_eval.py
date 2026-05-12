#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RAM Llama evaluation results.")
    parser.add_argument("run_dir", help="Path like /tmp/shared-storage/ExpertGym/LLaMA/results/<run>/<model>")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    for metrics_path in sorted(run_dir.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        rows.append((m["benchmark"], m["accuracy"], m["correct"], m["total"]))
    if not rows:
        raise SystemExit(f"No metrics found under {run_dir}")
    print("| benchmark | accuracy | correct/total |")
    print("|---|---:|---:|")
    for name, acc, correct, total in rows:
        print(f"| {name} | {acc:.4f} | {correct}/{total} |")
    macro = sum(r[1] for r in rows) / len(rows)
    print(f"| macro | {macro:.4f} | - |")


if __name__ == "__main__":
    main()
