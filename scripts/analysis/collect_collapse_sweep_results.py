#!/usr/bin/env python3
"""Collect eval results from a collapse boundary sweep into a summary table.

Usage:
  python scripts/analysis/collect_collapse_sweep_results.py \
    /tmp/shared-storage/ExpertGym/analysis/collapse_sweep/tool_sweep

Reads sweep_manifest.jsonl + per-point eval summaries and produces:
  - results.tsv  (for spreadsheets)
  - results.json (for plotting)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_summary(eval_dir: Path, task: str) -> dict | None:
    """Search common eval output patterns for a task summary."""
    # The eval suite writes results under various paths; search them.
    patterns = [
        eval_dir / "tool" / "summary.json",
        eval_dir / "memory" / "summary.json",
        eval_dir / "code" / "summary.json",
    ]
    # Also search under experiment-level dirs
    for p in eval_dir.rglob("summary.json"):
        if task in str(p).lower():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def extract_tool_summary(sweep_dir: Path, tag: str) -> dict:
    """Extract tool metrics from various possible eval output locations."""
    # Search broadly
    for summary_path in sorted(sweep_dir.glob(f"{tag}/**/summary.json"), key=lambda p: len(str(p))):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # BFCL-style summary
        if "parallel" in data or "mean" in data:
            return {
                "tool_mean": data.get("mean", data.get("overall_mean")),
                "tool_live_mean": data.get("live_mean"),
                "tool_parallel": data.get("parallel"),
                "tool_parallel_multiple": data.get("parallel_multiple"),
            }
    return {}


def extract_memory_summary(sweep_dir: Path, tag: str) -> dict:
    """Extract memory metrics."""
    for summary_path in sorted(sweep_dir.glob(f"{tag}/**/summary.json"), key=lambda p: len(str(p))):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "mean_f1" in data or "eval_50" in data:
            return {
                "memory_em": data.get("mean_em"),
                "memory_f1": data.get("mean_f1"),
            }
    return {}


def extract_code_summary(sweep_dir: Path, tag: str) -> dict:
    """Extract code metrics."""
    for summary_path in sorted(sweep_dir.glob(f"{tag}/**/summary.json"), key=lambda p: len(str(p))):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "mean_acc" in data or "mean_bon" in data:
            return {
                "code_acc": data.get("mean_acc"),
                "code_bon": data.get("mean_bon"),
            }
    return {}


def main() -> None:
    sweep_dir = Path(sys.argv[1]).expanduser().resolve()
    manifest_path = sweep_dir / "sweep_manifest.jsonl"
    if not manifest_path.exists():
        print(f"[error] {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    results = []
    for entry in entries:
        tag = entry["tag"]
        row = {
            "tag": tag,
            "sweep_expert": entry["sweep_expert"],
            "sweep_value": entry["sweep_value"],
            **entry["coefficients"],
        }
        row.update(extract_tool_summary(sweep_dir, tag))
        row.update(extract_memory_summary(sweep_dir, tag))
        row.update(extract_code_summary(sweep_dir, tag))
        results.append(row)

    # Write JSON
    results_json = sweep_dir / "results.json"
    results_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Write TSV
    if results:
        cols = list(results[0].keys())
        results_tsv = sweep_dir / "results.tsv"
        with results_tsv.open("w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for row in results:
                f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
        print(f"[collect] Wrote {len(results)} rows to {results_tsv}")

    # Print table
    print()
    print(f"{'coeff':>8}  {'tool':>8}  {'mem_f1':>8}  {'code_bon':>8}")
    print("-" * 40)
    for row in results:
        tool = row.get("tool_mean", "")
        mem = row.get("memory_f1", "")
        code = row.get("code_bon", "")
        print(f"{row['sweep_value']:>8.4f}  {tool or '-':>8}  {mem or '-':>8}  {code or '-':>8}")


if __name__ == "__main__":
    main()
