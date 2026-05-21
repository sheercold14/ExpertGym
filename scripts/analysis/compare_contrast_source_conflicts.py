#!/usr/bin/env python3
"""Compare residual contrast signs across datasets/spans.

Inputs are `contrast_module_summary.jsonl` files produced by
`compare_signed_utility_contrast.py`.  The script aligns rows by
`(param_name, expert)`, then reports where two sources agree or conflict.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: load_module_summary(path) for name, path in args.source}
    pairs = []
    for left, right in combinations(sorted(sources), 2):
        pairs.append(compare_pair(left, sources[left], right, sources[right], top_k=args.top_k))
    payload = {
        "format": "contrast_source_conflict_summary_v1",
        "sources": {name: str(Path(path).expanduser().resolve()) for name, path in args.source},
        "pairs": pairs,
    }
    write_json(output_dir / "source_conflicts.json", payload)
    (output_dir / "source_conflicts.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "source_conflicts.json"), "pair_count": len(pairs)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("NAME", "CONTRAST_MODULE_SUMMARY_JSONL"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def load_module_summary(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["param_name"]), str(row["expert"]))
            rows[key] = row
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def compare_pair(
    left_name: str,
    left_rows: dict[tuple[str, str], dict[str, Any]],
    right_name: str,
    right_rows: dict[tuple[str, str], dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    keys = sorted(set(left_rows) & set(right_rows))
    if not keys:
        raise ValueError(f"No overlapping residual entries for {left_name} and {right_name}")
    xs = [float(left_rows[key]["contrast_signed_effect_mean"]) for key in keys]
    ys = [float(right_rows[key]["contrast_signed_effect_mean"]) for key in keys]
    conflicts = []
    agreements = []
    for key, x, y in zip(keys, xs, ys):
        item = {
            "param_name": key[0],
            "expert": key[1],
            "left_contrast": x,
            "right_contrast": y,
            "combined_abs": abs(x) + abs(y),
            "left_positive_fraction": float(left_rows[key].get("contrast_positive_fraction", 0.0)),
            "right_positive_fraction": float(right_rows[key].get("contrast_positive_fraction", 0.0)),
        }
        if x * y < 0.0:
            conflicts.append(item)
        elif x * y > 0.0:
            agreements.append(item)
    return {
        "left": left_name,
        "right": right_name,
        "overlap_count": len(keys),
        "pearson": pearson(xs, ys),
        "conflict_count": len(conflicts),
        "agreement_count": len(agreements),
        "zero_count": len(keys) - len(conflicts) - len(agreements),
        "mean_abs_left": mean(abs(x) for x in xs),
        "mean_abs_right": mean(abs(y) for y in ys),
        "top_conflicts": sorted(conflicts, key=lambda row: row["combined_abs"], reverse=True)[:top_k],
        "top_agreements": sorted(agreements, key=lambda row: row["combined_abs"], reverse=True)[:top_k],
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Contrast Source Conflicts",
        "",
        "## Sources",
        "",
    ]
    for name, path in sorted(payload["sources"].items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Pair Summary",
            "",
            "| left | right | overlap | pearson | conflicts | agreements | mean abs left | mean abs right |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in payload["pairs"]:
        lines.append(
            f"| {pair['left']} | {pair['right']} | {pair['overlap_count']} | {pair['pearson']:.4f} | "
            f"{pair['conflict_count']} | {pair['agreement_count']} | {pair['mean_abs_left']:.3e} | {pair['mean_abs_right']:.3e} |"
        )
    for pair in payload["pairs"]:
        lines.extend(["", f"## Top Conflicts: `{pair['left']}` vs `{pair['right']}`", ""])
        lines.extend(["| rank | expert | param | left | right | combined abs |", "|---:|---|---|---:|---:|---:|"])
        for idx, row in enumerate(pair["top_conflicts"], start=1):
            lines.append(
                f"| {idx} | {row['expert']} | `{row['param_name']}` | {row['left_contrast']:.3e} | "
                f"{row['right_contrast']:.3e} | {row['combined_abs']:.3e} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
