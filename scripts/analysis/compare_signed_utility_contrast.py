#!/usr/bin/env python3
"""Compare positive vs negative signed-utility probe rows.

The input files are ``signed_utility_rows.jsonl`` written by
``scripts/attention_pauh/probe_signed_utility.py --write-row-details``.

For each same-prompt pair and OP-VEC residual entry, this script computes:

    contrast = signed_effect(positive trajectory) - signed_effect(negative trajectory)

Positive contrast means the residual is more aligned with the successful
trajectory than with the failed trajectory. This is the mechanism-level signal
needed for pass/fail Code attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROLE_SUFFIX_RE = re.compile(
    r"^(?P<pair_id>.+?)__(?:reference_pass|target_hurt|positive|negative|pass|fail)__(?:full|code|response|reasoning)$"
)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def main() -> None:
    args = parse_args()
    positive = load_probe_rows(Path(args.positive_rows), pair_id_regex=args.pair_id_regex)
    negative = load_probe_rows(Path(args.negative_rows), pair_id_regex=args.pair_id_regex)
    module_rows, pair_rows = compare_probe_maps(positive, negative, min_pair_count=args.min_pair_count)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    module_jsonl = output_dir / "contrast_module_summary.jsonl"
    pair_jsonl = output_dir / "contrast_pair_rows.jsonl"
    summary_json = output_dir / "contrast_summary.json"
    summary_md = output_dir / "contrast_summary.md"

    write_jsonl(module_jsonl, module_rows)
    write_jsonl(pair_jsonl, pair_rows)
    summary = build_summary(
        module_rows=module_rows,
        pair_rows=pair_rows,
        positive_rows=str(Path(args.positive_rows).expanduser()),
        negative_rows=str(Path(args.negative_rows).expanduser()),
        output_dir=output_dir,
        top_k=args.top_k,
    )
    summary["files"] = {
        "module_jsonl": str(module_jsonl),
        "pair_jsonl": str(pair_jsonl),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-rows", required=True, help="signed_utility_rows.jsonl for successful trajectories.")
    parser.add_argument("--negative-rows", required=True, help="signed_utility_rows.jsonl for failed trajectories.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--min-pair-count", type=int, default=1)
    parser.add_argument(
        "--pair-id-regex",
        default="",
        help="Optional regex with a named `pair_id` group. Defaults to the CURE hurt span-pair naming convention.",
    )
    return parser.parse_args()


def load_probe_rows(path: Path, *, pair_id_regex: str = "") -> dict[tuple[str, str, int, str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    compiled = re.compile(pair_id_regex) if pair_id_regex else None
    buckets: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            for item in payload.get("details", []):
                row_id = str(item.get("row_id") or payload.get("row_id") or "")
                param_name = str(item.get("param_name") or "")
                layer = safe_int(item.get("layer"), default=layer_from_param(param_name))
                expert = str(item.get("expert") or "")
                if not row_id or not expert or not param_name:
                    continue
                pair_id = extract_pair_id(row_id, compiled)
                key = (pair_id, expert, layer, param_name)
                buckets[key]["signed_sum"] += safe_float(item.get("signed_effect"))
                buckets[key]["expression_sum"] += safe_float(item.get("expression"))
                buckets[key]["count"] += 1.0

    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for key, stats in buckets.items():
        count = max(stats["count"], 1.0)
        result[key] = {
            "pair_id": key[0],
            "expert": key[1],
            "layer": key[2],
            "param_name": key[3],
            "module": module_from_param(key[3]),
            "module_family": module_family(module_from_param(key[3])),
            "signed_effect": stats["signed_sum"] / count,
            "expression": stats["expression_sum"] / count,
            "source_count": int(stats["count"]),
        }
    return result


def extract_pair_id(row_id: str, compiled: re.Pattern[str] | None) -> str:
    if compiled is not None:
        match = compiled.search(row_id)
        if match:
            return str(match.group("pair_id"))
    match = ROLE_SUFFIX_RE.search(row_id)
    if match:
        return str(match.group("pair_id"))
    parts = row_id.rsplit("__", 2)
    if len(parts) == 3 and parts[-2] and parts[-1]:
        return parts[0]
    return row_id


def compare_probe_maps(
    positive: dict[tuple[str, str, int, str], dict[str, Any]],
    negative: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    min_pair_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common_keys = sorted(set(positive) & set(negative))
    pair_rows: list[dict[str, Any]] = []
    module_stats: dict[tuple[str, int, str], dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    for key in common_keys:
        pos = positive[key]
        neg = negative[key]
        contrast = float(pos["signed_effect"]) - float(neg["signed_effect"])
        pair_row = {
            "pair_id": key[0],
            "expert": key[1],
            "layer": key[2],
            "param_name": key[3],
            "module": pos["module"],
            "module_family": pos["module_family"],
            "positive_signed_effect": float(pos["signed_effect"]),
            "negative_signed_effect": float(neg["signed_effect"]),
            "contrast_signed_effect": contrast,
            "positive_expression": float(pos["expression"]),
            "negative_expression": float(neg["expression"]),
            "expression_delta": float(pos["expression"]) - float(neg["expression"]),
        }
        pair_rows.append(pair_row)
        mkey = (key[1], key[2], key[3])
        stats = module_stats[mkey]
        stats["expert"] = key[1]
        stats["layer"] = key[2]
        stats["param_name"] = key[3]
        stats["module"] = pos["module"]
        stats["module_family"] = pos["module_family"]
        stats["positive_signed_sum"] += pair_row["positive_signed_effect"]
        stats["negative_signed_sum"] += pair_row["negative_signed_effect"]
        stats["contrast_sum"] += contrast
        stats["contrast_abs_sum"] += abs(contrast)
        stats["positive_expression_sum"] += pair_row["positive_expression"]
        stats["negative_expression_sum"] += pair_row["negative_expression"]
        stats["contrast_positive_count"] += 1.0 if contrast > 0.0 else 0.0
        stats["positive_effect_count"] += 1.0 if pair_row["positive_signed_effect"] > 0.0 else 0.0
        stats["negative_effect_count"] += 1.0 if pair_row["negative_signed_effect"] > 0.0 else 0.0
        stats["pair_count"] += 1.0

    module_rows: list[dict[str, Any]] = []
    for stats in module_stats.values():
        pair_count = int(stats["pair_count"])
        if pair_count < min_pair_count:
            continue
        denom = float(pair_count)
        module_rows.append(
            {
                "expert": stats["expert"],
                "layer": int(stats["layer"]),
                "param_name": stats["param_name"],
                "module": stats["module"],
                "module_family": stats["module_family"],
                "pair_count": pair_count,
                "positive_signed_effect_mean": stats["positive_signed_sum"] / denom,
                "negative_signed_effect_mean": stats["negative_signed_sum"] / denom,
                "contrast_signed_effect_mean": stats["contrast_sum"] / denom,
                "contrast_abs_mean": stats["contrast_abs_sum"] / denom,
                "contrast_positive_fraction": stats["contrast_positive_count"] / denom,
                "positive_effect_fraction": stats["positive_effect_count"] / denom,
                "negative_effect_fraction": stats["negative_effect_count"] / denom,
                "positive_expression_mean": stats["positive_expression_sum"] / denom,
                "negative_expression_mean": stats["negative_expression_sum"] / denom,
            }
        )
    module_rows.sort(
        key=lambda row: (
            row["contrast_signed_effect_mean"],
            row["contrast_positive_fraction"],
            row["positive_expression_mean"],
        ),
        reverse=True,
    )
    pair_rows.sort(key=lambda row: row["contrast_signed_effect"], reverse=True)
    return module_rows, pair_rows


def build_summary(
    *,
    module_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    positive_rows: str,
    negative_rows: str,
    output_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    by_expert: dict[str, dict[str, Any]] = {}
    for expert in sorted({str(row["expert"]) for row in module_rows}):
        rows = [row for row in module_rows if row["expert"] == expert]
        by_expert[expert] = {
            "module_count": len(rows),
            "mean_contrast": mean(row["contrast_signed_effect_mean"] for row in rows),
            "mean_positive_effect": mean(row["positive_signed_effect_mean"] for row in rows),
            "mean_negative_effect": mean(row["negative_signed_effect_mean"] for row in rows),
            "top_positive_modules": rows[:top_k],
            "top_negative_modules": sorted(rows, key=lambda row: row["contrast_signed_effect_mean"])[:top_k],
        }
    layer_module = aggregate_layer_module(module_rows)
    return {
        "format": "signed_utility_contrast_summary_v1",
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "output_dir": str(output_dir),
        "module_count": len(module_rows),
        "pair_row_count": len(pair_rows),
        "pair_count": len({row["pair_id"] for row in pair_rows}),
        "by_expert": by_expert,
        "layer_module_top": layer_module[:top_k],
        "interpretation": {
            "contrast_signed_effect_mean": "mean signed_effect(pass trajectory) - signed_effect(fail trajectory) for the same prompt pair and residual entry",
            "positive_value": "residual is more aligned with successful behavior than failed behavior",
            "negative_value": "residual is more aligned with failed behavior or does not distinguish success",
        },
    }


def aggregate_layer_module(module_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str, str], dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    for row in module_rows:
        key = (int(row["layer"]), str(row["module"]), str(row["module_family"]))
        stats = buckets[key]
        stats["layer"] = key[0]
        stats["module"] = key[1]
        stats["module_family"] = key[2]
        stats["contrast_sum"] += float(row["contrast_signed_effect_mean"])
        stats["abs_sum"] += abs(float(row["contrast_signed_effect_mean"]))
        stats["count"] += 1.0
    result = []
    for stats in buckets.values():
        count = max(float(stats["count"]), 1.0)
        result.append(
            {
                "layer": int(stats["layer"]),
                "module": stats["module"],
                "module_family": stats["module_family"],
                "expert_count": int(stats["count"]),
                "mean_contrast": stats["contrast_sum"] / count,
                "mean_abs_contrast": stats["abs_sum"] / count,
            }
        )
    return sorted(result, key=lambda row: row["mean_abs_contrast"], reverse=True)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Signed Utility Pass/Fail Contrast",
        "",
        f"- Positive rows: `{summary['positive_rows']}`",
        f"- Negative rows: `{summary['negative_rows']}`",
        f"- Pair count: `{summary['pair_count']}`",
        f"- Module entries: `{summary['module_count']}`",
        "",
        "## Expert Summary",
        "",
        "| expert | modules | mean contrast | mean pass utility | mean fail utility |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for expert, payload in summary["by_expert"].items():
        lines.append(
            "| {expert} | {module_count} | {mean_contrast:.6g} | {mean_positive_effect:.6g} | {mean_negative_effect:.6g} |".format(
                expert=expert,
                **payload,
            )
        )
    lines.extend(["", "## Top Positive Contrast Modules", ""])
    for expert, payload in summary["by_expert"].items():
        lines.extend([f"### `{expert}`", "", "| layer | module | contrast | pass utility | fail utility | positive frac |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
        for row in payload["top_positive_modules"][:10]:
            lines.append(
                "| {layer} | `{module}` | {contrast_signed_effect_mean:.6g} | {positive_signed_effect_mean:.6g} | {negative_signed_effect_mean:.6g} | {contrast_positive_fraction:.3f} |".format(
                    **row
                )
            )
        lines.append("")
    lines.extend(["## Top Layer/Module Contrast Magnitude", "", "| layer | module | mean contrast | mean abs contrast |", "| ---: | --- | ---: | ---: |"])
    for row in summary["layer_module_top"][:24]:
        lines.append(
            "| {layer} | `{module}` | {mean_contrast:.6g} | {mean_abs_contrast:.6g} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def layer_from_param(param_name: str) -> int:
    match = LAYER_RE.search(str(param_name))
    return int(match.group(1)) if match else -1


def module_from_param(param_name: str) -> str:
    text = str(param_name)
    if ".self_attn.q_proj.weight" in text:
        return "q"
    if ".self_attn.k_proj.weight" in text:
        return "k"
    if ".self_attn.v_proj.weight" in text:
        return "v"
    if ".self_attn.o_proj.weight" in text:
        return "o"
    if ".mlp.gate_proj.weight" in text:
        return "gate"
    if ".mlp.up_proj.weight" in text:
        return "up"
    if ".mlp.down_proj.weight" in text:
        return "down"
    return "other"


def module_family(module: str) -> str:
    if module in {"q", "k", "v", "o"}:
        return module
    if module in {"gate", "up", "down"}:
        return "mlp"
    return "other"


if __name__ == "__main__":
    main()
