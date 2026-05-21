#!/usr/bin/env python3
"""Build a residual-level utility/harm/conflict evidence table.

This script is intentionally diagnostic.  It does not train, bake, or evaluate
models.  It aligns existing mechanism artifacts by `(param_name, expert)`:

* Code pass/fail contrast from multiple source/span probes.
* Task behavior signed-utility summaries for Tool/Memory/Code.
* Optional gate deltas from already-built variants.

The output is a single auditable table that can drive later conservative
residual routing rules without relying on scalar sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "residual_evidence_table_20260521"
DEFAULT_SIGNED_UTILITY = Path(
    "/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521/"
    "signed_utility_summary.json"
)

DEFAULT_CODE_SOURCES = [
    ("LB_code", ROOT / "contrast" / "livebench_code_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LB_prompt", ROOT / "contrast" / "livebench_prompt_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LB_reasoning", ROOT / "contrast" / "livebench_reasoning_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LCB_code", ROOT / "contrast" / "livecodebench_code_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LCB_prompt", ROOT / "contrast" / "livecodebench_prompt_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
]

DEFAULT_GATES = [
    ("v1_code_only", ROOT / "contrast_gates" / "rcrf_code_contrast_v1" / "gates.json"),
    ("v2_spanaware", ROOT / "contrast_gates" / "rcrf_code_spanaware_conservative_v2" / "gates.json"),
    ("v3_memory_hard_floor", ROOT / "contrast_gates" / "rcrf_code_spanaware_memory_preserve_v3" / "gates.json"),
    ("v4_memory_utility_floor", ROOT / "contrast_gates" / "rcrf_code_spanaware_memory_utility_preserve_v4" / "gates.json"),
]

TASKS = ("tool", "memory", "code")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    code_sources = args.code_source or DEFAULT_CODE_SOURCES
    gate_sources = args.gate or DEFAULT_GATES

    code_rows, source_stats = load_code_contrast_sources(code_sources)
    signed_utility_summaries = args.signed_utility_summary or [DEFAULT_SIGNED_UTILITY]
    utility_rows = load_signed_utility(signed_utility_summaries)
    gate_rows = load_gate_sources(gate_sources)

    evidence_rows = build_evidence_rows(
        code_rows=code_rows,
        source_stats=source_stats,
        utility_rows=utility_rows,
        gate_rows=gate_rows,
        informative_normalized_effect=args.informative_normalized_effect,
        utility_positive_fraction=args.utility_positive_fraction,
    )
    evidence_rows.sort(key=lambda row: (-float(row["conflict_priority"]), row["expert"], row["param_name"]))

    write_outputs(output_dir, evidence_rows, source_stats, code_sources, gate_sources, signed_utility_summaries)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "row_count": len(evidence_rows),
                "summary": str(output_dir / "residual_evidence_summary.json"),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--signed-utility-summary",
        type=Path,
        action="append",
        default=[],
        help="signed_utility_summary.json. Can be repeated; stats are count-weighted merged. Defaults to the small 20260521 signature.",
    )
    parser.add_argument(
        "--code-source",
        nargs=2,
        action="append",
        metavar=("NAME", "CONTRAST_MODULE_SUMMARY_JSONL"),
        help="Code contrast source. Defaults to the 2026-05-21 LB/LCB code-hurt probes.",
    )
    parser.add_argument(
        "--gate",
        nargs=2,
        action="append",
        metavar=("NAME", "GATES_JSON"),
        help="Optional gate checkpoint to align deltas. Defaults to v1-v4 RCRF variants.",
    )
    parser.add_argument(
        "--informative-normalized-effect",
        type=float,
        default=0.25,
        help=(
            "Minimum abs(effect / source_mean_abs) for a Code source to count as informative. "
            "Raw signs are still recorded separately."
        ),
    )
    parser.add_argument(
        "--utility-positive-fraction",
        type=float,
        default=0.5,
        help="Minimum positive_fraction for treating a task utility row as positive.",
    )
    return parser.parse_args()


def load_code_contrast_sources(
    sources: list[tuple[str, Path]],
) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], dict[str, dict[str, float]]]:
    rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    source_stats: dict[str, dict[str, float]] = {}
    for name, raw_path in sources:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        source_values: list[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["param_name"]), str(row["expert"]))
                value = safe_float(row.get("contrast_signed_effect_mean"))
                source_values.append(value)
                rows[key][name] = {
                    "effect": value,
                    "abs_effect": abs(value),
                    "pair_count": safe_float(row.get("pair_count")),
                    "positive_fraction": safe_float(row.get("contrast_positive_fraction")),
                    "positive_expression": safe_float(row.get("positive_expression_mean")),
                    "negative_expression": safe_float(row.get("negative_expression_mean")),
                    "module": row.get("module", ""),
                    "module_family": row.get("module_family", ""),
                    "layer": safe_int(row.get("layer"), default=layer_from_param(str(row["param_name"]))),
                }
        source_stats[name] = summarize_source_values(source_values, str(path))
    return rows, source_stats


def summarize_source_values(values: list[float], path: str) -> dict[str, float | str]:
    abs_values = [abs(value) for value in values]
    mean_abs = mean(abs_values) if abs_values else 0.0
    return {
        "path": path,
        "count": float(len(values)),
        "mean_abs": mean_abs,
        "max_abs": max(abs_values) if abs_values else 0.0,
        "positive_count": float(sum(1 for value in values if value > 0.0)),
        "negative_count": float(sum(1 for value in values if value < 0.0)),
    }


def load_signed_utility(paths: list[Path]) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    accum: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        module_summary = payload.get("module_summary", {})
        for task in TASKS:
            task_rows = module_summary.get(task, {})
            if not isinstance(task_rows, dict):
                continue
            for param_name, expert_map in task_rows.items():
                if not isinstance(expert_map, dict):
                    continue
                for expert, stats in expert_map.items():
                    if not isinstance(stats, dict):
                        continue
                    count = max(safe_float(stats.get("count")), 0.0)
                    if count <= 0.0:
                        continue
                    bucket = accum[(str(param_name), str(expert))][task]
                    bucket["count"] += count
                    for key in ("expression_mean", "harm_mean", "positive_fraction", "signed_effect_mean"):
                        bucket[f"{key}_weighted_sum"] += safe_float(stats.get(key)) * count
    result: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for residual_key, task_map in accum.items():
        for task, stats in task_map.items():
            count = max(stats.get("count", 0.0), 1.0)
            result[residual_key][task] = {
                "count": stats.get("count", 0.0),
                "expression_mean": stats.get("expression_mean_weighted_sum", 0.0) / count,
                "harm_mean": stats.get("harm_mean_weighted_sum", 0.0) / count,
                "positive_fraction": stats.get("positive_fraction_weighted_sum", 0.0) / count,
                "signed_effect_mean": stats.get("signed_effect_mean_weighted_sum", 0.0) / count,
            }
    return result


def load_gate_sources(sources: list[tuple[str, Path]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    result: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for name, raw_path in sources:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("decision_rows", []):
            key = (str(row.get("param_name", "")), str(row.get("expert", "")))
            if not key[0] or not key[1]:
                continue
            result[key][name] = {
                "coefficient": safe_float(row.get("coefficient")),
                "base_coefficient": safe_float(row.get("base_coefficient")),
                "delta": safe_float(row.get("delta")),
                "reason": str(row.get("reason", "")),
            }
    return result


def build_evidence_rows(
    *,
    code_rows: dict[tuple[str, str], dict[str, dict[str, Any]]],
    source_stats: dict[str, dict[str, float]],
    utility_rows: dict[tuple[str, str], dict[str, dict[str, float]]],
    gate_rows: dict[tuple[str, str], dict[str, dict[str, Any]]],
    informative_normalized_effect: float,
    utility_positive_fraction: float,
) -> list[dict[str, Any]]:
    keys = sorted(set(code_rows) | set(utility_rows) | set(gate_rows))
    evidence_rows: list[dict[str, Any]] = []
    for param_name, expert in keys:
        code = summarize_code_evidence(
            code_rows.get((param_name, expert), {}),
            source_stats,
            informative_normalized_effect=informative_normalized_effect,
        )
        utility = utility_rows.get((param_name, expert), {})
        utility_states = {
            task: classify_utility_state(utility.get(task), positive_fraction_threshold=utility_positive_fraction)
            for task in TASKS
        }
        recommendation, reason = recommend_action(code, utility_states)
        row: dict[str, Any] = {
            "param_name": param_name,
            "expert": expert,
            "layer": layer_from_param(param_name),
            "module": module_from_param(param_name),
            "module_family": module_family(module_from_param(param_name)),
            **code,
            "utility_states": json.dumps(utility_states, sort_keys=True, ensure_ascii=False),
            "positive_utility_tasks": ",".join(task for task, state in utility_states.items() if state == "positive"),
            "harm_utility_tasks": ",".join(task for task, state in utility_states.items() if state == "harm"),
            "recommendation": recommendation,
            "recommendation_reason": reason,
        }
        for task in TASKS:
            stats = utility.get(task, {})
            state = utility_states[task]
            row[f"{task}_utility_state"] = state
            row[f"{task}_utility_signed_effect"] = stats.get("signed_effect_mean", 0.0)
            row[f"{task}_utility_harm"] = stats.get("harm_mean", 0.0)
            row[f"{task}_utility_positive_fraction"] = stats.get("positive_fraction", 0.0)
            row[f"{task}_utility_expression"] = stats.get("expression_mean", 0.0)
            row[f"{task}_utility_count"] = stats.get("count", 0.0)
        for gate_name, gate in sorted(gate_rows.get((param_name, expert), {}).items()):
            row[f"{gate_name}_delta"] = gate["delta"]
            row[f"{gate_name}_coefficient"] = gate["coefficient"]
            row[f"{gate_name}_reason"] = gate["reason"]
        row["conflict_priority"] = compute_conflict_priority(row)
        evidence_rows.append(row)
    return evidence_rows


def summarize_code_evidence(
    rows: dict[str, dict[str, Any]],
    source_stats: dict[str, dict[str, float]],
    *,
    informative_normalized_effect: float,
) -> dict[str, Any]:
    raw_effects: dict[str, float] = {}
    norm_effects: dict[str, float] = {}
    raw_positive_sources: list[str] = []
    raw_negative_sources: list[str] = []
    positive_sources: list[str] = []
    negative_sources: list[str] = []
    weak_sources: list[str] = []
    source_positive_fractions: dict[str, float] = {}
    for source, row in rows.items():
        effect = safe_float(row.get("effect"))
        scale = float(source_stats[source].get("mean_abs", 0.0)) or 1.0
        normalized = effect / scale
        raw_effects[source] = effect
        norm_effects[source] = normalized
        source_positive_fractions[source] = safe_float(row.get("positive_fraction"))
        if normalized > 0.0:
            raw_positive_sources.append(source)
        elif normalized < 0.0:
            raw_negative_sources.append(source)
        if abs(normalized) < informative_normalized_effect:
            weak_sources.append(source)
        elif normalized > 0.0:
            positive_sources.append(source)
        elif normalized < 0.0:
            negative_sources.append(source)
    n = len(raw_effects)
    pair_count = n * (n - 1) / 2
    source_conflict_pairs = len(positive_sources) * len(negative_sources)
    if positive_sources and negative_sources:
        code_state = "source_conflict"
    elif positive_sources:
        code_state = "positive"
    elif negative_sources:
        code_state = "negative"
    else:
        code_state = "no_signal"
    return {
        "code_source_count": n,
        "code_informative_source_count": len(positive_sources) + len(negative_sources),
        "code_weak_source_count": len(weak_sources),
        "code_raw_positive_source_count": len(raw_positive_sources),
        "code_raw_negative_source_count": len(raw_negative_sources),
        "code_positive_source_count": len(positive_sources),
        "code_negative_source_count": len(negative_sources),
        "code_source_conflict_pairs": source_conflict_pairs,
        "code_source_conflict_rate": source_conflict_pairs / pair_count if pair_count else 0.0,
        "code_mean_raw_effect": mean(raw_effects.values()) if raw_effects else 0.0,
        "code_mean_abs_raw_effect": mean(abs(value) for value in raw_effects.values()) if raw_effects else 0.0,
        "code_mean_normalized_effect": mean(norm_effects.values()) if norm_effects else 0.0,
        "code_mean_abs_normalized_effect": mean(abs(value) for value in norm_effects.values()) if norm_effects else 0.0,
        "code_state": code_state,
        "code_positive_sources": ",".join(sorted(positive_sources)),
        "code_negative_sources": ",".join(sorted(negative_sources)),
        "code_weak_sources": ",".join(sorted(weak_sources)),
        "code_raw_positive_sources": ",".join(sorted(raw_positive_sources)),
        "code_raw_negative_sources": ",".join(sorted(raw_negative_sources)),
        "code_raw_effects": json.dumps(raw_effects, sort_keys=True),
        "code_normalized_effects": json.dumps(norm_effects, sort_keys=True),
        "code_source_positive_fractions": json.dumps(source_positive_fractions, sort_keys=True),
    }


def classify_utility_state(stats: dict[str, float] | None, *, positive_fraction_threshold: float) -> str:
    if not stats:
        return "missing"
    signed = safe_float(stats.get("signed_effect_mean"))
    harm = safe_float(stats.get("harm_mean"))
    positive_fraction = safe_float(stats.get("positive_fraction"))
    if signed > 0.0 and positive_fraction >= positive_fraction_threshold:
        return "positive"
    if signed < 0.0 or harm > abs(signed):
        return "harm"
    return "weak"


def recommend_action(code: dict[str, Any], utility_states: dict[str, str]) -> tuple[str, str]:
    positive_tasks = [task for task, state in utility_states.items() if state == "positive"]
    harm_tasks = [task for task, state in utility_states.items() if state == "harm"]
    code_state = str(code["code_state"])
    if code_state == "source_conflict":
        return "hold_conflict", "code source/span signs disagree"
    if code_state == "positive":
        if harm_tasks:
            return "hold_conflict", f"code positive but utility harm for {','.join(harm_tasks)}"
        return "keep_or_raise", "code positive and no observed task harm"
    if code_state == "negative":
        if positive_tasks:
            return "hold_conflict", f"code negative but utility positive for {','.join(positive_tasks)}"
        return "suppress", "code negative and no observed positive task utility"
    if positive_tasks:
        return "preserve", f"no code signal but utility positive for {','.join(positive_tasks)}"
    return "no_decision", "no strong aligned evidence"


def compute_conflict_priority(row: dict[str, Any]) -> float:
    code_conflict = safe_float(row.get("code_source_conflict_rate"))
    code_abs = safe_float(row.get("code_mean_abs_normalized_effect"))
    task_utility = 0.0
    for task in TASKS:
        task_utility += abs(safe_float(row.get(f"{task}_utility_signed_effect")))
        task_utility += safe_float(row.get(f"{task}_utility_harm"))
    return code_abs * (1.0 + code_conflict) + 100.0 * task_utility


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    source_stats: dict[str, dict[str, float]],
    code_sources: list[tuple[str, Path]],
    gate_sources: list[tuple[str, Path]],
    signed_utility_summaries: list[Path],
) -> None:
    csv_path = output_dir / "residual_evidence_rows.csv"
    jsonl_path = output_dir / "residual_evidence_rows.jsonl"
    summary_json_path = output_dir / "residual_evidence_summary.json"
    summary_md_path = output_dir / "residual_evidence_summary.md"

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = build_summary(rows, source_stats, code_sources, gate_sources, signed_utility_summaries)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md_path.write_text(render_markdown_summary(summary), encoding="utf-8")


def build_summary(
    rows: list[dict[str, Any]],
    source_stats: dict[str, dict[str, float]],
    code_sources: list[tuple[str, Path]],
    gate_sources: list[tuple[str, Path]],
    signed_utility_summaries: list[Path],
) -> dict[str, Any]:
    recommendation_counts = Counter(str(row["recommendation"]) for row in rows)
    code_state_counts = Counter(str(row["code_state"]) for row in rows)
    expert_recommendations: dict[str, dict[str, int]] = {}
    for expert in sorted({str(row["expert"]) for row in rows}):
        expert_rows = [row for row in rows if row["expert"] == expert]
        expert_recommendations[expert] = dict(Counter(str(row["recommendation"]) for row in expert_rows))
    top_conflicts = [
        {
            "param_name": row["param_name"],
            "expert": row["expert"],
            "layer": row["layer"],
            "module_family": row["module_family"],
            "recommendation": row["recommendation"],
            "code_state": row["code_state"],
            "code_positive_sources": row["code_positive_sources"],
            "code_negative_sources": row["code_negative_sources"],
            "conflict_priority": row["conflict_priority"],
            "positive_utility_tasks": row["positive_utility_tasks"],
            "harm_utility_tasks": row["harm_utility_tasks"],
        }
        for row in rows[:30]
    ]
    return {
        "format": "residual_evidence_table_v1",
        "row_count": len(rows),
        "code_sources": {name: str(Path(path).expanduser()) for name, path in code_sources},
        "gate_sources": {name: str(Path(path).expanduser()) for name, path in gate_sources},
        "signed_utility_summaries": [str(Path(path).expanduser()) for path in signed_utility_summaries],
        "source_stats": source_stats,
        "recommendation_counts": dict(recommendation_counts),
        "code_state_counts": dict(code_state_counts),
        "expert_recommendations": expert_recommendations,
        "top_conflict_priority_rows": top_conflicts,
        "interpretation": {
            "keep_or_raise": "Code source/span evidence is consistently positive and no task utility harm is observed.",
            "suppress": "Code source/span evidence is consistently negative and no task utility preservation signal is observed.",
            "preserve": "No Code contrast signal, but at least one task has positive behavior utility.",
            "hold_conflict": "Evidence conflicts across Code sources/spans or with task utility; do not make a strong scalar decision.",
            "no_decision": "Current probes do not provide enough evidence.",
        },
    }


def render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Residual Evidence Table Summary",
        "",
        "## Inputs",
        "",
        "- signed utility summaries:",
        "",
    ]
    for path in summary["signed_utility_summaries"]:
        lines.append(f"  - `{path}`")
    lines.extend(
        [
            "",
        "### Code sources",
        "",
        ]
    )
    for name, path in sorted(summary["code_sources"].items()):
        stats = summary["source_stats"].get(name, {})
        lines.append(
            f"- `{name}`: `{path}`; mean_abs={safe_float(stats.get('mean_abs')):.3e}, "
            f"pos={int(safe_float(stats.get('positive_count')))}, neg={int(safe_float(stats.get('negative_count')))}"
        )
    lines.extend(["", "### Gate sources", ""])
    for name, path in sorted(summary["gate_sources"].items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            "### Recommendation counts",
            "",
            "| recommendation | count |",
            "|---|---:|",
        ]
    )
    for name, count in sorted(summary["recommendation_counts"].items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "### Code state counts", "", "| code_state | count |", "|---|---:|"])
    for name, count in sorted(summary["code_state_counts"].items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "### Top conflict-priority rows", ""])
    lines.extend(
        [
            "| rank | expert | layer | family | recommendation | code state | code + | code - | utility + | utility harm |",
            "|---:|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for idx, row in enumerate(summary["top_conflict_priority_rows"][:20], start=1):
        lines.append(
            f"| {idx} | {row['expert']} | {row['layer']} | {row['module_family']} | "
            f"{row['recommendation']} | {row['code_state']} | {row['code_positive_sources']} | "
            f"{row['code_negative_sources']} | {row['positive_utility_tasks']} | {row['harm_utility_tasks']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `hold_conflict` rows are the most important analysis targets: they are where scalar task-vector scaling is likely unstable.",
            "- `keep_or_raise` / `suppress` rows are candidates for conservative residual routing, not proof of final gate values.",
            "- Utility columns come from the current small signed-utility signature and should be strengthened with task-specific behavior-span probes before becoming a final method.",
            "",
        ]
    )
    return "\n".join(lines)


def layer_from_param(param_name: str) -> int:
    match = LAYER_RE.search(param_name)
    return int(match.group(1)) if match else -1


def module_from_param(param_name: str) -> str:
    if ".self_attn.q_proj." in param_name:
        return "q"
    if ".self_attn.k_proj." in param_name:
        return "k"
    if ".self_attn.v_proj." in param_name:
        return "v"
    if ".self_attn.o_proj." in param_name:
        return "o"
    if ".mlp.gate_proj." in param_name:
        return "gate"
    if ".mlp.up_proj." in param_name:
        return "up"
    if ".mlp.down_proj." in param_name:
        return "down"
    return "unknown"


def module_family(module: str) -> str:
    if module in {"q", "k", "v", "o"}:
        return "attention"
    if module in {"gate", "up", "down"}:
        return "mlp"
    return "unknown"


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
