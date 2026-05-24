#!/usr/bin/env python3
"""Build an RCRF residual conflict atlas.

The atlas aligns three kinds of already-computed mechanism evidence by
``(param_name, expert)``:

* Code same-prompt pass/fail contrast.
* Tool/Memory behavior utility and harm probes.
* Optional gate decision rows from candidate operating points.

It is a lightweight diagnostic script.  It does not train models, bake
checkpoints, or run evaluation.  Its purpose is to make the residual conflict
structure inspectable at the layer/module/expert level.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_conflict_atlas_20260522"

DEFAULT_CODE_CONTRASTS = [
    ("LB_prompt", ROOT / "contrast" / "livebench_prompt_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LB_reasoning", ROOT / "contrast" / "livebench_reasoning_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LCB_code", ROOT / "contrast" / "livecodebench_code_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
    ("LCB_prompt", ROOT / "contrast" / "livecodebench_prompt_alllayers_s16_20260521" / "contrast_module_summary.jsonl"),
]
DEFAULT_BEHAVIOR_SUMMARIES = [
    ROOT / "probes" / "tool_memory_positive_signature_s32_20260521" / "signed_utility_summary.json",
    ROOT / "probes" / "memory_fulltraj_positive_s32_20260521" / "signed_utility_summary.json",
]
DEFAULT_CANDIDATE_DECISIONS = [
    ("v8_hard", ROOT / "contrast_gates" / "rcrf_code_spanaware_tmpos_s32_memoryfull_v8" / "decision_rows.jsonl"),
    ("v9_soft", ROOT / "contrast_gates" / "rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9" / "decision_rows.jsonl"),
    ("v10_ratio", ROOT / "contrast_gates" / "rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10" / "decision_rows.jsonl"),
    ("v11_tasktyped", ROOT / "contrast_gates" / "rcrf_code_spanaware_tmpos_s32_tasktyped_v11" / "decision_rows.jsonl"),
]

TASKS = ("tool", "memory", "code")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    code_sources = args.code_contrast or DEFAULT_CODE_CONTRASTS
    behavior_summaries = args.behavior_summary or DEFAULT_BEHAVIOR_SUMMARIES
    decision_sources = args.decision_rows or DEFAULT_CANDIDATE_DECISIONS

    code_rows, code_source_stats = load_code_contrasts(code_sources)
    behavior_rows = load_behavior_summaries(behavior_summaries)
    normalizers = build_behavior_normalizers(behavior_rows)
    decision_rows = load_decision_rows(decision_sources)
    protected_tasks = tuple(args.protected_task or ["tool", "memory"])
    rows = build_atlas_rows(
        code_rows=code_rows,
        code_source_stats=code_source_stats,
        behavior_rows=behavior_rows,
        normalizers=normalizers,
        decision_rows=decision_rows,
        code_threshold=args.code_normalized_threshold,
        harm_threshold=args.protected_harm_threshold,
        utility_threshold=args.protected_utility_threshold,
        utility_positive_fraction=args.utility_positive_fraction,
        protected_tasks=protected_tasks,
    )
    rows.sort(
        key=lambda row: (
            -safe_float(row["conflict_priority"]),
            str(row["role"]),
            str(row["expert"]),
            int(row["layer"]),
            str(row["module"]),
        )
    )

    summary = build_summary(
        rows=rows,
        code_sources=code_sources,
        code_source_stats=code_source_stats,
        behavior_summaries=behavior_summaries,
        decision_sources=decision_sources,
        normalizers=normalizers,
        protected_tasks=protected_tasks,
        top_k=args.top_k,
    )
    write_outputs(output_dir, rows, summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "row_count": len(rows),
                "summary": str(output_dir / "residual_conflict_atlas_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--code-contrast",
        nargs=2,
        action="append",
        metavar=("NAME", "CONTRAST_MODULE_SUMMARY_JSONL"),
        help="Code pass/fail contrast source. Can be repeated.",
    )
    parser.add_argument(
        "--behavior-summary",
        type=Path,
        action="append",
        default=[],
        help="signed_utility_summary.json for protected behavior evidence. Can be repeated.",
    )
    parser.add_argument(
        "--decision-rows",
        nargs=2,
        action="append",
        metavar=("NAME", "DECISION_ROWS_JSONL"),
        help="Optional gate decision_rows.jsonl. Can be repeated.",
    )
    parser.add_argument("--protected-task", action="append", default=[])
    parser.add_argument("--code-normalized-threshold", type=float, default=0.25)
    parser.add_argument("--protected-harm-threshold", type=float, default=0.4)
    parser.add_argument("--protected-utility-threshold", type=float, default=0.4)
    parser.add_argument("--utility-positive-fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def load_code_contrasts(
    sources: list[tuple[str, Path]],
) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    source_stats: dict[str, dict[str, Any]] = {}
    for name, raw_path in sources:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        values: list[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                param_name = str(row.get("param_name") or "")
                expert = str(row.get("expert") or "")
                if not param_name or not expert:
                    continue
                effect = safe_float(row.get("contrast_signed_effect_mean"))
                values.append(effect)
                rows[(param_name, expert)][name] = {
                    "effect": effect,
                    "pair_count": safe_float(row.get("pair_count")),
                    "positive_fraction": safe_float(row.get("contrast_positive_fraction")),
                    "positive_expression": safe_float(row.get("positive_expression_mean")),
                    "negative_expression": safe_float(row.get("negative_expression_mean")),
                    "layer": safe_int(row.get("layer"), layer_from_param(param_name)),
                    "module": str(row.get("module") or module_from_param(param_name)),
                    "module_family": str(row.get("module_family") or module_family(module_from_param(param_name))),
                }
        abs_values = [abs(value) for value in values]
        source_stats[name] = {
            "path": str(path),
            "count": len(values),
            "mean_abs": mean(abs_values) if abs_values else 0.0,
            "max_abs": max(abs_values) if abs_values else 0.0,
            "positive_count": sum(1 for value in values if value > 0.0),
            "negative_count": sum(1 for value in values if value < 0.0),
        }
    return rows, source_stats


def load_behavior_summaries(paths: list[Path]) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
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
                    count = safe_float(stats.get("count"))
                    if count <= 0.0:
                        continue
                    bucket = accum[(str(param_name), str(expert))][task]
                    bucket["count"] += count
                    for field in ("expression_mean", "harm_mean", "positive_fraction", "signed_effect_mean"):
                        bucket[f"{field}_weighted_sum"] += safe_float(stats.get(field)) * count

    result: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for key, task_map in accum.items():
        for task, stats in task_map.items():
            count = max(stats.get("count", 0.0), 1.0)
            result[key][task] = {
                "count": stats.get("count", 0.0),
                "expression_mean": stats.get("expression_mean_weighted_sum", 0.0) / count,
                "harm_mean": stats.get("harm_mean_weighted_sum", 0.0) / count,
                "positive_fraction": stats.get("positive_fraction_weighted_sum", 0.0) / count,
                "signed_effect_mean": stats.get("signed_effect_mean_weighted_sum", 0.0) / count,
            }
    return result


def build_behavior_normalizers(
    behavior_rows: dict[tuple[str, str], dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for task_map in behavior_rows.values():
        for task, stats in task_map.items():
            signed = safe_float(stats.get("signed_effect_mean"))
            harm = max(safe_float(stats.get("harm_mean")), -signed, 0.0)
            utility = max(signed, 0.0)
            values[task]["harm"].append(harm)
            values[task]["utility"].append(utility)
    normalizers: dict[str, dict[str, float]] = {}
    for task in TASKS:
        harm_values = [value for value in values[task]["harm"] if value > 0.0]
        utility_values = [value for value in values[task]["utility"] if value > 0.0]
        normalizers[task] = {
            "harm_mean_positive": mean(harm_values) if harm_values else 1.0,
            "utility_mean_positive": mean(utility_values) if utility_values else 1.0,
            "harm_positive_count": len(harm_values),
            "utility_positive_count": len(utility_values),
        }
    return normalizers


def load_decision_rows(sources: list[tuple[str, Path]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    result: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for name, raw_path in sources:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                param_name = str(row.get("param_name") or "")
                expert = str(row.get("expert") or "")
                if not param_name or not expert:
                    continue
                result[(param_name, expert)][name] = {
                    "delta": safe_float(row.get("delta")),
                    "coefficient": safe_float(row.get("coefficient")),
                    "base_coefficient": safe_float(row.get("base_coefficient")),
                    "reason": str(row.get("reason") or ""),
                }
    return result


def build_atlas_rows(
    *,
    code_rows: dict[tuple[str, str], dict[str, dict[str, Any]]],
    code_source_stats: dict[str, dict[str, Any]],
    behavior_rows: dict[tuple[str, str], dict[str, dict[str, float]]],
    normalizers: dict[str, dict[str, float]],
    decision_rows: dict[tuple[str, str], dict[str, dict[str, Any]]],
    code_threshold: float,
    harm_threshold: float,
    utility_threshold: float,
    utility_positive_fraction: float,
    protected_tasks: tuple[str, ...],
) -> list[dict[str, Any]]:
    keys = sorted(set(code_rows) | set(behavior_rows) | set(decision_rows))
    rows: list[dict[str, Any]] = []
    for param_name, expert in keys:
        code = summarize_code_signal(
            code_rows.get((param_name, expert), {}),
            code_source_stats,
            threshold=code_threshold,
        )
        behavior = summarize_behavior_signal(
            behavior_rows.get((param_name, expert), {}),
            normalizers,
            protected_tasks=protected_tasks,
            harm_threshold=harm_threshold,
            utility_threshold=utility_threshold,
            utility_positive_fraction=utility_positive_fraction,
        )
        role = classify_role(code, behavior)
        layer = layer_from_param(param_name)
        row: dict[str, Any] = {
            "param_name": param_name,
            "expert": expert,
            "layer": layer,
            "layer_band": layer_band(layer),
            "module": module_from_param(param_name),
            "module_family": module_family(module_from_param(param_name)),
            "role": role,
            "conflict_priority": compute_conflict_priority(code, behavior),
            **code,
            **behavior,
        }
        for candidate, decision in sorted(decision_rows.get((param_name, expert), {}).items()):
            row[f"{candidate}_delta"] = decision["delta"]
            row[f"{candidate}_coefficient"] = decision["coefficient"]
            row[f"{candidate}_reason"] = decision["reason"]
        rows.append(row)
    return rows


def summarize_code_signal(
    rows: dict[str, dict[str, Any]],
    source_stats: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    normalized: dict[str, float] = {}
    positive_sources: list[str] = []
    negative_sources: list[str] = []
    weak_sources: list[str] = []
    raw_effects: dict[str, float] = {}
    for source, row in rows.items():
        raw = safe_float(row.get("effect"))
        scale = safe_float(source_stats.get(source, {}).get("mean_abs")) or 1.0
        value = raw / scale
        raw_effects[source] = raw
        normalized[source] = value
        if abs(value) < threshold:
            weak_sources.append(source)
        elif value > 0.0:
            positive_sources.append(source)
        else:
            negative_sources.append(source)
    positive_strength = mean([normalized[src] for src in positive_sources]) if positive_sources else 0.0
    negative_strength = mean([-normalized[src] for src in negative_sources]) if negative_sources else 0.0
    if positive_sources and negative_sources:
        state = "source_conflict"
    elif positive_sources:
        state = "positive"
    elif negative_sources:
        state = "negative"
    else:
        state = "no_signal"
    return {
        "code_state": state,
        "code_source_count": len(rows),
        "code_positive_source_count": len(positive_sources),
        "code_negative_source_count": len(negative_sources),
        "code_weak_source_count": len(weak_sources),
        "code_positive_strength": positive_strength,
        "code_negative_strength": negative_strength,
        "code_mean_normalized_effect": mean(normalized.values()) if normalized else 0.0,
        "code_mean_abs_normalized_effect": mean(abs(value) for value in normalized.values()) if normalized else 0.0,
        "code_positive_sources": ",".join(sorted(positive_sources)),
        "code_negative_sources": ",".join(sorted(negative_sources)),
        "code_weak_sources": ",".join(sorted(weak_sources)),
        "code_raw_effects": json.dumps(raw_effects, sort_keys=True),
        "code_normalized_effects": json.dumps(normalized, sort_keys=True),
    }


def summarize_behavior_signal(
    task_rows: dict[str, dict[str, float]],
    normalizers: dict[str, dict[str, float]],
    *,
    protected_tasks: tuple[str, ...],
    harm_threshold: float,
    utility_threshold: float,
    utility_positive_fraction: float,
) -> dict[str, Any]:
    protected_harm_tasks: list[str] = []
    protected_support_tasks: list[str] = []
    max_harm = 0.0
    max_utility = 0.0
    result: dict[str, Any] = {}
    for task in TASKS:
        stats = task_rows.get(task, {})
        signed = safe_float(stats.get("signed_effect_mean"))
        raw_harm = max(safe_float(stats.get("harm_mean")), -signed, 0.0)
        raw_utility = max(signed, 0.0)
        harm_norm = raw_harm / (safe_float(normalizers.get(task, {}).get("harm_mean_positive")) or 1.0)
        utility_norm = raw_utility / (safe_float(normalizers.get(task, {}).get("utility_mean_positive")) or 1.0)
        positive_fraction = safe_float(stats.get("positive_fraction"))
        result[f"{task}_signed_effect"] = signed
        result[f"{task}_harm_raw"] = raw_harm
        result[f"{task}_harm_norm"] = harm_norm
        result[f"{task}_utility_raw"] = raw_utility
        result[f"{task}_utility_norm"] = utility_norm
        result[f"{task}_positive_fraction"] = positive_fraction
        result[f"{task}_expression"] = safe_float(stats.get("expression_mean"))
        result[f"{task}_count"] = safe_float(stats.get("count"))
        if task in protected_tasks:
            max_harm = max(max_harm, harm_norm)
            max_utility = max(max_utility, utility_norm)
            if harm_norm >= harm_threshold:
                protected_harm_tasks.append(task)
            if utility_norm >= utility_threshold and positive_fraction >= utility_positive_fraction:
                protected_support_tasks.append(task)
    result["protected_harm_tasks"] = ",".join(protected_harm_tasks)
    result["protected_support_tasks"] = ",".join(protected_support_tasks)
    result["protected_harm_count"] = len(protected_harm_tasks)
    result["protected_support_count"] = len(protected_support_tasks)
    result["protected_max_harm_norm"] = max_harm
    result["protected_max_utility_norm"] = max_utility
    return result


def classify_role(code: dict[str, Any], behavior: dict[str, Any]) -> str:
    code_state = str(code["code_state"])
    has_harm = safe_int(behavior.get("protected_harm_count"), 0) > 0
    has_support = safe_int(behavior.get("protected_support_count"), 0) > 0
    if code_state == "source_conflict":
        if has_harm or has_support:
            return "code_source_conflict_with_behavior"
        return "code_source_conflict"
    if code_state == "positive":
        if has_harm and has_support:
            return "code_repair_shared_and_harm"
        if has_harm:
            return "code_repair_vs_protected_harm"
        if has_support:
            return "shared_positive"
        return "code_repair_only"
    if code_state == "negative":
        if has_support:
            return "code_negative_but_protected_support"
        return "code_negative_noise"
    if has_harm and has_support:
        return "protected_mixed_behavior"
    if has_harm:
        return "protected_harm_only"
    if has_support:
        return "protected_support_only"
    return "uninformative"


def compute_conflict_priority(code: dict[str, Any], behavior: dict[str, Any]) -> float:
    code_strength = max(safe_float(code.get("code_positive_strength")), safe_float(code.get("code_negative_strength")))
    behavior_strength = safe_float(behavior.get("protected_max_harm_norm")) + safe_float(
        behavior.get("protected_max_utility_norm")
    )
    source_conflict_bonus = 1.0 if code.get("code_state") == "source_conflict" else 0.0
    return code_strength * (1.0 + behavior_strength + source_conflict_bonus)


def build_summary(
    *,
    rows: list[dict[str, Any]],
    code_sources: list[tuple[str, Path]],
    code_source_stats: dict[str, dict[str, Any]],
    behavior_summaries: list[Path],
    decision_sources: list[tuple[str, Path]],
    normalizers: dict[str, dict[str, float]],
    protected_tasks: tuple[str, ...],
    top_k: int,
) -> dict[str, Any]:
    candidate_names = [name for name, _ in decision_sources]
    return {
        "format": "rcrf_residual_conflict_atlas_v1",
        "row_count": len(rows),
        "protected_tasks": list(protected_tasks),
        "code_sources": {name: str(Path(path).expanduser()) for name, path in code_sources},
        "code_source_stats": code_source_stats,
        "behavior_summaries": [str(Path(path).expanduser()) for path in behavior_summaries],
        "decision_sources": {name: str(Path(path).expanduser()) for name, path in decision_sources},
        "behavior_normalizers": normalizers,
        "role_counts": dict(Counter(str(row["role"]) for row in rows)),
        "expert_role_counts": nested_counts(rows, "expert", "role"),
        "module_role_counts": nested_counts(rows, "module_family", "role"),
        "layer_band_role_counts": nested_counts(rows, "layer_band", "role"),
        "candidate_delta_by_role": candidate_delta_by_group(rows, "role", candidate_names),
        "candidate_delta_by_expert": candidate_delta_by_group(rows, "expert", candidate_names),
        "top_conflict_rows": slim_rows(rows[:top_k], candidate_names),
        "interpretation": {
            "code_repair_vs_protected_harm": "Residual supports Code pass/fail repair but is harmful to Tool/Memory behavior spans.",
            "shared_positive": "Residual supports Code repair and protected behavior; this is a likely synergy residual.",
            "code_negative_but_protected_support": "Residual looks harmful for Code pass/fail but useful for Tool/Memory; suppressing it can damage protected behavior.",
            "protected_harm_only": "No Code signal, but protected behavior probe marks it harmful; useful for pruning or veto rules.",
            "protected_support_only": "No Code signal, but protected behavior probe marks it useful; useful for preservation floors.",
        },
    }


def nested_counts(rows: list[dict[str, Any]], outer_key: str, inner_key: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[str(row.get(outer_key, ""))][str(row.get(inner_key, ""))] += 1
    return {outer: dict(counter) for outer, counter in sorted(result.items())}


def candidate_delta_by_group(
    rows: list[dict[str, Any]], group_key: str, candidate_names: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    groups = sorted({str(row.get(group_key, "")) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if str(row.get(group_key, "")) == group]
        result[group] = {}
        for candidate in candidate_names:
            deltas = [safe_float(row.get(f"{candidate}_delta")) for row in group_rows if f"{candidate}_delta" in row]
            changed = [delta for delta in deltas if abs(delta) > 1e-12]
            result[group][candidate] = {
                "row_count": len(deltas),
                "changed_count": len(changed),
                "positive_count": sum(1 for delta in changed if delta > 0.0),
                "negative_count": sum(1 for delta in changed if delta < 0.0),
                "mean_delta": mean(deltas) if deltas else 0.0,
                "mean_abs_delta": mean(abs(delta) for delta in deltas) if deltas else 0.0,
            }
    return result


def slim_rows(rows: list[dict[str, Any]], candidate_names: list[str]) -> list[dict[str, Any]]:
    keys = [
        "param_name",
        "expert",
        "layer",
        "module_family",
        "role",
        "conflict_priority",
        "code_state",
        "code_positive_sources",
        "code_negative_sources",
        "code_positive_strength",
        "code_negative_strength",
        "protected_harm_tasks",
        "protected_support_tasks",
        "protected_max_harm_norm",
        "protected_max_utility_norm",
    ]
    result: list[dict[str, Any]] = []
    for row in rows:
        slim = {key: row.get(key) for key in keys}
        for candidate in candidate_names:
            if f"{candidate}_delta" in row:
                slim[f"{candidate}_delta"] = row[f"{candidate}_delta"]
                slim[f"{candidate}_reason"] = row.get(f"{candidate}_reason", "")
        result.append(slim)
    return result


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    jsonl_path = output_dir / "residual_conflict_atlas_rows.jsonl"
    csv_path = output_dir / "residual_conflict_atlas_rows.csv"
    summary_json = output_dir / "residual_conflict_atlas_summary.json"
    summary_md = output_dir / "residual_conflict_atlas_summary.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RCRF Residual Conflict Atlas",
        "",
        "## Inputs",
        "",
        "### Code pass/fail contrasts",
        "",
    ]
    for name, path in sorted(summary["code_sources"].items()):
        stats = summary["code_source_stats"].get(name, {})
        lines.append(
            f"- `{name}`: `{path}`; mean_abs={safe_float(stats.get('mean_abs')):.3e}, "
            f"pos={safe_int(stats.get('positive_count'), 0)}, neg={safe_int(stats.get('negative_count'), 0)}"
        )
    lines.extend(["", "### Behavior summaries", ""])
    for path in summary["behavior_summaries"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Role Counts", "", "| role | count |", "|---|---:|"])
    for role, count in sorted(summary["role_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {role} | {count} |")
    lines.extend(["", "## Expert x Role", ""])
    lines.extend(render_nested_count_table(summary["expert_role_counts"], "expert"))
    lines.extend(["", "## Module Family x Role", ""])
    lines.extend(render_nested_count_table(summary["module_role_counts"], "module_family"))
    lines.extend(["", "## Candidate Delta by Role", ""])
    lines.extend(render_candidate_delta_table(summary["candidate_delta_by_role"]))
    lines.extend(["", "## Top Conflict Rows", ""])
    lines.extend(
        [
            "| expert | layer | module | role | priority | code+ | code- | protected harm | protected support |",
            "|---|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["top_conflict_rows"][:20]:
        lines.append(
            f"| {row.get('expert')} | {row.get('layer')} | {row.get('module_family')} | {row.get('role')} | "
            f"{safe_float(row.get('conflict_priority')):.3f} | "
            f"{safe_float(row.get('code_positive_strength')):.3f} | "
            f"{safe_float(row.get('code_negative_strength')):.3f} | "
            f"{row.get('protected_harm_tasks') or '-'} | {row.get('protected_support_tasks') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `code_repair_vs_protected_harm`: Code repair signal and Tool/Memory behavior harm collide on the same residual.",
            "- `shared_positive`: same residual supports Code repair and protected behavior; these are synergy candidates.",
            "- `code_negative_but_protected_support`: suppressing this residual may help Code but damage Tool/Memory.",
            "- `protected_support_only`: preservation floor candidates.",
            "- `protected_harm_only`: pruning or veto candidates when no Code evidence exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_nested_count_table(data: dict[str, dict[str, int]], first_col: str) -> list[str]:
    roles = sorted({role for counts in data.values() for role in counts})
    lines = ["| " + " | ".join([first_col, *roles]) + " |", "|" + "---|" * (len(roles) + 1)]
    for outer, counts in sorted(data.items()):
        values = [str(counts.get(role, 0)) for role in roles]
        lines.append("| " + " | ".join([outer, *values]) + " |")
    return lines


def render_candidate_delta_table(data: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    candidate_names = sorted({candidate for groups in data.values() for candidate in groups})
    lines = ["| role | candidate | changed | + | - | mean_abs_delta |", "|---|---|---:|---:|---:|---:|"]
    for role, candidate_map in sorted(data.items()):
        for candidate in candidate_names:
            stats = candidate_map.get(candidate, {})
            lines.append(
                f"| {role} | {candidate} | {safe_int(stats.get('changed_count'), 0)} | "
                f"{safe_int(stats.get('positive_count'), 0)} | {safe_int(stats.get('negative_count'), 0)} | "
                f"{safe_float(stats.get('mean_abs_delta')):.6f} |"
            )
    return lines


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def layer_from_param(param_name: str) -> int:
    match = LAYER_RE.search(param_name)
    return int(match.group(1)) if match else -1


def layer_band(layer: int) -> str:
    if layer < 0:
        return "unknown"
    if layer <= 9:
        return "early_00_09"
    if layer <= 19:
        return "middle_10_19"
    return "late_20_27"


def module_from_param(param_name: str) -> str:
    if "q_proj" in param_name:
        return "q"
    if "k_proj" in param_name:
        return "k"
    if "v_proj" in param_name:
        return "v"
    if "o_proj" in param_name:
        return "o"
    if "gate_proj" in param_name:
        return "gate"
    if "up_proj" in param_name:
        return "up"
    if "down_proj" in param_name:
        return "down"
    return "unknown"


def module_family(module: str) -> str:
    if module in {"q", "k", "v", "o"}:
        return "attention"
    if module in {"gate", "up", "down"}:
        return "mlp"
    return module


if __name__ == "__main__":
    main()
