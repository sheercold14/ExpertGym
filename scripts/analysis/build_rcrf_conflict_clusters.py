#!/usr/bin/env python3
"""Summarize RCF-BC residual rows into interpretable conflict clusters.

This script turns row-level operating-point diagnostics into paper-facing
mechanism clusters.  It is deliberately descriptive: it does not generate new
gates or tune any coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_DASHBOARD_DATA = ROOT / "analysis" / "rcrf_diagnostic_dashboard_20260522" / "dashboard_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_conflict_clusters_20260522"


ARCHETYPE_ORDER = (
    "clean_code_repair",
    "code_repair_with_behavior_harm",
    "code_source_conflict",
    "code_negative_with_behavior_support",
    "code_negative_noise",
    "behavior_only",
    "weak_or_uninformative",
)


def main() -> None:
    args = parse_args()
    dashboard = load_json(Path(args.dashboard_data).expanduser())
    rows = list(dashboard.get("rows") or [])
    if not rows:
        raise ValueError(f"No rows found in {args.dashboard_data}")
    primary = args.primary_candidate
    if primary not in dashboard.get("candidates", []):
        raise ValueError(f"--primary-candidate must be in {dashboard.get('candidates')}, got {primary}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched = [enrich_row(row, primary=primary) for row in rows]
    archetype_summary = aggregate(enriched, ["archetype"], primary=primary)
    role_expert_summary = aggregate(enriched, ["role", "expert"], primary=primary)
    layer_module_expert_summary = aggregate(enriched, ["layer_band", "module_family", "expert"], primary=primary)
    source_pattern_summary = aggregate(
        enriched,
        ["archetype", "code_positive_sources", "code_negative_sources", "protected_support_tasks", "protected_harm_tasks"],
        primary=primary,
    )
    top_rows = top_conflict_rows(enriched, primary=primary, top_k=args.top_k)
    summary = {
        "format": "rcrf_conflict_cluster_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dashboard_data": str(Path(args.dashboard_data).expanduser().resolve()),
        "primary_candidate": primary,
        "row_count": len(enriched),
        "archetype_summary": archetype_summary,
        "role_expert_summary": role_expert_summary,
        "layer_module_expert_summary": layer_module_expert_summary,
        "top_rows": top_rows,
        "interpretation": interpret(archetype_summary, primary=primary),
    }
    write_json(output_dir / "conflict_cluster_summary.json", summary)
    write_jsonl(output_dir / "conflict_cluster_rows.jsonl", enriched)
    write_csv(output_dir / "archetype_summary.csv", archetype_summary)
    write_csv(output_dir / "role_expert_summary.csv", role_expert_summary)
    write_csv(output_dir / "layer_module_expert_summary.csv", layer_module_expert_summary)
    write_csv(output_dir / "source_pattern_summary.csv", source_pattern_summary)
    (output_dir / "conflict_cluster_report.md").write_text(render_markdown(summary, output_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "report": str(output_dir / "conflict_cluster_report.md"),
                "row_count": len(enriched),
                "primary_candidate": primary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-data", type=Path, default=DEFAULT_DASHBOARD_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-candidate", default="v18_rcf_bc")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def enrich_row(row: dict[str, Any], *, primary: str) -> dict[str, Any]:
    enriched = dict(row)
    support_tasks = split_tasks(row.get("protected_support_tasks"))
    harm_tasks = split_tasks(row.get("protected_harm_tasks"))
    positive_sources = split_tasks(row.get("code_positive_sources"))
    negative_sources = split_tasks(row.get("code_negative_sources"))
    code_pos = safe_float(row.get("code_positive_strength"))
    code_neg = safe_float(row.get("code_negative_strength"))
    harm = safe_float(row.get("protected_max_harm_norm"))
    support = 1.0 if support_tasks else 0.0
    delta = safe_float(row.get(f"{primary}_delta"))
    enriched["archetype"] = classify_archetype(
        role=str(row.get("role") or ""),
        code_pos=code_pos,
        code_neg=code_neg,
        positive_sources=positive_sources,
        negative_sources=negative_sources,
        support_tasks=support_tasks,
        harm_tasks=harm_tasks,
    )
    enriched["primary_delta"] = delta
    enriched["primary_abs_delta"] = abs(delta)
    enriched["primary_changed"] = abs(delta) > 1e-12
    enriched["primary_direction"] = sign(delta)
    enriched["code_signal_strength"] = max(code_pos, code_neg)
    enriched["code_source_conflict_count"] = min(len(positive_sources), len(negative_sources))
    enriched["behavior_support_count"] = len(support_tasks)
    enriched["behavior_harm_count"] = len(harm_tasks)
    enriched["behavior_conflict_score"] = min(max(code_pos, code_neg), harm) if harm_tasks else 0.0
    enriched["capability_preserve_tension"] = max(code_neg, 0.0) * support
    enriched["paper_use"] = paper_use_label(enriched)
    return enriched


def classify_archetype(
    *,
    role: str,
    code_pos: float,
    code_neg: float,
    positive_sources: set[str],
    negative_sources: set[str],
    support_tasks: set[str],
    harm_tasks: set[str],
) -> str:
    if positive_sources and negative_sources:
        return "code_source_conflict"
    if code_pos > 0.0 and harm_tasks:
        return "code_repair_with_behavior_harm"
    if code_pos > 0.0:
        return "clean_code_repair"
    if code_neg > 0.0 and support_tasks:
        return "code_negative_with_behavior_support"
    if code_neg > 0.0:
        return "code_negative_noise"
    if support_tasks or harm_tasks or role.startswith("protected"):
        return "behavior_only"
    return "weak_or_uninformative"


def paper_use_label(row: dict[str, Any]) -> str:
    archetype = str(row.get("archetype") or "")
    changed = bool(row.get("primary_changed"))
    if archetype == "clean_code_repair" and changed:
        return "positive evidence that capability residuals are sparse but distributed"
    if archetype == "code_repair_with_behavior_harm" and changed:
        return "central conflict case: capability repair under behavior constraint"
    if archetype == "code_source_conflict" and changed:
        return "evidence for continuous field over hard role routing"
    if archetype == "code_negative_with_behavior_support":
        return "preservation case: do not suppress behavior-supporting residuals blindly"
    if archetype == "code_negative_noise" and changed:
        return "negative evidence: remove residuals aligned with failing trajectories"
    if archetype == "behavior_only":
        return "behavior constraint case"
    return "audit/background"


def aggregate(rows: list[dict[str, Any]], group_keys: list[str], *, primary: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group, group_rows in sorted(grouped.items(), key=group_sort_key(group_keys)):
        item = {key: value for key, value in zip(group_keys, group)}
        deltas = [safe_float(row.get("primary_delta")) for row in group_rows]
        changed = [value for value in deltas if abs(value) > 1e-12]
        item.update(
            {
                "row_count": len(group_rows),
                "changed": len(changed),
                "positive": sum(1 for value in changed if value > 0.0),
                "negative": sum(1 for value in changed if value < 0.0),
                "mean_delta": mean(deltas) if deltas else 0.0,
                "mean_abs_delta": mean(abs(value) for value in deltas) if deltas else 0.0,
                "mean_code_positive_strength": mean(safe_float(row.get("code_positive_strength")) for row in group_rows),
                "mean_code_negative_strength": mean(safe_float(row.get("code_negative_strength")) for row in group_rows),
                "mean_behavior_harm": mean(safe_float(row.get("protected_max_harm_norm")) for row in group_rows),
                "tool_rows": sum(1 for row in group_rows if row.get("expert") == "tool"),
                "memory_rows": sum(1 for row in group_rows if row.get("expert") == "memory"),
                "code_rows": sum(1 for row in group_rows if row.get("expert") == "code"),
            }
        )
        output.append(item)
    return output


def group_sort_key(group_keys: list[str]):
    def _key(item: tuple[Any, ...]) -> tuple[Any, ...]:
        values = dict(zip(group_keys, item))
        if "archetype" in values:
            order = ARCHETYPE_ORDER.index(str(values["archetype"])) if values["archetype"] in ARCHETYPE_ORDER else 999
            return (order, *tuple(str(x) for x in item))
        return tuple(str(x) for x in item)

    return _key


def top_conflict_rows(rows: list[dict[str, Any]], *, primary: str, top_k: int) -> dict[str, list[dict[str, Any]]]:
    fields = [
        "key",
        "archetype",
        "paper_use",
        "role",
        "expert",
        "layer",
        "module",
        "primary_delta",
        "code_positive_sources",
        "code_negative_sources",
        "code_positive_strength",
        "code_negative_strength",
        "protected_support_tasks",
        "protected_harm_tasks",
        "protected_max_harm_norm",
        "behavior_conflict_score",
        "capability_preserve_tension",
    ]

    def pack(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{field: row.get(field) for field in fields} for row in items[:top_k]]

    return {
        "largest_primary_delta": pack(sorted(rows, key=lambda row: safe_float(row.get("primary_abs_delta")), reverse=True)),
        "strongest_behavior_conflict": pack(
            sorted(rows, key=lambda row: safe_float(row.get("behavior_conflict_score")), reverse=True)
        ),
        "strongest_capability_preserve_tension": pack(
            sorted(rows, key=lambda row: safe_float(row.get("capability_preserve_tension")), reverse=True)
        ),
        "changed_source_conflicts": pack(
            sorted(
                [row for row in rows if row.get("archetype") == "code_source_conflict" and row.get("primary_changed")],
                key=lambda row: safe_float(row.get("primary_abs_delta")),
                reverse=True,
            )
        ),
    }


def interpret(archetype_summary: list[dict[str, Any]], *, primary: str) -> list[str]:
    by_name = {str(row.get("archetype")): row for row in archetype_summary}
    lines = []
    source = by_name.get("code_source_conflict", {})
    if source:
        lines.append(
            f"`code_source_conflict` has {source.get('row_count', 0)} rows and "
            f"{source.get('changed', 0)} changed rows under `{primary}`; this is the main evidence "
            "against hard role routing."
        )
    repair_harm = by_name.get("code_repair_with_behavior_harm", {})
    if repair_harm:
        lines.append(
            f"`code_repair_with_behavior_harm` has {repair_harm.get('row_count', 0)} rows; these rows define "
            "the core Pareto boundary between Code repair and Tool/Memory behavior preservation."
        )
    neg_support = by_name.get("code_negative_with_behavior_support", {})
    if neg_support:
        lines.append(
            f"`code_negative_with_behavior_support` has {neg_support.get('row_count', 0)} rows; these rows explain why "
            "blindly suppressing negative Code evidence can hurt behavior tasks."
        )
    noise = by_name.get("code_negative_noise", {})
    if noise:
        lines.append(
            f"`code_negative_noise` has {noise.get('row_count', 0)} rows and is the cleanest suppression target."
        )
    return lines


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# RCF-BC Residual Conflict Clusters",
        "",
        "## Metadata",
        "",
        f"- Primary candidate: `{summary['primary_candidate']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Dashboard data: `{summary['dashboard_data']}`",
        f"- Output dir: `{output_dir}`",
        "",
        "## Archetype Summary",
        "",
        "| archetype | rows | changed | + | - | mean abs delta | mean code+ | mean code- | mean harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["archetype_summary"]:
        lines.append(
            f"| `{row.get('archetype')}` | {row.get('row_count')} | {row.get('changed')} | "
            f"{row.get('positive')} | {row.get('negative')} | {safe_float(row.get('mean_abs_delta')):.6f} | "
            f"{safe_float(row.get('mean_code_positive_strength')):.4f} | "
            f"{safe_float(row.get('mean_code_negative_strength')):.4f} | "
            f"{safe_float(row.get('mean_behavior_harm')):.4f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in summary["interpretation"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Top Conflict Rows",
            "",
            "### Strongest Behavior Conflict",
            "",
            "| key | role | expert | layer | module | delta | code+ | code- | support | harm | conflict |",
            "|---|---|---|---:|---|---:|---|---|---|---|---:|",
        ]
    )
    for row in summary["top_rows"]["strongest_behavior_conflict"][:15]:
        lines.append(render_top_row(row))
    lines.extend(
        [
            "",
            "### Changed Source Conflicts",
            "",
            "| key | role | expert | layer | module | delta | code+ | code- | support | harm | conflict |",
            "|---|---|---|---:|---|---:|---|---|---|---|---:|",
        ]
    )
    for row in summary["top_rows"]["changed_source_conflicts"][:15]:
        lines.append(render_top_row(row))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON summary: `{output_dir / 'conflict_cluster_summary.json'}`",
            f"- Row JSONL: `{output_dir / 'conflict_cluster_rows.jsonl'}`",
            f"- Archetype CSV: `{output_dir / 'archetype_summary.csv'}`",
            f"- Role/expert CSV: `{output_dir / 'role_expert_summary.csv'}`",
            f"- Layer/module/expert CSV: `{output_dir / 'layer_module_expert_summary.csv'}`",
            f"- Source-pattern CSV: `{output_dir / 'source_pattern_summary.csv'}`",
            "",
            "## Method Implication",
            "",
            "The next method change should operate on archetypes, not global task coefficients. "
            "The cleanest hierarchy is: suppress `code_negative_noise`; preserve `code_negative_with_behavior_support`; "
            "allow continuous small deltas for `code_source_conflict`; and treat `code_repair_with_behavior_harm` as "
            "the Pareto boundary where Tool/Memory behavior evidence constrains Code repair.",
            "",
        ]
    )
    return "\n".join(lines)


def render_top_row(row: dict[str, Any]) -> str:
    return (
        f"| `{row.get('key')}` | `{row.get('role')}` | {row.get('expert')} | {row.get('layer')} | "
        f"{row.get('module')} | {safe_float(row.get('primary_delta')):.6f} | "
        f"{row.get('code_positive_sources') or '-'} | {row.get('code_negative_sources') or '-'} | "
        f"{row.get('protected_support_tasks') or '-'} | {row.get('protected_harm_tasks') or '-'} | "
        f"{safe_float(row.get('behavior_conflict_score')):.4f} |"
    )


def split_tasks(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def sign(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "zero"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
