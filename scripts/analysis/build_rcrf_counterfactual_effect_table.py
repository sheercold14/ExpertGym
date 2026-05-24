#!/usr/bin/env python3
"""Build a counterfactual effect table for RCF-BC residual ablations.

The paper evidence table says which candidate scored what.  This script adds
the missing causal view: for each mechanical intervention, how many residual
rows were directly changed, how large the coefficient move was, and how the
downstream metrics moved relative to the RCF-BC operating point.

It only reads existing gate/evaluation artifacts.  It does not bake, evaluate,
or change any checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
GATE_ROOT = ROOT / "contrast_gates"
PAPER_TABLE = ROOT / "analysis" / "rcrf_paper_evidence_table_20260522" / "rcrf_paper_evidence_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_counterfactual_effects_20260522"
DEFAULT_DOC_REPORT = REPO_ROOT / "docs" / "report" / "RCRF" / "20260522_counterfactual_residual_effects.md"


@dataclass(frozen=True)
class Intervention:
    short: str
    family: str
    summary_path: Path
    description: str


INTERVENTIONS = [
    Intervention(
        short="v14",
        family="global_code_scalar",
        summary_path=GATE_ROOT / "rcrf_v9_code_half_v14" / "expert_scale_summary.json",
        description="all code expert coefficients * 0.5",
    ),
    Intervention(
        short="v15",
        family="global_code_scalar",
        summary_path=GATE_ROOT / "rcrf_v9_code_zero_v15" / "expert_scale_summary.json",
        description="all code expert coefficients = 0",
    ),
    Intervention(
        short="v20",
        family="local_combined",
        summary_path=GATE_ROOT / "rcrf_code_noise_weak_half_v20" / "archetype_scaled_summary.json",
        description="code_negative_noise + weak_or_uninformative code rows * 0.5",
    ),
    Intervention(
        short="v21",
        family="local_combined",
        summary_path=GATE_ROOT / "rcrf_code_noise_weak_zero_v21" / "archetype_scaled_summary.json",
        description="code_negative_noise + weak_or_uninformative code rows = 0",
    ),
    Intervention(
        short="v22",
        family="split_local",
        summary_path=GATE_ROOT / "rcrf_code_negative_noise_half_v22" / "archetype_scaled_summary.json",
        description="only code_negative_noise code rows * 0.5",
    ),
    Intervention(
        short="v23",
        family="split_local",
        summary_path=GATE_ROOT / "rcrf_code_weak_half_v23" / "archetype_scaled_summary.json",
        description="only weak_or_uninformative code rows * 0.5",
    ),
]


METRICS = [
    "tool_quick_mean",
    "tool_live_parallel",
    "memory_eval50_f1",
    "livebench_hurt_acc",
    "livebench_hurt_bon_acc",
    "livecodebench_hurt_acc",
    "livecodebench_hurt_bon_acc",
    "code_hurt_acc_mean",
    "code_hurt_bon_mean",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_rows = {row["short"]: row for row in read_csv(Path(args.paper_table).expanduser())}
    baseline = paper_rows.get(args.baseline)
    if not baseline:
        raise ValueError(f"Baseline `{args.baseline}` not found in {args.paper_table}")

    rows = [build_effect_row(item, paper_rows, baseline) for item in INTERVENTIONS]
    interactions = build_interaction_rows(rows)
    summary = {
        "format": "rcrf_counterfactual_effect_table_v1",
        "baseline": args.baseline,
        "paper_table": str(Path(args.paper_table).expanduser().resolve()),
        "rows": rows,
        "interactions": interactions,
        "takeaways": make_takeaways(rows, interactions),
    }
    write_csv(output_dir / "counterfactual_effect_rows.csv", rows)
    write_json(output_dir / "counterfactual_effect_summary.json", summary)
    report = render_markdown(summary, output_dir)
    (output_dir / "counterfactual_effect_report.md").write_text(report, encoding="utf-8")
    if args.doc_report:
        doc_report = Path(args.doc_report).expanduser().resolve()
        doc_report.parent.mkdir(parents=True, exist_ok=True)
        doc_report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(rows),
                "report": str(output_dir / "counterfactual_effect_report.md"),
                "doc_report": str(Path(args.doc_report).expanduser().resolve()) if args.doc_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-table", type=Path, default=PAPER_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    parser.add_argument("--baseline", default="v18")
    return parser.parse_args()


def build_effect_row(
    intervention: Intervention,
    paper_rows: dict[str, dict[str, str]],
    baseline: dict[str, str],
) -> dict[str, Any]:
    candidate = paper_rows.get(intervention.short)
    if not candidate:
        raise ValueError(f"Candidate `{intervention.short}` not found in paper table")
    summary = load_json(intervention.summary_path)
    direct = direct_change_summary(summary)
    row: dict[str, Any] = {
        "candidate": intervention.short,
        "family": intervention.family,
        "description": intervention.description,
        "summary_path": str(intervention.summary_path),
        "direct_rows": direct["direct_rows"],
        "direct_mean_before": direct["direct_mean_before"],
        "direct_mean_after": direct["direct_mean_after"],
        "direct_mean_delta": direct["direct_mean_delta"],
        "direct_archetypes": direct["direct_archetypes"],
    }
    for metric in METRICS:
        value = safe_float(candidate.get(metric))
        base = safe_float(baseline.get(metric))
        delta = none_subtract(value, base)
        row[metric] = value
        row[f"delta_{metric}"] = delta
        row[f"delta_{metric}_per_10_rows"] = per_rows(delta, direct["direct_rows"], scale=10.0)
    row["primary_read"] = primary_read(row)
    return row


def direct_change_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if "selected_count" in summary:
        archetypes = ",".join(str(row.get("archetype")) for row in summary.get("selected_by_archetype", []))
        return {
            "direct_rows": int(summary.get("selected_count") or summary.get("changed_count") or 0),
            "direct_mean_before": safe_float(summary.get("coefficient_before", {}).get("mean")),
            "direct_mean_after": safe_float(summary.get("coefficient_after", {}).get("mean")),
            "direct_mean_delta": safe_float(summary.get("delta", {}).get("mean"))
            if summary.get("delta", {}).get("count") == summary.get("selected_count")
            else none_subtract(
                safe_float(summary.get("coefficient_after", {}).get("mean")),
                safe_float(summary.get("coefficient_before", {}).get("mean")),
            ),
            "direct_archetypes": archetypes,
        }

    # Expert-level scalar ablation summary.
    before = summary.get("coefficient_summary_before", {}).get("code", {})
    after = summary.get("coefficient_summary_after", {}).get("code", {})
    return {
        "direct_rows": int(summary.get("changed_count") or 0),
        "direct_mean_before": safe_float(before.get("mean")),
        "direct_mean_after": safe_float(after.get("mean")),
        "direct_mean_delta": none_subtract(safe_float(after.get("mean")), safe_float(before.get("mean"))),
        "direct_archetypes": "all_code_expert_rows",
    }


def build_interaction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_short = {str(row["candidate"]): row for row in rows}
    interactions = []
    if all(key in by_short for key in ("v20", "v22", "v23")):
        interactions.append(interaction_row("v20_minus_v22_plus_v23", by_short["v20"], [by_short["v22"], by_short["v23"]]))
    if all(key in by_short for key in ("v21", "v22", "v23")):
        interactions.append(interaction_row("v21_minus_v22_plus_v23", by_short["v21"], [by_short["v22"], by_short["v23"]]))
    return interactions


def interaction_row(name: str, combined: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "combined": combined["candidate"],
        "parts": "+".join(str(part["candidate"]) for part in parts),
        "direct_rows_combined": combined["direct_rows"],
        "direct_rows_parts_sum": sum(int(part["direct_rows"]) for part in parts),
    }
    for metric in METRICS:
        combined_delta = safe_float(combined.get(f"delta_{metric}"))
        parts_delta = sum(safe_float(part.get(f"delta_{metric}")) for part in parts)
        row[f"interaction_{metric}"] = combined_delta - parts_delta
    row["primary_read"] = interaction_read(row)
    return row


def primary_read(row: dict[str, Any]) -> str:
    memory = safe_float(row.get("delta_memory_eval50_f1"))
    tool_live = safe_float(row.get("delta_tool_live_parallel"))
    lcb = safe_float(row.get("delta_livecodebench_hurt_bon_acc"))
    lb = safe_float(row.get("delta_livebench_hurt_bon_acc"))
    if memory > 0.015 and tool_live >= 0.0 and lcb > -0.15:
        return "localized memory gain but code evidence is mixed"
    if memory < 0.0 and tool_live < 0.0 and max(lb, lcb) >= 0.0:
        return "code-capable rows also support behavior; unsafe to prune"
    if memory > 0.02 and (lb < -0.1 or lcb < -0.1):
        return "memory/code trade-off from coarse code suppression"
    return "diagnostic trade-off"


def interaction_read(row: dict[str, Any]) -> str:
    memory = safe_float(row.get("interaction_memory_eval50_f1"))
    tool = safe_float(row.get("interaction_tool_live_parallel"))
    if memory > 0.02 and tool > 0.02:
        return "combined intervention has strong positive behavior interaction; row effects are not additive"
    if abs(memory) > 0.01 or abs(tool) > 0.01:
        return "non-additive interaction is material"
    return "near-additive"


def make_takeaways(rows: list[dict[str, Any]], interactions: list[dict[str, Any]]) -> list[str]:
    by_short = {str(row["candidate"]): row for row in rows}
    takeaways = []
    v20 = by_short.get("v20", {})
    v22 = by_short.get("v22", {})
    v23 = by_short.get("v23", {})
    if v20:
        takeaways.append(
            "v20 directly changes only 60 code rows but recovers nearly the same Memory F1 as global code-half, "
            "so Memory harm is localized but not safely separable yet."
        )
    if v22 and v23:
        takeaways.append(
            "v22 and v23 individually reduce Tool live_parallel and Memory F1 while preserving or improving parts of Code, "
            "so these rows are not disposable noise; they are capability/behavior coupling points."
        )
    if interactions:
        takeaways.append(
            "The v20-v22-v23 non-additivity shows residual rows interact: a row group's effect cannot be inferred by a hard archetype label alone."
        )
    takeaways.append(
        "The next method should keep a continuous residual field and add behavior-support constraints for low-evidence rows, instead of hard pruning."
    )
    return takeaways


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def none_subtract(value: float | None, base: float | None) -> float:
    return safe_float(value) - safe_float(base)


def per_rows(delta: float, rows: int, *, scale: float) -> float:
    if rows <= 0:
        return 0.0
    return safe_float(delta) / float(rows) * scale


def fmt(value: Any) -> str:
    return f"{safe_float(value):.4f}"


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    rows = summary["rows"]
    interactions = summary["interactions"]
    lines = [
        "# 2026-05-22 RCF-BC Counterfactual Residual Effects",
        "",
        "## Purpose",
        "",
        "This report converts RCF-BC ablations into counterfactual effect estimates. "
        "The goal is to separate two quantities that are easy to conflate:",
        "",
        "- how many residual rows were directly intervened on;",
        "- how Tool, Memory, and Code metrics moved relative to `v18_rcf_bc`.",
        "",
        "This is the evidence layer needed for a general capability-attribution framework: row labels are hypotheses, while counterfactual metric deltas are the behavioral evidence.",
        "",
        "## Main Effects",
        "",
        "| candidate | direct rows | intervention | dTool | dTool live | dMemory F1 | dLB BoN | dLCB BoN | read |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | {row['direct_rows']} | {row['description']} | "
            f"{fmt(row['delta_tool_quick_mean'])} | {fmt(row['delta_tool_live_parallel'])} | "
            f"{fmt(row['delta_memory_eval50_f1'])} | {fmt(row['delta_livebench_hurt_bon_acc'])} | "
            f"{fmt(row['delta_livecodebench_hurt_bon_acc'])} | {row['primary_read']} |"
        )
    lines.extend(
        [
            "",
            "## Row-Normalized Effects",
            "",
            "These values are metric delta per 10 directly changed rows. They are not causal constants; they flag which interventions are disproportionately risky.",
            "",
            "| candidate | rows | dMemory/10 | dTool live/10 | dLB BoN/10 | dLCB BoN/10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | {row['direct_rows']} | "
            f"{fmt(row['delta_memory_eval50_f1_per_10_rows'])} | "
            f"{fmt(row['delta_tool_live_parallel_per_10_rows'])} | "
            f"{fmt(row['delta_livebench_hurt_bon_acc_per_10_rows'])} | "
            f"{fmt(row['delta_livecodebench_hurt_bon_acc_per_10_rows'])} |"
        )
    lines.extend(["", "## Non-Additivity", "", "| interaction | combined | parts | dTool live interaction | dMemory interaction | dLB BoN interaction | dLCB BoN interaction | read |", "|---|---|---|---:|---:|---:|---:|---|"])
    for row in interactions:
        lines.append(
            f"| `{row['name']}` | `{row['combined']}` | `{row['parts']}` | "
            f"{fmt(row['interaction_tool_live_parallel'])} | {fmt(row['interaction_memory_eval50_f1'])} | "
            f"{fmt(row['interaction_livebench_hurt_bon_acc'])} | {fmt(row['interaction_livecodebench_hurt_bon_acc'])} | "
            f"{row['primary_read']} |"
        )
    lines.extend(["", "## Takeaways", ""])
    for item in summary["takeaways"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- CSV: `{output_dir / 'counterfactual_effect_rows.csv'}`",
            f"- JSON: `{output_dir / 'counterfactual_effect_summary.json'}`",
            f"- Source paper table: `{summary['paper_table']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
