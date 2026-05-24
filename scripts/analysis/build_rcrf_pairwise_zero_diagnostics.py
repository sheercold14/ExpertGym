#!/usr/bin/env python3
"""Build pairwise-zero diagnostics for the 8765/RCRF workbench.

This is an analysis-only script.  It does not bake checkpoints or run model
evaluation.  Starting from a reference residual gate, it constructs three
virtual pairwise views:

* tool + memory, code coefficients set to zero
* tool + code, memory coefficients set to zero
* memory + code, tool coefficients set to zero

The goal is to make the mechanism story easier to audit: each view isolates one
expert as the removed source and reports what residual roles, behavior supports,
and task conflicts are removed with it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_ATLAS = ROOT / "analysis" / "rcrf_conflict_atlas_20260522" / "residual_conflict_atlas_rows.csv"
DEFAULT_REFERENCE_GATE = (
    ROOT / "contrast_gates" / "residual_capability_field_behavior_constraints_v18" / "gates.json"
)
DEFAULT_EVIDENCE_TABLE = ROOT / "analysis" / "rcrf_paper_evidence_table_20260522" / "rcrf_paper_evidence_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "pairwise_zero_diagnostics_20260523"

EXPERTS = ("tool", "memory", "code")
TASKS = ("tool", "memory", "code")
PAIRS = {
    "tool_memory__code_zero": ("tool", "memory", "code"),
    "tool_code__memory_zero": ("tool", "code", "memory"),
    "memory_code__tool_zero": ("memory", "code", "tool"),
}
KEY_ROLES = (
    "code_repair_only",
    "shared_positive",
    "code_repair_vs_protected_harm",
    "code_negative_but_protected_support",
    "code_source_conflict_with_behavior",
    "uninformative",
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    atlas_rows = read_csv_dicts(Path(args.atlas_rows).expanduser())
    reference_payload = read_json(Path(args.reference_gate).expanduser())
    reference_gates = extract_gate_map(reference_payload)
    evidence_rows = read_csv_dicts(Path(args.evidence_table).expanduser()) if Path(args.evidence_table).exists() else []

    ensure_gate_alignment(atlas_rows, reference_gates)

    zero_rows, virtual_gate_manifests = build_zero_expert_rows(atlas_rows, reference_gates, output_dir, args.reference_gate)
    conflict_rows, module_conflict_rows = build_pairwise_conflict_rows(atlas_rows)
    evidence_context = extract_existing_zero_eval_context(evidence_rows)

    write_csv(output_dir / "zero_expert_summary.csv", zero_rows)
    write_json(output_dir / "zero_expert_summary.json", zero_rows)
    write_csv(output_dir / "pairwise_conflict_summary.csv", conflict_rows)
    write_json(output_dir / "pairwise_conflict_summary.json", conflict_rows)
    write_csv(output_dir / "pairwise_module_conflict_summary.csv", module_conflict_rows)
    write_json(output_dir / "virtual_gate_manifest.json", virtual_gate_manifests)

    make_conflict_heatmap(conflict_rows, figures_dir / "pairwise_conflict_rate_heatmap.png")
    make_conflict_heatmap(conflict_rows, figures_dir / "pairwise_conflict_rate_heatmap.svg")
    make_zero_role_heatmap(zero_rows, figures_dir / "zero_expert_role_risk_heatmap.png")
    make_zero_role_heatmap(zero_rows, figures_dir / "zero_expert_role_risk_heatmap.svg")
    make_expression_dominance_plot(conflict_rows, figures_dir / "pairwise_expression_dominance.png")
    make_expression_dominance_plot(conflict_rows, figures_dir / "pairwise_expression_dominance.svg")

    report = render_report(
        atlas_rows=atlas_rows,
        zero_rows=zero_rows,
        conflict_rows=conflict_rows,
        evidence_context=evidence_context,
        virtual_gate_manifests=virtual_gate_manifests,
        output_dir=output_dir,
    )
    (output_dir / "pairwise_zero_diagnostic_report.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(output_dir), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "report": str(output_dir / "pairwise_zero_diagnostic_report.md"),
                "figures": str(figures_dir),
                "zero_views": list(PAIRS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-rows", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--reference-gate", type=Path, default=DEFAULT_REFERENCE_GATE)
    parser.add_argument("--evidence-table", type=Path, default=DEFAULT_EVIDENCE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def split_tasks(text: Any) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def residual_key(row: dict[str, Any]) -> str:
    return f"{row.get('param_name', '')}::{row.get('expert', '')}"


def extract_gate_map(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("gates", payload)
    if not isinstance(raw, dict):
        raise TypeError("gate payload must be a dict or contain dict-valued `gates`")
    return {str(key): safe_float(value) for key, value in raw.items() if isinstance(value, int | float)}


def expert_from_key(key: str) -> str:
    return key.rsplit("::", 1)[1] if "::" in key else ""


def ensure_gate_alignment(atlas_rows: list[dict[str, Any]], gates: dict[str, float]) -> None:
    missing = [residual_key(row) for row in atlas_rows if residual_key(row) not in gates]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} atlas rows are missing from reference gates: {preview}")


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": mean(values),
        "max": ordered[-1],
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def build_zero_expert_rows(
    atlas_rows: list[dict[str, Any]],
    reference_gates: dict[str, float],
    output_dir: Path,
    reference_gate_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    by_expert = {expert: [row for row in atlas_rows if row.get("expert") == expert] for expert in EXPERTS}

    for pair_name, (expert_a, expert_b, zero_expert) in PAIRS.items():
        removed = by_expert[zero_expert]
        before_values = [reference_gates[residual_key(row)] for row in removed]
        role_counts = Counter(str(row.get("role") or "unknown") for row in removed)
        layer_band_counts = Counter(str(row.get("layer_band") or "unknown") for row in removed)
        module_family_counts = Counter(str(row.get("module_family") or "unknown") for row in removed)
        support_counts = Counter()
        harm_counts = Counter()
        for row in removed:
            support_counts.update(split_tasks(row.get("protected_support_tasks")))
            harm_counts.update(split_tasks(row.get("protected_harm_tasks")))

        item: dict[str, Any] = {
            "pair_name": pair_name,
            "keep_experts": f"{expert_a},{expert_b}",
            "zero_expert": zero_expert,
            "removed_rows": len(removed),
            "mean_removed_coefficient": describe(before_values).get("mean", 0.0),
            "min_removed_coefficient": describe(before_values).get("min", 0.0),
            "max_removed_coefficient": describe(before_values).get("max", 0.0),
            "own_task_signed_effect_mean": mean([task_signal(row, zero_expert) for row in removed]),
            "own_task_expression_mean": mean([task_expression(row, zero_expert) for row in removed]),
            "tool_support_rows": support_counts.get("tool", 0),
            "memory_support_rows": support_counts.get("memory", 0),
            "tool_harm_rows": harm_counts.get("tool", 0),
            "memory_harm_rows": harm_counts.get("memory", 0),
            "code_positive_strength_sum": sum(safe_float(row.get("code_positive_strength")) for row in removed),
            "code_negative_strength_sum": sum(safe_float(row.get("code_negative_strength")) for row in removed),
            "protected_max_harm_norm_mean": mean([safe_float(row.get("protected_max_harm_norm")) for row in removed]),
        }
        for role in KEY_ROLES:
            item[f"role_{role}"] = role_counts.get(role, 0)
        for band in ("early_00_09", "middle_10_19", "late_20_27"):
            item[f"layer_{band}"] = layer_band_counts.get(band, 0)
        for family in ("attention", "mlp"):
            item[f"module_{family}"] = module_family_counts.get(family, 0)
        rows.append(item)

        virtual_gate = dict(reference_gates)
        decision_rows = []
        for key, before in sorted(reference_gates.items()):
            expert = expert_from_key(key)
            after = 0.0 if expert == zero_expert else before
            virtual_gate[key] = after
            decision_rows.append(
                {
                    "key": key,
                    "expert": expert,
                    "before": before,
                    "after": after,
                    "delta": after - before,
                    "reason": "zero_removed_expert" if expert == zero_expert else "kept_pair_expert",
                }
            )
        gate_dir = output_dir / "virtual_gates" / pair_name
        gate_payload = {
            "format": "rcrf_pairwise_zero_gate_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference_gate": str(reference_gate_path),
            "pair_name": pair_name,
            "keep_experts": [expert_a, expert_b],
            "zero_expert": zero_expert,
            "gates": virtual_gate,
            "summary": item,
            "decision_rows": decision_rows,
        }
        write_json(gate_dir / "gates.json", gate_payload)
        write_json(gate_dir / "summary.json", item)
        (gate_dir / "summary.md").write_text(render_gate_summary_markdown(item), encoding="utf-8")
        manifests.append(
            {
                "pair_name": pair_name,
                "keep_experts": f"{expert_a},{expert_b}",
                "zero_expert": zero_expert,
                "gate_path": str(gate_dir / "gates.json"),
                "summary_path": str(gate_dir / "summary.md"),
            }
        )
    return rows, manifests


def build_pairwise_conflict_rows(atlas_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_param: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in atlas_rows:
        by_param[str(row.get("param_name"))][str(row.get("expert"))] = row

    summary_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for pair_name, (expert_a, expert_b, zero_expert) in PAIRS.items():
        pair_records = [
            (param_name, experts[expert_a], experts[expert_b])
            for param_name, experts in by_param.items()
            if expert_a in experts and expert_b in experts
        ]
        for task in TASKS:
            task_stats = summarize_pair_records(pair_records, expert_a, expert_b, task)
            summary_rows.append(
                {
                    "pair_name": pair_name,
                    "expert_a": expert_a,
                    "expert_b": expert_b,
                    "zero_expert": zero_expert,
                    "task": task,
                    **task_stats,
                }
            )
            grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for param_name, row_a, row_b in pair_records:
                group = (str(row_a.get("layer_band") or ""), str(row_a.get("module_family") or ""))
                grouped[group].append((param_name, row_a, row_b))
            for (layer_band, module_family), records in sorted(grouped.items()):
                module_rows.append(
                    {
                        "pair_name": pair_name,
                        "expert_a": expert_a,
                        "expert_b": expert_b,
                        "zero_expert": zero_expert,
                        "task": task,
                        "layer_band": layer_band,
                        "module_family": module_family,
                        **summarize_pair_records(records, expert_a, expert_b, task),
                    }
                )
    return summary_rows, module_rows


def summarize_pair_records(
    records: list[tuple[str, dict[str, Any], dict[str, Any]]],
    expert_a: str,
    expert_b: str,
    task: str,
) -> dict[str, Any]:
    opposite = both_positive = both_negative = same_sign = active = 0
    expr_a_values: list[float] = []
    expr_b_values: list[float] = []
    abs_a_values: list[float] = []
    abs_b_values: list[float] = []
    a_dominates = b_dominates = balanced = 0
    comparable_records = 0
    for _, row_a, row_b in records:
        if not has_task_signal(row_a, task) or not has_task_signal(row_b, task):
            continue
        comparable_records += 1
        eff_a = task_signal(row_a, task)
        eff_b = task_signal(row_b, task)
        expr_a = task_expression(row_a, task)
        expr_b = task_expression(row_b, task)
        expr_a_values.append(expr_a)
        expr_b_values.append(expr_b)
        abs_a_values.append(abs(eff_a))
        abs_b_values.append(abs(eff_b))
        if abs(eff_a) > 1e-12 and abs(eff_b) > 1e-12:
            active += 1
            if eff_a * eff_b < 0:
                opposite += 1
            elif eff_a > 0 and eff_b > 0:
                both_positive += 1
                same_sign += 1
            elif eff_a < 0 and eff_b < 0:
                both_negative += 1
                same_sign += 1
            else:
                same_sign += 1
        ratio = (expr_a + 1e-12) / (expr_b + 1e-12)
        if ratio >= 3.0:
            a_dominates += 1
        elif ratio <= 1.0 / 3.0:
            b_dominates += 1
        else:
            balanced += 1
    total = comparable_records
    return {
        "matched_modules": len(records),
        "comparable_modules": total,
        "active_modules": active,
        "opposite_sign_count": opposite,
        "opposite_sign_rate": opposite / active if active else 0.0,
        "same_sign_rate": same_sign / active if active else 0.0,
        "both_positive_rate": both_positive / active if active else 0.0,
        "both_negative_rate": both_negative / active if active else 0.0,
        "expert_a_expression_mean": mean(expr_a_values) if expr_a_values else 0.0,
        "expert_b_expression_mean": mean(expr_b_values) if expr_b_values else 0.0,
        "expert_a_abs_effect_mean": mean(abs_a_values) if abs_a_values else 0.0,
        "expert_b_abs_effect_mean": mean(abs_b_values) if abs_b_values else 0.0,
        "expert_a_expression_dominates_frac": a_dominates / total if total else 0.0,
        "expert_b_expression_dominates_frac": b_dominates / total if total else 0.0,
        "balanced_expression_frac": balanced / total if total else 0.0,
    }


def extract_existing_zero_eval_context(evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for row in evidence_rows:
        text = " ".join(str(row.get(key, "")) for key in ("candidate", "short", "method", "checkpoint", "gate_path"))
        if "code_zero" in text or "v15" in text:
            context["code_zero_v15"] = {
                "candidate": row.get("candidate"),
                "short": row.get("short"),
                "method": row.get("method"),
                "tool_quick_mean": safe_float(row.get("tool_quick_mean")),
                "memory_eval50_f1": safe_float(row.get("memory_eval50_f1")),
                "livebench_hurt_acc": safe_float(row.get("livebench_hurt_acc")),
                "livebench_hurt_bon_acc": safe_float(row.get("livebench_hurt_bon_acc")),
                "livecodebench_hurt_acc": safe_float(row.get("livecodebench_hurt_acc")),
                "livecodebench_hurt_bon_acc": safe_float(row.get("livecodebench_hurt_bon_acc")),
                "code_hurt_acc_mean": safe_float(row.get("code_hurt_acc_mean")),
                "code_hurt_bon_mean": safe_float(row.get("code_hurt_bon_mean")),
            }
        if ("softveto_v9" in text or "v9" in text) and "code_zero" not in text:
            context.setdefault(
                "reference_v9",
                {
                    "candidate": row.get("candidate"),
                    "short": row.get("short"),
                    "method": row.get("method"),
                    "tool_quick_mean": safe_float(row.get("tool_quick_mean")),
                    "memory_eval50_f1": safe_float(row.get("memory_eval50_f1")),
                    "code_hurt_acc_mean": safe_float(row.get("code_hurt_acc_mean")),
                    "code_hurt_bon_mean": safe_float(row.get("code_hurt_bon_mean")),
                },
            )
    return context


def task_signal(row: dict[str, Any], task: str) -> float:
    """Return the signed diagnostic signal for a task.

    Tool and Memory use teacher-forced signed effect directly.  Code uses the
    outcome-contrast atlas signal because code expert rows store pass/fail
    evidence as source strengths rather than as a direct behavior signed effect.
    """

    if task == "code":
        return safe_float(row.get("code_positive_strength")) - safe_float(row.get("code_negative_strength"))
    return safe_float(row.get(f"{task}_signed_effect"))


def task_expression(row: dict[str, Any], task: str) -> float:
    if task == "code":
        return safe_float(row.get("code_positive_strength")) + safe_float(row.get("code_negative_strength"))
    return safe_float(row.get(f"{task}_expression"))


def has_task_signal(row: dict[str, Any], task: str) -> bool:
    """Whether the atlas row actually contains a signal for this task.

    Code expert rows in the current atlas store code pass/fail contrast, but do
    not contain Tool/Memory behavior signed effects.  Treating those missing
    behavior probes as zeros would falsely imply "no conflict".
    """

    expert = str(row.get("expert") or "")
    if task == "code":
        return (
            safe_float(row.get("code_positive_strength")) > 0.0
            or safe_float(row.get("code_negative_strength")) > 0.0
        )
    if expert == "code":
        return False
    return (
        abs(safe_float(row.get(f"{task}_signed_effect"))) > 0.0
        or safe_float(row.get(f"{task}_expression")) > 0.0
    )


def make_conflict_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    pair_names = list(PAIRS)
    values = [[find_row(rows, pair, task, "opposite_sign_rate") for task in TASKS] for pair in pair_names]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    finite_values = [value for row in values for value in row if not math.isnan(value)]
    im = ax.imshow(values, cmap="Reds", vmin=0.0, vmax=max(0.8, max(finite_values) if finite_values else 0.8))
    ax.set_xticks(range(len(TASKS)), TASKS)
    ax.set_yticks(range(len(pair_names)), [pretty_pair(pair) for pair in pair_names])
    ax.set_title("Pairwise opposite-sign rate by task")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            text = "N/A" if math.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_zero_role_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    role_labels = list(KEY_ROLES)
    values = [[safe_float(row.get(f"role_{role}")) for role in role_labels] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(role_labels)), [role.replace("_", "\n") for role in role_labels], fontsize=8)
    ax.set_yticks(range(len(rows)), [pretty_pair(str(row["pair_name"])) for row in rows])
    ax.set_title("Residual roles removed by zeroing one expert")
    for i, row_values in enumerate(values):
        for j, value in enumerate(row_values):
            ax.text(j, i, f"{int(value)}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_expression_dominance_plot(rows: list[dict[str, Any]], output: Path) -> None:
    pair_names = list(PAIRS)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 3.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        a_values = [find_row(rows, pair, task, "expert_a_expression_dominates_frac") for pair in pair_names]
        b_values = [find_row(rows, pair, task, "expert_b_expression_dominates_frac") for pair in pair_names]
        y = list(range(len(pair_names)))
        ax.barh([v - 0.18 for v in y], a_values, height=0.34, label="expert_a")
        ax.barh([v + 0.18 for v in y], b_values, height=0.34, label="expert_b")
        ax.set_title(task)
        ax.set_xlim(0, 1)
        ax.set_yticks(y, [pretty_pair(pair) for pair in pair_names])
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Expression dominance fraction in each pair")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def find_row(rows: list[dict[str, Any]], pair_name: str, task: str, key: str) -> float:
    for row in rows:
        if row.get("pair_name") == pair_name and row.get("task") == task:
            if safe_int(row.get("comparable_modules")) <= 0:
                return float("nan")
            return safe_float(row.get(key))
    return float("nan")


def pretty_pair(pair_name: str) -> str:
    label = pair_name.replace("__", " / ").replace("_", " ")
    return label


def render_gate_summary_markdown(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {row['pair_name']}",
            "",
            f"- Keep experts: `{row['keep_experts']}`",
            f"- Zero expert: `{row['zero_expert']}`",
            f"- Removed rows: `{row['removed_rows']}`",
            f"- Mean removed coefficient: `{safe_float(row['mean_removed_coefficient']):.4f}`",
            "",
            "This is a virtual diagnostic gate. It has not been baked or evaluated.",
            "",
        ]
    )


def render_report(
    *,
    atlas_rows: list[dict[str, Any]],
    zero_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
    evidence_context: dict[str, Any],
    virtual_gate_manifests: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    lines: list[str] = [
        "# Pairwise-Zero Diagnostics for 8765",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 为什么做两两分析",
        "",
        "三专家联合分析容易把互补、冗余和伤害混在一起。这里每次只保留两个 expert，把第三个 expert 的 196 个 residual coefficient 置为 0，形成三个可审查的二元视角：",
        "",
        "| 视角 | 保留 | 置零 | 目的 |",
        "| --- | --- | --- | --- |",
        "| TM(code=0) | Tool + Memory | Code | 看 Code residual 到底是能力还是干扰源 |",
        "| TC(memory=0) | Tool + Code | Memory | 看 Memory residual 是否是 Code 冲突主因，以及 Memory 能力代价 |",
        "| MC(tool=0) | Memory + Code | Tool | 看 Tool 是否主要是格式行为保护，而不是能力主干 |",
        "",
        "注意：这些 gate 是诊断视角，不是最终方法。它们的作用是解释 residual 结构，而不是直接声称评测性能。",
        "",
        "## 2. 二元 residual 冲突",
        "",
        "| pair | task | active modules | opposite-sign rate | both-positive rate | expression dominance |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in conflict_rows:
        if safe_int(row.get("comparable_modules")) <= 0:
            lines.append(
                f"| `{row['pair_name']}` | {row['task']} | N/A | N/A | N/A | "
                "unavailable: one expert lacks this task-span signal |"
            )
            continue
        dominance = (
            f"{row['expert_a']} {safe_float(row['expert_a_expression_dominates_frac']):.2f} / "
            f"{row['expert_b']} {safe_float(row['expert_b_expression_dominates_frac']):.2f}"
        )
        lines.append(
            f"| `{row['pair_name']}` | {row['task']} | {safe_int(row['active_modules'])} | "
            f"{safe_float(row['opposite_sign_rate']):.3f} | {safe_float(row['both_positive_rate']):.3f} | {dominance} |"
        )
    lines.extend(
        [
            "",
            "读法：`opposite-sign rate` 高，说明两个 expert 在同一任务 span 上经常推向相反方向；`both-positive rate` 高，说明它们更像协同。",
            "",
            "## 3. 置零一个 expert 会删掉什么",
            "",
            "| view | zero expert | removed rows | own expression | tool support | memory support | code positive strength | code negative strength | key removed roles |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in zero_rows:
        key_roles = ", ".join(
            f"{role}:{safe_int(row.get(f'role_{role}'))}"
            for role in KEY_ROLES
            if safe_int(row.get(f"role_{role}")) > 0
        )
        lines.append(
            f"| `{row['pair_name']}` | {row['zero_expert']} | {safe_int(row['removed_rows'])} | "
            f"{safe_float(row['own_task_expression_mean']):.4g} | {safe_int(row['tool_support_rows'])} | "
            f"{safe_int(row['memory_support_rows'])} | {safe_float(row['code_positive_strength_sum']):.2f} | "
            f"{safe_float(row['code_negative_strength_sum']):.2f} | {key_roles} |"
        )
    lines.extend(["", "## 4. 现在最有价值的结论", ""])
    lines.extend(render_findings(zero_rows, conflict_rows, evidence_context))
    lines.extend(
        [
            "",
            "## 5. 与已有评测证据的连接",
            "",
        ]
    )
    ref = evidence_context.get("reference_v9")
    code_zero = evidence_context.get("code_zero_v15")
    if ref and code_zero:
        lines.extend(
            [
                "已有 `code=0` 的真实 quick / code-hurt 评测可以作为 TM(code=0) 的外部验证：",
                "",
                "| candidate | Tool quick | Memory eval50 F1 | Code hurt acc mean | Code hurt BoN mean |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| reference `{ref.get('short')}` | {safe_float(ref.get('tool_quick_mean')):.4f} | {safe_float(ref.get('memory_eval50_f1')):.4f} | {safe_float(ref.get('code_hurt_acc_mean')):.4f} | {safe_float(ref.get('code_hurt_bon_mean')):.4f} |",
                f"| code zero `{code_zero.get('short')}` | {safe_float(code_zero.get('tool_quick_mean')):.4f} | {safe_float(code_zero.get('memory_eval50_f1')):.4f} | {safe_float(code_zero.get('code_hurt_acc_mean')):.4f} | {safe_float(code_zero.get('code_hurt_bon_mean')):.4f} |",
                "",
                "这条证据说明：删除 Code residual 的确能释放一部分 Memory，但 Code 能力显著下降。因此 `code=0` 是 negative control，不是方法。",
                "",
            ]
        )
    else:
        lines.append("未在 evidence table 中找到完整 `code=0` 外部评测；当前只报告机制诊断。")
    lines.extend(
        [
            "## 6. 可视化产物",
            "",
            f"- `{output_dir / 'figures' / 'pairwise_conflict_rate_heatmap.png'}`",
            f"- `{output_dir / 'figures' / 'zero_expert_role_risk_heatmap.png'}`",
            f"- `{output_dir / 'figures' / 'pairwise_expression_dominance.png'}`",
            "",
            "## 7. 虚拟 gate 产物",
            "",
        ]
    )
    for item in virtual_gate_manifests:
        lines.append(f"- `{item['pair_name']}`: `{item['gate_path']}`")
    lines.extend(
        [
            "",
            "这些 gate 只用于诊断。如果后续要 bake / eval，应单独记录为 ablation，不要和主方法 checkpoint 混在一起。",
            "",
            "## 8. 下一步建议",
            "",
            "1. 论文图优先画二元冲突热图，而不是三专家总热图；读者更容易理解哪个 pair 在哪个 task 上冲突。",
            "2. 将 `code=0` 放在 paper 里作为 scalar negative control：它释放 Memory，但摧毁 Code，证明全局置零不是方法。",
            "3. 对 `memory=0` 和 `tool=0` 暂时只作为机制预测，不立刻跑完整评测；先用小规模 quick eval 验证预测方向。",
            "4. 方法主线继续保持简单：二元诊断定位冲突，最终算法只做 residual-level soft constraint，而不是三专家黑盒联合调参。",
            "",
        ]
    )
    return "\n".join(lines)


def render_findings(
    zero_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
    evidence_context: dict[str, Any],
) -> list[str]:
    def row_for_zero(expert: str) -> dict[str, Any]:
        for row in zero_rows:
            if row.get("zero_expert") == expert:
                return row
        return {}

    code_zero = row_for_zero("code")
    memory_zero = row_for_zero("memory")
    tool_zero = row_for_zero("tool")

    def fmt_conflict(pair: str, task: str) -> str:
        value = find_row(conflict_rows, pair, task, "opposite_sign_rate")
        return "N/A" if math.isnan(value) else f"{value:.3f}"

    lines = [
        f"1. **Code 不是纯干扰项。** `code=0` 会删掉 {safe_int(code_zero.get('role_code_repair_only'))} 个 `code_repair_only` row 和 {safe_int(code_zero.get('role_shared_positive'))} 个 `shared_positive` row；已有 v15 评测也显示 Memory 上升但 Code 明显下降。所以不能把 Code expert 整体压掉。",
        f"2. **Memory-Code 是最需要解释的二元关系。** 在 `memory_code__tool_zero` 视角下，Code task 的 opposite-sign rate 为 {fmt_conflict('memory_code__tool_zero', 'code')}；Memory task 不能直接比较，因为当前 atlas 没有 Code expert 在 Memory span 上的 signed-effect。这仍然说明 Memory/Code 冲突需要 residual key + span 级诊断，不能简化成“memory 系数太高”。",
        f"3. **Memory 不能粗暴置零。** `memory=0` 会删掉 {safe_int(memory_zero.get('memory_support_rows'))} 个 Memory support row，同时移除大量 Code 正/负混合证据。它可能让部分 Code 冲突减少，但代价是 Memory 轨迹能力失去主通道。",
        f"4. **Tool 更像格式行为保护。** `tool=0` 删除的 own-task expression 均值是 {safe_float(tool_zero.get('own_task_expression_mean')):.4g}，通常比 Memory 小，但它删掉的是 tool-call span 上的格式行为锚点；这解释了 Tool live 的波动为什么不能只靠整体 reward 分析。",
        "5. **论文主图应从三专家联合图改成二元图。** 二元图能清楚展示：哪个 pair 冲突、在哪个 task 上冲突、置零第三方会删掉哪些 role。这比三专家合在一起更符合第一性诊断。",
    ]
    return [line + "\n" for line in lines]


def render_readme(output_dir: Path) -> str:
    return "\n".join(
        [
            "# Pairwise-Zero Diagnostics",
            "",
            "This directory is generated by:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \\",
            "  scripts/analysis/build_rcrf_pairwise_zero_diagnostics.py",
            "```",
            "",
            "Main report:",
            "",
            f"```text\n{output_dir / 'pairwise_zero_diagnostic_report.md'}\n```",
            "",
            "The virtual gates are diagnostic only.  They have not been baked or evaluated.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
