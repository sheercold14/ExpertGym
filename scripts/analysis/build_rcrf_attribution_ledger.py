#!/usr/bin/env python3
"""Build a row-level attribution ledger for RCF-BC.

The ledger joins three evidence layers:

1. row-level residual evidence from the conflict-cluster table;
2. the current RCF-BC coefficient / delta for each residual row;
3. group-level counterfactual effects from mechanical ablations.

It is intentionally an audit artifact rather than a new gate generator.  The
goal is to make each residual row's status explicit: capability evidence,
behavior evidence, counterfactual risk, and the recommended handling rule.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_CLUSTER_ROWS = ROOT / "analysis" / "rcrf_conflict_clusters_20260522" / "conflict_cluster_rows.jsonl"
DEFAULT_EFFECT_SUMMARY = ROOT / "analysis" / "rcrf_counterfactual_effects_20260522" / "counterfactual_effect_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_attribution_ledger_20260522"
DEFAULT_DOC_REPORT = REPO_ROOT / "docs" / "report" / "RCRF" / "20260522_rcrf_attribution_ledger.md"


LEDGER_FIELDS = [
    "key",
    "param_name",
    "expert",
    "layer",
    "layer_band",
    "module",
    "module_family",
    "archetype",
    "role",
    "decision",
    "decision_reason",
    "routing_action",
    "validation_priority",
    "next_validation",
    "counterfactual_group",
    "cf_direct_rows",
    "cf_delta_tool_live_parallel",
    "cf_delta_memory_eval50_f1",
    "cf_delta_livebench_hurt_bon_acc",
    "cf_delta_livecodebench_hurt_bon_acc",
    "cf_primary_read",
    "v18_coefficient",
    "v18_delta",
    "base_coefficient",
    "code_positive_strength",
    "code_negative_strength",
    "code_positive_sources",
    "code_negative_sources",
    "protected_support_tasks",
    "protected_harm_tasks",
    "protected_max_harm_norm",
    "behavior_support_count",
    "behavior_harm_count",
    "primary_abs_delta",
    "paper_use",
]


def main() -> None:
    args = parse_args()
    cluster_rows = load_jsonl(Path(args.cluster_rows).expanduser())
    effect_summary = load_json(Path(args.effect_summary).expanduser())
    effect_by_candidate = {str(row["candidate"]): row for row in effect_summary.get("rows", [])}

    ledger_rows = [build_ledger_row(row, effect_by_candidate) for row in cluster_rows]
    summary = build_summary(
        ledger_rows,
        cluster_rows=Path(args.cluster_rows).expanduser().resolve(),
        effect_summary=Path(args.effect_summary).expanduser().resolve(),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "rcrf_attribution_ledger.csv", ledger_rows, LEDGER_FIELDS)
    write_jsonl(output_dir / "rcrf_attribution_ledger.jsonl", ledger_rows)
    write_json(output_dir / "rcrf_attribution_ledger_summary.json", summary)
    report = render_markdown(summary, output_dir)
    (output_dir / "rcrf_attribution_ledger_report.md").write_text(report, encoding="utf-8")
    if args.doc_report:
        doc_report = Path(args.doc_report).expanduser().resolve()
        doc_report.parent.mkdir(parents=True, exist_ok=True)
        doc_report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(ledger_rows),
                "report": str(output_dir / "rcrf_attribution_ledger_report.md"),
                "doc_report": str(Path(args.doc_report).expanduser().resolve()) if args.doc_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-rows", type=Path, default=DEFAULT_CLUSTER_ROWS)
    parser.add_argument("--effect-summary", type=Path, default=DEFAULT_EFFECT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    return parser.parse_args()


def build_ledger_row(row: dict[str, Any], effect_by_candidate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    group = counterfactual_group(row)
    effect = effect_by_candidate.get(group, {}) if group else {}
    decision, reason = decision_for(row, effect)
    routing_action, validation_priority, next_validation = routing_for(decision, row, effect)
    return {
        "key": str(row.get("key", "")),
        "param_name": str(row.get("param_name", "")),
        "expert": str(row.get("expert", "")),
        "layer": int(safe_float(row.get("layer"))),
        "layer_band": str(row.get("layer_band", "")),
        "module": str(row.get("module", "")),
        "module_family": str(row.get("module_family", "")),
        "archetype": str(row.get("archetype", "")),
        "role": str(row.get("role", "")),
        "decision": decision,
        "decision_reason": reason,
        "routing_action": routing_action,
        "validation_priority": validation_priority,
        "next_validation": next_validation,
        "counterfactual_group": group,
        "cf_direct_rows": int(safe_float(effect.get("direct_rows"))),
        "cf_delta_tool_live_parallel": safe_float(effect.get("delta_tool_live_parallel")),
        "cf_delta_memory_eval50_f1": safe_float(effect.get("delta_memory_eval50_f1")),
        "cf_delta_livebench_hurt_bon_acc": safe_float(effect.get("delta_livebench_hurt_bon_acc")),
        "cf_delta_livecodebench_hurt_bon_acc": safe_float(effect.get("delta_livecodebench_hurt_bon_acc")),
        "cf_primary_read": str(effect.get("primary_read", "")),
        "v18_coefficient": safe_float(row.get("v18_rcf_bc_coefficient")),
        "v18_delta": safe_float(row.get("v18_rcf_bc_delta")),
        "base_coefficient": safe_float(row.get("base_coefficient")),
        "code_positive_strength": safe_float(row.get("code_positive_strength")),
        "code_negative_strength": safe_float(row.get("code_negative_strength")),
        "code_positive_sources": str(row.get("code_positive_sources", "")),
        "code_negative_sources": str(row.get("code_negative_sources", "")),
        "protected_support_tasks": str(row.get("protected_support_tasks", "")),
        "protected_harm_tasks": str(row.get("protected_harm_tasks", "")),
        "protected_max_harm_norm": safe_float(row.get("protected_max_harm_norm")),
        "behavior_support_count": int(safe_float(row.get("behavior_support_count"))),
        "behavior_harm_count": int(safe_float(row.get("behavior_harm_count"))),
        "primary_abs_delta": safe_float(row.get("primary_abs_delta")),
        "paper_use": str(row.get("paper_use", "")),
    }


def counterfactual_group(row: dict[str, Any]) -> str:
    expert = str(row.get("expert", ""))
    archetype = str(row.get("archetype", ""))
    if expert != "code":
        return ""
    if archetype == "code_negative_noise":
        return "v22"
    if archetype == "weak_or_uninformative":
        return "v23"
    return ""


def decision_for(row: dict[str, Any], effect: dict[str, Any]) -> tuple[str, str]:
    archetype = str(row.get("archetype", ""))
    expert = str(row.get("expert", ""))
    support = int(safe_float(row.get("behavior_support_count")))
    harm = int(safe_float(row.get("behavior_harm_count")))
    changed = bool(row.get("primary_changed"))
    cf_tool = safe_float(effect.get("delta_tool_live_parallel"))
    cf_memory = safe_float(effect.get("delta_memory_eval50_f1"))
    cf_code_bon = max(
        safe_float(effect.get("delta_livebench_hurt_bon_acc")),
        safe_float(effect.get("delta_livecodebench_hurt_bon_acc")),
    )

    if expert == "code" and archetype in {"code_negative_noise", "weak_or_uninformative"}:
        if cf_tool < 0.0 or cf_memory < 0.0:
            return (
                "audit_before_prune",
                "counterfactual shrink hurts Tool/Memory; low or negative Code evidence is not sufficient for pruning",
            )
        return "candidate_for_soft_shrink", "counterfactual shrink does not hurt protected behavior"
    if archetype == "clean_code_repair":
        return "keep_capability_delta", "positive Code evidence with little behavior harm"
    if archetype == "code_repair_with_behavior_harm":
        return "soft_constrained_capability", "Code repair conflicts with protected behavior; keep continuous small deltas only"
    if archetype == "code_source_conflict":
        return "keep_continuous_field", "mixed Code evidence across sources; avoid hard routing"
    if support > 0 and cf_code_bon >= 0.0:
        return "protect_behavior_support", "row supports protected behavior or has positive counterfactual Code evidence"
    if harm > 0:
        return "behavior_constraint", "row has protected behavior harm evidence"
    if archetype == "behavior_only":
        return "behavior_audit", "row has behavior evidence without Code contrast"
    if not changed:
        return "hold", "RCF-BC leaves row unchanged"
    return "low_confidence_keep_small", "changed row lacks decisive counterfactual support for pruning"


def routing_for(decision: str, row: dict[str, Any], effect: dict[str, Any]) -> tuple[str, str, str]:
    code_pos = safe_float(row.get("code_positive_strength"))
    code_neg = safe_float(row.get("code_negative_strength"))
    behavior_harm = safe_float(row.get("protected_max_harm_norm"))
    changed = bool(row.get("primary_changed"))
    cf_tool = safe_float(effect.get("delta_tool_live_parallel"))
    cf_memory = safe_float(effect.get("delta_memory_eval50_f1"))
    cf_lcb_bon = safe_float(effect.get("delta_livecodebench_hurt_bon_acc"))

    if decision == "audit_before_prune":
        return (
            "do_not_prune_without_counterfactual",
            "high",
            "先不要剪；只有在 isolated shrink/restore 同时通过 Tool/Memory quick 与 Code hurt 后，才能把它当作可压低 residual。",
        )
    if decision == "keep_capability_delta":
        priority = "high" if code_pos >= 1.0 and behavior_harm <= 0.1 else "medium"
        return (
            "retain_capability_delta",
            priority,
            "保留当前能力 delta；后续可做 restore/drop 单行组验证它是否稳定提升 Code hurt。",
        )
    if decision == "soft_constrained_capability":
        priority = "high" if changed and (code_pos >= 1.0 or behavior_harm >= 1.0) else "medium"
        return (
            "retain_with_behavior_constraint",
            priority,
            "保留连续小 delta，但任何增强都必须先过 Tool/Memory 行为约束；适合作为 soft constraint 论文案例。",
        )
    if decision == "keep_continuous_field":
        priority = "high" if changed and code_pos >= 1.0 and code_neg >= 1.0 else "medium"
        return (
            "retain_continuous_field",
            priority,
            "保持连续场，不做 hard routing；需要按 source/span 做更细 counterfactual，验证 mixed evidence 的真实方向。",
        )
    if decision == "protect_behavior_support":
        priority = "high" if changed or behavior_harm >= 1.0 else "medium"
        return (
            "protect_behavior_anchor",
            priority,
            "把它作为 Tool/Memory behavior anchor；如果未来压低该 row，必须触发行为回归评测。",
        )
    if decision == "behavior_constraint":
        priority = "high" if changed or behavior_harm >= 1.0 else "medium"
        return (
            "behavior_guard",
            priority,
            "优先作为负约束或 veto 信号；不应为了 Code 单侧收益直接放大。",
        )
    if decision == "low_confidence_keep_small":
        priority = "medium" if changed else "low"
        return (
            "keep_small_until_validated",
            priority,
            "当前只允许小幅连续 delta；需要通过局部反事实确认后再扩大或压低。",
        )
    if decision == "candidate_for_soft_shrink":
        priority = "medium" if cf_lcb_bon >= 0.0 and cf_tool >= 0.0 and cf_memory >= 0.0 else "high"
        return (
            "soft_shrink_candidate",
            priority,
            "可作为 residual-level shrink 候选，但仍需完整 Tool/Memory/Code 三项验证。",
        )
    return (
        "hold_base",
        "low",
        "维持 base 系数；除非新证据改变该 row 的 source/span attribution。",
    )


def build_summary(ledger_rows: list[dict[str, Any]], *, cluster_rows: Path, effect_summary: Path) -> dict[str, Any]:
    by_decision = aggregate(ledger_rows, "decision")
    by_action = aggregate(ledger_rows, "routing_action")
    by_priority = aggregate(ledger_rows, "validation_priority")
    by_archetype = aggregate(ledger_rows, "archetype")
    by_group = aggregate([row for row in ledger_rows if row["counterfactual_group"]], "counterfactual_group")
    high_risk = sorted(
        [
            row
            for row in ledger_rows
            if row["decision"] == "audit_before_prune"
            or (row["cf_delta_tool_live_parallel"] < 0.0 and row["cf_delta_memory_eval50_f1"] < 0.0)
        ],
        key=lambda row: (
            row["cf_delta_tool_live_parallel"] + row["cf_delta_memory_eval50_f1"],
            -row["code_positive_strength"],
        ),
    )[:30]
    return {
        "format": "rcrf_attribution_ledger_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cluster_rows": str(cluster_rows),
        "effect_summary": str(effect_summary),
        "row_count": len(ledger_rows),
        "decision_summary": by_decision,
        "routing_action_summary": by_action,
        "validation_priority_summary": by_priority,
        "archetype_summary": by_archetype,
        "counterfactual_group_summary": by_group,
        "high_risk_rows": compact_rows(high_risk),
        "takeaways": [
            "ledger 将 proxy row label 与 counterfactual effect 分开，避免把相关性标签当作因果结论。",
            "当 code_negative_noise 或 weak_or_uninformative group shrink 会伤 Tool/Memory 时，这些 row 被标为 audit_before_prune，而不是直接剪掉。",
            "当前证据支持 continuous residual field + behavior-support audit，而不是 hard pruning 或全局 task scalar suppression。",
        ],
    }


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    output = []
    for value, items in sorted(grouped.items()):
        output.append(
            {
                key: value,
                "row_count": len(items),
                "changed_rows": sum(1 for row in items if abs(float(row.get("v18_delta", 0.0))) > 1e-12),
                "mean_abs_v18_delta": mean(abs(float(row.get("v18_delta", 0.0))) for row in items) if items else 0.0,
                "mean_code_positive_strength": mean(float(row.get("code_positive_strength", 0.0)) for row in items)
                if items
                else 0.0,
                "mean_code_negative_strength": mean(float(row.get("code_negative_strength", 0.0)) for row in items)
                if items
                else 0.0,
                "mean_behavior_harm": mean(float(row.get("protected_max_harm_norm", 0.0)) for row in items)
                if items
                else 0.0,
            }
        )
    return output


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "key",
        "expert",
        "layer",
        "module",
        "archetype",
        "decision",
        "routing_action",
        "validation_priority",
        "counterfactual_group",
        "cf_delta_tool_live_parallel",
        "cf_delta_memory_eval50_f1",
        "cf_delta_livebench_hurt_bon_acc",
        "cf_delta_livecodebench_hurt_bon_acc",
        "code_positive_sources",
        "code_negative_sources",
        "protected_support_tasks",
        "protected_harm_tasks",
        "v18_delta",
    ]
    return [{field: row.get(field) for field in fields} for row in rows]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value: Any) -> str:
    return f"{safe_float(value):.4f}"


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# 2026-05-22 RCF-BC Attribution Ledger",
        "",
        "## 目的",
        "",
        "这是 RCF-BC 框架的逐 residual row 审计表。它把 residual 机制证据、当前系数和 counterfactual group effect 合并到同一张表里，让 row label 保持为可验证假设，而不是未经验证的因果结论。",
        "",
        "## 决策汇总",
        "",
        "| decision | rows | changed | mean code+ | mean code- | mean behavior harm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["decision_summary"]:
        lines.append(
            f"| `{row['decision']}` | {row['row_count']} | {row['changed_rows']} | "
            f"{fmt(row['mean_code_positive_strength'])} | {fmt(row['mean_code_negative_strength'])} | "
            f"{fmt(row['mean_behavior_harm'])} |"
        )
    lines.extend(
        [
            "",
            "## 可执行动作",
            "",
            "| action | rows | changed | mean code+ | mean code- | mean behavior harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["routing_action_summary"]:
        lines.append(
            f"| `{row['routing_action']}` | {row['row_count']} | {row['changed_rows']} | "
            f"{fmt(row['mean_code_positive_strength'])} | {fmt(row['mean_code_negative_strength'])} | "
            f"{fmt(row['mean_behavior_harm'])} |"
        )
    lines.extend(
        [
            "",
            "## 验证优先级",
            "",
            "| priority | rows | changed | mean code+ | mean code- | mean behavior harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["validation_priority_summary"]:
        lines.append(
            f"| `{row['validation_priority']}` | {row['row_count']} | {row['changed_rows']} | "
            f"{fmt(row['mean_code_positive_strength'])} | {fmt(row['mean_code_negative_strength'])} | "
            f"{fmt(row['mean_behavior_harm'])} |"
        )
    lines.extend(
        [
            "",
            "## 反事实分组",
            "",
            "| group | rows | changed | mean code+ | mean code- | mean behavior harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["counterfactual_group_summary"]:
        lines.append(
            f"| `{row['counterfactual_group']}` | {row['row_count']} | {row['changed_rows']} | "
            f"{fmt(row['mean_code_positive_strength'])} | {fmt(row['mean_code_negative_strength'])} | "
            f"{fmt(row['mean_behavior_harm'])} |"
        )
    lines.extend(
        [
            "",
            "## 高风险行",
            "",
            "| key | archetype | decision | group | dTool live | dMemory | dLB BoN | dLCB BoN |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["high_risk_rows"][:20]:
        lines.append(
            f"| `{row['key']}` | `{row['archetype']}` | `{row['decision']}` | `{row['counterfactual_group']}` | "
            f"{fmt(row['cf_delta_tool_live_parallel'])} | {fmt(row['cf_delta_memory_eval50_f1'])} | "
            f"{fmt(row['cf_delta_livebench_hurt_bon_acc'])} | {fmt(row['cf_delta_livecodebench_hurt_bon_acc'])} |"
        )
    lines.extend(["", "## 框架结论", ""])
    for item in summary["takeaways"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- CSV: `{output_dir / 'rcrf_attribution_ledger.csv'}`",
            f"- JSONL: `{output_dir / 'rcrf_attribution_ledger.jsonl'}`",
            f"- Summary JSON: `{output_dir / 'rcrf_attribution_ledger_summary.json'}`",
            f"- Cluster rows: `{summary['cluster_rows']}`",
            f"- Counterfactual effects: `{summary['effect_summary']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
