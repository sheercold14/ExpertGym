#!/usr/bin/env python3
"""Build an executable validation plan from the RCF-BC attribution ledger.

The ledger tells us how each residual row should be treated.  This script turns
that row-level evidence into grouped validation cards: what to test next, why it
matters, and which rows should be inspected first.

It does not generate gates or run evaluation.  It is a planning artifact for the
next counterfactual loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_LEDGER = ROOT / "analysis" / "rcrf_attribution_ledger_20260522" / "rcrf_attribution_ledger.csv"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_validation_plan_20260522"
DEFAULT_DOC_REPORT = REPO_ROOT / "docs" / "report" / "RCRF" / "20260522_rcrf_validation_plan.md"


CARD_FIELDS = [
    "card_id",
    "priority_rank",
    "validation_priority",
    "routing_action",
    "archetype",
    "expert",
    "layer_band",
    "module_family",
    "row_count",
    "changed_rows",
    "mean_abs_v18_delta",
    "mean_code_positive_strength",
    "mean_code_negative_strength",
    "mean_behavior_harm",
    "card_score",
    "validation_type",
    "hypothesis",
    "success_criterion",
    "failure_read",
    "representative_keys",
]


ACTION_WEIGHT = {
    "retain_with_behavior_constraint": 95.0,
    "retain_capability_delta": 90.0,
    "protect_behavior_anchor": 85.0,
    "behavior_guard": 82.0,
    "do_not_prune_without_counterfactual": 80.0,
    "retain_continuous_field": 74.0,
    "keep_small_until_validated": 55.0,
    "hold_base": 5.0,
}

PRIORITY_WEIGHT = {"high": 100.0, "medium": 35.0, "low": 0.0}


def main() -> None:
    args = parse_args()
    ledger_path = Path(args.ledger).expanduser().resolve()
    rows = read_csv(ledger_path)
    if not rows:
        raise ValueError(f"No rows loaded from {ledger_path}")

    cards = build_cards(rows, max_cards=args.max_cards, sample_keys=args.sample_keys)
    summary = build_summary(rows, cards, ledger_path)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_cards.csv", cards, CARD_FIELDS)
    write_jsonl(output_dir / "validation_cards.jsonl", cards)
    write_json(output_dir / "validation_plan_summary.json", summary)
    report = render_markdown(summary, cards, output_dir)
    (output_dir / "validation_plan.md").write_text(report, encoding="utf-8")

    if args.doc_report:
        doc_report = Path(args.doc_report).expanduser().resolve()
        doc_report.parent.mkdir(parents=True, exist_ok=True)
        doc_report.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cards": len(cards),
                "report": str(output_dir / "validation_plan.md"),
                "doc_report": str(Path(args.doc_report).expanduser().resolve()) if args.doc_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    parser.add_argument("--max-cards", type=int, default=48)
    parser.add_argument("--sample-keys", type=int, default=8)
    return parser.parse_args()


def build_cards(rows: list[dict[str, str]], *, max_cards: int, sample_keys: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["routing_action"],
            row["validation_priority"],
            row["archetype"],
            row["expert"],
            row["layer_band"],
            row["module_family"],
        )
        grouped[key].append(row)

    cards: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        action, priority, archetype, expert, layer_band, module_family = group_key
        card_score = score_group(group_rows)
        validation_type, hypothesis, success_criterion, failure_read = validation_protocol(action, archetype)
        top_rows = sorted(group_rows, key=row_score, reverse=True)[:sample_keys]
        card_id = slugify(f"{priority}-{action}-{archetype}-{expert}-{layer_band}-{module_family}")
        cards.append(
            {
                "card_id": card_id,
                "priority_rank": 0,
                "validation_priority": priority,
                "routing_action": action,
                "archetype": archetype,
                "expert": expert,
                "layer_band": layer_band,
                "module_family": module_family,
                "row_count": len(group_rows),
                "changed_rows": sum(1 for row in group_rows if abs(safe_float(row.get("v18_delta"))) > 1e-12),
                "mean_abs_v18_delta": avg(abs(safe_float(row.get("v18_delta"))) for row in group_rows),
                "mean_code_positive_strength": avg(safe_float(row.get("code_positive_strength")) for row in group_rows),
                "mean_code_negative_strength": avg(safe_float(row.get("code_negative_strength")) for row in group_rows),
                "mean_behavior_harm": avg(safe_float(row.get("protected_max_harm_norm")) for row in group_rows),
                "card_score": card_score,
                "validation_type": validation_type,
                "hypothesis": hypothesis,
                "success_criterion": success_criterion,
                "failure_read": failure_read,
                "representative_keys": ";".join(row["key"] for row in top_rows),
            }
        )

    cards.sort(key=lambda card: (-safe_float(card["card_score"]), card["card_id"]))
    for index, card in enumerate(cards[:max_cards], start=1):
        card["priority_rank"] = index
    return cards[:max_cards]


def validation_protocol(action: str, archetype: str) -> tuple[str, str, str, str]:
    if action == "retain_capability_delta":
        return (
            "capability_drop_or_restore",
            "这些 row 是最干净的能力 residual；如果把它们投回 base，Code hurt 应该下降。",
            "drop/restore 对照中 Code hurt BoN 下降，同时 Tool/Memory 不出现足以解释该下降的反向收益。",
            "若 drop 后 Code 不降，说明当前 pass/fail contrast 不是因果能力证据。",
        )
    if action == "retain_with_behavior_constraint":
        return (
            "pareto_boundary_scale_check",
            "这些 row 同时有能力和行为风险，是 Pareto 边界而非单任务能力。",
            "连续缩放能沿 Code 与 Tool/Memory trade-off 移动；hard drop 或 hard keep 都更差。",
            "若缩放不影响任何正式指标，说明 behavior harm probe 只是相关性。",
        )
    if action == "retain_continuous_field":
        return (
            "source_span_counterfactual",
            "mixed source/span row 需要连续场；离散 source dominant routing 会丢失能力。",
            "按 source/span 分组干预能解释 v18 优于 hard routing 的 Code 差异。",
            "若 source/span 分组没有指标差异，continuous field 的证据需要回到 probe 设计。",
        )
    if action == "protect_behavior_anchor":
        return (
            "behavior_anchor_drop_test",
            "这些 row 承载 Tool/Memory behavior，不能因为 Code 负证据就压低。",
            "drop 或 shrink 后 Tool/Memory quick 下降，证明它们是行为 anchor。",
            "若 drop 后行为不降，应从 behavior constraint 中移除该组。",
        )
    if action == "behavior_guard":
        return (
            "veto_release_test",
            "这些 row 的主要作用是行为 guard，不应该为了 Code 单侧收益放大。",
            "释放 veto 后 Tool/Memory 下降，或 Code 收益不足以补偿行为损失。",
            "若释放后三项都升，当前 veto 过强。",
        )
    if action == "do_not_prune_without_counterfactual":
        return (
            "prune_safety_audit",
            "低置信/负向 code row 可能仍有行为或组合价值，不能直接剪。",
            "isolated shrink 必须同时通过 Tool/Memory quick 与 Code hurt，才允许进入剪枝候选。",
            "若 shrink 稳定提升三项，可将该组升级为 soft_shrink_candidate。",
        )
    if action == "keep_small_until_validated":
        return (
            "low_confidence_delta_audit",
            "这些 row 只能维持小 delta，不能作为主 claim。",
            "只有反事实指标稳定时才扩大其权重或纳入主方法解释。",
            "若无指标影响，应回退为 hold_base。",
        )
    return (
        "holdout_monitor",
        "这些 row 当前没有足够证据，保持 base。",
        "未来新增 source/span 证据前不应主动干预。",
        "若新增证据改变其 decision，再进入对应验证协议。",
    )


def build_summary(rows: list[dict[str, str]], cards: list[dict[str, Any]], ledger_path: Path) -> dict[str, Any]:
    priority_counts = count_by(rows, "validation_priority")
    action_counts = count_by(rows, "routing_action")
    p0_cards = cards[:12]
    return {
        "format": "rcrf_validation_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ledger": str(ledger_path),
        "row_count": len(rows),
        "card_count": len(cards),
        "priority_counts": priority_counts,
        "action_counts": action_counts,
        "p0_cards": p0_cards,
        "core_loop": [
            "select one validation card",
            "build the minimal isolated gate intervention for its representative rows",
            "bake checkpoint",
            "run Tool/Memory quick first",
            "run Code hurt only if Tool/Memory do not fail the agreed guardrail",
            "write the counterfactual result back into the ledger/effect table",
        ],
    }


def score_group(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    action = rows[0].get("routing_action", "")
    priority = rows[0].get("validation_priority", "")
    code_pos = avg(safe_float(row.get("code_positive_strength")) for row in rows)
    code_neg = avg(safe_float(row.get("code_negative_strength")) for row in rows)
    behavior = avg(safe_float(row.get("protected_max_harm_norm")) for row in rows)
    delta = avg(abs(safe_float(row.get("v18_delta"))) for row in rows)
    changed_bonus = sum(1 for row in rows if abs(safe_float(row.get("v18_delta"))) > 1e-12) / max(len(rows), 1)
    size_bonus = min(len(rows), 12) / 12.0
    return (
        PRIORITY_WEIGHT.get(priority, 0.0)
        + ACTION_WEIGHT.get(action, 0.0)
        + 6.0 * code_pos
        + 4.0 * code_neg
        + 5.0 * behavior
        + 200.0 * delta
        + 8.0 * changed_bonus
        + 4.0 * size_bonus
    )


def row_score(row: dict[str, str]) -> float:
    return (
        6.0 * safe_float(row.get("code_positive_strength"))
        + 4.0 * safe_float(row.get("code_negative_strength"))
        + 5.0 * safe_float(row.get("protected_max_harm_norm"))
        + 200.0 * abs(safe_float(row.get("v18_delta")))
    )


def count_by(rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    changed: dict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key, "")
        counts[value] += 1
        if abs(safe_float(row.get("v18_delta"))) > 1e-12:
            changed[value] += 1
    return [{key: value, "row_count": counts[value], "changed_rows": changed[value]} for value in sorted(counts)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_markdown(summary: dict[str, Any], cards: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# 2026-05-22 RCF-BC Validation Plan",
        "",
        "## 目的",
        "",
        "这个文件把 attribution ledger 转成下一轮可执行验证卡片。它不改 gate、不跑评测，只规定哪些 residual group 最值得做反事实实验，以及通过什么指标判断该机制是否成立。",
        "",
        "## 闭环协议",
        "",
    ]
    for step in summary["core_loop"]:
        lines.append(f"- {step}")

    lines.extend(
        [
            "",
            "## 优先级概览",
            "",
            "| priority | rows | changed |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["priority_counts"]:
        lines.append(f"| `{row['validation_priority']}` | {row['row_count']} | {row['changed_rows']} |")

    lines.extend(
        [
            "",
            "## P0 验证卡片",
            "",
            "| rank | card | action | rows | changed | score | validation |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for card in cards[:12]:
        lines.append(
            f"| {card['priority_rank']} | `{card['card_id']}` | `{card['routing_action']}` | "
            f"{card['row_count']} | {card['changed_rows']} | {safe_float(card['card_score']):.2f} | "
            f"{card['validation_type']} |"
        )

    lines.extend(["", "## 卡片详情", ""])
    for card in cards[:12]:
        keys = str(card["representative_keys"]).split(";") if card["representative_keys"] else []
        lines.extend(
            [
                f"### {card['priority_rank']}. `{card['card_id']}`",
                "",
                f"- action: `{card['routing_action']}`",
                f"- priority: `{card['validation_priority']}`",
                f"- group: `{card['archetype']} / {card['expert']} / {card['layer_band']} / {card['module_family']}`",
                f"- rows: `{card['row_count']}`, changed: `{card['changed_rows']}`",
                f"- score: `{safe_float(card['card_score']):.2f}`",
                f"- hypothesis: {card['hypothesis']}",
                f"- success: {card['success_criterion']}",
                f"- failure read: {card['failure_read']}",
                "- representative keys:",
            ]
        )
        for key in keys[:8]:
            lines.append(f"  - `{key}`")
        lines.append("")

    lines.extend(
        [
            "## 产物",
            "",
            f"- cards CSV: `{output_dir / 'validation_cards.csv'}`",
            f"- cards JSONL: `{output_dir / 'validation_cards.jsonl'}`",
            f"- summary JSON: `{output_dir / 'validation_plan_summary.json'}`",
            f"- ledger: `{summary['ledger']}`",
            "",
        ]
    )
    return "\n".join(lines)


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def avg(values: Any) -> float:
    vals = [float(value) for value in values]
    return mean(vals) if vals else 0.0


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


if __name__ == "__main__":
    main()
