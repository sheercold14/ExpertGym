#!/usr/bin/env python3
"""Materialize validation-card interventions as OP-VEC gate files.

This script is the next step after `build_rcrf_validation_plan.py`: it takes the
top validation cards and builds minimal isolated gate interventions for them.

It does not bake checkpoints or run evaluation.  The generated gate files are
counterfactual candidates that can be evaluated through the existing OP-VEC bake
and quick-eval harness.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_CARDS = ROOT / "analysis" / "rcrf_validation_plan_20260522" / "validation_cards.csv"
DEFAULT_LEDGER = ROOT / "analysis" / "rcrf_attribution_ledger_20260522" / "rcrf_attribution_ledger.csv"
DEFAULT_SOURCE_GATE = ROOT / "contrast_gates" / "residual_capability_field_behavior_constraints_v18" / "gates.json"
DEFAULT_BASE_GATE = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_OUTPUT_DIR = ROOT / "contrast_gates" / "validation_card_interventions_20260522"
DEFAULT_DOC_REPORT = REPO_ROOT / "docs" / "report" / "RCRF" / "20260522_rcrf_validation_interventions.md"


DECISION_ROW_FIELDS = [
    "key",
    "param_name",
    "expert",
    "layer",
    "layer_band",
    "module_family",
    "archetype",
    "routing_action",
    "validation_priority",
    "operation",
    "before",
    "after",
    "base",
    "delta_before",
    "delta_after",
    "changed_by_intervention",
]


def main() -> None:
    args = parse_args()
    cards = read_csv(Path(args.cards).expanduser().resolve())
    ledger_rows = read_csv(Path(args.ledger).expanduser().resolve())
    source_payload = load_json(Path(args.source_gate).expanduser().resolve())
    base_payload = load_json(Path(args.base_gate).expanduser().resolve())
    source_gates = extract_gate_map(source_payload)
    base_gates = extract_gate_map(base_payload)

    selected_cards = select_cards(cards, args.card_rank, args.card_id, args.top_k)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for card in selected_cards:
        operation = args.operation if args.operation != "auto" else auto_operation(card)
        candidate = build_candidate(
            card=card,
            ledger_rows=ledger_rows,
            source_payload=source_payload,
            source_gates=source_gates,
            base_gates=base_gates,
            source_gate_path=Path(args.source_gate).expanduser().resolve(),
            operation=operation,
            shrink_scale=args.shrink_scale,
            delta_scale=args.delta_scale,
            output_root=output_dir,
        )
        generated.append(candidate)

    manifest = {
        "format": "rcrf_validation_interventions_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cards": str(Path(args.cards).expanduser().resolve()),
        "ledger": str(Path(args.ledger).expanduser().resolve()),
        "source_gate": str(Path(args.source_gate).expanduser().resolve()),
        "base_gate": str(Path(args.base_gate).expanduser().resolve()),
        "output_dir": str(output_dir),
        "top_k": args.top_k,
        "requested_card_rank": args.card_rank,
        "requested_card_id": args.card_id,
        "operation": args.operation,
        "shrink_scale": args.shrink_scale,
        "delta_scale": args.delta_scale,
        "generated": generated,
    }
    write_json(output_dir / "validation_interventions_manifest.json", manifest)
    report = render_markdown(manifest)
    (output_dir / "validation_interventions.md").write_text(report, encoding="utf-8")
    if args.doc_report:
        doc_report = Path(args.doc_report).expanduser().resolve()
        doc_report.parent.mkdir(parents=True, exist_ok=True)
        doc_report.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "generated": len(generated),
                "report": str(output_dir / "validation_interventions.md"),
                "doc_report": str(Path(args.doc_report).expanduser().resolve()) if args.doc_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--source-gate", type=Path, default=DEFAULT_SOURCE_GATE)
    parser.add_argument("--base-gate", type=Path, default=DEFAULT_BASE_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--card-rank", type=int, action="append", default=[])
    parser.add_argument("--card-id", action="append", default=[])
    parser.add_argument(
        "--operation",
        choices=["auto", "drop-delta", "half-delta", "shrink-coeff"],
        default="auto",
    )
    parser.add_argument("--shrink-scale", type=float, default=0.5)
    parser.add_argument("--delta-scale", type=float, default=0.5)
    return parser.parse_args()


def select_cards(cards: list[dict[str, str]], ranks: list[int], ids: list[str], top_k: int) -> list[dict[str, str]]:
    by_rank = {int(safe_float(card.get("priority_rank"))): card for card in cards}
    by_id = {card["card_id"]: card for card in cards}
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    if ranks or ids:
        for rank in ranks:
            if rank not in by_rank:
                raise ValueError(f"Unknown card rank: {rank}")
            card = by_rank[rank]
            if card["card_id"] not in seen:
                selected.append(card)
                seen.add(card["card_id"])
        for card_id in ids:
            if card_id not in by_id:
                raise ValueError(f"Unknown card id: {card_id}")
            card = by_id[card_id]
            if card["card_id"] not in seen:
                selected.append(card)
                seen.add(card["card_id"])
        return selected
    return cards[:top_k]


def build_candidate(
    *,
    card: dict[str, str],
    ledger_rows: list[dict[str, str]],
    source_payload: dict[str, Any],
    source_gates: dict[str, float],
    base_gates: dict[str, float],
    source_gate_path: Path,
    operation: str,
    shrink_scale: float,
    delta_scale: float,
    output_root: Path,
) -> dict[str, Any]:
    keys = [key for key in card.get("representative_keys", "").split(";") if key]
    if not keys:
        raise ValueError(f"Card has no representative keys: {card['card_id']}")

    ledger_by_key = {row["key"]: row for row in ledger_rows}
    gates = dict(source_gates)
    decision_rows = []
    for key in keys:
        if key not in gates:
            raise KeyError(f"Missing source gate key: {key}")
        before = float(gates[key])
        base = float(base_gates.get(key, before))
        after = apply_operation(before, base, operation, shrink_scale=shrink_scale, delta_scale=delta_scale)
        gates[key] = after
        row = ledger_by_key.get(key, {})
        decision_rows.append(
            {
                "key": key,
                "param_name": row.get("param_name", key.rsplit("::", 1)[0] if "::" in key else key),
                "expert": row.get("expert", key.rsplit("::", 1)[1] if "::" in key else ""),
                "layer": row.get("layer", ""),
                "layer_band": row.get("layer_band", card.get("layer_band", "")),
                "module_family": row.get("module_family", card.get("module_family", "")),
                "archetype": row.get("archetype", card.get("archetype", "")),
                "routing_action": row.get("routing_action", card.get("routing_action", "")),
                "validation_priority": row.get("validation_priority", card.get("validation_priority", "")),
                "operation": operation,
                "before": before,
                "after": after,
                "base": base,
                "delta_before": before - base,
                "delta_after": after - base,
                "changed_by_intervention": abs(after - before) > 1e-12,
            }
        )

    candidate_id = f"card{int(safe_float(card['priority_rank'])):02d}_{card['card_id']}__{operation}"
    candidate_dir = output_root / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_candidate(card, operation, decision_rows, candidate_dir)
    payload = {
        "format": "rcrf_validation_card_intervention_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": candidate_id,
        "source_gate_variant": source_payload.get("variant", "v18_rcf_bc"),
        "source_gate_checkpoint": str(source_gate_path),
        "principle": {
            "unit": "validation-card representative residual rows",
            "rule": "apply a minimal isolated intervention to representative rows only",
            "purpose": "counterfactual validation of row-level attribution hypotheses",
        },
        "card": card,
        "operation": {
            "name": operation,
            "shrink_scale": shrink_scale,
            "delta_scale": delta_scale,
        },
        "summary": summary,
        "gates": gates,
        "decision_rows": decision_rows,
    }
    write_json(candidate_dir / "gates.json", payload)
    write_json(candidate_dir / "intervention_summary.json", summary)
    write_csv(candidate_dir / "decision_rows.csv", decision_rows, DECISION_ROW_FIELDS)
    (candidate_dir / "intervention_summary.md").write_text(render_candidate_markdown(summary), encoding="utf-8")
    return summary


def auto_operation(card: dict[str, str]) -> str:
    action = card.get("routing_action", "")
    if action in {"protect_behavior_anchor", "do_not_prune_without_counterfactual"}:
        return "shrink-coeff"
    if action in {"retain_with_behavior_constraint"}:
        return "half-delta"
    if action in {"retain_capability_delta", "retain_continuous_field", "behavior_guard", "keep_small_until_validated"}:
        return "drop-delta"
    return "drop-delta"


def apply_operation(before: float, base: float, operation: str, *, shrink_scale: float, delta_scale: float) -> float:
    if operation == "drop-delta":
        return base
    if operation == "half-delta":
        return base + delta_scale * (before - base)
    if operation == "shrink-coeff":
        return shrink_scale * before
    raise ValueError(f"Unsupported operation: {operation}")


def summarize_candidate(
    card: dict[str, str],
    operation: str,
    decision_rows: list[dict[str, Any]],
    candidate_dir: Path,
) -> dict[str, Any]:
    changed = [row for row in decision_rows if row["changed_by_intervention"]]
    return {
        "candidate_id": candidate_dir.name,
        "card_id": card["card_id"],
        "priority_rank": int(safe_float(card["priority_rank"])),
        "routing_action": card["routing_action"],
        "validation_type": card["validation_type"],
        "operation": operation,
        "gate_path": str(candidate_dir / "gates.json"),
        "summary_path": str(candidate_dir / "intervention_summary.json"),
        "row_count": len(decision_rows),
        "changed_rows": len(changed),
        "mean_before": avg(row["before"] for row in decision_rows),
        "mean_after": avg(row["after"] for row in decision_rows),
        "mean_base": avg(row["base"] for row in decision_rows),
        "mean_delta_before": avg(row["delta_before"] for row in decision_rows),
        "mean_delta_after": avg(row["delta_after"] for row in decision_rows),
        "hypothesis": card["hypothesis"],
        "success_criterion": card["success_criterion"],
        "failure_read": card["failure_read"],
        "representative_keys": [row["key"] for row in decision_rows],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# 2026-05-22 RCF-BC Validation Interventions",
        "",
        "## 目的",
        "",
        "这个文件把 validation cards 物化为可 bake 的最小 OP-VEC gate 干预。它不启动评测；每个候选只改一张卡片中的代表 residual rows，用于后续反事实验证。",
        "",
        "## 生成候选",
        "",
        "| candidate | card | operation | rows | changed | mean delta before | mean delta after |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in manifest["generated"]:
        lines.append(
            f"| `{row['candidate_id']}` | `{row['card_id']}` | `{row['operation']}` | "
            f"{row['row_count']} | {row['changed_rows']} | {row['mean_delta_before']:.6f} | "
            f"{row['mean_delta_after']:.6f} |"
        )
    lines.extend(["", "## 后续评测命令模板", ""])
    lines.append("对任一候选，先 bake，再跑 Tool/Memory quick；只有行为不过 guardrail 时再跑 Code hurt：")
    lines.extend(
        [
            "",
            "```bash",
            "PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python",
            "MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json",
            "CFG=configs/gated_grpo.yaml",
            "GATE=/path/to/validation_card_intervention/gates.json",
            "OUT=/tmp/shared-storage/OnPolicy/checkpoints/<candidate_id>",
            "$PY scripts/eval/opvec_bake_checkpoint.py --config $CFG --mode-manifest $MODE --gate-checkpoint $GATE --output $OUT",
            "```",
            "",
            "## 产物",
            "",
            f"- manifest: `{Path(manifest['output_dir']) / 'validation_interventions_manifest.json'}`",
            f"- output root: `{manifest['output_dir']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_candidate_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['candidate_id']}",
        "",
        "## Card",
        "",
        f"- card: `{summary['card_id']}`",
        f"- rank: `{summary['priority_rank']}`",
        f"- action: `{summary['routing_action']}`",
        f"- validation: `{summary['validation_type']}`",
        f"- operation: `{summary['operation']}`",
        "",
        "## Hypothesis",
        "",
        summary["hypothesis"],
        "",
        "## Success Criterion",
        "",
        summary["success_criterion"],
        "",
        "## Failure Read",
        "",
        summary["failure_read"],
        "",
        "## Changed Rows",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- changed: `{summary['changed_rows']}`",
        f"- mean delta before: `{summary['mean_delta_before']:.6f}`",
        f"- mean delta after: `{summary['mean_delta_after']:.6f}`",
        "",
    ]
    for key in summary["representative_keys"]:
        lines.append(f"- `{key}`")
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_gate_map(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("gates", payload)
    if not isinstance(raw, dict):
        raise TypeError("Gate payload must be a dict or contain dict-valued `gates`")
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, int | float)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def avg(values: Any) -> float:
    vals = [float(value) for value in values]
    return mean(vals) if vals else 0.0


if __name__ == "__main__":
    main()
