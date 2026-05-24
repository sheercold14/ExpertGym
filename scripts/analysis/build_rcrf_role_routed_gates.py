#!/usr/bin/env python3
"""Materialize role-routed RCRF gates from a residual conflict atlas.

This script is the routing counterpart of ``build_rcrf_conflict_atlas.py``.
It intentionally uses simple, auditable role rules instead of learned scalar
coefficients:

* raise clean Code repair and shared-positive residuals;
* softly raise Memory-conflict repair residuals;
* do not raise Tool-conflict repair residuals;
* optionally suppress Code-negative noise and protected-harm-only residuals;
* keep Code-negative but protected-support residuals unchanged.

The goal is to turn the atlas into a reproducible Pareto operating point while
keeping the method easy to inspect.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_ATLAS_ROWS = ROOT / "analysis" / "rcrf_conflict_atlas_20260522" / "residual_conflict_atlas_rows.jsonl"
DEFAULT_BASE_GATES = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_OUTPUT_DIR = ROOT / "contrast_gates" / "rcrf_role_routed_v12"


def main() -> None:
    args = parse_args()
    atlas_rows = load_jsonl(Path(args.atlas_rows).expanduser())
    base_payload = load_json(Path(args.base_gates).expanduser())
    base_gates = extract_gates(base_payload)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_scale = robust_scale(
        [
            safe_float(row.get("code_positive_strength"))
            for row in atlas_rows
            if str(row.get("role")) in POSITIVE_CODE_ROLES and safe_float(row.get("code_positive_strength")) > 0.0
        ],
        quantile=args.strength_quantile,
    )
    negative_scale = robust_scale(
        [
            safe_float(row.get("code_negative_strength"))
            for row in atlas_rows
            if str(row.get("role")) in NEGATIVE_CODE_ROLES and safe_float(row.get("code_negative_strength")) > 0.0
        ],
        quantile=args.strength_quantile,
    )
    harm_scale = robust_scale(
        [
            safe_float(row.get("protected_max_harm_norm"))
            for row in atlas_rows
            if str(row.get("role")) == "protected_harm_only" and safe_float(row.get("protected_max_harm_norm")) > 0.0
        ],
        quantile=args.strength_quantile,
    )

    rows_by_key = {residual_key(row): row for row in atlas_rows}
    gates: dict[str, float] = {}
    decision_rows: list[dict[str, Any]] = []
    for key in sorted(base_gates):
        base_coeff = safe_float(base_gates[key])
        atlas = rows_by_key.get(key)
        if atlas is None:
            coeff = clamp(base_coeff, args.min_coefficient, args.max_coefficient)
            delta = coeff - base_coeff
            decision = {
                "key": key,
                "param_name": key.rsplit("::", 1)[0] if "::" in key else key,
                "expert": key.rsplit("::", 1)[1] if "::" in key else "",
                "base_coefficient": base_coeff,
                "coefficient": coeff,
                "delta": delta,
                "role": "missing_atlas",
                "reason": "hold_missing_atlas",
                "metrics": {},
            }
        else:
            delta, reason, metrics = route_delta(
                atlas,
                max_delta=args.max_delta,
                positive_scale=positive_scale,
                negative_scale=negative_scale,
                harm_scale=harm_scale,
                tool_harm_positive_scale=args.tool_harm_positive_scale,
                memory_harm_positive_scale=args.memory_harm_positive_scale,
                mixed_harm_positive_scale=args.mixed_harm_positive_scale,
                code_negative_action=args.code_negative_action,
                protected_harm_action=args.protected_harm_action,
                source_conflict_action=args.source_conflict_action,
                source_conflict_min_strength=args.source_conflict_min_strength,
                source_conflict_dominance_ratio=args.source_conflict_dominance_ratio,
                source_conflict_protected_support_action=args.source_conflict_protected_support_action,
            )
            coeff = clamp(base_coeff + delta, args.min_coefficient, args.max_coefficient)
            delta = coeff - base_coeff
            decision = {
                "key": key,
                "param_name": atlas.get("param_name", key.rsplit("::", 1)[0] if "::" in key else key),
                "expert": atlas.get("expert", key.rsplit("::", 1)[1] if "::" in key else ""),
                "layer": atlas.get("layer", -1),
                "layer_band": atlas.get("layer_band", ""),
                "module": atlas.get("module", ""),
                "module_family": atlas.get("module_family", ""),
                "base_coefficient": base_coeff,
                "coefficient": coeff,
                "delta": delta,
                "role": atlas.get("role", ""),
                "reason": reason,
                "metrics": metrics,
            }
        gates[key] = coeff
        decision_rows.append(decision)

    summary = build_summary(
        decision_rows=decision_rows,
        positive_scale=positive_scale,
        negative_scale=negative_scale,
        harm_scale=harm_scale,
        args=args,
    )
    payload = {
        "format": "rcrf_role_routed_gate_checkpoint_v1",
        "created_at": now_iso(),
        "variant": args.variant_name,
        "principle": {
            "atlas": "route by residual role from Code pass/fail and Tool/Memory behavior utility/harm",
            "raise": "clean Code repair and shared-positive residuals",
            "protect": "Tool harm is near-hard; Memory harm is soft",
            "suppress": "optional suppression for Code-negative noise and protected-harm-only residuals",
        },
        "base_gates": str(Path(args.base_gates).expanduser()),
        "atlas_rows": str(Path(args.atlas_rows).expanduser()),
        "routing_config": vars(args),
        "routing_scales": {
            "positive_scale": positive_scale,
            "negative_scale": negative_scale,
            "harm_scale": harm_scale,
        },
        "decision_summary": summary,
        "coefficient_summary": coefficient_summary(decision_rows),
        "gates": gates,
        "decision_rows": decision_rows,
    }
    write_json(output_dir / "gates.json", payload)
    write_json(output_dir / "role_routing_summary.json", summary)
    (output_dir / "role_routing_summary.md").write_text(render_markdown(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))


POSITIVE_CODE_ROLES = {
    "code_repair_only",
    "shared_positive",
    "code_repair_vs_protected_harm",
    "code_repair_shared_and_harm",
}
NEGATIVE_CODE_ROLES = {"code_negative_noise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-rows", default=str(DEFAULT_ATLAS_ROWS))
    parser.add_argument("--base-gates", default=str(DEFAULT_BASE_GATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variant-name", default="rcrf_role_routed_v12")
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--min-coefficient", type=float, default=0.55)
    parser.add_argument("--max-coefficient", type=float, default=1.12)
    parser.add_argument("--strength-quantile", type=float, default=0.9)
    parser.add_argument(
        "--tool-harm-positive-scale",
        type=float,
        default=0.0,
        help="Scale for positive Code deltas when the same residual has protected Tool harm.",
    )
    parser.add_argument(
        "--memory-harm-positive-scale",
        type=float,
        default=0.5,
        help="Scale for positive Code deltas when the same residual has protected Memory harm but no Tool harm.",
    )
    parser.add_argument(
        "--mixed-harm-positive-scale",
        type=float,
        default=0.25,
        help="Fallback scale when both Tool and Memory harm appear and Tool-specific scale is not decisive.",
    )
    parser.add_argument(
        "--code-negative-action",
        choices=("suppress", "hold"),
        default="suppress",
        help="Action for `code_negative_noise`. Default preserves the v12 behavior.",
    )
    parser.add_argument(
        "--protected-harm-action",
        choices=("suppress", "hold"),
        default="suppress",
        help="Action for `protected_harm_only`. Default preserves the v12 behavior.",
    )
    parser.add_argument(
        "--source-conflict-action",
        choices=("hold", "suppress-dominant", "route-dominant"),
        default="hold",
        help=(
            "Action for `code_source_conflict*` rows. Default preserves old behavior. "
            "`suppress-dominant` only suppresses rows whose negative source evidence dominates; "
            "`route-dominant` also raises rows whose positive source evidence dominates."
        ),
    )
    parser.add_argument(
        "--source-conflict-min-strength",
        type=float,
        default=1.0,
        help="Minimum source strength needed before routing a source-conflict row.",
    )
    parser.add_argument(
        "--source-conflict-dominance-ratio",
        type=float,
        default=1.25,
        help="Required dominant/opposing source strength ratio for source-conflict routing.",
    )
    parser.add_argument(
        "--source-conflict-protected-support-action",
        choices=("hold", "allow"),
        default="hold",
        help="Whether suppression may lower rows that also support Tool/Memory behavior.",
    )
    return parser.parse_args()


def route_delta(
    row: dict[str, Any],
    *,
    max_delta: float,
    positive_scale: float,
    negative_scale: float,
    harm_scale: float,
    tool_harm_positive_scale: float,
    memory_harm_positive_scale: float,
    mixed_harm_positive_scale: float,
    code_negative_action: str,
    protected_harm_action: str,
    source_conflict_action: str,
    source_conflict_min_strength: float,
    source_conflict_dominance_ratio: float,
    source_conflict_protected_support_action: str,
) -> tuple[float, str, dict[str, Any]]:
    role = str(row.get("role") or "")
    positive_strength = safe_float(row.get("code_positive_strength"))
    negative_strength = safe_float(row.get("code_negative_strength"))
    harm_strength = safe_float(row.get("protected_max_harm_norm"))
    harm_tasks = split_tasks(row.get("protected_harm_tasks"))
    support_tasks = split_tasks(row.get("protected_support_tasks"))
    metrics = {
        "code_positive_strength": positive_strength,
        "code_negative_strength": negative_strength,
        "protected_max_harm_norm": harm_strength,
        "protected_max_utility_norm": safe_float(row.get("protected_max_utility_norm")),
        "protected_harm_tasks": sorted(harm_tasks),
        "protected_support_tasks": sorted(support_tasks),
    }
    metrics["code_negative_action"] = code_negative_action
    metrics["protected_harm_action"] = protected_harm_action
    metrics["source_conflict_action"] = source_conflict_action
    metrics["source_conflict_min_strength"] = source_conflict_min_strength
    metrics["source_conflict_dominance_ratio"] = source_conflict_dominance_ratio
    metrics["source_conflict_protected_support_action"] = source_conflict_protected_support_action

    if role in {"code_repair_only", "shared_positive"}:
        return positive_delta(positive_strength, max_delta, positive_scale), f"raise_{role}", metrics
    if role in {"code_repair_vs_protected_harm", "code_repair_shared_and_harm"}:
        scale = harm_positive_scale(
            harm_tasks,
            tool_harm_positive_scale=tool_harm_positive_scale,
            memory_harm_positive_scale=memory_harm_positive_scale,
            mixed_harm_positive_scale=mixed_harm_positive_scale,
        )
        metrics["harm_positive_scale"] = scale
        return positive_delta(positive_strength, max_delta, positive_scale) * scale, f"soft_raise_{role}", metrics
    if role == "code_negative_noise":
        if code_negative_action == "hold":
            return 0.0, "hold_code_negative_noise", metrics
        return -positive_delta(negative_strength, max_delta, negative_scale), "suppress_code_negative_noise", metrics
    if role == "protected_harm_only":
        if protected_harm_action == "hold":
            return 0.0, "hold_protected_harm_only", metrics
        return -positive_delta(harm_strength, max_delta, harm_scale), "suppress_protected_harm_only", metrics
    if role == "protected_support_only":
        return 0.0, "hold_protected_support_only", metrics
    if role == "code_negative_but_protected_support":
        return 0.0, "hold_code_negative_but_protected_support", metrics
    if role.startswith("code_source_conflict"):
        return route_source_conflict_delta(
            row,
            role=role,
            max_delta=max_delta,
            positive_scale=positive_scale,
            negative_scale=negative_scale,
            tool_harm_positive_scale=tool_harm_positive_scale,
            memory_harm_positive_scale=memory_harm_positive_scale,
            mixed_harm_positive_scale=mixed_harm_positive_scale,
            source_conflict_action=source_conflict_action,
            source_conflict_min_strength=source_conflict_min_strength,
            source_conflict_dominance_ratio=source_conflict_dominance_ratio,
            source_conflict_protected_support_action=source_conflict_protected_support_action,
            metrics=metrics,
        )
    return 0.0, f"hold_{role or 'unknown'}", metrics


def route_source_conflict_delta(
    row: dict[str, Any],
    *,
    role: str,
    max_delta: float,
    positive_scale: float,
    negative_scale: float,
    tool_harm_positive_scale: float,
    memory_harm_positive_scale: float,
    mixed_harm_positive_scale: float,
    source_conflict_action: str,
    source_conflict_min_strength: float,
    source_conflict_dominance_ratio: float,
    source_conflict_protected_support_action: str,
    metrics: dict[str, Any],
) -> tuple[float, str, dict[str, Any]]:
    if source_conflict_action == "hold":
        return 0.0, f"hold_{role}", metrics

    positive_strength = safe_float(row.get("code_positive_strength"))
    negative_strength = safe_float(row.get("code_negative_strength"))
    support_tasks = split_tasks(row.get("protected_support_tasks"))
    harm_tasks = split_tasks(row.get("protected_harm_tasks"))
    ratio = max(float(source_conflict_dominance_ratio), 1.0)
    min_strength = max(float(source_conflict_min_strength), 0.0)
    eps = 1e-12

    metrics["source_positive_sources"] = split_csv(row.get("code_positive_sources"))
    metrics["source_negative_sources"] = split_csv(row.get("code_negative_sources"))
    metrics["source_positive_strength"] = positive_strength
    metrics["source_negative_strength"] = negative_strength
    metrics["source_negative_over_positive"] = negative_strength / (positive_strength + eps)
    metrics["source_positive_over_negative"] = positive_strength / (negative_strength + eps)

    negative_dominates = negative_strength >= min_strength and negative_strength >= ratio * max(positive_strength, eps)
    if negative_dominates:
        if support_tasks and source_conflict_protected_support_action == "hold":
            return 0.0, f"hold_{role}_negative_dominant_protected_support", metrics
        return -positive_delta(negative_strength, max_delta, negative_scale), f"suppress_{role}_negative_dominant", metrics

    if source_conflict_action != "route-dominant":
        return 0.0, f"hold_{role}_not_negative_dominant", metrics

    positive_dominates = positive_strength >= min_strength and positive_strength >= ratio * max(negative_strength, eps)
    if not positive_dominates:
        return 0.0, f"hold_{role}_no_dominant_source", metrics

    scale = 1.0
    if harm_tasks:
        scale = harm_positive_scale(
            harm_tasks,
            tool_harm_positive_scale=tool_harm_positive_scale,
            memory_harm_positive_scale=memory_harm_positive_scale,
            mixed_harm_positive_scale=mixed_harm_positive_scale,
        )
    metrics["source_conflict_positive_harm_scale"] = scale
    return positive_delta(positive_strength, max_delta, positive_scale) * scale, f"raise_{role}_positive_dominant", metrics


def positive_delta(strength: float, max_delta: float, scale: float) -> float:
    if strength <= 0.0 or max_delta <= 0.0:
        return 0.0
    return max_delta * min(strength / max(scale, 1e-12), 1.0)


def harm_positive_scale(
    harm_tasks: set[str],
    *,
    tool_harm_positive_scale: float,
    memory_harm_positive_scale: float,
    mixed_harm_positive_scale: float,
) -> float:
    if "tool" in harm_tasks and "memory" in harm_tasks:
        return min(tool_harm_positive_scale, mixed_harm_positive_scale)
    if "tool" in harm_tasks:
        return tool_harm_positive_scale
    if "memory" in harm_tasks:
        return memory_harm_positive_scale
    return mixed_harm_positive_scale


def build_summary(
    *,
    decision_rows: list[dict[str, Any]],
    positive_scale: float,
    negative_scale: float,
    harm_scale: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    role_counts = Counter(str(row.get("role")) for row in decision_rows)
    reason_counts = Counter(str(row.get("reason")) for row in decision_rows)
    return {
        "format": "rcrf_role_routing_summary_v1",
        "variant": args.variant_name,
        "row_count": len(decision_rows),
        "changed_count": sum(1 for row in decision_rows if abs(safe_float(row.get("delta"))) > 1e-12),
        "positive_delta_count": sum(1 for row in decision_rows if safe_float(row.get("delta")) > 1e-12),
        "negative_delta_count": sum(1 for row in decision_rows if safe_float(row.get("delta")) < -1e-12),
        "mean_abs_delta": mean(abs(safe_float(row.get("delta"))) for row in decision_rows) if decision_rows else 0.0,
        "routing_scales": {
            "positive_scale": positive_scale,
            "negative_scale": negative_scale,
            "harm_scale": harm_scale,
            "max_delta": args.max_delta,
            "strength_quantile": args.strength_quantile,
            "tool_harm_positive_scale": args.tool_harm_positive_scale,
            "memory_harm_positive_scale": args.memory_harm_positive_scale,
            "mixed_harm_positive_scale": args.mixed_harm_positive_scale,
            "code_negative_action": args.code_negative_action,
            "protected_harm_action": args.protected_harm_action,
            "source_conflict_action": args.source_conflict_action,
            "source_conflict_min_strength": args.source_conflict_min_strength,
            "source_conflict_dominance_ratio": args.source_conflict_dominance_ratio,
            "source_conflict_protected_support_action": args.source_conflict_protected_support_action,
        },
        "role_counts": dict(role_counts),
        "reason_counts": dict(reason_counts),
        "delta_by_role": delta_by_group(decision_rows, "role"),
        "delta_by_expert": delta_by_group(decision_rows, "expert"),
        "delta_by_layer_band": delta_by_group(decision_rows, "layer_band"),
        "top_positive": top_rows(decision_rows, reverse=True),
        "top_negative": top_rows(decision_rows, reverse=False),
    }


def coefficient_summary(decision_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in decision_rows:
        buckets[str(row.get("expert"))].append(safe_float(row.get("coefficient")))
    return {key: stats(values) for key, values in sorted(buckets.items())}


def delta_by_group(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(group_key, ""))].append(safe_float(row.get("delta")))
    return {key: delta_stats(values) for key, values in sorted(buckets.items())}


def delta_stats(values: list[float]) -> dict[str, float]:
    changed = [value for value in values if abs(value) > 1e-12]
    return {
        "row_count": float(len(values)),
        "changed_count": float(len(changed)),
        "positive_count": float(sum(1 for value in changed if value > 0.0)),
        "negative_count": float(sum(1 for value in changed if value < 0.0)),
        "mean_delta": mean(values) if values else 0.0,
        "mean_abs_delta": mean(abs(value) for value in values) if values else 0.0,
    }


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {"count": float(len(values)), "mean": mean(values), "min": min(values), "max": max(values)}


def top_rows(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: safe_float(row.get("delta")), reverse=reverse)
    return [
        {
            "key": row.get("key"),
            "expert": row.get("expert"),
            "layer": row.get("layer"),
            "module_family": row.get("module_family"),
            "role": row.get("role"),
            "reason": row.get("reason"),
            "delta": row.get("delta"),
            "coefficient": row.get("coefficient"),
            "metrics": row.get("metrics", {}),
        }
        for row in sorted_rows[:20]
        if (safe_float(row.get("delta")) > 1e-12 if reverse else safe_float(row.get("delta")) < -1e-12)
    ]


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# RCRF Role-Routed Gate Summary",
        "",
        f"- output_dir: `{output_dir}`",
        f"- variant: `{summary['variant']}`",
        f"- changed: `{summary['changed_count']}/{summary['row_count']}`",
        f"- positive_delta_count: `{summary['positive_delta_count']}`",
        f"- negative_delta_count: `{summary['negative_delta_count']}`",
        f"- mean_abs_delta: `{safe_float(summary['mean_abs_delta']):.6f}`",
        "",
        "## Routing Scales",
        "",
        "| key | value |",
        "|---|---:|",
    ]
    for key, value in summary["routing_scales"].items():
        if isinstance(value, (int, float)):
            rendered_value = f"{safe_float(value):.6f}"
        else:
            rendered_value = f"`{value}`"
        lines.append(f"| {key} | {rendered_value} |")
    lines.extend(["", "## Delta by Role", "", "| role | changed | + | - | mean_delta | mean_abs_delta |", "|---|---:|---:|---:|---:|---:|"])
    for role, stats_map in sorted(summary["delta_by_role"].items()):
        lines.append(
            f"| {role} | {safe_int(stats_map.get('changed_count'))} | {safe_int(stats_map.get('positive_count'))} | "
            f"{safe_int(stats_map.get('negative_count'))} | {safe_float(stats_map.get('mean_delta')):.6f} | "
            f"{safe_float(stats_map.get('mean_abs_delta')):.6f} |"
        )
    lines.extend(["", "## Delta by Expert", "", "| expert | changed | + | - | mean_delta | mean_abs_delta |", "|---|---:|---:|---:|---:|---:|"])
    for expert, stats_map in sorted(summary["delta_by_expert"].items()):
        lines.append(
            f"| {expert} | {safe_int(stats_map.get('changed_count'))} | {safe_int(stats_map.get('positive_count'))} | "
            f"{safe_int(stats_map.get('negative_count'))} | {safe_float(stats_map.get('mean_delta')):.6f} | "
            f"{safe_float(stats_map.get('mean_abs_delta')):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append("- This gate is not a sweep result; it is a direct materialization of atlas roles.")
    lines.append("- Clean Code repair and shared-positive residuals are raised.")
    lines.append("- Tool-conflict residuals are not raised; Memory-conflict residuals are softly raised.")
    lines.append(f"- Code-negative noise action: `{summary['routing_scales']['code_negative_action']}`.")
    lines.append(f"- Protected-harm-only action: `{summary['routing_scales']['protected_harm_action']}`.")
    lines.append("- Code-negative but protected-support residuals are held instead of suppressed.")
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_gates(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("gates") or payload.get("final_gates") or payload
    return {str(key): safe_float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def residual_key(row: dict[str, Any]) -> str:
    return f"{row.get('param_name')}::{row.get('expert')}"


def split_tasks(value: Any) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def robust_scale(values: list[float], *, quantile: float) -> float:
    positives = sorted(value for value in values if value > 0.0)
    if not positives:
        return 1.0
    q = max(0.0, min(1.0, quantile))
    index = min(len(positives) - 1, int(round(q * (len(positives) - 1))))
    return max(positives[index], 1e-12)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
