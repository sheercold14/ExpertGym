#!/usr/bin/env python3
"""Build a conservative gate checkpoint from a residual evidence table.

The script materializes the simplest testable version of the attribution
framework:

* move only residual entries with aligned evidence (`keep_or_raise`/`suppress`);
* keep conflict and no-decision entries at the base coefficient;
* do not train, bake, evaluate, or change reward logic.

It is intentionally conservative so that any later evaluation can test whether
the evidence table identifies safe residual edits, not whether a tuned scalar
search found a lucky checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping


DEFAULT_BASE_GATES = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_EVIDENCE_ROWS = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/"
    "residual_evidence_table_20260521/residual_evidence_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/"
    "rcrf_evidence_routed_v1"
)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def main() -> None:
    args = parse_args()
    base_payload = load_json(args.base_gates)
    base_gates = extract_gate_map(base_payload)
    evidence_rows = load_evidence_rows(args.evidence_rows)

    gates, decision_rows = route_gates(
        base_gates=base_gates,
        evidence_rows=evidence_rows,
        max_delta=args.max_delta,
        min_abs_score=args.min_abs_score,
        min_coeff=args.min_coeff,
        max_coeff=args.max_coeff,
        allowed_recommendations=set(args.recommendation),
    )
    if args.preserve_expert_mean:
        gates = recenter_by_expert(
            gates=gates,
            base_gates=base_gates,
            min_coeff=args.min_coeff,
            max_coeff=args.max_coeff,
            passes=args.recenter_passes,
        )
        decision_rows = refresh_decision_deltas(decision_rows, gates=gates, base_gates=base_gates)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "evidence_routed_residual_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_gate_checkpoint": str(args.base_gates.expanduser().resolve()),
        "evidence_rows": str(args.evidence_rows.expanduser().resolve()),
        "mode_manifest": base_payload.get("mode_manifest"),
        "principle": {
            "unit": "parameter-level OP-VEC residual coefficient",
            "signal": "residual evidence table with source/span contrast and behavior utility",
            "rule": "raise/suppress only entries with non-conflicting evidence; leave conflict/no-decision entries at base",
            "purpose": "test whether residual-level attribution can avoid mode conflict without scalar sweeps",
        },
        "config": {
            "max_delta": args.max_delta,
            "min_abs_score": args.min_abs_score,
            "min_coeff": args.min_coeff,
            "max_coeff": args.max_coeff,
            "recommendation": args.recommendation,
            "preserve_expert_mean": args.preserve_expert_mean,
            "recenter_passes": args.recenter_passes,
        },
        "gates": gates,
        "coefficient_summary": coefficient_summary(gates),
        "delta_summary": delta_summary(base_gates, gates),
        "decision_summary": decision_summary(decision_rows),
        "decision_rows": sorted(decision_rows, key=lambda row: abs(float(row["delta"])), reverse=True),
    }
    write_json(output_dir / "gates.json", payload)
    write_jsonl(output_dir / "decision_rows.jsonl", payload["decision_rows"])
    (output_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "gates.json"),
                "changed": payload["delta_summary"]["overall"]["changed_count"],
                "decision_summary": payload["decision_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gates", type=Path, default=DEFAULT_BASE_GATES)
    parser.add_argument("--evidence-rows", type=Path, default=DEFAULT_EVIDENCE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-delta", type=float, default=0.04)
    parser.add_argument("--min-abs-score", type=float, default=0.25)
    parser.add_argument("--min-coeff", type=float, default=0.55)
    parser.add_argument("--max-coeff", type=float, default=1.12)
    parser.add_argument(
        "--recommendation",
        action="append",
        default=[],
        help="Recommendation allowlist to materialize. Defaults to keep_or_raise and suppress.",
    )
    parser.add_argument(
        "--preserve-expert-mean",
        action="store_true",
        default=False,
        help="Optional recentering to preserve each expert mean after routing. Default off to keep no-decision rows unchanged.",
    )
    parser.add_argument("--recenter-passes", type=int, default=3)
    args = parser.parse_args()
    if not args.recommendation:
        args.recommendation = ["keep_or_raise", "suppress"]
    return args


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_gate_map(payload: Mapping[str, Any]) -> dict[str, float]:
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("Expected gate checkpoint with dict field `gates`")
    return {str(key): safe_float(value) for key, value in gates.items()}


def load_evidence_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def route_gates(
    *,
    base_gates: dict[str, float],
    evidence_rows: list[dict[str, str]],
    max_delta: float,
    min_abs_score: float,
    min_coeff: float,
    max_coeff: float,
    allowed_recommendations: set[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gates = dict(base_gates)
    decision_rows: list[dict[str, Any]] = []
    for row in evidence_rows:
        recommendation = str(row.get("recommendation", ""))
        param_name = str(row.get("param_name", ""))
        expert = str(row.get("expert", ""))
        key = f"{param_name}::{expert}"
        if key not in gates:
            continue
        if recommendation not in allowed_recommendations:
            continue
        score = safe_float(row.get("code_mean_normalized_effect"))
        if abs(score) < min_abs_score:
            continue
        direction = 1.0 if recommendation == "keep_or_raise" else -1.0
        signed_delta = direction * max_delta * min(1.0, abs(score))
        before = gates[key]
        after = min(max(before + signed_delta, min_coeff), max_coeff)
        if abs(after - before) <= 1.0e-12:
            continue
        gates[key] = after
        decision_rows.append(
            {
                "key": key,
                "param_name": param_name,
                "expert": expert,
                "layer": layer_from_param(param_name),
                "module": module_from_param(param_name),
                "module_family": module_family(module_from_param(param_name)),
                "base_coefficient": before,
                "coefficient": after,
                "delta": after - before,
                "recommendation": recommendation,
                "recommendation_reason": row.get("recommendation_reason", ""),
                "code_state": row.get("code_state", ""),
                "code_positive_sources": row.get("code_positive_sources", ""),
                "code_negative_sources": row.get("code_negative_sources", ""),
                "code_mean_normalized_effect": score,
                "positive_utility_tasks": row.get("positive_utility_tasks", ""),
                "harm_utility_tasks": row.get("harm_utility_tasks", ""),
            }
        )
    return gates, decision_rows


def recenter_by_expert(
    *,
    gates: dict[str, float],
    base_gates: dict[str, float],
    min_coeff: float,
    max_coeff: float,
    passes: int,
) -> dict[str, float]:
    result = dict(gates)
    for _ in range(max(1, passes)):
        by_expert: dict[str, list[str]] = defaultdict(list)
        for key in result:
            by_expert[key.rsplit("::", 1)[-1]].append(key)
        for expert, keys in by_expert.items():
            base_mean = mean(base_gates[key] for key in keys if key in base_gates)
            current_mean = mean(result[key] for key in keys)
            offset = base_mean - current_mean
            for key in keys:
                result[key] = min(max(result[key] + offset, min_coeff), max_coeff)
    return result


def refresh_decision_deltas(
    decision_rows: list[dict[str, Any]], *, gates: dict[str, float], base_gates: dict[str, float]
) -> list[dict[str, Any]]:
    refreshed = []
    for row in decision_rows:
        key = row["key"]
        updated = dict(row)
        updated["coefficient"] = gates[key]
        updated["delta"] = gates[key] - base_gates[key]
        refreshed.append(updated)
    return refreshed


def coefficient_summary(gates: Mapping[str, float]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        expert = key.rsplit("::", 1)[-1]
        buckets[expert].append(float(value))
    return {
        expert: {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
        for expert, values in sorted(buckets.items())
    }


def delta_summary(base_gates: Mapping[str, float], gates: Mapping[str, float]) -> dict[str, Any]:
    by_expert: dict[str, list[float]] = defaultdict(list)
    all_deltas: list[float] = []
    for key, value in gates.items():
        delta = float(value) - float(base_gates.get(key, value))
        by_expert[key.rsplit("::", 1)[-1]].append(delta)
        all_deltas.append(delta)
    summary = {expert: summarize_deltas(values) for expert, values in sorted(by_expert.items())}
    summary["overall"] = summarize_deltas(all_deltas)
    return summary


def summarize_deltas(values: list[float]) -> dict[str, float]:
    changed = [value for value in values if abs(value) > 1.0e-12]
    return {
        "count": len(values),
        "changed_count": len(changed),
        "positive_count": sum(1 for value in changed if value > 0.0),
        "negative_count": sum(1 for value in changed if value < 0.0),
        "mean": mean(values) if values else 0.0,
        "mean_abs": mean(abs(value) for value in values) if values else 0.0,
        "max_abs": max((abs(value) for value in values), default=0.0),
    }


def decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "recommendation_counts": dict(Counter(str(row.get("recommendation", "")) for row in rows)),
        "expert_counts": dict(Counter(str(row.get("expert", "")) for row in rows)),
        "module_family_counts": dict(Counter(str(row.get("module_family", "")) for row in rows)),
    }


def render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Evidence-Routed Residual Gates",
        "",
        f"- base gates: `{payload['base_gate_checkpoint']}`",
        f"- evidence rows: `{payload['evidence_rows']}`",
        "",
        "## Config",
        "",
    ]
    for key, value in payload["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Decision Summary", ""])
    summary = payload["decision_summary"]
    lines.append(f"- decision rows: `{summary['count']}`")
    lines.extend(["", "### Recommendation counts", "", "| recommendation | count |", "|---|---:|"])
    for key, value in sorted(summary["recommendation_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "### Delta summary", "", "| expert | changed | positive | negative | mean abs | max abs |", "|---|---:|---:|---:|---:|---:|"])
    for expert, row in payload["delta_summary"].items():
        if expert == "overall":
            continue
        lines.append(
            f"| {expert} | {row['changed_count']} | {row['positive_count']} | {row['negative_count']} | "
            f"{row['mean_abs']:.4f} | {row['max_abs']:.4f} |"
        )
    overall = payload["delta_summary"]["overall"]
    lines.append(
        f"| overall | {overall['changed_count']} | {overall['positive_count']} | {overall['negative_count']} | "
        f"{overall['mean_abs']:.4f} | {overall['max_abs']:.4f} |"
    )
    lines.append("")
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


if __name__ == "__main__":
    main()
