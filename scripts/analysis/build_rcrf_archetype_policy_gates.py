#!/usr/bin/env python3
"""Build an archetype-consistent RCF-BC gate checkpoint.

The input is an already-built RCF-BC gate plus conflict-cluster rows.  The
script applies a small semantic projection:

* keep clean positive Code repair deltas;
* keep negative Code-noise suppression;
* keep continuous source-conflict deltas;
* reset deltas that contradict their archetype, e.g. suppressing a residual
  that also supports Tool/Memory behavior;
* reset weak/uninformative and behavior-only recenter drift.

This is not a sweep or learned update.  It is a deterministic consistency
check over the residual attribution table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_BASE_GATES = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_SOURCE_GATES = ROOT / "contrast_gates" / "residual_capability_field_behavior_constraints_v18" / "gates.json"
DEFAULT_CLUSTER_ROWS = ROOT / "analysis" / "rcrf_conflict_clusters_20260522" / "conflict_cluster_rows.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "contrast_gates" / "rcrf_archetype_consistency_v19"


def main() -> None:
    args = parse_args()
    base_payload = load_json(Path(args.base_gates).expanduser())
    source_payload = load_json(Path(args.source_gates).expanduser())
    base_gates = extract_gates(base_payload)
    source_gates = extract_gates(source_payload)
    cluster_rows = load_jsonl(Path(args.cluster_rows).expanduser())
    cluster_by_key = {str(row["key"]): row for row in cluster_rows}
    if set(base_gates) != set(source_gates):
        raise ValueError("Base and source gate keys do not match")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gates: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    for key in sorted(source_gates):
        base = float(base_gates[key])
        source = float(source_gates[key])
        delta = source - base
        cluster = cluster_by_key.get(key, {})
        archetype = str(cluster.get("archetype") or "missing_cluster")
        projected_delta, reason = project_delta(archetype, delta)
        coefficient = base + projected_delta
        gates[key] = coefficient
        decisions.append(
            {
                "key": key,
                "param_name": key.rsplit("::", 1)[0] if "::" in key else key,
                "expert": key.rsplit("::", 1)[1] if "::" in key else "",
                "archetype": archetype,
                "role": cluster.get("role", ""),
                "layer": cluster.get("layer", -1),
                "layer_band": cluster.get("layer_band", ""),
                "module": cluster.get("module", ""),
                "module_family": cluster.get("module_family", ""),
                "base_coefficient": base,
                "source_coefficient": source,
                "coefficient": coefficient,
                "source_delta": delta,
                "delta": projected_delta,
                "projection_delta": projected_delta - delta,
                "reason": reason,
                "code_positive_sources": cluster.get("code_positive_sources", ""),
                "code_negative_sources": cluster.get("code_negative_sources", ""),
                "protected_support_tasks": cluster.get("protected_support_tasks", ""),
                "protected_harm_tasks": cluster.get("protected_harm_tasks", ""),
            }
        )

    summary = build_summary(base_gates, source_gates, gates, decisions)
    payload = {
        "format": "rcrf_archetype_policy_gate_checkpoint_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant_name,
        "principle": {
            "unit": "parameter-level OP-VEC residual coefficient",
            "source": "RCF-BC continuous capability field",
            "projection": "deterministic archetype consistency over residual conflict clusters",
            "not_a_sweep": True,
        },
        "config": {
            "base_gates": str(Path(args.base_gates).expanduser().resolve()),
            "source_gates": str(Path(args.source_gates).expanduser().resolve()),
            "cluster_rows": str(Path(args.cluster_rows).expanduser().resolve()),
            "policy": "archetype_consistency",
        },
        "gates": gates,
        "decision_rows": decisions,
        "summary": summary,
        "coefficient_summary": coefficient_summary(gates),
        "delta_summary": delta_summary(base_gates, gates),
    }
    write_json(output_dir / "gates.json", payload)
    write_jsonl(output_dir / "decision_rows.jsonl", decisions)
    write_json(output_dir / "archetype_policy_summary.json", summary)
    (output_dir / "archetype_policy_summary.md").write_text(render_markdown(summary, output_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "gates.json"),
                "changed": summary["projected"]["changed_count"],
                "reset_count": summary["projection"]["reset_count"],
                "summary": str(output_dir / "archetype_policy_summary.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gates", type=Path, default=DEFAULT_BASE_GATES)
    parser.add_argument("--source-gates", type=Path, default=DEFAULT_SOURCE_GATES)
    parser.add_argument("--cluster-rows", type=Path, default=DEFAULT_CLUSTER_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant-name", default="rcrf_archetype_consistency_v19")
    return parser.parse_args()


def project_delta(archetype: str, delta: float) -> tuple[float, str]:
    if abs(delta) <= 1e-12:
        return 0.0, "hold_zero_delta"
    if archetype == "clean_code_repair":
        if delta < 0.0:
            return 0.0, "reset_clean_repair_negative_delta"
        return delta, "keep_clean_repair_positive_delta"
    if archetype == "code_repair_with_behavior_harm":
        if delta < 0.0:
            return 0.0, "reset_repair_harm_negative_delta"
        return delta, "keep_soft_constrained_repair_delta"
    if archetype == "code_source_conflict":
        return delta, "keep_continuous_source_conflict_delta"
    if archetype == "code_negative_with_behavior_support":
        return 0.0, "reset_negative_delta_on_behavior_support"
    if archetype == "code_negative_noise":
        if delta > 0.0:
            return 0.0, "reset_negative_noise_positive_delta"
        return delta, "keep_negative_noise_suppression"
    if archetype in {"behavior_only", "weak_or_uninformative"}:
        return 0.0, f"reset_{archetype}_drift"
    return delta, "keep_missing_or_unknown_archetype_delta"


def build_summary(
    base_gates: dict[str, float],
    source_gates: dict[str, float],
    gates: dict[str, float],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    projection_rows = [row for row in decisions if abs(float(row["projection_delta"])) > 1e-12]
    return {
        "source": summarize_delta(base_gates, source_gates),
        "projected": summarize_delta(base_gates, gates),
        "projection": {
            "reset_count": len(projection_rows),
            "mean_abs_projection_delta": mean(abs(float(row["projection_delta"])) for row in projection_rows)
            if projection_rows
            else 0.0,
            "reason_counts": dict(Counter(str(row["reason"]) for row in projection_rows).most_common()),
            "archetype_counts": dict(Counter(str(row["archetype"]) for row in projection_rows).most_common()),
            "expert_counts": dict(Counter(str(row["expert"]) for row in projection_rows).most_common()),
            "top_projection_rows": compact_rows(
                sorted(projection_rows, key=lambda row: abs(float(row["projection_delta"])), reverse=True)[:30]
            ),
        },
        "projected_by_archetype": summarize_decisions(decisions, "archetype"),
        "projected_by_expert": summarize_decisions(decisions, "expert"),
        "projected_by_layer_module_expert": summarize_decisions(decisions, "layer_band", "module_family", "expert"),
    }


def summarize_delta(base: dict[str, float], gates: dict[str, float]) -> dict[str, Any]:
    deltas = [float(gates[key]) - float(base[key]) for key in sorted(base)]
    changed = [value for value in deltas if abs(value) > 1e-12]
    return {
        "count": len(deltas),
        "changed_count": len(changed),
        "positive_count": sum(1 for value in changed if value > 0.0),
        "negative_count": sum(1 for value in changed if value < 0.0),
        "mean": mean(deltas) if deltas else 0.0,
        "mean_abs": mean(abs(value) for value in deltas) if deltas else 0.0,
        "max_abs": max((abs(value) for value in deltas), default=0.0),
        "min": min(deltas) if deltas else 0.0,
        "max": max(deltas) if deltas else 0.0,
        "std": pstdev(deltas) if len(deltas) > 1 else 0.0,
    }


def summarize_decisions(decisions: list[dict[str, Any]], *group_keys: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item = {key: value for key, value in zip(group_keys, group)}
        deltas = [float(row["delta"]) for row in rows]
        changed = [value for value in deltas if abs(value) > 1e-12]
        projected = [row for row in rows if abs(float(row["projection_delta"])) > 1e-12]
        item.update(
            {
                "row_count": len(rows),
                "changed_count": len(changed),
                "positive_count": sum(1 for value in changed if value > 0.0),
                "negative_count": sum(1 for value in changed if value < 0.0),
                "reset_count": len(projected),
                "mean_delta": mean(deltas) if deltas else 0.0,
                "mean_abs_delta": mean(abs(value) for value in deltas) if deltas else 0.0,
            }
        )
        output.append(item)
    return output


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "key",
        "archetype",
        "role",
        "expert",
        "layer",
        "module",
        "source_delta",
        "delta",
        "projection_delta",
        "reason",
        "code_positive_sources",
        "code_negative_sources",
        "protected_support_tasks",
        "protected_harm_tasks",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def coefficient_summary(gates: dict[str, float]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        expert = key.rsplit("::", 1)[1] if "::" in key else ""
        grouped[expert].append(float(value))
    return {expert: describe(values) for expert, values in sorted(grouped.items())}


def delta_summary(base: dict[str, float], gates: dict[str, float]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    all_deltas = []
    for key, value in gates.items():
        expert = key.rsplit("::", 1)[1] if "::" in key else ""
        delta = float(value) - float(base[key])
        grouped[expert].append(delta)
        all_deltas.append(delta)
    return {"overall": describe_deltas(all_deltas), "by_expert": {k: describe_deltas(v) for k, v in sorted(grouped.items())}}


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def describe_deltas(values: list[float]) -> dict[str, Any]:
    result = describe(values)
    result.update(
        {
            "changed_count": sum(1 for value in values if abs(value) > 1e-12),
            "positive_count": sum(1 for value in values if value > 1e-12),
            "negative_count": sum(1 for value in values if value < -1e-12),
            "mean_abs": mean(abs(value) for value in values) if values else 0.0,
            "max_abs": max((abs(value) for value in values), default=0.0),
        }
    )
    return result


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    projection = summary["projection"]
    lines = [
        "# RCF-BC Archetype Consistency Projection",
        "",
        "## Summary",
        "",
        f"- Source changed rows: `{summary['source']['changed_count']}`",
        f"- Projected changed rows: `{summary['projected']['changed_count']}`",
        f"- Reset rows: `{projection['reset_count']}`",
        f"- Source mean abs delta: `{summary['source']['mean_abs']:.6f}`",
        f"- Projected mean abs delta: `{summary['projected']['mean_abs']:.6f}`",
        "",
        "## Reset Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in projection["reason_counts"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Archetype Summary", "", "| archetype | rows | changed | reset | + | - | mean abs delta |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in summary["projected_by_archetype"]:
        lines.append(
            f"| `{row.get('archetype')}` | {row.get('row_count')} | {row.get('changed_count')} | "
            f"{row.get('reset_count')} | {row.get('positive_count')} | {row.get('negative_count')} | "
            f"{float(row.get('mean_abs_delta', 0.0)):.6f} |"
        )
    lines.extend(["", "## Top Reset Rows", "", "| key | archetype | expert | source delta | projected delta | reason |", "|---|---|---|---:|---:|---|"])
    for row in projection["top_projection_rows"][:20]:
        lines.append(
            f"| `{row.get('key')}` | `{row.get('archetype')}` | {row.get('expert')} | "
            f"{float(row.get('source_delta', 0.0)):.6f} | {float(row.get('delta', 0.0)):.6f} | `{row.get('reason')}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Gate: `{output_dir / 'gates.json'}`",
            f"- Decision rows: `{output_dir / 'decision_rows.jsonl'}`",
            f"- Summary JSON: `{output_dir / 'archetype_policy_summary.json'}`",
            "",
            "## Interpretation",
            "",
            "This candidate tests whether an explicit archetype-consistency projection can remove residual drift "
            "without collapsing back to hard role routing. It should be evaluated as a mechanism ablation against "
            "`v18_rcf_bc`, not as a tuned replacement.",
            "",
        ]
    )
    return "\n".join(lines)


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
    raw = payload.get("gates", payload)
    if not isinstance(raw, dict):
        raise TypeError("Gate payload must be a dict or contain a dict-valued `gates` field")
    gates = {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}
    if not gates:
        raise ValueError("No numeric gates found")
    return gates


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
