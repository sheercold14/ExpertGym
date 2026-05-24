#!/usr/bin/env python3
"""Build a filtered coefficient ablation from RCF-BC conflict clusters.

This is a mechanical diagnostic tool.  It starts from an existing gate file,
selects rows by conflict-cluster metadata, and changes only those coefficients.
It does not read rewards, rerun probes, or learn any new parameters.
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
DEFAULT_SOURCE_GATES = ROOT / "contrast_gates" / "residual_capability_field_behavior_constraints_v18" / "gates.json"
DEFAULT_CLUSTER_ROWS = ROOT / "analysis" / "rcrf_conflict_clusters_20260522" / "conflict_cluster_rows.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "contrast_gates" / "rcrf_archetype_scaled_ablation"


def main() -> None:
    args = parse_args()
    if (args.scale_coefficient is None) == (args.set_coefficient is None):
        raise ValueError("Specify exactly one of --scale-coefficient or --set-coefficient")

    source_path = Path(args.source_gates).expanduser().resolve()
    cluster_path = Path(args.cluster_rows).expanduser().resolve()
    source_payload = load_json(source_path)
    source_gates = extract_gates(source_payload)
    cluster_by_key = {str(row["key"]): row for row in load_jsonl(cluster_path)}
    missing = sorted(set(source_gates) - set(cluster_by_key))
    if missing:
        raise ValueError(f"{len(missing)} gate rows are missing cluster metadata; first={missing[0]}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gates: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    for key in sorted(source_gates):
        before = float(source_gates[key])
        cluster = cluster_by_key[key]
        selected, selected_reasons = row_selected(cluster, args)
        if selected:
            after = (
                before * float(args.scale_coefficient)
                if args.scale_coefficient is not None
                else float(args.set_coefficient)
            )
            action = "scale_coefficient" if args.scale_coefficient is not None else "set_coefficient"
        else:
            after = before
            action = "unchanged"
        gates[key] = after
        decisions.append(
            {
                "key": key,
                "param_name": key.rsplit("::", 1)[0] if "::" in key else key,
                "expert": cluster.get("expert", key.rsplit("::", 1)[-1]),
                "archetype": cluster.get("archetype", ""),
                "role": cluster.get("role", ""),
                "layer": cluster.get("layer", -1),
                "layer_band": cluster.get("layer_band", ""),
                "module": cluster.get("module", ""),
                "module_family": cluster.get("module_family", ""),
                "before": before,
                "after": after,
                "delta": after - before,
                "selected": selected,
                "selected_reasons": selected_reasons,
                "action": action,
                "code_positive_sources": cluster.get("code_positive_sources", ""),
                "code_negative_sources": cluster.get("code_negative_sources", ""),
                "protected_support_tasks": cluster.get("protected_support_tasks", ""),
                "protected_harm_tasks": cluster.get("protected_harm_tasks", ""),
                "code_positive_strength": cluster.get("code_positive_strength", 0.0),
                "code_negative_strength": cluster.get("code_negative_strength", 0.0),
                "protected_max_harm_norm": cluster.get("protected_max_harm_norm", 0.0),
            }
        )

    summary = build_summary(
        args=args,
        source_path=source_path,
        cluster_path=cluster_path,
        source_gates=source_gates,
        gates=gates,
        decisions=decisions,
    )
    payload = {
        "format": "rcrf_archetype_scaled_gate_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant_name,
        "principle": {
            "unit": "parameter-level OP-VEC residual coefficient",
            "operation": "mechanically rewrite coefficients selected by residual conflict archetype",
            "purpose": "test whether task-level coefficient shrinkage should be localized to mechanism rows",
            "not_a_sweep": True,
        },
        "config": config_dict(args, source_path, cluster_path),
        "gates": gates,
        "decision_rows": decisions,
        "summary": summary,
        "coefficient_summary": coefficient_summary(gates),
    }
    write_json(output_dir / "gates.json", payload)
    write_jsonl(output_dir / "decision_rows.jsonl", decisions)
    write_json(output_dir / "archetype_scaled_summary.json", summary)
    (output_dir / "archetype_scaled_summary.md").write_text(render_markdown(summary, output_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "gates.json"),
                "summary": str(output_dir / "archetype_scaled_summary.md"),
                "selected_count": summary["selected_count"],
                "changed_count": summary["changed_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gates", type=Path, default=DEFAULT_SOURCE_GATES)
    parser.add_argument("--cluster-rows", type=Path, default=DEFAULT_CLUSTER_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant-name", default="rcrf_archetype_scaled_ablation")
    parser.add_argument("--archetype", action="append", default=[], help="Select these archetypes. Empty means all.")
    parser.add_argument("--expert", action="append", default=[], help="Select these experts. Empty means all.")
    parser.add_argument("--role", action="append", default=[], help="Select exact atlas roles. Empty means all.")
    parser.add_argument("--module-family", action="append", default=[], help="Select module families. Empty means all.")
    parser.add_argument("--layer-band", action="append", default=[], help="Select layer bands. Empty means all.")
    parser.add_argument("--min-code-negative-strength", type=float, default=None)
    parser.add_argument("--min-code-positive-strength", type=float, default=None)
    parser.add_argument("--min-protected-harm", type=float, default=None)
    parser.add_argument("--scale-coefficient", type=float, default=None)
    parser.add_argument("--set-coefficient", type=float, default=None)
    return parser.parse_args()


def row_selected(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    checks = [
        ("archetype", args.archetype, str(row.get("archetype", ""))),
        ("expert", args.expert, str(row.get("expert", ""))),
        ("role", args.role, str(row.get("role", ""))),
        ("module_family", args.module_family, str(row.get("module_family", ""))),
        ("layer_band", args.layer_band, str(row.get("layer_band", ""))),
    ]
    reasons: list[str] = []
    for name, allowlist, value in checks:
        if allowlist and value not in set(allowlist):
            return False, []
        if allowlist:
            reasons.append(f"{name}={value}")
    threshold_checks = [
        ("code_negative_strength", args.min_code_negative_strength),
        ("code_positive_strength", args.min_code_positive_strength),
        ("protected_max_harm_norm", args.min_protected_harm),
    ]
    for field, threshold in threshold_checks:
        if threshold is None:
            continue
        value = safe_float(row.get(field))
        if value < float(threshold):
            return False, []
        reasons.append(f"{field}>={threshold:g}")
    if not reasons:
        reasons.append("all_rows")
    return True, reasons


def build_summary(
    *,
    args: argparse.Namespace,
    source_path: Path,
    cluster_path: Path,
    source_gates: dict[str, float],
    gates: dict[str, float],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in decisions if row["selected"]]
    changed = [row for row in decisions if abs(float(row["delta"])) > 1e-12]
    return {
        "variant": args.variant_name,
        "source_gates": str(source_path),
        "cluster_rows": str(cluster_path),
        "config": config_dict(args, source_path, cluster_path),
        "row_count": len(decisions),
        "selected_count": len(selected),
        "changed_count": len(changed),
        "coefficient_before": describe([float(row["before"]) for row in selected]),
        "coefficient_after": describe([float(row["after"]) for row in selected]),
        "delta": describe_deltas([float(row["delta"]) for row in decisions]),
        "selected_by_archetype": summarize_decisions(selected, "archetype"),
        "selected_by_expert": summarize_decisions(selected, "expert"),
        "selected_by_layer_module_expert": summarize_decisions(selected, "layer_band", "module_family", "expert"),
        "selected_reason_counts": dict(Counter(reason for row in selected for reason in row["selected_reasons"]).most_common()),
        "top_changed_rows": compact_rows(sorted(changed, key=lambda row: abs(float(row["delta"])), reverse=True)[:30]),
        "source_coefficient_summary": coefficient_summary(source_gates),
        "coefficient_summary": coefficient_summary(gates),
    }


def config_dict(args: argparse.Namespace, source_path: Path, cluster_path: Path) -> dict[str, Any]:
    return {
        "source_gates": str(source_path),
        "cluster_rows": str(cluster_path),
        "archetype": list(args.archetype),
        "expert": list(args.expert),
        "role": list(args.role),
        "module_family": list(args.module_family),
        "layer_band": list(args.layer_band),
        "min_code_negative_strength": args.min_code_negative_strength,
        "min_code_positive_strength": args.min_code_positive_strength,
        "min_protected_harm": args.min_protected_harm,
        "scale_coefficient": args.scale_coefficient,
        "set_coefficient": args.set_coefficient,
    }


def summarize_decisions(rows: list[dict[str, Any]], *group_keys: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item = {key: value for key, value in zip(group_keys, group)}
        deltas = [float(row["delta"]) for row in group_rows]
        item.update(
            {
                "row_count": len(group_rows),
                "changed_count": sum(1 for value in deltas if abs(value) > 1e-12),
                "mean_before": mean(float(row["before"]) for row in group_rows),
                "mean_after": mean(float(row["after"]) for row in group_rows),
                "mean_delta": mean(deltas) if deltas else 0.0,
                "mean_abs_delta": mean(abs(value) for value in deltas) if deltas else 0.0,
            }
        )
        output.append(item)
    return output


def coefficient_summary(gates: dict[str, float]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        expert = key.rsplit("::", 1)[1] if "::" in key else ""
        grouped[expert].append(float(value))
    return {expert: describe(values) for expert, values in sorted(grouped.items())}


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "key",
        "archetype",
        "role",
        "expert",
        "layer",
        "module",
        "before",
        "after",
        "delta",
        "code_positive_sources",
        "code_negative_sources",
        "protected_support_tasks",
        "protected_harm_tasks",
    )
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


def extract_gates(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("gates", payload)
    if not isinstance(raw, dict):
        raise TypeError("Gate payload must be a dict or contain a dict-valued `gates` field")
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    config = summary["config"]
    lines = [
        f"# {summary['variant']}",
        "",
        "## Purpose",
        "",
        "Residual-level coefficient ablation. This tests whether the harmful part of a task vector can be localized by RCF-BC archetypes instead of shrinking a whole expert.",
        "",
        "## Config",
        "",
        f"- Source gates: `{summary['source_gates']}`",
        f"- Cluster rows: `{summary['cluster_rows']}`",
        f"- Archetypes: `{config['archetype']}`",
        f"- Experts: `{config['expert']}`",
        f"- Scale coefficient: `{config['scale_coefficient']}`",
        f"- Set coefficient: `{config['set_coefficient']}`",
        f"- Selected rows: `{summary['selected_count']}` / `{summary['row_count']}`",
        f"- Changed rows: `{summary['changed_count']}`",
        "",
        "## Selected Archetypes",
        "",
        "| archetype | rows | mean before | mean after | mean delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["selected_by_archetype"]:
        lines.append(
            f"| `{row.get('archetype')}` | {row.get('row_count')} | "
            f"{float(row.get('mean_before', 0.0)):.6f} | {float(row.get('mean_after', 0.0)):.6f} | "
            f"{float(row.get('mean_delta', 0.0)):.6f} |"
        )
    lines.extend(["", "## Selected Experts", "", "| expert | rows | mean before | mean after | mean delta |", "|---|---:|---:|---:|---:|"])
    for row in summary["selected_by_expert"]:
        lines.append(
            f"| `{row.get('expert')}` | {row.get('row_count')} | "
            f"{float(row.get('mean_before', 0.0)):.6f} | {float(row.get('mean_after', 0.0)):.6f} | "
            f"{float(row.get('mean_delta', 0.0)):.6f} |"
        )
    lines.extend(["", "## Top Changed Rows", "", "| key | archetype | expert | before | after | delta |", "|---|---|---|---:|---:|---:|"])
    for row in summary["top_changed_rows"][:20]:
        lines.append(
            f"| `{row.get('key')}` | `{row.get('archetype')}` | `{row.get('expert')}` | "
            f"{float(row.get('before', 0.0)):.6f} | {float(row.get('after', 0.0)):.6f} | "
            f"{float(row.get('delta', 0.0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Gate: `{output_dir / 'gates.json'}`",
            f"- Decision rows: `{output_dir / 'decision_rows.jsonl'}`",
            f"- Summary JSON: `{output_dir / 'archetype_scaled_summary.json'}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
