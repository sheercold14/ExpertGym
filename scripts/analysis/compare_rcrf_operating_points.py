#!/usr/bin/env python3
"""Compare RCRF operating points against a residual conflict atlas.

The script aligns gate checkpoints by ``(param_name, expert)`` and reports
which residual decisions are kept, dropped, or changed between operating
points.  It is diagnostic only: it does not generate gates, bake models, or run
evaluation.
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
DEFAULT_ATLAS_ROWS = ROOT / "analysis" / "rcrf_conflict_atlas_20260522" / "residual_conflict_atlas_rows.jsonl"
DEFAULT_BASE_GATES = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_operating_point_compare_20260522"
DEFAULT_GATES = {
    "v9": ROOT / "contrast_gates" / "rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9" / "gates.json",
    "v13": ROOT / "contrast_gates" / "rcrf_role_routed_positive_only_v13" / "gates.json",
    "v16": ROOT / "contrast_gates" / "rcrf_source_conflict_suppress_v16" / "gates.json",
    "v17": ROOT / "contrast_gates" / "rcrf_source_conflict_route_v17" / "gates.json",
    "v18_rcf_bc": ROOT
    / "contrast_gates"
    / "residual_capability_field_behavior_constraints_v18"
    / "gates.json",
    "v19_archetype_consistency": ROOT / "contrast_gates" / "rcrf_archetype_consistency_v19" / "gates.json",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_rows = load_jsonl(Path(args.atlas_rows).expanduser())
    base_gates = extract_gates(load_json(Path(args.base_gates).expanduser()))
    candidate_paths = parse_gate_args(args.gate)
    candidate_gates = {name: extract_gates(load_json(path.expanduser())) for name, path in candidate_paths.items()}
    ensure_aligned(atlas_rows, base_gates, candidate_gates)

    rows = build_rows(atlas_rows, base_gates, candidate_gates)
    reference = args.reference
    if reference not in candidate_gates:
        raise ValueError(f"--reference must be one of {sorted(candidate_gates)}")
    compares = args.compare or [name for name in candidate_gates if name != reference]

    summary = {
        "format": "rcrf_operating_point_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "atlas_rows": str(Path(args.atlas_rows).expanduser()),
        "base_gates": str(Path(args.base_gates).expanduser()),
        "reference": reference,
        "compares": compares,
        "candidate_paths": {name: str(path) for name, path in candidate_paths.items()},
        "candidate_overview": candidate_overview(rows, candidate_gates),
        "reference_gap_summary": {
            name: reference_gap_summary(rows, reference=reference, candidate=name, gap_threshold=args.gap_threshold)
            for name in compares
        },
    }

    write_json(output_dir / "operating_point_comparison_summary.json", summary)
    write_jsonl(output_dir / "operating_point_rows.jsonl", rows)
    write_csv(output_dir / "delta_by_role.csv", delta_by_group(rows, ["role"], candidate_gates))
    write_csv(output_dir / "delta_by_role_expert.csv", delta_by_group(rows, ["role", "expert"], candidate_gates))
    write_csv(output_dir / "delta_by_layer_module_expert.csv", delta_by_group(rows, ["layer_band", "module_family", "expert"], candidate_gates))
    write_csv(output_dir / "reference_lost_by_role.csv", lost_by_group(rows, reference, compares, ["role"]))
    write_csv(output_dir / "reference_lost_by_source_pattern.csv", lost_by_group(rows, reference, compares, ["role", "code_positive_sources", "code_negative_sources"]))
    (output_dir / "operating_point_comparison.md").write_text(render_markdown(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "operating_point_comparison_summary.json")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-rows", type=Path, default=DEFAULT_ATLAS_ROWS)
    parser.add_argument("--base-gates", type=Path, default=DEFAULT_BASE_GATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gate",
        nargs=2,
        action="append",
        metavar=("NAME", "GATES_JSON"),
        help="Candidate gate checkpoint. Defaults to v9/v13/v16/v17/v18_rcf_bc/v19_archetype_consistency.",
    )
    parser.add_argument("--reference", default="v9")
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--gap-threshold", type=float, default=0.03)
    return parser.parse_args()


def parse_gate_args(raw: list[list[str]] | None) -> dict[str, Path]:
    if not raw:
        return dict(DEFAULT_GATES)
    return {name: Path(path) for name, path in raw}


def build_rows(
    atlas_rows: list[dict[str, Any]],
    base_gates: dict[str, float],
    candidate_gates: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atlas in atlas_rows:
        key = residual_key(atlas)
        base = float(base_gates[key])
        row = {
            "key": key,
            "param_name": atlas.get("param_name", ""),
            "expert": atlas.get("expert", ""),
            "layer": safe_int(atlas.get("layer")),
            "layer_band": atlas.get("layer_band", ""),
            "module": atlas.get("module", ""),
            "module_family": atlas.get("module_family", ""),
            "role": atlas.get("role", ""),
            "code_positive_sources": atlas.get("code_positive_sources", ""),
            "code_negative_sources": atlas.get("code_negative_sources", ""),
            "code_positive_strength": safe_float(atlas.get("code_positive_strength")),
            "code_negative_strength": safe_float(atlas.get("code_negative_strength")),
            "protected_support_tasks": atlas.get("protected_support_tasks", ""),
            "protected_harm_tasks": atlas.get("protected_harm_tasks", ""),
            "protected_max_harm_norm": safe_float(atlas.get("protected_max_harm_norm")),
            "base_coefficient": base,
        }
        for name, gates in candidate_gates.items():
            coeff = float(gates[key])
            delta = coeff - base
            row[f"{name}_coefficient"] = coeff
            row[f"{name}_delta"] = delta
            row[f"{name}_sign"] = sign(delta)
        rows.append(row)
    rows.sort(key=lambda row: (str(row["role"]), str(row["expert"]), int(row["layer"]), str(row["module"])))
    return rows


def candidate_overview(rows: list[dict[str, Any]], candidate_gates: dict[str, dict[str, float]]) -> dict[str, Any]:
    overview = {}
    for name in candidate_gates:
        deltas = [safe_float(row.get(f"{name}_delta")) for row in rows]
        changed = [value for value in deltas if abs(value) > 1e-12]
        overview[name] = {
            "changed_count": len(changed),
            "positive_count": sum(1 for value in changed if value > 0.0),
            "negative_count": sum(1 for value in changed if value < 0.0),
            "mean_delta": mean(deltas) if deltas else 0.0,
            "mean_abs_delta": mean(abs(value) for value in deltas) if deltas else 0.0,
            "delta_by_expert": summarize_by(rows, [f"{name}_delta"], ["expert"])[f"{name}_delta"],
        }
    return overview


def reference_gap_summary(rows: list[dict[str, Any]], *, reference: str, candidate: str, gap_threshold: float) -> dict[str, Any]:
    ref_key = f"{reference}_delta"
    cand_key = f"{candidate}_delta"
    ref_changed = [row for row in rows if abs(safe_float(row.get(ref_key))) > 1e-12]
    lost = [row for row in ref_changed if abs(safe_float(row.get(cand_key))) <= 1e-12]
    sign_mismatch = [
        row
        for row in rows
        if sign(safe_float(row.get(ref_key))) != "zero"
        and sign(safe_float(row.get(cand_key))) != "zero"
        and sign(safe_float(row.get(ref_key))) != sign(safe_float(row.get(cand_key)))
    ]
    big_gap = [row for row in rows if abs(safe_float(row.get(ref_key)) - safe_float(row.get(cand_key))) >= gap_threshold]
    return {
        "reference_changed_count": len(ref_changed),
        "lost_count": len(lost),
        "sign_mismatch_count": len(sign_mismatch),
        "big_gap_count": len(big_gap),
        "lost_by_role": counter_dict(row.get("role") for row in lost),
        "lost_by_layer_module_expert": counter_dict(
            (row.get("layer_band"), row.get("module_family"), row.get("expert")) for row in lost
        ),
        "big_gap_by_role": counter_dict(row.get("role") for row in big_gap),
        "top_lost_rows": compact_rows(lost[:30], reference=reference, candidate=candidate),
        "top_big_gap_rows": compact_rows(sorted(big_gap, key=lambda row: abs(safe_float(row.get(ref_key)) - safe_float(row.get(cand_key))), reverse=True)[:30], reference=reference, candidate=candidate),
    }


def delta_by_group(rows: list[dict[str, Any]], group_keys: list[str], candidate_gates: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        base = {key: value for key, value in zip(group_keys, group)}
        base["row_count"] = len(group_rows)
        for name in candidate_gates:
            values = [safe_float(row.get(f"{name}_delta")) for row in group_rows]
            changed = [value for value in values if abs(value) > 1e-12]
            base[f"{name}_changed"] = len(changed)
            base[f"{name}_positive"] = sum(1 for value in changed if value > 0.0)
            base[f"{name}_negative"] = sum(1 for value in changed if value < 0.0)
            base[f"{name}_mean_delta"] = mean(values) if values else 0.0
            base[f"{name}_mean_abs_delta"] = mean(abs(value) for value in values) if values else 0.0
        output.append(base)
    return output


def lost_by_group(rows: list[dict[str, Any]], reference: str, candidates: list[str], group_keys: list[str]) -> list[dict[str, Any]]:
    ref_key = f"{reference}_delta"
    grouped: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_counts: Counter[tuple[Any, ...]] = Counter()
    ref_changed_counts: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        group = tuple(row.get(key, "") for key in group_keys)
        row_counts[group] += 1
        if abs(safe_float(row.get(ref_key))) <= 1e-12:
            continue
        ref_changed_counts[group] += 1
        for candidate in candidates:
            if abs(safe_float(row.get(f"{candidate}_delta"))) <= 1e-12:
                grouped[group][candidate] += 1
    output = []
    for group in sorted(row_counts, key=lambda item: tuple(str(x) for x in item)):
        row = {key: value for key, value in zip(group_keys, group)}
        row["row_count"] = row_counts[group]
        row[f"{reference}_changed"] = ref_changed_counts[group]
        for candidate in candidates:
            row[f"{candidate}_lost_from_{reference}"] = grouped[group][candidate]
        output.append(row)
    return output


def summarize_by(rows: list[dict[str, Any]], value_keys: list[str], group_keys: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for value_key in value_keys:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            group = "|".join(str(row.get(key, "")) for key in group_keys)
            grouped[group].append(safe_float(row.get(value_key)))
        result[value_key] = {
            group: {
                "count": len(values),
                "changed": sum(1 for value in values if abs(value) > 1e-12),
                "mean": mean(values) if values else 0.0,
                "mean_abs": mean(abs(value) for value in values) if values else 0.0,
            }
            for group, values in sorted(grouped.items())
        }
    return result


def compact_rows(rows: list[dict[str, Any]], *, reference: str, candidate: str) -> list[dict[str, Any]]:
    return [
        {
            "key": row.get("key"),
            "role": row.get("role"),
            "expert": row.get("expert"),
            "layer_band": row.get("layer_band"),
            "module_family": row.get("module_family"),
            "code_positive_sources": row.get("code_positive_sources"),
            "code_negative_sources": row.get("code_negative_sources"),
            f"{reference}_delta": row.get(f"{reference}_delta"),
            f"{candidate}_delta": row.get(f"{candidate}_delta"),
        }
        for row in rows
    ]


def render_markdown(summary: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# RCRF Operating Point Comparison",
        "",
        f"- output_dir: `{output_dir}`",
        f"- reference: `{summary['reference']}`",
        f"- compares: `{', '.join(summary['compares'])}`",
        "",
        "## Candidate Overview",
        "",
        "| candidate | changed | + | - | mean abs delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in summary["candidate_overview"].items():
        lines.append(
            f"| {name} | {stats['changed_count']} | {stats['positive_count']} | {stats['negative_count']} | "
            f"{safe_float(stats['mean_abs_delta']):.6f} |"
        )
    lines.extend(["", "## Reference Gaps", "", "| candidate | ref changed | lost | sign mismatch | big gap | top lost roles |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for name, stats in summary["reference_gap_summary"].items():
        top_roles = ", ".join(f"{role}:{count}" for role, count in list(stats["lost_by_role"].items())[:6])
        lines.append(
            f"| {name} | {stats['reference_changed_count']} | {stats['lost_count']} | "
            f"{stats['sign_mismatch_count']} | {stats['big_gap_count']} | {top_roles} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `operating_point_rows.jsonl`: per residual row with candidate coefficients/deltas.",
            "- `delta_by_role.csv`: aggregate by atlas role.",
            "- `delta_by_role_expert.csv`: aggregate by role and expert.",
            "- `delta_by_layer_module_expert.csv`: aggregate by layer band, module family, and expert.",
            "- `reference_lost_by_role.csv`: rows where reference changed but candidate held.",
            "- `reference_lost_by_source_pattern.csv`: same, grouped by Code source pattern.",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_aligned(
    atlas_rows: list[dict[str, Any]],
    base_gates: dict[str, float],
    candidate_gates: dict[str, dict[str, float]],
) -> None:
    expected = {residual_key(row) for row in atlas_rows}
    missing_base = expected - set(base_gates)
    if missing_base:
        raise ValueError(f"Base gate missing {len(missing_base)} atlas keys, first={sorted(missing_base)[:3]}")
    for name, gates in candidate_gates.items():
        missing = expected - set(gates)
        if missing:
            raise ValueError(f"{name} gate missing {len(missing)} atlas keys, first={sorted(missing)[:3]}")


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
    return {str(key): safe_float(value) for key, value in raw.items() if isinstance(value, int | float)}


def residual_key(row: dict[str, Any]) -> str:
    return f"{row.get('param_name')}::{row.get('expert')}"


def counter_dict(values: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(values).most_common()}


def sign(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "zero"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


if __name__ == "__main__":
    main()
