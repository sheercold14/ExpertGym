#!/usr/bin/env python3
"""Build no-training gates from activation-update geometry.

The rule is deliberately conservative:

* start from all-one expert coefficients;
* mildly preserve blocks with stable positive alignment on successful probes;
* only shrink an expert when activation-update geometry shows negative pair
  agreement and that expert is anti-aligned on the same successful probe;
* never zero a residual.

This is a smoke-test implementation for trajectory-conditioned activation
geometry. It is not an optimizer and does not use held-out metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_MODULE_STATS = (
    "/tmp/shared-storage/ExpertGym/task_vector_activation_distributions/"
    "opvec4_rcrf_calibration_20260522/activation_module_stats.csv"
)
DEFAULT_CONFLICT_STATS = (
    "/tmp/shared-storage/ExpertGym/task_vector_activation_distributions/"
    "opvec4_rcrf_calibration_20260522/activation_conflict_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/tmp/shared-storage/ExpertGym/activation_update_geometry/"
    "opvec4_rcrf_calibration_20260522/activation_geometry_soft_v1"
)

EXPERTS = ("tool", "memory", "code")
MODULE_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
CALIBRATION_OWNER = {
    "tool_signature_s32": "tool",
    "memory_signature_s32": "memory",
    "memory_fulltraj_s32": "memory",
    "livebench_code_s16": "code",
    "livecodebench_code_s16": "code",
}
SUCCESS_CALIBRATIONS = tuple(CALIBRATION_OWNER)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    module_rows = read_csv(Path(args.module_stats).expanduser())
    conflict_rows = read_csv(Path(args.conflict_stats).expanduser())

    module_index = index_module_stats(module_rows)
    conflict_index = index_conflict_stats(conflict_rows)
    scales = calibration_scales(module_rows)

    gates: dict[str, float] = {}
    decision_rows: list[dict[str, Any]] = []
    for entry in sorted(manifest["basis_entries"], key=lambda item: (str(item["param_name"]), str(item["expert"]))):
        expert = str(entry["expert"])
        if expert not in EXPERTS:
            continue
        param_name = str(entry["param_name"])
        layer = layer_from_param(param_name)
        module = module_from_param(param_name)
        coeff, decision = coefficient_for_entry(
            expert=expert,
            layer=layer,
            module=module,
            module_index=module_index,
            conflict_index=conflict_index,
            scales=scales,
            min_coeff=float(args.min_coeff),
            max_coeff=float(args.max_coeff),
            preserve_weight=float(args.preserve_weight),
            shrink_weight=float(args.shrink_weight),
        )
        gates[f"{param_name}::{expert}"] = coeff
        decision_rows.append(
            {
                "param_name": param_name,
                "expert": expert,
                "layer": layer,
                "module": module,
                **decision,
                "coefficient": coeff,
                "delta_from_init1": coeff - 1.0,
            }
        )

    payload = {
        "format": "activation_update_geometry_soft_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "module_stats": str(Path(args.module_stats).expanduser().resolve()),
        "conflict_stats": str(Path(args.conflict_stats).expanduser().resolve()),
        "principle": [
            "Start from init1/all-one coefficients.",
            "Use successful/positive probe signed effects as alignment evidence.",
            "Use activation-update cosine only as a conflict geometry signal.",
            "Shrink only anti-aligned experts in negative update-geometry regions.",
            "Never hard-mask residuals.",
        ],
        "config": {
            "min_coeff": float(args.min_coeff),
            "max_coeff": float(args.max_coeff),
            "preserve_weight": float(args.preserve_weight),
            "shrink_weight": float(args.shrink_weight),
        },
        "coefficient_summary": summarize_coefficients(gates),
        "decision_summary": summarize_decisions(decision_rows),
        "gates": gates,
        "decision_rows": decision_rows,
    }
    write_json(output_dir / "gates.json", payload)
    write_csv(output_dir / "decision_rows.csv", decision_rows)
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps({"gate_checkpoint": str(output_dir / "gates.json"), "summary": payload["coefficient_summary"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--module-stats", default=DEFAULT_MODULE_STATS)
    parser.add_argument("--conflict-stats", default=DEFAULT_CONFLICT_STATS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-coeff", type=float, default=0.85)
    parser.add_argument("--max-coeff", type=float, default=1.08)
    parser.add_argument("--preserve-weight", type=float, default=0.05)
    parser.add_argument("--shrink-weight", type=float, default=0.10)
    return parser.parse_args()


def coefficient_for_entry(
    *,
    expert: str,
    layer: int,
    module: str,
    module_index: dict[tuple[str, str, int, str], dict[str, float]],
    conflict_index: dict[tuple[str, int, str], dict[str, float]],
    scales: dict[str, float],
    min_coeff: float,
    max_coeff: float,
    preserve_weight: float,
    shrink_weight: float,
) -> tuple[float, dict[str, Any]]:
    preserve_terms: list[float] = []
    shrink_terms: list[float] = []
    evidence: dict[str, Any] = {}

    for calibration in SUCCESS_CALIBRATIONS:
        owner = CALIBRATION_OWNER[calibration]
        current = module_index.get((calibration, expert, layer, module))
        owner_row = module_index.get((calibration, owner, layer, module))
        if current is None:
            continue
        signed = float(current["signed_effect_mean"])
        positive_frac = float(current["signed_effect_positive_frac"])
        normalized = signed / scales[calibration]
        signed_scaled = math.asinh(normalized)

        # Positive successful-probe alignment is preservation evidence. It is
        # intentionally mild and does not imply ownership.
        if signed > 0.0:
            consistency = positive_frac if signed >= 0.0 else 1.0 - positive_frac
            preserve_terms.append(max(0.0, signed_scaled) * max(0.0, consistency - 0.5) * 2.0)

        if owner_row is None or owner == expert:
            continue
        owner_signed = float(owner_row["signed_effect_mean"])
        owner_scaled = math.asinh(owner_signed / scales[calibration])
        pair = pair_key(expert, owner)
        conflict = conflict_index.get((calibration, layer, pair))
        if conflict is None:
            continue
        negative_fraction = float(conflict["negative_fraction"])
        cosine_mean = float(conflict["cosine_mean"])

        # Shrink only when geometry is mostly negative and the current expert is
        # anti-aligned while the owner is aligned. Negative geometry alone is not
        # enough: orthogonal/cooperative successful probes should survive.
        if signed_scaled < 0.0 and owner_scaled > 0.0 and negative_fraction > 0.60 and cosine_mean < 0.0:
            conflict_strength = max(0.0, negative_fraction - 0.60) / 0.40
            cosine_strength = min(1.0, abs(cosine_mean) / 0.08)
            anti_strength = min(1.0, abs(signed_scaled) / 3.0)
            shrink_terms.append(conflict_strength * (0.5 + 0.5 * cosine_strength) * anti_strength)

    preserve = float(np.mean(preserve_terms)) if preserve_terms else 0.0
    shrink = float(np.mean(shrink_terms)) if shrink_terms else 0.0
    raw_coeff = 1.0 + preserve_weight * min(1.0, preserve) - shrink_weight * min(1.0, shrink)
    coeff = min(max(raw_coeff, min_coeff), max_coeff)
    evidence.update(
        {
            "preserve_score": preserve,
            "shrink_score": shrink,
            "raw_coefficient": raw_coeff,
            "num_preserve_terms": len(preserve_terms),
            "num_shrink_terms": len(shrink_terms),
        }
    )
    return coeff, evidence


def index_module_stats(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, str], dict[str, float]]:
    output = {}
    for row in rows:
        calibration = str(row["calibration"])
        if calibration not in CALIBRATION_OWNER:
            continue
        key = (calibration, str(row["expert"]), int(row["layer"]), str(row["module"]))
        output[key] = {
            "signed_effect_mean": safe_float(row.get("signed_effect_mean")),
            "signed_effect_positive_frac": safe_float(row.get("signed_effect_positive_frac")),
            "expression_mean": safe_float(row.get("expression_mean")),
        }
    return output


def index_conflict_stats(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, float]]:
    output = {}
    for row in rows:
        calibration = str(row["calibration"])
        if calibration not in CALIBRATION_OWNER:
            continue
        pair = str(row["pair"])
        output[(calibration, int(row["layer"]), pair)] = {
            "cosine_mean": safe_float(row.get("cosine_mean")),
            "negative_fraction": safe_float(row.get("negative_fraction")),
            "count": safe_float(row.get("count")),
        }
    return output


def calibration_scales(rows: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        calibration = str(row["calibration"])
        if calibration in CALIBRATION_OWNER:
            value = abs(safe_float(row.get("signed_effect_mean")))
            if value > 0.0:
                values[calibration].append(value)
    scales = {}
    for calibration in SUCCESS_CALIBRATIONS:
        arr = np.asarray(values.get(calibration) or [1.0], dtype=np.float64)
        scales[calibration] = max(float(np.nanmedian(arr)), 1.0e-30)
    return scales


def summarize_coefficients(gates: dict[str, float]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        expert = key.rsplit("::", 1)[1]
        grouped[expert].append(float(value))
    output = {}
    for expert, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        output[expert] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "changed": int(np.sum(np.abs(arr - 1.0) > 1.0e-8)),
            "shrunk": int(np.sum(arr < 1.0 - 1.0e-8)),
            "boosted": int(np.sum(arr > 1.0 + 1.0e-8)),
        }
    return output


def summarize_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for expert in EXPERTS:
        selected = [row for row in rows if row["expert"] == expert]
        output[expert] = {
            "changed": sum(1 for row in selected if abs(float(row["coefficient"]) - 1.0) > 1.0e-8),
            "shrunk": sum(1 for row in selected if float(row["coefficient"]) < 1.0 - 1.0e-8),
            "boosted": sum(1 for row in selected if float(row["coefficient"]) > 1.0 + 1.0e-8),
            "mean_preserve_score": mean(float(row["preserve_score"]) for row in selected),
            "mean_shrink_score": mean(float(row["shrink_score"]) for row in selected),
        }
    return output


def pair_key(left: str, right: str) -> str:
    return "|".join(sorted([left, right]))


def module_from_param(param_name: str) -> str:
    mapping = {
        ".self_attn.q_proj.": "q",
        ".self_attn.k_proj.": "k",
        ".self_attn.v_proj.": "v",
        ".self_attn.o_proj.": "o",
        ".mlp.gate_proj.": "gate",
        ".mlp.up_proj.": "up",
        ".mlp.down_proj.": "down",
    }
    for needle, module in mapping.items():
        if needle in param_name:
            return module
    raise ValueError(f"Unknown module for {param_name}")


def layer_from_param(param_name: str) -> int:
    match = re.search(r"model\.layers\.(\d+)\.", param_name)
    return int(match.group(1)) if match else -1


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def mean(values: Any) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Activation-Update Geometry Soft Gates v1",
        "",
        "No training and no metric tuning. This starts from init1/all-one coefficients and applies mild geometry-derived soft coefficients.",
        "",
        "Coefficient summary:",
        "",
        "| expert | mean | min | max | changed | shrunk | boosted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for expert in EXPERTS:
        stats = payload["coefficient_summary"][expert]
        lines.append(
            f"| {expert} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | "
            f"{stats['changed']} | {stats['shrunk']} | {stats['boosted']} |"
        )
    lines.extend(["", "Decision summary:", ""])
    lines.append("| expert | changed | shrunk | boosted | mean preserve | mean shrink |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for expert in EXPERTS:
        stats = payload["decision_summary"][expert]
        lines.append(
            f"| {expert} | {stats['changed']} | {stats['shrunk']} | {stats['boosted']} | "
            f"{stats['mean_preserve_score']:.4f} | {stats['mean_shrink_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Guardrail: coefficients are soft and bounded; residuals are never zeroed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
