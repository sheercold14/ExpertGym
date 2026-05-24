#!/usr/bin/env python3
"""Build gates from activation-update projection decomposition.

This is a no-training behavior-support constraint for an existing residual
field.  It starts from a base gate checkpoint, reads successful-trajectory
projection summaries, and only attenuates non-owner residual entries whose
activation update has more signed conflict than alignment with the successful
owner update direction.

The intended claim boundary is narrow:

* dense residual support overlap is not treated as interference;
* orthogonal/shared activation-update components are preserved;
* only stable negative projection components receive a small bounded shrink.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_BASE_GATES = (
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/"
    "residual_capability_field_behavior_constraints_v18/gates.json"
)
DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_OUTPUT_DIR = (
    "/tmp/shared-storage/ExpertGym/activation_update_geometry/"
    "opvec4_rcrf_calibration_20260522/projection_conflict_v1"
)

EXPERTS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(Path(args.mode_manifest).expanduser())
    base_payload = read_json(Path(args.base_gate_checkpoint).expanduser())
    base_gates = gate_values(base_payload)
    rows = []
    for item in args.projection_csv:
        label, path = parse_labeled_path(item)
        rows.extend(read_projection_csv(path, label=label))
    protected_experts = set(split_csv(args.protect_expert))

    row_index = index_projection_rows(rows)
    gates = dict(base_gates)
    decision_rows = []
    for entry in sorted(manifest["basis_entries"], key=lambda item: (str(item["param_name"]), str(item["expert"]))):
        param_name = str(entry["param_name"])
        expert = str(entry["expert"])
        if expert not in EXPERTS:
            continue
        key = f"{param_name}::{expert}"
        before = float(gates.get(key, base_gates.get(key, 1.0)))
        decision = decision_for_entry(
            param_name=param_name,
            expert=expert,
            rows=row_index.get(key, []),
            protected_experts=protected_experts,
            min_negative_fraction=float(args.min_negative_fraction),
            min_conflict_ratio=float(args.min_conflict_ratio),
            min_conflict_over_align=float(args.min_conflict_over_align),
            score_scale=float(args.score_scale),
            max_delta=float(args.max_delta),
        )
        after = max(float(args.min_coeff), min(float(args.max_coeff), before + decision["delta"]))
        gates[key] = after
        decision_rows.append(
            {
                "param_name": param_name,
                "expert": expert,
                "layer": layer_from_param(param_name),
                "module": module_from_param(param_name),
                "family": family_from_param(param_name),
                "before": before,
                "after": after,
                "delta": after - before,
                **decision,
            }
        )

    payload = {
        "format": "activation_update_projection_conflict_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_gate_checkpoint": str(Path(args.base_gate_checkpoint).expanduser().resolve()),
        "mode_manifest": str(Path(args.mode_manifest).expanduser().resolve()),
        "projection_csv": [str(parse_labeled_path(item)[1]) for item in args.projection_csv],
        "principle": [
            "Start from an existing residual field.",
            "Use successful trajectory owner activation updates as local capability directions.",
            "Preserve aligned and orthogonal/shared components.",
            "Apply a small bounded shrink only when conflict projection exceeds alignment.",
            "Never hard-mask residual entries.",
        ],
        "config": {
            "min_negative_fraction": float(args.min_negative_fraction),
            "min_conflict_ratio": float(args.min_conflict_ratio),
            "min_conflict_over_align": float(args.min_conflict_over_align),
            "score_scale": float(args.score_scale),
            "max_delta": float(args.max_delta),
            "min_coeff": float(args.min_coeff),
            "max_coeff": float(args.max_coeff),
            "protect_expert": sorted(protected_experts),
        },
        "gate_summary": summarize_gates(base_gates, gates),
        "decision_summary": summarize_decisions(decision_rows),
        "gates": gates,
    }
    write_json(output_dir / "gates.json", payload)
    write_csv(output_dir / "decision_rows.csv", decision_rows)
    write_jsonl(output_dir / "decision_rows.jsonl", decision_rows)
    write_markdown(output_dir / "summary.md", payload, decision_rows)
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "gates.json"),
                "changed": payload["gate_summary"]["overall"]["changed"],
                "mean_abs_delta": payload["gate_summary"]["overall"]["mean_abs_delta"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gate-checkpoint", default=DEFAULT_BASE_GATES)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--projection-csv", action="append", required=True, help="label=/path/to/projection.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-negative-fraction", type=float, default=0.62)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.03)
    parser.add_argument("--min-conflict-over-align", type=float, default=0.005)
    parser.add_argument("--score-scale", type=float, default=0.05)
    parser.add_argument("--max-delta", type=float, default=0.015)
    parser.add_argument("--min-coeff", type=float, default=0.55)
    parser.add_argument("--max-coeff", type=float, default=1.12)
    parser.add_argument(
        "--protect-expert",
        default="",
        help="Comma-separated experts whose own residual field is protected from projection shrink.",
    )
    return parser.parse_args()


def decision_for_entry(
    *,
    param_name: str,
    expert: str,
    rows: list[dict[str, Any]],
    protected_experts: set[str],
    min_negative_fraction: float,
    min_conflict_ratio: float,
    min_conflict_over_align: float,
    score_scale: float,
    max_delta: float,
) -> dict[str, Any]:
    if expert in protected_experts:
        return {
            "delta": 0.0,
            "reason": "preserve_protected_expert",
            "projection_score": 0.0,
            "source": "",
            "task": "",
            "align_ratio": 0.0,
            "conflict_ratio": 0.0,
            "orth_ratio": 0.0,
            "negative_fraction": 0.0,
            "cosine_mean": 0.0,
        }
    candidates = []
    for row in rows:
        task = str(row["task"])
        owner = owner_expert(task)
        if owner == expert:
            continue
        conflict = float(row["conflict_ratio"])
        align = float(row["align_ratio"])
        negative_fraction = float(row["negative_fraction"])
        margin = conflict - align - min_conflict_over_align
        if conflict < min_conflict_ratio or margin <= 0.0 or negative_fraction < min_negative_fraction:
            continue
        neg_strength = (negative_fraction - min_negative_fraction) / max(1.0 - min_negative_fraction, 1.0e-12)
        score = margin * max(0.0, min(1.0, neg_strength))
        candidates.append(
            {
                "source": str(row.get("source", "")),
                "task": task,
                "score": score,
                "align_ratio": align,
                "conflict_ratio": conflict,
                "orth_ratio": float(row["orth_ratio"]),
                "negative_fraction": negative_fraction,
                "cosine_mean": float(row["cosine_mean"]),
            }
        )
    if not candidates:
        return {
            "delta": 0.0,
            "reason": "preserve_no_conflict_projection",
            "projection_score": 0.0,
            "source": "",
            "task": "",
            "align_ratio": 0.0,
            "conflict_ratio": 0.0,
            "orth_ratio": 0.0,
            "negative_fraction": 0.0,
            "cosine_mean": 0.0,
        }
    best = max(candidates, key=lambda item: float(item["score"]))
    strength = min(1.0, float(best["score"]) / max(score_scale, 1.0e-12))
    return {
        "delta": -float(max_delta) * strength,
        "reason": "shrink_conflict_projection",
        "projection_score": float(best["score"]),
        **{key: best[key] for key in ("source", "task", "align_ratio", "conflict_ratio", "orth_ratio", "negative_fraction", "cosine_mean")},
    }


def index_projection_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['param_name']}::{row['expert']}"
        grouped[key].append(row)
    return grouped


def read_projection_csv(path: Path, *, label: str) -> list[dict[str, Any]]:
    output = []
    with path.expanduser().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = dict(row)
            item["source"] = label
            for key in (
                "layer",
                "count",
                "token_count",
                "align_ratio",
                "conflict_ratio",
                "orth_ratio",
                "total_norm_mean",
                "owner_norm_mean",
                "cosine_mean",
                "negative_fraction",
                "positive_fraction",
            ):
                item[key] = safe_float(item.get(key))
            output.append(item)
    return output


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw).expanduser().resolve()
        return path.parent.name, path
    label, path = raw.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def gate_values(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("gates") if isinstance(payload.get("gates"), Mapping) else payload
    return {str(key): float(value) for key, value in raw.items()}


def summarize_gates(base_gates: Mapping[str, float], gates: Mapping[str, float]) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        delta = float(value) - float(base_gates.get(key, value))
        expert = key.rsplit("::", 1)[-1] if "::" in key else "unknown"
        by_group[expert].append(delta)
        by_group["overall"].append(delta)
    return {group: summarize_deltas(values) for group, values in sorted(by_group.items())}


def summarize_deltas(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "changed": 0.0, "positive": 0.0, "negative": 0.0, "mean_delta": 0.0, "mean_abs_delta": 0.0}
    changed = [value for value in values if abs(value) > 1.0e-12]
    return {
        "count": float(len(values)),
        "changed": float(len(changed)),
        "positive": float(sum(1 for value in changed if value > 0.0)),
        "negative": float(sum(1 for value in changed if value < 0.0)),
        "mean_delta": sum(values) / float(len(values)),
        "mean_abs_delta": sum(abs(value) for value in values) / float(len(values)),
        "max_abs_delta": max(abs(value) for value in values),
    }


def summarize_decisions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    by_expert: dict[str, int] = defaultdict(int)
    for row in rows:
        reasons[str(row["reason"])] += 1
        if abs(float(row["delta"])) > 1.0e-12:
            by_task[str(row["task"])] += 1
            by_expert[str(row["expert"])] += 1
    return {
        "reason_counts": dict(sorted(reasons.items())),
        "changed_by_task": dict(sorted(by_task.items())),
        "changed_by_expert": dict(sorted(by_expert.items())),
    }


def write_markdown(path: Path, payload: Mapping[str, Any], decision_rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Activation-Update Projection Conflict Gates",
        "",
        "No training and no held-out metric tuning. This starts from the v18/v9 residual field and applies only bounded conflict-projection shrink.",
        "",
        "## Gate Summary",
        "",
        "| group | changed | + | - | mean delta | mean abs delta | max abs delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stats in payload["gate_summary"].items():
        lines.append(
            f"| {group} | {stats['changed']:.0f} | {stats['positive']:.0f} | {stats['negative']:.0f} | "
            f"{stats['mean_delta']:.6f} | {stats['mean_abs_delta']:.6f} | {stats['max_abs_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Summary",
            "",
            f"- reason_counts: `{json.dumps(payload['decision_summary']['reason_counts'], sort_keys=True)}`",
            f"- changed_by_task: `{json.dumps(payload['decision_summary']['changed_by_task'], sort_keys=True)}`",
            f"- changed_by_expert: `{json.dumps(payload['decision_summary']['changed_by_expert'], sort_keys=True)}`",
            "",
            "## Top Shrinks",
            "",
            "| rank | expert | task | param | delta | score | conflict | align | orth | neg frac | cosine |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    changed = [row for row in decision_rows if abs(float(row["delta"])) > 1.0e-12]
    changed.sort(key=lambda row: float(row["delta"]))
    for rank, row in enumerate(changed[:40], start=1):
        lines.append(
            f"| {rank} | {row['expert']} | {row['task']} | `{row['param_name']}` | {float(row['delta']):.6f} | "
            f"{float(row['projection_score']):.4f} | {float(row['conflict_ratio']):.3f} | "
            f"{float(row['align_ratio']):.3f} | {float(row['orth_ratio']):.3f} | "
            f"{float(row['negative_fraction']):.3f} | {float(row['cosine_mean']):.3f} |"
        )
    lines.extend(["", "Guardrail: aligned and orthogonal projection mass is never removed explicitly; the checkpoint only changes scalar residual coefficients."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def owner_expert(task: str) -> str:
    task = str(task).strip().lower()
    if task in {"tool", "memory", "code"}:
        return task
    if task in {"bfcl", "toolrl"}:
        return "tool"
    if task in {"hotpotqa", "memagent"}:
        return "memory"
    if task in {"livebench", "livecodebench", "coding"}:
        return "code"
    return task


def layer_from_param(param_name: str) -> int:
    match = re.search(r"model\.layers\.(\d+)\.", param_name)
    if not match:
        raise ValueError(f"Cannot infer layer from {param_name}")
    return int(match.group(1))


def module_from_param(param_name: str) -> str:
    for module in ("q", "k", "v", "o"):
        if param_name.endswith(f"self_attn.{module}_proj.weight"):
            return module
    for module in ("gate", "up", "down"):
        if param_name.endswith(f"mlp.{module}_proj.weight"):
            return module
    return "unknown"


def family_from_param(param_name: str) -> str:
    if ".self_attn." in param_name:
        return "attention"
    if ".mlp." in param_name:
        return "mlp"
    return "unknown"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
