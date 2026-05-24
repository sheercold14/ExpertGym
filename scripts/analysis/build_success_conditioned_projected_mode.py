#!/usr/bin/env python3
"""Build a success-conditioned projected OP-VEC mode.

This is the projection candidate builder for the 8765 -> TRC geometry route.
It does not train reward weights and it does not shrink scalar gates.  It:

1. reads the success-conditioned projection ledger;
2. permits edits only on `mixed_success_geometry` residual rows;
3. recomputes source-specific activation projection coefficients;
4. materializes a new OP-VEC mode by removing capped anti-success projection.

The heavy lifting is delegated to build_activation_residual_projected_mode.py.
This file only replaces its task-prior candidate selector.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.build_activation_residual_projected_mode import (  # noqa: E402
    collect_residual_projection_coefficients,
    index_entries,
    load_parameter_coefficients,
    materialize_residual_projected_mode,
    read_json,
    read_probe_spec,
    read_projection_rows,
    split_csv,
    write_csv,
    write_json,
    write_markdown,
)
from scripts.attention_pauh.core import default_owner_task  # noqa: E402
from scripts.attention_pauh.probe_signed_utility import normalize_task_name  # noqa: E402


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_GATE_CHECKPOINT = "/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517/gate_values.json"
DEFAULT_LEDGER = (
    "/tmp/shared-storage/ExpertGym/activation_update_geometry/"
    "success_conditioned_ledger_20260523/success_projection_ledger.csv"
)
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/OnPolicy/modes/opvec4_success_project_init1_tm_guard_v0"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    mode_dir = manifest_path.parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [read_probe_spec(item) for item in args.probe_dir]
    eligible = read_eligible_ledger_rows(
        Path(args.success_ledger).expanduser().resolve(),
        allowed_roles=set(split_csv(args.allowed_role)),
    )
    owner_filter = set(split_csv(args.owner_filter))
    coefficient_map, gate_metadata = load_parameter_coefficients(
        mode_manifest_path=manifest_path,
        gate_checkpoint=Path(args.gate_checkpoint).expanduser().resolve(),
    )
    groups = select_success_conditioned_groups(
        specs=specs,
        eligible=eligible,
        owner_filter=owner_filter,
        coefficient_map=coefficient_map,
        min_negative_fraction=float(args.min_negative_fraction),
        min_conflict_ratio=float(args.min_conflict_ratio),
        min_conflict_over_align=float(args.min_conflict_over_align),
        min_coeff_l2=float(args.min_coeff_l2),
    )
    if args.max_candidates > 0:
        groups = dict(sorted(groups.items(), key=lambda item: item[1]["score"], reverse=True)[: int(args.max_candidates)])
    if not groups:
        raise ValueError("No success-conditioned projection groups selected.")
    if args.selection_only:
        payload = {
            "format": "success_conditioned_projection_selection_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_groups": len(groups),
            "groups": groups,
            "summary": summarize_groups(groups),
        }
        write_json(output_dir / "selection_plan.json", payload)
        print(json.dumps({key: value for key, value in payload.items() if key != "groups"}, indent=2, sort_keys=True))
        return

    entry_index = index_entries(manifest, mode_dir)
    alpha_stats = collect_residual_projection_coefficients(
        specs=specs,
        groups=groups,
        manifest=manifest,
        manifest_dir=mode_dir,
        coefficient_map=coefficient_map,
        base_model=str(args.base_model or manifest["base_model"]),
        device=str(args.device),
        dtype=str(args.torch_dtype),
    )
    edits, group_rows = materialize_residual_projected_mode(
        manifest=manifest,
        manifest_dir=mode_dir,
        output_dir=output_dir,
        entry_index=entry_index,
        groups=groups,
        alpha_stats=alpha_stats,
        coefficient_map=coefficient_map,
        source_policy=str(args.source_policy),
        required_sources=[spec.label for spec in specs],
        edit_basis=str(args.edit_basis),
        max_abs_edit_scale=float(args.max_abs_edit_scale),
        projection_strength=float(args.projection_strength),
        max_abs_alpha=float(args.max_abs_alpha),
        min_coeff_l2=float(args.min_coeff_l2),
    )
    output_manifest = dict(manifest)
    output_manifest["format"] = "opvec4_success_conditioned_projected_mode_v1"
    output_manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    output_manifest["source_mode_manifest"] = str(manifest_path)
    output_manifest["projection_metadata"] = {
        "projection": "success_conditioned_mixed_rows_activation_metric",
        "gate_checkpoint": str(Path(args.gate_checkpoint).expanduser().resolve()),
        "gate_metadata": gate_metadata,
        "success_ledger": str(Path(args.success_ledger).expanduser().resolve()),
        "allowed_role": sorted(set(split_csv(args.allowed_role))),
        "probe_dirs": [str(spec.directory) for spec in specs],
        "owner_filter": sorted(owner_filter),
        "min_negative_fraction": float(args.min_negative_fraction),
        "min_conflict_ratio": float(args.min_conflict_ratio),
        "min_conflict_over_align": float(args.min_conflict_over_align),
        "min_coeff_l2": float(args.min_coeff_l2),
        "source_policy": str(args.source_policy),
        "edit_basis": str(args.edit_basis),
        "max_abs_edit_scale": float(args.max_abs_edit_scale),
        "projection_strength": float(args.projection_strength),
        "max_abs_alpha": float(args.max_abs_alpha),
        "num_groups": len(groups),
        "num_projected_groups": sum(1 for item in group_rows if item["projected"]),
        "num_projected_tensor_edits": len(edits),
    }
    write_json(output_dir / "mode_manifest.json", output_manifest)
    write_json(output_dir / "projection_summary.json", {"groups": groups, "group_rows": group_rows, "edits": edits})
    write_csv(output_dir / "projection_groups.csv", group_rows)
    write_csv(output_dir / "projection_edits.csv", edits)
    write_markdown(output_dir / "projection_summary.md", output_manifest["projection_metadata"], group_rows, edits)
    print(
        json.dumps(
            {
                "mode_manifest": str(output_dir / "mode_manifest.json"),
                "num_groups": len(groups),
                "num_projected_groups": output_manifest["projection_metadata"]["num_projected_groups"],
                "num_projected_tensor_edits": len(edits),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--gate-checkpoint", default=DEFAULT_GATE_CHECKPOINT)
    parser.add_argument("--success-ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--allowed-role", default="mixed_success_geometry")
    parser.add_argument("--probe-dir", action="append", required=True, help="label=/path/to/probe_dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--owner-filter", default="tool,memory", help="Successful source owners to protect/project against.")
    parser.add_argument("--min-negative-fraction", type=float, default=0.62)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.03)
    parser.add_argument("--min-conflict-over-align", type=float, default=0.005)
    parser.add_argument("--min-coeff-l2", type=float, default=1.0e-8)
    parser.add_argument("--source-policy", choices=["pooled", "agreement"], default="pooled")
    parser.add_argument("--edit-basis", choices=["owner", "editable"], default="owner")
    parser.add_argument("--max-abs-edit-scale", type=float, default=0.10)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--max-abs-alpha", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_eligible_ledger_rows(path: Path, *, allowed_roles: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    import csv

    eligible: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            role = str(row.get("role") or "")
            if allowed_roles and role not in allowed_roles:
                continue
            key = (str(row["param_name"]), str(row["expert"]))
            eligible[key] = dict(row)
    if not eligible:
        raise ValueError(f"No ledger rows matched roles {sorted(allowed_roles)} in {path}")
    return eligible


def select_success_conditioned_groups(
    *,
    specs: list[Any],
    eligible: Mapping[tuple[str, str], Mapping[str, Any]],
    owner_filter: set[str],
    coefficient_map: Mapping[str, float],
    min_negative_fraction: float,
    min_conflict_ratio: float,
    min_conflict_over_align: float,
    min_coeff_l2: float,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for spec in specs:
        for row in read_projection_rows(spec.projection_csv):
            task = normalize_task_name(str(row["task"]))
            owner = default_owner_task(task)
            if owner_filter and owner not in owner_filter:
                continue
            expert = str(row["expert"])
            if expert == owner:
                continue
            param_name = str(row["param_name"])
            ledger_row = eligible.get((param_name, expert))
            if ledger_row is None:
                continue
            coeff = float(coefficient_map.get(f"{param_name}::{expert}", 0.0))
            if abs(coeff) <= 0.0:
                continue
            conflict = float(row["conflict_ratio"])
            align = float(row["align_ratio"])
            negative_fraction = float(row["negative_fraction"])
            margin = conflict - align - min_conflict_over_align
            if conflict < min_conflict_ratio or margin <= 0.0 or negative_fraction < min_negative_fraction:
                continue
            neg_strength = (negative_fraction - min_negative_fraction) / max(1.0 - min_negative_fraction, 1.0e-12)
            score = margin * max(0.0, min(1.0, neg_strength))
            key = f"{param_name}::{owner}::{task}"
            item = selected.setdefault(
                key,
                {
                    "key": key,
                    "task": task,
                    "owner": owner,
                    "param_name": param_name,
                    "editable_experts": [],
                    "coeff_l2": 0.0,
                    "score": 0.0,
                    "sources": [],
                    "trigger_experts": [],
                    "max_conflict_ratio": 0.0,
                    "min_align_ratio": 1.0,
                    "max_negative_fraction": 0.0,
                    "mean_orth_ratio": 0.0,
                    "num_triggers": 0,
                    "ledger_roles": [],
                },
            )
            if expert not in item["editable_experts"]:
                item["editable_experts"].append(expert)
                item["coeff_l2"] += coeff * coeff
            if spec.label not in item["sources"]:
                item["sources"].append(spec.label)
            if expert not in item["trigger_experts"]:
                item["trigger_experts"].append(expert)
            ledger_role = str(ledger_row.get("role") or "")
            if ledger_role and ledger_role not in item["ledger_roles"]:
                item["ledger_roles"].append(ledger_role)
            item["score"] = max(float(item["score"]), score)
            item["max_conflict_ratio"] = max(float(item["max_conflict_ratio"]), conflict)
            item["min_align_ratio"] = min(float(item["min_align_ratio"]), align)
            item["max_negative_fraction"] = max(float(item["max_negative_fraction"]), negative_fraction)
            n = int(item["num_triggers"])
            item["mean_orth_ratio"] = (float(item["mean_orth_ratio"]) * n + float(row["orth_ratio"])) / float(n + 1)
            item["num_triggers"] = n + 1
            source_counts[key][spec.label] += 1

    pruned = {}
    for key, item in selected.items():
        item["editable_experts"] = sorted(item["editable_experts"])
        item["sources"] = sorted(item["sources"])
        item["trigger_experts"] = sorted(item["trigger_experts"])
        item["ledger_roles"] = sorted(item["ledger_roles"])
        if float(item["coeff_l2"]) <= min_coeff_l2:
            continue
        pruned[key] = item
    return pruned


def summarize_groups(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_owner: dict[str, int] = defaultdict(int)
    by_trigger: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    for item in groups.values():
        by_owner[str(item["owner"])] += 1
        for expert in item.get("trigger_experts", []):
            by_trigger[str(expert)] += 1
        for source in item.get("sources", []):
            by_source[str(source)] += 1
    return {
        "by_owner": dict(sorted(by_owner.items())),
        "by_trigger_expert": dict(sorted(by_trigger.items())),
        "by_source": dict(sorted(by_source.items())),
        "top_groups": [
            {
                "key": str(item["key"]),
                "score": float(item["score"]),
                "owner": str(item["owner"]),
                "param_name": str(item["param_name"]),
                "editable_experts": list(item.get("editable_experts", [])),
                "sources": list(item.get("sources", [])),
                "max_conflict_ratio": float(item.get("max_conflict_ratio", 0.0)),
                "max_negative_fraction": float(item.get("max_negative_fraction", 0.0)),
            }
            for item in sorted(groups.values(), key=lambda row: float(row["score"]), reverse=True)[:20]
        ],
    }


if __name__ == "__main__":
    main()
