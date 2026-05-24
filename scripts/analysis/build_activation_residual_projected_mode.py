#!/usr/bin/env python3
"""Build an OP-VEC mode with residual merged activation projection.

Unlike ``build_activation_projected_mode.py``, this script does not project
each non-owner expert independently.  For a successful owner task and module it
first forms the actually baked non-owner residual direction

    r_t = sum_{e in editable} c_{param,e} Delta W_e h_t

and compares it to the owner update

    b_t = Delta W_owner h_t.

If the residual has a negative activation-metric component along the owner
direction,

    alpha_R = sum_t <r_t, b_t> / sum_t ||b_t||^2 < 0,

we remove only that residual component.  The tensor correction is distributed
back to editable experts with the least-norm coefficient split under the fixed
gate coefficients:

    Delta W_e' = Delta W_e - alpha_R * c_e / sum_j c_j^2 * Delta W_owner.

With the same gate checkpoint used for calibration, the merged residual update
changes by exactly ``-alpha_R Delta W_owner`` before trust-region capping.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.modeling.bake import create_bake_plan, load_gate_values  # noqa: E402
from scripts.attention_pauh.core import apply_linear_delta, default_owner_task  # noqa: E402
from scripts.attention_pauh.probe_signed_utility import (  # noqa: E402
    ProbeEntry,
    encode_teacher_forced,
    index_probe_entries,
    load_trajectory_rows,
    normalize_task_name,
)


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_GATE_CHECKPOINT = (
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/"
    "contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json"
)
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/OnPolicy/modes/opvec4_activation_residual_projected_codeguard_v1"
EXPERTS = ("tool", "memory", "code")


@dataclass(frozen=True)
class ProbeSpec:
    label: str
    directory: Path
    config: dict[str, Any]
    projection_csv: Path


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
    protected_experts = set(split_csv(args.protect_expert))
    owner_filter = set(split_csv(args.owner_filter))
    coefficient_map, gate_metadata = load_parameter_coefficients(
        mode_manifest_path=manifest_path,
        gate_checkpoint=Path(args.gate_checkpoint).expanduser().resolve(),
    )
    groups = select_candidate_groups(
        specs=specs,
        protected_experts=protected_experts,
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
        raise ValueError("No residual projection groups selected.")

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
    output_manifest["format"] = "opvec4_activation_residual_projected_mode_v1"
    output_manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    output_manifest["source_mode_manifest"] = str(manifest_path)
    output_manifest["projection_metadata"] = {
        "projection": "editable_non_owner_residual_activation_metric",
        "gate_checkpoint": str(Path(args.gate_checkpoint).expanduser().resolve()),
        "gate_metadata": gate_metadata,
        "probe_dirs": [str(spec.directory) for spec in specs],
        "protect_expert": sorted(protected_experts),
        "owner_filter": sorted(owner_filter),
        "min_negative_fraction": float(args.min_negative_fraction),
        "min_conflict_ratio": float(args.min_conflict_ratio),
        "min_conflict_over_align": float(args.min_conflict_over_align),
        "min_coeff_l2": float(args.min_coeff_l2),
        "source_policy": str(args.source_policy),
        "required_sources": [spec.label for spec in specs],
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
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--gate-checkpoint", default=DEFAULT_GATE_CHECKPOINT)
    parser.add_argument("--probe-dir", action="append", required=True, help="label=/path/to/probe_dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--protect-expert", default="code")
    parser.add_argument("--owner-filter", default="code", help="Comma-separated owner experts to project; empty means all.")
    parser.add_argument("--min-negative-fraction", type=float, default=0.62)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.03)
    parser.add_argument("--min-conflict-over-align", type=float, default=0.005)
    parser.add_argument("--min-coeff-l2", type=float, default=1.0e-8)
    parser.add_argument(
        "--source-policy",
        choices=["pooled", "agreement"],
        default="pooled",
        help="pooled uses all selected probe rows together; agreement requires every probe source to have alpha_R < 0.",
    )
    parser.add_argument(
        "--edit-basis",
        choices=["owner", "editable"],
        default="owner",
        help="owner adds the owner task vector direction; editable stays inside the current editable residual span.",
    )
    parser.add_argument("--max-abs-edit-scale", type=float, default=0.20)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--max-abs-alpha", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_parameter_coefficients(*, mode_manifest_path: Path, gate_checkpoint: Path) -> tuple[dict[str, float], dict[str, Any]]:
    gate_values = load_gate_values(gate_checkpoint)
    plan = create_bake_plan(mode_manifest_path=mode_manifest_path, gate_values=gate_values, output_dir=gate_checkpoint.parent / "_plan_only")
    coefficients: dict[str, float] = {}
    for param_name, entries in plan["entries_by_param"].items():
        for entry in entries:
            coefficients[f"{param_name}::{entry['expert']}"] = float(entry["coefficient"])
    return coefficients, {
        "gate_parameterization": plan["gate_parameterization"],
        "num_delta_entries": plan["num_delta_entries"],
    }


def select_candidate_groups(
    *,
    specs: list[ProbeSpec],
    protected_experts: set[str],
    owner_filter: set[str],
    coefficient_map: Mapping[str, float],
    min_negative_fraction: float,
    min_conflict_ratio: float,
    min_conflict_over_align: float,
    min_coeff_l2: float,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for row in read_projection_rows(spec.projection_csv):
            task = normalize_task_name(str(row["task"]))
            owner = default_owner_task(task)
            if owner_filter and owner not in owner_filter:
                continue
            expert = str(row["expert"])
            if expert == owner or expert in protected_experts:
                continue
            param_name = str(row["param_name"])
            editable = editable_experts(owner=owner, protected_experts=protected_experts, param_name=param_name, coefficient_map=coefficient_map)
            coeff_l2 = sum(float(coefficient_map.get(f"{param_name}::{item}", 0.0)) ** 2 for item in editable)
            if expert not in editable or coeff_l2 <= min_coeff_l2:
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
                    "editable_experts": editable,
                    "coeff_l2": coeff_l2,
                    "score": 0.0,
                    "sources": [],
                    "trigger_experts": [],
                    "max_conflict_ratio": 0.0,
                    "min_align_ratio": 1.0,
                    "max_negative_fraction": 0.0,
                    "mean_orth_ratio": 0.0,
                    "num_triggers": 0,
                },
            )
            if spec.label not in item["sources"]:
                item["sources"].append(spec.label)
            if expert not in item["trigger_experts"]:
                item["trigger_experts"].append(expert)
            item["score"] = max(float(item["score"]), score)
            item["max_conflict_ratio"] = max(float(item["max_conflict_ratio"]), conflict)
            item["min_align_ratio"] = min(float(item["min_align_ratio"]), align)
            item["max_negative_fraction"] = max(float(item["max_negative_fraction"]), negative_fraction)
            n = int(item["num_triggers"])
            item["mean_orth_ratio"] = (float(item["mean_orth_ratio"]) * n + float(row["orth_ratio"])) / float(n + 1)
            item["num_triggers"] = n + 1
    for item in selected.values():
        item["sources"] = sorted(item["sources"])
        item["trigger_experts"] = sorted(item["trigger_experts"])
    return selected


def editable_experts(
    *,
    owner: str,
    protected_experts: set[str],
    param_name: str,
    coefficient_map: Mapping[str, float],
) -> list[str]:
    output = []
    for expert in EXPERTS:
        if expert == owner or expert in protected_experts:
            continue
        if abs(float(coefficient_map.get(f"{param_name}::{expert}", 0.0))) > 0.0:
            output.append(expert)
    return output


def collect_residual_projection_coefficients(
    *,
    specs: list[ProbeSpec],
    groups: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    coefficient_map: Mapping[str, float],
    base_model: str,
    device: str,
    dtype: str,
) -> dict[str, dict[str, float]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype_map[dtype], trust_remote_code=True)
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_params = {str(item["param_name"]) for item in groups.values()}
    entries_by_param = index_probe_entries(
        manifest=manifest,
        manifest_dir=manifest_dir,
        experts=set(EXPERTS),
        layers=None,
        scope="all-linear",
    )
    entries_by_param = {param: entries for param, entries in entries_by_param.items() if param in target_params}
    captured: dict[str, torch.Tensor] = {}
    hooks = register_input_hooks(model, set(entries_by_param), captured)
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    try:
        with torch.no_grad():
            for spec in specs:
                source_groups = [item for item in groups.values() if spec.label in set(item["sources"])]
                if not source_groups:
                    continue
                by_task_param: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
                for item in source_groups:
                    by_task_param[(str(item["task"]), str(item["param_name"]))].append(item)
                cfg = spec.config
                rows = load_trajectory_rows(
                    [Path(path).expanduser() for path in cfg["trajectory_jsonl"]],
                    tasks=tuple(normalize_task_name(task) for task in cfg["tasks"]),
                    samples_per_task=int(cfg["samples_per_task"]),
                )
                for row in rows:
                    captured.clear()
                    encoded = encode_teacher_forced(
                        tokenizer,
                        row,
                        max_seq_length=int(cfg["max_seq_length"]),
                        span=str(cfg["span"]),
                        response_tail_tokens=int(cfg["response_tail_tokens"]),
                    )
                    token_mask = encoded.pop("probe_token_mask").to(device)
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    _ = model(**encoded, use_cache=False)
                    task = normalize_task_name(str(row["task"]))
                    for (row_task, param), items in by_task_param.items():
                        if row_task != task:
                            continue
                        inputs = captured.get(param)
                        if inputs is None:
                            continue
                        selected_inputs = select_inputs(inputs.detach(), token_mask)
                        if selected_inputs.numel() == 0:
                            continue
                        owner = str(items[0]["owner"])
                        owner_entry = entry_for(entries_by_param[param], owner)
                        owner_delta = load_delta(owner_entry.storage_path)
                        owner_update = apply_linear_delta(delta=owner_delta, selected_inputs=selected_inputs)
                        denom = float(owner_update.float().pow(2).sum().item())
                        if denom <= 1.0e-24:
                            continue
                        expert_updates: dict[str, torch.Tensor] = {}
                        for item in items:
                            key = str(item["key"])
                            numerator = 0.0
                            coeff_l2 = 0.0
                            for expert in item["editable_experts"]:
                                coeff = float(coefficient_map.get(f"{param}::{expert}", 0.0))
                                if coeff == 0.0:
                                    continue
                                if expert not in expert_updates:
                                    entry = entry_for(entries_by_param[param], str(expert))
                                    delta = load_delta(entry.storage_path)
                                    expert_updates[str(expert)] = apply_linear_delta(delta=delta, selected_inputs=selected_inputs)
                                dot = float((expert_updates[str(expert)].float() * owner_update.float()).sum().item())
                                numerator += coeff * dot
                                coeff_l2 += coeff * coeff
                                stats[key][f"numerator::{expert}"] += coeff * dot
                            stats[key]["numerator"] += numerator
                            stats[key]["denominator"] += denom
                            stats[key]["coeff_l2_accum"] += coeff_l2
                            stats[key]["count"] += 1.0
                            stats[key][f"source::{spec.label}::numerator"] += numerator
                            stats[key][f"source::{spec.label}::denominator"] += denom
                            stats[key][f"source::{spec.label}::count"] += 1.0
                        del owner_delta, owner_update, expert_updates
                    del encoded, token_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        for hook in hooks:
            hook.remove()
        model.cpu()
        del model
    output: dict[str, dict[str, float]] = {}
    for key, item in stats.items():
        denom = float(item.get("denominator", 0.0))
        alpha = float(item.get("numerator", 0.0)) / denom if denom > 1.0e-24 else 0.0
        count = max(float(item.get("count", 0.0)), 1.0)
        source_alphas: dict[str, float] = {}
        source_counts: dict[str, float] = {}
        for stat_key, numerator in item.items():
            if not stat_key.startswith("source::") or not stat_key.endswith("::numerator"):
                continue
            source = stat_key[len("source::") : -len("::numerator")]
            source_denom = float(item.get(f"source::{source}::denominator", 0.0))
            if source_denom > 1.0e-24:
                source_alphas[source] = float(numerator) / source_denom
                source_counts[source] = float(item.get(f"source::{source}::count", 0.0))
        output[key] = {
            **item,
            "alpha": alpha,
            "source_alphas": source_alphas,
            "source_counts": source_counts,
            "coeff_l2": float(item.get("coeff_l2_accum", 0.0)) / count,
        }
    return output


def materialize_residual_projected_mode(
    *,
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    output_dir: Path,
    entry_index: Mapping[str, ProbeEntry],
    groups: Mapping[str, Mapping[str, Any]],
    alpha_stats: Mapping[str, Mapping[str, float]],
    coefficient_map: Mapping[str, float],
    source_policy: str,
    required_sources: list[str],
    edit_basis: str,
    max_abs_edit_scale: float,
    projection_strength: float,
    max_abs_alpha: float,
    min_coeff_l2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projected_groups: list[dict[str, Any]] = []
    groups_by_param_expert: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        key = str(group["key"])
        stats = alpha_stats.get(key, {})
        pooled_alpha = float(stats.get("alpha", 0.0))
        source_alphas = {
            str(source): float(value)
            for source, value in dict(stats.get("source_alphas", {})).items()
        }
        alpha, source_decision = decide_projection_alpha(
            pooled_alpha=pooled_alpha,
            source_alphas=source_alphas,
            source_policy=source_policy,
            required_sources=required_sources,
        )
        coeff_l2 = float(group.get("coeff_l2", 0.0))
        projected = alpha < 0.0 and coeff_l2 > min_coeff_l2
        applied_alpha = max(alpha, -float(max_abs_alpha)) if projected else 0.0
        row = {
            **group,
            "alpha": alpha,
            "pooled_alpha": pooled_alpha,
            "source_policy": source_policy,
            "source_decision": source_decision,
            "source_alphas_json": json.dumps(source_alphas, sort_keys=True),
            "source_alpha_min": min(source_alphas.values()) if source_alphas else 0.0,
            "source_alpha_max": max(source_alphas.values()) if source_alphas else 0.0,
            "num_negative_sources": sum(1 for value in source_alphas.values() if value < 0.0),
            "applied_alpha": applied_alpha,
            "projected": bool(projected),
            "stats_count": float(stats.get("count", 0.0)),
            "stats_denominator": float(stats.get("denominator", 0.0)),
        }
        projected_groups.append(row)
        if not projected:
            continue
        param_name = str(group["param_name"])
        if edit_basis == "owner":
            for expert in group["editable_experts"]:
                coeff = float(coefficient_map.get(f"{param_name}::{expert}", 0.0))
                if coeff == 0.0:
                    continue
                edit_scale = -float(projection_strength) * applied_alpha * coeff / coeff_l2
                groups_by_param_expert[f"{param_name}::{expert}"].append(
                    {**row, "edit_expert": expert, "edit_scale": edit_scale, "edit_basis": edit_basis}
                )
        elif edit_basis == "editable":
            denom = float(row["stats_denominator"])
            target_numerator = -float(projection_strength) * applied_alpha * denom
            contributions = {
                str(expert): float(stats.get(f"numerator::{expert}", 0.0))
                for expert in group["editable_experts"]
            }
            contribution_l2 = sum(value * value for value in contributions.values())
            if contribution_l2 <= 1.0e-24:
                continue
            for expert, contribution in contributions.items():
                raw_edit_scale = target_numerator * contribution / contribution_l2
                edit_scale = max(-float(max_abs_edit_scale), min(float(max_abs_edit_scale), raw_edit_scale))
                if edit_scale == 0.0:
                    continue
                groups_by_param_expert[f"{param_name}::{expert}"].append(
                    {
                        **row,
                        "edit_expert": expert,
                        "edit_scale": edit_scale,
                        "raw_edit_scale": raw_edit_scale,
                        "edit_basis": edit_basis,
                        "projection_contribution": contribution,
                        "projection_contribution_l2": contribution_l2,
                    }
                )
        else:
            raise ValueError(f"Unsupported edit_basis={edit_basis!r}")

    edits: list[dict[str, Any]] = []
    for raw in manifest["basis_entries"]:
        key = f"{raw['param_name']}::{raw['expert']}"
        source = manifest_dir / str(raw["storage_path"])
        target = output_dir / str(raw["storage_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        pending = groups_by_param_expert.get(key, [])
        if not pending:
            os.symlink(source, target)
            continue
        expert_entry = entry_index[key]
        expert_delta = torch.load(expert_entry.storage_path, map_location="cpu").float()
        updated = expert_delta.clone()
        edit_norm_sq = 0.0
        for item in pending:
            if item.get("edit_basis") == "editable":
                correction_basis = expert_delta
            else:
                owner_key = f"{item['param_name']}::{item['owner']}"
                correction_basis = torch.load(entry_index[owner_key].storage_path, map_location="cpu").float()
            correction = float(item["edit_scale"]) * correction_basis
            updated = updated + correction
            edit_norm_sq += float(correction.norm().item()) ** 2
            edits.append(
                {
                    "param_name": item["param_name"],
                    "task": item["task"],
                    "owner": item["owner"],
                    "expert": item["edit_expert"],
                    "alpha": item["alpha"],
                    "pooled_alpha": item["pooled_alpha"],
                    "source_policy": item["source_policy"],
                    "source_decision": item["source_decision"],
                    "source_alphas_json": item["source_alphas_json"],
                    "source_alpha_min": item["source_alpha_min"],
                    "source_alpha_max": item["source_alpha_max"],
                    "applied_alpha": item["applied_alpha"],
                    "edit_scale": item["edit_scale"],
                    "raw_edit_scale": item.get("raw_edit_scale", item["edit_scale"]),
                    "edit_basis": item.get("edit_basis", "owner"),
                    "projection_contribution": item.get("projection_contribution", 0.0),
                    "projection_contribution_l2": item.get("projection_contribution_l2", 0.0),
                    "coeff_l2": item["coeff_l2"],
                    "score": item["score"],
                    "sources": ",".join(item["sources"]),
                    "trigger_experts": ",".join(item["trigger_experts"]),
                }
            )
            del correction
            if item.get("edit_basis") != "editable":
                del correction_basis
        torch.save(updated.to(dtype=expert_delta.dtype), target)
        base_norm = float(expert_delta.norm().item())
        edit_norm = edit_norm_sq ** 0.5
        for item in edits[-len(pending) :]:
            item["base_norm"] = base_norm
            item["edit_norm"] = edit_norm
            item["relative_edit_norm"] = edit_norm / max(base_norm, 1.0e-12) if base_norm > 0.0 else 0.0
        del expert_delta, updated
    projected_groups = sorted(projected_groups, key=lambda item: abs(float(item["applied_alpha"])), reverse=True)
    edits = sorted(edits, key=lambda item: abs(float(item["edit_scale"])), reverse=True)
    return edits, projected_groups


def decide_projection_alpha(
    *,
    pooled_alpha: float,
    source_alphas: Mapping[str, float],
    source_policy: str,
    required_sources: list[str],
) -> tuple[float, str]:
    if source_policy == "pooled":
        return pooled_alpha, "pooled"
    if source_policy != "agreement":
        raise ValueError(f"Unsupported source_policy={source_policy!r}")
    missing = [source for source in required_sources if source not in source_alphas]
    if missing:
        return 0.0, f"missing:{','.join(missing)}"
    required_values = [float(source_alphas[source]) for source in required_sources]
    if any(value >= 0.0 for value in required_values):
        return 0.0, "sign_disagreement"
    return max(required_values), "all_negative"


def select_inputs(inputs: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask.to(device=inputs.device, dtype=torch.bool)
    if mask.ndim == 1:
        return inputs[:, mask, :]
    if mask.ndim == 2:
        return inputs[mask].unsqueeze(0)
    raise ValueError(f"Unsupported token_mask ndim={mask.ndim}")


def register_input_hooks(
    model: torch.nn.Module,
    target_param_names: set[str],
    captured: dict[str, torch.Tensor],
) -> list[Any]:
    hooks = []
    for module_name, module in model.named_modules():
        param_name = f"{module_name}.weight"
        if param_name not in target_param_names:
            continue

        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any, *, name: str = param_name) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            hidden = inputs[0]
            if hidden.ndim == 3:
                captured[name] = hidden

        hooks.append(module.register_forward_hook(hook))
    return hooks


def read_probe_spec(raw: str) -> ProbeSpec:
    label, path = parse_labeled_path(raw)
    directory = path.expanduser().resolve()
    config = read_json(directory / "probe_config.json")
    projection_csv = directory / "activation_update_projection_summary.csv"
    if not projection_csv.exists():
        raise FileNotFoundError(projection_csv)
    return ProbeSpec(label=label, directory=directory, config=config, projection_csv=projection_csv)


def read_projection_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = dict(row)
            for key in (
                "align_ratio",
                "conflict_ratio",
                "orth_ratio",
                "negative_fraction",
                "cosine_mean",
            ):
                item[key] = float(item[key])
            rows.append(item)
    return rows


def index_entries(manifest: Mapping[str, Any], manifest_dir: Path) -> dict[str, ProbeEntry]:
    output = {}
    for entries in index_probe_entries(
        manifest=manifest,
        manifest_dir=manifest_dir,
        experts=set(EXPERTS),
        layers=None,
        scope="all-linear",
    ).values():
        for entry in entries:
            output[f"{entry.param_name}::{entry.expert}"] = entry
    return output


def entry_for(entries: list[ProbeEntry], expert: str) -> ProbeEntry:
    for entry in entries:
        if entry.expert == expert:
            return entry
    raise KeyError(expert)


def load_delta(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw).expanduser().resolve()
        return path.name, path
    label, path = raw.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def write_markdown(
    path: Path,
    metadata: Mapping[str, Any],
    group_rows: list[Mapping[str, Any]],
    edits: list[Mapping[str, Any]],
) -> None:
    lines = [
        "# Activation Residual-Projected OP-VEC Mode",
        "",
        "This artifact edits task-vector tensors directly. Scalar gates are unchanged.",
        "",
        "Projection target: coefficient-weighted editable non-owner residual, measured on successful trajectory activations.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Top Projection Groups",
        "",
        "| rank | owner | task | param | alpha_R | pooled | source max | applied | coeff_l2 | decision | triggers | sources |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for rank, item in enumerate(group_rows[:50], start=1):
        lines.append(
            f"| {rank} | {item['owner']} | {item['task']} | `{item['param_name']}` | "
            f"{float(item['alpha']):.5f} | {float(item.get('pooled_alpha', item['alpha'])):.5f} | "
            f"{float(item.get('source_alpha_max', 0.0)):.5f} | {float(item['applied_alpha']):.5f} | "
            f"{float(item['coeff_l2']):.5f} | {item.get('source_decision', 'pooled')} | "
            f"{','.join(item['trigger_experts'])} | {','.join(item['sources'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Tensor Edits",
            "",
            "| rank | basis | expert | owner | task | param | edit scale | rel edit | alpha_R |",
            "|---:|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for rank, item in enumerate(edits[:50], start=1):
        lines.append(
            f"| {rank} | {item.get('edit_basis', 'owner')} | {item['expert']} | {item['owner']} | "
            f"{item['task']} | `{item['param_name']}` | "
            f"{float(item['edit_scale']):.5f} | {float(item['relative_edit_norm']):.5f} | {float(item['alpha']):.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
