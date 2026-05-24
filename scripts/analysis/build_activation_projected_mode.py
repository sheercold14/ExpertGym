#!/usr/bin/env python3
"""Build a projected OP-VEC mode by editing task-vector modules directly.

This script turns activation-update conflict diagnostics into actual task
vector edits.  For a selected linear module and non-owner expert ``e``, it
uses the successful task owner's update as the capability direction:

    b_t = Delta W_owner h_t
    u_t = Delta W_e h_t

The projection coefficient is computed in the activation metric induced by the
successful trajectory span:

    alpha = sum_t <u_t, b_t> / sum_t ||b_t||^2

If alpha is negative, the edited tensor is

    Delta W_e' = Delta W_e - alpha Delta W_owner

which removes the negative owner-direction component while preserving the
orthogonal component in that module.  The scalar gate checkpoint can then stay
unchanged.
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

from scripts.attention_pauh.core import apply_linear_delta, default_owner_task  # noqa: E402
from scripts.attention_pauh.probe_signed_utility import (  # noqa: E402
    ProbeEntry,
    encode_teacher_forced,
    index_probe_entries,
    load_trajectory_rows,
    normalize_task_name,
)


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/OnPolicy/modes/opvec4_activation_projected_codeguard_v1"
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
    candidates = select_candidates(
        specs=specs,
        protected_experts=protected_experts,
        min_negative_fraction=float(args.min_negative_fraction),
        min_conflict_ratio=float(args.min_conflict_ratio),
        min_conflict_over_align=float(args.min_conflict_over_align),
    )
    if args.max_candidates > 0:
        candidates = dict(
            sorted(candidates.items(), key=lambda item: item[1]["score"], reverse=True)[: int(args.max_candidates)]
        )
    if not candidates:
        raise ValueError("No projection candidates selected.")

    entry_index = index_entries(manifest, mode_dir)
    alpha_stats = collect_projection_coefficients(
        specs=specs,
        candidates=candidates,
        manifest=manifest,
        manifest_dir=mode_dir,
        base_model=str(args.base_model or manifest["base_model"]),
        device=str(args.device),
        dtype=str(args.torch_dtype),
    )
    edits = materialize_projected_mode(
        manifest=manifest,
        manifest_dir=mode_dir,
        output_dir=output_dir,
        entry_index=entry_index,
        candidates=candidates,
        alpha_stats=alpha_stats,
        projection_strength=float(args.projection_strength),
        max_abs_alpha=float(args.max_abs_alpha),
    )
    output_manifest = dict(manifest)
    output_manifest["format"] = "opvec4_activation_projected_mode_v1"
    output_manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    output_manifest["source_mode_manifest"] = str(manifest_path)
    output_manifest["projection_metadata"] = {
        "probe_dirs": [str(spec.directory) for spec in specs],
        "protect_expert": sorted(protected_experts),
        "min_negative_fraction": float(args.min_negative_fraction),
        "min_conflict_ratio": float(args.min_conflict_ratio),
        "min_conflict_over_align": float(args.min_conflict_over_align),
        "projection_strength": float(args.projection_strength),
        "max_abs_alpha": float(args.max_abs_alpha),
        "num_candidates": len(candidates),
        "num_projected": sum(1 for item in edits if item["projected"]),
    }
    write_json(output_dir / "mode_manifest.json", output_manifest)
    write_json(output_dir / "projection_summary.json", {"candidates": candidates, "edits": edits})
    write_csv(output_dir / "projection_edits.csv", edits)
    write_markdown(output_dir / "projection_summary.md", output_manifest["projection_metadata"], edits)
    print(
        json.dumps(
            {
                "mode_manifest": str(output_dir / "mode_manifest.json"),
                "num_candidates": len(candidates),
                "num_projected": output_manifest["projection_metadata"]["num_projected"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--probe-dir", action="append", required=True, help="label=/path/to/probe_dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--protect-expert", default="code")
    parser.add_argument("--min-negative-fraction", type=float, default=0.62)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.03)
    parser.add_argument("--min-conflict-over-align", type=float, default=0.005)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--max-abs-alpha", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_candidates(
    *,
    specs: list[ProbeSpec],
    protected_experts: set[str],
    min_negative_fraction: float,
    min_conflict_ratio: float,
    min_conflict_over_align: float,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for row in read_projection_rows(spec.projection_csv):
            task = normalize_task_name(str(row["task"]))
            expert = str(row["expert"])
            owner = default_owner_task(task)
            if expert == owner or expert in protected_experts:
                continue
            conflict = float(row["conflict_ratio"])
            align = float(row["align_ratio"])
            negative_fraction = float(row["negative_fraction"])
            margin = conflict - align - min_conflict_over_align
            if conflict < min_conflict_ratio or margin <= 0.0 or negative_fraction < min_negative_fraction:
                continue
            neg_strength = (negative_fraction - min_negative_fraction) / max(1.0 - min_negative_fraction, 1.0e-12)
            score = margin * max(0.0, min(1.0, neg_strength))
            key = f"{row['param_name']}::{expert}"
            item = {
                "key": key,
                "source": spec.label,
                "task": task,
                "owner": owner,
                "expert": expert,
                "param_name": str(row["param_name"]),
                "score": score,
                "align_ratio": align,
                "conflict_ratio": conflict,
                "orth_ratio": float(row["orth_ratio"]),
                "negative_fraction": negative_fraction,
                "cosine_mean": float(row["cosine_mean"]),
            }
            if key not in selected or score > float(selected[key]["score"]):
                selected[key] = item
    return selected


def collect_projection_coefficients(
    *,
    specs: list[ProbeSpec],
    candidates: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    manifest_dir: Path,
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

    target_params = {str(item["param_name"]) for item in candidates.values()}
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
                source_candidates = [item for item in candidates.values() if item["source"] == spec.label]
                if not source_candidates:
                    continue
                by_task_param: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
                for item in source_candidates:
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
                    for param_name, items in by_task_param.items():
                        row_task, param = param_name
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
                            expert = str(item["expert"])
                            if expert not in expert_updates:
                                entry = entry_for(entries_by_param[param], expert)
                                delta = load_delta(entry.storage_path)
                                expert_updates[expert] = apply_linear_delta(delta=delta, selected_inputs=selected_inputs)
                            update = expert_updates[expert]
                            numerator = float((update.float() * owner_update.float()).sum().item())
                            key = str(item["key"])
                            stats[key]["numerator"] += numerator
                            stats[key]["denominator"] += denom
                            stats[key]["count"] += 1.0
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
        output[key] = {**item, "alpha": alpha}
    return output


def materialize_projected_mode(
    *,
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    output_dir: Path,
    entry_index: Mapping[str, ProbeEntry],
    candidates: Mapping[str, Mapping[str, Any]],
    alpha_stats: Mapping[str, Mapping[str, float]],
    projection_strength: float,
    max_abs_alpha: float,
) -> list[dict[str, Any]]:
    edits = []
    changed_keys = set(candidates)
    for raw in manifest["basis_entries"]:
        key = f"{raw['param_name']}::{raw['expert']}"
        source = manifest_dir / str(raw["storage_path"])
        target = output_dir / str(raw["storage_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if key not in changed_keys:
            os.symlink(source, target)
            continue
        candidate = candidates[key]
        alpha = float(alpha_stats.get(key, {}).get("alpha", 0.0))
        projected = alpha < 0.0
        applied_alpha = max(alpha, -float(max_abs_alpha)) if projected else 0.0
        if projected:
            expert_entry = entry_index[key]
            owner_entry = entry_index[f"{candidate['param_name']}::{candidate['owner']}"]
            expert_delta = torch.load(expert_entry.storage_path, map_location="cpu").float()
            owner_delta = torch.load(owner_entry.storage_path, map_location="cpu").float()
            updated = expert_delta - float(projection_strength) * float(applied_alpha) * owner_delta
            torch.save(updated.to(dtype=expert_delta.dtype), target)
            edit_norm = float((updated - expert_delta).norm().item())
            base_norm = float(expert_delta.norm().item())
            del expert_delta, owner_delta, updated
        else:
            os.symlink(source, target)
            edit_norm = 0.0
            base_norm = 0.0
        edits.append(
            {
                **candidate,
                "alpha": alpha,
                "applied_alpha": applied_alpha,
                "projected": bool(projected),
                "edit_norm": edit_norm,
                "base_norm": base_norm,
                "relative_edit_norm": edit_norm / max(base_norm, 1.0e-12) if base_norm > 0.0 else 0.0,
            }
        )
    return sorted(edits, key=lambda item: abs(float(item["applied_alpha"])), reverse=True)


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


def write_markdown(path: Path, metadata: Mapping[str, Any], edits: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Activation-Projected OP-VEC Mode",
        "",
        "This artifact edits task-vector tensors directly. Scalar gates are unchanged.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Top Projection Edits",
        "",
        "| rank | expert | owner | task | param | alpha | applied alpha | rel edit | score | conflict | align | orth |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(edits[:50], start=1):
        lines.append(
            f"| {rank} | {item['expert']} | {item['owner']} | {item['task']} | `{item['param_name']}` | "
            f"{float(item['alpha']):.5f} | {float(item['applied_alpha']):.5f} | "
            f"{float(item['relative_edit_norm']):.5f} | {float(item['score']):.5f} | "
            f"{float(item['conflict_ratio']):.3f} | {float(item['align_ratio']):.3f} | {float(item['orth_ratio']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
