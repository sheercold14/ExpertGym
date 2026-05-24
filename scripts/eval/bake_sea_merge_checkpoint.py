#!/usr/bin/env python3
"""Bake a SEA-merged OP-VEC checkpoint.

SEA (Signed Energy-Amplitude) is not representable by the existing linear gate
bake path because it changes each parameter coordinate non-linearly:

    delta = sign(sum_i w_i tau_i) * |sum_i w_i tau_i|^beta
            * (sum_i |w_i tau_i|)^(1 - beta)

This script reads the OP-VEC mode manifest, computes the SEA delta per
mergeable tensor, applies tensor-level energy rescaling, and writes a standard
HuggingFace safetensors checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.modeling.bake import copy_model_sidecars, ensure_tokenizer_chat_template  # noqa: E402


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_OUTPUT = "/tmp/shared-storage/OnPolicy/checkpoints/sea_beta07_lam1_rescale_opvec4_20260522"
EXPERT_ORDER = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.beta) <= 1.0):
        raise ValueError("--beta must be in (0, 1].")
    if float(args.lam) <= 0.0:
        raise ValueError("--lam must be positive.")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(max(1, int(args.torch_threads)))

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    mode_dir = manifest_path.parent
    base_dir = Path(manifest["base_model"]).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    expert_weights = parse_expert_weights(args.expert_weights, manifest)
    entries_by_param = group_entries(manifest, expert_weights)
    base_index = read_json(base_dir / "model.safetensors.index.json")
    weight_map = base_index["weight_map"]

    config = {
        "format": "sea_merge_bake_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "SEA Merge: Signed Energy-Amplitude Merge",
        "mode_manifest": str(manifest_path),
        "base_model": str(base_dir),
        "output_dir": str(output_dir),
        "beta": float(args.beta),
        "lambda": float(args.lam),
        "eps": float(args.eps),
        "expert_weights": expert_weights,
        "tensor_energy_rescale": True,
        "num_params": len(entries_by_param),
        "num_delta_entries": sum(len(items) for items in entries_by_param.values()),
    }
    write_json(output_dir / "sea_merge_config.json", config)
    if args.plan_only:
        write_json(output_dir / "sea_merge_plan.json", {**config, "entries_by_param": entries_by_param})
        print(json.dumps({"output": str(output_dir), "plan_only": True, **config}, ensure_ascii=False, indent=2))
        return

    copy_model_sidecars(base_dir, output_dir)
    ensure_tokenizer_chat_template(output_dir / "tokenizer_config.json")

    stats_rows: list[dict[str, Any]] = []
    shard_names = sorted(set(weight_map.values()))
    for shard_idx, shard_name in enumerate(shard_names, start=1):
        print(f"[sea-bake] shard {shard_idx}/{len(shard_names)} {shard_name}", flush=True)
        shard_keys = [name for name, mapped in weight_map.items() if mapped == shard_name]
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(base_dir / shard_name, framework="pt", device="cpu") as handle:
            for key_idx, key in enumerate(shard_keys, start=1):
                tensor = handle.get_tensor(key)
                entries = entries_by_param.get(key, [])
                if entries:
                    delta, stats = sea_delta_for_param(
                        entries=entries,
                        mode_dir=mode_dir,
                        beta=float(args.beta),
                        lam=float(args.lam),
                        eps=float(args.eps),
                        torch=torch,
                    )
                    updated = tensor.to(dtype=torch.float32).add_(delta)
                    tensor = updated.to(dtype=tensor.dtype)
                    stats_rows.append({"param_name": key, **stats})
                    if key_idx % 16 == 0:
                        print(f"  updated {key_idx}/{len(shard_keys)} params in {shard_name}", flush=True)
                    del delta, updated
                tensors[key] = tensor
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        del tensors
    write_json(output_dir / "model.safetensors.index.json", base_index)
    write_csv(output_dir / "sea_tensor_stats.csv", stats_rows)
    summary = summarize_stats(config, stats_rows)
    write_json(output_dir / "sea_bake_summary.json", summary)
    print(json.dumps({"output": str(output_dir), "plan_only": False, **summary["summary"]}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1.0e-12)
    parser.add_argument(
        "--expert-weights",
        default="tool=1.0,memory=1.0,code=1.0",
        help="Comma-separated non-negative expert weights, e.g. tool=1,memory=1,code=1.",
    )
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def parse_expert_weights(raw: str, manifest: dict[str, Any]) -> dict[str, float]:
    experts = manifest_expert_names(manifest)
    weights = {expert: 0.0 for expert in experts}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected expert=weight in --expert-weights, got {item!r}")
        expert, value = item.split("=", 1)
        expert = expert.strip()
        if expert not in weights:
            raise ValueError(f"Unknown expert {expert!r}; available experts: {sorted(weights)}")
        weight = float(value)
        if weight < 0:
            raise ValueError("SEA total variation currently expects non-negative expert weights.")
        weights[expert] = weight
    return weights


def manifest_expert_names(manifest: dict[str, Any]) -> list[str]:
    raw = manifest.get("experts")
    if isinstance(raw, dict) and raw:
        names = [expert for expert in EXPERT_ORDER if expert in raw]
        names.extend(str(expert) for expert in raw if str(expert) not in names)
        return names
    names: list[str] = []
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        if expert not in names:
            names.append(expert)
    return names


def group_entries(manifest: dict[str, Any], expert_weights: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        weight = float(expert_weights.get(expert, 0.0))
        if weight == 0.0:
            continue
        entries[str(entry["param_name"])].append({**entry, "weight": weight})
    return dict(sorted(entries.items()))


def sea_delta_for_param(
    *,
    entries: list[dict[str, Any]],
    mode_dir: Path,
    beta: float,
    lam: float,
    eps: float,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    m = None
    total_variation = None
    weighted_norm_sum = 0.0
    expert_norms: dict[str, float] = {}
    shape = None
    for entry in entries:
        path = Path(str(entry["storage_path"]))
        if not path.is_absolute():
            path = mode_dir / path
        delta = torch.load(path, map_location="cpu").to(dtype=torch.float32)
        weight = float(entry["weight"])
        shape = list(delta.shape)
        weighted = delta.mul(weight)
        if m is None:
            m = torch.zeros_like(weighted, dtype=torch.float32)
            total_variation = torch.zeros_like(weighted, dtype=torch.float32)
        m.add_(weighted)
        total_variation.add_(delta.abs(), alpha=abs(weight))
        norm_value = float(torch.linalg.vector_norm(weighted).item())
        weighted_norm_sum += norm_value
        expert_norms[str(entry["expert"])] = norm_value
        del delta, weighted

    if m is None or total_variation is None:
        raise ValueError("SEA delta requested with no entries.")

    sum_norm = float(torch.linalg.vector_norm(m).item())
    tv_l1 = float(total_variation.sum().item())
    sum_l1 = float(m.abs().sum().item())
    cancellation_l1 = 1.0 - (sum_l1 / (tv_l1 + eps))
    raw = torch.sign(m)
    amplitude = m.abs().add_(eps).pow_(beta)
    total_variation.add_(eps).pow_(1.0 - beta)
    raw.mul_(amplitude).mul_(total_variation)
    raw_norm = float(torch.linalg.vector_norm(raw).item())
    budget = (sum_norm + eps) ** beta * (weighted_norm_sum + eps) ** (1.0 - beta)
    if raw_norm > 0.0 and budget > 0.0:
        raw.mul_(float(lam) * float(budget / (raw_norm + eps)))
    else:
        raw.zero_()
    output_norm = float(torch.linalg.vector_norm(raw).item())
    stats = {
        "expert_count": len(entries),
        "shape": shape,
        "numel": int(raw.numel()),
        "beta": beta,
        "lambda": lam,
        "sum_norm_l2": sum_norm,
        "weighted_expert_norm_sum_l2": weighted_norm_sum,
        "sea_budget_l2": float(budget),
        "sea_output_norm_l2": output_norm,
        "raw_norm_l2_before_rescale": raw_norm,
        "sum_l1": sum_l1,
        "total_variation_l1": tv_l1,
        "cancellation_l1": cancellation_l1,
        "output_vs_sum_norm_ratio": output_norm / (sum_norm + eps),
        "expert_norms_l2": expert_norms,
    }
    return raw, stats


def summarize_stats(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {**config, "summary": {"num_tensors": 0}}
    cancellation = [float(row["cancellation_l1"]) for row in rows]
    ratio = [float(row["output_vs_sum_norm_ratio"]) for row in rows]
    output_norms = [float(row["sea_output_norm_l2"]) for row in rows]
    sum_norms = [float(row["sum_norm_l2"]) for row in rows]
    summary = {
        "num_tensors": len(rows),
        "mean_cancellation_l1": sum(cancellation) / len(cancellation),
        "median_cancellation_l1": percentile(cancellation, 50),
        "p90_cancellation_l1": percentile(cancellation, 90),
        "mean_output_vs_sum_norm_ratio": sum(ratio) / len(ratio),
        "median_output_vs_sum_norm_ratio": percentile(ratio, 50),
        "total_sum_norm_l2_blocks": sum(sum_norms),
        "total_sea_output_norm_l2_blocks": sum(output_norms),
    }
    return {**config, "summary": summary, "tensor_stats": rows}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite or another path.")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


if __name__ == "__main__":
    main()
