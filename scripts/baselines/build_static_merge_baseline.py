#!/usr/bin/env python3
"""Build static task-vector merging baselines from an OP-VEC mode manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.modeling.bake import copy_model_sidecars, ensure_tokenizer_chat_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True, help="OP-VEC mode_manifest.json")
    parser.add_argument("--output-dir", required=True, help="HF checkpoint output directory")
    parser.add_argument(
        "--method",
        required=True,
        choices=["task_arithmetic", "ties", "dare_ta", "dare_ties"],
        help="Static merge baseline to build.",
    )
    parser.add_argument("--scaling-coefficient", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--ties-keep-ratio",
        type=float,
        default=0.2,
        help="TIES keeps this fraction of largest-magnitude values per expert delta.",
    )
    parser.add_argument(
        "--dare-drop-rate",
        type=float,
        default=0.8,
        help="DARE random drop probability before rescaling kept deltas.",
    )
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument(
        "--experts",
        nargs="+",
        default=None,
        help="Expert names to merge. Defaults to all experts in manifest order.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it exists.")
    parser.add_argument("--plan-only", action="store_true", help="Only write baseline_config.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ties_keep_ratio <= 1.0:
        raise ValueError("--ties-keep-ratio must be in (0, 1]")
    if not 0.0 <= args.dare_drop_rate < 1.0:
        raise ValueError("--dare-drop-rate must be in [0, 1)")

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    mode_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = Path(manifest["base_model"]).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experts = tuple(args.experts or _manifest_expert_names(manifest))
    entries_by_param = _entries_by_param(manifest, experts)
    config = {
        "format": "expertgym_static_baseline_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "mode_manifest": str(manifest_path),
        "base_model": str(base_dir),
        "experts": list(experts),
        "output_dir": str(output_dir),
        "scaling_coefficient": args.scaling_coefficient,
        "ties_keep_ratio": args.ties_keep_ratio,
        "dare_drop_rate": args.dare_drop_rate,
        "seed": args.seed,
        "num_params": len(entries_by_param),
        "num_delta_entries": sum(len(items) for items in entries_by_param.values()),
    }
    _write_json(output_dir / "baseline_config.json", config)
    if args.plan_only:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    base_index = json.loads((base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = base_index["weight_map"]
    copy_model_sidecars(base_dir, output_dir)
    ensure_tokenizer_chat_template(output_dir / "tokenizer_config.json")

    summary: dict[str, Any] = {**config, "shards": []}
    for shard in sorted(set(weight_map.values())):
        shard_keys = [name for name, mapped in weight_map.items() if mapped == shard]
        tensors = {}
        shard_updated = 0
        with safe_open(base_dir / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensor = handle.get_tensor(key)
                entries = entries_by_param.get(key)
                if entries:
                    delta = _merge_param_delta(
                        method=args.method,
                        entries=entries,
                        mode_dir=mode_dir,
                        scaling_coefficient=args.scaling_coefficient,
                        ties_keep_ratio=args.ties_keep_ratio,
                        dare_drop_rate=args.dare_drop_rate,
                        seed=args.seed,
                        param_name=key,
                    )
                    tensor = (tensor.to(torch.float32) + delta).to(dtype=tensor.dtype)
                    shard_updated += 1
                tensors[key] = tensor
        save_file(tensors, output_dir / shard, metadata={"format": "pt"})
        summary["shards"].append({"name": shard, "updated_params": shard_updated})
        print(json.dumps(summary["shards"][-1], ensure_ascii=False))

    _write_json(output_dir / "model.safetensors.index.json", base_index)
    _write_json(output_dir / "baseline_summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "method": args.method}, indent=2, ensure_ascii=False))


def _manifest_expert_names(manifest: dict[str, Any]) -> list[str]:
    configured = manifest.get("expert_names")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    experts = manifest.get("experts")
    if isinstance(experts, dict) and experts:
        return [str(item) for item in experts]
    names: list[str] = []
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        if expert not in names:
            names.append(expert)
    return names


def _entries_by_param(manifest: dict[str, Any], experts: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    expert_set = set(experts)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        if expert in expert_set:
            grouped[str(entry["param_name"])][expert] = entry
    missing = {
        param: [expert for expert in experts if expert not in entries]
        for param, entries in grouped.items()
        if any(expert not in entries for expert in experts)
    }
    if missing:
        sample = next(iter(missing.items()))
        raise ValueError(f"Manifest missing expert deltas for {sample[0]}: {sample[1]}")
    return {param: [entries[expert] for expert in experts] for param, entries in grouped.items()}


def _merge_param_delta(
    *,
    method: str,
    entries: list[dict[str, Any]],
    mode_dir: Path,
    scaling_coefficient: float,
    ties_keep_ratio: float,
    dare_drop_rate: float,
    seed: int,
    param_name: str,
) -> torch.Tensor:
    deltas = [
        torch.load(mode_dir / entry["storage_path"], map_location="cpu").to(torch.float32)
        for entry in entries
    ]
    if method in {"dare_ta", "dare_ties"}:
        deltas = [
            _dare_drop_and_rescale(delta, drop_rate=dare_drop_rate, seed=seed, name=f"{param_name}:{idx}")
            for idx, delta in enumerate(deltas)
        ]
    if method in {"task_arithmetic", "dare_ta"}:
        return torch.stack(deltas, dim=0).sum(dim=0).mul(float(scaling_coefficient))
    if method in {"ties", "dare_ties"}:
        return _ties_merge(deltas, keep_ratio=ties_keep_ratio).mul(float(scaling_coefficient))
    raise ValueError(f"Unsupported method: {method}")


def _dare_drop_and_rescale(delta: torch.Tensor, *, drop_rate: float, seed: int, name: str) -> torch.Tensor:
    keep_prob = 1.0 - float(drop_rate)
    if keep_prob <= 0.0:
        raise ValueError("DARE keep probability must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, name))
    mask = torch.rand(delta.shape, dtype=torch.float32, generator=generator) < keep_prob
    return delta.mul(mask).div(keep_prob)


def _ties_merge(deltas: list[torch.Tensor], *, keep_ratio: float) -> torch.Tensor:
    stacked = torch.stack(deltas, dim=0)
    flat = stacked.reshape(stacked.shape[0], -1)
    keep_mask = torch.zeros_like(flat, dtype=torch.bool)
    kth = max(1, min(flat.shape[1], int(flat.shape[1] * (1.0 - keep_ratio))))
    for idx in range(flat.shape[0]):
        threshold = flat[idx].abs().kthvalue(kth).values
        keep_mask[idx] = flat[idx].abs() >= threshold
    masked = flat * keep_mask
    signs = torch.sign(masked.sum(dim=0))
    majority_sign = torch.sign(signs.sum())
    if majority_sign != 0:
        signs[signs == 0] = majority_sign
    selected = torch.where(signs.unsqueeze(0) > 0, masked > 0, masked < 0)
    selected_values = masked * selected
    counts = selected.sum(dim=0).to(torch.float32).clamp_min(1.0)
    merged = selected_values.sum(dim=0) / counts
    return merged.reshape_as(stacked[0])


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
