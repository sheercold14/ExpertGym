#!/usr/bin/env python3
"""Bake OP-VEC task-vector deltas onto a different anchor checkpoint.

The standard OP-VEC bake path applies expert deltas to the manifest base model:

    theta = theta_base + sum_i alpha_i * (theta_expert_i - theta_base)

This diagnostic script keeps the same stored deltas but applies them to an
explicit anchor model:

    theta = theta_anchor + sum_i alpha_i * (theta_expert_i - theta_manifest_base)

It is intended for compatibility probes such as adding Tool/Memory/Code agent
vectors onto a DeepSeek/R1 anchor. The default coefficients are full-strength
agent vectors: tool=1.0,memory=1.0,code=1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.modeling.bake import copy_model_sidecars, ensure_tokenizer_chat_template  # noqa: E402


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_EXPERT_WEIGHTS = "tool=1.0,memory=1.0,code=1.0"


def main() -> None:
    args = parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(max(1, int(args.torch_threads)))

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    mode_dir = manifest_path.parent
    anchor_dir = Path(args.anchor_model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    expert_weights = parse_expert_weights(args.expert_weights, manifest)
    entries_by_param = group_entries(manifest, expert_weights)
    anchor_index = read_json(anchor_dir / "model.safetensors.index.json")
    anchor_weight_map = anchor_index["weight_map"]
    validate_entries(entries_by_param, anchor_weight_map)

    plan = {
        "format": "anchor_task_vector_bake_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "anchor + OP-VEC task vectors",
        "mode_manifest": str(manifest_path),
        "manifest_base_model": manifest["base_model"],
        "anchor_model": str(anchor_dir),
        "output_dir": str(output_dir),
        "expert_models": manifest.get("experts", {}),
        "expert_weights": expert_weights,
        "num_params": len(entries_by_param),
        "num_delta_entries": sum(len(items) for items in entries_by_param.values()),
        "entries_by_param": entries_by_param,
    }
    write_json(output_dir / "anchor_task_vector_bake_plan.json", plan)
    if args.plan_only:
        print(json.dumps({"output": str(output_dir), "plan_only": True, **plan}, ensure_ascii=False, indent=2))
        return

    copy_model_sidecars(anchor_dir, output_dir)
    ensure_tokenizer_chat_template(output_dir / "tokenizer_config.json")

    stats: list[dict[str, Any]] = []
    shard_names = sorted(set(anchor_weight_map.values()))
    for shard_idx, shard_name in enumerate(shard_names, start=1):
        print(f"[anchor-tv-bake] shard {shard_idx}/{len(shard_names)} {shard_name}", flush=True)
        shard_keys = [name for name, mapped in anchor_weight_map.items() if mapped == shard_name]
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(anchor_dir / shard_name, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensor = handle.get_tensor(key)
                entries = entries_by_param.get(key, [])
                if entries:
                    updated = tensor.to(dtype=torch.float32)
                    per_expert_norms: dict[str, float] = {}
                    for entry in entries:
                        delta_path = mode_dir / str(entry["storage_path"])
                        delta = torch.load(delta_path, map_location="cpu").to(dtype=torch.float32)
                        coefficient = float(entry["coefficient"])
                        if list(delta.shape) != list(tensor.shape):
                            raise ValueError(
                                f"Delta shape mismatch for {key}: delta={list(delta.shape)} anchor={list(tensor.shape)}"
                            )
                        updated = updated + delta * coefficient
                        per_expert_norms[str(entry["expert"])] = float(torch.linalg.vector_norm(delta * coefficient).item())
                        del delta
                    stats.append(
                        {
                            "param_name": key,
                            "shape": list(tensor.shape),
                            "num_entries": len(entries),
                            "weighted_delta_norms_l2": per_expert_norms,
                            "anchor_tensor_norm_l2": float(torch.linalg.vector_norm(tensor.float()).item()),
                            "updated_tensor_norm_l2": float(torch.linalg.vector_norm(updated).item()),
                        }
                    )
                    tensor = updated.to(dtype=tensor.dtype)
                    del updated
                tensors[key] = tensor
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        del tensors

    write_json(output_dir / "model.safetensors.index.json", anchor_index)
    summary = {
        **{key: value for key, value in plan.items() if key != "entries_by_param"},
        "checkpoint_frozen": True,
        "checkpoint_path": str(output_dir),
        "tensor_stats": stats,
        "summary": summarize_stats(stats),
    }
    write_json(output_dir / "anchor_task_vector_bake_summary.json", summary)
    print(json.dumps({"output": str(output_dir), **summary["summary"]}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-weights", default=DEFAULT_EXPERT_WEIGHTS)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def parse_expert_weights(raw: str, manifest: Mapping[str, Any]) -> dict[str, float]:
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
        weights[expert] = float(value)
    return weights


def manifest_expert_names(manifest: Mapping[str, Any]) -> list[str]:
    experts = manifest.get("experts")
    if isinstance(experts, Mapping) and experts:
        ordered = [name for name in ("tool", "memory", "code") if name in experts]
        ordered.extend(str(name) for name in experts if str(name) not in ordered)
        return ordered
    names: list[str] = []
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        if expert not in names:
            names.append(expert)
    return names


def group_entries(manifest: Mapping[str, Any], expert_weights: Mapping[str, float]) -> dict[str, list[dict[str, Any]]]:
    entries: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        coefficient = float(expert_weights.get(expert, 0.0))
        if coefficient == 0.0:
            continue
        param_name = str(entry["param_name"])
        entries.setdefault(param_name, []).append({**entry, "coefficient": coefficient})
    return dict(sorted(entries.items()))


def validate_entries(entries_by_param: Mapping[str, list[dict[str, Any]]], weight_map: Mapping[str, str]) -> None:
    missing = [name for name in entries_by_param if name not in weight_map]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Anchor model is missing {len(missing)} merge parameters; first missing: {preview}")


def summarize_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_tensors": 0}
    delta_norm_totals: dict[str, float] = {}
    for row in rows:
        for expert, value in row["weighted_delta_norms_l2"].items():
            delta_norm_totals[expert] = delta_norm_totals.get(expert, 0.0) + float(value)
    return {
        "num_tensors": len(rows),
        "weighted_delta_norm_l2_block_sum": dict(sorted(delta_norm_totals.items())),
        "mean_anchor_tensor_norm_l2": sum(float(row["anchor_tensor_norm_l2"]) for row in rows) / len(rows),
        "mean_updated_tensor_norm_l2": sum(float(row["updated_tensor_norm_l2"]) for row in rows) / len(rows),
    }


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {output_dir}")
        for item in output_dir.iterdir():
            if item.is_dir():
                import shutil

                shutil.rmtree(item)
            else:
                item.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
