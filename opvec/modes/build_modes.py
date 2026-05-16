"""Build OP-VEC-4 physical expert-delta basis artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from opvec.config import write_json


def select_mergeable_params(
    keys: Sequence[str],
    *,
    include_regex: Sequence[str] | None = None,
    exclude_regex: Sequence[str] | None = None,
    max_params: int | None = None,
) -> list[str]:
    include_patterns = [re.compile(pattern) for pattern in include_regex or []]
    exclude_patterns = [re.compile(pattern) for pattern in exclude_regex or []]
    selected = []
    for key in sorted(str(item) for item in keys):
        if include_patterns and not any(pattern.search(key) for pattern in include_patterns):
            continue
        if any(pattern.search(key) for pattern in exclude_patterns):
            continue
        selected.append(key)
        if max_params is not None and len(selected) >= max_params:
            break
    return selected


def build_opvec4_modes(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    max_params: int | None = None,
) -> dict[str, Any]:
    """Build or plan physical expert deltas used by OP-VEC gates."""

    models = config["models"]
    experts = models["experts"]
    expert_names = _ordered_expert_names(experts)
    modes_cfg = config.get("modes", {})
    output = Path(output_dir or modes_cfg["artifact_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_dir = Path(models["base"]).expanduser()
    base_index = load_safetensors_index(base_dir)
    params = select_mergeable_params(
        list(base_index["weight_map"].keys()),
        include_regex=modes_cfg.get("include_regex"),
        exclude_regex=modes_cfg.get("exclude_regex"),
        max_params=max_params if max_params is not None else modes_cfg.get("max_params"),
    )
    if not params:
        raise ValueError("No mergeable parameters selected")

    manifest: dict[str, Any] = {
        "format": "opvec4_mode_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_set": "opvec4",
        "base_model": str(base_dir),
        "experts": dict(experts),
        "expert_names": expert_names,
        "selection": {
            "include_regex": list(modes_cfg.get("include_regex") or []),
            "exclude_regex": list(modes_cfg.get("exclude_regex") or []),
            "num_params": len(params),
            "params": params,
        },
        "physical_basis": "expert_delta",
        "gate_parameterization": "coeff_i = common + zero_mean_residual_i",
        "dry_run": dry_run,
        "basis_entries": [],
    }

    if dry_run:
        write_mode_outputs(output, manifest, diagnostics={"dry_run": True})
        return manifest

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Mode building requires torch and safetensors") from error

    dtype = _torch_dtype(torch, str(modes_cfg.get("delta_dtype", "float32")))
    base_weight_map = base_index["weight_map"]
    diagnostics: dict[str, Any] = {"experts": {}, "params": {}, "total_l2_sq_by_expert": defaultdict(float)}

    for expert_name in expert_names:
        expert_dir = Path(experts[expert_name]).expanduser()
        expert_weight_map = load_safetensors_index(expert_dir)["weight_map"]
        missing = [param for param in params if param not in expert_weight_map]
        if missing:
            raise KeyError(f"Expert {expert_name} missing selected params: {missing[:5]}")
        diagnostics["experts"][expert_name] = {"path": str(expert_dir), "num_params": len(params)}
        for param_name in params:
            base_tensor = _read_safetensor(base_dir, base_weight_map[param_name], param_name)
            expert_tensor = _read_safetensor(expert_dir, expert_weight_map[param_name], param_name)
            if list(base_tensor.shape) != list(expert_tensor.shape):
                raise ValueError(f"Shape mismatch for {expert_name}:{param_name}")
            delta = expert_tensor.to(dtype=torch.float32) - base_tensor.to(dtype=torch.float32)
            if dtype is not None:
                delta = delta.to(dtype=dtype)
            rel_path = basis_storage_path(expert_name, param_name)
            abs_path = output / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(delta, abs_path)
            l2_sq = float(delta.float().pow(2).sum().item())
            diagnostics["total_l2_sq_by_expert"][expert_name] += l2_sq
            diagnostics["params"].setdefault(param_name, {})[expert_name] = {
                "shape": list(delta.shape),
                "dtype": str(delta.dtype),
                "l2_norm": l2_sq ** 0.5,
                "max_abs": float(delta.float().abs().max().item()) if delta.numel() else 0.0,
                "storage_path": rel_path,
            }
            manifest["basis_entries"].append(
                {
                    "expert": expert_name,
                    "param_name": param_name,
                    "storage_path": rel_path,
                    "storage_format": "torch_pt",
                    "shape": list(delta.shape),
                    "dtype": str(delta.dtype),
                }
            )
    diagnostics["total_l2_norm_by_expert"] = {
        key: value ** 0.5 for key, value in diagnostics.pop("total_l2_sq_by_expert").items()
    }
    write_mode_outputs(output, manifest, diagnostics=diagnostics)
    return manifest


def write_mode_outputs(output: Path, manifest: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> None:
    write_json(output / "mode_manifest.json", manifest)
    write_json(output / "diagnostics.json", diagnostics)
    with (output / "basis_index.jsonl").open("w", encoding="utf-8") as handle:
        for entry in manifest.get("basis_entries", []):
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def basis_storage_path(expert_name: str, param_name: str) -> str:
    safe = param_name.replace(".", "__")
    return f"expert_deltas/{expert_name}/{safe}.pt"


def _ordered_expert_names(experts: Mapping[str, Any]) -> list[str]:
    names = [expert for expert in ("tool", "memory", "code") if expert in experts]
    names.extend(str(expert) for expert in experts if str(expert) not in names)
    if not names:
        raise ValueError("models.experts must contain at least one expert")
    return names


def load_safetensors_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("weight_map"), dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    return payload


def _read_safetensor(model_dir: Path, shard: str, param_name: str):
    from safetensors import safe_open

    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(param_name)


def _torch_dtype(torch_module: Any, dtype_name: str | None):
    if dtype_name is None or str(dtype_name).lower() in {"none", "keep"}:
        return None
    normalized = str(dtype_name).replace("torch.", "")
    if not hasattr(torch_module, normalized):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch_module, normalized)
