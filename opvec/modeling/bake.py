"""Bake OP-VEC gates into a standard HF safetensors checkpoint."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from opvec.config import write_json
from opvec.modes.gates import expert_coefficients, project_gates
from opvec.modeling.gate_parameters import DEFAULT_LAYER_BANDS, EXPERT_NAMES, layer_band_for_param


MODEL_SIDECAR_SUFFIXES = {
    ".json",
    ".model",
    ".tiktoken",
    ".py",
}
MODEL_SIDECAR_NAMES = {
    "tokenizer_config.json",
    "tokenizer.json",
    "generation_config.json",
    "special_tokens_map.json",
    "config.json",
    "merges.txt",
    "vocab.json",
}

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


def load_gate_values(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "gates" in payload and isinstance(payload["gates"], Mapping):
        payload = payload["gates"]
    if "final" in payload and isinstance(payload["final"], Mapping) and "gates" in payload["final"]:
        payload = payload["final"]["gates"]
    numeric = {str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}
    if any("::" in key for key in numeric):
        return numeric
    if _is_global_coefficient_values(numeric):
        return _global_coefficient_values(numeric)
    if any("." in key for key in numeric):
        return numeric
    return {
        "common": float(payload.get("common", payload.get("a_common", 0.5))),
        "tool_residual": float(payload.get("tool_residual", payload.get("a_tool", 0.0))),
        "memory_residual": float(payload.get("memory_residual", payload.get("a_memory", 0.0))),
        "code_residual": float(payload.get("code_residual", payload.get("a_code", 0.0))),
    }


def create_bake_plan(
    *,
    mode_manifest_path: str | Path,
    gate_values: Mapping[str, float],
    output_dir: str | Path,
    layer_bands: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(mode_manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expert_names = _manifest_expert_names(manifest)
    parameter_mode = _is_parameter_coefficient_values(gate_values)
    global_parameter_mode = parameter_mode and any(str(key).startswith("__global__::") for key in gate_values)
    layer_band_mode = (not parameter_mode) and _is_layer_band_gate_values(gate_values)
    layer_band_parameter_mode = layer_band_mode and any(str(key).startswith("__global__::") for key in gate_values)
    global_coefficient_mode = (not parameter_mode) and (not layer_band_mode) and _is_global_coefficient_values(gate_values, expert_names)
    if parameter_mode:
        projected = {str(key): float(value) for key, value in gate_values.items() if "::" in str(key)}
        coeffs = {"parameter_coefficients": projected}
    elif layer_band_mode:
        projected = project_layer_band_gates(gate_values, expert_names=expert_names)
        coeffs = {
            band: _expert_coefficients_for_experts(_band_gate_values(projected, band, expert_names=expert_names), expert_names)
            for band in _gate_band_names(projected)
        }
    elif global_coefficient_mode:
        projected = _global_coefficient_values(gate_values, expert_names)
        coeffs = dict(projected)
    else:
        projected = project_gates(gate_values).as_dict()
        coeffs = expert_coefficients(projected)
    entries_by_param: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bake_layer_bands = dict(layer_bands or DEFAULT_LAYER_BANDS)
    default_layer_band = next(iter(bake_layer_bands), "early")
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        if parameter_mode:
            coeff = float(projected.get(f"{entry['param_name']}::{expert}", 0.0))
        elif layer_band_mode:
            band = layer_band_for_param(str(entry["param_name"]), bake_layer_bands, default=default_layer_band)
            coeff = float(coeffs[band][expert])
        else:
            coeff = float(coeffs[expert])
        if coeff == 0.0:
            continue
        entries_by_param[str(entry["param_name"])].append({**entry, "coefficient": coeff})
    return {
        "format": "opvec4_bake_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "base_model": manifest["base_model"],
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "gate_values": projected,
        "expert_coefficients": coeffs,
        "gate_parameterization": (
            "global-parameter"
            if global_parameter_mode
            else (
                "parameter"
                if parameter_mode
                else (
                    "layer-band-parameter"
                    if layer_band_parameter_mode
                    else ("layer-band" if layer_band_mode else ("global-coefficient" if global_coefficient_mode else "global"))
                )
            )
        ),
        "num_params": len(entries_by_param),
        "num_delta_entries": sum(len(value) for value in entries_by_param.values()),
        "entries_by_param": entries_by_param,
    }


def bake_checkpoint(
    *,
    mode_manifest_path: str | Path,
    gate_values: Mapping[str, float],
    output_dir: str | Path,
    plan_only: bool = False,
    layer_bands: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    plan = create_bake_plan(
        mode_manifest_path=mode_manifest_path,
        gate_values=gate_values,
        output_dir=output_dir,
        layer_bands=layer_bands,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "bake_plan.json", plan)
    write_json(output / "gate_values.json", plan["gate_values"])
    if plan_only:
        return plan
    if plan["num_delta_entries"] <= 0:
        raise ValueError("Bake plan has no delta entries. Build modes without --dry-run first.")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Checkpoint baking requires torch and safetensors") from error

    mode_dir = Path(mode_manifest_path).expanduser().resolve().parent
    base_dir = Path(plan["base_model"]).expanduser()
    base_index = json.loads((base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = base_index["weight_map"]
    selected = plan["entries_by_param"]

    copy_model_sidecars(base_dir, output)
    ensure_tokenizer_chat_template(output / "tokenizer_config.json")
    for shard in sorted(set(weight_map.values())):
        shard_keys = [name for name, mapped in weight_map.items() if mapped == shard]
        tensors = {}
        with safe_open(base_dir / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensor = handle.get_tensor(key)
                entries = selected.get(key, [])
                if entries:
                    updated = tensor.to(dtype=torch.float32)
                    for entry in entries:
                        delta = torch.load(mode_dir / entry["storage_path"], map_location="cpu")
                        updated = updated + delta.to(dtype=torch.float32) * float(entry["coefficient"])
                    tensor = updated.to(dtype=tensor.dtype)
                tensors[key] = tensor
        save_file(tensors, output / shard, metadata={"format": "pt"})
    write_json(output / "model.safetensors.index.json", base_index)
    summary = {**plan, "checkpoint_frozen": True, "checkpoint_path": str(output)}
    write_json(output / "bake_summary.json", summary)
    return summary


def copy_model_sidecars(base_dir: Path, output: Path) -> list[str]:
    copied = []
    for item in base_dir.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.is_file() and (item.name in MODEL_SIDECAR_NAMES or item.suffix in MODEL_SIDECAR_SUFFIXES):
            shutil.copy2(item, output / item.name)
            copied.append(item.name)
    return copied


def ensure_tokenizer_chat_template(tokenizer_config_path: Path) -> bool:
    """Add a minimal Qwen chat template when local sidecars omit one."""

    if not tokenizer_config_path.exists():
        return False
    payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    if payload.get("chat_template"):
        return False
    payload["chat_template"] = QWEN_CHAT_TEMPLATE
    tokenizer_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def project_layer_band_gates(gate_values: Mapping[str, float], expert_names: tuple[str, ...] = EXPERT_NAMES) -> dict[str, float]:
    projected = {}
    for band in _gate_band_names(gate_values):
        band_projected = _project_common_residual_gates(_band_gate_values(gate_values, band, expert_names=expert_names), expert_names)
        for key, value in band_projected.items():
            projected[f"{band}.{key}"] = value
    return projected


def _project_common_residual_gates(values: Mapping[str, float], expert_names: tuple[str, ...]) -> dict[str, float]:
    """Project a common-plus-residual gate set for an arbitrary expert list."""

    if _is_global_coefficient_values_for_experts(values, expert_names):
        return _global_coefficient_values(values, expert_names)
    common = float(values.get("common", values.get("a_common", 0.0)))
    residuals = [
        float(values.get(f"{expert}_residual", values.get(f"a_{expert}", 0.0)))
        for expert in expert_names
    ]
    mean_residual = sum(residuals) / float(len(residuals))
    projected = {"common": common}
    for expert, residual in zip(expert_names, residuals):
        projected[f"{expert}_residual"] = residual - mean_residual
    return projected


def _expert_coefficients_for_experts(values: Mapping[str, float], expert_names: tuple[str, ...]) -> dict[str, float]:
    if _is_global_coefficient_values_for_experts(values, expert_names):
        return _global_coefficient_values(values, expert_names)
    common = float(values.get("common", 0.0))
    return {expert: common + float(values.get(f"{expert}_residual", 0.0)) for expert in expert_names}


def _is_layer_band_gate_values(gate_values: Mapping[str, float]) -> bool:
    return any("." in str(key) for key in gate_values if "::" not in str(key))


def _is_parameter_coefficient_values(gate_values: Mapping[str, float]) -> bool:
    return any("::" in str(key) and not str(key).startswith("__global__::") for key in gate_values)


def _is_global_coefficient_values_for_experts(gate_values: Mapping[str, float], expert_names: tuple[str, ...]) -> bool:
    keys = {str(key) for key in gate_values}
    return all(expert in keys for expert in expert_names) or all(f"global.{expert}" in keys for expert in expert_names)


def _is_global_coefficient_values(gate_values: Mapping[str, float], expert_names: tuple[str, ...] = EXPERT_NAMES) -> bool:
    return _is_global_coefficient_values_for_experts(gate_values, expert_names)


def _global_coefficient_values(gate_values: Mapping[str, float], expert_names: tuple[str, ...] = EXPERT_NAMES) -> dict[str, float]:
    return {
        expert: float(gate_values[expert] if expert in gate_values else gate_values[f"global.{expert}"])
        for expert in expert_names
    }


def _manifest_expert_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    configured = manifest.get("expert_names")
    if isinstance(configured, list) and configured:
        return tuple(str(item) for item in configured)
    experts = manifest.get("experts")
    if isinstance(experts, Mapping) and experts:
        ordered = [expert for expert in EXPERT_NAMES if expert in experts]
        ordered.extend(str(expert) for expert in experts if str(expert) not in ordered)
        return tuple(ordered)
    entries = manifest.get("basis_entries") or []
    names: list[str] = []
    for entry in entries:
        expert = str(entry.get("expert", ""))
        if expert and expert not in names:
            names.append(expert)
    return tuple(names or EXPERT_NAMES)


def _gate_band_names(gate_values: Mapping[str, float]) -> list[str]:
    names = sorted({str(key).split(".", 1)[0] for key in gate_values if "." in str(key)})
    return names or list(DEFAULT_LAYER_BANDS)


def _band_gate_values(
    gate_values: Mapping[str, float],
    band: str,
    expert_names: tuple[str, ...] = EXPERT_NAMES,
) -> dict[str, float]:
    values = {"common": float(gate_values.get(f"{band}.common", gate_values.get("common", 0.5)))}
    for expert in expert_names:
        values[f"{expert}_residual"] = float(
            gate_values.get(f"{band}.{expert}_residual", gate_values.get(f"{expert}_residual", 0.0))
        )
        if f"{band}.{expert}" in gate_values or expert in gate_values:
            values[expert] = float(gate_values.get(f"{band}.{expert}", gate_values.get(expert, 0.0)))
    return values
