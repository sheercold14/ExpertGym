"""Helpers for installing gated modules."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .gated_linear import GatedLinear


def replace_child_module(root, dotted_name: str, new_module) -> None:
    """Replace a child module by dotted path."""

    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def get_child_module(root, dotted_name: str):
    current = root
    for part in dotted_name.split("."):
        current = getattr(current, part)
    return current


def install_gated_linears_from_manifest(
    torch_module: Any,
    model: Any,
    *,
    mode_manifest_path: str | Path,
    gate_manager: Any,
    max_modules: int | None = None,
    device: str | None = None,
) -> list[str]:
    """Replace selected Linear modules with GatedLinear wrappers."""

    manifest_path = Path(mode_manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode_dir = manifest_path.parent
    entries_by_param: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in manifest.get("basis_entries", []):
        entries_by_param[str(entry["param_name"])].append(entry)

    installed = []
    for param_name in sorted(entries_by_param):
        if not param_name.endswith(".weight"):
            continue
        module_name = param_name[: -len(".weight")]
        base_linear = get_child_module(model, module_name)
        target_device = _module_device(base_linear)
        load_device = device if device and str(device) != "auto" else "cpu"
        deltas = {}
        for entry in entries_by_param[param_name]:
            delta = torch_module.load(mode_dir / entry["storage_path"], map_location=load_device or "cpu")
            if load_device and str(load_device) != "cpu":
                delta = delta.to(load_device)
            deltas[str(entry["expert"])] = delta
        wrapped = GatedLinear(torch_module, base_linear, deltas, gate_manager, param_name=param_name).module
        replace_child_module(model, module_name, wrapped)
        installed.append(module_name)
        if max_modules is not None and len(installed) >= max_modules:
            break
    return installed


def _module_device(module) -> Any | None:
    for param in module.parameters(recurse=False):
        return param.device
    for buffer in module.buffers(recurse=False):
        return buffer.device
    return None
