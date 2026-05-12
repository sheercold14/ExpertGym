"""VeRL-facing helpers for installing OP-VEC gates into a HF actor model.

This module intentionally avoids importing VeRL directly.  It is meant to be
called from a small VeRL worker/model-construction patch after the HuggingFace
actor model has been created.  The helper freezes the base model, installs
OP-VEC ``GatedLinear`` modules from a mode manifest, and returns a gate manager
whose parameters are the only trainable tensors.

Typical use inside a VeRL actor worker patch::

    from opvec.frameworks.verl_gated_actor import install_opvec_gate_actor
    gate_manager, audit = install_opvec_gate_actor(torch, model)
    optimizer = torch.optim.AdamW(gate_manager.parameters(), lr=cfg.lr)

Environment variables are supported so the same patch can be reused across
small experiments:

    OPVEC_CONFIG=configs/gated_grpo.yaml
    OPVEC_MODE_MANIFEST=/path/to/mode_manifest.json
    OPVEC_GATE_PARAMETERIZATION=global
    OPVEC_INIT_GATE_CHECKPOINT=/path/to/gates.json   # optional
    OPVEC_MAX_GATED_MODULES=0                        # 0/all, 1 smoke
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from opvec.config import load_config
from opvec.modeling.apply_gates import install_gated_linears_from_manifest
from opvec.modeling.bake import load_gate_values
from opvec.modeling.gate_parameters import make_torch_gate_manager
from opvec.modeling.manifest import manifest_param_names


PARAMETERIZED_GATES = {"parameter", "global-parameter"}


def install_opvec_gate_actor(
    torch_module: Any,
    model: Any,
    *,
    config_path: str | Path | None = None,
    mode_manifest_path: str | Path | None = None,
    gate_parameterization: str | None = None,
    init_gate_checkpoint: str | Path | None = None,
    max_gated_modules: int | None = None,
    device: str | None = None,
    freeze_base: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Install OP-VEC gates into ``model`` and freeze all non-gate weights.

    Returns ``(gate_manager, audit)``.  The caller should pass only
    ``gate_manager.parameters()`` to the optimizer when doing gate-only GRPO.
    """

    config_path = config_path or os.environ.get("OPVEC_CONFIG") or "configs/gated_grpo.yaml"
    mode_manifest_path = mode_manifest_path or os.environ.get("OPVEC_MODE_MANIFEST")
    if not mode_manifest_path:
        raise ValueError("mode_manifest_path or OPVEC_MODE_MANIFEST is required")
    gate_parameterization = normalize_gate_parameterization(
        gate_parameterization or os.environ.get("OPVEC_GATE_PARAMETERIZATION") or "global"
    )
    init_gate_checkpoint = init_gate_checkpoint or os.environ.get("OPVEC_INIT_GATE_CHECKPOINT") or None
    if max_gated_modules is None:
        env_max = os.environ.get("OPVEC_MAX_GATED_MODULES")
        max_gated_modules = int(env_max) if env_max not in (None, "", "none", "None") else None
    if max_gated_modules == 0:
        max_gated_modules = None

    config = load_config(str(config_path))
    if init_gate_checkpoint:
        config = {**config, "initial_gates": load_gate_values(init_gate_checkpoint)}

    if freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)

    param_names = manifest_param_names(mode_manifest_path) if gate_parameterization in PARAMETERIZED_GATES else None
    gate_manager = make_torch_gate_manager(
        torch_module,
        config,
        parameterization=gate_parameterization,
        param_names=param_names,
    )
    if device:
        gate_manager = gate_manager.to(device)

    installed = install_gated_linears_from_manifest(
        torch_module,
        model,
        mode_manifest_path=mode_manifest_path,
        gate_manager=gate_manager,
        max_modules=max_gated_modules,
        device=device,
    )

    # GatedLinear calls into gate_manager, but keeping it as an attribute helps
    # framework code discover the trainable gate module after patching.
    setattr(model, "opvec_gate_manager", gate_manager)
    setattr(model, "opvec_gate_audit", {
        "config_path": str(config_path),
        "mode_manifest_path": str(mode_manifest_path),
        "gate_parameterization": gate_parameterization,
        "init_gate_checkpoint": str(init_gate_checkpoint) if init_gate_checkpoint else None,
        "installed_modules": installed,
        "max_gated_modules": max_gated_modules,
        "trainable_parameter_names": trainable_parameter_names(gate_manager),
    })
    return gate_manager, model.opvec_gate_audit


def trainable_parameter_names(module: Any) -> list[str]:
    """Return gate-manager parameter names that require gradients."""

    output: list[str] = []
    for name, param in module.named_parameters():
        if getattr(param, "requires_grad", False):
            output.append(name)
    return output


def normalize_gate_parameterization(value: str) -> str:
    aliases = {
        "layer_band": "layer-band",
        "global_parameter": "global-parameter",
        "global_param": "global-parameter",
        "param": "parameter",
    }
    return aliases.get(str(value), str(value))
