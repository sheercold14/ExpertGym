"""Experimental VeRL external_lib hook for OP-VEC Gated-GRPO.

VeRL can import user supplied ``actor_rollout_ref.model.external_libs`` before
loading the Hugging Face actor.  This file provides a deliberately small import
hook that patches ``transformers.AutoModelForCausalLM.from_pretrained`` so the
loaded model is immediately converted into an OP-VEC gated policy.

Why this exists
---------------
The recommended first path remains ``scripts/train/opvec_gated_grpo_loop.py``:
it is auditable and only optimizes task-vector coefficients.  This hook is for
experiments where you want VeRL's GRPO runner to own rollout/training while the
actor still exposes only gate parameters as trainable.

Required environment variables
------------------------------
OPVEC_ENABLE_VERL_PATCH=1
OPVEC_CONFIG=/path/to/configs/gated_grpo.yaml
OPVEC_MODE_MANIFEST=/path/to/mode_manifest.json

Optional environment variables
------------------------------
OPVEC_GATE_PARAMETERIZATION=global|layer-band|parameter|global-parameter
OPVEC_GATE_CHECKPOINT=/path/to/gates.json
OPVEC_MAX_GATED_MODULES=0      # 0/empty means all modules; use 1 for smoke tests
OPVEC_FREEZE_BASE=1            # default: freeze non-gate parameters

Caveat
------
This is an integration shim, not a stable VeRL fork.  If a VeRL version changes
its model loader order, use the native OP-VEC loop and keep this hook as a
reference implementation for a custom VeRL worker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_PATCHED = False
_ORIGINAL_FROM_PRETRAINED = None


def maybe_patch_transformers() -> bool:
    """Install the AutoModelForCausalLM patch when explicitly enabled.

    Returns True when the patch is active.  Importing this module is otherwise a
    no-op, which keeps ordinary OP-VEC scripts safe.
    """

    global _PATCHED, _ORIGINAL_FROM_PRETRAINED
    if _PATCHED:
        return True
    if not _env_truthy("OPVEC_ENABLE_VERL_PATCH", default=False):
        return False
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as error:  # pragma: no cover - only relevant inside VeRL envs
        raise RuntimeError("OP-VEC VeRL patch requires transformers") from error

    _ORIGINAL_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained

    def _patched_from_pretrained(*args: Any, **kwargs: Any):
        model = _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)
        return install_opvec_gates_into_model(model)

    AutoModelForCausalLM.from_pretrained = _patched_from_pretrained  # type: ignore[assignment]
    _PATCHED = True
    return True


def install_opvec_gates_into_model(model: Any) -> Any:
    """Attach OP-VEC gated Linear modules to an already-loaded HF model."""

    config_path = os.environ.get("OPVEC_CONFIG")
    mode_manifest = os.environ.get("OPVEC_MODE_MANIFEST")
    if not config_path or not mode_manifest:
        raise RuntimeError("Set OPVEC_CONFIG and OPVEC_MODE_MANIFEST before enabling OPVEC_ENABLE_VERL_PATCH=1")

    import torch

    from opvec.config import load_config
    from opvec.modeling.apply_gates import install_gated_linears_from_manifest
    from opvec.modeling.bake import load_gate_values
    from opvec.modeling.gate_parameters import make_torch_gate_manager
    from opvec.modeling.manifest import manifest_param_names

    config = load_config(config_path)
    gate_checkpoint = os.environ.get("OPVEC_GATE_CHECKPOINT")
    if gate_checkpoint:
        config = {**config, "initial_gates": load_gate_values(gate_checkpoint)}

    parameterization = _normalize_gate_parameterization(os.environ.get("OPVEC_GATE_PARAMETERIZATION", "global"))
    param_names = manifest_param_names(mode_manifest) if parameterization in {"parameter", "global-parameter"} else None
    gate_manager = make_torch_gate_manager(
        torch,
        config,
        parameterization=parameterization,
        param_names=param_names,
    )

    # Register as submodule so VeRL/FSDP optimizers can see gate parameters.
    model.opvec_gate_manager = gate_manager
    max_modules = _parse_max_gated_modules(os.environ.get("OPVEC_MAX_GATED_MODULES"))
    installed = install_gated_linears_from_manifest(
        torch,
        model,
        mode_manifest_path=mode_manifest,
        gate_manager=gate_manager,
        max_modules=max_modules,
        device=None,
    )
    model.opvec_installed_modules = installed
    model.opvec_gate_parameterization = parameterization

    if _env_truthy("OPVEC_FREEZE_BASE", default=True):
        for name, param in model.named_parameters():
            if not name.startswith("opvec_gate_manager."):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
    return model


def _parse_max_gated_modules(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    value = int(str(raw))
    return None if value <= 0 else value


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_gate_parameterization(value: str) -> str:
    aliases = {
        "layer_band": "layer-band",
        "param": "parameter",
        "param-coefficients": "parameter",
        "parameter-coefficients": "parameter",
        "global_parameter": "global-parameter",
        "global-param": "global-parameter",
        "global_param": "global-parameter",
        "global-residual": "global-parameter",
        "global_residual": "global-parameter",
    }
    return aliases.get(str(value), str(value))


# VeRL imports external_libs for side effects.
maybe_patch_transformers()
