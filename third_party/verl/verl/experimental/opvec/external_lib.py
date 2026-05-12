"""VeRL external_lib hook that installs OP-VEC gates into HF actors.

Set:

```text
actor_rollout_ref.model.external_lib=verl.experimental.opvec.external_lib
OPVEC_ENABLE_VERL_PATCH=1
OPVEC_CONFIG=/path/to/configs/gated_grpo.yaml
OPVEC_MODE_MANIFEST=/path/to/mode_manifest.json
```
"""

from __future__ import annotations

import os
from typing import Any

from .gate_actor import install_opvec_gate_actor

_PATCHED = False
_ORIGINAL_FROM_PRETRAINED = None


def maybe_patch_transformers() -> bool:
    global _PATCHED, _ORIGINAL_FROM_PRETRAINED
    if _PATCHED:
        return True
    if not _env_truthy("OPVEC_ENABLE_VERL_PATCH", default=False):
        return False

    from transformers import AutoModelForCausalLM

    _ORIGINAL_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained

    def _patched_from_pretrained(*args: Any, **kwargs: Any):
        model = _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)
        return install_opvec_gates_into_model(model)

    AutoModelForCausalLM.from_pretrained = _patched_from_pretrained
    _PATCHED = True
    return True


def install_opvec_gates_into_model(model: Any) -> Any:
    config_path = os.environ.get("OPVEC_CONFIG")
    mode_manifest = os.environ.get("OPVEC_MODE_MANIFEST")
    if not config_path or not mode_manifest:
        raise RuntimeError("OPVEC_CONFIG and OPVEC_MODE_MANIFEST are required when OPVEC_ENABLE_VERL_PATCH=1")

    import torch

    gate_manager, audit = install_opvec_gate_actor(
        torch,
        model,
        config_path=config_path,
        mode_manifest_path=mode_manifest,
        gate_parameterization=_normalize_gate_parameterization(os.environ.get("OPVEC_GATE_PARAMETERIZATION", "global")),
        init_gate_checkpoint=os.environ.get("OPVEC_GATE_CHECKPOINT") or None,
        max_gated_modules=_parse_max_gated_modules(os.environ.get("OPVEC_MAX_GATED_MODULES")),
        device=os.environ.get("OPVEC_GATE_DEVICE") or None,
        freeze_base=_env_truthy("OPVEC_FREEZE_BASE", default=True),
    )
    model.opvec_gate_manager = gate_manager
    model.opvec_gate_audit = audit
    model.opvec_enable_effective_weight_sync = True
    return model


def _parse_max_gated_modules(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    value = int(raw)
    return None if value <= 0 else value


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_gate_parameterization(value: str) -> str:
    aliases = {
        "global_parameter": "global-parameter",
        "global_param": "global-parameter",
        "global-param": "global-parameter",
        "layer_band": "layer-band",
        "param": "parameter",
    }
    return aliases.get(value, value)


maybe_patch_transformers()
