"""Gate-only actor helpers for running OP-VEC inside verl."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Mapping

from .path_utils import ensure_opvec_on_path

ensure_opvec_on_path()

from opvec.frameworks.verl_gated_actor import install_opvec_gate_actor as _install_opvec_gate_actor  # noqa: E402


def install_opvec_gate_actor(
    torch_module: Any,
    model: Any,
    *,
    config_path: str | Path,
    mode_manifest_path: str | Path,
    gate_parameterization: str = "global",
    init_gate_checkpoint: str | Path | None = None,
    max_gated_modules: int | None = None,
    device: str | None = None,
    freeze_base: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Install OP-VEC gates and freeze all non-gate model parameters."""

    return _install_opvec_gate_actor(
        torch_module,
        model,
        config_path=config_path,
        mode_manifest_path=mode_manifest_path,
        gate_parameterization=gate_parameterization,
        init_gate_checkpoint=init_gate_checkpoint,
        max_gated_modules=max_gated_modules,
        device=device,
        freeze_base=freeze_base,
    )


def export_effective_hf_state_dict(model: Any, *, dtype: Any | None = None) -> dict[str, Any]:
    """Materialize a normal HF state_dict from a model patched with OP-VEC gates.

    This is the missing bridge needed by verl's actor-to-vLLM weight sync.  vLLM
    should receive ordinary HF keys, not ``GatedLinear`` internal keys.
    """

    return dict(iter_effective_hf_state_dict(model, dtype=dtype))


def has_opvec_gate_actor(model: Any) -> bool:
    """Return True when ``model`` contains an OP-VEC gate manager."""

    return _find_gate_manager(model) is not None


def gate_metrics(model: Any, *, prefix: str = "opvec/gates", exact_limit: int = 16) -> dict[str, float]:
    """Return lightweight scalar metrics for the current gate manager."""

    gate_manager = _find_gate_manager(model)
    if gate_manager is None or not hasattr(gate_manager, "gate_values"):
        return {}
    try:
        values = gate_manager.gate_values()
    except Exception:
        return {}
    scalars = {str(key): float(value) for key, value in values.items() if _is_finite_number(value)}
    if not scalars:
        return {}

    metrics: dict[str, float] = {
        f"{prefix}/count": float(len(scalars)),
        f"{prefix}/mean": sum(scalars.values()) / len(scalars),
        f"{prefix}/min": min(scalars.values()),
        f"{prefix}/max": max(scalars.values()),
    }
    if len(scalars) > 1:
        mean = metrics[f"{prefix}/mean"]
        metrics[f"{prefix}/std"] = math.sqrt(sum((value - mean) ** 2 for value in scalars.values()) / len(scalars))
    else:
        metrics[f"{prefix}/std"] = 0.0

    by_expert: dict[str, list[float]] = {"tool": [], "memory": [], "code": []}
    for key, value in scalars.items():
        for expert in by_expert:
            if key == expert or key.endswith(f"::{expert}") or key.endswith(f".{expert}"):
                by_expert[expert].append(value)
    for expert, expert_values in by_expert.items():
        if expert_values:
            metrics[f"{prefix}/{expert}_mean"] = sum(expert_values) / len(expert_values)
            metrics[f"{prefix}/{expert}_min"] = min(expert_values)
            metrics[f"{prefix}/{expert}_max"] = max(expert_values)

    if len(scalars) <= exact_limit:
        for key, value in sorted(scalars.items()):
            metrics[f"{prefix}/exact/{_metric_safe_key(key)}"] = value
    return metrics


def iter_effective_hf_state_dict(model: Any, *, dtype: Any | None = None) -> Iterator[tuple[str, Any]]:
    """Yield a normal HF state_dict with OP-VEC gated Linear weights materialized.

    The implementation works from ``state_dict()`` rather than live module
    parameters so it can be used after FSDP wrapping, where module attributes may
    hold sharded/flattened views.
    """

    state = model.state_dict()
    gate_manager = _find_gate_manager(model)
    if gate_manager is None:
        yield from _iter_plain_state_dict(state, dtype=dtype)
        return

    gated_prefixes = {
        key[: -len(".base_linear.weight")]
        for key in state
        if key.endswith(".base_linear.weight")
    }
    yielded: set[str] = set()
    for prefix in sorted(gated_prefixes):
        base_key = f"{prefix}.base_linear.weight"
        out_key = f"{prefix}.weight"
        param_name = f"{_strip_framework_prefix(prefix)}.weight"
        weight = _effective_weight_from_state(
            state,
            prefix=prefix,
            gate_manager=gate_manager,
            param_name=param_name,
        )
        yielded.add(out_key)
        yield out_key, _maybe_to_dtype(weight, dtype)
        bias_key = f"{prefix}.base_linear.bias"
        if bias_key in state:
            out_bias_key = f"{prefix}.bias"
            yielded.add(out_bias_key)
            yield out_bias_key, _maybe_to_dtype(_detach(state[bias_key]), dtype)

    for key, tensor in state.items():
        if key in yielded:
            continue
        if _is_gate_manager_key(key):
            continue
        if _is_internal_gated_linear_key(key, gated_prefixes):
            continue
        yielded.add(key)
        yield key, _maybe_to_dtype(_detach(tensor), dtype)


def _is_opvec_gated_linear(module: Any) -> bool:
    return all(hasattr(module, attr) for attr in ("base_linear", "gate_manager", "param_name"))


def _effective_linear_weight(module: Any) -> Any:
    base_weight = module.base_linear.weight.detach()
    weight = base_weight.float()
    try:
        coeffs = module.gate_manager.expert_coefficients(param_name=module.param_name)
    except TypeError:
        coeffs = module.gate_manager.expert_coefficients()
    for expert in ("tool", "memory", "code"):
        buffer_name = f"delta_{expert}"
        if not hasattr(module, buffer_name):
            continue
        delta = getattr(module, buffer_name).detach().to(device=weight.device, dtype=weight.dtype)
        raw_coeff = coeffs[expert]
        coeff = (
            raw_coeff.to(device=weight.device, dtype=weight.dtype)
            if hasattr(raw_coeff, "to")
            else weight.new_tensor(float(raw_coeff))
        )
        weight = weight + coeff * delta
    return weight.to(dtype=base_weight.dtype).detach()


def _effective_weight_from_state(
    state: Mapping[str, Any],
    *,
    prefix: str,
    gate_manager: Any,
    param_name: str,
) -> Any:
    base = _detach(state[f"{prefix}.base_linear.weight"])
    weight = base.float() if hasattr(base, "float") else base
    try:
        coeffs = gate_manager.expert_coefficients(param_name=param_name)
    except TypeError:
        coeffs = gate_manager.expert_coefficients()
    for expert in ("tool", "memory", "code"):
        delta_key = f"{prefix}.delta_{expert}"
        if delta_key not in state:
            continue
        delta = _detach(state[delta_key])
        if hasattr(delta, "to") and hasattr(weight, "device"):
            delta = delta.to(device=weight.device, dtype=weight.dtype)
        coeff = _coefficient_tensor(coeffs[expert], weight)
        weight = weight + coeff * delta
    return weight.to(dtype=base.dtype).detach() if hasattr(weight, "to") else weight


def _find_gate_manager(model: Any) -> Any | None:
    if hasattr(model, "opvec_gate_manager"):
        return model.opvec_gate_manager
    for _, module in model.named_modules():
        if hasattr(module, "opvec_gate_manager"):
            return module.opvec_gate_manager
    return None


def _coefficient_tensor(value: Any, reference: Any) -> Any:
    if hasattr(value, "to") and hasattr(reference, "device"):
        return value.to(device=reference.device, dtype=reference.dtype)
    if hasattr(reference, "new_tensor"):
        return reference.new_tensor(float(value))
    return float(value)


def _detach(tensor: Any) -> Any:
    return tensor.detach() if hasattr(tensor, "detach") else tensor


def _maybe_to_dtype(tensor: Any, dtype: Any | None) -> Any:
    if dtype is None or not hasattr(tensor, "to"):
        return tensor
    return tensor.to(dtype=dtype)


def _iter_plain_state_dict(state: Mapping[str, Any], *, dtype: Any | None) -> Iterator[tuple[str, Any]]:
    for key, tensor in state.items():
        yield key, _maybe_to_dtype(_detach(tensor), dtype)


def _is_gate_manager_key(key: str) -> bool:
    return "opvec_gate_manager." in key


def _is_internal_gated_linear_key(key: str, prefixes: set[str]) -> bool:
    for prefix in prefixes:
        if key.startswith(f"{prefix}.base_linear.") or key.startswith(f"{prefix}.delta_"):
            return True
    return False


def _strip_framework_prefix(prefix: str) -> str:
    output = prefix
    for marker in ("_fsdp_wrapped_module.", "module."):
        output = output.replace(marker, "")
    return output


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _metric_safe_key(key: str) -> str:
    return key.replace("/", "_").replace("::", "__").replace(" ", "_")
