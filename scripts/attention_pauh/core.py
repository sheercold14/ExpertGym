"""Core utilities for PromptAttention-UtilityHarm (PAUH).

PAUH is intentionally training-free. It turns prompt-only activation statistics
into OP-VEC parameter coefficients that can be baked by the existing OP-VEC
checkpoint baker.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


ATTENTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class LayerEnergy:
    utility: float
    harm: float
    raw_score: float
    score: float


def parse_layer_index(param_name: str) -> int:
    match = LAYER_RE.search(param_name)
    if not match:
        raise ValueError(f"Cannot infer layer index from parameter name: {param_name}")
    return int(match.group(1))


def is_attention_param(param_name: str) -> bool:
    return any(param_name.endswith(suffix) for suffix in ATTENTION_SUFFIXES)


def manifest_expert_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    configured = manifest.get("expert_names")
    if isinstance(configured, list) and configured:
        return tuple(str(item) for item in configured)
    experts = manifest.get("experts")
    if isinstance(experts, Mapping) and experts:
        return tuple(str(item) for item in experts)
    names: list[str] = []
    for entry in manifest.get("basis_entries", []):
        expert = str(entry["expert"])
        if expert not in names:
            names.append(expert)
    return tuple(names)


def manifest_layers(manifest: Mapping[str, Any]) -> list[int]:
    layers = {
        parse_layer_index(str(entry["param_name"]))
        for entry in manifest.get("basis_entries", [])
        if "param_name" in entry and LAYER_RE.search(str(entry["param_name"]))
    }
    return sorted(layers)


def normalize_task_name(task: str) -> str:
    normalized = str(task or "").strip().lower()
    if normalized in {"tool", "toolcall", "tool_call", "bfcl", "toolrl"}:
        return "tool"
    if normalized in {"memory", "mem", "hotpotqa", "memagent"}:
        return "memory"
    if normalized in {"code", "coding", "cure", "livebench", "livecodebench"}:
        return "code"
    return normalized


def default_owner_task(expert: str) -> str:
    expert = str(expert)
    if expert in {"tool", "memory", "code"}:
        return expert
    if expert in {"reasoning", "r1", "deepseek"}:
        return "code"
    return normalize_task_name(expert)


def entry_activation_energy(
    delta: torch.Tensor,
    activation_diag: torch.Tensor,
    *,
    normalization: str = "delta-norm",
    eps: float = 1.0e-12,
) -> float:
    """Return diagonal-covariance approximation of E[||X @ delta.T||^2].

    ``delta`` is a linear weight delta with shape ``[out_dim, in_dim]``.
    ``activation_diag`` is E[x_i^2] for the same linear input dimension.
    With ``normalization='delta-norm'`` the score becomes activation exposure
    per unit delta norm, reducing raw task-vector scale dominance.
    """

    if delta.ndim != 2:
        raise ValueError(f"Expected a rank-2 weight delta, got shape={tuple(delta.shape)}")
    if activation_diag.numel() != delta.shape[1]:
        raise ValueError(
            f"Activation dimension mismatch: diag={activation_diag.numel()} delta_in={delta.shape[1]}"
        )
    delta_f = delta.detach().to(dtype=torch.float32, device="cpu")
    diag_f = activation_diag.detach().to(dtype=torch.float32, device="cpu")
    column_energy = delta_f.pow(2).sum(dim=0)
    raw = float(torch.dot(column_energy, diag_f).item())
    if normalization in {"none", "raw"}:
        return raw
    if normalization in {"delta-norm", "delta_norm"}:
        denom = float(column_energy.sum().item())
        return raw / max(denom, eps)
    raise ValueError(f"Unsupported energy normalization: {normalization}")


def linear_delta_probe(
    *,
    delta: torch.Tensor,
    inputs: torch.Tensor,
    output_grads: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> tuple[float, float, torch.Tensor]:
    """Estimate first-order loss effect of a linear weight delta.

    For a linear projection ``y = W x`` and an expert delta ``D``, the induced
    hidden-state change is ``D x``. If ``output_grads`` stores
    ``d loss / d y``, then ``-<d loss / d y, D x>`` is the first-order
    improvement in teacher-forced loss from adding this delta. Positive values
    therefore mean the delta points toward the supervised trajectory.

    Returns ``(expression, signed_effect, mean_update)``:
    - expression: mean ``||D x||^2`` over selected tokens;
    - signed_effect: mean negative gradient inner product over selected tokens;
    - mean_update: mean ``D x`` vector, useful for expert-conflict cosines.
    """

    if delta.ndim != 2:
        raise ValueError(f"Expected rank-2 delta, got shape={tuple(delta.shape)}")
    if inputs.ndim != 3 or output_grads.ndim != 3:
        raise ValueError(
            f"Expected rank-3 inputs/output_grads, got {tuple(inputs.shape)} and {tuple(output_grads.shape)}"
        )
    if inputs.shape[:2] != output_grads.shape[:2]:
        raise ValueError(
            f"Token shape mismatch: inputs={tuple(inputs.shape)} output_grads={tuple(output_grads.shape)}"
        )
    if delta.shape[1] != inputs.shape[-1] or delta.shape[0] != output_grads.shape[-1]:
        raise ValueError(
            "Delta shape does not match linear input/output dimensions: "
            f"delta={tuple(delta.shape)} inputs={tuple(inputs.shape)} output_grads={tuple(output_grads.shape)}"
        )

    if token_mask is None:
        selected_inputs = inputs
        selected_grads = output_grads
    else:
        mask = token_mask.to(device=inputs.device, dtype=torch.bool)
        if mask.ndim == 1:
            if mask.numel() != inputs.shape[1]:
                raise ValueError(f"1D token_mask length {mask.numel()} != seq_len {inputs.shape[1]}")
            selected_inputs = inputs[:, mask, :]
            selected_grads = output_grads[:, mask, :]
        elif mask.ndim == 2:
            if tuple(mask.shape) != tuple(inputs.shape[:2]):
                raise ValueError(f"2D token_mask shape {tuple(mask.shape)} != token shape {tuple(inputs.shape[:2])}")
            selected_inputs = inputs[mask].unsqueeze(0)
            selected_grads = output_grads[mask].unsqueeze(0)
        else:
            raise ValueError(f"Unsupported token_mask ndim={mask.ndim}")
    if selected_inputs.numel() == 0:
        zero = torch.zeros(delta.shape[0], dtype=torch.float32, device=delta.device)
        return 0.0, 0.0, zero

    delta_local = delta.to(device=selected_inputs.device, dtype=selected_inputs.dtype)
    induced = torch.einsum("bsi,oi->bso", selected_inputs, delta_local)
    selected_grads = selected_grads.to(dtype=induced.dtype)
    expression = induced.pow(2).sum(dim=-1).mean()
    signed_effect = -(selected_grads * induced).sum(dim=-1).mean()
    mean_update = induced.mean(dim=(0, 1)).detach().to(dtype=torch.float32, device="cpu")
    return float(expression.detach().float().item()), float(signed_effect.detach().float().item()), mean_update


def aggregate_layer_energy_scores(
    *,
    layer_task_energy: Mapping[str, Mapping[int, Mapping[str, float]]],
    experts: Iterable[str],
    layers: Iterable[int],
    tasks: Iterable[str],
    eps: float = 1.0e-12,
) -> dict[str, dict[int, LayerEnergy]]:
    """Build z-scored utility-harm layer scores per expert."""

    task_list = [normalize_task_name(task) for task in tasks]
    result: dict[str, dict[int, LayerEnergy]] = {}
    for expert in experts:
        owner = default_owner_task(expert)
        raw_by_layer: dict[int, tuple[float, float, float]] = {}
        for layer in layers:
            task_energy = layer_task_energy.get(expert, {}).get(int(layer), {})
            utility = float(task_energy.get(owner, 0.0))
            protected = [float(task_energy.get(task, 0.0)) for task in task_list if task != owner]
            harm = sum(protected) / float(len(protected)) if protected else 0.0
            raw_score = math.log(utility + eps) - math.log(harm + eps)
            raw_by_layer[int(layer)] = (utility, harm, raw_score)
        raw_values = [item[2] for item in raw_by_layer.values()]
        mean = sum(raw_values) / float(len(raw_values)) if raw_values else 0.0
        variance = sum((value - mean) ** 2 for value in raw_values) / float(len(raw_values)) if raw_values else 0.0
        std = math.sqrt(variance)
        expert_scores: dict[int, LayerEnergy] = {}
        for layer, (utility, harm, raw_score) in raw_by_layer.items():
            score = 0.0 if std <= eps else (raw_score - mean) / std
            expert_scores[layer] = LayerEnergy(
                utility=utility,
                harm=harm,
                raw_score=raw_score,
                score=score,
            )
        result[expert] = expert_scores
    return result


def layer_coefficients_from_scores(
    scores: Mapping[str, Mapping[int, LayerEnergy | float]],
    *,
    alpha_by_expert: Mapping[str, float],
    beta: float,
    min_coeff: float,
    max_coeff: float,
) -> dict[str, dict[int, float]]:
    """Convert layer scores to coefficients while preserving expert budgets."""

    coeffs: dict[str, dict[int, float]] = {}
    for expert, layer_scores in scores.items():
        alpha = float(alpha_by_expert.get(expert, 0.75))
        layers = sorted(int(layer) for layer in layer_scores)
        raw_scores = []
        for layer in layers:
            value = layer_scores[layer]
            raw_scores.append(float(value.score if isinstance(value, LayerEnergy) else value))
        weights = [math.exp(float(beta) * score) for score in raw_scores]
        mean_weight = sum(weights) / float(len(weights)) if weights else 1.0
        weights = [weight / max(mean_weight, 1.0e-12) for weight in weights]
        weights = _renormalize_clipped_weights(
            weights,
            target_mean=1.0,
            lower=min_coeff / alpha if alpha > 0.0 else 0.0,
            upper=max_coeff / alpha if alpha > 0.0 else 0.0,
        )
        coeffs[expert] = {layer: float(alpha * weight) for layer, weight in zip(layers, weights)}
    return coeffs


def transform_layer_scores(
    scores: Mapping[str, Mapping[int, LayerEnergy | float]],
    *,
    transform: str,
    smooth_radius: int = 1,
    shuffle_seed: int = 0,
) -> dict[str, dict[int, float]]:
    """Return transformed per-expert layer scores for PAUH ablations.

    These transformations are intentionally deterministic so the resulting
    gates can be used as mechanism checks in a paper table.
    """

    import random

    normalized = str(transform or "identity").strip().lower()
    result: dict[str, dict[int, float]] = {}
    for expert, layer_scores in scores.items():
        layers = sorted(int(layer) for layer in layer_scores)
        values = []
        for layer in layers:
            value = layer_scores[layer]
            values.append(float(value.score if isinstance(value, LayerEnergy) else value))
        if normalized in {"identity", "none"}:
            transformed = values
        elif normalized == "inverse":
            transformed = [-value for value in values]
        elif normalized == "shuffle":
            transformed = list(values)
            rng = random.Random(int(shuffle_seed) + stable_string_hash(str(expert)))
            rng.shuffle(transformed)
        elif normalized == "smooth":
            transformed = smooth_values(values, radius=smooth_radius)
        elif normalized in {"smooth-inverse", "inverse-smooth"}:
            transformed = [-value for value in smooth_values(values, radius=smooth_radius)]
        else:
            raise ValueError(f"Unsupported PAUH score transform: {transform}")
        result[str(expert)] = {layer: float(value) for layer, value in zip(layers, transformed)}
    return result


def smooth_values(values: list[float], *, radius: int) -> list[float]:
    if radius <= 0:
        return list(values)
    smoothed: list[float] = []
    for index in range(len(values)):
        lo = max(0, index - int(radius))
        hi = min(len(values), index + int(radius) + 1)
        window = values[lo:hi]
        smoothed.append(sum(window) / float(len(window)))
    return smoothed


def stable_string_hash(value: str) -> int:
    total = 0
    for char in str(value):
        total = (total * 131 + ord(char)) % 1_000_000_007
    return total


def apply_coefficient_floors_preserve_mean(
    coefficients: Mapping[str, Mapping[int, float]],
    *,
    alpha_by_expert: Mapping[str, float],
    floors: Mapping[str, Mapping[int, float]],
    min_coeff: float,
    max_coeff: float,
    eps: float = 1.0e-10,
) -> dict[str, dict[int, float]]:
    """Apply expert/layer coefficient floors and preserve each expert mean.

    The adjustment is a simple bounded water-fill around the already generated
    PAUH coefficients. It is used for mechanism ablations such as memory-safe
    late-layer floors, not for the default PAUH path.
    """

    adjusted: dict[str, dict[int, float]] = {}
    for expert, layer_values in coefficients.items():
        alpha = float(alpha_by_expert.get(expert, 0.75))
        layers = sorted(int(layer) for layer in layer_values)
        if not layers:
            adjusted[str(expert)] = {}
            continue
        lower = {
            layer: max(float(min_coeff), float(floors.get(expert, {}).get(layer, min_coeff)))
            for layer in layers
        }
        upper = {layer: max(float(max_coeff), lower[layer]) for layer in layers}
        values = {
            layer: min(max(float(layer_values[layer]), lower[layer]), upper[layer])
            for layer in layers
        }
        target_sum = alpha * float(len(layers))
        min_sum = sum(lower.values())
        max_sum = sum(upper.values())
        target_sum = min(max(target_sum, min_sum), max_sum)
        values = _shift_values_to_sum(values, lower=lower, upper=upper, target_sum=target_sum, eps=eps)
        adjusted[str(expert)] = values
    return adjusted


def _shift_values_to_sum(
    values: Mapping[int, float],
    *,
    lower: Mapping[int, float],
    upper: Mapping[int, float],
    target_sum: float,
    eps: float,
) -> dict[int, float]:
    shifted = {int(layer): float(value) for layer, value in values.items()}
    for _ in range(128):
        diff = float(target_sum) - sum(shifted.values())
        if abs(diff) <= eps:
            break
        if diff > 0.0:
            free = [layer for layer, value in shifted.items() if value < upper[layer] - eps]
            if not free:
                break
            step = diff / float(len(free))
            for layer in free:
                shifted[layer] = min(upper[layer], shifted[layer] + step)
        else:
            free = [layer for layer, value in shifted.items() if value > lower[layer] + eps]
            if not free:
                break
            step = (-diff) / float(len(free))
            for layer in free:
                shifted[layer] = max(lower[layer], shifted[layer] - step)
    return shifted


def gate_values_from_layer_coefficients(
    manifest: Mapping[str, Any],
    coefficients: Mapping[str, Mapping[int, float]],
    *,
    scope: str = "layer-all",
    alpha_by_expert: Mapping[str, float] | None = None,
    mlp_residual_scale: float = 0.5,
) -> dict[str, float]:
    """Materialize OP-VEC parameter gate values from layer coefficients."""

    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in {"layer-all", "attn-only", "attention-only", "hybrid"}:
        raise ValueError(f"Unsupported PAUH scope: {scope}")
    gates: dict[str, float] = {}
    for entry in manifest.get("basis_entries", []):
        param_name = str(entry["param_name"])
        expert = str(entry["expert"])
        if normalized_scope in {"attn-only", "attention-only"} and not is_attention_param(param_name):
            continue
        layer = parse_layer_index(param_name)
        coeff = float(coefficients.get(expert, {}).get(layer, 0.0))
        if normalized_scope == "hybrid" and not is_attention_param(param_name):
            alpha = float((alpha_by_expert or {}).get(expert, 0.75))
            coeff = alpha + float(mlp_residual_scale) * (coeff - alpha)
        if coeff != 0.0:
            gates[f"{param_name}::{expert}"] = coeff
    return gates


def summarize_coefficients(coefficients: Mapping[str, Mapping[int, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for expert, layer_values in coefficients.items():
        values = [float(value) for _, value in sorted(layer_values.items())]
        if not values:
            summary[expert] = {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        summary[expert] = {
            "count": float(len(values)),
            "mean": sum(values) / float(len(values)),
            "min": min(values),
            "max": max(values),
        }
    return summary


def task_layer_energy_from_entries(
    *,
    manifest: Mapping[str, Any],
    activation_diags: Mapping[str, Mapping[str, torch.Tensor]],
    mode_dir: str,
    normalization: str,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute expert/layer/task attention exposure from activation diagonals."""

    from pathlib import Path

    root = Path(mode_dir)
    sums: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for entry in manifest.get("basis_entries", []):
        param_name = str(entry["param_name"])
        if not is_attention_param(param_name):
            continue
        expert = str(entry["expert"])
        layer = parse_layer_index(param_name)
        delta = torch.load(root / str(entry["storage_path"]), map_location="cpu")
        for task, task_diags in activation_diags.items():
            diag = task_diags.get(param_name)
            if diag is None:
                continue
            energy = entry_activation_energy(delta, diag, normalization=normalization)
            sums[expert][layer][task] += energy
            counts[expert][layer][task] += 1
        del delta
    averaged: dict[str, dict[int, dict[str, float]]] = {}
    for expert, layer_map in sums.items():
        averaged[expert] = {}
        for layer, task_map in layer_map.items():
            averaged[expert][layer] = {}
            for task, total in task_map.items():
                averaged[expert][layer][task] = total / float(max(counts[expert][layer][task], 1))
    return averaged


def _renormalize_clipped_weights(
    weights: list[float],
    *,
    target_mean: float,
    lower: float,
    upper: float,
) -> list[float]:
    if not weights:
        return []
    if upper <= 0.0:
        return [0.0 for _ in weights]
    lower = max(0.0, lower)
    upper = max(lower, upper)

    def clipped_mean(scale: float) -> float:
        return sum(min(max(weight * scale, lower), upper) for weight in weights) / float(len(weights))

    min_mean = clipped_mean(0.0)
    max_mean = clipped_mean(1.0e12)
    if target_mean <= min_mean:
        return [lower for _ in weights]
    if target_mean >= max_mean:
        return [upper for _ in weights]
    lo, hi = 0.0, 1.0
    while clipped_mean(hi) < target_mean:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if clipped_mean(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    return [min(max(weight * hi, lower), upper) for weight in weights]
