#!/usr/bin/env python3
"""Build training-free Response-Conditioned Residual Filtering gates.

RCRF treats the functional unit of a task vector as the induced residual
``Delta W @ h`` on task response spans, not as an expert-level scalar.  The
builder consumes the signed-utility probe summary and materializes a small
gate family:

- ``rcrf``: keep useful response-conditioned residuals, mildly amplify
  cross-task agreement, and shrink conflict/noise.
- ``energy_only``: a single ablation that only uses expression energy.

This script does not train, roll out, or change reward code.  It only writes
OP-VEC gate JSON files consumable by the existing checkpoint baker.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.build_signature_preserving_gates import module_family  # noqa: E402
from scripts.attention_pauh.core import (  # noqa: E402
    default_owner_task,
    parse_layer_index,
    summarize_coefficients,
)


SENSITIVE_ROUTING_FAMILIES = frozenset({"attn_q", "attn_k"})
WRITEBACK_ATTENTION_FAMILIES = frozenset({"attn_v", "attn_o"})
MLP_FAMILIES = frozenset({"mlp_gate", "mlp_up", "mlp_down"})


@dataclass(frozen=True)
class RcrfProfile:
    name: str
    description: str
    default_alpha: float = 1.0
    utility_weight: float = 0.12
    synergy_weight: float = 0.05
    harm_weight: float = 0.18
    noise_weight: float = 0.10
    min_coeff: float = 0.55
    max_coeff: float = 1.12
    use_utility: bool = True
    use_conflict: bool = True
    use_synergy: bool = True


DEFAULT_PROFILES = (
    RcrfProfile(
        name="rcrf",
        description=(
            "Main method: response-conditioned residual filtering with owner utility, "
            "cross-task agreement, conflict, and noise."
        ),
    ),
    RcrfProfile(
        name="no_conflict",
        description="Ablation: owner utility and agreement are kept, but conflict/harm projection is disabled.",
        harm_weight=0.0,
        use_conflict=False,
    ),
    RcrfProfile(
        name="owner_only",
        description="Ablation: use only owner utility plus noise filtering, without cross-task agreement or conflict.",
        synergy_weight=0.0,
        harm_weight=0.0,
        use_conflict=False,
        use_synergy=False,
    ),
    RcrfProfile(
        name="energy_only",
        description="Ablation: filter only by DeltaW h expression energy, without signed utility or conflict.",
        utility_weight=0.08,
        synergy_weight=0.0,
        harm_weight=0.0,
        noise_weight=0.12,
        min_coeff=0.70,
        max_coeff=1.08,
        use_utility=False,
        use_conflict=False,
        use_synergy=False,
    ),
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    summary_path = Path(args.signed_utility_summary).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profiles = selected_profiles(args.profiles)
    stats_index = build_stats_index(signed_summary)
    scale_index = build_scale_index(stats_index)
    payloads = []

    for profile in profiles:
        payload = build_candidate_payload(
            manifest=manifest,
            signed_summary=signed_summary,
            stats_index=stats_index,
            scale_index=scale_index,
            profile=profile,
            mode_manifest=str(manifest_path),
            signed_utility_summary=str(summary_path),
        )
        candidate_dir = output_dir / profile.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "gates.json", payload)
        write_markdown(candidate_dir / "summary.md", payload)
        payloads.append(
            {
                "name": profile.name,
                "gate_checkpoint": str(candidate_dir / "gates.json"),
                "summary": str(candidate_dir / "summary.md"),
                "profile": asdict(profile),
                "coefficient_summary": payload["coefficient_summary"],
                "decision_summary": payload["decision_summary"],
            }
        )

    manifest_payload = {
        "format": "response_conditioned_residual_filtering_family_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "signed_utility_summary": str(summary_path),
        "output_dir": str(output_dir),
        "principle": [
            "The functional unit is DeltaW h on task response spans, not an expert scalar.",
            "Keep residuals with stable owner utility and low cross-task conflict.",
            "Mildly amplify cross-task agreement as possible synergy.",
            "Suppress low-energy, unstable, or harmful residuals as noise/conflict.",
        ],
        "candidates": payloads,
    }
    write_json(output_dir / "candidate_manifest.json", manifest_payload)
    write_family_markdown(output_dir / "README.md", manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--signed-utility-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profiles",
        default="rcrf,energy_only",
        help="Comma-separated profile names. Available: rcrf, no_conflict, owner_only, energy_only.",
    )
    return parser.parse_args()


def selected_profiles(raw: str) -> list[RcrfProfile]:
    by_name = {profile.name: profile for profile in DEFAULT_PROFILES}
    result = []
    for name in [item.strip() for item in str(raw).split(",") if item.strip()]:
        if name not in by_name:
            raise ValueError(f"Unknown RCRF profile {name!r}; available={sorted(by_name)}")
        result.append(by_name[name])
    if not result:
        raise ValueError("At least one profile is required")
    return result


def build_candidate_payload(
    *,
    manifest: Mapping[str, Any],
    signed_summary: Mapping[str, Any],
    stats_index: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    scale_index: Mapping[str, Mapping[str, float]],
    profile: RcrfProfile,
    mode_manifest: str,
    signed_utility_summary: str,
) -> dict[str, Any]:
    gates, decisions = build_rcrf_gates(
        manifest=manifest,
        signed_summary=signed_summary,
        stats_index=stats_index,
        scale_index=scale_index,
        profile=profile,
    )
    coefficients = layer_mean_coefficients(decisions)
    return {
        "format": "response_conditioned_residual_filtering_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": mode_manifest,
        "signed_utility_summary": signed_utility_summary,
        "profile": asdict(profile),
        "principle": {
            "unit": "module-level induced residual DeltaW h on task response spans",
            "keep": "stable owner utility with low harm/conflict",
            "amplify": "cross-task agreement/synergy",
            "suppress": "low energy, unstable sign, or cross-task conflict",
        },
        "gates": gates,
        "coefficients": {
            expert: {str(layer): value for layer, value in sorted(layer_map.items())}
            for expert, layer_map in coefficients.items()
        },
        "coefficient_summary": summarize_coefficients(coefficients),
        "decision_summary": summarize_decisions(decisions),
        "decision_rows": decisions,
    }


def build_rcrf_gates(
    *,
    manifest: Mapping[str, Any],
    signed_summary: Mapping[str, Any],
    stats_index: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    scale_index: Mapping[str, Mapping[str, float]],
    profile: RcrfProfile,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gates: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    probed_layers = sorted({int(layer) for expert_map in stats_index.values() for fam_map in expert_map.values() for layer in fam_map})
    for entry in manifest.get("basis_entries", []):
        param_name = str(entry["param_name"])
        expert = str(entry["expert"])
        layer = parse_layer_index(param_name)
        family = module_family(param_name)
        nearest_layer = nearest_value(probed_layers, layer)
        stats = stats_index.get(expert, {}).get(family, {}).get(nearest_layer, {})
        scales = scale_index.get(expert, {})
        coeff, metrics, reason = coefficient_for_entry(
            expert=expert,
            layer=layer,
            probed_layer=nearest_layer,
            family=family,
            stats=stats,
            scales=scales,
            signed_summary=signed_summary,
            profile=profile,
        )
        gates[f"{param_name}::{expert}"] = float(coeff)
        decisions.append(
            {
                "expert": expert,
                "layer": layer,
                "probed_layer": nearest_layer,
                "family": family,
                "param_name": param_name,
                "coefficient": float(coeff),
                "reason": reason,
                "metrics": metrics,
            }
        )
    return gates, decisions


def coefficient_for_entry(
    *,
    expert: str,
    layer: int,
    probed_layer: int,
    family: str,
    stats: Mapping[str, Any],
    scales: Mapping[str, float],
    signed_summary: Mapping[str, Any],
    profile: RcrfProfile,
) -> tuple[float, dict[str, float], str]:
    owner = default_owner_task(expert)
    owner_stats = stats.get(owner, {})
    owner_effect = float(owner_stats.get("signed_effect_mean", 0.0))
    owner_expression = float(owner_stats.get("expression_mean", 0.0))
    positive_fraction = float(owner_stats.get("positive_fraction", 0.0))
    protected = [task_stats for task, task_stats in stats.items() if str(task) != owner]
    cross_positive = mean_or_zero(max(0.0, float(item.get("signed_effect_mean", 0.0))) for item in protected)
    cross_harm = mean_or_zero(
        max(0.0, -float(item.get("signed_effect_mean", 0.0))) + float(item.get("harm_mean", 0.0))
        for item in protected
    )
    owner_scale = max(float(scales.get("owner_effect_scale", 1.0e-12)), 1.0e-12)
    expression_scale = max(float(scales.get("expression_scale", 1.0e-12)), 1.0e-12)

    owner_signal = math.tanh(max(0.0, owner_effect) / owner_scale) * max(0.0, min(1.0, positive_fraction))
    synergy_signal = math.tanh(cross_positive / owner_scale)
    direct_harm = math.tanh((cross_harm + max(0.0, -owner_effect)) / owner_scale)
    conflict_score = conflict_for_entry(
        signed_summary=signed_summary,
        target_task=owner,
        expert=expert,
        layer=probed_layer,
    )
    expression_ratio = owner_expression / expression_scale
    low_energy = 1.0 if expression_ratio < 0.35 else 0.0
    unstable = 1.0 if positive_fraction < 0.45 else 0.0
    noise_score = max(low_energy, unstable if owner_effect <= 0.0 else 0.0)

    family_boost = 1.0
    conflict_boost = 1.0
    if family in MLP_FAMILIES:
        family_boost = 1.05
    elif family in WRITEBACK_ATTENTION_FAMILIES:
        family_boost = 0.90
    elif family in SENSITIVE_ROUTING_FAMILIES:
        family_boost = 0.55
        conflict_boost = 1.35

    if profile.use_utility:
        delta = profile.utility_weight * family_boost * owner_signal
    else:
        delta = profile.utility_weight * math.tanh(max(0.0, expression_ratio - 1.0))
    if profile.use_synergy:
        delta += profile.synergy_weight * family_boost * synergy_signal
    if profile.use_conflict:
        delta -= profile.harm_weight * conflict_boost * max(direct_harm, conflict_score)
    delta -= profile.noise_weight * noise_score
    coeff = clamp(profile.default_alpha + delta, profile.min_coeff, profile.max_coeff)

    metrics = {
        "owner_effect": owner_effect,
        "owner_expression": owner_expression,
        "positive_fraction": positive_fraction,
        "cross_positive": cross_positive,
        "cross_harm": cross_harm,
        "owner_signal": owner_signal,
        "synergy_signal": synergy_signal,
        "direct_harm": direct_harm,
        "conflict_score": conflict_score,
        "expression_ratio": expression_ratio,
        "noise_score": noise_score,
        "delta": delta,
    }
    reason = classify_reason(metrics, coeff=coeff, default_alpha=profile.default_alpha)
    return coeff, metrics, reason


def build_stats_index(summary: Mapping[str, Any]) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    result: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for expert, module_map in summary.get("module_summary", {}).items():
        for param_name, task_stats in module_map.items():
            layer = parse_layer_index(str(param_name))
            family = module_family(str(param_name))
            result[str(expert)][family][layer] = task_stats
    return {expert: {family: dict(layer_map) for family, layer_map in family_map.items()} for expert, family_map in result.items()}


def build_scale_index(
    stats_index: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for expert, family_map in stats_index.items():
        owner = default_owner_task(expert)
        effects = []
        expressions = []
        for layer_map in family_map.values():
            for task_stats in layer_map.values():
                owner_stats = task_stats.get(owner, {})
                effect = float(owner_stats.get("signed_effect_mean", 0.0))
                expression = float(owner_stats.get("expression_mean", 0.0))
                if effect > 0.0:
                    effects.append(effect)
                if expression > 0.0:
                    expressions.append(expression)
        result[str(expert)] = {
            "owner_effect_scale": robust_median(effects, fallback=1.0e-12),
            "expression_scale": robust_median(expressions, fallback=1.0e-12),
        }
    return result


def conflict_for_entry(
    *,
    signed_summary: Mapping[str, Any],
    target_task: str,
    expert: str,
    layer: int,
) -> float:
    conflicts = signed_summary.get("conflict_summary", {}).get(str(target_task), {})
    values = []
    prefix = f"layer_{int(layer)}:"
    for key, stats in conflicts.items():
        if not str(key).startswith(prefix):
            continue
        pair = str(key).split(":", 1)[1].split("|")
        if str(expert) not in pair:
            continue
        negative_fraction = float(stats.get("negative_fraction", 0.0))
        cosine_mean = float(stats.get("cosine_mean", 0.0))
        values.append(max(0.0, negative_fraction - 0.5) + max(0.0, -cosine_mean))
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / float(len(values))))


def nearest_value(values: list[int], target: int) -> int:
    if not values:
        return int(target)
    return min(values, key=lambda value: (abs(int(value) - int(target)), int(value)))


def robust_median(values: list[float], *, fallback: float) -> float:
    clean = [float(value) for value in values if float(value) > 0.0 and math.isfinite(float(value))]
    if not clean:
        return float(fallback)
    return max(float(statistics.median(clean)), float(fallback))


def mean_or_zero(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / float(len(collected))


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def classify_reason(metrics: Mapping[str, float], *, coeff: float, default_alpha: float) -> str:
    if metrics["noise_score"] > 0.0 and coeff < default_alpha:
        return "suppress_low_energy_or_unstable_residual"
    if max(metrics["direct_harm"], metrics["conflict_score"]) > 0.25 and coeff < default_alpha:
        return "suppress_conflict_or_harm"
    if metrics["synergy_signal"] > 0.25 and coeff > default_alpha:
        return "amplify_cross_task_agreement"
    if metrics["owner_signal"] > 0.25 and coeff > default_alpha:
        return "keep_or_amplify_owner_utility"
    if coeff < default_alpha:
        return "mild_filter"
    if coeff > default_alpha:
        return "mild_amplify"
    return "neutral"


def layer_mean_coefficients(rows: list[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        expert = str(row["expert"])
        layer = int(row["layer"])
        sums[expert][layer] += float(row["coefficient"])
        counts[expert][layer] += 1
    return {
        expert: {layer: total / float(counts[expert][layer]) for layer, total in layer_map.items()}
        for expert, layer_map in sums.items()
    }


def summarize_decisions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = defaultdict(int)
    family_coeffs: dict[str, list[float]] = defaultdict(list)
    group_coeffs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reason_counts[str(row["reason"])] += 1
        family_coeffs[f"{row['expert']}:{row['family']}"].append(float(row["coefficient"]))
        group = "attention" if str(row["family"]).startswith("attn_") else "mlp"
        group_coeffs[f"{row['expert']}:{group}"].append(float(row["coefficient"]))
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "family_mean_coefficients": {
            key: sum(values) / float(len(values)) for key, values in sorted(family_coeffs.items())
        },
        "group_mean_coefficients": {
            key: sum(values) / float(len(values)) for key, values in sorted(group_coeffs.items())
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    profile = payload["profile"]
    lines = [
        f"# RCRF Gate: {profile['name']}",
        "",
        profile["description"],
        "",
        "## Coefficient Summary",
        "",
        "| expert | layers | mean | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for expert, stats in sorted(payload["coefficient_summary"].items()):
        lines.append(
            f"| {expert} | {int(stats['count'])} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "## Decision Counts", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in payload["decision_summary"]["reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Response-Conditioned Residual Filtering Gate Family",
        "",
        "This directory contains one main method and one minimal ablation.",
        "",
        "## Principle",
        "",
    ]
    for item in payload["principle"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Candidates", "", "| candidate | gate | memory mean | tool mean | code mean |", "| --- | --- | ---: | ---: | ---: |"])
    for candidate in payload["candidates"]:
        summary = candidate["coefficient_summary"]
        lines.append(
            f"| {candidate['name']} | `{candidate['gate_checkpoint']}` | "
            f"{summary.get('memory', {}).get('mean', 0.0):.4f} | "
            f"{summary.get('tool', {}).get('mean', 0.0):.4f} | "
            f"{summary.get('code', {}).get('mean', 0.0):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
