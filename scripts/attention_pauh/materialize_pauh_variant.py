#!/usr/bin/env python3
"""Materialize PAUH gate variants from an existing PAUH score payload.

This script reuses the expensive prompt-activation scores produced by
build_prompt_attention_utility_harm_gates.py. It creates deterministic
mechanism checks and memory-safe variants without re-forwarding the base model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import (  # noqa: E402
    LayerEnergy,
    apply_coefficient_floors_preserve_mean,
    gate_values_from_layer_coefficients,
    layer_coefficients_from_scores,
    manifest_expert_names,
    summarize_coefficients,
    transform_layer_scores,
)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_gates).expanduser().resolve()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experts = tuple(args.experts.split(",")) if args.experts else manifest_expert_names(manifest)
    alpha_by_expert = parse_alpha(args.alpha, experts=experts, default=args.default_alpha)
    scores = parse_scores(source["scores"])
    transformed_scores = transform_layer_scores(
        scores,
        transform=args.score_transform,
        smooth_radius=args.smooth_radius,
        shuffle_seed=args.shuffle_seed,
    )
    coefficients = layer_coefficients_from_scores(
        transformed_scores,
        alpha_by_expert=alpha_by_expert,
        beta=args.beta,
        min_coeff=args.min_coeff,
        max_coeff=args.max_coeff,
    )
    floors = parse_layer_floors(args.layer_floor)
    if floors:
        coefficients = apply_coefficient_floors_preserve_mean(
            coefficients,
            alpha_by_expert=alpha_by_expert,
            floors=floors,
            min_coeff=args.min_coeff,
            max_coeff=args.max_coeff,
        )
    gates = gate_values_from_layer_coefficients(
        manifest,
        coefficients,
        scope=args.scope,
        alpha_by_expert=alpha_by_expert,
        mlp_residual_scale=args.mlp_residual_scale,
    )
    config = {
        "format": "prompt_attention_utility_harm_variant_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_gates": str(source_path),
        "mode_manifest": str(manifest_path),
        "variant_name": args.variant_name,
        "score_transform": args.score_transform,
        "smooth_radius": args.smooth_radius,
        "shuffle_seed": args.shuffle_seed,
        "beta": args.beta,
        "alpha_by_expert": alpha_by_expert,
        "min_coeff": args.min_coeff,
        "max_coeff": args.max_coeff,
        "scope": args.scope,
        "mlp_residual_scale": args.mlp_residual_scale,
        "layer_floor": args.layer_floor,
    }
    payload = {
        "format": "prompt_attention_utility_harm_gates_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "gates": gates,
        "coefficients": {
            expert: {str(layer): value for layer, value in sorted(values.items())}
            for expert, values in coefficients.items()
        },
        "coefficient_summary": summarize_coefficients(coefficients),
        "source_score_summary": summarize_scores(transformed_scores),
    }
    write_json(output_dir / "pauh_gates.json", payload)
    write_json(output_dir / "pauh_config.json", config)
    write_markdown_summary(output_dir / "pauh_summary.md", payload)
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "pauh_gates.json"),
                "num_gates": len(gates),
                "coefficient_summary": payload["coefficient_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gates", required=True)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--experts", default="tool,memory,code")
    parser.add_argument("--score-transform", choices=["identity", "inverse", "shuffle", "smooth", "smooth-inverse"], default="identity")
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=20260521)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--default-alpha", type=float, default=1.0)
    parser.add_argument("--alpha", default="", help="Comma-separated overrides, e.g. tool=1,memory=1,code=1")
    parser.add_argument("--min-coeff", type=float, default=0.35)
    parser.add_argument("--max-coeff", type=float, default=1.65)
    parser.add_argument("--scope", choices=["layer-all", "attn-only", "hybrid"], default="layer-all")
    parser.add_argument("--mlp-residual-scale", type=float, default=0.5)
    parser.add_argument(
        "--layer-floor",
        action="append",
        default=[],
        help="Expert/layer floor, e.g. memory:18-27=0.75 or code:10,11,12=0.6",
    )
    return parser.parse_args()


def parse_scores(raw_scores: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[int, LayerEnergy]]:
    parsed: dict[str, dict[int, LayerEnergy]] = {}
    for expert, layer_map in raw_scores.items():
        parsed[expert] = {}
        for layer, item in layer_map.items():
            parsed[expert][int(layer)] = LayerEnergy(
                utility=float(item["utility"]),
                harm=float(item["harm"]),
                raw_score=float(item["raw_score"]),
                score=float(item["score"]),
            )
    return parsed


def parse_alpha(raw: str, *, experts: tuple[str, ...], default: float) -> dict[str, float]:
    values = {expert: float(default) for expert in experts}
    for item in [part.strip() for part in str(raw or "").split(",") if part.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid --alpha item: {item}")
        key, value = item.split("=", 1)
        values[str(key).strip()] = float(value)
    return values


def parse_layer_floors(items: list[str]) -> dict[str, dict[int, float]]:
    floors: dict[str, dict[int, float]] = {}
    for item in items:
        if ":" not in item or "=" not in item:
            raise ValueError(f"Invalid --layer-floor item: {item}")
        expert, rest = item.split(":", 1)
        layer_part, value_part = rest.split("=", 1)
        value = float(value_part)
        target = floors.setdefault(expert.strip(), {})
        for layer in parse_layers(layer_part):
            target[int(layer)] = value
    return floors


def parse_layers(raw: str) -> list[int]:
    layers: list[int] = []
    for part in [item.strip() for item in str(raw).split(",") if item.strip()]:
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    return layers


def summarize_scores(scores: dict[str, dict[int, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for expert, layer_scores in scores.items():
        values = list(layer_scores.values())
        if not values:
            summary[expert] = {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        summary[expert] = {
            "count": float(len(values)),
            "mean": sum(values) / float(len(values)),
            "min": min(values),
            "max": max(values),
        }
    return summary


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# PAUH Variant: {payload['config']['variant_name']}",
        "",
        "## Config",
        "",
        f"- source: `{payload['config']['source_gates']}`",
        f"- transform: `{payload['config']['score_transform']}`",
        f"- scope: `{payload['config']['scope']}`",
        f"- beta: `{payload['config']['beta']}`",
        f"- alpha_by_expert: `{payload['config']['alpha_by_expert']}`",
        f"- min/max coeff: `{payload['config']['min_coeff']}` / `{payload['config']['max_coeff']}`",
        f"- layer_floor: `{payload['config']['layer_floor']}`",
        "",
        "## Coefficients",
        "",
        "| expert | layers | mean | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for expert, stats in sorted(payload["coefficient_summary"].items()):
        lines.append(
            f"| {expert} | {int(stats['count'])} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
