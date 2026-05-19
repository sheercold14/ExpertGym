#!/usr/bin/env python3
"""Build an explicit zero-initialized OP-VEC gate checkpoint.

The checkpoint means: start exactly from the base model, then learn task-vector
coefficients from data. For parameterized gates we write every mergeable
parameter/expert coefficient as 0.0 so fallback config defaults cannot silently
pull the initialization back to common=0.5.
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

from opvec.config import load_config, write_json
from opvec.modeling.gate_parameters import DEFAULT_LAYER_BANDS, EXPERT_NAMES
from opvec.modeling.manifest import manifest_param_names


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else None
    expert_names = _manifest_expert_names(args.mode_manifest)
    gates, metadata = build_zero_gate_checkpoint(
        mode_manifest=args.mode_manifest,
        parameterization=args.gate_parameterization,
        config=config,
        include_bias=bool(args.include_bias),
        expert_names=expert_names,
    )
    payload = {
        "format": "opvec_zero_gate_checkpoint_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(Path(args.mode_manifest).expanduser().resolve()),
        "gate_parameterization": normalize_parameterization(args.gate_parameterization),
        "gates": gates,
        **metadata,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_zero_gate_checkpoint(
    *,
    mode_manifest: str | Path,
    parameterization: str,
    config: dict[str, Any] | None = None,
    include_bias: bool = False,
    expert_names: tuple[str, ...] = EXPERT_NAMES,
) -> tuple[dict[str, float], dict[str, Any]]:
    parameterization = normalize_parameterization(parameterization)
    if parameterization == "global-coefficient":
        gates: dict[str, float] = {expert: 0.0 for expert in expert_names}
    else:
        gates = {"common": 0.0}
        for expert in expert_names:
            gates[f"{expert}_residual"] = 0.0
    param_names: list[str] = []

    if parameterization == "layer-band":
        for band_name in _layer_band_names(config):
            gates[f"{band_name}.common"] = 0.0
            for expert in expert_names:
                gates[f"{band_name}.{expert}_residual"] = 0.0

    if parameterization == "layer-band-coefficient":
        for band_name in _layer_band_names(config):
            for expert in expert_names:
                gates[f"{band_name}.{expert}"] = 0.0

    if parameterization == "layer-band-parameter":
        for expert in expert_names:
            gates[f"__global__::{expert}"] = 0.0
        for band_name in _layer_band_names(config):
            for expert in expert_names:
                gates[f"{band_name}.{expert}"] = 0.0

    if parameterization in {"parameter", "global-parameter"}:
        param_names = manifest_param_names(mode_manifest, weight_only=not include_bias)
        if not param_names:
            raise ValueError(
                "No mergeable parameters found in mode manifest. "
                "Use a real mode manifest with basis_entries, not a dry-run manifest."
            )
        if parameterization == "global-parameter":
            for expert in expert_names:
                gates[f"__global__::{expert}"] = 0.0
        for param_name in param_names:
            for expert in expert_names:
                gates[f"{param_name}::{expert}"] = 0.0

    return gates, {
        "num_mergeable_params": len(param_names),
        "num_gate_values": len(gates),
        "experts": list(expert_names),
        "zero_init_meaning": "all task-vector coefficients are 0.0; policy starts as the base model",
    }


def normalize_parameterization(value: str) -> str:
    aliases = {
        "layer_band": "layer-band",
        "layer_band_coefficient": "layer-band-coefficient",
        "layer-band-coefficients": "layer-band-coefficient",
        "layer_band_coefficients": "layer-band-coefficient",
        "layer-band-direct": "layer-band-coefficient",
        "layer_band_direct": "layer-band-coefficient",
        "layer_band_parameter": "layer-band-parameter",
        "layer-band-param": "layer-band-parameter",
        "layer_band_param": "layer-band-parameter",
        "layer-band-residual": "layer-band-parameter",
        "layer_band_residual": "layer-band-parameter",
        "layer-band-hierarchical": "layer-band-parameter",
        "layer_band_hierarchical": "layer-band-parameter",
        "hierarchical-layer-band": "layer-band-parameter",
        "hierarchical_layer_band": "layer-band-parameter",
        "param": "parameter",
        "param-coefficients": "parameter",
        "parameter-coefficients": "parameter",
        "global_parameter": "global-parameter",
        "global-param": "global-parameter",
        "global_param": "global-parameter",
        "global-residual": "global-parameter",
        "global_residual": "global-parameter",
        "global_coefficient": "global-coefficient",
        "global-coefficients": "global-coefficient",
        "global_coefficients": "global-coefficient",
        "global-direct": "global-coefficient",
        "global_direct": "global-coefficient",
        "expert-coefficient": "global-coefficient",
        "expert_coefficient": "global-coefficient",
    }
    normalized = aliases.get(str(value), str(value))
    if normalized not in {"global", "layer-band", "layer-band-coefficient", "layer-band-parameter", "parameter", "global-parameter", "global-coefficient"}:
        raise ValueError(f"Unknown gate parameterization: {value}")
    return normalized


def _layer_band_names(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return list(DEFAULT_LAYER_BANDS)
    raw = config.get("layer_bands") or config.get("modes", {}).get("layer_bands") or DEFAULT_LAYER_BANDS
    return [str(name) for name in raw.keys()]


def _manifest_expert_names(mode_manifest: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(mode_manifest).expanduser().read_text(encoding="utf-8"))
    configured = payload.get("expert_names")
    if isinstance(configured, list) and configured:
        return tuple(str(item) for item in configured)
    experts = payload.get("experts")
    if isinstance(experts, dict) and experts:
        ordered = [expert for expert in EXPERT_NAMES if expert in experts]
        ordered.extend(str(expert) for expert in experts if str(expert) not in ordered)
        return tuple(ordered)
    return EXPERT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None, help="Optional config, only needed to mirror custom layer-band names.")
    parser.add_argument(
        "--gate-parameterization",
        default="global-parameter",
        choices=["global", "layer-band", "layer-band-coefficient", "layer-band-parameter", "parameter", "global-parameter", "global-coefficient"],
    )
    parser.add_argument("--include-bias", action="store_true", help="Include non-weight manifest entries if present.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
