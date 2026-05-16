#!/usr/bin/env python3
"""Build a constant-initialized OP-VEC gate checkpoint."""

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
    gates, metadata = build_constant_gate_checkpoint(
        mode_manifest=args.mode_manifest,
        parameterization=args.gate_parameterization,
        value=float(args.value),
        config=config,
        include_bias=bool(args.include_bias),
        expert_names=expert_names,
    )
    payload = {
        "format": "opvec_constant_gate_checkpoint_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(Path(args.mode_manifest).expanduser().resolve()),
        "gate_parameterization": normalize_parameterization(args.gate_parameterization),
        "constant_value": float(args.value),
        "gates": gates,
        **metadata,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_constant_gate_checkpoint(
    *,
    mode_manifest: str | Path,
    parameterization: str,
    value: float,
    config: dict[str, Any] | None = None,
    include_bias: bool = False,
    expert_names: tuple[str, ...] = EXPERT_NAMES,
) -> tuple[dict[str, float], dict[str, Any]]:
    parameterization = normalize_parameterization(parameterization)
    if parameterization == "global-coefficient":
        gates: dict[str, float] = {expert: value for expert in expert_names}
    else:
        gates = {"common": value}
        for expert in expert_names:
            gates[f"{expert}_residual"] = 0.0
    param_names: list[str] = []
    band_names: list[str] = []

    if parameterization == "layer-band":
        band_names = _layer_band_names(config)
        for band_name in band_names:
            gates[f"{band_name}.common"] = value
            for expert in expert_names:
                gates[f"{band_name}.{expert}_residual"] = 0.0

    if parameterization in {"parameter", "global-parameter"}:
        param_names = manifest_param_names(mode_manifest, weight_only=not include_bias)
        if not param_names:
            raise ValueError("No mergeable parameters found in mode manifest.")
        if parameterization == "global-parameter":
            for expert in expert_names:
                gates[f"__global__::{expert}"] = value
        for param_name in param_names:
            for expert in expert_names:
                gates[f"{param_name}::{expert}"] = value

    return gates, {
        "num_mergeable_params": len(param_names),
        "num_layer_bands": len(band_names),
        "num_gate_values": len(gates),
        "experts": list(expert_names),
        "constant_init_meaning": "all effective task-vector coefficients start at the requested value",
    }


def normalize_parameterization(value: str) -> str:
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
        "global_coefficient": "global-coefficient",
        "global-coefficients": "global-coefficient",
        "global_coefficients": "global-coefficient",
        "global-direct": "global-coefficient",
        "global_direct": "global-coefficient",
        "expert-coefficient": "global-coefficient",
        "expert_coefficient": "global-coefficient",
    }
    normalized = aliases.get(str(value), str(value))
    if normalized not in {"global", "layer-band", "parameter", "global-parameter", "global-coefficient"}:
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
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--gate-parameterization",
        default="global",
        choices=["global", "layer-band", "parameter", "global-parameter", "global-coefficient"],
    )
    parser.add_argument("--include-bias", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
