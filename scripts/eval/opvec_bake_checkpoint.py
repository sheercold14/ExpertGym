#!/usr/bin/env python3
"""Bake OP-VEC gates into a standard model checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config
from opvec.modeling.bake import bake_checkpoint, load_gate_values


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode_manifest = args.mode_manifest or Path(config["modes"]["artifact_dir"]) / "mode_manifest.json"
    if args.gate_checkpoint:
        gate_values = load_gate_values(args.gate_checkpoint)
    else:
        gate_values = config["initial_gates"]
    output = args.output or Path(config["storage"]["root"]) / "checkpoints" / f"{config['run']['name']}-bake"
    summary = bake_checkpoint(
        mode_manifest_path=mode_manifest,
        gate_values=gate_values,
        output_dir=output,
        plan_only=args.plan_only,
    )
    print(json.dumps({"output": str(output), "num_delta_entries": summary["num_delta_entries"], "plan_only": args.plan_only}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--mode-manifest", default=None)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
