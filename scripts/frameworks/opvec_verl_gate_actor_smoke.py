#!/usr/bin/env python3
"""Smoke-test the OP-VEC VeRL actor patch on a tiny local HF model.

This does not require VeRL.  It validates that the hook can freeze the actor,
install gates, and expose only gate-manager parameters as trainable.  Use a
small model/mode manifest for CI; for the real Qwen run, prefer the native
``scripts/train/opvec_gated_grpo_loop.py`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.frameworks.verl_gated_actor import install_opvec_gate_actor


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.torch_dtype),
        low_cpu_mem_usage=True,
    )
    gate_manager, audit = install_opvec_gate_actor(
        torch,
        model,
        config_path=args.config,
        mode_manifest_path=args.mode_manifest,
        gate_parameterization=args.gate_parameterization,
        init_gate_checkpoint=args.init_gate_checkpoint,
        max_gated_modules=args.max_gated_modules,
        device=args.device if args.device != "cpu" else None,
    )
    trainable_model_params = [name for name, p in model.named_parameters() if p.requires_grad]
    summary = {
        "audit": audit,
        "gate_parameter_count": sum(p.numel() for p in gate_manager.parameters() if p.requires_grad),
        "trainable_model_parameters": trainable_model_params,
        "num_trainable_model_parameters": len(trainable_model_params),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--gate-parameterization", choices=["global", "layer-band", "parameter", "global-parameter"], default="global")
    parser.add_argument("--init-gate-checkpoint", default=None)
    parser.add_argument("--max-gated-modules", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    return parser.parse_args()


if __name__ == "__main__":
    main()
