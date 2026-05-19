#!/usr/bin/env python3
"""Bake FRS-QP coefficients into a merged model checkpoint.

Usage:
    python scripts/analysis/bake_frs_qp_model.py \
        --results /tmp/shared-storage/ExpertGym/analysis/frs_qp_v3/frs_qp_results.json \
        --output /tmp/shared-storage/ExpertGym/models/frs_qp_merge

    # Or with custom coefficients:
    python scripts/analysis/bake_frs_qp_model.py \
        --alpha-tool 0.574 --alpha-memory 0.341 --alpha-code 0.0 \
        --output /tmp/shared-storage/ExpertGym/models/frs_qp_custom
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_MODEL = "/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct"
EXPERTS = {
    "tool": "/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold",
    "memory": "/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B",
    "code": "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B",
}

INCLUDE_REGEX = [
    r"^model\.layers\.[0-9]+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=str, default=None, help="Path to frs_qp_results.json (reads alpha from there)")
    p.add_argument("--alpha-tool", type=float, default=None)
    p.add_argument("--alpha-memory", type=float, default=None)
    p.add_argument("--alpha-code", type=float, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dry-run", action="store_true", help="Print coefficients without saving")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine coefficients
    if args.results:
        with open(args.results) as f:
            data = json.load(f)
        alpha = data["alpha"]
    else:
        alpha = {}

    if args.alpha_tool is not None:
        alpha["tool"] = args.alpha_tool
    if args.alpha_memory is not None:
        alpha["memory"] = args.alpha_memory
    if args.alpha_code is not None:
        alpha["code"] = args.alpha_code

    print("FRS-QP Merge Coefficients:")
    for e, a in alpha.items():
        print(f"  {e}: {a:.6f}")

    if args.dry_run:
        print("\n[dry-run] No model saved.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    base_state = base_model.state_dict()
    param_names = [n for n in base_state.keys() if any(re.match(rx, n) for rx in INCLUDE_REGEX)]
    print(f"  Merge parameters: {len(param_names)}")

    # Apply per-expert scaling
    merged_state = {k: v.clone() for k, v in base_state.items()}

    for expert_name, coeff in alpha.items():
        if abs(coeff) < 1e-10:
            print(f"  Skipping {expert_name} (coeff={coeff:.6f})")
            continue

        expert_path = EXPERTS[expert_name]
        print(f"  Loading expert: {expert_name} ({expert_path}), coeff={coeff:.4f}")
        expert_model = AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        expert_state = expert_model.state_dict()

        with torch.no_grad():
            for name in param_names:
                tau = expert_state[name].to(merged_state[name].device) - base_state[name].to(merged_state[name].device)
                merged_state[name] += coeff * tau

        del expert_model, expert_state

    # Save
    base_model.load_state_dict(merged_state)
    print(f"\nSaving merged model to {output_dir}...")
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    meta = {
        "method": "FRS-QP (Fisher-Regression Shrinkage QP)",
        "alpha": alpha,
        "base_model": BASE_MODEL,
        "experts": EXPERTS,
    }
    with open(output_dir / "frs_qp_merge_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
