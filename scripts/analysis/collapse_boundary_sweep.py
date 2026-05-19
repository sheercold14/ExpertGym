#!/usr/bin/env python3
"""Sweep one expert coefficient while fixing others → bake checkpoints for eval.

Generates a grid of gate-value JSON files and bakes each into a HF checkpoint.
After baking, run eval separately (see --help for examples).

Usage examples:

  # 1. Sweep tool coefficient (fix memory=0.75, code=0.75)
  python scripts/analysis/collapse_boundary_sweep.py \
    --sweep tool --range 0.0 1.2 0.1 \
    --fix memory=0.75 code=0.75 \
    --output-dir /tmp/shared-storage/ExpertGym/analysis/collapse_sweep/tool_sweep

  # 2. Sweep memory coefficient (fix tool=0.60, code=0.75)
  python scripts/analysis/collapse_boundary_sweep.py \
    --sweep memory --range 0.0 1.2 0.1 \
    --fix tool=0.60 code=0.75 \
    --output-dir /tmp/shared-storage/ExpertGym/analysis/collapse_sweep/memory_sweep

  # 3. Plan only (no baking, just generate gate JSONs)
  python scripts/analysis/collapse_boundary_sweep.py \
    --sweep tool --range 0.0 1.2 0.1 \
    --fix memory=0.75 code=0.75 \
    --output-dir /tmp/sweep --plan-only

  # 4. With 4 experts (R1-scaled)
  python scripts/analysis/collapse_boundary_sweep.py \
    --sweep tool --range 0.0 1.2 0.1 \
    --fix memory=0.75 code=0.75 reasoning=0.0 \
    --config configs/gated_grpo_4expert_r1scaled.yaml \
    --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4_r1scaled_20260518/mode_manifest.json \
    --output-dir /tmp/sweep

After baking, evaluate with:
  bash scripts/analysis/run_collapse_sweep_eval.sh /tmp/shared-storage/ExpertGym/analysis/collapse_sweep/tool_sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="Expert to sweep: tool, memory, code, reasoning")
    p.add_argument("--range", nargs=3, type=float, metavar=("START", "END", "STEP"),
                   default=[0.0, 1.2, 0.1], help="Sweep range: start end step")
    p.add_argument("--fix", nargs="+", metavar="EXPERT=VALUE",
                   help="Fixed expert coefficients, e.g. memory=0.75 code=0.75")
    p.add_argument("--config", default="configs/gated_grpo.yaml",
                   help="Config YAML (determines mode manifest and layer bands)")
    p.add_argument("--mode-manifest", default=None,
                   help="Override mode manifest path")
    p.add_argument("--output-dir", required=True, help="Root output directory for sweep")
    p.add_argument("--plan-only", action="store_true", help="Only generate gate JSONs, skip baking")
    p.add_argument("--py", default="/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python",
                   help="Python interpreter for bake script")
    return p.parse_args()


def frange(start: float, end: float, step: float) -> list[float]:
    vals = []
    v = start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_expert = args.sweep
    sweep_values = frange(*args.range)
    fixed = {}
    if args.fix:
        for pair in args.fix:
            k, v = pair.split("=")
            fixed[k.strip()] = float(v.strip())

    all_experts = sorted(set([sweep_expert] + list(fixed.keys())))
    print(f"[sweep] Expert: {sweep_expert}")
    print(f"[sweep] Range: {args.range[0]} → {args.range[1]}, step {args.range[2]} ({len(sweep_values)} points)")
    print(f"[sweep] Fixed: {fixed}")
    print(f"[sweep] Output: {output_dir}")
    print()

    # Generate gate JSONs and bake plan
    manifest = []
    for val in sweep_values:
        coeffs = dict(fixed)
        coeffs[sweep_expert] = val
        tag = f"{sweep_expert}_{val:.4f}"
        point_dir = output_dir / tag

        # Write gate JSON (global-coefficient format)
        gate_path = point_dir / "gates.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(coeffs, indent=2) + "\n", encoding="utf-8")

        checkpoint_dir = point_dir / "checkpoint"
        entry = {
            "tag": tag,
            "sweep_expert": sweep_expert,
            "sweep_value": val,
            "coefficients": coeffs,
            "gate_json": str(gate_path),
            "checkpoint_dir": str(checkpoint_dir),
            "baked": False,
        }
        manifest.append(entry)

        if not args.plan_only:
            if (checkpoint_dir / "config.json").exists():
                print(f"[bake] {tag}: reusing existing checkpoint")
                entry["baked"] = True
                continue
            print(f"[bake] {tag}: baking {coeffs}")
            cmd = [
                args.py, str(REPO_ROOT / "scripts" / "eval" / "opvec_bake_checkpoint.py"),
                "--config", args.config,
                "--gate-checkpoint", str(gate_path),
                "--output", str(checkpoint_dir),
            ]
            if args.mode_manifest:
                cmd += ["--mode-manifest", args.mode_manifest]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [ERROR] bake failed: {result.stderr[-500:]}")
                entry["baked"] = False
            else:
                entry["baked"] = True
                print(f"  done → {checkpoint_dir}")

    # Write manifest
    manifest_path = output_dir / "sweep_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    # Write summary
    summary = {
        "sweep_expert": sweep_expert,
        "sweep_range": list(args.range),
        "sweep_values": sweep_values,
        "fixed_coefficients": fixed,
        "num_points": len(manifest),
        "num_baked": sum(1 for e in manifest if e["baked"]),
        "config": args.config,
        "output_dir": str(output_dir),
    }
    (output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"[sweep] Generated {len(manifest)} points, baked {summary['num_baked']}")
    print(f"[sweep] Manifest: {manifest_path}")
    print()
    if not args.plan_only:
        print("Next: run eval on baked checkpoints:")
        print(f"  bash scripts/analysis/run_collapse_sweep_eval.sh {output_dir}")


if __name__ == "__main__":
    main()
