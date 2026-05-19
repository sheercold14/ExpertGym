#!/usr/bin/env python3
"""Build 4-expert mode artifacts with R1 delta scaled to agent-expert range.

Reads the existing raw 4-expert mode manifest, scales only the reasoning
(R1) deltas by a factor that brings their per-param norm to the mean of the
three agent experts (tool/memory/code), and writes a new mode directory.

Agent expert deltas are symlinked (not copied) to save disk space.

Usage:
    python build_scaled_r1_modes.py \
        --source-manifest /path/to/opvec4_reasoning/mode_manifest.json \
        --output-dir /path/to/opvec4_r1scaled_YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


AGENT_EXPERTS = ("tool", "memory", "code")
REASONING_EXPERT = "reasoning"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-manifest",
        required=True,
        help="Path to existing 4-expert mode_manifest.json",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for scaled artifacts")
    p.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Manual scale factor for reasoning deltas. If omitted, computed automatically.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    return p.parse_args()


def compute_scale_factor(source_dir: Path, manifest: dict) -> tuple[float, dict]:
    """Compute s = mean_agent_per_param_norm / reasoning_per_param_norm."""
    import torch

    expert_norms: dict[str, list[float]] = defaultdict(list)
    for entry in manifest["basis_entries"]:
        delta = torch.load(
            source_dir / entry["storage_path"],
            map_location="cpu",
            weights_only=True,
        )
        norm = delta.float().norm().item()
        expert_norms[entry["expert"]].append(norm)

    stats = {}
    for expert, norms in expert_norms.items():
        stats[expert] = {
            "count": len(norms),
            "mean_norm": sum(norms) / len(norms),
            "total_norm": sum(n**2 for n in norms) ** 0.5,
        }

    agent_mean = sum(stats[e]["mean_norm"] for e in AGENT_EXPERTS) / len(AGENT_EXPERTS)
    reasoning_mean = stats[REASONING_EXPERT]["mean_norm"]
    scale = agent_mean / reasoning_mean

    stats["_scale"] = {
        "agent_per_param_mean_norm": agent_mean,
        "reasoning_per_param_mean_norm": reasoning_mean,
        "scale_factor": scale,
    }
    return scale, stats


def main() -> None:
    args = parse_args()
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_dir = source_manifest_path.parent
    output_dir = Path(args.output_dir).expanduser().resolve()

    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    # Compute or use provided scale factor
    print("[r1-scale] Computing delta norms...")
    auto_scale, norm_stats = compute_scale_factor(source_dir, manifest)
    scale = args.scale_factor if args.scale_factor is not None else auto_scale

    print(f"[r1-scale] Agent experts mean per-param norm: {norm_stats['_scale']['agent_per_param_mean_norm']:.4f}")
    print(f"[r1-scale] Reasoning mean per-param norm:     {norm_stats['_scale']['reasoning_per_param_mean_norm']:.4f}")
    print(f"[r1-scale] Scale factor: {scale:.6f}")
    for expert in list(AGENT_EXPERTS) + [REASONING_EXPERT]:
        s = norm_stats[expert]
        print(f"  {expert}: count={s['count']}, mean_norm={s['mean_norm']:.4f}, total_norm={s['total_norm']:.4f}")

    if args.dry_run:
        print("[r1-scale] Dry run, exiting.")
        return

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    new_entries = []
    n_scaled = 0
    n_linked = 0

    for entry in manifest["basis_entries"]:
        src_path = source_dir / entry["storage_path"]
        dst_path = output_dir / entry["storage_path"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if entry["expert"] == REASONING_EXPERT:
            # Scale and save
            delta = torch.load(src_path, map_location="cpu", weights_only=True)
            scaled = delta.float() * scale
            scaled = scaled.to(dtype=delta.dtype)
            torch.save(scaled, dst_path)
            n_scaled += 1
            new_entry = {**entry, "scaled_by": scale}
        else:
            # Symlink agent expert deltas
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.symlink(src_path, dst_path)
            n_linked += 1
            new_entry = dict(entry)

        new_entries.append(new_entry)

    # Write new manifest
    new_manifest = {
        **manifest,
        "format": "opvec4_r1scaled_mode_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest_path),
        "reasoning_scale_factor": scale,
        "norm_stats": norm_stats,
        "basis_entries": new_entries,
    }
    (output_dir / "mode_manifest.json").write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    # Write basis index
    with (output_dir / "basis_index.jsonl").open("w", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"[r1-scale] Done. Scaled {n_scaled} reasoning deltas, symlinked {n_linked} agent deltas.")
    print(f"[r1-scale] Output: {output_dir}")


if __name__ == "__main__":
    main()
