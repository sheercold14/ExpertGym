#!/usr/bin/env python3
"""Build low-rank delta factors for Tool HiddenSteer projection hooks.

The online HiddenSteer hook should not multiply by full task-vector tensors.
This builder uses existing activation-update projection diagnostics to pick a
small set of Tool-risk modules, then stores truncated low-rank factors for the
Tool owner delta and the Memory/Code non-owner deltas at those modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_PROJECTION_CSV = (
    "/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/"
    "projection_probes/tool_memory_signature_s2_20260523/activation_update_projection_summary.csv"
)
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/hiddensteer/tool_lowrank_basis_smoke_20260524"
EXPERTS = ("tool", "memory", "code")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    manifest_dir = manifest_path.parent
    candidates = select_tool_candidates(
        csv_paths=[Path(item).expanduser().resolve() for item in args.projection_csv],
        max_modules=int(args.max_modules),
        family=str(args.family),
        min_negative_fraction=float(args.min_negative_fraction),
        min_conflict_ratio=float(args.min_conflict_ratio),
        min_conflict_over_align=float(args.min_conflict_over_align),
    )
    if not candidates:
        raise SystemExit("No Tool projection candidates selected.")

    entry_index = index_manifest_entries(manifest, manifest_dir)
    coefficient_map = load_coefficients(Path(args.gate_checkpoint).expanduser() if args.gate_checkpoint else None)
    rank = int(args.rank)
    factors: dict[str, dict[str, Any]] = {}
    module_rows = []
    for param_name, info in candidates.items():
        factors[param_name] = {}
        for expert in EXPERTS:
            entry = entry_index.get(f"{param_name}::{expert}")
            if entry is None:
                raise KeyError(f"Missing mode entry for {param_name}::{expert}")
            factor = factorize_delta(
                entry["storage_path"],
                rank=rank,
                oversample=int(args.oversample),
                niter=int(args.niter),
                device=str(args.device),
                factor_dtype=str(args.factor_dtype),
            )
            factors[param_name][expert] = factor
        row = {
            **info,
            "rank": min(rank, int(min(factors[param_name]["tool"]["shape"]))),
            "coeff_tool": coefficient_map.get(f"{param_name}::tool", 1.0),
            "coeff_memory": coefficient_map.get(f"{param_name}::memory", 1.0),
            "coeff_code": coefficient_map.get(f"{param_name}::code", 1.0),
        }
        module_rows.append(row)

    factor_path = output_dir / "lowrank_factors.pt"
    torch.save(factors, factor_path)
    payload = {
        "format": "hiddensteer_tool_lowrank_basis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "projection_csv": [str(Path(item).expanduser().resolve()) for item in args.projection_csv],
        "gate_checkpoint": str(Path(args.gate_checkpoint).expanduser().resolve()) if args.gate_checkpoint else None,
        "factor_path": str(factor_path),
        "owner_expert": "tool",
        "non_owner_experts": ["memory", "code"],
        "rank": int(args.rank),
        "family": str(args.family),
        "selection": {
            "max_modules": int(args.max_modules),
            "min_negative_fraction": float(args.min_negative_fraction),
            "min_conflict_ratio": float(args.min_conflict_ratio),
            "min_conflict_over_align": float(args.min_conflict_over_align),
        },
        "runtime_semantics": (
            "For selected linear modules, estimate Tool owner update and Memory/Code residual update "
            "with low-rank task-vector factors. If the non-owner residual has a negative projection "
            "onto the Tool update, add the opposite component to the module output."
        ),
        "modules": module_rows,
    }
    write_json(output_dir / "basis_manifest.json", payload)
    write_csv(output_dir / "selected_modules.csv", module_rows)
    write_markdown(output_dir / "README.md", payload)
    print(json.dumps({"basis_manifest": str(output_dir / "basis_manifest.json"), "modules": len(module_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--projection-csv", action="append", default=[DEFAULT_PROJECTION_CSV])
    parser.add_argument("--gate-checkpoint", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--max-modules", type=int, default=4)
    parser.add_argument("--family", choices=["attention", "mlp", "all"], default="attention")
    parser.add_argument("--min-negative-fraction", type=float, default=0.70)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.03)
    parser.add_argument("--min-conflict-over-align", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--factor-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def select_tool_candidates(
    *,
    csv_paths: list[Path],
    max_modules: int,
    family: str,
    min_negative_fraction: float,
    min_conflict_ratio: float,
    min_conflict_over_align: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in csv_paths:
        for row in read_csv(path):
            if str(row.get("task")) != "tool":
                continue
            expert = str(row.get("expert"))
            if expert not in {"memory", "code"}:
                continue
            param_name = str(row["param_name"])
            row_family = str(row.get("family") or family_from_param(param_name))
            if family != "all" and row_family != family:
                continue
            conflict = float(row["conflict_ratio"])
            align = float(row["align_ratio"])
            negative_fraction = float(row["negative_fraction"])
            margin = conflict - align - min_conflict_over_align
            if conflict < min_conflict_ratio or margin <= 0.0 or negative_fraction < min_negative_fraction:
                continue
            score = margin * negative_fraction
            item = grouped.setdefault(
                param_name,
                {
                    "param_name": param_name,
                    "layer": layer_from_param(param_name),
                    "module": str(row.get("module") or module_from_param(param_name)),
                    "family": row_family,
                    "score": 0.0,
                    "max_conflict_ratio": 0.0,
                    "min_align_ratio": 1.0,
                    "max_negative_fraction": 0.0,
                    "trigger_experts": [],
                    "sources": [],
                },
            )
            item["score"] = max(float(item["score"]), score)
            item["max_conflict_ratio"] = max(float(item["max_conflict_ratio"]), conflict)
            item["min_align_ratio"] = min(float(item["min_align_ratio"]), align)
            item["max_negative_fraction"] = max(float(item["max_negative_fraction"]), negative_fraction)
            if expert not in item["trigger_experts"]:
                item["trigger_experts"].append(expert)
            if str(path) not in item["sources"]:
                item["sources"].append(str(path))
    selected = dict(sorted(grouped.items(), key=lambda item: float(item[1]["score"]), reverse=True))
    if max_modules > 0:
        selected = dict(list(selected.items())[:max_modules])
    for item in selected.values():
        item["trigger_experts"] = sorted(item["trigger_experts"])
    return selected


def factorize_delta(
    path: Path,
    *,
    rank: int,
    oversample: int,
    niter: int,
    device: str,
    factor_dtype: str,
) -> dict[str, Any]:
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    delta = torch.load(path, map_location="cpu").float()
    max_rank = max(1, min(int(rank), min(delta.shape)))
    q = min(max_rank + max(0, int(oversample)), min(delta.shape))
    work = delta.to(device=device)
    with torch.no_grad():
        u, s, v = torch.svd_lowrank(work, q=q, niter=int(niter))
        u = u[:, :max_rank].detach().cpu().to(dtype_map[factor_dtype])
        s = s[:max_rank].detach().cpu().to(torch.float32)
        v = v[:, :max_rank].detach().cpu().to(dtype_map[factor_dtype])
    del work, delta
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "u": u,
        "s": s,
        "v": v,
        "shape": [int(u.shape[0]), int(v.shape[0])],
        "rank": int(max_rank),
        "source_path": str(path),
    }


def index_manifest_entries(manifest: Mapping[str, Any], manifest_dir: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for raw in manifest.get("basis_entries", []):
        expert = str(raw["expert"])
        if expert not in EXPERTS:
            continue
        param_name = str(raw["param_name"])
        output[f"{param_name}::{expert}"] = {**raw, "storage_path": manifest_dir / str(raw["storage_path"])}
    return output


def load_coefficients(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    gates = payload.get("gates", payload if isinstance(payload, dict) else {})
    return {str(key): float(value) for key, value in gates.items() if "::" in str(key)}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# HiddenSteer Tool Low-Rank Basis",
        "",
        f"- Created: `{payload['created_at']}`",
        f"- Mode manifest: `{payload['mode_manifest']}`",
        f"- Gate checkpoint: `{payload['gate_checkpoint']}`",
        f"- Rank: `{payload['rank']}`",
        f"- Selected modules: `{len(payload['modules'])}`",
        "",
        "| rank | layer | module | param | score | conflict | align | neg frac | triggers |",
        "|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, item in enumerate(payload["modules"], start=1):
        lines.append(
            f"| {idx} | {item['layer']} | {item['module']} | `{item['param_name']}` | "
            f"{float(item['score']):.5f} | {float(item['max_conflict_ratio']):.3f} | "
            f"{float(item['min_align_ratio']):.3f} | {float(item['max_negative_fraction']):.3f} | "
            f"{','.join(item['trigger_experts'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def layer_from_param(param_name: str) -> int:
    match = LAYER_RE.search(param_name)
    return int(match.group(1)) if match else -1


def module_from_param(param_name: str) -> str:
    if ".self_attn." in param_name:
        return param_name.rsplit(".self_attn.", 1)[1].split("_proj", 1)[0]
    if ".mlp." in param_name:
        return param_name.rsplit(".mlp.", 1)[1].split("_proj", 1)[0]
    return "unknown"


def family_from_param(param_name: str) -> str:
    if ".self_attn." in param_name:
        return "attention"
    if ".mlp." in param_name:
        return "mlp"
    return "unknown"


if __name__ == "__main__":
    main()

