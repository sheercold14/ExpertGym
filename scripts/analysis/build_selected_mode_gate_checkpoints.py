#!/usr/bin/env python3
"""Build OP-VEC gate checkpoints from expert-specific selected modes.

The output checkpoints use parameter coefficients so selected modes can be kept
exactly at the requested coefficient and non-selected modes can be pruned to 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import write_json
from opvec.modeling.manifest import manifest_param_names


def main() -> None:
    args = parse_args()
    expert_sources = _parse_expert_sources(args.expert_selected_modes)
    if not expert_sources and not args.reasoning_selected_modes:
        raise SystemExit("Provide at least one --expert-selected-modes or --reasoning-selected-modes.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[dict[str, Any]] = []
    if expert_sources:
        created.append(
            build_expert_prune_checkpoint(
                mode_manifest=Path(args.mode_manifest).expanduser(),
                expert_sources=expert_sources,
                output=output_dir / args.expert_prune_name,
                selected_coefficient=args.selected_coefficient,
                pruned_coefficient=args.pruned_coefficient,
                top_k_per_expert=args.top_k_per_expert,
                diagnostics_path=Path(args.diagnostics).expanduser() if args.diagnostics else None,
            )
        )
    if args.reasoning_selected_modes:
        created.append(
            build_reasoning_add_checkpoint(
                mode_manifest=Path(args.reasoning_mode_manifest).expanduser(),
                selected_modes_path=Path(args.reasoning_selected_modes).expanduser(),
                output=output_dir / args.reasoning_add_name,
                base_experts=[item.strip() for item in args.base_experts.split(",") if item.strip()],
                reasoning_expert=args.reasoning_expert,
                base_coefficient=args.base_coefficient,
                reasoning_coefficient=args.reasoning_coefficient,
            )
        )

    write_json(output_dir / "manifest.json", {"created_at": _now(), "checkpoints": created})
    print(json.dumps({"output_dir": str(output_dir), "checkpoints": created}, ensure_ascii=False, indent=2, sort_keys=True))


def build_expert_prune_checkpoint(
    *,
    mode_manifest: Path,
    expert_sources: dict[str, Path],
    output: Path,
    selected_coefficient: float,
    pruned_coefficient: float,
    top_k_per_expert: int | None = None,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_json(mode_manifest)
    param_names = manifest_param_names(mode_manifest)
    expert_names = _manifest_expert_names(manifest)
    selected_by_expert: dict[str, set[str]] = {}
    selected_order_by_expert: dict[str, list[str]] = {}
    supplement_counts: dict[str, int] = {}
    selected_source_counts: dict[str, int] = {}
    missing_by_expert: dict[str, list[str]] = {}

    for expert, source in expert_sources.items():
        selected_order = _selected_param_names_ordered(source, source_expert=expert)
        selected = set(selected_order)
        selected_source_counts[expert] = len(selected_order)
        mapped_order = [name for name in selected_order if name in param_names]
        if top_k_per_expert is not None and len(mapped_order) < top_k_per_expert:
            ranked = _delta_l2_ranked_params(
                diagnostics_path=diagnostics_path or mode_manifest.parent / "diagnostics.json",
                expert=expert,
            )
            seen = set(mapped_order)
            for param_name in ranked:
                if param_name in param_names and param_name not in seen:
                    mapped_order.append(param_name)
                    seen.add(param_name)
                if len(mapped_order) >= top_k_per_expert:
                    break
        if top_k_per_expert is not None:
            mapped_order = mapped_order[:top_k_per_expert]
        mapped = set(mapped_order)
        selected_order_by_expert[expert] = mapped_order
        selected_by_expert[expert] = mapped
        supplement_counts[expert] = max(0, len(mapped_order) - len([name for name in selected_order if name in param_names]))
        missing_by_expert[expert] = sorted(selected.difference(param_names))

    gates: dict[str, float] = {}
    for param_name in param_names:
        for expert in expert_names:
            if expert in selected_by_expert:
                value = selected_coefficient if param_name in selected_by_expert[expert] else pruned_coefficient
            else:
                value = selected_coefficient
            gates[f"{param_name}::{expert}"] = float(value)

    payload = {
        "format": "opvec_selected_mode_gate_checkpoint_v1",
        "created_at": _now(),
        "mode_manifest": str(mode_manifest.resolve()),
        "gate_parameterization": "parameter",
        "variant": "expert_specific_structured_prune",
        "selected_coefficient": float(selected_coefficient),
        "pruned_coefficient": float(pruned_coefficient),
        "experts": expert_names,
        "expert_selected_modes": {expert: str(path.resolve()) for expert, path in expert_sources.items()},
        "top_k_per_expert": top_k_per_expert,
        "selected_source_counts": selected_source_counts,
        "supplement_counts_from_delta_l2": supplement_counts,
        "selected_mapped_counts": {expert: len(params) for expert, params in selected_by_expert.items()},
        "selected_order_by_expert": selected_order_by_expert,
        "missing_selected_modes": missing_by_expert,
        "num_mergeable_params": len(param_names),
        "num_gate_values": len(gates),
        "gates": gates,
    }
    write_json(output, payload)
    _write_checkpoint_summary(output.with_suffix(".md"), payload, selected_by_expert, selected_order_by_expert)
    return {
        "name": output.name,
        "path": str(output.resolve()),
        "variant": payload["variant"],
        "selected_mapped_counts": payload["selected_mapped_counts"],
        "supplement_counts_from_delta_l2": supplement_counts,
        "num_gate_values": payload["num_gate_values"],
    }


def build_reasoning_add_checkpoint(
    *,
    mode_manifest: Path,
    selected_modes_path: Path,
    output: Path,
    base_experts: list[str],
    reasoning_expert: str,
    base_coefficient: float,
    reasoning_coefficient: float,
) -> dict[str, Any]:
    manifest = _load_json(mode_manifest)
    param_names = manifest_param_names(mode_manifest)
    expert_names = _manifest_expert_names(manifest)
    if reasoning_expert not in expert_names:
        raise ValueError(f"Reasoning expert {reasoning_expert!r} not found in manifest experts: {expert_names}")

    selected = _selected_param_names(selected_modes_path)
    mapped = selected.intersection(param_names)
    missing = sorted(selected.difference(param_names))

    gates: dict[str, float] = {}
    for param_name in param_names:
        for expert in expert_names:
            if expert in base_experts:
                value = base_coefficient
            elif expert == reasoning_expert:
                value = reasoning_coefficient if param_name in mapped else 0.0
            else:
                value = 0.0
            gates[f"{param_name}::{expert}"] = float(value)

    payload = {
        "format": "opvec_selected_mode_gate_checkpoint_v1",
        "created_at": _now(),
        "mode_manifest": str(mode_manifest.resolve()),
        "gate_parameterization": "parameter",
        "variant": "init1_plus_reasoning_selected_modes",
        "base_experts": base_experts,
        "base_coefficient": float(base_coefficient),
        "reasoning_expert": reasoning_expert,
        "reasoning_coefficient": float(reasoning_coefficient),
        "reasoning_selected_modes": str(selected_modes_path.resolve()),
        "reasoning_selected_source_count": len(selected),
        "reasoning_selected_mapped_count": len(mapped),
        "missing_reasoning_selected_modes": missing,
        "experts": expert_names,
        "num_mergeable_params": len(param_names),
        "num_gate_values": len(gates),
        "gates": gates,
    }
    write_json(output, payload)
    _write_checkpoint_summary(output.with_suffix(".md"), payload, {reasoning_expert: mapped}, {reasoning_expert: sorted(mapped)})
    return {
        "name": output.name,
        "path": str(output.resolve()),
        "variant": payload["variant"],
        "reasoning_selected_mapped_count": len(mapped),
        "num_gate_values": payload["num_gate_values"],
    }


def _parse_expert_sources(items: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--expert-selected-modes expects EXPERT=PATH, got: {item}")
        expert, path = item.split("=", 1)
        expert = expert.strip()
        if not expert:
            raise ValueError(f"Empty expert in --expert-selected-modes: {item}")
        sources[expert] = Path(path).expanduser()
    return sources


def _selected_param_names(path: Path, *, source_expert: str | None = None) -> set[str]:
    return set(_selected_param_names_ordered(path, source_expert=source_expert))


def _selected_param_names_ordered(path: Path, *, source_expert: str | None = None) -> list[str]:
    payload = _load_json(path)
    rows = _rows_from_payload(payload, source_path=path)
    if not isinstance(rows, list):
        raise ValueError(f"Could not find selected rows in {path}")

    rows = sorted(rows, key=_row_sort_key)
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        basis_id = str(row.get("basis_id") or row.get("mode") or row.get("name") or "")
        if not basis_id:
            continue
        expert, param_name = _split_basis_id(basis_id)
        row_expert = str(row.get("expert") or expert or "")
        if source_expert is not None and row_expert and row_expert != source_expert:
            continue
        if param_name not in seen:
            selected.append(param_name)
            seen.add(param_name)
    return selected


def _rows_from_payload(payload: Any, *, source_path: Path) -> list[dict[str, Any]] | None:
    if isinstance(payload, dict) and isinstance(payload.get("rows_path"), str):
        rows_path = Path(payload["rows_path"]).expanduser()
        if rows_path.exists() and rows_path.suffix == ".jsonl":
            return _read_jsonl_rows(rows_path)
    if isinstance(payload, dict) and isinstance(payload.get("selected"), list):
        return payload["selected"]
    if isinstance(payload, dict) and isinstance(payload.get("top_rows"), list):
        return payload["top_rows"]
    if isinstance(payload, dict) and isinstance(payload.get("plans"), list):
        return payload["plans"]
    if isinstance(payload, dict) and isinstance(payload.get("coefficients"), dict):
        return [{"basis_id": key, "coefficient": value} for key, value in payload["coefficients"].items() if isinstance(value, (int, float))]
    if isinstance(payload, dict):
        numeric = [{"basis_id": key, "coefficient": value} for key, value in payload.items() if isinstance(value, (int, float))]
        return numeric or None
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return None


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _row_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    rank = row.get("selected_rank")
    if rank is not None:
        try:
            return (0, float(rank), str(row.get("basis_id") or ""))
        except (TypeError, ValueError):
            pass
    for key in ("score", "boundary_score", "proposal_score", "prior_direction_utility", "utility"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return (1, -float(value), str(row.get("basis_id") or ""))
    value = row.get("z", row.get("coefficient", 0.0))
    if isinstance(value, (int, float)):
        return (2, -abs(float(value)), str(row.get("basis_id") or ""))
    return (3, 0.0, str(row.get("basis_id") or ""))


def _delta_l2_ranked_params(*, diagnostics_path: Path, expert: str) -> list[str]:
    diagnostics = _load_json(diagnostics_path)
    params = diagnostics.get("params", {})
    rows = []
    for param_name, by_expert in params.items():
        if not isinstance(by_expert, dict) or expert not in by_expert:
            continue
        value = by_expert[expert].get("l2_norm")
        if isinstance(value, (int, float)):
            rows.append((float(value), str(param_name)))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in rows]


def _split_basis_id(basis_id: str) -> tuple[str | None, str]:
    if ":" not in basis_id:
        return None, basis_id
    expert, param_name = basis_id.split(":", 1)
    return expert, param_name


def _manifest_expert_names(manifest: dict[str, Any]) -> list[str]:
    configured = manifest.get("expert_names")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    experts = manifest.get("experts")
    if isinstance(experts, dict) and experts:
        preferred = ["tool", "memory", "code", "reasoning"]
        ordered = [expert for expert in preferred if expert in experts]
        ordered.extend(str(expert) for expert in experts if str(expert) not in ordered)
        return ordered
    seen: list[str] = []
    for entry in manifest.get("basis_entries", []):
        expert = str(entry.get("expert", ""))
        if expert and expert not in seen:
            seen.append(expert)
    return seen


def _write_checkpoint_summary(
    path: Path,
    payload: dict[str, Any],
    selected_by_expert: dict[str, set[str]],
    selected_order_by_expert: dict[str, list[str]] | None = None,
) -> None:
    lines = [
        f"# {payload['variant']}",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- mode_manifest: `{payload['mode_manifest']}`",
        f"- gate_parameterization: `{payload['gate_parameterization']}`",
        f"- checkpoint: `{path.with_suffix('.json')}`",
        f"- num_mergeable_params: `{payload['num_mergeable_params']}`",
        f"- num_gate_values: `{payload['num_gate_values']}`",
        f"- top_k_per_expert: `{payload.get('top_k_per_expert')}`",
        "",
        "## Selected Counts",
        "",
        "| expert | selected mapped | supplemented from delta L2 |",
        "|---|---:|---:|",
    ]
    for expert in sorted(selected_by_expert):
        supplement = payload.get("supplement_counts_from_delta_l2", {}).get(expert, 0)
        lines.append(f"| {expert} | {len(selected_by_expert[expert])} | {supplement} |")
    lines.extend(["", "## Selected Params", ""])
    for expert in sorted(selected_by_expert):
        lines.append(f"### {expert}")
        ordered = (selected_order_by_expert or {}).get(expert) or sorted(selected_by_expert[expert])
        lines.extend(f"- `{name}`" for name in ordered)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default="/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json")
    parser.add_argument("--reasoning-mode-manifest", default="/tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expert-selected-modes",
        action="append",
        default=[],
        help="Expert-specific selected modes as EXPERT=PATH. May be repeated.",
    )
    parser.add_argument("--selected-coefficient", type=float, default=1.0)
    parser.add_argument("--pruned-coefficient", type=float, default=0.0)
    parser.add_argument("--top-k-per-expert", type=int, default=None, help="If set, keep at most this many params per selected expert and supplement short rankings with delta-L2 top params.")
    parser.add_argument("--diagnostics", default=None, help="Diagnostics JSON used for delta-L2 top-k supplementation. Defaults to mode_manifest/diagnostics.json.")
    parser.add_argument("--expert-prune-name", default="expert_selected_prune_init1.parameter.json")
    parser.add_argument("--reasoning-selected-modes", default=None)
    parser.add_argument("--reasoning-add-name", default="init1_plus_reasoning64_z0001.parameter.json")
    parser.add_argument("--base-experts", default="tool,memory,code")
    parser.add_argument("--reasoning-expert", default="reasoning")
    parser.add_argument("--base-coefficient", type=float, default=1.0)
    parser.add_argument("--reasoning-coefficient", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    main()
