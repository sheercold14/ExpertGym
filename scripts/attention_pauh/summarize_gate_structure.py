#!/usr/bin/env python3
"""Summarize OP-VEC gate checkpoints by expert, layer, and module family.

The project has two gate formats:

- full OP-VEC gates: ``model.layers.N.xxx.weight::expert``;
- layer-band TRC gates: ``layerN.expert``.

This script expands both into a common structure with help from the mode
manifest, then reports expert/family/layer statistics. It is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import parse_layer_index  # noqa: E402


LAYER_GATE_RE = re.compile(r"^layer(\d+)\.([A-Za-z0-9_-]+)$")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summaries = []
    for raw_path in args.gate_json:
        path = Path(raw_path).expanduser().resolve()
        summary = summarize_gate_file(path, manifest=manifest)
        summaries.append(summary)
    payload = {
        "format": "gate_structure_summary_v1",
        "mode_manifest": str(manifest_path),
        "summaries": summaries,
    }
    if args.output_json:
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default="/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json")
    parser.add_argument("--gate-json", action="append", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def summarize_gate_file(path: Path, *, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates") or payload.get("final_gates") or payload
    flat_rows = expand_gate_rows(gates, manifest=manifest)
    return {
        "path": str(path),
        "source_format": infer_gate_format(gates),
        "num_raw_gates": len(gates),
        "num_expanded_rows": len(flat_rows),
        "expert_summary": summarize_by(flat_rows, ("expert",)),
        "family_summary": summarize_by(flat_rows, ("expert", "family")),
        "group_summary": summarize_groups(flat_rows),
        "layer_summary": summarize_by(flat_rows, ("expert", "layer")),
    }


def infer_gate_format(gates: Mapping[str, Any]) -> str:
    keys = list(gates)
    if any("::" in str(key) for key in keys):
        return "full_param_gate"
    if any(LAYER_GATE_RE.match(str(key)) for key in keys):
        return "layer_expert_gate"
    return "unknown"


def expand_gate_rows(gates: Mapping[str, Any], *, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_rows = [
        {
            "param_name": str(entry["param_name"]),
            "expert": str(entry["expert"]),
            "layer": parse_layer_index(str(entry["param_name"])),
            "family": module_family(str(entry["param_name"])),
        }
        for entry in manifest.get("basis_entries", [])
    ]
    manifest_by_layer_expert: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        manifest_by_layer_expert[(int(row["layer"]), str(row["expert"]))].append(row)

    for key, value in gates.items():
        key_text = str(key)
        coeff = float(value)
        if "::" in key_text:
            param_name, expert = key_text.rsplit("::", 1)
            rows.append(
                {
                    "raw_key": key_text,
                    "expert": expert,
                    "layer": parse_layer_index(param_name),
                    "family": module_family(param_name),
                    "coefficient": coeff,
                }
            )
            continue
        match = LAYER_GATE_RE.match(key_text)
        if match:
            layer = int(match.group(1))
            expert = match.group(2)
            expanded = manifest_by_layer_expert.get((layer, expert))
            if expanded:
                for item in expanded:
                    rows.append(
                        {
                            "raw_key": key_text,
                            "expert": expert,
                            "layer": layer,
                            "family": item["family"],
                            "coefficient": coeff,
                        }
                    )
            else:
                rows.append(
                    {
                        "raw_key": key_text,
                        "expert": expert,
                        "layer": layer,
                        "family": "layer_all",
                        "coefficient": coeff,
                    }
                )
            continue
        rows.append(
            {
                "raw_key": key_text,
                "expert": key_text,
                "layer": -1,
                "family": "global",
                "coefficient": coeff,
            }
        )
    return rows


def module_family(param_name: str) -> str:
    if ".self_attn.q_proj.weight" in param_name:
        return "attn_q"
    if ".self_attn.k_proj.weight" in param_name:
        return "attn_k"
    if ".self_attn.v_proj.weight" in param_name:
        return "attn_v"
    if ".self_attn.o_proj.weight" in param_name:
        return "attn_o"
    if ".mlp.gate_proj.weight" in param_name:
        return "mlp_gate"
    if ".mlp.up_proj.weight" in param_name:
        return "mlp_up"
    if ".mlp.down_proj.weight" in param_name:
        return "mlp_down"
    return "other"


def summarize_by(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = "|".join(str(row[field]) for field in fields)
        buckets[key].append(float(row["coefficient"]))
    return {key: coefficient_stats(values) for key, values in sorted(buckets.items())}


def summarize_groups(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family = str(row["family"])
        if family.startswith("attn_"):
            group = "attention"
        elif family.startswith("mlp_"):
            group = "mlp"
        else:
            group = family
        buckets[f"{row['expert']}|{group}"].append(float(row["coefficient"]))
    return {key: coefficient_stats(values) for key, values in sorted(buckets.items())}


def coefficient_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": float(len(values)),
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "std": var**0.5,
    }


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate Structure Summary",
        "",
        f"- mode_manifest: `{payload['mode_manifest']}`",
        "",
    ]
    for summary in payload["summaries"]:
        lines.extend(
            [
                f"## {Path(summary['path']).name}",
                "",
                f"- path: `{summary['path']}`",
                f"- source_format: `{summary['source_format']}`",
                f"- raw/expanded gates: `{summary['num_raw_gates']}` / `{summary['num_expanded_rows']}`",
                "",
                "### Expert Mean",
                "",
                "| expert | mean | min | max | std |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, stats in summary["expert_summary"].items():
            lines.append(format_stats_row(key, stats))
        lines.extend(["", "### Expert Group Mean", "", "| expert/group | mean | min | max | std |", "| --- | ---: | ---: | ---: | ---: |"])
        for key, stats in summary["group_summary"].items():
            lines.append(format_stats_row(key, stats))
        lines.extend(["", "### Expert Family Mean", "", "| expert/family | mean | min | max | std |", "| --- | ---: | ---: | ---: | ---: |"])
        for key, stats in summary["family_summary"].items():
            lines.append(format_stats_row(key, stats))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_stats_row(key: str, stats: Mapping[str, float]) -> str:
    return f"| {key} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['std']:.4f} |"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

