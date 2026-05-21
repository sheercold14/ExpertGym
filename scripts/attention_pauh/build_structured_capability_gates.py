#!/usr/bin/env python3
"""Build a small first-principles family of structured capability gates.

The generator encodes the current mechanism findings directly:

- Memory needs both MLP residual magnitude and attention routing, so keep it.
- Tool source behavior is fragile and mostly preserved by keeping Tool intact.
- Code has weak middle-layer positive pass/fail signal, but late-layer conflict,
  so only middle-layer positive families are opened aggressively.

This is not a reward sweep. It creates a tiny frontier of mechanism-constrained
candidate gates that can be baked and verified with the existing evaluation
harnesses.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.build_signature_preserving_gates import module_family  # noqa: E402
from scripts.attention_pauh.core import parse_layer_index, summarize_coefficients  # noqa: E402


POSITIVE_CODE_FAMILIES = frozenset({"mlp_gate", "mlp_up", "attn_o", "attn_v"})
WEAK_CODE_FAMILIES = frozenset({"mlp_down", "attn_q", "attn_k"})


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    description: str
    memory_scale: float = 1.0
    tool_scale: float = 1.0
    code_background_scale: float = 0.75
    code_mid_positive_scale: float = 1.0
    code_mid_weak_scale: float = 0.85
    code_conflict_scale: float = 0.50


DEFAULT_PROFILES = (
    CapabilityProfile(
        name="balanced",
        description="Preserve Memory/Tool; open Code middle positive families while suppressing conflict layers.",
        code_background_scale=0.75,
        code_mid_positive_scale=1.00,
        code_mid_weak_scale=0.85,
        code_conflict_scale=0.50,
    ),
    CapabilityProfile(
        name="code_mid_push",
        description="Slightly push Code middle positive families; keep late conflict strongly suppressed.",
        code_background_scale=0.75,
        code_mid_positive_scale=1.15,
        code_mid_weak_scale=0.90,
        code_conflict_scale=0.35,
    ),
    CapabilityProfile(
        name="code_safe",
        description="Conservative Code expression for protecting Memory/Tool while keeping middle signal alive.",
        code_background_scale=0.70,
        code_mid_positive_scale=0.95,
        code_mid_weak_scale=0.75,
        code_conflict_scale=0.25,
    ),
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    profiles = selected_profiles(args.profiles)
    mid_layers = parse_int_set(args.code_mid_layers)
    conflict_layers = parse_int_set(args.code_conflict_layers)
    payloads = []
    for profile in profiles:
        payload = build_candidate_payload(
            manifest=manifest,
            profile=profile,
            mode_manifest=str(manifest_path),
            code_mid_layers=mid_layers,
            code_conflict_layers=conflict_layers,
        )
        candidate_dir = output_dir / profile.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "gates.json", payload)
        write_markdown(candidate_dir / "summary.md", payload)
        payloads.append(
            {
                "name": profile.name,
                "gate_checkpoint": str(candidate_dir / "gates.json"),
                "summary": str(candidate_dir / "summary.md"),
                "profile": asdict(profile),
                "coefficient_summary": payload["coefficient_summary"],
                "decision_summary": payload["decision_summary"],
            }
        )

    manifest_payload = {
        "format": "structured_capability_gate_family_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "code_mid_layers": sorted(mid_layers),
        "code_conflict_layers": sorted(conflict_layers),
        "principles": [
            "Preserve Memory MLP and attention because both are causally useful.",
            "Preserve Tool unless evaluation shows a specific BFCL behavior deficit.",
            "Express Code only through middle-layer positive families from pass/fail contrast.",
            "Suppress late Code conflict layers, especially layer 27.",
        ],
        "candidates": payloads,
    }
    write_json(output_dir / "candidate_manifest.json", manifest_payload)
    write_family_markdown(output_dir / "README.md", manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profiles",
        default="balanced,code_mid_push,code_safe",
        help="Comma-separated profile names. Available: balanced, code_mid_push, code_safe.",
    )
    parser.add_argument("--code-mid-layers", default="8-20")
    parser.add_argument("--code-conflict-layers", default="24,27")
    return parser.parse_args()


def selected_profiles(raw: str) -> list[CapabilityProfile]:
    by_name = {profile.name: profile for profile in DEFAULT_PROFILES}
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    profiles = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Unknown profile {name!r}; available={sorted(by_name)}")
        profiles.append(by_name[name])
    if not profiles:
        raise ValueError("At least one profile is required")
    return profiles


def parse_int_set(raw: str) -> set[int]:
    result: set[int] = set()
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(piece))
    return result


def build_candidate_payload(
    *,
    manifest: Mapping[str, Any],
    profile: CapabilityProfile,
    mode_manifest: str,
    code_mid_layers: set[int],
    code_conflict_layers: set[int],
) -> dict[str, Any]:
    gates, decisions = build_structured_capability_gates(
        manifest=manifest,
        profile=profile,
        code_mid_layers=code_mid_layers,
        code_conflict_layers=code_conflict_layers,
    )
    coefficients = layer_mean_coefficients(decisions)
    return {
        "format": "structured_capability_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": mode_manifest,
        "profile": asdict(profile),
        "principles": {
            "memory": "keep MLP and attention; memory-attn-calm ablation showed F1 drop",
            "tool": "keep source behavior; use BFCL/ToolRL evaluation for verification",
            "code": "middle positive families only; suppress late conflict layers",
        },
        "code_mid_layers": sorted(code_mid_layers),
        "code_conflict_layers": sorted(code_conflict_layers),
        "gates": gates,
        "coefficients": {
            expert: {str(layer): value for layer, value in sorted(layer_map.items())}
            for expert, layer_map in coefficients.items()
        },
        "coefficient_summary": summarize_coefficients(coefficients),
        "decision_summary": summarize_decisions(decisions),
        "decision_rows": decisions,
    }


def build_structured_capability_gates(
    *,
    manifest: Mapping[str, Any],
    profile: CapabilityProfile,
    code_mid_layers: set[int],
    code_conflict_layers: set[int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gates: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    for entry in manifest.get("basis_entries", []):
        param_name = str(entry["param_name"])
        expert = str(entry["expert"])
        layer = parse_layer_index(param_name)
        family = module_family(param_name)
        coeff, reason = coefficient_for_entry(
            expert=expert,
            layer=layer,
            family=family,
            profile=profile,
            code_mid_layers=code_mid_layers,
            code_conflict_layers=code_conflict_layers,
        )
        gates[f"{param_name}::{expert}"] = float(coeff)
        decisions.append(
            {
                "expert": expert,
                "layer": layer,
                "family": family,
                "param_name": param_name,
                "coefficient": float(coeff),
                "reason": reason,
            }
        )
    return gates, decisions


def coefficient_for_entry(
    *,
    expert: str,
    layer: int,
    family: str,
    profile: CapabilityProfile,
    code_mid_layers: set[int],
    code_conflict_layers: set[int],
) -> tuple[float, str]:
    if expert == "memory":
        return float(profile.memory_scale), "preserve_memory_attention_and_mlp"
    if expert == "tool":
        return float(profile.tool_scale), "preserve_tool_behavior"
    if expert != "code":
        return 0.0, "unknown_expert_suppressed"
    if int(layer) in code_conflict_layers:
        return float(profile.code_conflict_scale), "suppress_code_late_conflict"
    if int(layer) in code_mid_layers and family in POSITIVE_CODE_FAMILIES:
        return float(profile.code_mid_positive_scale), "open_code_mid_positive_family"
    if int(layer) in code_mid_layers and family in WEAK_CODE_FAMILIES:
        return float(profile.code_mid_weak_scale), "guard_code_mid_weak_family"
    return float(profile.code_background_scale), "code_background_prior"


def layer_mean_coefficients(rows: list[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        expert = str(row["expert"])
        layer = int(row["layer"])
        sums[expert][layer] += float(row["coefficient"])
        counts[expert][layer] += 1
    return {
        expert: {layer: total / float(counts[expert][layer]) for layer, total in layer_map.items()}
        for expert, layer_map in sums.items()
    }


def summarize_decisions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = defaultdict(int)
    family_coeffs: dict[str, list[float]] = defaultdict(list)
    group_coeffs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reason_counts[str(row["reason"])] += 1
        key = f"{row['expert']}:{row['family']}"
        family_coeffs[key].append(float(row["coefficient"]))
        group = "attention" if str(row["family"]).startswith("attn_") else "mlp"
        group_coeffs[f"{row['expert']}:{group}"].append(float(row["coefficient"]))
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "family_mean_coefficients": {
            key: sum(values) / float(len(values)) for key, values in sorted(family_coeffs.items())
        },
        "group_mean_coefficients": {
            key: sum(values) / float(len(values)) for key, values in sorted(group_coeffs.items())
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    profile = payload["profile"]
    lines = [
        f"# Structured Capability Gate: {profile['name']}",
        "",
        profile["description"],
        "",
        "## Profile",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    for key, value in profile.items():
        if key in {"name", "description"}:
            continue
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Coefficient Summary", "", "| expert | layers | mean | min | max |", "| --- | ---: | ---: | ---: | ---: |"])
    for expert, stats in sorted(payload["coefficient_summary"].items()):
        lines.append(
            f"| {expert} | {int(stats['count'])} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "## Decision Counts", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in payload["decision_summary"]["reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Structured Capability Gate Family",
        "",
        "This directory contains a tiny mechanism-constrained frontier, not a reward sweep.",
        "",
        "## Principles",
        "",
    ]
    for item in payload["principles"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Candidates", "", "| candidate | gate | memory mean | tool mean | code mean |", "| --- | --- | ---: | ---: | ---: |"])
    for candidate in payload["candidates"]:
        summary = candidate["coefficient_summary"]
        lines.append(
            f"| {candidate['name']} | `{candidate['gate_checkpoint']}` | "
            f"{summary.get('memory', {}).get('mean', 0.0):.4f} | "
            f"{summary.get('tool', {}).get('mean', 0.0):.4f} | "
            f"{summary.get('code', {}).get('mean', 0.0):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
