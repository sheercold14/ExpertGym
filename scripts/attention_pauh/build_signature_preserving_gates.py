#!/usr/bin/env python3
"""Build simple first-principles OP-VEC gates from task-pattern diagnostics.

SPRE (Signature-Preserving Residual Editing) starts from the full expert prior
``alpha=1`` and only shrinks module families with consistent owner/protected
evidence against them. It intentionally avoids reward sweeps and training.

Implemented variants:
- keep MLP residuals by default, because diagnostics show MLP is the main
  residual-expression channel;
- keep Memory and Tool intact unless both prompt and response exposure ratios
  indicate clear cross-task risk;
- shrink Code q/k/v attention when both prompt and response owner/protected
  ratios are below threshold, matching the observed Code attention harm pattern.
- optionally apply an architecture-level rule: keep all MLP at alpha=1 while
  calming all attention deltas, because attention edits alter routing/format.
- optionally calm only Memory attention to isolate whether Memory relies on
  attention routing in addition to its strong MLP residual channel.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import parse_layer_index, summarize_coefficients  # noqa: E402


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exposure = load_exposure_summary(args.exposure_summary)
    gates, coefficients, decision_rows = build_spre_gates(
        manifest=manifest,
        exposure=exposure,
        default_alpha=args.default_alpha,
        shrink_threshold=args.shrink_threshold,
        min_coeff=args.min_coeff,
        method=args.method,
        code_attn_qkv_scale=args.code_attn_qkv_scale,
        code_attn_o_scale=args.code_attn_o_scale,
        attention_calm_scale=args.attention_calm_scale,
    )
    config = {
        "format": "signature_preserving_residual_editing_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "mode_manifest": str(manifest_path),
        "exposure_summary": str(Path(args.exposure_summary).expanduser().resolve()) if args.exposure_summary else None,
        "default_alpha": args.default_alpha,
        "shrink_threshold": args.shrink_threshold,
        "min_coeff": args.min_coeff,
        "code_attn_qkv_scale": args.code_attn_qkv_scale,
        "code_attn_o_scale": args.code_attn_o_scale,
        "attention_calm_scale": args.attention_calm_scale,
        "principle": [
            "Start from init=1 full expert prior.",
            "Do not prune MLP by default because MLP dominates residual expression.",
            "Shrink only families with consistent owner/protected evidence of cross-task risk.",
        ],
    }
    payload = {
        "format": "signature_preserving_residual_editing_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "gates": gates,
        "coefficients": {
            expert: {str(layer): value for layer, value in sorted(layer_map.items())}
            for expert, layer_map in coefficients.items()
        },
        "coefficient_summary": summarize_coefficients(coefficients),
        "decision_rows": decision_rows,
        "decision_summary": summarize_decisions(decision_rows),
    }
    write_json(output_dir / "spre_gates.json", payload)
    write_json(output_dir / "spre_config.json", config)
    write_markdown_summary(output_dir / "spre_summary.md", payload)
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "spre_gates.json"),
                "num_gates": len(gates),
                "decision_summary": payload["decision_summary"],
                "coefficient_summary": payload["coefficient_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exposure-summary", default="", help="Raw linear exposure summary JSON from probe_linear_module_exposure_patterns.py")
    parser.add_argument(
        "--method",
        choices=[
            "exposure-shrink",
            "static-code-attn-shrink",
            "mlp-preserve-attn-calm",
            "memory-attn-calm",
        ],
        default="exposure-shrink",
    )
    parser.add_argument("--default-alpha", type=float, default=1.0)
    parser.add_argument("--shrink-threshold", type=float, default=0.85)
    parser.add_argument("--min-coeff", type=float, default=0.70)
    parser.add_argument("--code-attn-qkv-scale", type=float, default=0.75)
    parser.add_argument("--code-attn-o-scale", type=float, default=0.90)
    parser.add_argument("--attention-calm-scale", type=float, default=0.75)
    return parser.parse_args()


def load_exposure_summary(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def build_spre_gates(
    *,
    manifest: Mapping[str, Any],
    exposure: Mapping[str, Any] | None,
    default_alpha: float,
    shrink_threshold: float,
    min_coeff: float,
    method: str,
    code_attn_qkv_scale: float,
    code_attn_o_scale: float,
    attention_calm_scale: float,
) -> tuple[dict[str, float], dict[str, dict[int, float]], list[dict[str, Any]]]:
    gates: dict[str, float] = {}
    layer_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    decisions: list[dict[str, Any]] = []
    for entry in manifest.get("basis_entries", []):
        param_name = str(entry["param_name"])
        expert = str(entry["expert"])
        layer = parse_layer_index(param_name)
        family = module_family(param_name)
        coeff = float(default_alpha)
        reason = "default_full_expert_prior"
        ratios = exposure_ratios(exposure, expert=expert, family=family) if exposure is not None else {}

        if method == "mlp-preserve-attn-calm":
            if family.startswith("attn_"):
                coeff *= float(attention_calm_scale)
                reason = "attention_calm_mlp_preserved"
        elif method == "memory-attn-calm":
            if expert == "memory" and family.startswith("attn_"):
                coeff *= float(attention_calm_scale)
                reason = "memory_attention_calm"
        elif method == "static-code-attn-shrink":
            if expert == "code" and family in {"attn_q", "attn_k", "attn_v"}:
                coeff *= float(code_attn_qkv_scale)
                reason = "static_code_qkv_attention_shrink"
            elif expert == "code" and family == "attn_o":
                coeff *= float(code_attn_o_scale)
                reason = "static_code_o_attention_mild_shrink"
        else:
            should_shrink = should_shrink_family(
                expert=expert,
                family=family,
                ratios=ratios,
                shrink_threshold=shrink_threshold,
            )
            if should_shrink:
                coeff = max(float(min_coeff), min(float(default_alpha), exposure_shrink_value(ratios, shrink_threshold)))
                reason = "owner_protected_exposure_shrink"

        gates[f"{param_name}::{expert}"] = float(coeff)
        layer_values[expert][layer].append(float(coeff))
        decisions.append(
            {
                "expert": expert,
                "layer": layer,
                "family": family,
                "param_name": param_name,
                "coefficient": float(coeff),
                "reason": reason,
                "ratios": ratios,
            }
        )
    coefficients = {
        expert: {layer: sum(values) / float(len(values)) for layer, values in layer_map.items()}
        for expert, layer_map in layer_values.items()
    }
    return gates, coefficients, decisions


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


def exposure_ratios(exposure: Mapping[str, Any] | None, *, expert: str, family: str) -> dict[str, float]:
    if exposure is None:
        return {}
    summary = exposure.get("owner_protected_summary", {})
    span_map = summary.get(expert, {})
    ratios: dict[str, float] = {}
    for span in ("prompt", "response"):
        stats = span_map.get(span, {}).get(f"all:{family}")
        if stats:
            ratios[span] = float(stats.get("owner_over_protected", 1.0))
    return ratios


def should_shrink_family(*, expert: str, family: str, ratios: Mapping[str, float], shrink_threshold: float) -> bool:
    if family.startswith("mlp_"):
        return False
    if expert == "memory":
        return False
    if expert == "code" and family in {"attn_q", "attn_k", "attn_v", "attn_o"}:
        prompt = float(ratios.get("prompt", 1.0))
        response = float(ratios.get("response", 1.0))
        return response < float(shrink_threshold) and prompt < 1.0
    if expert == "tool" and family != "attn_o":
        return False
    prompt = float(ratios.get("prompt", 1.0))
    response = float(ratios.get("response", 1.0))
    return prompt < float(shrink_threshold) and response < float(shrink_threshold)


def exposure_shrink_value(ratios: Mapping[str, float], shrink_threshold: float) -> float:
    if not ratios:
        return 1.0
    min_ratio = min(float(value) for value in ratios.values())
    return min(1.0, min_ratio / max(float(shrink_threshold), 1.0e-12))


def summarize_decisions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = defaultdict(int)
    family_coeffs: dict[str, list[float]] = defaultdict(list)
    expert_coeffs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reason_counts[str(row["reason"])] += 1
        family_coeffs[f"{row['expert']}:{row['family']}"].append(float(row["coefficient"]))
        expert_coeffs[str(row["expert"])].append(float(row["coefficient"]))
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "expert_mean_coefficients": {
            expert: sum(values) / float(len(values)) for expert, values in sorted(expert_coeffs.items())
        },
        "family_mean_coefficients": {
            family: sum(values) / float(len(values)) for family, values in sorted(family_coeffs.items())
        },
    }


def write_markdown_summary(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# SPRE Gate Summary",
        "",
        "## Principle",
        "",
        "- Start from `init=1` full expert prior.",
        "- Keep MLP residuals by default because diagnostics show MLP is the main residual-expression channel.",
        "- Shrink only module families with consistent owner/protected evidence of cross-task risk.",
        "",
        "## Config",
        "",
        f"- method: `{payload['config']['method']}`",
        f"- mode_manifest: `{payload['config']['mode_manifest']}`",
        f"- exposure_summary: `{payload['config']['exposure_summary']}`",
        f"- default_alpha: `{payload['config']['default_alpha']}`",
        f"- shrink_threshold/min_coeff: `{payload['config']['shrink_threshold']}` / `{payload['config']['min_coeff']}`",
        "",
        "## Expert Coefficients",
        "",
        "| expert | layers | mean | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for expert, stats in sorted(payload["coefficient_summary"].items()):
        lines.append(
            f"| {expert} | {int(stats['count'])} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "## Decision Summary", ""])
    lines.append("### Reasons")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("| --- | ---: |")
    for reason, count in payload["decision_summary"]["reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "### Family Mean Coefficients", ""])
    lines.append("| expert:family | mean coefficient |")
    lines.append("| --- | ---: |")
    for family, value in payload["decision_summary"]["family_mean_coefficients"].items():
        lines.append(f"| {family} | {value:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
