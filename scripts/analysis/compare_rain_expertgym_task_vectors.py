#!/usr/bin/env python3
"""Compare RAIN task-vector artifacts with ExpertGym OP-VEC deltas.

The goal is diagnostic, not training.  RAIN and ExpertGym use different
anchors, vector semantics, and slicing schemes; this script makes those
differences explicit from already-generated manifests/stat files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RAIN_STAGE1_CONFIG = (
    "/tmp/shared-storage/RAIN/rain_paper_eq_formula_fp64_full_05003cc_20260522_203804/"
    "stage1/projected_task_vectors_config.json"
)
DEFAULT_RAIN_LAMBDA09_STATS = (
    "/tmp/shared-storage/RAIN/rain_fp64_lambda_sweep_05003cc_20260522_223000/"
    "runs/lambda_0.9/stage3/unified_merge_stats.json"
)
DEFAULT_RAIN_LAMBDA_GRID = (
    "/tmp/shared-storage/RAIN/rain_fp64_lambda_sweep_05003cc_20260522_223000/"
    "lambda_grid_summary.json"
)
DEFAULT_OPVEC_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4-full-bf16-real/mode_manifest.json"
DEFAULT_OPVEC_DIAGNOSTICS = "/tmp/shared-storage/OnPolicy/modes/opvec4-full-bf16-real/diagnostics.json"
DEFAULT_OPVEC_R1_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519/mode_manifest.json"
DEFAULT_OPVEC_R1_DIAGNOSTICS = "/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519/diagnostics.json"


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    if args.output_json:
        write_json(Path(args.output_json), summary)
    if args.output_md:
        write_text(Path(args.output_md), render_markdown(summary))
    print(json.dumps(brief(summary), ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rain-stage1-config", default=DEFAULT_RAIN_STAGE1_CONFIG)
    parser.add_argument("--rain-stage3-stats", default=DEFAULT_RAIN_LAMBDA09_STATS)
    parser.add_argument("--rain-lambda-grid", default=DEFAULT_RAIN_LAMBDA_GRID)
    parser.add_argument("--opvec-manifest", default=DEFAULT_OPVEC_MANIFEST)
    parser.add_argument("--opvec-diagnostics", default=DEFAULT_OPVEC_DIAGNOSTICS)
    parser.add_argument("--opvec-r1-manifest", default=DEFAULT_OPVEC_R1_MANIFEST)
    parser.add_argument("--opvec-r1-diagnostics", default=DEFAULT_OPVEC_R1_DIAGNOSTICS)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rain_stage1 = read_json(Path(args.rain_stage1_config))
    rain_stage3 = read_json(Path(args.rain_stage3_stats))
    rain_grid = read_json(Path(args.rain_lambda_grid)) if Path(args.rain_lambda_grid).exists() else {}
    opvec_manifest = read_json(Path(args.opvec_manifest))
    opvec_diag = read_json(Path(args.opvec_diagnostics))
    r1_manifest = read_json(Path(args.opvec_r1_manifest)) if Path(args.opvec_r1_manifest).exists() else {}
    r1_diag = read_json(Path(args.opvec_r1_diagnostics)) if Path(args.opvec_r1_diagnostics).exists() else {}

    rain_norms = summarize_rain_stage3(rain_stage3)
    opvec = summarize_opvec(opvec_manifest, opvec_diag)
    opvec_r1 = summarize_opvec(r1_manifest, r1_diag) if r1_diag else {}

    return {
        "format": "rain_expertgym_task_vector_diagnosis_v1",
        "sources": {
            "rain_stage1_config": str(args.rain_stage1_config),
            "rain_stage3_stats": str(args.rain_stage3_stats),
            "rain_lambda_grid": str(args.rain_lambda_grid),
            "opvec_manifest": str(args.opvec_manifest),
            "opvec_diagnostics": str(args.opvec_diagnostics),
            "opvec_r1_manifest": str(args.opvec_r1_manifest),
            "opvec_r1_diagnostics": str(args.opvec_r1_diagnostics),
        },
        "rain_semantics": {
            "anchor": "DeepSeek-R1-Distill-Qwen-7B",
            "added_vector": "Qwen2.5-7B-Instruct - Qwen2.5-7B",
            "reasoning_vector": "none; reasoning model is the protected target/anchor",
            "stage1_projection": "null-space projection against reasoning calibration features",
            "stage2_coefficients": "instruction-attention utility/harm alpha",
            "selected_layers": rain_stage1.get("selected_layers"),
            "selected_heads": rain_stage1.get("selected_heads"),
            "merge_types": rain_stage1.get("merge_types"),
            "compute_dtype": rain_stage1.get("compute_dtype"),
            "lambda_grid_best": (rain_grid.get("best_mean") or {}).get("lambda"),
        },
        "rain_norm_summary": rain_norms,
        "expertgym_semantics": {
            "anchor": opvec_manifest.get("base_model"),
            "added_vectors": opvec_manifest.get("experts"),
            "physical_basis": opvec_manifest.get("physical_basis"),
            "gate_parameterization": opvec_manifest.get("gate_parameterization"),
            "num_params": (opvec_manifest.get("selection") or {}).get("num_params"),
            "selected_params": (opvec_manifest.get("selection") or {}).get("params", [])[:8],
        },
        "expertgym_norm_summary": opvec,
        "expertgym_r1_semantics": {
            "anchor": r1_manifest.get("base_model"),
            "experts": r1_manifest.get("experts"),
            "delta_bases": r1_manifest.get("delta_bases"),
            "warning": "R1 delta is heterogenous and much larger than Tool/Memory/Code; it is not commensurate with ordinary RL expert deltas.",
        },
        "expertgym_r1_norm_summary": opvec_r1,
        "derived_findings": derive_findings(rain_norms, opvec, opvec_r1),
    }


def summarize_rain_stage3(stats: dict[str, Any]) -> dict[str, Any]:
    by_module: dict[str, list[float]] = {"q": [], "k": [], "v": [], "o": [], "ffn": []}
    all_values: list[float] = []
    for layer in (stats.get("layer_stats") or {}).values():
        for head_stat in (layer.get("heads") or {}).values():
            for group, key in [("q", "norm_q"), ("k", "norm_k"), ("v", "norm_v"), ("o", "norm_o")]:
                value = float(head_stat.get(key) or 0.0)
                if value:
                    by_module[group].append(value)
                    all_values.append(value)
        ffn = layer.get("ffn") or {}
        for key in ["norm_gate", "norm_up", "norm_down"]:
            value = float(ffn.get(key) or 0.0)
            if value:
                by_module["ffn"].append(value)
                all_values.append(value)
    return {
        "params_modified": stats.get("total_params_modified"),
        "num_slices": len(all_values),
        "sum_slice_norm": sum(all_values),
        "sqrt_sum_slice_norm": sqrt_sum_sq(all_values),
        "max_slice_norm": max(all_values) if all_values else 0.0,
        "by_module": {key: summarize_values(values) for key, values in by_module.items() if values},
    }


def summarize_opvec(manifest: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    totals = {key: float(value) for key, value in (diagnostics.get("total_l2_norm_by_expert") or {}).items()}
    params = diagnostics.get("params") or {}
    experts = sorted(totals)
    module_counts = {expert: {"total": 0, "attn": 0, "mlp": 0} for expert in experts}
    module_energy_sq = {expert: {"attn": 0.0, "mlp": 0.0, "other": 0.0} for expert in experts}
    max_abs = {expert: 0.0 for expert in experts}
    sum_param_l2 = {expert: 0.0 for expert in experts}
    for param_name, row in params.items():
        family = "mlp" if ".mlp." in param_name else "attn" if ".self_attn." in param_name else "other"
        for expert in experts:
            if expert not in row:
                continue
            l2 = float(row[expert].get("l2_norm") or 0.0)
            module_counts[expert]["total"] += 1
            module_counts[expert][family] = module_counts[expert].get(family, 0) + 1
            module_energy_sq[expert][family] = module_energy_sq[expert].get(family, 0.0) + l2 * l2
            sum_param_l2[expert] += l2
            max_abs[expert] = max(max_abs[expert], float(row[expert].get("max_abs") or 0.0))
    block_energy = {}
    for expert in experts:
        total_sq = sum(module_energy_sq[expert].values()) or 1.0
        block_energy[expert] = {
            key: {"l2": math.sqrt(value), "energy_frac": value / total_sq}
            for key, value in module_energy_sq[expert].items()
            if value
        }
    return {
        "base_model": manifest.get("base_model"),
        "experts": manifest.get("experts"),
        "delta_bases": manifest.get("delta_bases"),
        "num_basis_entries": len(manifest.get("basis_entries") or []),
        "num_params": (manifest.get("selection") or {}).get("num_params"),
        "total_l2_norm_by_expert": totals,
        "sum_param_l2_by_expert": sum_param_l2,
        "max_abs_by_expert": max_abs,
        "module_counts": module_counts,
        "block_energy": block_energy,
    }


def derive_findings(rain: dict[str, Any], opvec: dict[str, Any], opvec_r1: dict[str, Any]) -> list[str]:
    findings = [
        "RAIN does not learn or add a reasoning task vector; the reasoning model is the anchor being protected.",
        "ExpertGym adds multiple RL expert deltas relative to the instruction anchor, so its deltas are capability priors rather than behavior anchors.",
        "RAIN's alpha/grid controls a projected instruction delta after behavior protection; ExpertGym gates directly scale unprojected expert deltas unless an explicit behavior constraint is added.",
    ]
    totals = opvec.get("total_l2_norm_by_expert") or {}
    if {"code", "memory", "tool"}.issubset(totals):
        findings.append(
            f"ExpertGym delta norms are highly unbalanced: code={totals['code']:.3f}, "
            f"tool={totals['tool']:.3f}, memory={totals['memory']:.3f}."
        )
    r1_totals = opvec_r1.get("total_l2_norm_by_expert") or {}
    if "reasoning" in r1_totals and "code" in totals and "memory" in totals:
        findings.append(
            f"The raw R1/Math reasoning delta norm is {r1_totals['reasoning']:.3f}, "
            f"about {r1_totals['reasoning'] / totals['code']:.1f}x code and "
            f"{r1_totals['reasoning'] / totals['memory']:.1f}x memory."
        )
    if rain.get("sqrt_sum_slice_norm"):
        findings.append(
            f"RAIN lambda=0.9 applies a projected instruction perturbation with slice sqrt-norm "
            f"{rain['sqrt_sum_slice_norm']:.3f}; this is not directly comparable to OP-VEC total L2, "
            "but it shows RAIN operates after projection and alpha selection, not on raw expert deltas."
        )
    return findings


def render_markdown(summary: dict[str, Any]) -> str:
    rain = summary["rain_norm_summary"]
    opvec = summary["expertgym_norm_summary"]
    r1 = summary["expertgym_r1_norm_summary"]
    lines = [
        "# RAIN vs ExpertGym Task Vector Diagnosis",
        "",
        "## Question",
        "",
        "比较 RAIN-Merging 中的 instruction / reasoning 角色与 ExpertGym 中 Tool / Memory / Code / R1 task vector 的差异，判断下一步是否应继续沿 gate learning 推进，还是把 RAIN 的 behavior-preserving idea 迁移到 ExpertGym。",
        "",
        "## Key Conclusion",
        "",
        "RAIN 和 ExpertGym 当前使用的 `task vector` 不是同一种对象。RAIN 的 reasoning model 不是 additive vector，而是被保护的 anchor；ExpertGym 的 Tool/Memory/Code 是相对同一个 instruction base 的 RL expert delta。直接把 RAIN 的成功解释成“多 expert gate 会自动学好”是不成立的。",
        "",
        "## Source Artifacts",
        "",
    ]
    for key, value in summary["sources"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Semantic Comparison",
            "",
            "| item | RAIN-Merging | ExpertGym OP-VEC |",
            "|---|---|---|",
            "| anchor | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-7B-Instruct |",
            "| additive vector | Qwen2.5-Instruct - Qwen2.5-Base | Tool / Memory / Code expert - Qwen2.5-Instruct |",
            "| reasoning role | protected target behavior, not a vector | optional fourth heterogeneous R1 delta |",
            "| primary control | null-space projection + attention utility/harm alpha + lambda grid | reward/OPD/TRC/hand-designed gates over expert deltas |",
            "| risk | instruction delta may break reasoning rollout | expert deltas are unbalanced and can conflict across behaviors |",
            "",
            "## Norm / Structure Evidence",
            "",
            "### RAIN lambda=0.9 projected instruction perturbation",
            "",
            f"- params modified: `{rain.get('params_modified')}`",
            f"- slices: `{rain.get('num_slices')}`",
            f"- sum slice norm: `{rain.get('sum_slice_norm', 0.0):.4f}`",
            f"- sqrt-sum slice norm: `{rain.get('sqrt_sum_slice_norm', 0.0):.4f}`",
            f"- max slice norm: `{rain.get('max_slice_norm', 0.0):.4f}`",
            "",
            "| module | n | sum norm | sqrt-sum norm | mean | max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for module, stats in rain.get("by_module", {}).items():
        lines.append(
            f"| {module} | {stats['n']} | {stats['sum']:.4f} | {stats['sqrt_sum']:.4f} | "
            f"{stats['mean']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "### ExpertGym Tool / Memory / Code deltas", "", "| expert | total L2 | sum param L2 | max abs | MLP energy | attn energy |", "|---|---:|---:|---:|---:|---:|"])
    totals = opvec.get("total_l2_norm_by_expert") or {}
    sums = opvec.get("sum_param_l2_by_expert") or {}
    max_abs = opvec.get("max_abs_by_expert") or {}
    block = opvec.get("block_energy") or {}
    for expert in sorted(totals):
        b = block.get(expert, {})
        lines.append(
            f"| {expert} | {totals[expert]:.4f} | {sums.get(expert, 0.0):.4f} | {max_abs.get(expert, 0.0):.6f} | "
            f"{b.get('mlp', {}).get('energy_frac', 0.0):.4f} | {b.get('attn', {}).get('energy_frac', 0.0):.4f} |"
        )
    if r1:
        lines.extend(["", "### ExpertGym raw R1/Math delta diagnostic", "", "| expert | total L2 | sum param L2 | max abs |", "|---|---:|---:|---:|"])
        r1_totals = r1.get("total_l2_norm_by_expert") or {}
        r1_sums = r1.get("sum_param_l2_by_expert") or {}
        r1_max = r1.get("max_abs_by_expert") or {}
        for expert in sorted(r1_totals):
            lines.append(f"| {expert} | {r1_totals[expert]:.4f} | {r1_sums.get(expert, 0.0):.4f} | {r1_max.get(expert, 0.0):.6f} |")
    lines.extend(["", "## Findings", ""])
    for item in summary["derived_findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Research Implication",
            "",
            "下一步不应直接把 RAIN 的 alpha 公式照搬到 ExpertGym，也不应继续只调全局 gate。更合理的主线是：",
            "",
            "1. 用 RAIN 的第一性原则定义 ExpertGym 的 behavior anchors：Tool call span、Memory full trajectory、Code pass/fail execution span。",
            "2. 对每个 candidate expert delta 先做 behavior-preserving projection 或 soft constraint，再估计 utility/harm。",
            "3. 把 calibration 从“训练答案”降级为“诊断 residual 是否支持/伤害某个行为”的 probe；算法输出是 residual-level routing，而不是 reward-fitting gate。",
            "",
            "这条线能把 RAIN/RAM/ExpertGym 统一为一个更简洁的论文叙事：能力向量必须在行为子空间约束下组合。",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "sum": 0.0, "sqrt_sum": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "sum": sum(values),
        "sqrt_sum": sqrt_sum_sq(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def sqrt_sum_sq(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def brief(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "rain_anchor": summary["rain_semantics"]["anchor"],
        "rain_added_vector": summary["rain_semantics"]["added_vector"],
        "expertgym_anchor": summary["expertgym_semantics"]["anchor"],
        "expertgym_norms": summary["expertgym_norm_summary"].get("total_l2_norm_by_expert"),
        "r1_norms": summary["expertgym_r1_norm_summary"].get("total_l2_norm_by_expert"),
        "findings": summary["derived_findings"],
    }


if __name__ == "__main__":
    main()
