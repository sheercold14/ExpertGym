#!/usr/bin/env python3
"""Analyze OP-VEC expert delta norms and build norm-aware gate checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_DIAGNOSTICS = "/tmp/shared-storage/OnPolicy/modes/opvec4/diagnostics.json"


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics).expanduser().resolve()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    diagnostics = _read_json(diagnostics_path)
    manifest = _read_json(manifest_path)

    stats = analyze_norms(
        diagnostics=diagnostics,
        manifest=manifest,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        reference_sum=float(args.reference_sum),
    )

    if args.output_json:
        _write_json(Path(args.output_json), stats)
    if args.output_report:
        _write_text(Path(args.output_report), render_report(stats))
    if args.output_gate_dir:
        write_gate_checkpoints(
            output_dir=Path(args.output_gate_dir),
            stats=stats,
            manifest_path=manifest_path,
            parameterizations=[item.strip() for item in args.gate_parameterization.split(",") if item.strip()],
        )

    print(json.dumps(_brief(stats), ensure_ascii=False, indent=2, sort_keys=True))


def analyze_norms(
    *,
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    diagnostics_path: Path,
    manifest_path: Path,
    reference_sum: float,
) -> dict[str, Any]:
    experts = _expert_names(manifest, diagnostics)
    params = diagnostics.get("params") or {}
    rows = [_param_row(name, value, experts) for name, value in params.items()]
    total_l2 = {expert: float(diagnostics["total_l2_norm_by_expert"][expert]) for expert in experts}
    norm_inits = _norm_aware_initializations(total_l2, reference_sum=reference_sum)
    return {
        "format": "opvec_task_vector_norm_diagnostics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "diagnostics": str(diagnostics_path),
        "experts": experts,
        "num_params": len(rows),
        "param_names": [row["param_name"] for row in rows],
        "total_l2_norm_by_expert": total_l2,
        "relative_total_l2": _relative_total_l2(total_l2),
        "module_counts": _module_counts(rows, experts),
        "module_l2_summary": _module_l2_summary(rows, experts),
        "block_energy": _energy_by(rows, experts, key="block"),
        "module_energy": _energy_by(rows, experts, key="module"),
        "layer_energy": _energy_by(rows, experts, key="layer"),
        "code_effective_sparsity": _effective_sparsity(rows, expert="code"),
        "code_small_relative_counts": _small_relative_counts(rows, expert="code", baselines=["tool", "memory"]),
        "top_code_modules": _top_modules(rows, expert="code", limit=20),
        "norm_aware_initializations": norm_inits,
        "notes": [
            "module count is not sparsity; compare total_l2 and energy concentration for effective strength",
            "sum1_equal_effective_l2 preserves total coefficient sum while equalizing expert L2 contribution",
            "baseline_mean_effective_l2 preserves the average expert perturbation of equal reference coefficients",
        ],
    }


def write_gate_checkpoints(
    *,
    output_dir: Path,
    stats: dict[str, Any],
    manifest_path: Path,
    parameterizations: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for init_name, init in stats["norm_aware_initializations"].items():
        coefficients = init["coefficients"]
        for parameterization in parameterizations:
            gates = _gates_for_parameterization(
                parameterization=parameterization,
                coefficients=coefficients,
                stats=stats,
            )
            payload = {
                "format": "opvec_norm_aware_gate_checkpoint_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mode_manifest": str(manifest_path),
                "gate_parameterization": parameterization,
                "init_name": init_name,
                "init_definition": init["definition"],
                "experts": stats["experts"],
                "total_l2_norm_by_expert": stats["total_l2_norm_by_expert"],
                "coefficients": coefficients,
                "effective_l2_by_expert": init["effective_l2_by_expert"],
                "gates": gates,
                "num_gate_values": len(gates),
            }
            _write_json(output_dir / f"{init_name}.{parameterization}.json", payload)


def render_report(stats: dict[str, Any]) -> str:
    total = stats["total_l2_norm_by_expert"]
    rel = stats["relative_total_l2"]
    code_sparse = stats["code_effective_sparsity"]
    small = stats["code_small_relative_counts"]
    lines = [
        "# Task Vector Norm Diagnostics",
        "",
        f"生成时间：`{stats['created_at']}`",
        "",
        "## 输入",
        "",
        f"- mode manifest: `{stats['mode_manifest']}`",
        f"- diagnostics: `{stats['diagnostics']}`",
        f"- mergeable params: `{stats['num_params']}`",
        "",
        "## 总范数",
        "",
        "| expert | total L2 | 相对 code |",
        "|---|---:|---:|",
    ]
    for expert in stats["experts"]:
        lines.append(f"| {expert} | {total[expert]:.6f} | {rel[expert]:.4f} |")
    lines.extend(
        [
            "",
            "解读：当前 code 不是缺少 module 覆盖，而是 delta 幅度显著小。"
            f"code total L2 只有 tool 的 `{total['code'] / total['tool']:.4f}`，"
            f"只有 memory 的 `{total['code'] / total['memory']:.4f}`。",
            "",
            "## 覆盖与能量分布",
            "",
            "| expert | params | MLP params | attention params | MLP energy frac | attention energy frac |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for expert in stats["experts"]:
        counts = stats["module_counts"][expert]
        block = stats["block_energy"][expert]
        lines.append(
            f"| {expert} | {counts['total']} | {counts.get('mlp', 0)} | {counts.get('attn', 0)} | "
            f"{block.get('mlp', {}).get('energy_frac', 0.0):.4f} | {block.get('attn', {}).get('energy_frac', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Code 有效稀疏性",
            "",
            "| code energy coverage | modules needed | total modules |",
            "|---:|---:|---:|",
        ]
    )
    for target, count in code_sparse.items():
        lines.append(f"| {float(target):.2f} | {count} | {stats['num_params']} |")
    lines.extend(["", "Code 相对其他 expert 的 module 级范数：", ""])
    lines.append("| condition | count | total |")
    lines.append("|---|---:|---:|")
    for key, value in small.items():
        lines.append(f"| {key} | {value} | {stats['num_params']} |")
    lines.extend(
        [
            "",
            "## Code Top Modules",
            "",
            "| rank | param | layer | module | code L2 | code/tool | code/memory |",
            "|---:|---|---:|---|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(stats["top_code_modules"], start=1):
        lines.append(
            f"| {idx} | `{row['param_name']}` | {row['layer']} | {row['module']} | "
            f"{row['code_l2']:.6f} | {row['code_over_tool']:.4f} | {row['code_over_memory']:.4f} |"
        )
    lines.extend(["", "## Norm-aware Init 候选", ""])
    for name, init in stats["norm_aware_initializations"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(init["definition"])
        lines.append("")
        lines.append("| expert | coefficient | effective L2 |")
        lines.append("|---|---:|---:|")
        for expert in stats["experts"]:
            lines.append(
                f"| {expert} | {init['coefficients'][expert]:.6f} | {init['effective_l2_by_expert'][expert]:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 结论",
            "",
            "- `1/3` 系数并不等价于三类能力同等注入；memory delta 的真实模型位移远大于 code。",
            "- 如果要让 code 能力在 merged model 中可见，必须采用 norm-aware init、code-specific modes 或更强 code/reasoning delta。",
            "- sweep 只应作为 oracle/diagnostic；论文方法应使用确定性 norm-aware init，然后用 on-policy gate learning 微调。",
        ]
    )
    return "\n".join(lines) + "\n"


def _param_row(param_name: str, value: dict[str, Any], experts: list[str]) -> dict[str, Any]:
    layer_match = re.search(r"model\.layers\.(\d+)\.", param_name)
    layer = int(layer_match.group(1)) if layer_match else -1
    if ".mlp." in param_name:
        block = "mlp"
        module = param_name.split(".mlp.", 1)[1].split(".weight", 1)[0]
    elif ".self_attn." in param_name:
        block = "attn"
        module = param_name.split(".self_attn.", 1)[1].split(".weight", 1)[0]
    else:
        block = "other"
        module = "other"
    row = {"param_name": param_name, "layer": layer, "block": block, "module": module}
    shape = value[experts[0]].get("shape") or []
    numel = 1
    for dim in shape:
        numel *= int(dim)
    row["numel"] = numel
    for expert in experts:
        row[f"{expert}_l2"] = float(value[expert]["l2_norm"])
        row[f"{expert}_max_abs"] = float(value[expert]["max_abs"])
        row[f"{expert}_rms"] = float(value[expert]["l2_norm"]) / math.sqrt(numel)
    return row


def _expert_names(manifest: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    experts = manifest.get("expert_names")
    if isinstance(experts, list) and experts:
        return [str(item) for item in experts]
    raw = manifest.get("experts")
    if isinstance(raw, dict) and raw:
        preferred = [item for item in ["tool", "memory", "code", "reasoning"] if item in raw]
        preferred.extend(str(item) for item in raw if str(item) not in preferred)
        return preferred
    return list(diagnostics.get("total_l2_norm_by_expert", {}).keys())


def _relative_total_l2(total_l2: dict[str, float]) -> dict[str, float]:
    reference = total_l2.get("code") or min(total_l2.values())
    return {expert: value / reference for expert, value in total_l2.items()}


def _module_counts(rows: list[dict[str, Any]], experts: list[str]) -> dict[str, dict[str, int]]:
    block_counts = Counter(row["block"] for row in rows)
    return {
        expert: {
            "total": len(rows),
            **dict(block_counts),
        }
        for expert in experts
    }


def _module_l2_summary(rows: list[dict[str, Any]], experts: list[str]) -> dict[str, dict[str, float]]:
    output = {}
    for expert in experts:
        values = [row[f"{expert}_l2"] for row in rows]
        rms_values = [row[f"{expert}_rms"] for row in rows]
        output[expert] = {
            "min_l2": min(values),
            "median_l2": median(values),
            "mean_l2": mean(values),
            "max_l2": max(values),
            "median_rms": median(rms_values),
            "max_rms": max(rms_values),
        }
    return output


def _energy_by(rows: list[dict[str, Any]], experts: list[str], *, key: str) -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    for expert in experts:
        sums: dict[str, float] = defaultdict(float)
        for row in rows:
            label = str(row[key])
            sums[label] += row[f"{expert}_l2"] ** 2
        total = sum(sums.values())
        output[expert] = {
            label: {
                "l2": math.sqrt(value),
                "energy_frac": value / total if total else 0.0,
            }
            for label, value in sorted(sums.items(), key=lambda item: item[0])
        }
    return output


def _effective_sparsity(rows: list[dict[str, Any]], *, expert: str) -> dict[str, int]:
    values = sorted((row[f"{expert}_l2"] ** 2 for row in rows), reverse=True)
    total = sum(values)
    output = {}
    for target in [0.5, 0.8, 0.9, 0.95]:
        acc = 0.0
        count = 0
        for value in values:
            acc += value
            count += 1
            if total and acc / total >= target:
                output[str(target)] = count
                break
    return output


def _small_relative_counts(rows: list[dict[str, Any]], *, expert: str, baselines: list[str]) -> dict[str, int]:
    output = {}
    for baseline in baselines:
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            key = f"{expert}_lt_{ratio:g}x_{baseline}"
            output[key] = sum(1 for row in rows if row[f"{expert}_l2"] < ratio * row[f"{baseline}_l2"])
    return output


def _top_modules(rows: list[dict[str, Any]], *, expert: str, limit: int) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda item: item[f"{expert}_l2"], reverse=True)[:limit]:
        item = {
            "param_name": row["param_name"],
            "layer": row["layer"],
            "block": row["block"],
            "module": row["module"],
            "code_l2": row["code_l2"],
            "tool_l2": row.get("tool_l2", 0.0),
            "memory_l2": row.get("memory_l2", 0.0),
            "code_over_tool": row["code_l2"] / (row.get("tool_l2", 0.0) + 1e-12),
            "code_over_memory": row["code_l2"] / (row.get("memory_l2", 0.0) + 1e-12),
        }
        output.append(item)
    return output


def _norm_aware_initializations(total_l2: dict[str, float], *, reference_sum: float) -> dict[str, Any]:
    experts = list(total_l2)
    inv = {expert: 1.0 / max(total_l2[expert], 1e-12) for expert in experts}
    inv_sum = sum(inv.values())
    sum1 = {expert: reference_sum * inv[expert] / inv_sum for expert in experts}
    sum1_eff = {expert: sum1[expert] * total_l2[expert] for expert in experts}

    baseline_coeff = reference_sum / len(experts)
    target_effective = baseline_coeff * mean(total_l2.values())
    baseline_mean = {expert: target_effective / max(total_l2[expert], 1e-12) for expert in experts}
    baseline_eff = {expert: baseline_mean[expert] * total_l2[expert] for expert in experts}
    all_ones = {expert: 1.0 for expert in experts}
    all_ones_eff = {expert: total_l2[expert] for expert in experts}
    strongest = max(total_l2.values())
    sqrt_comp = {expert: math.sqrt(strongest / max(total_l2[expert], 1e-12)) for expert in experts}
    sqrt_eff = {expert: sqrt_comp[expert] * total_l2[expert] for expert in experts}
    linear_comp = {expert: strongest / max(total_l2[expert], 1e-12) for expert in experts}
    linear_eff = {expert: linear_comp[expert] * total_l2[expert] for expert in experts}

    return {
        "all_ones": {
            "definition": "all expert coefficients are initialized to 1.0; preserves full expert task-vector strength before learning",
            "coefficients": all_ones,
            "effective_l2_by_expert": all_ones_eff,
        },
        "all1_sqrt_weak_compensation": {
            "definition": (
                "strongest expert stays at 1.0; weaker experts are compensated by sqrt(max ||Delta|| / ||Delta_e||). "
                "This is a conservative non-sweep stress test for weak code/tool deltas."
            ),
            "coefficients": sqrt_comp,
            "effective_l2_by_expert": sqrt_eff,
        },
        "all1_linear_weak_compensation": {
            "definition": (
                "strongest expert stays at 1.0; weaker experts are linearly compensated to equal effective L2. "
                "This is aggressive and should be used only as a short stability diagnostic."
            ),
            "coefficients": linear_comp,
            "effective_l2_by_expert": linear_eff,
        },
        "sum1_equal_effective_l2": {
            "definition": f"coefficients are proportional to 1 / ||Delta_e|| and sum to {reference_sum:g}",
            "coefficients": sum1,
            "effective_l2_by_expert": sum1_eff,
        },
        "baseline_mean_effective_l2": {
            "definition": (
                f"coefficients equalize effective L2 to the mean perturbation produced by "
                f"equal coefficients summing to {reference_sum:g}"
            ),
            "coefficients": baseline_mean,
            "effective_l2_by_expert": baseline_eff,
        },
    }


def _gates_for_parameterization(
    *,
    parameterization: str,
    coefficients: dict[str, float],
    stats: dict[str, Any],
) -> dict[str, float]:
    if parameterization == "global-coefficient":
        return {expert: float(coefficients[expert]) for expert in stats["experts"]}
    if parameterization == "global-parameter":
        gates = {f"__global__::{expert}": float(coefficients[expert]) for expert in stats["experts"]}
        param_names = list(stats.get("param_names") or [])
        for param_name in param_names:
            for expert in stats["experts"]:
                gates[f"{param_name}::{expert}"] = float(coefficients[expert])
        return gates
    raise ValueError(f"Unsupported norm-aware output gate parameterization: {parameterization}")


def _brief(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_l2_norm_by_expert": stats["total_l2_norm_by_expert"],
        "relative_total_l2": stats["relative_total_l2"],
        "code_effective_sparsity": stats["code_effective_sparsity"],
        "norm_aware_initializations": stats["norm_aware_initializations"],
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--diagnostics", default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--output-gate-dir", default=None)
    parser.add_argument("--reference-sum", type=float, default=1.0)
    parser.add_argument(
        "--gate-parameterization",
        default="global-coefficient,global-parameter",
        help="Comma-separated output checkpoint parameterizations. Supported: global-coefficient,global-parameter.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
