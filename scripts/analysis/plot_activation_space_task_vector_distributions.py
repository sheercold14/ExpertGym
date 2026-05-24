#!/usr/bin/env python3
"""Plot activation-space task-vector distributions from signed utility probes.

The raw task-vector plots show where parameter deltas are large. This script
uses cached ``linear_delta_probe`` outputs to show how those deltas are actually
expressed on calibration activations:

    expression = mean ||Delta W h||^2

The default inputs are the RCF-BC positive calibration probes from 2026-05-21.
No model execution is performed here; the script only aggregates cached JSONL
probe rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROBE_ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes")
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/ExpertGym/task_vector_activation_distributions/opvec4_rcrf_calibration_20260522"
)
DEFAULT_PROBES = [
    (
        "tm_signature_s32",
        PROBE_ROOT / "tool_memory_positive_signature_s32_20260521" / "signed_utility_rows.jsonl",
    ),
    (
        "memory_fulltraj_s32",
        PROBE_ROOT / "memory_fulltraj_positive_s32_20260521" / "signed_utility_rows.jsonl",
    ),
    (
        "livebench_code_s16",
        PROBE_ROOT / "livebench_positive_code_alllayers_s16_20260521" / "signed_utility_rows.jsonl",
    ),
    (
        "livecodebench_code_s16",
        PROBE_ROOT / "livecodebench_positive_code_alllayers_s16_20260521" / "signed_utility_rows.jsonl",
    ),
]
EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
FAMILY_ORDER = ["attention", "mlp"]
EPS = 1.0e-30


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_specs = parse_probe_args(args.probe) if args.probe else DEFAULT_PROBES
    rows = load_probe_rows(probe_specs)
    if not rows:
        raise ValueError("No activation probe rows were loaded.")

    calibrations = ordered_calibrations(rows)
    experts = ordered_experts(rows)
    expr_bins = make_bins([row["log_expression"] for row in rows], bins=int(args.expression_bins), pad=0.25)
    effect_scale = robust_abs_scale([row["signed_effect"] for row in rows])
    for row in rows:
        row["signed_effect_scaled"] = signed_log_scale(float(row["signed_effect"]), effect_scale)

    global_rows = summarize_global(rows)
    module_rows = summarize_module(rows)
    owner_rows = compute_owner_ratios(rows)
    conflict_rows = load_conflict_rows(probe_specs, rows)

    write_csv(output_dir / "activation_global_stats.csv", global_rows)
    write_csv(output_dir / "activation_module_stats.csv", module_rows)
    write_csv(output_dir / "activation_owner_ratios.csv", owner_rows)
    write_csv(output_dir / "activation_conflict_summary.csv", conflict_rows)
    write_json(
        output_dir / "activation_histogram_metadata.json",
        {
            "expression_log10_bin_edges": expr_bins.tolist(),
            "signed_effect_scale": effect_scale,
            "probes": [{"label": label, "rows": str(path)} for label, path in probe_specs],
            "calibrations": calibrations,
            "experts": experts,
        },
    )

    plot_expression_density(
        rows=rows,
        bins=expr_bins,
        calibrations=calibrations,
        experts=experts,
        output_path=output_dir / "01_activation_expression_density.png",
    )
    plot_expression_heatmaps(
        module_rows=module_rows,
        calibrations=calibrations,
        experts=experts,
        output_path=output_dir / "02_activation_expression_heatmaps.png",
    )
    plot_owner_ratio_heatmaps(
        owner_rows=owner_rows,
        calibrations=calibrations,
        output_path=output_dir / "03_owner_vs_nonowner_expression_ratio.png",
    )
    plot_signed_effect_density(
        rows=rows,
        calibrations=calibrations,
        experts=experts,
        effect_scale=effect_scale,
        output_path=output_dir / "04_signed_effect_density.png",
    )
    if conflict_rows:
        plot_conflict_cosines(
            conflict_rows=conflict_rows,
            calibrations=calibrations,
            output_path=output_dir / "05_activation_update_cosines.png",
        )
    write_report(
        output_dir / "README.md",
        output_dir=output_dir,
        probe_specs=probe_specs,
        rows=rows,
        global_rows=global_rows,
        owner_rows=owner_rows,
        conflict_rows=conflict_rows,
        effect_scale=effect_scale,
    )
    print(f"Wrote activation-space distribution plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help="Optional probe in label=/path/to/signed_utility_rows.jsonl form. Defaults to RCF-BC positives.",
    )
    parser.add_argument("--expression-bins", type=int, default=180)
    return parser.parse_args()


def parse_probe_args(raw_items: list[str]) -> list[tuple[str, Path]]:
    probes: list[tuple[str, Path]] = []
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"Probe must be label=path, got {raw!r}")
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Probe label is empty in {raw!r}")
        probes.append((label, Path(path).expanduser().resolve()))
    return probes


def load_probe_rows(probe_specs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_label, path in probe_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                row_id = str(payload.get("row_id") or "")
                task = normalize_task(str(payload.get("task") or ""))
                calibration = calibration_label(source_label, task)
                for detail in payload.get("details", []):
                    param_name = str(detail["param_name"])
                    expression = float(detail.get("expression", 0.0))
                    signed_effect = float(detail.get("signed_effect", 0.0))
                    module = module_from_param(param_name)
                    rows.append(
                        {
                            "source": source_label,
                            "calibration": calibration,
                            "task": task,
                            "owner": task,
                            "row_id": row_id,
                            "expert": str(detail["expert"]),
                            "layer": int(detail.get("layer", layer_from_param(param_name))),
                            "param_name": param_name,
                            "module": module,
                            "family": module_family(module),
                            "expression": expression,
                            "log_expression": math.log10(max(expression, 0.0) + EPS),
                            "signed_effect": signed_effect,
                        }
                    )
    return rows


def calibration_label(source_label: str, task: str) -> str:
    if source_label == "tm_signature_s32":
        if task == "tool":
            return "tool_signature_s32"
        if task == "memory":
            return "memory_signature_s32"
    if source_label == "memory_fulltraj_s32":
        return "memory_fulltraj_s32"
    if source_label == "livebench_code_s16":
        return "livebench_code_s16"
    if source_label == "livecodebench_code_s16":
        return "livecodebench_code_s16"
    return f"{source_label}:{task}"


def normalize_task(raw: str) -> str:
    raw = raw.lower().strip()
    aliases = {
        "tool_call": "tool",
        "tool-use": "tool",
        "memory_update": "memory",
        "coding": "code",
    }
    return aliases.get(raw, raw)


def ordered_calibrations(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "tool_signature_s32",
        "memory_signature_s32",
        "memory_fulltraj_s32",
        "livebench_code_s16",
        "livecodebench_code_s16",
    ]
    present = {str(row["calibration"]) for row in rows}
    ordered = [item for item in preferred if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def ordered_experts(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["expert"]) for row in rows}
    ordered = [item for item in EXPERT_ORDER if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def make_bins(values: list[float], *, bins: int, pad: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    lo = float(np.nanpercentile(arr, 0.2))
    hi = float(np.nanpercentile(arr, 99.8))
    lo = math.floor((lo - pad) * 2.0) / 2.0
    hi = math.ceil((hi + pad) * 2.0) / 2.0
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1)


def robust_abs_scale(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        return 1.0
    scale = float(np.nanmedian(arr))
    return scale if scale > 0 else float(np.nanpercentile(arr, 50))


def signed_log_scale(value: float, scale: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.log10(1.0 + abs(value) / max(scale, EPS)), value)


def summarize_global(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["calibration"]), str(row["task"]), str(row["expert"]))].append(row)
    output: list[dict[str, Any]] = []
    for (calibration, task, expert), items in sorted(grouped.items()):
        expr = np.asarray([float(item["expression"]) for item in items], dtype=np.float64)
        log_expr = np.asarray([float(item["log_expression"]) for item in items], dtype=np.float64)
        signed = np.asarray([float(item["signed_effect"]) for item in items], dtype=np.float64)
        output.append(
            {
                "calibration": calibration,
                "task": task,
                "expert": expert,
                "count": len(items),
                "expression_mean": float(np.mean(expr)),
                "expression_p50": float(np.quantile(expr, 0.50)),
                "expression_p90": float(np.quantile(expr, 0.90)),
                "expression_p99": float(np.quantile(expr, 0.99)),
                "log10_expression_p10": float(np.quantile(log_expr, 0.10)),
                "log10_expression_p50": float(np.quantile(log_expr, 0.50)),
                "log10_expression_p90": float(np.quantile(log_expr, 0.90)),
                "log10_expression_p99": float(np.quantile(log_expr, 0.99)),
                "signed_effect_mean": float(np.mean(signed)),
                "signed_effect_positive_frac": float(np.mean(signed > 0.0)),
                "signed_effect_negative_frac": float(np.mean(signed < 0.0)),
            }
        )
    return output


def summarize_module(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["calibration"]),
                str(row["task"]),
                str(row["expert"]),
                int(row["layer"]),
                str(row["module"]),
                str(row["family"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (calibration, task, expert, layer, module, family), items in sorted(grouped.items()):
        expr = np.asarray([float(item["expression"]) for item in items], dtype=np.float64)
        log_expr = np.asarray([float(item["log_expression"]) for item in items], dtype=np.float64)
        signed = np.asarray([float(item["signed_effect"]) for item in items], dtype=np.float64)
        output.append(
            {
                "calibration": calibration,
                "task": task,
                "expert": expert,
                "layer": layer,
                "module": module,
                "family": family,
                "count": len(items),
                "expression_mean": float(np.mean(expr)),
                "log10_expression_p50": float(np.quantile(log_expr, 0.50)),
                "log10_expression_p90": float(np.quantile(log_expr, 0.90)),
                "signed_effect_mean": float(np.mean(signed)),
                "signed_effect_positive_frac": float(np.mean(signed > 0.0)),
            }
        )
    return output


def compute_owner_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["calibration"]), str(row["task"]), str(row["row_id"]), str(row["param_name"]))].append(row)

    ratio_samples: dict[tuple[str, str, int, str, str], list[dict[str, float]]] = defaultdict(list)
    for (calibration, task, _row_id, param_name), items in grouped.items():
        by_expert = {str(item["expert"]): item for item in items}
        if task not in by_expert:
            continue
        owner_expr = float(by_expert[task]["expression"])
        other_expr = [float(item["expression"]) for expert, item in by_expert.items() if expert != task]
        if not other_expr:
            continue
        max_other = max(other_expr)
        total_expr = owner_expr + sum(other_expr)
        layer = int(by_expert[task]["layer"])
        module = str(by_expert[task]["module"])
        family = str(by_expert[task]["family"])
        ratio_samples[(calibration, task, layer, module, family)].append(
            {
                "owner_log10_vs_max_other": math.log10((owner_expr + EPS) / (max_other + EPS)),
                "owner_expression_share": owner_expr / (total_expr + EPS),
                "owner_rank": 1.0 + sum(1 for value in other_expr if value > owner_expr),
            }
        )

    output: list[dict[str, Any]] = []
    for (calibration, task, layer, module, family), items in sorted(ratio_samples.items()):
        log_ratio = np.asarray([item["owner_log10_vs_max_other"] for item in items], dtype=np.float64)
        share = np.asarray([item["owner_expression_share"] for item in items], dtype=np.float64)
        rank = np.asarray([item["owner_rank"] for item in items], dtype=np.float64)
        output.append(
            {
                "calibration": calibration,
                "task": task,
                "owner": task,
                "layer": layer,
                "module": module,
                "family": family,
                "count": len(items),
                "owner_log10_vs_max_other_mean": float(np.mean(log_ratio)),
                "owner_log10_vs_max_other_p50": float(np.quantile(log_ratio, 0.50)),
                "owner_log10_vs_max_other_p10": float(np.quantile(log_ratio, 0.10)),
                "owner_log10_vs_max_other_p90": float(np.quantile(log_ratio, 0.90)),
                "owner_expression_share_mean": float(np.mean(share)),
                "owner_expression_share_p50": float(np.quantile(share, 0.50)),
                "owner_top1_frac": float(np.mean(rank <= 1.0)),
                "owner_mean_rank": float(np.mean(rank)),
            }
        )
    return output


def load_conflict_rows(probe_specs: list[tuple[str, Path]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        tasks_by_source[str(row["source"])].add(str(row["task"]))

    output: list[dict[str, Any]] = []
    for source_label, rows_path in probe_specs:
        summary_path = rows_path.with_name("signed_utility_summary.json")
        if not summary_path.exists():
            continue
        payload = read_json(summary_path)
        conflict_summary = payload.get("conflict_summary") or {}
        for task in sorted(tasks_by_source[source_label]):
            calibration = calibration_label(source_label, task)
            task_payload = conflict_summary.get(task) or {}
            for key, stats in task_payload.items():
                match = re.match(r"layer_(\d+):([^|]+)\|([^|]+)$", str(key))
                if not match:
                    continue
                layer = int(match.group(1))
                left = match.group(2)
                right = match.group(3)
                output.append(
                    {
                        "source": source_label,
                        "calibration": calibration,
                        "task": task,
                        "layer": layer,
                        "pair": f"{left}|{right}",
                        "cosine_mean": float(stats.get("cosine_mean", float("nan"))),
                        "negative_fraction": float(stats.get("negative_fraction", float("nan"))),
                        "count": float(stats.get("count", 0.0)),
                    }
                )
    return output


def plot_expression_density(
    *,
    rows: list[dict[str, Any]],
    bins: np.ndarray,
    calibrations: list[str],
    experts: list[str],
    output_path: Path,
) -> None:
    fig, axes = subplot_grid(len(calibrations), width=5.4, height=3.4, sharex=True, sharey=True)
    centers = (bins[:-1] + bins[1:]) / 2.0
    for ax, calibration in zip(axes, calibrations):
        for expert in experts:
            values = [float(row["log_expression"]) for row in rows if row["calibration"] == calibration and row["expert"] == expert]
            if not values:
                continue
            hist = np.histogram(values, bins=bins)[0].astype(np.float64)
            density = normalize_density(hist, bins)
            ax.plot(centers, density, label=expert, linewidth=2.0)
        ax.set_title(calibration)
        ax.set_xlabel("log10 mean ||Delta W h||^2")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("density")
    axes[min(len(axes) - 1, 2)].legend(ncol=len(experts), fontsize=9)
    fig.suptitle("Activation-space expression density", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_expression_heatmaps(
    *,
    module_rows: list[dict[str, Any]],
    calibrations: list[str],
    experts: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(experts),
        len(calibrations),
        figsize=(3.4 * len(calibrations), 3.25 * len(experts)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(experts), len(calibrations))
    all_values = [float(row["log10_expression_p50"]) for row in module_rows]
    vmin = float(np.nanpercentile(all_values, 2))
    vmax = float(np.nanpercentile(all_values, 98))
    image = None
    for i, expert in enumerate(experts):
        for j, calibration in enumerate(calibrations):
            ax = axes[i, j]
            mat = matrix_from_rows(
                module_rows,
                calibration=calibration,
                expert=expert,
                value_key="log10_expression_p50",
            )
            image = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_title(calibration, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{expert}\nlayer")
            ax.set_yticks(range(0, 28, 4))
            ax.set_xticks(range(len(MODULE_ORDER)))
            ax.set_xticklabels(MODULE_ORDER, rotation=45, ha="right")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="median log10 expression")
    fig.suptitle("Layer-module activation expression", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_owner_ratio_heatmaps(
    *,
    owner_rows: list[dict[str, Any]],
    calibrations: list[str],
    output_path: Path,
) -> None:
    fig, axes = subplot_grid(len(calibrations), width=4.8, height=3.8, sharex=True, sharey=True)
    all_values = [float(row["owner_log10_vs_max_other_p50"]) for row in owner_rows]
    if all_values:
        bound = float(np.nanpercentile(np.abs(all_values), 95))
        bound = max(0.5, min(4.0, bound))
    else:
        bound = 1.0
    image = None
    for ax, calibration in zip(axes, calibrations):
        mat = matrix_from_rows(
            owner_rows,
            calibration=calibration,
            expert=None,
            value_key="owner_log10_vs_max_other_p50",
        )
        image = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        ax.axhline(15.5, color="black", linewidth=0.6, alpha=0.25)
        ax.set_title(calibration)
        ax.set_ylabel("layer")
        ax.set_xlabel("module")
        ax.set_yticks(range(0, 28, 4))
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels(MODULE_ORDER, rotation=45, ha="right")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="median log10(owner / max other)")
    fig.suptitle("Owner expression dominance on calibration activations", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_signed_effect_density(
    *,
    rows: list[dict[str, Any]],
    calibrations: list[str],
    experts: list[str],
    effect_scale: float,
    output_path: Path,
) -> None:
    values = np.asarray([float(row["signed_effect_scaled"]) for row in rows], dtype=np.float64)
    bound = float(np.nanpercentile(np.abs(values), 99.3))
    bound = max(0.5, min(6.0, bound))
    bins = np.linspace(-bound, bound, 181)
    fig, axes = subplot_grid(len(calibrations), width=5.4, height=3.4, sharex=True, sharey=True)
    centers = (bins[:-1] + bins[1:]) / 2.0
    for ax, calibration in zip(axes, calibrations):
        for expert in experts:
            subset = [
                float(row["signed_effect_scaled"])
                for row in rows
                if row["calibration"] == calibration and row["expert"] == expert
            ]
            if not subset:
                continue
            hist = np.histogram(subset, bins=bins)[0].astype(np.float64)
            density = normalize_density(hist, bins)
            ax.plot(centers, density, label=expert, linewidth=1.8)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(calibration)
        ax.set_xlabel("sign(s) log10(1 + |s| / median|s|)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("density")
    axes[min(len(axes) - 1, 2)].legend(ncol=len(experts), fontsize=9)
    fig.suptitle(f"Signed first-order effect density; median|s|={effect_scale:.3e}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_conflict_cosines(
    *,
    conflict_rows: list[dict[str, Any]],
    calibrations: list[str],
    output_path: Path,
) -> None:
    pairs = sorted({str(row["pair"]) for row in conflict_rows})
    fig, axes = subplot_grid(len(calibrations), width=4.9, height=2.8, sharex=True, sharey=True)
    image = None
    for ax, calibration in zip(axes, calibrations):
        mat = np.full((len(pairs), 28), np.nan, dtype=np.float64)
        for row in conflict_rows:
            if row["calibration"] != calibration:
                continue
            pair_idx = pairs.index(str(row["pair"]))
            layer = int(row["layer"])
            if 0 <= layer < 28:
                mat[pair_idx, layer] = float(row["cosine_mean"])
        image = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-0.08, vmax=0.08)
        ax.set_title(calibration)
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, 28, 4))
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="mean cosine of mean Delta W h")
    fig.suptitle("Activation-space update cosine", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def matrix_from_rows(
    rows: list[dict[str, Any]],
    *,
    calibration: str,
    expert: str | None,
    value_key: str,
) -> np.ndarray:
    mat = np.full((28, len(MODULE_ORDER)), np.nan, dtype=np.float64)
    for row in rows:
        if row["calibration"] != calibration:
            continue
        if expert is not None and row.get("expert") != expert:
            continue
        layer = int(row["layer"])
        module = str(row["module"])
        if module not in MODULE_ORDER or not (0 <= layer < 28):
            continue
        mat[layer, MODULE_ORDER.index(module)] = float(row[value_key])
    return mat


def subplot_grid(
    count: int,
    *,
    width: float,
    height: float,
    sharex: bool,
    sharey: bool,
) -> tuple[plt.Figure, list[plt.Axes]]:
    cols = min(3, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), sharex=sharex, sharey=sharey)
    axes_list = list(np.asarray(axes).reshape(-1))
    for ax in axes_list[count:]:
        ax.axis("off")
    return fig, axes_list[:count]


def normalize_density(hist: np.ndarray, bins: np.ndarray) -> np.ndarray:
    total = float(hist.sum())
    if total <= 0:
        return hist
    widths = np.diff(bins)
    return hist / (total * widths)


def module_from_param(param_name: str) -> str:
    mapping = {
        ".self_attn.q_proj.": "q",
        ".self_attn.k_proj.": "k",
        ".self_attn.v_proj.": "v",
        ".self_attn.o_proj.": "o",
        ".mlp.gate_proj.": "gate",
        ".mlp.up_proj.": "up",
        ".mlp.down_proj.": "down",
    }
    for needle, module in mapping.items():
        if needle in param_name:
            return module
    raise ValueError(f"unknown module for {param_name}")


def module_family(module: str) -> str:
    return "attention" if module in {"q", "k", "v", "o"} else "mlp"


def layer_from_param(param_name: str) -> int:
    match = re.search(r"model\.layers\.(\d+)\.", param_name)
    return int(match.group(1)) if match else -1


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    output_dir: Path,
    probe_specs: list[tuple[str, Path]],
    rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    owner_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
    effect_scale: float,
) -> None:
    stats_by_key = {(row["calibration"], row["expert"]): row for row in global_rows}
    owner_by_calibration = defaultdict(list)
    for row in owner_rows:
        owner_by_calibration[row["calibration"]].append(row)

    lines = [
        "# Activation-Space Task-Vector Distributions",
        "",
        "This report aggregates cached signed utility probes. No model forward/backward pass is run here.",
        "",
        "Definition:",
        "- expression = mean ||Delta W h||^2 over selected calibration tokens",
        "- signed_effect = - <d loss / d y, Delta W h>; positive means first-order teacher-forced loss reduction",
        "",
        "Inputs:",
    ]
    for label, probe_path in probe_specs:
        lines.append(f"- {label}: `{probe_path}`")
    lines.extend(
        [
            "",
            "Outputs:",
            f"- `01_activation_expression_density.png`: distribution of log10 expression by expert and calibration",
            f"- `02_activation_expression_heatmaps.png`: layer-module median expression heatmaps",
            f"- `03_owner_vs_nonowner_expression_ratio.png`: median log10(owner expression / strongest non-owner)",
            f"- `04_signed_effect_density.png`: signed first-order effect density, scaled by median |s| = {effect_scale:.3e}",
            f"- `05_activation_update_cosines.png`: activation-space cosine of mean Delta W h, if summaries are present",
            "",
            "Global expression summary:",
            "",
            "| calibration | tool p50 | memory p50 | code p50 | owner share p50 | owner top1 frac |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for calibration in ordered_calibrations(rows):
        p50 = {
            expert: stats_by_key.get((calibration, expert), {}).get("log10_expression_p50", float("nan"))
            for expert in EXPERT_ORDER
        }
        owner_items = owner_by_calibration.get(calibration, [])
        if owner_items:
            owner_share = float(np.nanmedian([float(item["owner_expression_share_p50"]) for item in owner_items]))
            owner_top1 = float(np.nanmean([float(item["owner_top1_frac"]) for item in owner_items]))
        else:
            owner_share = float("nan")
            owner_top1 = float("nan")
        lines.append(
            "| "
            + f"{calibration} | {p50['tool']:.3f} | {p50['memory']:.3f} | {p50['code']:.3f} | "
            + f"{owner_share:.3f} | {owner_top1:.3f} |"
        )

    if conflict_rows:
        lines.extend(["", "Activation cosine summary:", ""])
        pair_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        neg_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in conflict_rows:
            key = (str(row["calibration"]), str(row["pair"]))
            pair_groups[key].append(float(row["cosine_mean"]))
            neg_groups[key].append(float(row["negative_fraction"]))
        lines.extend(["| calibration | pair | mean cosine | mean negative frac |", "| --- | --- | ---: | ---: |"])
        for key in sorted(pair_groups):
            calibration, pair = key
            lines.append(
                f"| {calibration} | {pair} | {np.nanmean(pair_groups[key]):.4f} | {np.nanmean(neg_groups[key]):.3f} |"
            )

    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- High expression means the residual is active on the calibration hidden states; it is not automatically useful.",
            "- Positive signed_effect is the local first-order utility signal; expression and utility can diverge.",
            "- Owner dominance near or below zero means another expert produces equal or larger activation updates on that behavior span.",
            "",
            f"Output directory: `{output_dir}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
