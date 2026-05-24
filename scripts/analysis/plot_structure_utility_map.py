#!/usr/bin/env python3
"""Plot structure-vs-utility maps for task-vector merge diagnostics.

This figure joins two cached analyses:

1. parameter-space relative structure, measured by each expert residual RMS
   against the strongest other expert at the same layer/module block;
2. activation-space local utility, measured by cached signed utility probes.

The resulting map is diagnostic only. It is meant to show where a residual is
structurally large/small and locally useful/harmful, without directly turning
the score into a merge gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PARAM_STATS = Path(
    "/tmp/shared-storage/ExpertGym/task_vector_value_distributions/opvec4_20260522/task_vector_value_block_stats.csv"
)
DEFAULT_ACTIVATION_STATS = Path(
    "/tmp/shared-storage/ExpertGym/task_vector_activation_distributions/"
    "opvec4_rcrf_calibration_20260522/activation_module_stats.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/ExpertGym/structure_utility_maps/opvec4_rcrf_calibration_20260522"
)
EXPERT_ORDER = ["tool", "memory", "code"]
CALIBRATION_ORDER = [
    "tool_signature_s32",
    "memory_signature_s32",
    "memory_fulltraj_s32",
    "livebench_code_s16",
    "livecodebench_code_s16",
]
FAMILY_MARKERS = {"attention": "o", "mlp": "^"}
EXPERT_COLORS = {"tool": "#1f77b4", "memory": "#ff7f0e", "code": "#2ca02c"}
EPS = 1.0e-30


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    param_rows = read_csv(Path(args.param_stats).expanduser())
    activation_rows = read_csv(Path(args.activation_stats).expanduser())
    structure_rows = compute_structure_rows(param_rows)
    joined_rows = join_structure_and_utility(structure_rows, activation_rows)
    add_scaled_utility(joined_rows)
    add_expression_size(joined_rows)
    add_quadrants(joined_rows)

    write_csv(output_dir / "structure_utility_points.csv", joined_rows)
    quadrant_rows = summarize_quadrants(joined_rows)
    write_csv(output_dir / "structure_utility_quadrants.csv", quadrant_rows)
    write_json(
        output_dir / "structure_utility_metadata.json",
        {
            "param_stats": str(Path(args.param_stats).expanduser()),
            "activation_stats": str(Path(args.activation_stats).expanduser()),
            "x_axis": "log10(rms_delta_expert / max_other_expert_rms) at the same layer/module",
            "y_axis": "asinh(signed_effect_mean / median_abs_signed_effect_mean_per_calibration)",
            "point_size": "relative activation expression_mean within each calibration",
        },
    )
    plot_structure_utility_map(joined_rows, output_dir / "01_structure_utility_map.png")
    plot_quadrant_composition(joined_rows, output_dir / "02_quadrant_composition.png")
    write_report(output_dir / "README.md", joined_rows, quadrant_rows, output_dir)
    print(f"Wrote structure-utility maps to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param-stats", default=str(DEFAULT_PARAM_STATS))
    parser.add_argument("--activation-stats", default=str(DEFAULT_ACTIVATION_STATS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def compute_structure_rows(param_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in param_rows:
        expert = str(row["expert"])
        layer = int(row["layer"])
        module = str(row["module"])
        grouped[(layer, module)][expert] = row

    output: list[dict[str, Any]] = []
    for (layer, module), by_expert in sorted(grouped.items()):
        rms_by_expert = {expert: float(row["rms"]) for expert, row in by_expert.items()}
        total_rms = sum(rms_by_expert.values()) + EPS
        for expert, row in by_expert.items():
            rms = rms_by_expert[expert]
            other = [value for name, value in rms_by_expert.items() if name != expert]
            max_other = max(other) if other else EPS
            output.append(
                {
                    "expert": expert,
                    "layer": layer,
                    "module": module,
                    "family": str(row["module_family"]),
                    "param_name": str(row["param_name"]),
                    "param_rms": rms,
                    "param_log10_rms": math.log10(rms + EPS),
                    "param_log10_vs_max_other": math.log10((rms + EPS) / (max_other + EPS)),
                    "param_rms_share": rms / total_rms,
                    "param_p99_log10_abs": float(row["p99_log10_abs"]),
                }
            )
    return output


def join_structure_and_utility(
    structure_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    structure_by_key = {
        (str(row["expert"]), int(row["layer"]), str(row["module"])): row
        for row in structure_rows
    }
    output: list[dict[str, Any]] = []
    for row in activation_rows:
        key = (str(row["expert"]), int(row["layer"]), str(row["module"]))
        structure = structure_by_key.get(key)
        if structure is None:
            continue
        joined = {
            **structure,
            "calibration": str(row["calibration"]),
            "task": str(row["task"]),
            "expression_mean": float(row["expression_mean"]),
            "log10_expression_p50": float(row["log10_expression_p50"]),
            "log10_expression_p90": float(row["log10_expression_p90"]),
            "signed_effect_mean": float(row["signed_effect_mean"]),
            "signed_effect_positive_frac": float(row["signed_effect_positive_frac"]),
        }
        output.append(joined)
    return output


def add_scaled_utility(rows: list[dict[str, Any]]) -> None:
    by_calibration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_calibration[str(row["calibration"])].append(row)
    for calibration, items in by_calibration.items():
        values = np.asarray([abs(float(row["signed_effect_mean"])) for row in items], dtype=np.float64)
        nonzero = values[values > 0.0]
        scale = float(np.nanmedian(nonzero)) if nonzero.size else 1.0
        scale = max(scale, EPS)
        for row in items:
            signed = float(row["signed_effect_mean"])
            row["utility_scale"] = scale
            row["utility_asinh"] = math.asinh(signed / scale)
            row["utility_sign"] = "positive" if signed > 0.0 else "negative" if signed < 0.0 else "zero"


def add_expression_size(rows: list[dict[str, Any]]) -> None:
    by_calibration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_calibration[str(row["calibration"])].append(row)
    for items in by_calibration.values():
        logs = np.asarray([math.log10(max(float(row["expression_mean"]), 0.0) + EPS) for row in items])
        lo = float(np.nanpercentile(logs, 10))
        hi = float(np.nanpercentile(logs, 95))
        if hi <= lo:
            hi = lo + 1.0
        for row, value in zip(items, logs):
            norm = min(1.0, max(0.0, (float(value) - lo) / (hi - lo)))
            row["log10_expression_mean"] = float(value)
            row["plot_size"] = 22.0 + 105.0 * norm


def add_quadrants(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        structure_pos = float(row["param_log10_vs_max_other"]) >= 0.0
        utility_pos = float(row["signed_effect_mean"]) >= 0.0
        if structure_pos and utility_pos:
            quadrant = "structural_anchor"
        elif structure_pos and not utility_pos:
            quadrant = "structural_spillover"
        elif not structure_pos and utility_pos:
            quadrant = "hidden_positive"
        else:
            quadrant = "weak_or_harmful"
        row["quadrant"] = quadrant


def summarize_quadrants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["calibration"]), str(row["expert"]))].append(row)
    output: list[dict[str, Any]] = []
    for (calibration, expert), items in sorted(grouped.items()):
        counts = Counter(str(row["quadrant"]) for row in items)
        total = len(items)
        output.append(
            {
                "calibration": calibration,
                "expert": expert,
                "total": total,
                "structural_anchor_frac": counts["structural_anchor"] / total,
                "structural_spillover_frac": counts["structural_spillover"] / total,
                "hidden_positive_frac": counts["hidden_positive"] / total,
                "weak_or_harmful_frac": counts["weak_or_harmful"] / total,
                "median_structure_log10_vs_other": float(
                    np.nanmedian([float(row["param_log10_vs_max_other"]) for row in items])
                ),
                "median_utility_asinh": float(np.nanmedian([float(row["utility_asinh"]) for row in items])),
                "median_expression_log10": float(
                    np.nanmedian([float(row["log10_expression_mean"]) for row in items])
                ),
            }
        )
    return output


def plot_structure_utility_map(rows: list[dict[str, Any]], output_path: Path) -> None:
    calibrations = ordered_values((str(row["calibration"]) for row in rows), CALIBRATION_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.8), sharex=True, sharey=True)
    axes_list = list(axes.reshape(-1))
    for ax in axes_list[len(calibrations) :]:
        ax.axis("off")

    x_values = np.asarray([float(row["param_log10_vs_max_other"]) for row in rows], dtype=np.float64)
    y_values = np.asarray([float(row["utility_asinh"]) for row in rows], dtype=np.float64)
    x_bound = max(0.8, float(np.nanpercentile(np.abs(x_values), 99)))
    y_bound = max(1.0, float(np.nanpercentile(np.abs(y_values), 98)))
    x_bound = min(4.0, x_bound)
    y_bound = min(7.0, y_bound)

    for ax, calibration in zip(axes_list, calibrations):
        subset = [row for row in rows if row["calibration"] == calibration]
        ax.axvspan(0.0, x_bound, color="#f5f1dc", alpha=0.35, linewidth=0)
        ax.axhspan(0.0, y_bound, color="#e8f3e6", alpha=0.28, linewidth=0)
        ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)
        for family, marker in FAMILY_MARKERS.items():
            for expert in EXPERT_ORDER:
                items = [
                    row
                    for row in subset
                    if row["expert"] == expert and row["family"] == family
                ]
                if not items:
                    continue
                ax.scatter(
                    [float(row["param_log10_vs_max_other"]) for row in items],
                    [float(row["utility_asinh"]) for row in items],
                    s=[float(row["plot_size"]) for row in items],
                    marker=marker,
                    color=EXPERT_COLORS.get(expert, "tab:gray"),
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.68,
                    label=f"{expert} {family}",
                )
        for expert in EXPERT_ORDER:
            items = [row for row in subset if row["expert"] == expert]
            if not items:
                continue
            ax.scatter(
                [float(np.nanmedian([float(row["param_log10_vs_max_other"]) for row in items]))],
                [float(np.nanmedian([float(row["utility_asinh"]) for row in items]))],
                s=210,
                marker="X",
                color=EXPERT_COLORS.get(expert, "tab:gray"),
                edgecolor="black",
                linewidth=0.8,
                alpha=0.98,
            )
        ax.set_title(calibration)
        ax.set_xlim(-x_bound, x_bound)
        ax.set_ylim(-y_bound, y_bound)
        ax.grid(alpha=0.18)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=expert, markersize=8)
        for expert, color in EXPERT_COLORS.items()
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], marker="o", linestyle="", color="black", label="attention", markersize=7),
            plt.Line2D([0], [0], marker="^", linestyle="", color="black", label="mlp", markersize=7),
            plt.Line2D([0], [0], marker="X", linestyle="", color="black", label="expert median", markersize=9),
        ]
    )
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.055), frameon=True)
    fig.supylabel("activation utility: asinh(signed_effect / median |signed_effect|)", x=0.012)
    fig.supxlabel("relative parameter structure: log10(rms expert / max other rms)", y=0.018)
    fig.suptitle("Structure-Utility Map: parameter dominance vs activation utility", y=0.985)
    fig.tight_layout(rect=(0.035, 0.045, 0.94, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quadrant_composition(rows: list[dict[str, Any]], output_path: Path) -> None:
    calibrations = ordered_values((str(row["calibration"]) for row in rows), CALIBRATION_ORDER)
    quadrants = ["structural_anchor", "structural_spillover", "hidden_positive", "weak_or_harmful"]
    colors = {
        "structural_anchor": "#6cae75",
        "structural_spillover": "#d97171",
        "hidden_positive": "#7aa6d9",
        "weak_or_harmful": "#b9b9b9",
    }
    fig, axes = plt.subplots(len(calibrations), 1, figsize=(10.5, 2.1 * len(calibrations)), sharex=True)
    axes = list(np.asarray(axes).reshape(-1))
    for ax, calibration in zip(axes, calibrations):
        subset = [row for row in rows if row["calibration"] == calibration]
        left = np.zeros(len(EXPERT_ORDER), dtype=np.float64)
        for quadrant in quadrants:
            values = []
            for expert in EXPERT_ORDER:
                items = [row for row in subset if row["expert"] == expert]
                if not items:
                    values.append(0.0)
                    continue
                values.append(sum(row["quadrant"] == quadrant for row in items) / len(items))
            ax.barh(EXPERT_ORDER, values, left=left, color=colors[quadrant], label=quadrant)
            left += np.asarray(values)
        ax.set_title(calibration)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.18)
    axes[-1].set_xlabel("fraction of layer-module blocks")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.55), fontsize=9)
    fig.suptitle("Structure-utility quadrant composition", y=1.005)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ordered_values(values: Any, preferred: list[str]) -> list[str]:
    present = set(values)
    ordered = [item for item in preferred if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    quadrant_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    lines = [
        "# Structure-Utility Map",
        "",
        "This diagnostic joins relative parameter structure with activation-space signed utility.",
        "",
        "Axes:",
        "- x = log10(rms expert residual / strongest other expert residual) for the same layer/module block",
        "- y = asinh(signed_effect_mean / median absolute signed_effect_mean within the calibration)",
        "- point size = activation expression_mean percentile within the calibration",
        "- marker = attention or MLP family; X marker = per-expert median",
        "",
        "Quadrants:",
        "- structural_anchor: structurally dominant and positive utility",
        "- structural_spillover: structurally dominant but negative utility",
        "- hidden_positive: structurally subordinate but positive utility",
        "- weak_or_harmful: structurally subordinate and non-positive utility",
        "",
        "Median summary:",
        "",
        "| calibration | expert | median structure | median utility | anchor | spillover | hidden positive | weak/harmful |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in quadrant_rows:
        lines.append(
            "| "
            + f"{row['calibration']} | {row['expert']} | "
            + f"{float(row['median_structure_log10_vs_other']):.3f} | "
            + f"{float(row['median_utility_asinh']):.3f} | "
            + f"{float(row['structural_anchor_frac']):.3f} | "
            + f"{float(row['structural_spillover_frac']):.3f} | "
            + f"{float(row['hidden_positive_frac']):.3f} | "
            + f"{float(row['weak_or_harmful_frac']):.3f} |"
        )

    lines.extend(
        [
            "",
            "Files:",
            "- `01_structure_utility_map.png`",
            "- `02_quadrant_composition.png`",
            "- `structure_utility_points.csv`",
            "- `structure_utility_quadrants.csv`",
            "",
            f"Output directory: `{output_dir}`",
            f"Rows: {len(rows)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
