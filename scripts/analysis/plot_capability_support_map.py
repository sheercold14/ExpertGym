#!/usr/bin/env python3
"""Plot trajectory-conditioned capability support at residual-block granularity.

This figure is a paper-style replacement for the dense parameter-offset overlay.
It intentionally uses successful/positive trajectory probes only.  The goal is
to show which expert/layer/module residual blocks are aligned with each
successful probe set, not to claim failure discrimination.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


DEFAULT_INPUT_CSV = (
    "/tmp/shared-storage/ExpertGym/task_vector_activation_distributions/"
    "opvec4_rcrf_calibration_20260522/activation_module_stats.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/tmp/shared-storage/ExpertGym/capability_support_map/"
    "opvec4_rcrf_calibration_20260522"
)

EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
EXPERT_COLORS = {
    "tool": "#4C78A8",
    "memory": "#F58518",
    "code": "#54A24B",
}
CALIBRATION_SPECS = [
    {
        "key": "tool_signature_s32",
        "title": "Tool-use\nsuccess signature",
        "span": "selected output tokens",
        "n": 32,
    },
    {
        "key": "memory_signature_s32",
        "title": "Memory\nsuccess signature",
        "span": "selected output tokens",
        "n": 32,
    },
    {
        "key": "memory_fulltraj_s32",
        "title": "Memory\nfull trajectory",
        "span": "selected trajectory outputs",
        "n": 32,
    },
    {
        "key": "livebench_code_s16",
        "title": "LiveBench code\nreference-pass",
        "span": "final code tokens",
        "n": 16,
    },
    {
        "key": "livecodebench_code_s16",
        "title": "LiveCodeBench code\nreference-pass",
        "span": "final code tokens",
        "n": 16,
    },
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.input_csv).expanduser())
    calibrations = [spec["key"] for spec in CALIBRATION_SPECS if spec["key"] in {r["calibration"] for r in rows}]
    if not calibrations:
        raise ValueError("No known calibrations found in input CSV.")

    fig_path_png = output_dir / "capability_support_map_v1.png"
    fig_path_pdf = output_dir / "capability_support_map_v1.pdf"
    plot_support_map(rows, calibrations, fig_path_png, fig_path_pdf)
    write_report(output_dir / "README.md", output_dir, calibrations)
    print(f"Wrote capability support map to {fig_path_png}")
    print(f"Wrote capability support map to {fig_path_pdf}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "calibration": row["calibration"],
                    "task": row.get("task", ""),
                    "expert": row["expert"],
                    "layer": int(row["layer"]),
                    "module": row["module"],
                    "family": row.get("family", ""),
                    "count": int(float(row.get("count", 0) or 0)),
                    "signed_effect_mean": float(row["signed_effect_mean"]),
                    "utility_asinh": float(row["utility_asinh"]) if row.get("utility_asinh") else math.nan,
                    "signed_effect_positive_frac": float(row["signed_effect_positive_frac"]),
                    "expression_mean": float(row["expression_mean"]),
                }
            )
    fill_missing_support_scale(rows)
    return rows


def fill_missing_support_scale(rows: list[dict[str, Any]]) -> None:
    by_calibration: dict[str, list[float]] = {}
    for row in rows:
        if math.isfinite(float(row["utility_asinh"])):
            continue
        by_calibration.setdefault(str(row["calibration"]), []).append(abs(float(row["signed_effect_mean"])))

    scales = {}
    for calibration, values in by_calibration.items():
        positive = [value for value in values if value > 0.0 and math.isfinite(value)]
        if not positive:
            scales[calibration] = 1.0
        else:
            scales[calibration] = max(float(np.nanmedian(np.asarray(positive, dtype=np.float64))), 1.0e-30)

    for row in rows:
        if math.isfinite(float(row["utility_asinh"])):
            continue
        scale = scales[str(row["calibration"])]
        row["utility_asinh"] = math.asinh(float(row["signed_effect_mean"]) / scale)


def plot_support_map(
    rows: list[dict[str, Any]],
    calibrations: list[str],
    png_path: Path,
    pdf_path: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    row_keys = [(expert, module) for expert in EXPERT_ORDER for module in MODULE_ORDER]
    row_labels = [module for _expert, module in row_keys]
    layers = list(range(28))
    specs = [spec for spec in CALIBRATION_SPECS if spec["key"] in calibrations]
    ncols = len(specs)

    values_by_calibration = {
        spec["key"]: matrix_for_calibration(rows, spec["key"], row_keys, layers)
        for spec in specs
    }
    consistency_by_calibration = {
        spec["key"]: consistency_for_calibration(rows, spec["key"], row_keys, layers)
        for spec in specs
    }

    finite_values = np.concatenate(
        [
            value[np.isfinite(value)].reshape(-1)
            for value in values_by_calibration.values()
            if np.isfinite(value).any()
        ]
    )
    bound = max(2.0, float(np.nanpercentile(np.abs(finite_values), 97)))
    bound = min(bound, 5.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)

    fig = plt.figure(figsize=(14.2, 6.95), dpi=180)
    grid = fig.add_gridspec(
        2,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.045],
        height_ratios=[5.6, 1.05],
        left=0.075,
        right=0.955,
        top=0.815,
        bottom=0.16,
        wspace=0.07,
        hspace=0.22,
    )

    fig.text(
        0.075,
        0.965,
        "Successful-probe residual alignment",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        "Color: normalized signed first-order alignment on selected successful outputs; dot: |alignment| >= 2 and sign agreement >= 75%. Negative values are anti-aligned with these probes.",
        ha="left",
        va="top",
        fontsize=7.5,
    )

    image = None
    for col, spec in enumerate(specs):
        ax = fig.add_subplot(grid[0, col])
        value = values_by_calibration[spec["key"]]
        image = ax.imshow(value, aspect="auto", interpolation="nearest", cmap="RdBu", norm=norm)
        ax.set_title(f"{spec['title']}\n{spec['span']}, n={spec['n']}", pad=5)
        ax.set_xticks([0, 7, 15, 23, 27])
        ax.set_xticklabels(["0", "7", "15", "23", "27"])
        ax.set_xlabel("Layer")
        ax.set_xlim(-0.5, len(layers) - 0.5)

        if col == 0:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
        else:
            ax.set_yticks([])

        draw_structure_guides(ax, row_keys, layers)
        draw_consistency_dots(ax, value, consistency_by_calibration[spec["key"]])

        if col == 0:
            annotate_expert_groups(ax)

        bar_ax = fig.add_subplot(grid[1, col])
        plot_support_share(bar_ax, rows, spec["key"])
        if col == 0:
            bar_ax.set_ylabel("Positive signed-effect\nmass share", rotation=0, labelpad=34, va="center")

    cax = fig.add_subplot(grid[0, -1])
    if image is not None:
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Normalized signed alignment\nasinh(effect / calibration median |effect|)")

    legend_ax = fig.add_axes([0.075, 0.055, 0.86, 0.05])
    legend_ax.axis("off")
    x = 0.0
    for expert in EXPERT_ORDER:
        legend_ax.add_patch(
            mpl.patches.Rectangle((x, 0.35), 0.025, 0.25, color=EXPERT_COLORS[expert], transform=legend_ax.transAxes)
        )
        legend_ax.text(x + 0.032, 0.475, expert, transform=legend_ax.transAxes, va="center", fontsize=7)
        x += 0.11
    legend_ax.text(
        x + 0.03,
        0.475,
        "Successful/positive probes only; negative colors are anti-alignment with these probes, not a failure or harm claim.",
        transform=legend_ax.transAxes,
        va="center",
        fontsize=7,
        color="#333333",
    )

    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def matrix_for_calibration(
    rows: list[dict[str, Any]],
    calibration: str,
    row_keys: list[tuple[str, str]],
    layers: list[int],
) -> np.ndarray:
    indexed = {
        (row["expert"], row["module"], row["layer"]): row
        for row in rows
        if row["calibration"] == calibration
    }
    matrix = np.full((len(row_keys), len(layers)), np.nan, dtype=np.float64)
    for row_idx, (expert, module) in enumerate(row_keys):
        for col_idx, layer in enumerate(layers):
            row = indexed.get((expert, module, layer))
            if row is not None:
                matrix[row_idx, col_idx] = row["utility_asinh"]
    return matrix


def consistency_for_calibration(
    rows: list[dict[str, Any]],
    calibration: str,
    row_keys: list[tuple[str, str]],
    layers: list[int],
) -> np.ndarray:
    indexed = {
        (row["expert"], row["module"], row["layer"]): row
        for row in rows
        if row["calibration"] == calibration
    }
    matrix = np.full((len(row_keys), len(layers)), np.nan, dtype=np.float64)
    for row_idx, (expert, module) in enumerate(row_keys):
        for col_idx, layer in enumerate(layers):
            row = indexed.get((expert, module, layer))
            if row is None:
                continue
            frac = row["signed_effect_positive_frac"]
            direction = 1.0 if row["signed_effect_mean"] >= 0.0 else -1.0
            matrix[row_idx, col_idx] = frac if direction >= 0.0 else 1.0 - frac
    return matrix


def draw_structure_guides(ax: plt.Axes, row_keys: list[tuple[str, str]], layers: list[int]) -> None:
    for boundary in [7.5, 15.5, 23.5]:
        ax.axvline(boundary, color="black", linewidth=0.55, alpha=0.28)
    for row_boundary in [6.5, 13.5]:
        ax.axhline(row_boundary, color="black", linewidth=0.7, alpha=0.35)
    for row_idx, (_expert, module) in enumerate(row_keys):
        if module == "gate":
            ax.axhline(row_idx - 0.5, color="black", linewidth=0.35, alpha=0.18)
    ax.tick_params(length=2.0, width=0.5)


def draw_consistency_dots(ax: plt.Axes, values: np.ndarray, consistency: np.ndarray) -> None:
    ys, xs = np.where((np.abs(values) >= 2.0) & (consistency >= 0.75))
    if len(xs):
        ax.scatter(xs, ys, s=2.8, c="#111111", alpha=0.38, linewidths=0)


def annotate_expert_groups(ax: plt.Axes) -> None:
    centers = [3, 10, 17]
    for expert, center in zip(EXPERT_ORDER, centers):
        ax.text(
            -4.2,
            center,
            expert,
            ha="right",
            va="center",
            rotation=90,
            fontsize=7,
            fontweight="bold",
            color=EXPERT_COLORS.get(expert, "#333333"),
            clip_on=False,
        )


def plot_support_share(ax: plt.Axes, rows: list[dict[str, Any]], calibration: str) -> None:
    totals = []
    for expert in EXPERT_ORDER:
        positive = sum(
            max(float(row["signed_effect_mean"]), 0.0)
            for row in rows
            if row["calibration"] == calibration and row["expert"] == expert
        )
        totals.append(positive)
    total = sum(totals)
    shares = [value / total if total > 0.0 else 0.0 for value in totals]

    left = 0.0
    for expert, share in zip(EXPERT_ORDER, shares):
        ax.barh([0], [share], left=left, height=0.42, color=EXPERT_COLORS[expert], edgecolor="white", linewidth=0.4)
        if share >= 0.12:
            ax.text(left + share / 2, 0, f"{share:.0%}", ha="center", va="center", fontsize=6, color="white")
        left += share
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", ".5", "1"])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="x", length=2, width=0.5, pad=1)


def write_report(path: Path, output_dir: Path, calibrations: list[str]) -> None:
    lines = [
        "# Capability Support Map v1",
        "",
        "This draft reframes the previous signed-energy overlay as successful-probe residual alignment.",
        "",
        "Interpretation:",
        "- rows are expert/module residual blocks",
        "- columns are layers",
        "- each facet is one successful/positive trajectory probe set",
        "- color is normalized signed first-order alignment on selected successful outputs",
        "- black dots mark cells with |alignment| >= 2 and at least 75% sample sign agreement",
        "- bottom bars show each expert's share of summed positive signed-effect mass",
        "",
        "Guardrail: this figure does not use failure trajectories. Negative support should be read as anti-alignment with these successful probes, not as evidence of causing failure or harm.",
        "",
        "Calibrations:",
    ]
    for calibration in calibrations:
        lines.append(f"- `{calibration}`")
    lines.extend(
        [
            "",
            f"Output directory: `{output_dir}`",
            "",
            "Files:",
            "- `capability_support_map_v1.png`",
            "- `capability_support_map_v1.pdf`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
