#!/usr/bin/env python3
"""Overlay activation utility on the dense signed task-vector field.

This keeps the previous flattened signed residual visualization:

    x axis: global OP-VEC parameter position
    y axis: signed Delta theta value
    density: sampled parameter count

and adds an aligned utility strip for each expert:

    strip color: activation signed_effect_mean for the layer/module block

The goal is diagnostic: see which regions of the signed residual field are
locally useful or harmful on a behavior calibration without converting that
local signal into a merge gate.
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
from matplotlib.colors import TwoSlopeNorm


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_SIGNED_HISTOGRAMS = (
    "/tmp/shared-storage/ExpertGym/task_vector_signed_flattened_positions/"
    "opvec4_20260522/signed_flattened_position_histograms.json"
)
DEFAULT_ACTIVATION_STATS = (
    "/tmp/shared-storage/ExpertGym/task_vector_activation_distributions/"
    "opvec4_rcrf_calibration_20260522/activation_module_stats.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/tmp/shared-storage/ExpertGym/signed_energy_utility_overlay/"
    "opvec4_rcrf_calibration_20260522"
)
EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
CALIBRATION_ORDER = [
    "tool_signature_s32",
    "memory_signature_s32",
    "memory_fulltraj_s32",
    "livebench_code_s16",
    "livecodebench_code_s16",
]
EPS = 1.0e-30


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    blocks = build_blocks(manifest)
    assign_offsets(blocks)
    layer_boundaries = compute_layer_boundaries(blocks)

    signed_payload = read_json(Path(args.signed_histograms).expanduser())
    x_edges = np.asarray(signed_payload["x_bin_edges"], dtype=np.float64)
    y_edges = np.asarray(signed_payload["signed_value_bin_edges"], dtype=np.float64)
    experts = ordered_values(signed_payload["experts"], EXPERT_ORDER)
    histograms = {
        expert: np.asarray(signed_payload["histograms"][expert], dtype=np.float64)
        for expert in experts
    }

    activation_rows = read_csv(Path(args.activation_stats).expanduser())
    activation = index_activation_rows(activation_rows)
    calibrations = ordered_values(
        [str(row["calibration"]) for row in activation_rows],
        CALIBRATION_ORDER,
    )

    overlay_rows = []
    for calibration in calibrations:
        scale = utility_scale(activation, calibration)
        overlay_rows.extend(
            block_overlay_rows(
                blocks=blocks,
                activation=activation,
                calibration=calibration,
                scale=scale,
                experts=experts,
            )
        )
        plot_calibration_overlay(
            calibration=calibration,
            scale=scale,
            blocks=blocks,
            layer_boundaries=layer_boundaries,
            x_edges=x_edges,
            y_edges=y_edges,
            histograms=histograms,
            activation=activation,
            experts=experts,
            output_path=output_dir / f"01_{calibration}_signed_field_with_utility.png",
        )

    write_csv(output_dir / "signed_energy_utility_overlay_blocks.csv", overlay_rows)
    write_report(output_dir / "README.md", output_dir, calibrations, experts)
    print(f"Wrote signed energy utility overlays to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--signed-histograms", default=DEFAULT_SIGNED_HISTOGRAMS)
    parser.add_argument("--activation-stats", default=DEFAULT_ACTIVATION_STATS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_blocks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in manifest["basis_entries"]:
        param_name = str(entry["param_name"])
        shape = [int(dim) for dim in entry["shape"]]
        grouped.setdefault(
            param_name,
            {
                "param_name": param_name,
                "layer": layer_from_param(param_name),
                "module": module_from_param(param_name),
                "shape": shape,
                "numel": math.prod(shape),
            },
        )
    return sorted(grouped.values(), key=lambda row: (row["layer"], MODULE_ORDER.index(row["module"])))


def assign_offsets(blocks: list[dict[str, Any]]) -> None:
    offset = 0
    for block in blocks:
        block["start"] = offset
        offset += int(block["numel"])
        block["end"] = offset


def compute_layer_boundaries(blocks: list[dict[str, Any]]) -> list[int]:
    starts = {}
    for block in blocks:
        starts.setdefault(int(block["layer"]), int(block["start"]))
    return [starts[layer] for layer in sorted(starts)]


def index_activation_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, str], dict[str, float]]:
    output: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for row in rows:
        key = (
            str(row["calibration"]),
            str(row["expert"]),
            int(row["layer"]),
            str(row["module"]),
        )
        output[key] = {
            "signed_effect_mean": float(row["signed_effect_mean"]),
            "signed_effect_positive_frac": float(row["signed_effect_positive_frac"]),
            "expression_mean": float(row["expression_mean"]),
            "log10_expression_p50": float(row["log10_expression_p50"]),
        }
    return output


def utility_scale(activation: dict[tuple[str, str, int, str], dict[str, float]], calibration: str) -> float:
    values = [
        abs(float(stats["signed_effect_mean"]))
        for key, stats in activation.items()
        if key[0] == calibration and abs(float(stats["signed_effect_mean"])) > 0.0
    ]
    if not values:
        return 1.0
    scale = float(np.nanmedian(np.asarray(values, dtype=np.float64)))
    return max(scale, EPS)


def block_overlay_rows(
    *,
    blocks: list[dict[str, Any]],
    activation: dict[tuple[str, str, int, str], dict[str, float]],
    calibration: str,
    scale: float,
    experts: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expert in experts:
        for block in blocks:
            stats = activation.get((calibration, expert, int(block["layer"]), str(block["module"])))
            if stats is None:
                continue
            signed = float(stats["signed_effect_mean"])
            rows.append(
                {
                    "calibration": calibration,
                    "expert": expert,
                    "layer": int(block["layer"]),
                    "module": str(block["module"]),
                    "param_name": str(block["param_name"]),
                    "start": int(block["start"]),
                    "end": int(block["end"]),
                    "start_billion": int(block["start"]) / 1e9,
                    "end_billion": int(block["end"]) / 1e9,
                    "signed_effect_mean": signed,
                    "utility_scale": scale,
                    "utility_asinh": math.asinh(signed / scale),
                    "signed_effect_positive_frac": float(stats["signed_effect_positive_frac"]),
                    "expression_mean": float(stats["expression_mean"]),
                    "log10_expression_p50": float(stats["log10_expression_p50"]),
                }
            )
    return rows


def plot_calibration_overlay(
    *,
    calibration: str,
    scale: float,
    blocks: list[dict[str, Any]],
    layer_boundaries: list[int],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    histograms: dict[str, np.ndarray],
    activation: dict[tuple[str, str, int, str], dict[str, float]],
    experts: list[str],
    output_path: Path,
) -> None:
    x_extent = [x_edges[0] / 1e9, x_edges[-1] / 1e9]
    y_extent = [y_edges[0], y_edges[-1]]
    density_vmax = max(float(np.percentile(np.log1p(histograms[expert]), 99.6)) for expert in experts)
    utility_values = [
        math.asinh(float(stats["signed_effect_mean"]) / scale)
        for key, stats in activation.items()
        if key[0] == calibration
    ]
    utility_bound = max(1.0, float(np.nanpercentile(np.abs(utility_values), 97))) if utility_values else 1.0
    utility_bound = min(6.0, utility_bound)
    utility_norm = TwoSlopeNorm(vcenter=0.0, vmin=-utility_bound, vmax=utility_bound)

    height_ratios = []
    for _expert in experts:
        height_ratios.extend([0.28, 2.25])
    fig, axes = plt.subplots(
        len(experts) * 2,
        1,
        figsize=(18, 2.75 * len(experts)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )
    axes = list(np.asarray(axes).reshape(-1))
    density_image = None
    utility_image = None

    for idx, expert in enumerate(experts):
        strip_ax = axes[2 * idx]
        field_ax = axes[2 * idx + 1]
        utility_strip = utility_array_for_x(
            calibration=calibration,
            expert=expert,
            blocks=blocks,
            x_edges=x_edges,
            activation=activation,
            scale=scale,
        )

        utility_image = strip_ax.imshow(
            utility_strip[np.newaxis, :],
            aspect="auto",
            origin="lower",
            extent=[x_extent[0], x_extent[1], 0, 1],
            cmap="RdYlGn",
            norm=utility_norm,
        )
        strip_ax.set_yticks([])
        strip_ax.set_ylabel(f"{expert}\nutility", rotation=0, labelpad=42, va="center", fontsize=9)
        strip_ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)

        density = np.log1p(histograms[expert]).T
        density_image = field_ax.imshow(
            density,
            aspect="auto",
            origin="lower",
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
            cmap="Greys",
            vmin=0.0,
            vmax=density_vmax,
        )
        field_ax.imshow(
            utility_strip[np.newaxis, :],
            aspect="auto",
            origin="lower",
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
            cmap="RdYlGn",
            norm=utility_norm,
            alpha=0.18,
        )
        field_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        field_ax.set_ylabel(f"{expert}\nDelta theta", rotation=0, labelpad=48, va="center")
        field_ax.set_ylim(y_extent[0], y_extent[1])
        field_ax.grid(axis="x", alpha=0.05)

        for boundary in layer_boundaries:
            x = boundary / 1e9
            strip_ax.axvline(x, color="black", linewidth=0.25, alpha=0.15)
            field_ax.axvline(x, color="black", linewidth=0.25, alpha=0.15)

    axes[-1].set_xlabel("global OP-VEC parameter offset (billions)")
    fig.suptitle(
        f"Signed task-vector field with activation utility overlay: {calibration}\n"
        f"utility color = asinh(signed_effect_mean / median |signed_effect_mean|), scale={scale:.3e}",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.035, 0.02, 0.86, 0.94))
    if density_image is not None:
        density_cax = fig.add_axes([0.885, 0.18, 0.012, 0.64])
        fig.colorbar(density_image, cax=density_cax, label="log(1 + sampled count)")
    if utility_image is not None:
        utility_cax = fig.add_axes([0.935, 0.18, 0.012, 0.64])
        fig.colorbar(utility_image, cax=utility_cax, label="activation utility")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def utility_array_for_x(
    *,
    calibration: str,
    expert: str,
    blocks: list[dict[str, Any]],
    x_edges: np.ndarray,
    activation: dict[tuple[str, str, int, str], dict[str, float]],
    scale: float,
) -> np.ndarray:
    centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    block_idx = 0
    values = np.zeros(len(centers), dtype=np.float64)
    for x_idx, center in enumerate(centers):
        while block_idx + 1 < len(blocks) and center >= int(blocks[block_idx]["end"]):
            block_idx += 1
        block = blocks[block_idx]
        stats = activation.get((calibration, expert, int(block["layer"]), str(block["module"])))
        if stats is None:
            values[x_idx] = np.nan
        else:
            values[x_idx] = math.asinh(float(stats["signed_effect_mean"]) / scale)
    return values


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


def layer_from_param(param_name: str) -> int:
    match = re.search(r"model\.layers\.(\d+)\.", param_name)
    return int(match.group(1)) if match else -1


def ordered_values(values: Any, preferred: list[str]) -> list[str]:
    present = {str(value) for value in values}
    ordered = [item for item in preferred if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_report(path: Path, output_dir: Path, calibrations: list[str], experts: list[str]) -> None:
    lines = [
        "# Signed Energy Field With Activation Utility",
        "",
        "These plots overlay cached activation utility on the original flattened signed task-vector field.",
        "",
        "How to read:",
        "- x axis: global OP-VEC parameter position, same ordering as the signed flattened field",
        "- y axis: signed Delta theta value from sampled task-vector parameters",
        "- grayscale density: sampled parameter count at that signed value and position",
        "- top strip and translucent tint: activation signed_effect_mean for the corresponding layer/module block",
        "- green utility: locally positive first-order effect on the calibration trajectory",
        "- red utility: locally negative first-order effect on the calibration trajectory",
        "",
        "Generated calibrations:",
    ]
    for calibration in calibrations:
        lines.append(f"- `{calibration}`: `01_{calibration}_signed_field_with_utility.png`")
    lines.extend(
        [
            "",
            f"Experts: {', '.join(experts)}",
            "",
            "Important caveat: the utility color is block-level activation utility; the density field is sampled parameter-level signed residual mass.",
            "The overlay shows where local utility lands on the dense signed field, not a final merge gate.",
            "",
            f"Output directory: `{output_dir}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
