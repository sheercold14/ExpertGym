#!/usr/bin/env python3
"""Visualize task-vector magnitudes on a flattened 1D parameter axis.

The x-axis is a deterministic global parameter offset over OP-VEC mergeable
linear weights, ordered by layer and module. Because plotting every element of a
7B-scale model is not useful, the script samples positions and renders binned
quantile traces plus density rasters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/task_vector_flattened_positions/opvec4_20260522"
EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
LOG_ABS_BINS = np.linspace(-12.0, -2.5, 220)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    experts = expert_names(manifest)
    blocks = build_blocks(manifest, manifest_path.parent)
    total_numel = sum(block["numel"] for block in blocks)
    assign_offsets(blocks)

    x_bins = int(args.x_bins)
    x_edges = np.linspace(0, total_numel, x_bins + 1)
    histograms = {expert: np.zeros((x_bins, len(LOG_ABS_BINS) - 1), dtype=np.float64) for expert in experts}
    sample_rows: list[dict[str, Any]] = []

    for block in blocks:
        for expert in experts:
            path = block["paths"][expert]
            tensor = torch.load(path, map_location="cpu").float()
            n = sample_count(
                numel=block["numel"],
                total_numel=total_numel,
                total_samples=int(args.total_samples_per_expert),
                min_samples=int(args.min_samples_per_tensor),
                max_samples=int(args.max_samples_per_tensor),
            )
            indices = sample_indices(block["numel"], n=n, seed=stable_seed(expert + "::" + block["param_name"]))
            values = tensor.reshape(-1).index_select(0, indices).abs().numpy()
            global_indices = block["start"] + indices.numpy().astype(np.int64)
            x_idx = np.searchsorted(x_edges, global_indices, side="right") - 1
            x_idx = np.clip(x_idx, 0, x_bins - 1)
            y = np.log10(values + 1e-12)
            y_idx = np.searchsorted(LOG_ABS_BINS, y, side="right") - 1
            valid = (y_idx >= 0) & (y_idx < len(LOG_ABS_BINS) - 1)
            np.add.at(histograms[expert], (x_idx[valid], y_idx[valid]), 1)
            sample_rows.append(
                {
                    "expert": expert,
                    "layer": block["layer"],
                    "module": block["module"],
                    "param_name": block["param_name"],
                    "start": block["start"],
                    "end": block["end"],
                    "numel": block["numel"],
                    "samples": int(valid.sum()),
                    "sample_p50_log10_abs": float(np.quantile(y, 0.50)),
                    "sample_p90_log10_abs": float(np.quantile(y, 0.90)),
                    "sample_p99_log10_abs": float(np.quantile(y, 0.99)),
                }
            )
            del tensor, indices, values, global_indices, x_idx, y, y_idx, valid

    quantiles = {
        expert: quantile_traces(histograms[expert], [0.5, 0.9, 0.99])
        for expert in experts
    }
    write_csv(output_dir / "flattened_position_sample_stats.csv", sample_rows)
    write_json(
        output_dir / "flattened_position_histograms.json",
        {
            "total_numel": total_numel,
            "x_bin_edges": x_edges.tolist(),
            "log_abs_bin_edges": LOG_ABS_BINS.tolist(),
            "experts": experts,
            "histograms": {expert: histograms[expert].astype(int).tolist() for expert in experts},
        },
    )
    plot_quantile_traces(
        quantiles=quantiles,
        x_edges=x_edges,
        blocks=blocks,
        experts=experts,
        output_path=output_dir / "01_flattened_position_quantile_traces.png",
    )
    plot_density_rasters(
        histograms=histograms,
        x_edges=x_edges,
        experts=experts,
        output_path=output_dir / "02_flattened_position_density_rasters.png",
    )
    plot_expert_overlay(
        quantiles=quantiles,
        x_edges=x_edges,
        experts=experts,
        output_path=output_dir / "03_flattened_position_p99_overlay.png",
    )
    write_report(output_dir / "README.md", manifest_path, output_dir, total_numel, blocks, sample_rows, experts)
    print(f"Wrote flattened position plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-bins", type=int, default=2200)
    parser.add_argument("--total-samples-per-expert", type=int, default=3_000_000)
    parser.add_argument("--min-samples-per-tensor", type=int, default=512)
    parser.add_argument("--max-samples-per-tensor", type=int, default=40_000)
    return parser.parse_args()


def build_blocks(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in manifest["basis_entries"]:
        param_name = str(entry["param_name"])
        expert = str(entry["expert"])
        path = Path(str(entry["storage_path"]))
        if not path.is_absolute():
            path = root / path
        shape = [int(dim) for dim in entry["shape"]]
        numel = math.prod(shape)
        row = grouped.setdefault(
            param_name,
            {
                "param_name": param_name,
                "shape": shape,
                "numel": numel,
                "layer": layer_from_param(param_name),
                "module": module_from_param(param_name),
                "paths": {},
            },
        )
        row["paths"][expert] = path
    return sorted(grouped.values(), key=lambda row: (row["layer"], MODULE_ORDER.index(row["module"])))


def assign_offsets(blocks: list[dict[str, Any]]) -> None:
    offset = 0
    for block in blocks:
        block["start"] = offset
        offset += int(block["numel"])
        block["end"] = offset


def sample_count(
    *,
    numel: int,
    total_numel: int,
    total_samples: int,
    min_samples: int,
    max_samples: int,
) -> int:
    proportional = int(round(total_samples * numel / total_numel))
    return min(numel, max(min_samples, min(max_samples, proportional)))


def sample_indices(numel: int, *, n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(numel, (n,), generator=generator)


def quantile_traces(hist: np.ndarray, qs: list[float]) -> dict[str, np.ndarray]:
    centers = (LOG_ABS_BINS[:-1] + LOG_ABS_BINS[1:]) / 2.0
    output: dict[str, np.ndarray] = {}
    cumsum = np.cumsum(hist, axis=1)
    totals = cumsum[:, -1]
    for q in qs:
        values = np.full(hist.shape[0], np.nan, dtype=float)
        for idx in range(hist.shape[0]):
            if totals[idx] <= 0:
                continue
            target = q * totals[idx]
            bin_idx = int(np.searchsorted(cumsum[idx], target, side="left"))
            bin_idx = min(max(bin_idx, 0), len(centers) - 1)
            values[idx] = centers[bin_idx]
        output[f"p{int(q * 100):02d}"] = values
    return output


def plot_quantile_traces(
    *,
    quantiles: dict[str, dict[str, np.ndarray]],
    x_edges: np.ndarray,
    blocks: list[dict[str, Any]],
    experts: list[str],
    output_path: Path,
) -> None:
    x = (x_edges[:-1] + x_edges[1:]) / 2.0 / 1e9
    fig, axes = plt.subplots(len(experts), 1, figsize=(15, 3.3 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    for ax, expert in zip(axes, experts):
        q = quantiles[expert]
        ax.plot(x, q["p50"], color="#9ecae1", linewidth=0.8, label="p50")
        ax.plot(x, q["p90"], color="#4292c6", linewidth=1.0, label="p90")
        ax.plot(x, q["p99"], color="#084594", linewidth=1.1, label="p99")
        ax.fill_between(x, q["p50"], q["p99"], color="#9ecae1", alpha=0.22)
        add_layer_guides(ax, blocks)
        ax.set_title(f"{expert}: flattened parameter-position magnitude quantiles")
        ax.set_ylabel("log10(|Delta theta|)")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("global OP-VEC parameter offset (billions)")
    axes[0].legend(ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_density_rasters(
    *,
    histograms: dict[str, np.ndarray],
    x_edges: np.ndarray,
    experts: list[str],
    output_path: Path,
) -> None:
    x_extent = [x_edges[0] / 1e9, x_edges[-1] / 1e9, LOG_ABS_BINS[0], LOG_ABS_BINS[-1]]
    fig, axes = plt.subplots(len(experts), 1, figsize=(15, 3.4 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    vmax = max(float(np.percentile(np.log1p(histograms[expert]), 99.5)) for expert in experts)
    for ax, expert in zip(axes, experts):
        image = np.log1p(histograms[expert]).T
        im = ax.imshow(image, aspect="auto", origin="lower", extent=x_extent, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{expert}: density over flattened parameter positions")
        ax.set_ylabel("log10(|Delta theta|)")
    axes[-1].set_xlabel("global OP-VEC parameter offset (billions)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="log(1 + sampled count)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_expert_overlay(
    *,
    quantiles: dict[str, dict[str, np.ndarray]],
    x_edges: np.ndarray,
    experts: list[str],
    output_path: Path,
) -> None:
    colors = {"tool": "#1f77b4", "memory": "#ff7f0e", "code": "#2ca02c"}
    x = (x_edges[:-1] + x_edges[1:]) / 2.0 / 1e9
    fig, ax = plt.subplots(figsize=(15, 4.8))
    for expert in experts:
        ax.plot(x, quantiles[expert]["p99"], label=f"{expert} p99", linewidth=1.1, color=colors.get(expert))
    ax.set_title("Flattened parameter positions: p99 magnitude overlay")
    ax.set_xlabel("global OP-VEC parameter offset (billions)")
    ax.set_ylabel("p99 log10(|Delta theta|)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(experts))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def add_layer_guides(ax: plt.Axes, blocks: list[dict[str, Any]]) -> None:
    seen = set()
    for block in blocks:
        layer = int(block["layer"])
        if layer in seen:
            continue
        seen.add(layer)
        if layer % 4 == 0:
            ax.axvline(block["start"] / 1e9, color="black", linewidth=0.35, alpha=0.2)


def write_report(
    path: Path,
    manifest_path: Path,
    output_dir: Path,
    total_numel: int,
    blocks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    experts: list[str],
) -> None:
    lines = [
        "# Flattened Task-Vector Parameter Position Plots",
        "",
        f"- manifest: `{manifest_path}`",
        f"- output dir: `{output_dir}`",
        f"- flattened OP-VEC parameters: `{total_numel:,}`",
        "",
        "The x-axis follows layer/module order over mergeable OP-VEC linear weights, not embeddings or norm/lm_head parameters.",
        "",
        "## Figures",
        "",
        "- `01_flattened_position_quantile_traces.png`: p50/p90/p99 magnitude traces along the flattened parameter axis.",
        "- `02_flattened_position_density_rasters.png`: x-position vs log-magnitude density raster.",
        "- `03_flattened_position_p99_overlay.png`: p99 overlay across experts.",
        "",
        "## Sample Summary",
        "",
        "| expert | sampled positions | mean p99 log10 abs | strongest block | strongest p99 |",
        "|---|---:|---:|---|---:|",
    ]
    for expert in experts:
        selected = [row for row in rows if row["expert"] == expert]
        samples = sum(int(row["samples"]) for row in selected)
        mean_p99 = float(np.mean([float(row["sample_p99_log10_abs"]) for row in selected]))
        top = max(selected, key=lambda row: float(row["sample_p99_log10_abs"]))
        lines.append(
            f"| {expert} | {samples:,} | {mean_p99:.3f} | "
            f"L{top['layer']} {top['module']} | {float(top['sample_p99_log10_abs']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expert_names(manifest: dict[str, Any]) -> list[str]:
    raw = manifest.get("experts") or {}
    experts = [expert for expert in EXPERT_ORDER if expert in raw]
    experts.extend(str(expert) for expert in raw if str(expert) not in experts)
    return experts


def layer_from_param(param_name: str) -> int:
    match = re.search(r"model\.layers\.(\d+)\.", param_name)
    return int(match.group(1)) if match else -1


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


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
