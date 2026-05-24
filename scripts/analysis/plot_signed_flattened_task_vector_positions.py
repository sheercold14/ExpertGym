#!/usr/bin/env python3
"""Plot signed task-vector values on a flattened 1D parameter axis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/task_vector_signed_flattened_positions/opvec4_20260522"
EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
SIGNED_BINS = np.linspace(-4.0e-4, 4.0e-4, 241)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    experts = expert_names(manifest)
    blocks = build_blocks(manifest, manifest_path.parent)
    assign_offsets(blocks)
    total_numel = sum(int(block["numel"]) for block in blocks)
    x_bins = int(args.x_bins)
    x_edges = np.linspace(0, total_numel, x_bins + 1)

    histograms = {expert: np.zeros((x_bins, len(SIGNED_BINS) - 1), dtype=np.float64) for expert in experts}
    sign_rows: list[dict[str, Any]] = []
    for block in blocks:
        for expert in experts:
            tensor = torch.load(block["paths"][expert], map_location="cpu").float()
            n = sample_count(
                numel=int(block["numel"]),
                total_numel=total_numel,
                total_samples=int(args.total_samples_per_expert),
                min_samples=int(args.min_samples_per_tensor),
                max_samples=int(args.max_samples_per_tensor),
            )
            indices = sample_indices(int(block["numel"]), n=n, seed=stable_seed(expert + "::" + block["param_name"]))
            values = tensor.reshape(-1).index_select(0, indices).numpy()
            global_indices = int(block["start"]) + indices.numpy().astype(np.int64)
            x_idx = np.searchsorted(x_edges, global_indices, side="right") - 1
            x_idx = np.clip(x_idx, 0, x_bins - 1)
            y_idx = np.searchsorted(SIGNED_BINS, values, side="right") - 1
            valid = (y_idx >= 0) & (y_idx < len(SIGNED_BINS) - 1)
            np.add.at(histograms[expert], (x_idx[valid], y_idx[valid]), 1)
            sign_rows.append(
                {
                    "expert": expert,
                    "layer": block["layer"],
                    "module": block["module"],
                    "param_name": block["param_name"],
                    "start": block["start"],
                    "end": block["end"],
                    "numel": block["numel"],
                    "samples": int(len(values)),
                    "mean": float(np.mean(values)),
                    "p01": float(np.quantile(values, 0.01)),
                    "p10": float(np.quantile(values, 0.10)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                    "p99": float(np.quantile(values, 0.99)),
                    "positive_frac": float(np.mean(values > 0)),
                    "negative_frac": float(np.mean(values < 0)),
                    "near_zero_frac": float(np.mean(np.abs(values) < 1.0e-8)),
                }
            )
            del tensor, indices, values, global_indices, x_idx, y_idx, valid

    quantiles = {
        expert: quantile_traces(histograms[expert], [0.01, 0.10, 0.50, 0.90, 0.99])
        for expert in experts
    }
    write_csv(output_dir / "signed_flattened_position_sample_stats.csv", sign_rows)
    write_json(
        output_dir / "signed_flattened_position_histograms.json",
        {
            "total_numel": total_numel,
            "x_bin_edges": x_edges.tolist(),
            "signed_value_bin_edges": SIGNED_BINS.tolist(),
            "experts": experts,
            "histograms": {expert: histograms[expert].astype(int).tolist() for expert in experts},
        },
    )
    plot_signed_density_rasters(
        histograms=histograms,
        x_edges=x_edges,
        experts=experts,
        output_path=output_dir / "01_signed_value_density_rasters.png",
    )
    plot_signed_quantile_traces(
        quantiles=quantiles,
        x_edges=x_edges,
        experts=experts,
        output_path=output_dir / "02_signed_quantile_traces.png",
    )
    plot_positive_fraction(
        rows=sign_rows,
        output_path=output_dir / "03_positive_fraction_by_layer_module.png",
        experts=experts,
    )
    write_report(output_dir / "README.md", manifest_path, output_dir, total_numel, sign_rows, experts)
    print(f"Wrote signed flattened position plots to {output_dir}")


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
        row = grouped.setdefault(
            param_name,
            {
                "param_name": param_name,
                "layer": layer_from_param(param_name),
                "module": module_from_param(param_name),
                "shape": shape,
                "numel": math.prod(shape),
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
    centers = (SIGNED_BINS[:-1] + SIGNED_BINS[1:]) / 2.0
    cumsum = np.cumsum(hist, axis=1)
    totals = cumsum[:, -1]
    output: dict[str, np.ndarray] = {}
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


def plot_signed_density_rasters(
    *,
    histograms: dict[str, np.ndarray],
    x_edges: np.ndarray,
    experts: list[str],
    output_path: Path,
) -> None:
    x_extent = [x_edges[0] / 1e9, x_edges[-1] / 1e9, SIGNED_BINS[0], SIGNED_BINS[-1]]
    fig, axes = plt.subplots(len(experts), 1, figsize=(15, 3.4 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    vmax = max(float(np.percentile(np.log1p(histograms[expert]), 99.5)) for expert in experts)
    for ax, expert in zip(axes, experts):
        im = ax.imshow(
            np.log1p(histograms[expert]).T,
            aspect="auto",
            origin="lower",
            extent=x_extent,
            cmap="coolwarm",
            vmin=0,
            vmax=vmax,
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{expert}: signed residual density over flattened positions")
        ax.set_ylabel("Delta theta")
    axes[-1].set_xlabel("global OP-VEC parameter offset (billions)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="log(1 + sampled count)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_signed_quantile_traces(
    *,
    quantiles: dict[str, dict[str, np.ndarray]],
    x_edges: np.ndarray,
    experts: list[str],
    output_path: Path,
) -> None:
    x = (x_edges[:-1] + x_edges[1:]) / 2.0 / 1e9
    fig, axes = plt.subplots(len(experts), 1, figsize=(15, 3.3 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    for ax, expert in zip(axes, experts):
        q = quantiles[expert]
        ax.fill_between(x, q["p01"], q["p99"], color="#9ecae1", alpha=0.25, label="p01-p99")
        ax.fill_between(x, q["p10"], q["p90"], color="#4292c6", alpha=0.22, label="p10-p90")
        ax.plot(x, q["p50"], color="#084594", linewidth=0.9, label="p50")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{expert}: signed residual quantiles")
        ax.set_ylabel("Delta theta")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("global OP-VEC parameter offset (billions)")
    axes[0].legend(ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_positive_fraction(rows: list[dict[str, Any]], output_path: Path, experts: list[str]) -> None:
    fig, axes = plt.subplots(len(experts), 1, figsize=(12, 3.2 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    for ax, expert in zip(axes, experts):
        mat = np.full((28, len(MODULE_ORDER)), np.nan, dtype=float)
        for row in rows:
            if row["expert"] != expert:
                continue
            mat[int(row["layer"]), MODULE_ORDER.index(str(row["module"]))] = float(row["positive_frac"])
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=0.35, vmax=0.65)
        ax.set_title(f"{expert}: positive-value fraction")
        ax.set_ylabel("layer")
        ax.set_yticks(range(0, 28, 3))
    axes[-1].set_xticks(range(len(MODULE_ORDER)))
    axes[-1].set_xticklabels(MODULE_ORDER)
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    manifest_path: Path,
    output_dir: Path,
    total_numel: int,
    rows: list[dict[str, Any]],
    experts: list[str],
) -> None:
    lines = [
        "# Signed Flattened Task-Vector Position Plots",
        "",
        f"- manifest: `{manifest_path}`",
        f"- output dir: `{output_dir}`",
        f"- flattened OP-VEC parameters: `{total_numel:,}`",
        "",
        "## Figures",
        "",
        "- `01_signed_value_density_rasters.png`: signed value density over flattened positions.",
        "- `02_signed_quantile_traces.png`: signed p01/p10/p50/p90/p99 traces.",
        "- `03_positive_fraction_by_layer_module.png`: positive sign fraction per layer/module.",
        "",
        "## Global Sign Summary",
        "",
        "| expert | positive frac | negative frac | near-zero frac | mean signed value |",
        "|---|---:|---:|---:|---:|",
    ]
    for expert in experts:
        selected = [row for row in rows if row["expert"] == expert]
        weights = np.array([float(row["samples"]) for row in selected], dtype=float)
        pos = weighted_mean([float(row["positive_frac"]) for row in selected], weights)
        neg = weighted_mean([float(row["negative_frac"]) for row in selected], weights)
        near = weighted_mean([float(row["near_zero_frac"]) for row in selected], weights)
        mean = weighted_mean([float(row["mean"]) for row in selected], weights)
        lines.append(f"| {expert} | {pos:.4f} | {neg:.4f} | {near:.4f} | {mean:.3e} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def weighted_mean(values: list[float], weights: np.ndarray) -> float:
    arr = np.array(values, dtype=float)
    return float((arr * weights).sum() / max(weights.sum(), 1.0))


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
