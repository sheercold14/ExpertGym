#!/usr/bin/env python3
"""Plot elementwise task-vector value distributions.

This complements block-level norm/cosine plots by sampling individual parameter
positions from OP-VEC expert residual tensors and visualizing their magnitude
density by expert, module family, layer and selected matrix coordinates.
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
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/task_vector_value_distributions/opvec4_20260522"
EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
FAMILY_ORDER = ["attention", "mlp"]
LOG_BINS = np.linspace(-12.0, -0.5, 180)
EPS = 1e-12


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    experts = expert_names(manifest)
    entries = manifest["basis_entries"]
    total_numel_by_expert = total_numel(entries, experts)

    histograms = defaultdict(lambda: np.zeros(len(LOG_BINS) - 1, dtype=np.float64))
    block_rows: list[dict[str, Any]] = []
    position_panels: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    selected_params = selected_position_params()

    for item in entries:
        expert = str(item["expert"])
        if expert not in experts:
            continue
        param_name = str(item["param_name"])
        path = Path(str(item["storage_path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        tensor = torch.load(path, map_location="cpu").float()
        numel = int(tensor.numel())
        n = sample_count(
            numel=numel,
            total_numel=total_numel_by_expert[expert],
            total_samples=int(args.total_samples_per_expert),
            min_samples=int(args.min_samples_per_tensor),
            max_samples=int(args.max_samples_per_tensor),
        )
        sample = sample_tensor(tensor, n=n, seed=stable_seed(expert + "::" + param_name)).numpy()
        abs_sample = np.abs(sample)
        log_abs = np.log10(abs_sample + EPS)
        family = module_family(param_name)
        module = module_from_param(param_name)
        layer = layer_from_param(param_name)

        for key in [
            ("global", expert),
            ("family", expert, family),
            ("module", expert, module),
        ]:
            histograms[key] += np.histogram(log_abs, bins=LOG_BINS)[0]

        block_rows.append(
            {
                "expert": expert,
                "param_name": param_name,
                "layer": layer,
                "module": module,
                "module_family": family,
                "numel": numel,
                "samples": len(sample),
                "mean_abs": float(abs_sample.mean()),
                "rms": float(np.sqrt(np.mean(sample * sample))),
                "p50_log10_abs": float(np.quantile(log_abs, 0.50)),
                "p90_log10_abs": float(np.quantile(log_abs, 0.90)),
                "p99_log10_abs": float(np.quantile(log_abs, 0.99)),
                "max_abs_sample": float(abs_sample.max()),
                "zero_like_frac": float(np.mean(abs_sample <= EPS)),
            }
        )

        if param_name in selected_params:
            position_panels[param_name][expert] = sampled_position_grid(tensor, side=int(args.position_grid_side))
        del tensor

    write_csv(output_dir / "task_vector_value_block_stats.csv", block_rows)
    write_json(output_dir / "task_vector_histograms.json", histogram_payload(histograms))
    plot_global_density(histograms, experts, output_dir / "01_elementwise_abs_density_by_expert.png")
    plot_family_density(histograms, experts, output_dir / "02_elementwise_abs_density_by_family.png")
    plot_module_density(histograms, experts, output_dir / "03_elementwise_abs_density_by_module.png")
    plot_block_quantile_heatmaps(block_rows, experts, output_dir / "04_block_p99_logabs_heatmaps.png")
    plot_position_panels(position_panels, experts, output_dir / "05_selected_parameter_position_heatmaps.png")
    write_report(output_dir / "README.md", manifest_path, output_dir, block_rows, histograms, experts)
    print(f"Wrote elementwise value distribution plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total-samples-per-expert", type=int, default=2_000_000)
    parser.add_argument("--min-samples-per-tensor", type=int, default=2048)
    parser.add_argument("--max-samples-per-tensor", type=int, default=50_000)
    parser.add_argument("--position-grid-side", type=int, default=96)
    return parser.parse_args()


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


def sample_tensor(tensor: torch.Tensor, *, n: int, seed: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if n >= flat.numel():
        return flat.detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(flat.numel(), (n,), generator=generator)
    return flat.index_select(0, indices).detach().cpu()


def sampled_position_grid(tensor: torch.Tensor, *, side: int) -> np.ndarray:
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D tensor, got {tuple(tensor.shape)}")
    rows = min(side, tensor.shape[0])
    cols = min(side, tensor.shape[1])
    row_idx = torch.linspace(0, tensor.shape[0] - 1, rows).round().long()
    col_idx = torch.linspace(0, tensor.shape[1] - 1, cols).round().long()
    grid = tensor.index_select(0, row_idx).index_select(1, col_idx).abs()
    return torch.log10(grid + EPS).numpy()


def plot_global_density(histograms: dict[Any, np.ndarray], experts: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = bin_centers()
    for expert in experts:
        density = normalize_density(histograms[("global", expert)])
        ax.plot(centers, density, label=expert, linewidth=2)
    ax.set_title("Elementwise residual magnitude density")
    ax.set_xlabel("log10(|Delta theta|)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_family_density(histograms: dict[Any, np.ndarray], experts: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(experts), figsize=(5.2 * len(experts), 4.6), sharey=True)
    if len(experts) == 1:
        axes = [axes]
    centers = bin_centers()
    for ax, expert in zip(axes, experts):
        for family in FAMILY_ORDER:
            density = normalize_density(histograms[("family", expert, family)])
            ax.plot(centers, density, label=family, linewidth=2)
        ax.set_title(expert)
        ax.set_xlabel("log10(|Delta theta|)")
        ax.axvline(-6, color="black", linewidth=0.7, alpha=0.4)
    axes[0].set_ylabel("density")
    axes[-1].legend()
    fig.suptitle("Elementwise magnitude density by module family", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_module_density(histograms: dict[Any, np.ndarray], experts: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(len(experts), 1, figsize=(10, 3.8 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    centers = bin_centers()
    for ax, expert in zip(axes, experts):
        for module in MODULE_ORDER:
            density = normalize_density(histograms[("module", expert, module)])
            ax.plot(centers, density, label=module, linewidth=1.4)
        ax.set_title(expert)
        ax.set_ylabel("density")
    axes[-1].set_xlabel("log10(|Delta theta|)")
    axes[0].legend(ncol=7, fontsize=8)
    fig.suptitle("Elementwise magnitude density by module", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_block_quantile_heatmaps(rows: list[dict[str, Any]], experts: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(len(experts), 1, figsize=(12, 3.4 * len(experts)), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    values = [float(row["p99_log10_abs"]) for row in rows]
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))
    for ax, expert in zip(axes, experts):
        mat = np.full((28, len(MODULE_ORDER)), np.nan, dtype=float)
        for row in rows:
            if row["expert"] != expert:
                continue
            mat[int(row["layer"]), MODULE_ORDER.index(str(row["module"]))] = float(row["p99_log10_abs"])
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{expert}: p99 log10(|Delta theta|)")
        ax.set_ylabel("layer")
        ax.set_yticks(range(0, 28, 3))
    axes[-1].set_xticks(range(len(MODULE_ORDER)))
    axes[-1].set_xticklabels(MODULE_ORDER)
    fig.colorbar(im, ax=axes, shrink=0.86)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_position_panels(position_panels: dict[str, dict[str, np.ndarray]], experts: list[str], output_path: Path) -> None:
    params = list(position_panels)
    if not params:
        return
    fig, axes = plt.subplots(len(params), len(experts), figsize=(4.0 * len(experts), 3.2 * len(params)))
    if len(params) == 1:
        axes = np.array([axes])
    if len(experts) == 1:
        axes = axes[:, None]
    all_values = np.concatenate([grid.reshape(-1) for panels in position_panels.values() for grid in panels.values()])
    vmin = float(np.nanpercentile(all_values, 3))
    vmax = float(np.nanpercentile(all_values, 97))
    for row_idx, param_name in enumerate(params):
        for col_idx, expert in enumerate(experts):
            ax = axes[row_idx, col_idx]
            grid = position_panels[param_name].get(expert)
            if grid is None:
                ax.axis("off")
                continue
            im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(expert)
            if col_idx == 0:
                ax.set_ylabel(short_param_name(param_name), fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.84, label="log10(|Delta theta|)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    manifest_path: Path,
    output_dir: Path,
    block_rows: list[dict[str, Any]],
    histograms: dict[Any, np.ndarray],
    experts: list[str],
) -> None:
    lines = [
        "# OP-VEC Elementwise Value Distributions",
        "",
        f"- manifest: `{manifest_path}`",
        f"- output dir: `{output_dir}`",
        "",
        "## Figures",
        "",
        "- `01_elementwise_abs_density_by_expert.png`: global elementwise magnitude density.",
        "- `02_elementwise_abs_density_by_family.png`: attention vs MLP density per expert.",
        "- `03_elementwise_abs_density_by_module.png`: q/k/v/o/gate/up/down density per expert.",
        "- `04_block_p99_logabs_heatmaps.png`: per-block p99 elementwise magnitude.",
        "- `05_selected_parameter_position_heatmaps.png`: sampled matrix-coordinate magnitude maps.",
        "",
        "## Global Quantiles",
        "",
        "| expert | sample p50 | sample p90 | sample p99 | density mass > 1e-5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for expert in experts:
        selected = [row for row in block_rows if row["expert"] == expert]
        weights = np.array([float(row["samples"]) for row in selected], dtype=float)
        p50 = weighted_mean([float(row["p50_log10_abs"]) for row in selected], weights)
        p90 = weighted_mean([float(row["p90_log10_abs"]) for row in selected], weights)
        p99 = weighted_mean([float(row["p99_log10_abs"]) for row in selected], weights)
        mass = tail_mass(histograms[("global", expert)], threshold_log10=-5.0)
        lines.append(f"| {expert} | {p50:.3f} | {p90:.3f} | {p99:.3f} | {mass:.4f} |")
    lines.extend(["", "## Strongest Blocks by p99 Magnitude", "", "| rank | expert | layer | module | family | p99 log10 abs | p90 log10 abs | mean abs |", "|---:|---|---:|---|---|---:|---:|---:|"])
    top = sorted(block_rows, key=lambda row: float(row["p99_log10_abs"]), reverse=True)[:16]
    for idx, row in enumerate(top, 1):
        lines.append(
            f"| {idx} | {row['expert']} | {row['layer']} | {row['module']} | {row['module_family']} | "
            f"{float(row['p99_log10_abs']):.3f} | {float(row['p90_log10_abs']):.3f} | {float(row['mean_abs']):.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def histogram_payload(histograms: dict[Any, np.ndarray]) -> dict[str, Any]:
    return {
        "bin_edges_log10_abs": LOG_BINS.tolist(),
        "histograms": {"::".join(map(str, key)): value.tolist() for key, value in histograms.items()},
    }


def normalize_density(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return counts
    width = np.diff(LOG_BINS)
    return counts / total / width


def bin_centers() -> np.ndarray:
    return (LOG_BINS[:-1] + LOG_BINS[1:]) / 2.0


def tail_mass(counts: np.ndarray, *, threshold_log10: float) -> float:
    centers = bin_centers()
    total = counts.sum()
    if total <= 0:
        return 0.0
    return float(counts[centers > threshold_log10].sum() / total)


def weighted_mean(values: list[float], weights: np.ndarray) -> float:
    values_arr = np.array(values, dtype=float)
    return float((values_arr * weights).sum() / max(weights.sum(), 1.0))


def selected_position_params() -> set[str]:
    return {
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.27.self_attn.v_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
    }


def total_numel(entries: list[dict[str, Any]], experts: list[str]) -> dict[str, int]:
    output = {expert: 0 for expert in experts}
    for item in entries:
        expert = str(item["expert"])
        if expert not in output:
            continue
        numel = 1
        for dim in item["shape"]:
            numel *= int(dim)
        output[expert] += numel
    return output


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


def module_family(param_name: str) -> str:
    if ".self_attn." in param_name:
        return "attention"
    if ".mlp." in param_name:
        return "mlp"
    return "other"


def short_param_name(param_name: str) -> str:
    layer = layer_from_param(param_name)
    module = module_from_param(param_name)
    return f"L{layer} {module}"


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
