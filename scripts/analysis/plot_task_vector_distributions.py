#!/usr/bin/env python3
"""Plot OP-VEC task-vector distribution and conflict diagnostics.

This script is intentionally diagnostic-only: it reads expert residual tensors
from an OP-VEC mode manifest and writes static figures plus per-block metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/task_vector_visuals/opvec4_20260522"
EXPERT_ORDER = ["tool", "memory", "code"]
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
PAIR_ORDER = [("tool", "memory"), ("tool", "code"), ("memory", "code")]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    entries = build_entry_index(manifest, manifest_path.parent)
    experts = [expert for expert in EXPERT_ORDER if expert in manifest.get("experts", {})]
    experts.extend(expert for expert in manifest.get("experts", {}) if expert not in experts)

    rows = compute_block_metrics(entries, experts)
    write_csv(output_dir / "task_vector_block_metrics.csv", rows)
    write_json(output_dir / "task_vector_block_metrics.json", rows)

    plot_energy_heatmaps(rows, experts, output_dir / "01_energy_heatmaps.png")
    plot_cosine_heatmaps(rows, output_dir / "02_pairwise_cosine_heatmaps.png")
    plot_merge_geometry(rows, output_dir / "03_merge_geometry_heatmaps.png")
    plot_module_family_summary(rows, experts, output_dir / "04_module_family_summary.png")
    plot_conflict_scatter(rows, output_dir / "05_conflict_scatter.png")
    plot_sketched_spectra(
        rows=rows,
        entries=entries,
        experts=experts,
        output_path=output_dir / "06_sketched_spectra_selected_blocks.png",
        max_blocks=int(args.spectrum_blocks),
        sample_size=int(args.spectrum_sample_size),
    )
    write_report(output_dir / "README.md", manifest_path, output_dir, rows, experts)
    print(f"Wrote task-vector visualizations to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spectrum-blocks", type=int, default=4)
    parser.add_argument("--spectrum-sample-size", type=int, default=512)
    return parser.parse_args()


def build_entry_index(manifest: dict[str, Any], root: Path) -> dict[str, dict[str, Path]]:
    entries: dict[str, dict[str, Path]] = defaultdict(dict)
    for item in manifest["basis_entries"]:
        param_name = str(item["param_name"])
        expert = str(item["expert"])
        storage_path = Path(str(item["storage_path"]))
        if not storage_path.is_absolute():
            storage_path = root / storage_path
        entries[param_name][expert] = storage_path
    return dict(entries)


def compute_block_metrics(entries: dict[str, dict[str, Path]], experts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for param_name in sorted(entries, key=param_sort_key):
        tensors = {}
        for expert in experts:
            tensors[expert] = load_tensor(entries[param_name][expert])
        flat = {expert: tensors[expert].reshape(-1).float() for expert in experts}
        norms = {expert: float(torch.linalg.vector_norm(flat[expert]).item()) for expert in experts}
        dots: dict[tuple[str, str], float] = {}
        cosines: dict[tuple[str, str], float] = {}
        for a, b in PAIR_ORDER:
            dot = float(torch.dot(flat[a], flat[b]).item())
            dots[(a, b)] = dot
            denom = norms[a] * norms[b]
            cosines[(a, b)] = dot / denom if denom > 0 else 0.0

        sum_sq = sum(value * value for value in norms.values())
        for a, b in PAIR_ORDER:
            sum_sq += 2.0 * dots[(a, b)]
        sum_norm = math.sqrt(max(sum_sq, 0.0))
        norm_sum = sum(norms.values())
        rss_norm = math.sqrt(sum(value * value for value in norms.values()))
        min_cos = min(cosines.values())
        max_cos = max(cosines.values())
        mean_cos = sum(cosines.values()) / len(cosines)
        cancellation = 1.0 - (sum_norm / norm_sum if norm_sum > 0 else 0.0)
        amplification = sum_norm / rss_norm if rss_norm > 0 else 0.0

        row = {
            "param_name": param_name,
            "layer": layer_from_param(param_name),
            "module": module_from_param(param_name),
            "module_family": module_family(param_name),
            "sum_norm": sum_norm,
            "norm_sum": norm_sum,
            "rss_norm": rss_norm,
            "union_ratio": sum_norm / norm_sum if norm_sum > 0 else 0.0,
            "rss_ratio": amplification,
            "cancellation_score": cancellation,
            "min_pairwise_cosine": min_cos,
            "max_pairwise_cosine": max_cos,
            "mean_pairwise_cosine": mean_cos,
            "negative_cosine_mass": sum(max(0.0, -value) for value in cosines.values()),
        }
        for expert in experts:
            row[f"{expert}_norm"] = norms[expert]
        for a, b in PAIR_ORDER:
            row[f"cos_{a}_{b}"] = cosines[(a, b)]
            row[f"dot_{a}_{b}"] = dots[(a, b)]
        rows.append(row)
        del tensors, flat
        gc.collect()
    return rows


def plot_energy_heatmaps(rows: list[dict[str, Any]], experts: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(len(experts), 1, figsize=(13, 9), sharex=True, sharey=True)
    if len(experts) == 1:
        axes = [axes]
    vmax = max(math.log10(float(row[f"{expert}_norm"]) + 1e-12) for row in rows for expert in experts)
    vmin = min(math.log10(float(row[f"{expert}_norm"]) + 1e-12) for row in rows for expert in experts)
    for ax, expert in zip(axes, experts):
        mat = matrix_for(rows, f"{expert}_norm", transform=lambda x: math.log10(float(x) + 1e-12))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{expert} residual energy: log10(||Delta W||)")
        ax.set_ylabel("layer")
        ax.set_yticks(range(0, 28, 3))
    axes[-1].set_xticks(range(len(MODULE_ORDER)))
    axes[-1].set_xticklabels(MODULE_ORDER)
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cosine_heatmaps(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, (a, b) in zip(axes, PAIR_ORDER):
        mat = matrix_for(rows, f"cos_{a}_{b}")
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-0.3, vmax=0.3)
        ax.set_title(f"cosine({a}, {b})")
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels(MODULE_ORDER)
        ax.set_yticks(range(0, 28, 3))
        ax.set_ylabel("layer")
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_merge_geometry(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    configs = [
        ("union_ratio", "||sum Delta|| / sum ||Delta||", "magma_r", 0.0, 1.0),
        ("cancellation_score", "cancellation score", "magma", 0.0, 1.0),
        ("negative_cosine_mass", "negative cosine mass", "Reds", 0.0, None),
    ]
    for ax, (key, title, cmap, vmin, vmax) in zip(axes, configs):
        mat = matrix_for(rows, key)
        if vmax is None:
            vmax = float(np.nanpercentile(mat, 95))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels(MODULE_ORDER)
        ax.set_yticks(range(0, 28, 3))
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_module_family_summary(rows: list[dict[str, Any]], experts: list[str], output_path: Path) -> None:
    families = ["attention", "mlp"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    width = 0.25
    x = np.arange(len(families))
    for idx, expert in enumerate(experts):
        values = []
        for family in families:
            selected = [row for row in rows if row["module_family"] == family]
            values.append(sum(float(row[f"{expert}_norm"]) ** 2 for row in selected))
        total = sum(values) or 1.0
        axes[0].bar(x + (idx - 1) * width, [value / total for value in values], width, label=expert)
    axes[0].set_title("Energy fraction by module family")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    for key, title, ax in [
        ("min_pairwise_cosine", "Mean min pairwise cosine", axes[1]),
        ("cancellation_score", "Mean cancellation score", axes[2]),
    ]:
        values = []
        for family in families:
            selected = [row for row in rows if row["module_family"] == family]
            values.append(float(np.mean([float(row[key]) for row in selected])))
        ax.bar(families, values, color=["#4C78A8", "#F58518"])
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_conflict_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {"attention": "#4C78A8", "mlp": "#F58518"}
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for family in ["attention", "mlp"]:
        selected = [row for row in rows if row["module_family"] == family]
        ax.scatter(
            [float(row["norm_sum"]) for row in selected],
            [float(row["min_pairwise_cosine"]) for row in selected],
            s=[20 + 120 * float(row["cancellation_score"]) for row in selected],
            alpha=0.7,
            label=family,
            color=colors[family],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("sum of expert residual norms")
    ax.set_ylabel("minimum pairwise cosine")
    ax.set_title("Block-level conflict map; marker size = cancellation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sketched_spectra(
    *,
    rows: list[dict[str, Any]],
    entries: dict[str, dict[str, Path]],
    experts: list[str],
    output_path: Path,
    max_blocks: int,
    sample_size: int,
) -> None:
    selected = select_spectrum_blocks(rows, max_blocks)
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3.2 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, row in zip(axes, selected):
        param_name = str(row["param_name"])
        submatrices = {}
        for expert in experts:
            tensor = load_tensor(entries[param_name][expert]).float()
            submatrices[expert] = deterministic_submatrix(tensor, sample_size=sample_size)
            del tensor
        submatrices["sum"] = sum(submatrices[expert] for expert in experts)
        for name, matrix in submatrices.items():
            values = torch.linalg.svdvals(matrix).detach().cpu().numpy()
            values = values[: min(80, len(values))]
            if values[0] > 0:
                values = values / values[0]
            ax.plot(np.arange(1, len(values) + 1), values, label=name)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.set_title(
            f"sketched singular spectrum: L{row['layer']} {row['module']} "
            f"({row['module_family']}), min cos={float(row['min_pairwise_cosine']):.3f}"
        )
        ax.set_ylabel("normalized singular value")
        ax.legend(ncol=4, fontsize=8)
        del submatrices
        gc.collect()
    axes[-1].set_xlabel("rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_spectrum_blocks(rows: list[dict[str, Any]], max_blocks: int) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    for family in ["attention", "mlp"]:
        family_rows = [row for row in rows if row["module_family"] == family]
        family_rows.sort(key=lambda row: (float(row["negative_cosine_mass"]), float(row["norm_sum"])), reverse=True)
        picks.extend(family_rows[: max(1, max_blocks // 2)])
    seen = set()
    output = []
    for row in picks:
        if row["param_name"] in seen:
            continue
        seen.add(row["param_name"])
        output.append(row)
        if len(output) >= max_blocks:
            break
    return output


def deterministic_submatrix(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    rows = min(sample_size, tensor.shape[0])
    cols = min(sample_size, tensor.shape[1])
    row_idx = torch.linspace(0, tensor.shape[0] - 1, rows).round().long()
    col_idx = torch.linspace(0, tensor.shape[1] - 1, cols).round().long()
    return tensor.index_select(0, row_idx).index_select(1, col_idx).contiguous()


def matrix_for(rows: list[dict[str, Any]], key: str, transform=None) -> np.ndarray:
    mat = np.full((28, len(MODULE_ORDER)), np.nan, dtype=float)
    for row in rows:
        layer = int(row["layer"])
        module = str(row["module"])
        if 0 <= layer < 28 and module in MODULE_ORDER:
            value = row[key]
            if transform:
                value = transform(value)
            mat[layer, MODULE_ORDER.index(module)] = float(value)
    return mat


def write_report(
    path: Path,
    manifest_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    experts: list[str],
) -> None:
    family_summary = {}
    for family in ["attention", "mlp"]:
        selected = [row for row in rows if row["module_family"] == family]
        family_summary[family] = {
            "rows": len(selected),
            "mean_min_cos": float(np.mean([float(row["min_pairwise_cosine"]) for row in selected])),
            "mean_cancellation": float(np.mean([float(row["cancellation_score"]) for row in selected])),
            "mean_negative_cosine_mass": float(np.mean([float(row["negative_cosine_mass"]) for row in selected])),
        }
    top_conflict = sorted(rows, key=lambda row: (float(row["negative_cosine_mass"]), float(row["norm_sum"])), reverse=True)[:12]
    lines = [
        "# OP-VEC Task Vector Distribution Visuals",
        "",
        f"- manifest: `{manifest_path}`",
        f"- output dir: `{output_dir}`",
        f"- experts: `{', '.join(experts)}`",
        "",
        "## Figures",
        "",
        "- `01_energy_heatmaps.png`: per-expert residual energy by layer and module.",
        "- `02_pairwise_cosine_heatmaps.png`: pairwise parameter-space cosine by layer and module.",
        "- `03_merge_geometry_heatmaps.png`: direct-union cancellation/amplification geometry.",
        "- `04_module_family_summary.png`: attention vs MLP aggregate distribution.",
        "- `05_conflict_scatter.png`: block norm vs minimum pairwise cosine.",
        "- `06_sketched_spectra_selected_blocks.png`: sampled singular spectra for high-conflict blocks.",
        "",
        "## Module Family Summary",
        "",
        "| family | rows | mean min cosine | mean cancellation | mean negative cosine mass |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, item in family_summary.items():
        lines.append(
            f"| {family} | {item['rows']} | {item['mean_min_cos']:.4f} | "
            f"{item['mean_cancellation']:.4f} | {item['mean_negative_cosine_mass']:.4f} |"
        )
    lines.extend(["", "## Highest Conflict Blocks", "", "| rank | layer | module | family | min cosine | neg cosine mass | union ratio | norm sum |", "|---:|---:|---|---|---:|---:|---:|---:|"])
    for idx, row in enumerate(top_conflict, 1):
        lines.append(
            f"| {idx} | {row['layer']} | {row['module']} | {row['module_family']} | "
            f"{float(row['min_pairwise_cosine']):.4f} | {float(row['negative_cosine_mass']):.4f} | "
            f"{float(row['union_ratio']):.4f} | {float(row['norm_sum']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def param_sort_key(param_name: str) -> tuple[int, int]:
    return (layer_from_param(param_name), MODULE_ORDER.index(module_from_param(param_name)))


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


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


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
