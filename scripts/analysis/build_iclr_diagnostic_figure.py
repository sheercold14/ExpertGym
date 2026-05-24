#!/usr/bin/env python3
"""Build the main residual-diagnostic figure for the ICLR ExpertGym draft.

The figure is artifact-only and uses the existing RCRF analysis outputs:

1. Code source/span pair conflicts.
2. Residual role atlas counts.
3. RCF-BC operating-point trade-off rows.

It writes a PDF/PNG figure plus a short machine-readable summary.  It does not
run evaluation, load model weights, bake checkpoints, or alter training paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFLICTS = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/"
    "memory_code_conflict_20260521/source_conflict_pairs.csv"
)
DEFAULT_ARCHETYPES = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/"
    "rcrf_conflict_clusters_20260522/archetype_summary.csv"
)
DEFAULT_EVIDENCE = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/"
    "rcrf_paper_evidence_table_20260522/rcrf_paper_evidence_table.csv"
)
DEFAULT_FIGURE_DIR = ROOT / "docs/paper/ExpertGym_ICLR/figures"
DEFAULT_REPORT = ROOT / "docs/report/RCRF/20260523_iclr_main_diagnostic_figure.md"


SPAN_ORDER = ["LB_prompt", "LB_reasoning", "LB_code", "LCB_prompt", "LCB_code"]
ROLE_LABELS = {
    "behavior_only": "behavior only",
    "clean_code_repair": "clean code repair",
    "code_negative_noise": "code negative noise",
    "code_negative_with_behavior_support": "code negative + behavior support",
    "code_repair_with_behavior_harm": "code repair + behavior harm",
    "code_source_conflict": "code source conflict",
    "weak_or_uninformative": "weak / uninformative",
}
SELECTED_VARIANTS = ["v8", "v18", "v19", "v14", "v15"]


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")


def build_pair_matrix(df: pd.DataFrame, value_col: str, diagonal: float) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=SPAN_ORDER, columns=SPAN_ORDER, dtype=float)
    for span in SPAN_ORDER:
        matrix.loc[span, span] = diagonal
    for _, row in df.iterrows():
        left = str(row["left"])
        right = str(row["right"])
        if left not in matrix.index or right not in matrix.columns:
            continue
        value = float(row[value_col])
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def shorten_role(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " "))


def selected_evidence_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["short"].isin(SELECTED_VARIANTS)].copy()
    order = {name: idx for idx, name in enumerate(SELECTED_VARIANTS)}
    rows["_order"] = rows["short"].map(order)
    return rows.sort_values("_order")


def annotate_heatmap(ax: plt.Axes, matrix: pd.DataFrame, fmt: str = ".2f") -> None:
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            value = matrix.loc[row, col]
            if pd.isna(value):
                continue
            color = "white" if abs(float(value)) > 0.55 else "black"
            ax.text(j, i, format(float(value), fmt), ha="center", va="center", fontsize=7, color=color)


def build_figure(conflicts: pd.DataFrame, archetypes: pd.DataFrame, evidence: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    pearson = build_pair_matrix(conflicts, "pearson", 1.0)
    evidence_rows = selected_evidence_rows(evidence)

    fig = plt.figure(figsize=(12.8, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0])

    ax0 = fig.add_subplot(grid[0, 0])
    im = ax0.imshow(pearson.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax0.set_title("A. Code span evidence is not one direction", fontsize=10, pad=8)
    ax0.set_xticks(range(len(SPAN_ORDER)), SPAN_ORDER, rotation=45, ha="right", fontsize=8)
    ax0.set_yticks(range(len(SPAN_ORDER)), SPAN_ORDER, fontsize=8)
    annotate_heatmap(ax0, pearson)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson", fontsize=8)

    ax1 = fig.add_subplot(grid[0, 1])
    role_rows = archetypes.sort_values("row_count", ascending=True).copy()
    labels = [shorten_role(x) for x in role_rows["archetype"]]
    colors = ["#4C78A8" if "code source conflict" not in label else "#E45756" for label in labels]
    ax1.barh(labels, role_rows["row_count"], color=colors, alpha=0.9)
    ax1.set_title("B. Most residual rows are conflict or weak evidence", fontsize=10, pad=8)
    ax1.set_xlabel("residual rows", fontsize=8)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.tick_params(axis="x", labelsize=8)
    for y, value in enumerate(role_rows["row_count"]):
        ax1.text(float(value) + 3, y, str(int(value)), va="center", fontsize=7)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(grid[0, 2])
    x = evidence_rows["code_hurt_bon_mean"].astype(float)
    y = evidence_rows["memory_eval50_f1"].astype(float)
    sizes = 60 + 900 * (evidence_rows["tool_quick_mean"].astype(float) - 0.77).clip(lower=0)
    ax2.scatter(x, y, s=sizes, c="#54A24B", alpha=0.8, edgecolor="black", linewidth=0.6)
    for _, row in evidence_rows.iterrows():
        ax2.annotate(
            str(row["short"]),
            (float(row["code_hurt_bon_mean"]), float(row["memory_eval50_f1"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax2.set_title("C. Rules expose a Memory-Code trade-off", fontsize=10, pad=8)
    ax2.set_xlabel("Code hurt BoN mean", fontsize=8)
    ax2.set_ylabel("Memory eval50 F1", fontsize=8)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(True, linewidth=0.4, alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(conflicts: pd.DataFrame, archetypes: pd.DataFrame, evidence: pd.DataFrame, output_pdf: Path, output_png: Path) -> dict[str, Any]:
    top_conflicts = conflicts.sort_values("conflict_rate", ascending=False).head(5)
    role_total = int(archetypes["row_count"].sum())
    role_rows = [
        {
            "archetype": str(row["archetype"]),
            "row_count": int(row["row_count"]),
            "fraction": float(row["row_count"]) / role_total if role_total else None,
        }
        for _, row in archetypes.sort_values("row_count", ascending=False).iterrows()
    ]
    selected = selected_evidence_rows(evidence)
    tradeoff_rows = [
        {
            "variant": str(row["short"]),
            "method": str(row["method"]),
            "tool_quick_mean": float(row["tool_quick_mean"]),
            "memory_eval50_f1": float(row["memory_eval50_f1"]),
            "code_hurt_bon_mean": float(row["code_hurt_bon_mean"]),
            "code_hurt_acc_mean": float(row["code_hurt_acc_mean"]),
        }
        for _, row in selected.iterrows()
    ]
    return {
        "figure_pdf": str(output_pdf),
        "figure_png": str(output_png),
        "span_order": SPAN_ORDER,
        "top_conflict_pairs": [
            {
                "left": str(row["left"]),
                "right": str(row["right"]),
                "pearson": float(row["pearson"]),
                "conflict_rate": float(row["conflict_rate"]),
            }
            for _, row in top_conflicts.iterrows()
        ],
        "role_total": role_total,
        "role_rows": role_rows,
        "tradeoff_rows": tradeoff_rows,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ICLR Main Diagnostic Figure",
        "",
        f"PDF: `{summary['figure_pdf']}`",
        f"PNG: `{summary['figure_png']}`",
        "",
        "## What The Figure Shows",
        "",
        "1. Code evidence is span/source-conditioned rather than one smooth direction.",
        "2. Most residual rows are not clean task rows; they are conflict, behavior-related, or weak-evidence rows.",
        "3. RCF-BC variants expose a Memory-Code trade-off that scalar coefficients hide.",
        "",
        "## Top Code Span Conflicts",
        "",
        "| left | right | pearson | conflict rate |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary["top_conflict_pairs"]:
        lines.append(f"| {row['left']} | {row['right']} | {row['pearson']:.4f} | {row['conflict_rate']:.4f} |")
    lines.extend(["", "## Residual Role Counts", "", "| role | rows | fraction |", "| --- | ---: | ---: |"])
    for row in summary["role_rows"]:
        lines.append(f"| {row['archetype']} | {row['row_count']} | {row['fraction']:.4f} |")
    lines.extend(["", "## Trade-off Rows", "", "| variant | Tool | Memory F1 | Code BoN | Code Acc |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in summary["tradeoff_rows"]:
        lines.append(
            f"| {row['variant']} | {row['tool_quick_mean']:.4f} | {row['memory_eval50_f1']:.4f} | "
            f"{row['code_hurt_bon_mean']:.4f} | {row['code_hurt_acc_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paper Claim Supported",
            "",
            "This figure supports the mechanism claim that agent task vectors should be composed at residual-entry granularity under behavior constraints, not by a single expert-level scalar.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--archetypes", type=Path, default=DEFAULT_ARCHETYPES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_FIGURE_DIR / "diagnostic_residual_field_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_file(args.conflicts, "source conflict pairs")
    require_file(args.archetypes, "archetype summary")
    require_file(args.evidence, "paper evidence table")

    conflicts = pd.read_csv(args.conflicts)
    archetypes = pd.read_csv(args.archetypes)
    evidence = pd.read_csv(args.evidence)

    output_pdf = args.figure_dir / "diagnostic_residual_field.pdf"
    output_png = args.figure_dir / "diagnostic_residual_field.png"
    build_figure(conflicts, archetypes, evidence, output_pdf, output_png)
    summary = build_summary(conflicts, archetypes, evidence, output_pdf, output_png)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, summary)

    print(f"wrote {output_pdf}")
    print(f"wrote {output_png}")
    print(f"wrote {args.summary_json}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
