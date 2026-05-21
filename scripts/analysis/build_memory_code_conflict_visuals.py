#!/usr/bin/env python3
"""Build static visuals for the RCRF Memory/Code conflict analysis.

The script is deliberately read-only with respect to model artifacts. It
collects existing contrast summaries, gate decisions, and Code-hurt quick
evaluation outputs, then writes auditable CSV/JSON/SVG files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
OUTPUT_ROOT = ROOT / "analysis" / "memory_code_conflict_20260521"
CONFLICT_JSON = ROOT / "contrast" / "source_conflicts_20260521" / "source_conflicts.json"
CURE_OUTPUT_ROOT = Path("/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data")
GATE_ROOT = ROOT / "contrast_gates"


VARIANTS = {
    "v1_code_only": "rcrf_code_contrast_v1",
    "v2_spanaware": "rcrf_code_spanaware_conservative_v2",
    "v3_memory_hard_floor": "rcrf_code_spanaware_memory_preserve_v3",
    "v4_memory_utility_floor": "rcrf_code_spanaware_memory_utility_preserve_v4",
}

DATASETS = {
    "LiveBench": "LiveBenchCodeHurtRcrfVsTa16",
    "LiveCodeBench": "LiveCodeBenchCodeHurtRcrfVsTa16",
}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conflict_df = load_conflict_matrix()
    code_df = load_code_metrics()
    gate_df = load_gate_metrics()
    memory_delta_df = load_memory_delta_rows()
    preserve_df = load_preserve_rows()

    write_tables(output_dir, conflict_df, code_df, gate_df, memory_delta_df, preserve_df)
    write_figures(output_dir, conflict_df, code_df, gate_df, memory_delta_df)
    write_summary_json(output_dir, conflict_df, code_df, gate_df, memory_delta_df, preserve_df)

    print(json.dumps({"output_dir": str(output_dir), "figures": sorted(p.name for p in output_dir.glob("*.svg"))}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_conflict_matrix() -> pd.DataFrame:
    payload = load_json(CONFLICT_JSON)
    rows = []
    for row in payload["pairs"]:
        rows.append(
            {
                "left": row["left"],
                "right": row["right"],
                "pearson": float(row["pearson"]),
                "conflict_count": int(row["conflict_count"]),
                "agreement_count": int(row["agreement_count"]),
                "overlap_count": int(row["overlap_count"]),
                "conflict_rate": int(row["conflict_count"]) / int(row["overlap_count"]),
                "mean_abs_left": float(row["mean_abs_left"]),
                "mean_abs_right": float(row["mean_abs_right"]),
            }
        )
    return pd.DataFrame(rows)


def load_code_metrics() -> pd.DataFrame:
    rows = []
    for variant, slug in VARIANTS.items():
        for dataset_name, dataset_slug in DATASETS.items():
            path = CURE_OUTPUT_ROOT / f"outputs-eval-.tmp.shared-storage.OnPolicy.checkpoints.{slug}-{dataset_slug}.json"
            if not path.exists():
                continue
            metrics = summarize_cure_output(path)
            rows.append({"variant": variant, "dataset": dataset_name, **metrics, "source": str(path)})
    return pd.DataFrame(rows)


def summarize_cure_output(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pass_any = 0
    pass_count = 0
    total_points = 0
    pass_points = 0
    per_case_pass_count = []
    for row in data:
        table = row.get("test_bool_table") or []
        case_pass_count = sum(1 for candidate in table if candidate and all(candidate))
        per_case_pass_count.append(case_pass_count)
        pass_count += case_pass_count
        pass_any += int(case_pass_count > 0)
        for candidate in table:
            total_points += len(candidate)
            pass_points += sum(bool(item) for item in candidate)
    num_cases = len(data)
    num_candidates = num_cases * 4
    return {
        "num_cases": num_cases,
        "pass_any": pass_any / num_cases if num_cases else 0.0,
        "pass_any_count": pass_any,
        "candidate_pass_rate": pass_count / num_candidates if num_candidates else 0.0,
        "pass_count": pass_count,
        "test_point_rate": pass_points / total_points if total_points else 0.0,
        "per_case_pass_count": per_case_pass_count,
    }


def load_gate_metrics() -> pd.DataFrame:
    rows = []
    for variant, slug in VARIANTS.items():
        path = GATE_ROOT / slug / "gates.json"
        if not path.exists():
            continue
        payload = load_json(path)
        decision_summary = payload.get("decision_summary", {})
        reason_counts = decision_summary.get("reason_counts", {})
        preserve_summary = payload.get("preserve_signal_summary", {})
        memory_delta = payload.get("delta_summary", {}).get("memory", {})
        rows.append(
            {
                "variant": variant,
                "gate_path": str(path),
                "memory_changed": memory_delta.get("changed_count", 0),
                "memory_positive": memory_delta.get("positive_count", 0),
                "memory_negative": memory_delta.get("negative_count", 0),
                "memory_mean_abs_delta": memory_delta.get("mean_abs", 0.0),
                "memory_max_abs_delta": memory_delta.get("max_abs", 0.0),
                "protected_negative_overlay": reason_counts.get("protected_negative_overlay", 0),
                "preserve_utility_floor": reason_counts.get("preserve_utility_floor", 0),
                "preserved_keys": preserve_summary.get("num_preserved_keys", 0),
                "memory_mean_coeff": payload.get("coefficient_summary", {}).get("memory", {}).get("mean", 0.0),
            }
        )
    return pd.DataFrame(rows)


def load_memory_delta_rows() -> pd.DataFrame:
    rows = []
    for variant, slug in VARIANTS.items():
        path = GATE_ROOT / slug / "gates.json"
        if not path.exists():
            continue
        payload = load_json(path)
        for row in payload.get("decision_rows", []):
            if row.get("expert") != "memory":
                continue
            rows.append(
                {
                    "variant": variant,
                    "layer": parse_layer(str(row["param_name"])),
                    "family": module_family(str(row["param_name"])),
                    "param_name": row["param_name"],
                    "delta": float(row.get("delta", 0.0)),
                    "reason": row.get("reason", ""),
                    "score": float(row.get("metrics", {}).get("score", 0.0)),
                    "preserve_task": row.get("preserve_signal", {}).get("task", ""),
                    "preserve_utility": row.get("preserve_signal", {}).get("normalized_preserve_utility", 0.0),
                }
            )
    return pd.DataFrame(rows)


def load_preserve_rows() -> pd.DataFrame:
    rows = []
    for variant, slug in VARIANTS.items():
        path = GATE_ROOT / slug / "gates.json"
        if not path.exists():
            continue
        payload = load_json(path)
        for row in payload.get("decision_rows", []):
            preserve_signal = row.get("preserve_signal") or {}
            if not preserve_signal:
                continue
            rows.append(
                {
                    "variant": variant,
                    "layer": parse_layer(str(row["param_name"])),
                    "family": module_family(str(row["param_name"])),
                    "param_name": row["param_name"],
                    "delta": float(row.get("delta", 0.0)),
                    "reason": row.get("reason", ""),
                    "preserve_task": preserve_signal.get("task", ""),
                    "preserve_utility": float(preserve_signal.get("normalized_preserve_utility", 0.0)),
                    "positive_fraction": float(preserve_signal.get("positive_fraction", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def write_tables(
    output_dir: Path,
    conflict_df: pd.DataFrame,
    code_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    memory_delta_df: pd.DataFrame,
    preserve_df: pd.DataFrame,
) -> None:
    conflict_df.to_csv(output_dir / "source_conflict_pairs.csv", index=False)
    code_df.to_csv(output_dir / "code_hurt_metrics.csv", index=False)
    gate_df.to_csv(output_dir / "gate_memory_summary.csv", index=False)
    memory_delta_df.to_csv(output_dir / "memory_delta_rows.csv", index=False)
    preserve_df.to_csv(output_dir / "preserve_rows.csv", index=False)


def write_figures(output_dir: Path, conflict_df: pd.DataFrame, code_df: pd.DataFrame, gate_df: pd.DataFrame, memory_delta_df: pd.DataFrame) -> None:
    plot_conflict_heatmaps(output_dir, conflict_df)
    plot_code_metrics(output_dir, code_df)
    plot_memory_gate_summary(output_dir, gate_df)
    plot_memory_delta_by_layer(output_dir, memory_delta_df)


def plot_conflict_heatmaps(output_dir: Path, df: pd.DataFrame) -> None:
    labels = sorted(set(df["left"]) | set(df["right"]))
    pearson = pd.DataFrame(float("nan"), index=labels, columns=labels)
    conflict_rate = pd.DataFrame(float("nan"), index=labels, columns=labels)
    for label in labels:
        pearson.loc[label, label] = 1.0
        conflict_rate.loc[label, label] = 0.0
    for row in df.itertuples():
        pearson.loc[row.left, row.right] = row.pearson
        pearson.loc[row.right, row.left] = row.pearson
        conflict_rate.loc[row.left, row.right] = row.conflict_rate
        conflict_rate.loc[row.right, row.left] = row.conflict_rate
    heatmap(output_dir / "source_pearson_heatmap.svg", pearson, "Source Pearson", vmin=-1, vmax=1, cmap="coolwarm")
    heatmap(output_dir / "source_conflict_rate_heatmap.svg", conflict_rate, "Source Sign-Conflict Rate", vmin=0, vmax=1, cmap="magma")


def plot_code_metrics(output_dir: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    metric_names = ["pass_any", "test_point_rate"]
    for metric in metric_names:
        pivot = df.pivot(index="variant", columns="dataset", values=metric).loc[list(VARIANTS.keys())]
        ax = pivot.plot(kind="bar", figsize=(9.5, 4.8), ylim=(0, 1.0), width=0.78)
        ax.set_title(f"Code hurt {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
        plt.tight_layout()
        plt.savefig(output_dir / f"code_hurt_{metric}.svg", format="svg")
        plt.close()


def plot_memory_gate_summary(output_dir: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    plot_df = df.set_index("variant").loc[list(VARIANTS.keys())]
    columns = ["memory_positive", "memory_negative", "protected_negative_overlay", "preserve_utility_floor"]
    ax = plot_df[columns].plot(kind="bar", figsize=(10, 4.8), width=0.8)
    ax.set_title("Memory expert delta/protection counts")
    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(output_dir / "memory_delta_protection_counts.svg", format="svg")
    plt.close()


def plot_memory_delta_by_layer(output_dir: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    selected = df[df["variant"].isin(["v2_spanaware", "v3_memory_hard_floor", "v4_memory_utility_floor"])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, sharey=True)
    colors = {"attn_q": "#0072B2", "attn_k": "#009E73", "attn_v": "#56B4E9", "attn_o": "#CC79A7", "mlp_gate": "#E69F00", "mlp_up": "#D55E00", "mlp_down": "#000000"}
    for ax, variant in zip(axes, ["v2_spanaware", "v3_memory_hard_floor", "v4_memory_utility_floor"]):
        sub = selected[selected["variant"] == variant]
        for family, family_rows in sub.groupby("family"):
            ax.scatter(family_rows["layer"], family_rows["delta"], s=18, alpha=0.72, label=family, color=colors.get(family, "gray"))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(variant)
        ax.set_ylabel("memory delta")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("layer")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 0.95), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "memory_delta_by_layer.svg", format="svg", bbox_inches="tight")
    plt.close()


def heatmap(path: Path, matrix: pd.DataFrame, title: str, *, vmin: float, vmax: float, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix.values, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.values[i, j]
            if math.isfinite(float(value)):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def write_summary_json(
    output_dir: Path,
    conflict_df: pd.DataFrame,
    code_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    memory_delta_df: pd.DataFrame,
    preserve_df: pd.DataFrame,
) -> None:
    summary = {
        "output_dir": str(output_dir),
        "num_conflict_pairs": len(conflict_df),
        "num_code_metric_rows": len(code_df),
        "num_gate_rows": len(gate_df),
        "num_memory_delta_rows": len(memory_delta_df),
        "num_preserve_rows": len(preserve_df),
        "top_conflict_pairs": conflict_df.sort_values("conflict_rate", ascending=False).head(5).to_dict(orient="records"),
        "code_metrics": code_df.to_dict(orient="records"),
        "gate_metrics": gate_df.to_dict(orient="records"),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_layer(param_name: str) -> int:
    parts = param_name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            return int(parts[idx + 1])
    return -1


def module_family(param_name: str) -> str:
    if ".self_attn.q_proj." in param_name:
        return "attn_q"
    if ".self_attn.k_proj." in param_name:
        return "attn_k"
    if ".self_attn.v_proj." in param_name:
        return "attn_v"
    if ".self_attn.o_proj." in param_name:
        return "attn_o"
    if ".mlp.gate_proj." in param_name:
        return "mlp_gate"
    if ".mlp.up_proj." in param_name:
        return "mlp_up"
    if ".mlp.down_proj." in param_name:
        return "mlp_down"
    return "other"


if __name__ == "__main__":
    main()
