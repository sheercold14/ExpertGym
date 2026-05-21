#!/usr/bin/env python3
"""Export paper-ready RCRF mechanism figures from long-form workbench data."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

try:
    import plotly.express as px
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: plotly. Install workbench dependencies with "
        "`pip install -r scripts/visualization/attn/analysis_platform/requirements.txt`."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO_ROOT / "scripts" / "visualization" / "attn" / "analysis_platform"
sys.path.insert(0, str(PLATFORM_DIR))

from rcrf_schema import MODULE_ORDER, RCRF_CANDIDATES, load_mechanism_data  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    include_defaults = args.include_default_paths or not data_dir.exists()
    data = load_mechanism_data([str(data_dir)], include_defaults=include_defaults, example_on_empty=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    manifest: dict[str, object] = {"output_dir": str(output_dir), "figures": [], "warnings": []}

    figure_specs = build_figures(data)
    for stem, fig, table in figure_specs:
        result = export_one(fig, table, output_dir=output_dir, stem=stem, formats=formats)
        manifest["figures"].append(result["files"])
        manifest["warnings"].extend(result["warnings"])

    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(PLATFORM_DIR / "data"),
        help="Directory with residual/interference/gate/eval JSONL files.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for exported figures and CSV tables.")
    parser.add_argument("--formats", default="html,png,pdf,csv", help="Comma-separated output formats.")
    parser.add_argument("--include-default-paths", action="store_true", help="Also scan default live RCRF artifact paths.")
    return parser.parse_args()


def build_figures(data) -> list[tuple[str, object, pd.DataFrame]]:
    figures: list[tuple[str, object, pd.DataFrame]] = []

    if not data.residual.empty:
        signed = data.residual.groupby(["layer", "module"], as_index=False)["signed_effect"].mean()
        figures.append(
            (
                "module_map_signed_effect",
                heatmap_layer_module(
                    signed,
                    value="signed_effect",
                    title="RCRF Module Mechanism Map: signed_effect",
                    colorbar="mean[-<g, DeltaW h>] (loss units)",
                    diverging=True,
                ),
                signed,
            )
        )
        expression = data.residual.groupby(["layer", "module"], as_index=False)["expression"].mean()
        figures.append(
            (
                "module_map_expression",
                heatmap_layer_module(
                    expression,
                    value="expression",
                    title="RCRF Module Mechanism Map: expression",
                    colorbar="mean||DeltaW h||^2 (activation energy)",
                    diverging=False,
                ),
                expression,
            )
        )

    if not data.interference.empty:
        conflict = data.interference.groupby(["layer", "module"], as_index=False)["conflict_score"].mean()
        figures.append(
            (
                "interference_conflict_score",
                heatmap_layer_module(
                    conflict,
                    value="conflict_score",
                    title="RCRF Pairwise Interference: conflict_score",
                    colorbar="conflict score (unitless)",
                    diverging=False,
                ),
                conflict,
            )
        )

    if not data.gate.empty:
        candidate = first_candidate(data.gate["candidate"].unique())
        gate_rows = data.gate[data.gate["candidate"] == candidate]
        alpha = gate_rows.groupby(["layer", "module"], as_index=False)["alpha"].mean()
        figures.append(
            (
                f"gate_alpha_{candidate}",
                heatmap_layer_module(
                    alpha,
                    value="alpha",
                    title=f"RCRF Gate Alpha Heatmap: {candidate}",
                    colorbar="gate alpha (unitless)",
                    diverging=False,
                ),
                alpha,
            )
        )

        action_rows = data.gate.copy()
        action_rows["action"] = action_rows["alpha"].map(gate_action)
        action_counts = action_rows.groupby(["candidate", "expert", "action"], as_index=False).size().rename(columns={"size": "count"})
        fig_action = px.bar(
            action_counts,
            x="candidate",
            y="count",
            color="action",
            facet_col="expert",
            title="RCRF Gate Action Distribution by Candidate and Expert",
            labels={"candidate": "candidate", "count": "module count", "action": "gate action"},
        )
        figures.append(("gate_action_distribution", fig_action, action_counts))

    if not data.eval.empty:
        eval_rows = data.eval.copy()
        metric = "accuracy" if "accuracy" in set(eval_rows["metric"]) else str(eval_rows["metric"].iloc[0])
        eval_metric = eval_rows[eval_rows["metric"] == metric]
        fig_eval = px.bar(
            eval_metric,
            x="subset",
            y="score",
            color="candidate",
            facet_col="task",
            barmode="group",
            title=f"RCRF Eval Context: {metric}",
            labels={"subset": "subset", "score": f"{metric} score", "candidate": "candidate"},
        )
        figures.append((f"eval_{metric}", fig_eval, eval_metric))

    return figures


def heatmap_layer_module(df: pd.DataFrame, *, value: str, title: str, colorbar: str, diverging: bool):
    table = df.copy()
    table["module"] = pd.Categorical(table["module"], categories=MODULE_ORDER, ordered=True)
    pivot = table.pivot_table(index="layer", columns="module", values=value, aggfunc="mean", observed=False).sort_index()
    pivot = pivot.reindex(columns=[m for m in MODULE_ORDER if m in pivot.columns])
    kwargs = {
        "aspect": "auto",
        "title": title,
        "labels": {"x": "module", "y": "layer", "color": colorbar},
        "color_continuous_scale": "RdBu_r" if diverging else "Viridis",
    }
    if diverging:
        kwargs["color_continuous_midpoint"] = 0.0
    fig = px.imshow(pivot, **kwargs)
    fig.update_xaxes(title_text="module (q/k/v/o/gate/up/down)")
    fig.update_yaxes(title_text="layer")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def export_one(fig, table: pd.DataFrame, *, output_dir: Path, stem: str, formats: list[str]) -> dict[str, object]:
    files: dict[str, str] = {}
    warnings: list[str] = []
    if "csv" in formats:
        path = output_dir / f"{stem}.csv"
        table.to_csv(path, index=False)
        files["csv"] = str(path)
    if "html" in formats:
        path = output_dir / f"{stem}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        files["html"] = str(path)
    for fmt in ("png", "pdf"):
        if fmt not in formats:
            continue
        path = output_dir / f"{stem}.{fmt}"
        try:
            if fmt == "png":
                fig.write_image(path, scale=2)
            else:
                fig.write_image(path)
            files[fmt] = str(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{stem}.{fmt}: {exc}")
    return {"files": {stem: files}, "warnings": warnings}


def first_candidate(values) -> str:
    values = [str(value) for value in values]
    for candidate in RCRF_CANDIDATES[::-1]:
        if candidate in values:
            return candidate
    return values[0]


def gate_action(alpha: float) -> str:
    alpha = float(alpha)
    if alpha >= 1.03:
        return "amplify"
    if alpha >= 0.97:
        return "retain"
    if alpha >= 0.70:
        return "suppress"
    return "sparse"


if __name__ == "__main__":
    main()
