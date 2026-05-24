#!/usr/bin/env python3
"""Build a success-conditioned activation projection ledger.

The input CSVs are produced by:

    scripts/attention_pauh/probe_signed_utility.py --write-projection-summary

Each input row already decomposes an expert's activation update into aligned,
conflicting, and orthogonal norm mass relative to the successful span direction
used by that probe. This script treats task/source labels as provenance only and
aggregates the geometry by residual entry.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FLOAT_FIELDS = {
    "layer",
    "count",
    "token_count",
    "align_ratio",
    "conflict_ratio",
    "orth_ratio",
    "total_norm_mean",
    "owner_norm_mean",
    "cosine_mean",
    "negative_fraction",
    "positive_fraction",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for item in args.projection_csv:
        label, path = parse_labeled_path(item)
        source_rows.extend(read_projection_csv(label=label, path=path))
    if not source_rows:
        raise ValueError("No projection rows were loaded.")

    ledger_rows = build_ledger(
        rows=source_rows,
        vote_margin=float(args.vote_margin),
        min_support_score=float(args.min_support_score),
        min_conflict_score=float(args.min_conflict_score),
        stable_fraction=float(args.stable_fraction),
        null_score_threshold=float(args.null_score_threshold),
        strong_source_support=float(args.strong_source_support),
    )
    metadata = {
        "format": "success_conditioned_projection_ledger_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "projection_csv": [str(Path(parse_labeled_path(item)[1]).expanduser().resolve()) for item in args.projection_csv],
        "num_input_rows": len(source_rows),
        "num_ledger_rows": len(ledger_rows),
        "vote_margin": float(args.vote_margin),
        "min_support_score": float(args.min_support_score),
        "min_conflict_score": float(args.min_conflict_score),
        "stable_fraction": float(args.stable_fraction),
        "null_score_threshold": float(args.null_score_threshold),
        "strong_source_support": float(args.strong_source_support),
        "role_counts": dict(Counter(row["role"] for row in ledger_rows)),
    }

    write_json(output_dir / "success_projection_ledger_summary.json", metadata)
    write_csv(output_dir / "success_projection_ledger.csv", ledger_rows)
    write_markdown(output_dir / "success_projection_ledger.md", metadata, ledger_rows)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "success_projection_ledger_summary.json"),
                "ledger": str(output_dir / "success_projection_ledger.csv"),
                "num_rows": len(ledger_rows),
                "role_counts": metadata["role_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-csv", action="append", required=True, help="label=/path/to/activation_update_projection_summary.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vote-margin", type=float, default=0.005)
    parser.add_argument("--min-support-score", type=float, default=0.03)
    parser.add_argument("--min-conflict-score", type=float, default=0.03)
    parser.add_argument("--stable-fraction", type=float, default=0.5)
    parser.add_argument("--null-score-threshold", type=float, default=0.90)
    parser.add_argument(
        "--strong-source-support",
        type=float,
        default=0.25,
        help="If any success source has support above this value, conflict is mixed rather than scalar-suppressible.",
    )
    return parser.parse_args()


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path).expanduser().resolve()
    path = Path(raw).expanduser().resolve()
    return path.parent.name, path


def read_projection_csv(*, label: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in FLOAT_FIELDS:
                row[field] = float(row.get(field) or 0.0)
            row["source_label"] = label
            row["source_task"] = str(row.get("task") or "")
            rows.append(row)
    return rows


def build_ledger(
    *,
    rows: list[Mapping[str, Any]],
    vote_margin: float,
    min_support_score: float,
    min_conflict_score: float,
    stable_fraction: float,
    null_score_threshold: float,
    strong_source_support: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["param_name"]), str(row["expert"]))].append(row)

    output = []
    for (param_name, expert), items in grouped.items():
        total_weight = 0.0
        support_sum = 0.0
        conflict_sum = 0.0
        null_sum = 0.0
        cosine_sum = 0.0
        negative_sum = 0.0
        positive_sum = 0.0
        token_sum = 0.0
        votes = Counter()
        source_tasks = []
        breakdown = []
        max_source_support = 0.0
        max_source_conflict = 0.0
        for row in items:
            weight = projection_weight(row)
            total_weight += weight
            support_sum += float(row["align_ratio"]) * weight
            conflict_sum += float(row["conflict_ratio"]) * weight
            null_sum += float(row["orth_ratio"]) * weight
            cosine_sum += float(row["cosine_mean"]) * weight
            negative_sum += float(row["negative_fraction"]) * weight
            positive_sum += float(row["positive_fraction"]) * weight
            token_sum += float(row["token_count"])
            vote = projection_vote(row, margin=vote_margin)
            max_source_support = max(max_source_support, float(row["align_ratio"]))
            max_source_conflict = max(max_source_conflict, float(row["conflict_ratio"]))
            votes[vote] += 1
            source_key = f"{row['source_label']}:{row['source_task']}"
            source_tasks.append(source_key)
            breakdown.append(
                {
                    "source": source_key,
                    "support": round(float(row["align_ratio"]), 6),
                    "conflict": round(float(row["conflict_ratio"]), 6),
                    "null": round(float(row["orth_ratio"]), 6),
                    "cosine": round(float(row["cosine_mean"]), 6),
                    "vote": vote,
                }
            )

        denom = max(total_weight, 1.0e-24)
        n = max(len(items), 1)
        support_score = support_sum / denom
        conflict_score = conflict_sum / denom
        null_score = null_sum / denom
        support_fraction = float(votes["support"]) / n
        conflict_fraction = float(votes["conflict"]) / n
        mixed_fraction = float(votes["mixed"]) / n
        role = classify_role(
            support_score=support_score,
            conflict_score=conflict_score,
            null_score=null_score,
            support_fraction=support_fraction,
            conflict_fraction=conflict_fraction,
            mixed_fraction=mixed_fraction,
            min_support_score=min_support_score,
            min_conflict_score=min_conflict_score,
            stable_fraction=stable_fraction,
            null_score_threshold=null_score_threshold,
            max_source_support=max_source_support,
            strong_source_support=strong_source_support,
        )
        output.append(
            {
                "param_name": param_name,
                "expert": expert,
                "layer": int(float(items[0].get("layer", 0.0))),
                "module": str(items[0].get("module") or ""),
                "family": str(items[0].get("family") or ""),
                "role": role,
                "support_score": support_score,
                "conflict_score": conflict_score,
                "null_score": null_score,
                "signed_margin": support_score - conflict_score,
                "support_fraction": support_fraction,
                "conflict_fraction": conflict_fraction,
                "mixed_fraction": mixed_fraction,
                "max_source_support": max_source_support,
                "max_source_conflict": max_source_conflict,
                "mean_cosine": cosine_sum / denom,
                "negative_fraction": negative_sum / denom,
                "positive_fraction": positive_sum / denom,
                "num_observations": len(items),
                "num_source_tasks": len(set(source_tasks)),
                "token_count": token_sum,
                "projection_weight": total_weight,
                "source_tasks": ",".join(sorted(set(source_tasks))),
                "breakdown_json": json.dumps(breakdown, sort_keys=True),
            }
        )
    output.sort(
        key=lambda row: (
            role_rank(str(row["role"])),
            -abs(float(row["signed_margin"])),
            str(row["param_name"]),
            str(row["expert"]),
        )
    )
    return output


def projection_weight(row: Mapping[str, Any]) -> float:
    token_count = max(float(row.get("token_count", 0.0)), 1.0)
    update_norm = max(float(row.get("total_norm_mean", 0.0)), 1.0e-12)
    return token_count * update_norm


def projection_vote(row: Mapping[str, Any], *, margin: float) -> str:
    support = float(row["align_ratio"])
    conflict = float(row["conflict_ratio"])
    null = float(row["orth_ratio"])
    if support - conflict > margin:
        return "support"
    if conflict - support > margin:
        return "conflict"
    if null >= 0.90:
        return "null"
    return "mixed"


def classify_role(
    *,
    support_score: float,
    conflict_score: float,
    null_score: float,
    support_fraction: float,
    conflict_fraction: float,
    mixed_fraction: float,
    max_source_support: float,
    min_support_score: float,
    min_conflict_score: float,
    stable_fraction: float,
    null_score_threshold: float,
    strong_source_support: float,
) -> str:
    has_support = support_score >= min_support_score
    has_conflict = conflict_score >= min_conflict_score
    stable_support = support_fraction >= stable_fraction and has_support
    stable_conflict = conflict_fraction >= stable_fraction and has_conflict
    has_strong_source_support = max_source_support >= strong_source_support
    if stable_support and stable_conflict:
        return "mixed_success_geometry"
    if stable_support and has_conflict:
        return "mixed_success_geometry"
    if stable_conflict and has_support:
        return "mixed_success_geometry"
    if stable_conflict and has_strong_source_support:
        return "mixed_success_geometry"
    if stable_support:
        return "stable_support"
    if stable_conflict:
        return "stable_conflict"
    if has_support and has_conflict:
        return "mixed_success_geometry"
    if null_score >= null_score_threshold and mixed_fraction < stable_fraction:
        return "mostly_null"
    return "weak_or_unstable"


def role_rank(role: str) -> int:
    order = {
        "stable_conflict": 0,
        "mixed_success_geometry": 1,
        "stable_support": 2,
        "mostly_null": 3,
        "weak_or_unstable": 4,
    }
    return order.get(role, 99)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_markdown(path: Path, metadata: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Success-Conditioned Projection Ledger",
        "",
        "This is a geometry diagnostic, not a checkpoint edit. Task and expert names are provenance only.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Role Counts",
        "",
        "| role | count |",
        "| --- | ---: |",
    ]
    for role, count in sorted(metadata["role_counts"].items()):
        lines.append(f"| {role} | {count} |")
    lines.extend(["", "## Stable Conflict Candidates", "", table_for(rows, role="stable_conflict")])
    lines.extend(["", "## Mixed Success Geometry", "", table_for(rows, role="mixed_success_geometry")])
    lines.extend(["", "## Stable Support Anchors", "", table_for(rows, role="stable_support")])
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- `stable_conflict` means anti-alignment with observed successful directions, not proven failure causality.",
            "- `mostly_null` means outside the observed success bank, not useless.",
            "- Projection candidates should be capped and evaluated on the subset loop before any Eval6 promotion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def table_for(rows: list[Mapping[str, Any]], *, role: str, limit: int = 25) -> str:
    selected = [row for row in rows if row["role"] == role]
    if not selected:
        return "_No rows._"
    selected.sort(key=lambda row: abs(float(row["signed_margin"])), reverse=True)
    lines = [
        "| rank | expert | layer | module | support | conflict | null | margin | sources | param |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(selected[:limit], start=1):
        lines.append(
            f"| {rank} | {row['expert']} | {row['layer']} | {row['module']} | "
            f"{float(row['support_score']):.4f} | {float(row['conflict_score']):.4f} | "
            f"{float(row['null_score']):.4f} | {float(row['signed_margin']):.4f} | "
            f"{row['source_tasks']} | `{row['param_name']}` |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
