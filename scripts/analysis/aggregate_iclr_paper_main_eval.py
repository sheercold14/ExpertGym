#!/usr/bin/env python3
"""Aggregate paper-main full Eval6 outputs for the ICLR ExpertGym draft.

This script is artifact-only: it never launches evaluation.  It reads the
directory layout produced by skill/command/run_20260523_iclr_paper_main_eval.sh
and converts Tool / Memory / Code logs into a single paper-facing table.
Missing logs are kept as explicit `missing` cells so the submission blocker is
auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523")
DEFAULT_RUN_ID = "iclr_main_eval6_20260523"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "docs/paper/ExpertGym_ICLR/paper_main_eval6_aggregate.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "docs/paper/ExpertGym_ICLR/paper_main_eval6_aggregate.json"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model_name: str
    role: str
    checkpoint: str


CANDIDATES: dict[str, Candidate] = {
    "bcrc_v18_alias_v9": Candidate(
        candidate_id="bcrc_v18_alias_v9",
        model_name="bcrc-v18-alias-v9",
        role="main method: soft behavior-constrained residual field",
        checkpoint="/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9",
    ),
    "no_behavior_v1_code_only": Candidate(
        candidate_id="no_behavior_v1_code_only",
        model_name="no-behavior-v1-code-only",
        role="ablation: no behavior constraint",
        checkpoint="/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_contrast_v1",
    ),
    "hard_behavior_v8": Candidate(
        candidate_id="hard_behavior_v8",
        model_name="hard-behavior-v8",
        role="ablation: hard behavior constraint",
        checkpoint="/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_v8",
    ),
    "strict_cleanup_v19": Candidate(
        candidate_id="strict_cleanup_v19",
        model_name="strict-cleanup-v19",
        role="optional: strict residual-archetype cleanup",
        checkpoint="/tmp/shared-storage/OnPolicy/checkpoints/rcrf_archetype_consistency_v19",
    ),
    "scalar_code_half_v14": Candidate(
        candidate_id="scalar_code_half_v14",
        model_name="scalar-code-half-v14",
        role="optional negative control: code coefficient x0.5",
        checkpoint="/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_half_v14",
    ),
    "scalar_code_zero_v15": Candidate(
        candidate_id="scalar_code_zero_v15",
        model_name="scalar-code-zero-v15",
        role="optional negative control: code coefficient zero",
        checkpoint="/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_zero_v15",
    ),
}


def parse_candidate_list(text: str) -> list[str]:
    result = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in result if item not in CANDIDATES]
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    return result


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def find_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return objects
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        index = start + max(end, 1)


def last_json_with_key(path: Path, key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    objects = find_json_objects(path.read_text(encoding="utf-8", errors="ignore"))
    for obj in reversed(objects):
        if key in obj:
            return obj
    return None


def parse_tool_log(path: Path) -> dict[str, Any]:
    obj = last_json_with_key(path, "scores")
    if not obj:
        return {"status": "missing"}
    scores = obj.get("scores") or {}
    category_scores: dict[str, float] = {}
    for category, payload in scores.items():
        if isinstance(payload, dict) and isinstance(payload.get("accuracy"), (int, float)):
            category_scores[str(category)] = float(payload["accuracy"])
    live_scores = [value for key, value in category_scores.items() if key.startswith("live_")]
    return {
        "status": "ready" if category_scores else "missing",
        "tool_mean": mean(list(category_scores.values())),
        "tool_live_mean": mean(live_scores),
        "tool_scores": category_scores,
        "tool_log": str(path),
    }


def parse_memory_log(path: Path) -> dict[str, Any]:
    obj = last_json_with_key(path, "datasets")
    if not obj:
        return {"status": "missing"}
    dataset_rows = []
    for item in obj.get("datasets") or []:
        if not isinstance(item, dict):
            continue
        if "avg_f1" not in item and "exact_match_rate" not in item:
            continue
        dataset_rows.append(
            {
                "dataset": item.get("dataset"),
                "exact_match_rate": item.get("exact_match_rate"),
                "avg_f1": item.get("avg_f1"),
                "total_samples": item.get("total_samples"),
            }
        )
    ems = [float(row["exact_match_rate"]) for row in dataset_rows if isinstance(row.get("exact_match_rate"), (int, float))]
    f1s = [float(row["avg_f1"]) for row in dataset_rows if isinstance(row.get("avg_f1"), (int, float))]
    return {
        "status": "ready" if dataset_rows else "missing",
        "memory_em": mean(ems),
        "memory_f1": mean(f1s),
        "memory_datasets": dataset_rows,
        "memory_log": str(path),
    }


def parse_code_log(path: Path) -> dict[str, Any]:
    obj = last_json_with_key(path, "datasets")
    if not obj:
        return {"status": "missing"}
    dataset_rows = []
    for item in obj.get("datasets") or []:
        if not isinstance(item, dict) or "code_acc" not in item:
            continue
        bon_payload = item.get("bon") or {}
        bon_44 = bon_payload.get("(4, 4)") if isinstance(bon_payload, dict) else None
        dataset_rows.append(
            {
                "dataset": item.get("dataset"),
                "code_acc": item.get("code_acc"),
                "code_tp": item.get("code_accumulate_acc"),
                "code_bon": bon_44.get("acc") if isinstance(bon_44, dict) else None,
            }
        )
    accs = [float(row["code_acc"]) for row in dataset_rows if isinstance(row.get("code_acc"), (int, float))]
    tps = [float(row["code_tp"]) for row in dataset_rows if isinstance(row.get("code_tp"), (int, float))]
    bons = [float(row["code_bon"]) for row in dataset_rows if isinstance(row.get("code_bon"), (int, float))]
    return {
        "status": "ready" if dataset_rows else "missing",
        "code_acc": mean(accs),
        "code_tp": mean(tps),
        "code_bon": mean(bons),
        "code_datasets": dataset_rows,
        "code_log": str(path),
    }


def combine_status(*statuses: str) -> str:
    if all(status == "ready" for status in statuses):
        return "ready"
    if any(status == "ready" for status in statuses):
        return "partial"
    return "missing"


def row_for_candidate(root: Path, run_id: str, candidate_id: str) -> dict[str, Any]:
    candidate = CANDIDATES[candidate_id]
    base = root / candidate_id / run_id
    tool = parse_tool_log(base / "tool_memory/logs/tool_bfcl.log")
    memory = parse_memory_log(base / "tool_memory/logs/memory_hotpotqa.log")
    code = parse_code_log(base / "code/logs/code_cure.log")
    tool_status = str(tool.pop("status"))
    memory_status = str(memory.pop("status"))
    code_status = str(code.pop("status"))
    row: dict[str, Any] = {
        "candidate": candidate.candidate_id,
        "model": candidate.model_name,
        "role": candidate.role,
        "checkpoint": candidate.checkpoint,
        "run_id": run_id,
        "root": str(base),
        "status": combine_status(tool_status, memory_status, code_status),
        "tool_status": tool_status,
        "memory_status": memory_status,
        "code_status": code_status,
    }
    row.update(tool)
    row.update(memory)
    row.update(code)
    main_scores = [
        row.get("tool_mean"),
        row.get("memory_f1"),
        row.get("code_acc"),
    ]
    valid_main = [float(value) for value in main_scores if isinstance(value, (int, float))]
    row["simple_avg"] = mean(valid_main) if len(valid_main) == 3 else None
    row["worst_task"] = min(valid_main) if len(valid_main) == 3 else None
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[str], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def build_markdown(rows: list[dict[str, Any]], root: Path, run_id: str) -> str:
    columns = [
        "candidate",
        "role",
        "status",
        "tool_status",
        "memory_status",
        "code_status",
        "tool_mean",
        "tool_live_mean",
        "memory_em",
        "memory_f1",
        "code_acc",
        "code_tp",
        "code_bon",
        "simple_avg",
        "worst_task",
    ]
    headers = [
        "candidate",
        "role",
        "status",
        "Tool leg",
        "Memory leg",
        "Code leg",
        "Tool",
        "Tool live",
        "Memory EM",
        "Memory F1",
        "Code Acc",
        "Code TP",
        "Code BoN",
        "Avg(T/M/C)",
        "Worst",
    ]
    missing = [row for row in rows if row["status"] != "ready"]
    lines = [
        "# Paper-Main Eval6 Aggregate",
        "",
        f"Root: `{root}`",
        f"Run id: `{run_id}`",
        "",
        "This file is generated from existing evaluation artifacts only. Missing cells mean the corresponding full Eval6 leg has not been run or the expected log is absent.",
        "",
        markdown_table(rows, columns, headers),
        "",
        "## Missing / Partial Items",
    ]
    if missing:
        lines.append(markdown_table(missing, ["candidate", "status", "root"], ["candidate", "status", "expected artifact root"]))
    else:
        lines.append("All selected candidates have Tool, Memory, and Code logs.")
    lines.extend(
        [
            "",
            "## Detail Pointers",
            "",
            markdown_table(rows, ["candidate", "tool_log", "memory_log", "code_log"], ["candidate", "Tool log", "Memory log", "Code log"]),
            "",
            "## Paper Use Rule",
            "",
            "Use this table as the final RCF-BC full Eval6 block only when the minimum queue is `ready`: `bcrc_v18_alias_v9`, `no_behavior_v1_code_only`, and `hard_behavior_v8`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "candidate",
        "model",
        "role",
        "status",
        "tool_status",
        "memory_status",
        "code_status",
        "tool_mean",
        "tool_live_mean",
        "memory_em",
        "memory_f1",
        "code_acc",
        "code_tp",
        "code_bon",
        "simple_avg",
        "worst_task",
        "checkpoint",
        "run_id",
        "root",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--candidates",
        default="bcrc_v18_alias_v9,no_behavior_v1_code_only,hard_behavior_v8",
        help="Comma-separated candidate ids. See CANDIDATES in this script.",
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_ids = parse_candidate_list(args.candidates)
    rows = [row_for_candidate(args.root, args.run_id, candidate_id) for candidate_id in candidate_ids]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(rows, args.root, args.run_id), encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.output_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    ready = sum(1 for row in rows if row["status"] == "ready")
    print(f"ready {ready}/{len(rows)} candidates")


if __name__ == "__main__":
    main()
