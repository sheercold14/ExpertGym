#!/usr/bin/env python3
"""Build an ICLR main benchmark draft from existing ExpertGym artifacts.

The script is intentionally artifact-only.  It does not run evaluation, load
models, bake checkpoints, or modify training outputs.  Its job is to make the
current paper-readiness state explicit:

1. full Eval6 baseline rows that are already comparable;
2. RCF-BC mechanism rows that are useful but not full Eval6 rows;
3. completed paper-main BCRC-family rows that bound the claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_REPORT = ROOT / "docs/evaluation/20260518_baselines_eval6.md"
DEFAULT_STATIC_REPORT = ROOT / "docs/evaluation/20260517_p0_static_baselines_eval6.md"
DEFAULT_BEST_REPORT = ROOT / "docs/evaluation/best_ever_model.md"
DEFAULT_PAPER_EVAL_AGGREGATE = ROOT / "docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md"
DEFAULT_RCRF_TABLE = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/"
    "rcrf_paper_evidence_table_20260522/rcrf_paper_evidence_table.csv"
)
DEFAULT_OUTPUT_MD = ROOT / "docs/paper/ExpertGym_ICLR/MAIN_BENCHMARK_TABLE_DRAFT.md"
DEFAULT_OUTPUT_CSV = ROOT / "docs/paper/ExpertGym_ICLR/main_benchmark_table_draft.csv"
DEFAULT_OUTPUT_JSON = ROOT / "docs/paper/ExpertGym_ICLR/main_benchmark_table_draft.json"


FULL_COLUMNS = [
    "model",
    "type",
    "tool_mean",
    "tool_live_mean",
    "memory_em",
    "memory_f1",
    "code_acc",
    "code_tp",
    "code_bon",
    "source",
]

MECHANISM_COLUMNS = [
    "candidate",
    "rule",
    "changed",
    "tool_quick_mean",
    "memory_eval50_f1",
    "lb_hurt_acc",
    "lb_hurt_bon",
    "lcb_hurt_acc",
    "lcb_hurt_bon",
    "source",
]


def _clean_cell(cell: str) -> str:
    cell = cell.strip()
    if cell.startswith("`") and cell.endswith("`"):
        cell = cell[1:-1]
    cell = re.sub(r"\s+", " ", cell)
    return cell


def _parse_markdown_tables(text: str) -> list[list[dict[str, str]]]:
    tables: list[list[dict[str, str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") or "|" not in line[1:]:
            i += 1
            continue
        if i + 1 >= len(lines) or not re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            i += 1
            continue
        headers = [_clean_cell(x) for x in line.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        i += 2
        while i < len(lines) and lines[i].strip().startswith("|"):
            raw = [_clean_cell(x) for x in lines[i].strip().strip("|").split("|")]
            if len(raw) == len(headers):
                rows.append(dict(zip(headers, raw)))
            i += 1
        if rows:
            tables.append(rows)
        continue
    return tables


def _float_or_none(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"pending", "skipped", "nan", "none", "n/a", "-"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _row_from_eval6_table(row: dict[str, str], source: str) -> dict[str, Any] | None:
    required = {"model", "type", "Tool mean", "Tool live mean", "Memory EM", "Memory F1", "Code Acc", "Code TP", "Code BoN(4,4) Acc"}
    if not required.issubset(row):
        return None
    return {
        "model": row["model"],
        "type": row["type"],
        "tool_mean": _float_or_none(row["Tool mean"]),
        "tool_live_mean": _float_or_none(row["Tool live mean"]),
        "memory_em": _float_or_none(row["Memory EM"]),
        "memory_f1": _float_or_none(row["Memory F1"]),
        "code_acc": _float_or_none(row["Code Acc"]),
        "code_tp": _float_or_none(row["Code TP"]),
        "code_bon": _float_or_none(row["Code BoN(4,4) Acc"]),
        "source": source,
    }


def load_full_eval_rows(paths: list[Path]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for table in _parse_markdown_tables(path.read_text(encoding="utf-8")):
            for row in table:
                parsed = _row_from_eval6_table(row, str(path.relative_to(ROOT)))
                if parsed is None:
                    continue
                by_model[parsed["model"]] = parsed
    return list(by_model.values())


def load_best_ever(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    name_match = re.search(r"模型名\s*\n+\s*([^\n]+)", text)
    model = "tame-cg-r1calib-global-v2"
    if name_match:
        model = _clean_cell(name_match.group(1))
    patterns = {
        "tool_mean": r"Tool 均值\s*│\s*([0-9.]+)",
        "tool_live_mean": r"Tool Live 均值\s*│\s*([0-9.]+)",
        "memory_em": r"Memory EM\s*│\s*([0-9.]+)",
        "memory_f1": r"Memory F1\s*│\s*([0-9.]+)",
        "code_acc": r"Code Acc\s*│\s*([0-9.]+)",
        "code_bon": r"Code BoN\(4,4\)\s*│\s*([0-9.]+)",
    }
    row: dict[str, Any] = {
        "model": model,
        "type": "Historical best / TAME-style",
        "code_tp": None,
        "source": str(path.relative_to(ROOT)),
    }
    for key, pat in patterns.items():
        match = re.search(pat, text)
        row[key] = float(match.group(1)) if match else None
    return row


def load_paper_eval6_rows(path: Path) -> list[dict[str, Any]]:
    """Load the frozen BCRC-family full Eval6 rows from the paper aggregate."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for table in _parse_markdown_tables(path.read_text(encoding="utf-8")):
        for row in table:
            required = {"candidate", "role", "status", "Tool", "Tool live", "Memory EM", "Memory F1", "Code Acc", "Code TP", "Code BoN"}
            if not required.issubset(row) or row.get("status") != "ready":
                continue
            rows.append(
                {
                    "model": row["candidate"],
                    "type": row["role"],
                    "tool_mean": _float_or_none(row["Tool"]),
                    "tool_live_mean": _float_or_none(row["Tool live"]),
                    "memory_em": _float_or_none(row["Memory EM"]),
                    "memory_f1": _float_or_none(row["Memory F1"]),
                    "code_acc": _float_or_none(row["Code Acc"]),
                    "code_tp": _float_or_none(row["Code TP"]),
                    "code_bon": _float_or_none(row["Code BoN"]),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    order = ["bcrc_v18_alias_v9", "no_behavior_v1_code_only", "hard_behavior_v8"]
    return sorted(rows, key=lambda r: order.index(r["model"]) if r["model"] in order else 999)


def load_rcrf_mechanism_rows(path: Path) -> list[dict[str, Any]]:
    selected = {"v8", "v18", "v19", "v14", "v15"}
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            short = row.get("short", "")
            if short not in selected:
                continue
            rows.append(
                {
                    "candidate": short,
                    "rule": row.get("method", ""),
                    "changed": _float_or_none(row.get("changed_rows", "")),
                    "tool_quick_mean": _float_or_none(row.get("tool_quick_mean", "")),
                    "memory_eval50_f1": _float_or_none(row.get("memory_eval50_f1", "")),
                    "lb_hurt_acc": _float_or_none(row.get("livebench_hurt_acc", "")),
                    "lb_hurt_bon": _float_or_none(row.get("livebench_hurt_bon_acc", "")),
                    "lcb_hurt_acc": _float_or_none(row.get("livecodebench_hurt_acc", "")),
                    "lcb_hurt_bon": _float_or_none(row.get("livecodebench_hurt_bon_acc", "")),
                    "source": str(path),
                }
            )
    order = ["v18", "v8", "v19", "v14", "v15"]
    return sorted(rows, key=lambda r: order.index(r["candidate"]) if r["candidate"] in order else 999)


def compute_derived(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        numeric = [
            row.get("tool_mean"),
            row.get("memory_f1"),
            row.get("code_acc"),
        ]
        valid = [x for x in numeric if isinstance(x, float)]
        row["simple_avg"] = sum(valid) / len(valid) if len(valid) == 3 else None
        row["worst_task"] = min(valid) if len(valid) == 3 else None


def markdown_table(rows: list[dict[str, Any]], columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def build_report(
    full_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    paper_eval_rows: list[dict[str, Any]],
    best_row: dict[str, Any] | None,
) -> str:
    full_with_best = list(full_rows)
    if best_row:
        full_with_best.append(best_row)
    compute_derived(full_with_best)
    paper_eval_rows = list(paper_eval_rows)
    compute_derived(paper_eval_rows)

    full_columns = [
        "model",
        "type",
        "tool_mean",
        "memory_f1",
        "code_acc",
        "code_bon",
        "simple_avg",
        "worst_task",
        "source",
    ]
    mechanism_columns = [
        "candidate",
        "rule",
        "changed",
        "tool_quick_mean",
        "memory_eval50_f1",
        "lb_hurt_acc",
        "lb_hurt_bon",
        "lcb_hurt_acc",
        "lcb_hurt_bon",
    ]
    selected_baselines = [
        "Qwen2.5-7B-Instruct",
        "task-arithmetic-c033333",
        "ta-c075-global-20260517",
        "RAM-Merged ARM-R-v2",
        "wudi-qwen7b-3expert",
        "ties-c033333-k02",
        "dare-ta-c033333-d08",
        "dare-ties-c033333-k02-d08",
        "adamerging-taskwise-len1024",
        "mixture-grpo-ta13-l96-step1",
        "tame-cg-r1calib-global-v2",
    ]
    rows_by_name = {row["model"]: row for row in full_with_best}
    main_rows = [rows_by_name[name] for name in selected_baselines if name in rows_by_name]

    open_items = [
        {
            "item": "Broad SOTA claim",
            "status": "not supported",
            "action": "Keep the claim narrowed to mechanism and trade-off control; BCRC is below TA-0.75 / TAME-style on Code and average score.",
        },
        {
            "item": "ToolRL-80 in main table",
            "status": "optional",
            "action": "Either omit ToolRL from the main table or report it as an auxiliary source-distribution stability check.",
        },
        {
            "item": "RAM artifacts",
            "status": "discussion-only unless added",
            "action": "Add RAM artifact rows only if the RAIN/RAM comparison becomes an empirical claim.",
        },
    ]

    text = []
    text.append("# ICLR Main Benchmark Table Draft\n")
    text.append("This draft is generated from existing artifacts only. It does not run new evaluation.\n")
    text.append("## Comparable Full Eval6 Rows\n")
    text.append(
        "These rows share the full Eval6-style Tool / Memory / Code protocol recorded in "
        "`docs/evaluation/20260518_baselines_eval6.md` and `docs/evaluation/20260517_p0_static_baselines_eval6.md`.\n"
    )
    text.append(markdown_table(main_rows, full_columns, ["model", "type", "Tool", "Memory F1", "Code Acc", "Code BoN", "Avg(T/M/C)", "Worst", "source"]))
    text.append("\n## Paper-Main BCRC-Family Full Eval6 Rows\n")
    text.append(
        "The selected BCRC-family queue is complete.  These rows are comparable within the paper-main ablation block and bound the claim.\n"
    )
    text.append(markdown_table(paper_eval_rows, full_columns, ["model", "type", "Tool", "Memory F1", "Code Acc", "Code BoN", "Avg(T/M/C)", "Worst", "source"]))
    text.append("\n## RCF-BC Mechanism Rows\n")
    text.append(
        "These rows are mechanism evidence, not full Eval6 rows. Tool/Memory are quick metrics and Code is the code-hurt diagnostic subset.\n"
    )
    text.append(markdown_table(mechanism_rows, mechanism_columns, ["candidate", "rule", "changed", "Tool quick", "Mem eval50", "LB hurt acc", "LB hurt BoN", "LCB hurt acc", "LCB hurt BoN"]))
    text.append("\n## Claim Boundary / Open Items\n")
    text.append(
        "The paper-main queue has the required BCRC, no-behavior, and hard-behavior rows.  The remaining issue is claim framing, not missing selected Eval6 rows.\n"
    )
    text.append(markdown_table(open_items, ["item", "status", "action"], ["item", "status", "action"]))
    text.append("\n## Recommended Paper-Main Evaluation Set\n")
    text.append(
        "| role | candidate |\n| --- | --- |\n"
        "| main method | v18 / RCF-BC soft behavior constraints |\n"
        "| no behavior constraint | v1 or explicit code-field-only gate |\n"
        "| hard behavior constraint | v8 memoryfull hard veto or v19 strict cleanup |\n"
        "| scalar negative control | v14 code-half and/or v15 code-zero |\n"
        "| static baseline anchor | TA-0.75, DARE-TIES, AdaMerging taskwise, RAM-Merged ARM-R-v2 |\n"
    )
    return "\n".join(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--static-report", type=Path, default=DEFAULT_STATIC_REPORT)
    parser.add_argument("--best-report", type=Path, default=DEFAULT_BEST_REPORT)
    parser.add_argument("--paper-eval-aggregate", type=Path, default=DEFAULT_PAPER_EVAL_AGGREGATE)
    parser.add_argument("--rcrf-table", type=Path, default=DEFAULT_RCRF_TABLE)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_rows = load_full_eval_rows([args.baseline_report, args.static_report])
    best_row = load_best_ever(args.best_report)
    paper_eval_rows = load_paper_eval6_rows(args.paper_eval_aggregate)
    mechanism_rows = load_rcrf_mechanism_rows(args.rcrf_table)

    full_for_csv = list(full_rows)
    if best_row:
        full_for_csv.append(best_row)
    full_for_csv.extend(paper_eval_rows)
    compute_derived(full_for_csv)
    csv_columns = FULL_COLUMNS + ["simple_avg", "worst_task"]
    write_csv(args.output_csv, full_for_csv, csv_columns)

    payload = {
        "full_eval_rows": full_for_csv,
        "paper_eval_rows": paper_eval_rows,
        "rcrf_mechanism_rows": mechanism_rows,
        "sources": {
            "baseline_report": str(args.baseline_report),
            "static_report": str(args.static_report),
            "best_report": str(args.best_report),
            "paper_eval_aggregate": str(args.paper_eval_aggregate),
            "rcrf_table": str(args.rcrf_table),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_report(full_rows, mechanism_rows, paper_eval_rows, best_row), encoding="utf-8")

    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
