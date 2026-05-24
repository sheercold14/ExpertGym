#!/usr/bin/env python3
"""Build a paper-facing evidence table for RCF-BC candidates.

The table is assembled from existing evaluation artifacts.  It does not run
evaluation, bake checkpoints, or change any gate.  Its purpose is to keep the
RCF-BC mechanism story auditable from raw files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RCRF_ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf")
SUBSET_ROOT = RCRF_ROOT / "code_hurt_subset_20260521"
GATE_ROOT = SUBSET_ROOT / "contrast_gates"
FULL_EVAL_ROOT = RCRF_ROOT / "eval" / "full_suite"
MEMORY_ROOT = Path("/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory")
CURE_RESULTS = Path("/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results")
BASE_GATES = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_OUTPUT_DIR = SUBSET_ROOT / "analysis" / "rcrf_paper_evidence_table_20260522"
DEFAULT_DOC_REPORT = REPO_ROOT / "docs" / "report" / "RCRF" / "20260522_rcf_bc_paper_evidence_table.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    short: str
    checkpoint: str
    gate_dir: str
    eval_id: str
    method: str
    eval_alias: str = ""
    code_checkpoint: str = ""
    memory_summary: str = ""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_gates = load_gates(BASE_GATES)
    rows = [build_candidate_row(candidate, base_gates) for candidate in CANDIDATES]
    write_csv(output_dir / "rcrf_paper_evidence_table.csv", rows)
    write_json(output_dir / "rcrf_paper_evidence_table.json", rows)
    report = render_markdown(rows, output_dir)
    (output_dir / "rcrf_paper_evidence_table.md").write_text(report, encoding="utf-8")
    if args.doc_report:
        doc_report = Path(args.doc_report).expanduser().resolve()
        doc_report.parent.mkdir(parents=True, exist_ok=True)
        doc_report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(rows),
                "csv": str(output_dir / "rcrf_paper_evidence_table.csv"),
                "doc_report": str(Path(args.doc_report).expanduser().resolve()) if args.doc_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    return parser.parse_args()


CANDIDATES = [
    Candidate(
        name="v8_memoryfull_hard",
        short="v8",
        checkpoint="rcrf_code_spanaware_tmpos_s32_memoryfull_v8",
        gate_dir="rcrf_code_spanaware_tmpos_s32_memoryfull_v8",
        eval_id="rcrf_code_spanaware_tmpos_s32_memoryfull_v8",
        method="Code field + Tool/Memory hard behavior veto",
    ),
    Candidate(
        name="v9_soft_rcf_bc",
        short="v9",
        checkpoint="rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9",
        gate_dir="rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9",
        eval_id="rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9",
        method="Continuous Code field + soft Tool/Memory behavior constraint",
    ),
    Candidate(
        name="v10_ratio",
        short="v10",
        checkpoint="rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10",
        gate_dir="rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10",
        eval_id="rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10",
        method="Naive evidence-ratio veto",
    ),
    Candidate(
        name="v11_tasktyped",
        short="v11",
        checkpoint="rcrf_code_spanaware_tmpos_s32_tasktyped_v11",
        gate_dir="rcrf_code_spanaware_tmpos_s32_tasktyped_v11",
        eval_id="rcrf_code_spanaware_tmpos_s32_tasktyped_v11",
        method="Tool hard, Memory soft behavior constraint",
    ),
    Candidate(
        name="v12_role_routed",
        short="v12",
        checkpoint="rcrf_role_routed_v12",
        gate_dir="rcrf_role_routed_v12",
        eval_id="rcrf_role_routed_v12",
        method="Hard atlas role routing",
    ),
    Candidate(
        name="v13_positive_only_role",
        short="v13",
        checkpoint="rcrf_role_routed_positive_only_v13",
        gate_dir="rcrf_role_routed_positive_only_v13",
        eval_id="rcrf_role_routed_positive_only_v13",
        method="Positive-only role routing",
    ),
    Candidate(
        name="v14_code_half",
        short="v14",
        checkpoint="rcrf_v9_code_half_v14",
        gate_dir="rcrf_v9_code_half_v14",
        eval_id="rcrf_v9_code_half_v14",
        method="Negative control: v9 with code expert coefficients halved",
    ),
    Candidate(
        name="v15_code_zero",
        short="v15",
        checkpoint="rcrf_v9_code_zero_v15",
        gate_dir="rcrf_v9_code_zero_v15",
        eval_id="rcrf_v9_code_zero_v15",
        method="Negative control: v9 with code expert coefficients set to zero",
    ),
    Candidate(
        name="v16_source_suppress",
        short="v16",
        checkpoint="rcrf_source_conflict_suppress_v16",
        gate_dir="rcrf_source_conflict_suppress_v16",
        eval_id="rcrf_source_conflict_suppress_v16",
        method="Source-conflict dominant-negative suppression",
    ),
    Candidate(
        name="v17_source_route",
        short="v17",
        checkpoint="rcrf_source_conflict_route_v17",
        gate_dir="rcrf_source_conflict_route_v17",
        eval_id="rcrf_source_conflict_route_v17",
        method="Source-conflict dominant routing",
    ),
    Candidate(
        name="v18_rcf_bc",
        short="v18",
        checkpoint="residual_capability_field_behavior_constraints_v18",
        gate_dir="residual_capability_field_behavior_constraints_v18",
        eval_id="rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9",
        eval_alias="v9_soft_rcf_bc",
        code_checkpoint="rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9",
        method="Main method name for v9: RCF-BC",
    ),
    Candidate(
        name="v19_archetype_consistency",
        short="v19",
        checkpoint="rcrf_archetype_consistency_v19",
        gate_dir="rcrf_archetype_consistency_v19",
        eval_id="rcrf_archetype_consistency_v19",
        method="Ablation: strict archetype-consistency projection",
    ),
    Candidate(
        name="v20_code_noise_half",
        short="v20",
        checkpoint="rcrf_code_noise_weak_half_v20",
        gate_dir="rcrf_code_noise_weak_half_v20",
        eval_id="rcrf_code_noise_weak_half_v20",
        method="Residual-level ablation: halve code negative-noise and weak rows",
    ),
    Candidate(
        name="v21_code_noise_zero",
        short="v21",
        checkpoint="rcrf_code_noise_weak_zero_v21",
        gate_dir="rcrf_code_noise_weak_zero_v21",
        eval_id="rcrf_code_noise_weak_zero_v21",
        method="Residual-level ablation: zero code negative-noise and weak rows",
    ),
    Candidate(
        name="v22_code_negative_noise_half",
        short="v22",
        checkpoint="rcrf_code_negative_noise_half_v22",
        gate_dir="rcrf_code_negative_noise_half_v22",
        eval_id="rcrf_code_negative_noise_half_v22",
        method="Residual-level ablation: halve only code negative-noise rows",
    ),
    Candidate(
        name="v23_code_weak_half",
        short="v23",
        checkpoint="rcrf_code_weak_half_v23",
        gate_dir="rcrf_code_weak_half_v23",
        eval_id="rcrf_code_weak_half_v23",
        method="Residual-level ablation: halve only weak code rows",
    ),
]


def build_candidate_row(candidate: Candidate, base_gates: dict[str, float]) -> dict[str, Any]:
    gate_path = GATE_ROOT / candidate.gate_dir / "gates.json"
    gates = load_gates(gate_path)
    gate_stats = summarize_gate_delta(base_gates, gates)
    tool_scores, tool_log = parse_tool_scores(candidate.eval_id)
    memory_scores, memory_path = parse_memory_scores(candidate)
    code_checkpoint = candidate.code_checkpoint or candidate.checkpoint
    livebench_scores, livebench_path = parse_code_result(code_checkpoint, "LiveBenchCodeHurtRcrfVsTa16")
    livecodebench_scores, livecodebench_path = parse_code_result(code_checkpoint, "LiveCodeBenchCodeHurtRcrfVsTa16")
    row = {
        "candidate": candidate.name,
        "short": candidate.short,
        "method": candidate.method,
        "eval_alias": candidate.eval_alias,
        "checkpoint": candidate.checkpoint,
        "gate_path": str(gate_path),
        "changed_rows": gate_stats["changed_count"],
        "positive_rows": gate_stats["positive_count"],
        "negative_rows": gate_stats["negative_count"],
        "mean_abs_delta": gate_stats["mean_abs_delta"],
        "tool_parallel": tool_scores.get("parallel"),
        "tool_parallel_multiple": tool_scores.get("parallel_multiple"),
        "tool_live_parallel": tool_scores.get("live_parallel"),
        "tool_live_parallel_multiple": tool_scores.get("live_parallel_multiple"),
        "tool_quick_mean": mean_optional(
            [
                tool_scores.get("parallel"),
                tool_scores.get("parallel_multiple"),
                tool_scores.get("live_parallel"),
                tool_scores.get("live_parallel_multiple"),
            ]
        ),
        "memory_eval50_f1": memory_scores.get("avg_f1"),
        "memory_eval50_em": memory_scores.get("exact_match_rate"),
        "memory_eval50_sub_em": memory_scores.get("sub_exact_match_rate"),
        "livebench_hurt_acc": livebench_scores.get("code_acc"),
        "livebench_hurt_bon_acc": livebench_scores.get("bon_acc"),
        "livebench_hurt_accum": livebench_scores.get("code_accumulate_acc"),
        "livebench_hurt_bon_accum": livebench_scores.get("bon_accumulate_acc"),
        "livecodebench_hurt_acc": livecodebench_scores.get("code_acc"),
        "livecodebench_hurt_bon_acc": livecodebench_scores.get("bon_acc"),
        "livecodebench_hurt_accum": livecodebench_scores.get("code_accumulate_acc"),
        "livecodebench_hurt_bon_accum": livecodebench_scores.get("bon_accumulate_acc"),
        "code_hurt_acc_mean": mean_optional([livebench_scores.get("code_acc"), livecodebench_scores.get("code_acc")]),
        "code_hurt_bon_mean": mean_optional([livebench_scores.get("bon_acc"), livecodebench_scores.get("bon_acc")]),
        "tool_log": str(tool_log) if tool_log else "",
        "memory_summary": str(memory_path) if memory_path else "",
        "livebench_result": str(livebench_path) if livebench_path else "",
        "livecodebench_result": str(livecodebench_path) if livecodebench_path else "",
    }
    return row


def parse_tool_scores(eval_id: str) -> tuple[dict[str, float], Path | None]:
    path = FULL_EVAL_ROOT / eval_id / "quick_tool_memory" / "logs" / "tool_bfcl.log"
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Test completed: ([A-Za-z0-9_]+).*?Accuracy: ([0-9.]+)%", text)
    scores = {name: float(value) / 100.0 for name, value in matches}
    if scores:
        return scores, path
    json_scores = parse_last_json_scores(text)
    return json_scores, path


def parse_last_json_scores(text: str) -> dict[str, float]:
    marker = '"scores"'
    index = text.rfind(marker)
    if index < 0:
        return {}
    start = text.rfind("{", 0, index)
    if start < 0:
        return {}
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    scores = payload.get("scores", {})
    return {name: float(stats.get("accuracy")) for name, stats in scores.items() if isinstance(stats, dict)}


def parse_memory_scores(candidate: Candidate) -> tuple[dict[str, float], Path | None]:
    if candidate.memory_summary:
        path = Path(candidate.memory_summary)
    else:
        path = MEMORY_ROOT / "rcrf-memory" / candidate.eval_id / "quick_tool_memory" / "eval_50" / "evaluation_summary.json"
    if not path.exists():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("hotpotqa", payload)
    return {
        "avg_f1": safe_float(stats.get("avg_f1")),
        "exact_match_rate": safe_float(stats.get("exact_match_rate")),
        "sub_exact_match_rate": safe_float(stats.get("sub_exact_match_rate")),
    }, path


def parse_code_result(checkpoint: str, dataset: str) -> tuple[dict[str, float], Path | None]:
    path = CURE_RESULTS / f"results-eval-.tmp.shared-storage.OnPolicy.checkpoints.{checkpoint}-{dataset}.txt"
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "code_acc": regex_float(text, r"code acc .*?: ([0-9.]+)"),
        "code_accumulate_acc": regex_float(text, r"code accumulate acc .*?: ([0-9.]+)"),
        "bon_acc": regex_float(text, r"BoN setting .*?:\s*acc: ([0-9.]+)", flags=re.S),
        "bon_accumulate_acc": regex_float(text, r"BoN setting .*?:\s*acc: [0-9.]+, accumulate acc: ([0-9.]+)", flags=re.S),
    }, path


def summarize_gate_delta(base_gates: dict[str, float], gates: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(base_gates) & set(gates))
    deltas = [float(gates[key]) - float(base_gates[key]) for key in common]
    changed = [value for value in deltas if abs(value) > 1e-12]
    return {
        "count": len(common),
        "changed_count": len(changed),
        "positive_count": sum(1 for value in changed if value > 0.0),
        "negative_count": sum(1 for value in changed if value < 0.0),
        "mean_abs_delta": mean(abs(value) for value in deltas) if deltas else None,
    }


def load_gates(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("gates", payload)
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def regex_float(text: str, pattern: str, *, flags: int = 0) -> float | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return float(match.group(1))


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_optional(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# 2026-05-22 RCF-BC Paper Evidence Table",
        "",
        "## Purpose",
        "",
        "This table consolidates the RCF-BC mechanism candidates from raw gate and evaluation artifacts. "
        "It is the paper-facing evidence table for the claim that continuous residual capability fields are needed, "
        "while behavior evidence should act as constraints rather than hard routing rules.",
        "",
        "## Main Table",
        "",
        "| candidate | rule | changed | Tool mean | Memory F1 | LB hurt acc/BoN | LCB hurt acc/BoN | conclusion |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['short']}` | {row['method']} | {row['changed_rows']} | "
            f"{fmt(row['tool_quick_mean'])} | {fmt(row['memory_eval50_f1'])} | "
            f"{fmt(row['livebench_hurt_acc'])} / {fmt(row['livebench_hurt_bon_acc'])} | "
            f"{fmt(row['livecodebench_hurt_acc'])} / {fmt(row['livecodebench_hurt_bon_acc'])} | "
            f"{conclusion(row)} |"
        )
    lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            "- `v18_rcf_bc` is numerically identical to `v9`, but should be used as the semantic method name.",
            "- Global Code shrinkage (`v14/v15`) improves Memory but destroys Code, so task-level scalar control is too coarse.",
            "- Hard or semi-hard routing (`v13/v16/v17`) keeps Tool/Memory but loses Code, so discrete atlas roles are insufficient.",
            "- Strict archetype cleanup (`v19`) improves Memory and keeps Tool, but loses Code. This shows Code depends on low-confidence continuous residual deltas.",
            "- Residual-level Code shrinkage (`v20/v21`) recovers some Memory with far fewer changed rows than global shrinkage, but still hurts parts of Code. `v22/v23` split whether this comes from negative-noise rows or weak evidence rows.",
            "- The main paper story should therefore stay with `v18_rcf_bc`: continuous capability field plus behavior constraints.",
            "",
            "## Raw Artifacts",
            "",
            f"- CSV: `{output_dir / 'rcrf_paper_evidence_table.csv'}`",
            f"- JSON: `{output_dir / 'rcrf_paper_evidence_table.json'}`",
            "",
            "## Per-Candidate Provenance",
            "",
            "| candidate | gate | Tool log | Memory summary | LB result | LCB result |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['short']}` | `{row['gate_path']}` | `{row['tool_log']}` | "
            f"`{row['memory_summary']}` | `{row['livebench_result']}` | `{row['livecodebench_result']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


def conclusion(row: dict[str, Any]) -> str:
    short = row["short"]
    if short in {"v9", "v18"}:
        return "main operating point: best Code/behavior trade-off"
    if short == "v19":
        return "negative ablation: cleaner behavior, loses Code field"
    if short in {"v20", "v21", "v22", "v23"}:
        return "diagnostic: residual-level shrink is more localized but labels are incomplete"
    if short in {"v14", "v15"}:
        return "negative control: scalar Code shrinkage is too coarse"
    if short in {"v13", "v16", "v17"}:
        return "negative control: hard routing loses continuous Code signal"
    if short == "v8":
        return "behavior-safe but over-suppresses Code"
    return "mechanism ablation"


if __name__ == "__main__":
    main()
