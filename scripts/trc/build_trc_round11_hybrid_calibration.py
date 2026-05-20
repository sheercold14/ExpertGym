#!/usr/bin/env python3
"""Build the Round11 TRC hybrid calibration bank.

This is a data-only builder for the 2026-05-20 hybrid calibration pass. It
keeps the stable late3 Tool/Memory rows from Round8 and mixes Code rows from
Round10 tag quotas with a small RF-only CodeP0 supplement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.trc.build_trc_calibration_v1 import load_positive_candidates, materialize_row


DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round11_hybrid_v1/"
    "r11b_r11g_r10tag24_rf8_stablelate3"
)
DEFAULT_STABLE_TM_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/"
    "rf_only_late3/trc96_expert_trajectories.jsonl"
)
DEFAULT_R10_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round10_codep0_tag_v1/"
    "tag_quota_default_late3/trc96_expert_trajectories.jsonl"
)
DEFAULT_R8D_BANK = DEFAULT_STABLE_TM_BANK
DEFAULT_R8B_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/"
    "rf_then_ds_late3/trc96_expert_trajectories.jsonl"
)
DEFAULT_RF_RAW_ROLLOUT = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/"
    "code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl"
)

R10_EXTRA_DROP_PROMPT_ID = "code_p0v3__3175ad9d3d49acec"
RF_SUPPLEMENT_SPECS = [
    (
        "code_p0v3__e08aa7ed80a06eb9",
        "generation dynamic_programming multi-tag; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__e9f843a1eb08e58e",
        "generation dynamic_programming/greedy multi-tag; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__a5e6e2381fb1732e",
        "generation graph multi-tag; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__cbe4ee0799da565e",
        "frontier graph; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__702801cee1f4495b",
        "frontier graph; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__63718bf5a80920f3",
        "frontier greedy; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__d0fd7fc20a7cda49",
        "frontier greedy; use RF-only alternate successful sample",
    ),
    (
        "code_p0v3__fda4b9ef7fc21cb5",
        "R8D RF-only stable format row absent from R10; syntax-pass code-block-friendly sample",
    ),
]
STOP_TAGS = {"format_sensitive", "stdin_stdout"}
LEAKAGE_MARKERS = ("LiveBench", "LiveCodeBench", "CURE/data")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_rows = read_jsonl(Path(args.stable_tm_bank).expanduser())
    r10_rows = read_jsonl(Path(args.r10_bank).expanduser())
    tool_rows = select_task_rows(stable_rows, "tool", expected=32)
    memory_rows = select_task_rows(stable_rows, "memory", expected=32)
    r10_code_rows = select_task_rows(r10_rows, "code", expected=32)

    r10_by_prompt = {row["prompt_id"]: row for row in r10_code_rows}
    supplement_rows, supplement_details = build_rf_supplement_rows(
        rf_raw_rollout=Path(args.rf_raw_rollout).expanduser(),
        r10_by_prompt=r10_by_prompt,
    )
    supplement_prompt_ids = {row["prompt_id"] for row in supplement_rows}
    r10_excluded_prompt_ids = (supplement_prompt_ids & set(r10_by_prompt)) | {R10_EXTRA_DROP_PROMPT_ID}
    r10_core_rows = [row for row in r10_code_rows if row["prompt_id"] not in r10_excluded_prompt_ids]
    if len(r10_core_rows) != 24:
        raise RuntimeError(f"Expected 24 R10 core Code rows, got {len(r10_core_rows)}")

    rows: list[dict[str, Any]] = []
    rows.extend(with_provenance(row, component="stable_late3_tool", source_bank="r8d_rf_only_late3") for row in tool_rows)
    rows.extend(
        with_provenance(row, component="stable_late3_memory", source_bank="r8d_rf_only_late3") for row in memory_rows
    )
    rows.extend(with_provenance(row, component="r10_tag_quota_core", source_bank="r10_tag_quota") for row in r10_core_rows)
    for row in supplement_rows:
        rows.append(with_provenance(row, component="rf_only_supplement", source_bank="code_p0_v3_rf_raw"))
    renumber_trajectory_ids(rows)

    validate_final_rows(rows)
    out_jsonl = output_dir / "trc96_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, rows)

    summary = build_summary(
        rows=rows,
        out_jsonl=out_jsonl,
        args=args,
        r10_excluded_prompt_ids=sorted(r10_excluded_prompt_ids),
        r10_core_rows=r10_core_rows,
        supplement_rows=supplement_rows,
        supplement_details=supplement_details,
    )
    write_json(output_dir / "trc96_summary.json", summary)
    md = render_markdown(summary)
    (output_dir / "trc96_summary.md").write_text(md, encoding="utf-8")
    (output_dir / "README.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_rf_supplement_rows(*, rf_raw_rollout: Path, r10_by_prompt: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, _stats = load_positive_candidates(
        task="code",
        expert="code",
        sources=[("code_source_00", rf_raw_rollout)],
        positive_threshold=1.0,
        include_sample_expert=[],
        exclude_sample_expert=[],
        memory_response_source="final",
        memory_trajectory_max_update_turns=0,
        memory_trajectory_turn_policy="uniform",
        memory_trajectory_include_final_turn=True,
    )
    candidates_by_prompt: dict[str, list[Any]] = {}
    for candidate in candidates:
        candidates_by_prompt.setdefault(candidate.prompt_id, []).append(candidate)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    used_samples: set[str] = set()
    for prompt_id, reason in RF_SUPPLEMENT_SPECS:
        prompt_candidates = candidates_by_prompt.get(prompt_id) or []
        if not prompt_candidates:
            raise RuntimeError(f"No positive RF-only candidate found for supplement prompt {prompt_id}")
        r10_sample_id = str((r10_by_prompt.get(prompt_id) or {}).get("sample_id") or "")
        ordered = sorted(
            prompt_candidates,
            key=lambda item: (
                item.sample_id == r10_sample_id,
                item.sample_id in used_samples,
                item.length,
                item.sample_index,
                item.sample_id,
            ),
        )
        candidate = ordered[0]
        used_samples.add(candidate.sample_id)
        row = materialize_row(candidate, rank=0)
        row_metadata = dict(row.get("row_metadata") or {})
        row_metadata["round11_rf_supplement"] = {
            "selection_reason": reason,
            "r10_replaced_sample_id": r10_sample_id or None,
            "alternate_to_r10_sample": bool(r10_sample_id and candidate.sample_id != r10_sample_id),
        }
        row["row_metadata"] = row_metadata
        rows.append(row)
        details.append(
            {
                "prompt_id": prompt_id,
                "sample_id": candidate.sample_id,
                "r10_replaced_sample_id": r10_sample_id or None,
                "alternate_to_r10_sample": bool(r10_sample_id and candidate.sample_id != r10_sample_id),
                "role": code_role(row),
                "primary_tag": primary_code_tag(row),
                "tags": code_tags(row),
                "length": row.get("length"),
                "reason": reason,
            }
        )
    return rows, details


def build_summary(
    *,
    rows: list[dict[str, Any]],
    out_jsonl: Path,
    args: argparse.Namespace,
    r10_excluded_prompt_ids: list[str],
    r10_core_rows: list[dict[str, Any]],
    supplement_rows: list[dict[str, Any]],
    supplement_details: list[dict[str, Any]],
) -> dict[str, Any]:
    input_banks = {
        "stable_tool_memory_late3": str(Path(args.stable_tm_bank).expanduser().resolve()),
        "r10_tag_quota": str(Path(args.r10_bank).expanduser().resolve()),
        "r8d_rf_only_late3": str(Path(args.r8d_bank).expanduser().resolve()),
        "r8b_rf_then_ds_late3": str(Path(args.r8b_bank).expanduser().resolve()),
        "rf_raw_rollout": str(Path(args.rf_raw_rollout).expanduser().resolve()),
    }
    return {
        "format": "trc_round11_hybrid_calibration_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(out_jsonl),
        "num_rows": len(rows),
        "task_counts": count_by(rows, "task"),
        "unique_prompt_counts": unique_prompt_counts(rows),
        "duplicate_prompt_rows": duplicate_prompt_rows(rows),
        "input_banks": input_banks,
        "construction_policy": {
            "tool": "32 rows copied from Round8 RF-only late3 stable Tool rows; same rows as R8B/R5 stable late3.",
            "memory": "32 rows copied from Round8 RF-only late3 stable Memory trajectory-turn rows; late3 updates plus final.",
            "code": (
                "24 R10 tag-quota rows plus 8 RF-only CodeP0-v3 supplement rows. "
                f"R10 prompt {R10_EXTRA_DROP_PROMPT_ID} is dropped to replace one long DS greedy row with an RF-only stable row."
            ),
            "leakage_policy": "Code sources are CodeContests_train CodeP0-v3 expert successes; no LiveBench/LiveCodeBench/CURE hidden rows are used.",
        },
        "bank_statistics": {
            "r10_tag_quota": bank_stats(Path(args.r10_bank).expanduser()),
            "r8d_rf_only_late3": bank_stats(Path(args.r8d_bank).expanduser()),
            "r8b_rf_then_ds_late3": bank_stats(Path(args.r8b_bank).expanduser()),
            "stable_tool_memory_late3": bank_stats(Path(args.stable_tm_bank).expanduser(), tasks=("tool", "memory")),
        },
        "final_statistics": stats_for_rows(rows),
        "code_selection": {
            "r10_core_rows": len(r10_core_rows),
            "rf_only_supplement_rows": len(supplement_rows),
            "r10_excluded_prompt_ids": r10_excluded_prompt_ids,
            "r10_extra_drop_prompt_id": R10_EXTRA_DROP_PROMPT_ID,
            "supplement": supplement_details,
            "r10_core_distribution": code_distribution(r10_core_rows),
            "rf_supplement_distribution": code_distribution(supplement_rows),
            "final_code_distribution": code_distribution([row for row in rows if row.get("task") == "code"]),
        },
        "quality_checks": quality_checks(rows),
    }


def bank_stats(path: Path, *, tasks: tuple[str, ...] = ("tool", "memory", "code")) -> dict[str, Any]:
    rows = [row for row in read_jsonl(path) if str(row.get("task")) in tasks]
    return {
        "path": str(path.resolve()),
        **stats_for_rows(rows),
    }


def stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_counts": count_by(rows, "task"),
        "unique_prompt_counts": unique_prompt_counts(rows),
        "duplicate_prompt_rows": duplicate_prompt_rows(rows),
        "source_name_counts": nested_count(rows, "task", lambda row: str(row.get("source_name") or "unknown")),
        "source_path_counts": nested_count(rows, "task", lambda row: Path(str(row.get("source_path") or "unknown")).name),
        "expert_source_counts": nested_count(rows, "task", expert_source),
        "code_distribution": code_distribution([row for row in rows if row.get("task") == "code"]),
        "reward_train": reward_stats(rows),
    }


def code_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_prompts": len({row.get("prompt_id") for row in rows}),
        "duplicate_prompt_rows": sum(count - 1 for count in Counter(str(row.get("prompt_id")) for row in rows).values()),
        "source_name_counts": dict(sorted(Counter(str(row.get("source_name") or "unknown") for row in rows).items())),
        "source_path_counts": dict(sorted(Counter(Path(str(row.get("source_path") or "unknown")).name for row in rows).items())),
        "expert_source_counts": dict(sorted(Counter(expert_source(row) for row in rows).items())),
        "source_dataset_counts": dict(sorted(Counter(str(code_metadata(row).get("source_dataset") or "unknown") for row in rows).items())),
        "role_counts": dict(sorted(Counter(code_role(row) for row in rows).items())),
        "primary_tag_counts": dict(sorted(Counter(primary_code_tag(row) for row in rows).items())),
        "tag_counts": dict(sorted(Counter(tag for row in rows for tag in code_tags(row)).items())),
    }


def quality_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("trajectory_id") or "") for row in rows]
    blank_response = [row.get("trajectory_id") for row in rows if not str(row.get("response") or "").strip()]
    nonpositive = [row.get("trajectory_id") for row in rows if float(row.get("reward_train") or 0.0) < 1.0]
    code_rows = [row for row in rows if row.get("task") == "code"]
    leakage_hits = []
    for row in code_rows:
        haystack = json.dumps(
            {
                "prompt_id": row.get("prompt_id"),
                "source_path": row.get("source_path"),
                "reference_metadata": code_metadata(row),
                "row_metadata": row.get("row_metadata"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if any(marker in haystack for marker in LEAKAGE_MARKERS):
            leakage_hits.append(row.get("trajectory_id"))
    details_rows = [((row.get("sample_metadata") or {}).get("details") or {}) for row in code_rows]
    code_present_false = sum(1 for details in details_rows if details.get("code_present") is False)
    syntax_false = sum(1 for details in details_rows if details.get("syntax_ok") is False)
    return {
        "row_count_ok": len(rows) == 96,
        "task_balance_ok": count_by(rows, "task") == {"code": 32, "memory": 32, "tool": 32},
        "trajectory_id_unique_ok": len(ids) == len(set(ids)),
        "blank_response_count": len(blank_response),
        "nonpositive_reward_train_count": len(nonpositive),
        "code_leakage_marker_hits": leakage_hits,
        "code_leakage_marker_ok": not leakage_hits,
        "code_present_false_count": code_present_false,
        "syntax_false_count": syntax_false,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    final = summary["final_statistics"]
    code = summary["code_selection"]["final_code_distribution"]
    quality = summary["quality_checks"]
    bank_rows = []
    for name, stats in summary["bank_statistics"].items():
        bank_rows.append(
            "| {name} | {rows} | {unique} | {sources} |".format(
                name=name,
                rows=json.dumps(stats["row_counts"], sort_keys=True),
                unique=json.dumps(stats["unique_prompt_counts"], sort_keys=True),
                sources=json.dumps(stats["source_path_counts"], sort_keys=True),
            )
        )
    supplement_rows = []
    for item in summary["code_selection"]["supplement"]:
        supplement_rows.append(
            "| {prompt} | {sample} | {role} | {primary} | {alt} |".format(
                prompt=item["prompt_id"],
                sample=item["sample_id"],
                role=item["role"],
                primary=item["primary_tag"],
                alt=item["alternate_to_r10_sample"],
            )
        )
    return "\n".join(
        [
            "# TRC Round11 Hybrid Calibration",
            "",
            f"- Output: `{summary['output']}`",
            f"- Rows: `{summary['num_rows']}`",
            f"- Task counts: `{json.dumps(summary['task_counts'], sort_keys=True)}`",
            f"- Unique prompts: `{json.dumps(summary['unique_prompt_counts'], sort_keys=True)}`",
            "",
            "## Construction",
            "",
            "- Tool: stable Round8 late3 Tool rows, 32 rows.",
            "- Memory: stable Round8 late3 Memory trajectory rows, 32 rows.",
            "- Code: 24 R10 tag-quota rows plus 8 RF-only CodeP0-v3 supplement rows.",
            f"- Dropped R10 extra prompt: `{summary['code_selection']['r10_extra_drop_prompt_id']}`.",
            "- Leakage policy: CodeContests_train CodeP0-v3 only; no LiveBench/LiveCodeBench/CURE hidden rows.",
            "",
            "## Bank Statistics",
            "",
            "| bank | row counts | unique prompts | source paths |",
            "|---|---:|---:|---|",
            *bank_rows,
            "",
            "## Final Code Distribution",
            "",
            f"- Source paths: `{json.dumps(code['source_path_counts'], sort_keys=True)}`",
            f"- Roles: `{json.dumps(code['role_counts'], sort_keys=True)}`",
            f"- Primary tags: `{json.dumps(code['primary_tag_counts'], sort_keys=True)}`",
            f"- Tags: `{json.dumps(code['tag_counts'], sort_keys=True)}`",
            "",
            "## RF Supplement",
            "",
            "| prompt | sample | role | primary tag | alternate to R10 |",
            "|---|---|---|---|---:|",
            *supplement_rows,
            "",
            "## Quality Checks",
            "",
            f"- Final source names: `{json.dumps(final['source_name_counts'], sort_keys=True)}`",
            f"- Reward train: `{json.dumps(final['reward_train'], sort_keys=True)}`",
            f"- Checks: `{json.dumps(quality, sort_keys=True)}`",
            "",
        ]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_task_rows(rows: list[dict[str, Any]], task: str, *, expected: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("task") == task]
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} {task} rows, got {len(selected)}")
    return selected


def with_provenance(row: dict[str, Any], *, component: str, source_bank: str) -> dict[str, Any]:
    copied = deepcopy(row)
    row_metadata = dict(copied.get("row_metadata") or {})
    row_metadata["round11_hybrid"] = {
        "component": component,
        "source_bank": source_bank,
        "source_trajectory_id": copied.get("trajectory_id"),
    }
    copied["row_metadata"] = row_metadata
    return copied


def renumber_trajectory_ids(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("sample_id") or row.get("prompt_id") or index))[:96]
        row["trajectory_id"] = f"trc96_r11hybrid_v1__{index:03d}__{row.get('task')}__{slug}"


def validate_final_rows(rows: list[dict[str, Any]]) -> None:
    checks = quality_checks(rows)
    hard_failures = {
        "row_count_ok": checks["row_count_ok"],
        "task_balance_ok": checks["task_balance_ok"],
        "trajectory_id_unique_ok": checks["trajectory_id_unique_ok"],
        "code_leakage_marker_ok": checks["code_leakage_marker_ok"],
    }
    if not all(hard_failures.values()):
        raise RuntimeError(f"Final row validation failed: {hard_failures}")
    if checks["blank_response_count"] or checks["nonpositive_reward_train_count"]:
        raise RuntimeError(f"Final row quality failed: {checks}")


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "unknown") for row in rows).items()))


def unique_prompt_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    tasks = sorted({str(row.get("task") or "unknown") for row in rows})
    return {task: len({row.get("prompt_id") for row in rows if str(row.get("task") or "unknown") == task}) for task in tasks}


def duplicate_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for task in sorted({str(row.get("task") or "unknown") for row in rows}):
        counts = Counter(str(row.get("prompt_id")) for row in rows if str(row.get("task") or "unknown") == task)
        result[task] = sum(count - 1 for count in counts.values())
    return result


def nested_count(rows: list[dict[str, Any]], group_field: str, value_fn: Any) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {}
    for row in rows:
        group = str(row.get(group_field) or "unknown")
        result.setdefault(group, Counter())[str(value_fn(row) or "unknown")] += 1
    return {group: dict(sorted(counter.items())) for group, counter in sorted(result.items())}


def reward_stats(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(row.get("reward_train") or 0.0) for row in rows]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def expert_source(row: dict[str, Any]) -> str:
    details = (row.get("sample_metadata") or {}).get("details") or {}
    for key in ("expert_model", "expert_name", "model", "model_name"):
        value = details.get(key)
        if value:
            return str(value)
    return Path(str(row.get("source_path") or "unknown")).name


def code_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("reference") or {}).get("metadata") or {}


def code_tags(row: dict[str, Any]) -> list[str]:
    return [str(tag) for tag in code_metadata(row).get("code_tags") or []]


def primary_code_tag(row: dict[str, Any]) -> str:
    metadata = code_metadata(row)
    explicit = metadata.get("primary_code_tag") or metadata.get("primary_tag")
    if explicit:
        return str(explicit)
    for tag in code_tags(row):
        if tag not in STOP_TAGS:
            return tag
    return "unknown"


def code_role(row: dict[str, Any]) -> str:
    metadata = code_metadata(row)
    return str(metadata.get("code_bank_role") or metadata.get("role") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stable-tm-bank", default=str(DEFAULT_STABLE_TM_BANK))
    parser.add_argument("--r10-bank", default=str(DEFAULT_R10_BANK))
    parser.add_argument("--r8d-bank", default=str(DEFAULT_R8D_BANK))
    parser.add_argument("--r8b-bank", default=str(DEFAULT_R8B_BANK))
    parser.add_argument("--rf-raw-rollout", default=str(DEFAULT_RF_RAW_ROLLOUT))
    return parser.parse_args()


if __name__ == "__main__":
    main()
