#!/usr/bin/env python3
"""Build the Round12 RF-only tag-quota TRC calibration bank.

Round12D isolates whether the strong R8D LiveBench signal came from
ReasonFlux/RF-only CodeP0 purity. Tool and Memory are copied from the stable
late3 R11 hybrid bank; Code is selected only from ReasonFlux successful
CodeP0-v3 trajectories with primary-tag quotas.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.trc.build_trc_calibration_v1 import (
    Candidate,
    candidate_sort_key,
    load_positive_candidates,
    materialize_row,
)


DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/"
    "r12d_rfonly_primarytag_quota_stablelate3"
)
DEFAULT_STABLE_TM_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round11_hybrid_v1/"
    "r11b_r11g_r10tag24_rf8_stablelate3/trc96_expert_trajectories.jsonl"
)
DEFAULT_RF_RAW_ROLLOUT = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/"
    "code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl"
)
DEFAULT_PRIMARY_TAG_QUOTAS = OrderedDict(
    [
        ("string", 11),
        ("math", 7),
        ("graph", 5),
        ("dynamic_programming", 4),
        ("greedy", 3),
        ("format_sensitive", 2),
    ]
)
STOP_TAGS = {"stdin_stdout"}
LEAKAGE_MARKERS = ("LiveBench", "LiveCodeBench", "CURE/data")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    quotas = parse_quota(args.code_primary_tag_quota)
    if sum(quotas.values()) != 32:
        raise ValueError(f"Code primary tag quotas must sum to 32, got {dict(quotas)}")

    stable_rows = read_jsonl(Path(args.stable_tm_bank).expanduser())
    tool_rows = select_task_rows(stable_rows, "tool", expected=32)
    memory_rows = select_task_rows(stable_rows, "memory", expected=32)

    code_candidates, source_stats = load_positive_candidates(
        task="code",
        expert="code",
        sources=[("code_source_00", Path(args.rf_raw_rollout).expanduser())],
        positive_threshold=1.0,
        include_sample_expert=[],
        exclude_sample_expert=[],
        memory_response_source="final",
        memory_trajectory_max_update_turns=0,
        memory_trajectory_turn_policy="uniform",
        memory_trajectory_include_final_turn=True,
    )
    selected_code, code_selection = select_code_by_primary_tag_quota(code_candidates, quotas=quotas)
    code_rows = [materialize_row(candidate, rank=0) for candidate in selected_code]

    rows: list[dict[str, Any]] = []
    rows.extend(with_provenance(row, component="stable_late3_tool", source_bank="r11_hybrid_stable_late3") for row in tool_rows)
    rows.extend(
        with_provenance(row, component="stable_late3_memory", source_bank="r11_hybrid_stable_late3") for row in memory_rows
    )
    rows.extend(with_provenance(row, component="rf_only_primary_tag_quota_code", source_bank="code_p0_v3_rf_raw") for row in code_rows)
    renumber_trajectory_ids(rows)
    validate_final_rows(rows)

    out_jsonl = output_dir / "trc96_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, rows)
    summary = build_summary(
        rows=rows,
        out_jsonl=out_jsonl,
        args=args,
        quotas=quotas,
        source_stats=source_stats,
        code_candidates=code_candidates,
        code_selection=code_selection,
    )
    write_json(output_dir / "trc96_summary.json", summary)
    md = render_markdown(summary)
    (output_dir / "trc96_summary.md").write_text(md, encoding="utf-8")
    (output_dir / "README.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def select_code_by_primary_tag_quota(
    candidates: list[Candidate], *, quotas: OrderedDict[str, int]
) -> tuple[list[Candidate], dict[str, Any]]:
    ordered = sorted(candidates, key=candidate_sort_key)
    by_tag: dict[str, list[Candidate]] = {}
    for candidate in ordered:
        by_tag.setdefault(primary_code_tag_for_candidate(candidate), []).append(candidate)

    selected: list[Candidate] = []
    used_samples: set[tuple[str, str]] = set()
    used_prompts: set[str] = set()
    tag_stats: dict[str, Any] = {}

    for tag, quota in quotas.items():
        tag_candidates = by_tag.get(tag, [])
        unique_available = len({candidate.prompt_id for candidate in tag_candidates})
        tag_selected: list[Candidate] = []

        for candidate in tag_candidates:
            key = (candidate.source_name, candidate.sample_id)
            if key in used_samples or candidate.prompt_id in used_prompts:
                continue
            tag_selected.append(candidate)
            selected.append(candidate)
            used_samples.add(key)
            used_prompts.add(candidate.prompt_id)
            if len(tag_selected) >= quota:
                break

        if len(tag_selected) < quota:
            duplicate_pool = sorted(
                [
                    candidate
                    for candidate in tag_candidates
                    if (candidate.source_name, candidate.sample_id) not in used_samples
                ],
                key=lambda candidate: (
                    Counter(item.prompt_id for item in tag_selected)[candidate.prompt_id],
                    candidate.length,
                    candidate.sample_index,
                    candidate.sample_id,
                ),
            )
            for candidate in duplicate_pool:
                key = (candidate.source_name, candidate.sample_id)
                tag_selected.append(candidate)
                selected.append(candidate)
                used_samples.add(key)
                used_prompts.add(candidate.prompt_id)
                if len(tag_selected) >= quota:
                    break

        if len(tag_selected) != quota:
            raise RuntimeError(f"Unable to fill RF-only quota for {tag}: requested {quota}, selected {len(tag_selected)}")

        prompt_counts = Counter(candidate.prompt_id for candidate in tag_selected)
        tag_stats[tag] = {
            "quota": quota,
            "unique_available": unique_available,
            "selected": len(tag_selected),
            "selected_unique_prompts": len(prompt_counts),
            "selected_duplicate_prompt_rows": sum(count - 1 for count in prompt_counts.values()),
            "selected_prompt_counts": dict(sorted(prompt_counts.items())),
        }

    prompt_counts = Counter(candidate.prompt_id for candidate in selected)
    return selected, {
        "selected": len(selected),
        "selected_unique_prompts": len(prompt_counts),
        "selected_duplicate_prompt_rows": sum(count - 1 for count in prompt_counts.values()),
        "selected_duplicate_prompt_details": {
            prompt_id: count for prompt_id, count in sorted(prompt_counts.items()) if count > 1
        },
        "selected_primary_tag_counts": dict(sorted(Counter(primary_code_tag_for_candidate(item) for item in selected).items())),
        "selected_role_counts": dict(sorted(Counter(code_role_for_candidate(item) for item in selected).items())),
        "selected_tag_counts": dict(sorted(Counter(tag for item in selected for tag in code_tags_for_candidate(item)).items())),
        "per_quota": tag_stats,
    }


def build_summary(
    *,
    rows: list[dict[str, Any]],
    out_jsonl: Path,
    args: argparse.Namespace,
    quotas: OrderedDict[str, int],
    source_stats: dict[str, Any],
    code_candidates: list[Candidate],
    code_selection: dict[str, Any],
) -> dict[str, Any]:
    code_rows = [row for row in rows if row.get("task") == "code"]
    return {
        "format": "trc_round12_rfonly_tagquota_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(out_jsonl),
        "num_rows": len(rows),
        "task_counts": count_by(rows, "task"),
        "unique_prompt_counts": unique_prompt_counts(rows),
        "duplicate_prompt_rows": duplicate_prompt_rows(rows),
        "input_banks": {
            "stable_tool_memory_late3": str(Path(args.stable_tm_bank).expanduser().resolve()),
            "rf_raw_rollout": str(Path(args.rf_raw_rollout).expanduser().resolve()),
        },
        "construction_policy": {
            "tool": "32 rows copied from R11 hybrid stable late3 Tool rows.",
            "memory": "32 rows copied from R11 hybrid stable late3 Memory trajectory-turn rows.",
            "code": "32 Code rows selected only from ReasonFlux CodeP0-v3 successful trajectories by primary tag quota.",
            "leakage_policy": "CodeContests_train CodeP0-v3 only; no DeepSeek fallback and no LiveBench/LiveCodeBench/CURE hidden rows.",
        },
        "code_primary_tag_quotas": dict(quotas),
        "rf_candidate_capacity": candidate_capacity(code_candidates),
        "source_stats": source_stats,
        "code_selection": code_selection,
        "final_statistics": stats_for_rows(rows),
        "final_code_distribution": code_distribution(code_rows),
        "quality_checks": quality_checks(rows),
    }


def candidate_capacity(candidates: list[Candidate]) -> dict[str, Any]:
    rows = [candidate_to_row_view(candidate) for candidate in candidates]
    return {
        "positive_samples": len(candidates),
        "positive_unique_prompts": len({candidate.prompt_id for candidate in candidates}),
        "primary_tag_unique_prompt_counts": dict(
            sorted(
                (
                    tag,
                    len({row["prompt_id"] for row in rows if row["primary_tag"] == tag}),
                )
                for tag in sorted({row["primary_tag"] for row in rows})
            )
        ),
        "primary_tag_sample_counts": dict(sorted(Counter(row["primary_tag"] for row in rows).items())),
        "role_unique_prompt_counts": dict(
            sorted((role, len({row["prompt_id"] for row in rows if row["role"] == role})) for role in sorted({row["role"] for row in rows}))
        ),
        "tag_sample_counts": dict(sorted(Counter(tag for row in rows for tag in row["tags"]).items())),
    }


def candidate_to_row_view(candidate: Candidate) -> dict[str, Any]:
    return {
        "prompt_id": candidate.prompt_id,
        "sample_id": candidate.sample_id,
        "primary_tag": primary_code_tag_for_candidate(candidate),
        "role": code_role_for_candidate(candidate),
        "tags": code_tags_for_candidate(candidate),
    }


def stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_counts": count_by(rows, "task"),
        "unique_prompt_counts": unique_prompt_counts(rows),
        "duplicate_prompt_rows": duplicate_prompt_rows(rows),
        "duplicate_prompt_details": duplicate_prompt_details(rows),
        "source_path_counts": nested_count(rows, "task", lambda row: Path(str(row.get("source_path") or "unknown")).name),
        "expert_source_counts": nested_count(rows, "task", expert_source),
        "reward_train": reward_stats(rows),
    }


def code_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_prompts": len({row.get("prompt_id") for row in rows}),
        "duplicate_prompt_rows": sum(count - 1 for count in Counter(str(row.get("prompt_id")) for row in rows).values()),
        "duplicate_prompt_details": {
            prompt_id: count
            for prompt_id, count in sorted(Counter(str(row.get("prompt_id")) for row in rows).items())
            if count > 1
        },
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
    deepseek_hits = []
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
        if "deepseek" in haystack.lower():
            deepseek_hits.append(row.get("trajectory_id"))
    details_rows = [((row.get("sample_metadata") or {}).get("details") or {}) for row in code_rows]
    return {
        "row_count_ok": len(rows) == 96,
        "task_balance_ok": count_by(rows, "task") == {"code": 32, "memory": 32, "tool": 32},
        "trajectory_id_unique_ok": len(ids) == len(set(ids)),
        "blank_response_count": len(blank_response),
        "nonpositive_reward_train_count": len(nonpositive),
        "code_leakage_marker_hits": leakage_hits,
        "code_leakage_marker_ok": not leakage_hits,
        "code_deepseek_marker_hits": deepseek_hits,
        "code_rf_only_ok": not deepseek_hits,
        "code_present_false_count": sum(1 for details in details_rows if details.get("code_present") is False),
        "syntax_false_count": sum(1 for details in details_rows if details.get("syntax_ok") is False),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    code = summary["final_code_distribution"]
    duplicate_details = code["duplicate_prompt_details"]
    quota_rows = []
    for tag, stats in summary["code_selection"]["per_quota"].items():
        quota_rows.append(
            "| {tag} | {quota} | {avail} | {uniq} | {dup} |".format(
                tag=tag,
                quota=stats["quota"],
                avail=stats["unique_available"],
                uniq=stats["selected_unique_prompts"],
                dup=stats["selected_duplicate_prompt_rows"],
            )
        )
    return "\n".join(
        [
            "# TRC Round12 RF-only Tag-Quota Calibration",
            "",
            f"- Output: `{summary['output']}`",
            f"- Rows: `{summary['num_rows']}`",
            f"- Task counts: `{json.dumps(summary['task_counts'], sort_keys=True)}`",
            f"- Unique prompts: `{json.dumps(summary['unique_prompt_counts'], sort_keys=True)}`",
            "",
            "## Construction",
            "",
            "- Tool: 32 rows copied from R11 hybrid stable late3 Tool rows.",
            "- Memory: 32 rows copied from R11 hybrid stable late3 Memory rows.",
            "- Code: 32 rows selected only from ReasonFlux CodeP0-v3 successful trajectories.",
            "- Tag quota type: primary code tag.",
            "- No DeepSeek fallback; no LiveBench/LiveCodeBench/CURE hidden rows.",
            "",
            "## Quota Fill",
            "",
            "| primary tag | quota | RF unique available | selected unique | duplicate rows |",
            "|---|---:|---:|---:|---:|",
            *quota_rows,
            "",
            "## Final Code Distribution",
            "",
            f"- Source paths: `{json.dumps(code['source_path_counts'], sort_keys=True)}`",
            f"- Roles: `{json.dumps(code['role_counts'], sort_keys=True)}`",
            f"- Primary tags: `{json.dumps(code['primary_tag_counts'], sort_keys=True)}`",
            f"- Tags: `{json.dumps(code['tag_counts'], sort_keys=True)}`",
            f"- Duplicate prompt details: `{json.dumps(duplicate_details, sort_keys=True)}`",
            "",
            "## Quality Checks",
            "",
            f"`{json.dumps(summary['quality_checks'], sort_keys=True)}`",
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
    row_metadata["round12_rfonly_tagquota"] = {
        "component": component,
        "source_bank": source_bank,
        "source_trajectory_id": copied.get("trajectory_id"),
    }
    copied["row_metadata"] = row_metadata
    return copied


def renumber_trajectory_ids(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("sample_id") or row.get("prompt_id") or index))[:96]
        row["trajectory_id"] = f"trc96_r12rfonly_v1__{index:03d}__{row.get('task')}__{slug}"


def validate_final_rows(rows: list[dict[str, Any]]) -> None:
    checks = quality_checks(rows)
    required = {
        "row_count_ok": checks["row_count_ok"],
        "task_balance_ok": checks["task_balance_ok"],
        "trajectory_id_unique_ok": checks["trajectory_id_unique_ok"],
        "code_leakage_marker_ok": checks["code_leakage_marker_ok"],
        "code_rf_only_ok": checks["code_rf_only_ok"],
    }
    if not all(required.values()):
        raise RuntimeError(f"Final row validation failed: {required}")
    if checks["blank_response_count"] or checks["nonpositive_reward_train_count"]:
        raise RuntimeError(f"Final row quality failed: {checks}")


def parse_quota(items: list[str]) -> OrderedDict[str, int]:
    if not items:
        return OrderedDict(DEFAULT_PRIMARY_TAG_QUOTAS)
    quotas: OrderedDict[str, int] = OrderedDict()
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Quota must use tag=count format: {raw!r}")
        tag, value = raw.split("=", 1)
        quotas[tag.strip()] = int(value.strip())
    return quotas


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "unknown") for row in rows).items()))


def unique_prompt_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    tasks = sorted({str(row.get("task") or "unknown") for row in rows})
    return {task: len({row.get("prompt_id") for row in rows if str(row.get("task") or "unknown") == task}) for task in tasks}


def duplicate_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        task: sum(count - 1 for count in Counter(str(row.get("prompt_id")) for row in rows if str(row.get("task") or "unknown") == task).values())
        for task in sorted({str(row.get("task") or "unknown") for row in rows})
    }


def duplicate_prompt_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    details: dict[str, dict[str, int]] = {}
    for task in sorted({str(row.get("task") or "unknown") for row in rows}):
        counts = Counter(str(row.get("prompt_id")) for row in rows if str(row.get("task") or "unknown") == task)
        details[task] = {prompt_id: count for prompt_id, count in sorted(counts.items()) if count > 1}
    return details


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


def candidate_metadata(candidate: Candidate) -> dict[str, Any]:
    return (candidate.reference or {}).get("metadata") or {}


def code_tags_for_candidate(candidate: Candidate) -> list[str]:
    return [str(tag) for tag in candidate_metadata(candidate).get("code_tags") or []]


def primary_code_tag_for_candidate(candidate: Candidate) -> str:
    metadata = candidate_metadata(candidate)
    explicit = metadata.get("primary_code_tag") or metadata.get("primary_tag")
    if explicit:
        return str(explicit)
    for tag in code_tags_for_candidate(candidate):
        if tag not in STOP_TAGS:
            return tag
    return "unknown"


def code_role_for_candidate(candidate: Candidate) -> str:
    metadata = candidate_metadata(candidate)
    return str(metadata.get("code_bank_role") or metadata.get("role") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stable-tm-bank", default=str(DEFAULT_STABLE_TM_BANK))
    parser.add_argument("--rf-raw-rollout", default=str(DEFAULT_RF_RAW_ROLLOUT))
    parser.add_argument("--code-primary-tag-quota", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
