#!/usr/bin/env python3
"""Build TRC calibration with quota-controlled CodeP0-v3 trajectories.

This keeps the standard Tool/Memory selection from build_trc_calibration_v1 and
only replaces Code selection with role quotas from CodeP0 metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import write_json
from opvec.data.io import write_jsonl
from scripts.trc.build_trc_calibration_v1 import (
    DEFAULT_MEMORY_ROLLOUT,
    DEFAULT_TOOL_ROLLOUT,
    Candidate,
    candidate_sort_key,
    load_positive_candidates,
    materialize_row,
    select_balanced_task_trajectories,
)


DEFAULT_CODE_ROLLOUT = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/"
    "code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl"
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    role_quotas = parse_quota(args.code_role_quota)
    if sum(role_quotas.values()) != int(args.per_task):
        raise ValueError(f"Code role quotas must sum to --per-task={args.per_task}, got {role_quotas}")

    tool_candidates, tool_stats = load_positive_candidates(
        task="tool",
        expert="tool",
        sources=[(f"tool_source_{i:02d}", Path(p).expanduser()) for i, p in enumerate(args.tool_rollout)],
        positive_threshold=float(args.positive_threshold),
        include_sample_expert=[],
        exclude_sample_expert=[],
        memory_response_source="final",
        memory_trajectory_max_update_turns=0,
        memory_trajectory_turn_policy="uniform",
        memory_trajectory_include_final_turn=True,
    )
    memory_candidates, memory_stats = load_positive_candidates(
        task="memory",
        expert="memory",
        sources=[(f"memory_source_{i:02d}", Path(p).expanduser()) for i, p in enumerate(args.memory_rollout)],
        positive_threshold=float(args.positive_threshold),
        include_sample_expert=[],
        exclude_sample_expert=[],
        memory_response_source=str(args.memory_response_source),
        memory_trajectory_max_update_turns=int(args.memory_trajectory_max_update_turns),
        memory_trajectory_turn_policy=str(args.memory_trajectory_turn_policy),
        memory_trajectory_include_final_turn=not bool(args.memory_trajectory_no_final_turn),
    )
    code_candidates, code_stats = load_positive_candidates(
        task="code",
        expert="code",
        sources=[(f"code_source_{i:02d}", Path(p).expanduser()) for i, p in enumerate(args.code_rollout)],
        positive_threshold=float(args.positive_threshold),
        include_sample_expert=[],
        exclude_sample_expert=[],
        memory_response_source="final",
        memory_trajectory_max_update_turns=0,
        memory_trajectory_turn_policy="uniform",
        memory_trajectory_include_final_turn=True,
    )

    selected_tool, tool_selection = select_balanced_task_trajectories(tool_candidates, target_count=int(args.per_task))
    selected_memory, memory_selection = select_balanced_task_trajectories(memory_candidates, target_count=int(args.per_task))
    selected_code, code_selection = select_code_by_role_quota(code_candidates, quotas=role_quotas)
    selected = selected_tool + selected_memory + selected_code
    rows = [materialize_row(candidate, rank=index) for index, candidate in enumerate(selected)]
    out_jsonl = output_dir / "trc96_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, rows)

    summary = {
        "format": "trc_codep0_quota_calibration_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(out_jsonl),
        "num_rows": len(rows),
        "per_task_target": int(args.per_task),
        "positive_threshold": float(args.positive_threshold),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "unique_prompt_counts": {
            task: len({row["prompt_id"] for row in rows if row["task"] == task})
            for task in ("tool", "memory", "code")
        },
        "code_role_quotas": role_quotas,
        "source_stats": {"tool": tool_stats, "memory": memory_stats, "code": code_stats},
        "selection_stats": {
            "tool": tool_selection,
            "memory": memory_selection,
            "code": code_selection,
        },
        "memory_trajectory_policy": {
            "response_source": str(args.memory_response_source),
            "max_update_turns": int(args.memory_trajectory_max_update_turns),
            "turn_policy": str(args.memory_trajectory_turn_policy),
            "include_final_turn": not bool(args.memory_trajectory_no_final_turn),
        },
    }
    write_json(output_dir / "trc96_summary.json", summary)
    write_readme(output_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def select_code_by_role_quota(candidates: list[Candidate], *, quotas: dict[str, int]) -> tuple[list[Candidate], dict[str, Any]]:
    ordered = sorted(candidates, key=candidate_sort_key)
    selected: list[Candidate] = []
    used_prompts: set[str] = set()
    selected_keys: set[tuple[str, str]] = set()
    skipped_by_quota: dict[str, int] = {}
    for role, quota in quotas.items():
        role_candidates = [candidate for candidate in ordered if code_role(candidate) == role]
        picked = pick_unique_first(role_candidates, quota, used_prompts=used_prompts, selected_keys=selected_keys)
        selected.extend(picked)
        skipped_by_quota[role] = max(0, quota - len(picked))
    if len(selected) < sum(quotas.values()):
        fill = pick_unique_first(ordered, sum(quotas.values()) - len(selected), used_prompts=used_prompts, selected_keys=selected_keys)
        selected.extend(fill)
    if len(selected) != sum(quotas.values()):
        raise ValueError(f"Unable to select requested Code rows: got {len(selected)} for {quotas}")
    return selected, {
        "selected": len(selected),
        "selected_unique_prompts": len({item.prompt_id for item in selected}),
        "selected_duplicate_prompt_rows": len(selected) - len({item.prompt_id for item in selected}),
        "selected_source_counts": dict(Counter(item.source_name for item in selected)),
        "selected_role_counts": dict(Counter(code_role(item) for item in selected)),
        "selected_tag_counts": dict(Counter(tag for item in selected for tag in code_tags(item))),
        "unfilled_role_quota": skipped_by_quota,
    }


def pick_unique_first(
    candidates: list[Candidate],
    count: int,
    *,
    used_prompts: set[str],
    selected_keys: set[tuple[str, str]],
) -> list[Candidate]:
    picked: list[Candidate] = []
    for candidate in candidates:
        key = (candidate.source_name, candidate.sample_id)
        if key in selected_keys or candidate.prompt_id in used_prompts:
            continue
        picked.append(candidate)
        selected_keys.add(key)
        used_prompts.add(candidate.prompt_id)
        if len(picked) >= count:
            return picked
    for candidate in candidates:
        key = (candidate.source_name, candidate.sample_id)
        if key in selected_keys:
            continue
        picked.append(candidate)
        selected_keys.add(key)
        used_prompts.add(candidate.prompt_id)
        if len(picked) >= count:
            return picked
    return picked


def code_role(candidate: Candidate) -> str:
    metadata = (candidate.reference or {}).get("metadata") or {}
    return str(metadata.get("code_bank_role") or metadata.get("role") or "unknown")


def code_tags(candidate: Candidate) -> list[str]:
    metadata = (candidate.reference or {}).get("metadata") or {}
    tags = metadata.get("code_tags") or []
    return [str(tag) for tag in tags]


def parse_quota(items: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Quota must use role=count format: {raw!r}")
        role, value = raw.split("=", 1)
        quotas[role.strip()] = int(value.strip())
    if not quotas:
        quotas = {"frontier": 12, "partial_edge": 10, "generation": 8, "stable": 2}
    return quotas


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TRC CodeP0 Quota Calibration",
        "",
        "Generated by `scripts/trc/build_trc_codep0_quota_calibration.py`.",
        "",
        f"- Rows: {summary['num_rows']}",
        f"- Output: `{Path(summary['output']).name}`",
        "",
        "```json",
        json.dumps(
            {
                "task_counts": summary["task_counts"],
                "unique_prompt_counts": summary["unique_prompt_counts"],
                "code_role_quotas": summary["code_role_quotas"],
                "code_selection": summary["selection_stats"]["code"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-task", type=int, default=32)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--tool-rollout", action="append", default=None)
    parser.add_argument("--memory-rollout", action="append", default=None)
    parser.add_argument("--code-rollout", action="append", default=None)
    parser.add_argument("--code-role-quota", action="append", default=[])
    parser.add_argument("--memory-response-source", choices=["final", "trajectory-turns"], default="trajectory-turns")
    parser.add_argument("--memory-trajectory-max-update-turns", type=int, default=3)
    parser.add_argument("--memory-trajectory-turn-policy", choices=["uniform", "late", "first-last"], default="late")
    parser.add_argument("--memory-trajectory-no-final-turn", action="store_true")
    args = parser.parse_args()
    args.tool_rollout = args.tool_rollout or [str(DEFAULT_TOOL_ROLLOUT)]
    args.memory_rollout = args.memory_rollout or [str(DEFAULT_MEMORY_ROLLOUT)]
    args.code_rollout = args.code_rollout or [str(DEFAULT_CODE_ROLLOUT)]
    return args


if __name__ == "__main__":
    main()
