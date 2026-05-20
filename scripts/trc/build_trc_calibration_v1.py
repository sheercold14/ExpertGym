#!/usr/bin/env python3
"""Build TRC expert-trajectory calibration data.

This script is intentionally independent from the GRPO/OPD data builders.
It selects successful expert trajectories and emits exactly balanced rows for
the first TRC-Merging prototype.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import write_json
from opvec.data.io import read_jsonl, write_jsonl


DEFAULT_TOOL_ROLLOUT = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/"
    "tool_expert_paper96_s2_seed20260514.jsonl"
)
DEFAULT_MEMORY_ROLLOUT = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/"
    "memory_expert_paper96_s2_seed20260514.jsonl"
)
DEFAULT_CODE_ROLLOUTS = [
    Path(
        "/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/"
        "code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"
    ),
    Path(
        "/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/"
        "code_expert_reasonflux_coder7b_s8_seed20260516.jsonl"
    ),
    Path(
        "/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/"
        "code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl"
    ),
    Path(
        "/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/"
        "code_expert_paper96_s2_seed20260514.jsonl"
    ),
]


@dataclass(frozen=True)
class Candidate:
    task: str
    expert: str
    source_name: str
    source_path: str
    source_priority: int
    prompt_id: str
    group_id: str
    sample_id: str
    prompt: str
    rendered_prompt: str
    response: str
    reference: dict[str, Any]
    reward: float
    reward_train: float
    task_reward: float | None
    length: int
    sample_index: int
    row_index: int
    sample: dict[str, Any]
    row_metadata: dict[str, Any]
    trajectory_turns: list[dict[str, Any]]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tool_rollouts = [Path(path).expanduser() for path in (args.tool_rollout or [str(DEFAULT_TOOL_ROLLOUT)])]
    memory_rollouts = [Path(path).expanduser() for path in (args.memory_rollout or [str(DEFAULT_MEMORY_ROLLOUT)])]
    code_rollouts = [Path(path).expanduser() for path in (args.code_rollout or [str(path) for path in DEFAULT_CODE_ROLLOUTS])]
    sources = {
        "tool": [(f"tool_source_{idx:02d}", path) for idx, path in enumerate(tool_rollouts)],
        "memory": [(f"memory_source_{idx:02d}", path) for idx, path in enumerate(memory_rollouts)],
        "code": [(f"code_source_{idx:02d}", path) for idx, path in enumerate(code_rollouts)],
    }
    candidates_by_task: dict[str, list[Candidate]] = {}
    source_stats: dict[str, Any] = {}
    for task, task_sources in sources.items():
        candidates, stats = load_positive_candidates(
            task=task,
            expert=task,
            sources=task_sources,
            positive_threshold=float(args.positive_threshold),
            include_sample_expert=args.include_sample_expert,
            exclude_sample_expert=args.exclude_sample_expert,
            memory_response_source=str(args.memory_response_source),
            memory_trajectory_max_update_turns=int(args.memory_trajectory_max_update_turns),
            memory_trajectory_turn_policy=str(args.memory_trajectory_turn_policy),
            memory_trajectory_include_final_turn=not bool(args.memory_trajectory_no_final_turn),
        )
        candidates_by_task[task] = candidates
        source_stats[task] = stats

    selected: list[Candidate] = []
    selection_stats: dict[str, Any] = {}
    for task in ("tool", "memory", "code"):
        task_selected, stats = select_balanced_task_trajectories(
            candidates_by_task[task],
            target_count=int(args.per_task),
        )
        selected.extend(task_selected)
        selection_stats[task] = stats

    rows = [materialize_row(candidate, rank=index) for index, candidate in enumerate(selected)]
    if len(rows) != int(args.per_task) * 3:
        raise RuntimeError(f"Expected {int(args.per_task) * 3} rows, got {len(rows)}")

    out_jsonl = output_dir / "trc96_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, rows)
    summary = {
        "format": "trc_calibration_summary_v1",
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
        "duplicate_prompt_rows": {
            task: sum(count - 1 for count in Counter(row["prompt_id"] for row in rows if row["task"] == task).values())
            for task in ("tool", "memory", "code")
        },
        "source_stats": source_stats,
        "selection_stats": selection_stats,
        "source_policy": {
            "tool": "ToolRL expert paper96 successful samples; unique prompts first, then additional successful samples.",
            "memory": (
                "MemAgent expert paper96 successful samples; unique prompts first, then additional successful samples. "
                f"memory_response_source={args.memory_response_source}."
            ),
            "code": "ReasonFlux successful samples first, then DeepSeek-R1 and old code expert fallbacks.",
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


def load_positive_candidates(
    *,
    task: str,
    expert: str,
    sources: Iterable[tuple[str, Path]],
    positive_threshold: float,
    include_sample_expert: list[str],
    exclude_sample_expert: list[str],
    memory_response_source: str,
    memory_trajectory_max_update_turns: int,
    memory_trajectory_turn_policy: str,
    memory_trajectory_include_final_turn: bool,
) -> tuple[list[Candidate], dict[str, Any]]:
    candidates: list[Candidate] = []
    stats: dict[str, Any] = {
        "sources": {},
        "total_rows": 0,
        "total_samples": 0,
        "positive_samples": 0,
        "positive_prompts": 0,
    }
    positive_prompts: set[str] = set()
    for source_priority, (source_name, source_path) in enumerate(sources):
        rows = read_jsonl(source_path)
        source_positive = 0
        source_positive_prompts: set[str] = set()
        source_samples = 0
        for row_index, row in enumerate(rows):
            prompt_id = str(row.get("prompt_id") or row.get("group_id") or f"{task}_{row_index:05d}")
            group_id = str(row.get("group_id") or prompt_id)
            samples = list(row.get("samples") or [])
            source_samples += len(samples)
            for sample_index, sample in enumerate(samples):
                reward_train = sample_reward_train(sample)
                response = str(sample.get("text") or sample.get("response") or "")
                trajectory_turns: list[dict[str, Any]] = []
                if task == "memory" and memory_response_source == "trajectory-turns":
                    trajectory_turns = extract_memory_trajectory_turns(
                        sample,
                        max_update_turns=memory_trajectory_max_update_turns,
                        turn_policy=memory_trajectory_turn_policy,
                        include_final_turn=memory_trajectory_include_final_turn,
                    )
                    trajectory_response = format_trajectory_response(trajectory_turns)
                    if trajectory_response.strip():
                        response = trajectory_response
                if not response.strip():
                    continue
                if not sample_expert_filter_ok(
                    sample,
                    row=row,
                    task=task,
                    source_name=source_name,
                    include=include_sample_expert,
                    exclude=exclude_sample_expert,
                ):
                    continue
                if not (bool(sample.get("success")) or reward_train >= positive_threshold):
                    continue
                sample_id = str(sample.get("sample_id") or f"{prompt_id}__k{sample_index}")
                source_positive += 1
                source_positive_prompts.add(prompt_id)
                positive_prompts.add(prompt_id)
                candidates.append(
                    Candidate(
                        task=task,
                        expert=expert,
                        source_name=source_name,
                        source_path=str(source_path.resolve()),
                        source_priority=source_priority,
                        prompt_id=prompt_id,
                        group_id=group_id,
                        sample_id=sample_id,
                        prompt=str(row.get("prompt") or ""),
                        rendered_prompt=str(row.get("rendered_prompt") or row.get("prompt") or ""),
                        response=response,
                        reference=dict(row.get("reference") or {}),
                        reward=float(sample.get("reward", reward_train) or 0.0),
                        reward_train=reward_train,
                        task_reward=as_optional_float(sample.get("task_reward")),
                        length=int(sample.get("length") or len(response.split())),
                        sample_index=sample_index,
                        row_index=row_index,
                        sample=dict(sample),
                        row_metadata={
                            "run_id": row.get("run_id"),
                            "policy_id": row.get("policy_id"),
                            "gate_checkpoint": row.get("gate_checkpoint"),
                            "seed": row.get("seed"),
                            "frontier": row.get("frontier"),
                            "skip_reason": row.get("skip_reason"),
                            "memagent_trajectory": row.get("memagent_trajectory"),
                        },
                        trajectory_turns=trajectory_turns,
                    )
                )
        stats["sources"][source_name] = {
            "path": str(source_path.resolve()),
            "rows": len(rows),
            "samples": source_samples,
            "positive_samples": source_positive,
            "positive_prompts": len(source_positive_prompts),
        }
        stats["total_rows"] += len(rows)
        stats["total_samples"] += source_samples
        stats["positive_samples"] += source_positive
    stats["positive_prompts"] = len(positive_prompts)
    return candidates, stats


def select_balanced_task_trajectories(candidates: list[Candidate], *, target_count: int) -> tuple[list[Candidate], dict[str, Any]]:
    if len(candidates) < target_count:
        raise ValueError(f"Need {target_count} positive candidates, got {len(candidates)}")
    ordered = sorted(candidates, key=candidate_sort_key)
    by_prompt: dict[str, Candidate] = {}
    for candidate in ordered:
        by_prompt.setdefault(candidate.prompt_id, candidate)
    selected = sorted(by_prompt.values(), key=candidate_sort_key)[:target_count]
    selected_keys = {(item.source_name, item.sample_id) for item in selected}
    if len(selected) < target_count:
        extras = [item for item in ordered if (item.source_name, item.sample_id) not in selected_keys]
        selected.extend(extras[: target_count - len(selected)])
    selected = selected[:target_count]
    if len(selected) < target_count:
        raise ValueError(f"Unable to select {target_count} trajectories, got {len(selected)}")
    source_counts = Counter(item.source_name for item in selected)
    prompt_counts = Counter(item.prompt_id for item in selected)
    return selected, {
        "selected": len(selected),
        "selected_unique_prompts": len(prompt_counts),
        "selected_duplicate_prompt_rows": sum(count - 1 for count in prompt_counts.values()),
        "selected_source_counts": dict(source_counts),
        "min_reward_train": min(item.reward_train for item in selected),
        "mean_reward_train": sum(item.reward_train for item in selected) / len(selected),
        "max_reward_train": max(item.reward_train for item in selected),
    }


def candidate_sort_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.source_priority,
        candidate.prompt_id,
        -candidate.reward_train,
        candidate.length,
        candidate.sample_index,
        candidate.sample_id,
    )


def materialize_row(candidate: Candidate, *, rank: int) -> dict[str, Any]:
    row = {
        "format": "trc_expert_trajectory_v1",
        "trajectory_id": f"trc96__{rank:03d}__{candidate.task}__{candidate.source_name}__{candidate.sample_id}",
        "task": candidate.task,
        "expert": candidate.expert,
        "source_name": candidate.source_name,
        "source_path": candidate.source_path,
        "prompt_id": candidate.prompt_id,
        "group_id": candidate.group_id,
        "sample_id": candidate.sample_id,
        "prompt": candidate.prompt,
        "rendered_prompt": candidate.rendered_prompt,
        "response": candidate.response,
        "reference": candidate.reference,
        "reward": candidate.reward,
        "reward_train": candidate.reward_train,
        "task_reward": candidate.task_reward,
        "length": candidate.length,
        "success": True,
        "row_metadata": candidate.row_metadata,
        "sample_metadata": {
            "details": candidate.sample.get("details"),
            "critical_spans": candidate.sample.get("critical_spans"),
            "behavior_span_reward": candidate.sample.get("behavior_span_reward"),
        },
    }
    if candidate.trajectory_turns:
        row["trajectory_turns"] = candidate.trajectory_turns
        row["trajectory_turn_count"] = len(candidate.trajectory_turns)
    return row


def extract_memory_trajectory_turns(
    sample: dict[str, Any],
    *,
    max_update_turns: int,
    turn_policy: str,
    include_final_turn: bool,
) -> list[dict[str, Any]]:
    raw_turns = sample.get("trajectory") or []
    update_turns = [turn for turn in raw_turns if str(turn.get("kind")) == "memory_update"]
    final_turns = [turn for turn in raw_turns if str(turn.get("kind")) == "final_answer"]
    selected = select_memory_update_turns(update_turns, max_turns=max_update_turns, policy=turn_policy)
    if include_final_turn and final_turns:
        selected = selected + [final_turns[-1]]
    turns: list[dict[str, Any]] = []
    for turn in selected:
        prompt_text = str(turn.get("prompt_text") or "")
        text = str(turn.get("text") or turn.get("response") or "")
        if not prompt_text.strip() or not text.strip():
            continue
        turns.append(
            {
                "turn": turn.get("turn"),
                "kind": turn.get("kind"),
                "prompt_text": prompt_text,
                "text": text,
                "length": turn.get("length"),
            }
        )
    return turns


def select_memory_update_turns(raw_turns: list[dict[str, Any]], *, max_turns: int, policy: str) -> list[dict[str, Any]]:
    if max_turns <= 0 or len(raw_turns) <= max_turns:
        return list(raw_turns)
    if max_turns == 1:
        return [raw_turns[-1]]
    lowered = str(policy).strip().lower()
    if lowered == "late":
        return list(raw_turns[-max_turns:])
    if lowered == "first-last":
        if max_turns == 2:
            return [raw_turns[0], raw_turns[-1]]
        middle = select_memory_update_turns(raw_turns[1:-1], max_turns=max_turns - 2, policy="uniform")
        return [raw_turns[0], *middle, raw_turns[-1]]
    if lowered != "uniform":
        raise ValueError(f"Unsupported --memory-trajectory-turn-policy: {policy!r}")
    indices: list[int] = []
    denom = max_turns - 1
    last_index = len(raw_turns) - 1
    for index in range(max_turns):
        selected = round(index * last_index / denom)
        if selected not in indices:
            indices.append(selected)
    cursor = 0
    while len(indices) < max_turns and cursor < len(raw_turns):
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return [raw_turns[index] for index in sorted(indices[:max_turns])]


def format_trajectory_response(turns: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for turn in turns:
        kind = str(turn.get("kind") or "turn")
        turn_id = turn.get("turn")
        header = f"[{kind}:{turn_id}]" if turn_id is not None else f"[{kind}]"
        parts.append(f"{header}\n{turn.get('text') or ''}".strip())
    return "\n\n".join(part for part in parts if part)


def sample_reward_train(sample: dict[str, Any]) -> float:
    value = sample.get("reward_train", sample.get("reward", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sample_expert_filter_ok(
    sample: dict[str, Any],
    *,
    row: dict[str, Any],
    task: str,
    source_name: str,
    include: list[str],
    exclude: list[str],
) -> bool:
    if task != "code":
        return True
    identity = sample_expert_identity(sample, row=row, source_name=source_name)
    if exclude and any(pattern.lower() in identity for pattern in exclude):
        return False
    if include and not any(pattern.lower() in identity for pattern in include):
        return False
    return True


def sample_expert_identity(sample: dict[str, Any], *, row: dict[str, Any], source_name: str) -> str:
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    parts = [
        source_name,
        row.get("policy_id"),
        row.get("run_id"),
        sample.get("policy_id"),
        sample.get("model"),
        sample.get("model_name"),
        sample.get("expert"),
        sample.get("expert_name"),
        details.get("expert_name"),
        details.get("expert_model"),
        details.get("expert_temp_file"),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TRC96 Expert Trajectory Calibration",
        "",
        "This directory is generated by `scripts/trc/build_trc_calibration_v1.py`.",
        "",
        f"- Rows: {summary['num_rows']}",
        f"- Per-task target: {summary['per_task_target']}",
        f"- Positive threshold: {summary['positive_threshold']}",
        f"- Output: `{Path(summary['output']).name}`",
        "",
        "Important: this is 96 successful expert trajectories, not necessarily 96 unique prompts.",
        "Tool and memory have fewer than 32 unique successful prompts in the available paper96 expert rollouts,",
        "so the builder uses unique prompts first and then adds extra successful samples from repeated prompts.",
        "",
        "## Task Counts",
        "",
        "```json",
        json.dumps(
            {
                "task_counts": summary["task_counts"],
                "unique_prompt_counts": summary["unique_prompt_counts"],
                "duplicate_prompt_rows": summary["duplicate_prompt_rows"],
                "selection_stats": summary["selection_stats"],
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
    parser.add_argument(
        "--output-dir",
        default="/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1",
        help="Directory for TRC calibration outputs.",
    )
    parser.add_argument("--per-task", type=int, default=32, help="Number of successful trajectories per task.")
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument(
        "--tool-rollout",
        action="append",
        default=None,
        help="Tool rollout JSONL. Repeat to set ordered sources. Defaults to paper96 ToolRL rollout.",
    )
    parser.add_argument(
        "--memory-rollout",
        action="append",
        default=None,
        help="Memory rollout JSONL. Repeat to set ordered sources. Defaults to paper96 MemAgent rollout.",
    )
    parser.add_argument(
        "--code-rollout",
        action="append",
        default=None,
        help="Code rollout JSONL. Repeat to set ordered sources. Defaults to historical ReasonFlux/DeepSeek/paper96 order.",
    )
    parser.add_argument(
        "--include-sample-expert",
        action="append",
        default=[],
        help="Keep only samples whose source/policy/details identity contains this lowercase substring. Repeatable.",
    )
    parser.add_argument(
        "--exclude-sample-expert",
        action="append",
        default=[],
        help="Drop samples whose source/policy/details identity contains this lowercase substring. Repeatable.",
    )
    parser.add_argument(
        "--memory-response-source",
        choices=["final", "trajectory-turns"],
        default="final",
        help="For memory rows, keep the old final answer response or materialize MemAgent trajectory turns.",
    )
    parser.add_argument(
        "--memory-trajectory-max-update-turns",
        type=int,
        default=0,
        help="When --memory-response-source trajectory-turns, keep up to this many memory_update turns; 0 keeps all.",
    )
    parser.add_argument(
        "--memory-trajectory-turn-policy",
        choices=["uniform", "late", "first-last"],
        default="uniform",
        help="How to subsample memory_update turns when a max is set.",
    )
    parser.add_argument(
        "--memory-trajectory-no-final-turn",
        action="store_true",
        help="When using memory trajectory turns, omit the final_answer turn.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
