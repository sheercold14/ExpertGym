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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "tool": [("tool_paper96_s2", Path(args.tool_rollout).expanduser())],
        "memory": [("memory_paper96_s2", Path(args.memory_rollout).expanduser())],
        "code": [(f"code_source_{idx:02d}", Path(path).expanduser()) for idx, path in enumerate(args.code_rollout)],
    }
    candidates_by_task: dict[str, list[Candidate]] = {}
    source_stats: dict[str, Any] = {}
    for task, task_sources in sources.items():
        candidates, stats = load_positive_candidates(
            task=task,
            expert=task,
            sources=task_sources,
            positive_threshold=float(args.positive_threshold),
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
            "memory": "MemAgent expert paper96 successful samples; unique prompts first, then additional successful samples.",
            "code": "ReasonFlux successful samples first, then DeepSeek-R1 and old code expert fallbacks.",
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
                if not response.strip():
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
                        },
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
    return {
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


def sample_reward_train(sample: dict[str, Any]) -> float:
    value = sample.get("reward_train", sample.get("reward", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
    parser.add_argument("--tool-rollout", default=str(DEFAULT_TOOL_ROLLOUT))
    parser.add_argument("--memory-rollout", default=str(DEFAULT_MEMORY_ROLLOUT))
    parser.add_argument("--code-rollout", action="append", default=[str(path) for path in DEFAULT_CODE_ROLLOUTS])
    return parser.parse_args()


if __name__ == "__main__":
    main()
