#!/usr/bin/env python3
"""Build frontier/recoverable/stable/unsolved state tables from rollout JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl
from opvec.config import write_json


TASKS = ("tool", "memory", "code")
STATES = ("frontier", "recoverable", "stable", "unsolved", "other")


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.rollout:
        rows.extend(read_jsonl(path))
    expert_index = _build_expert_positive_index(
        args.expert_rollout,
        key=args.key,
        positive_threshold=float(args.expert_positive_threshold),
    )
    prompt_rows = [
        _classify_row(
            row,
            key=args.key,
            expert_index=expert_index,
            success_threshold=float(args.current_success_threshold),
            stable_min_success_rate=float(args.stable_min_success_rate),
        )
        for row in rows
    ]
    summary = _summarize(prompt_rows, args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, summary)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["state_counts"], ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", action="append", required=True, help="Current policy rollout JSONL. Can be repeated.")
    parser.add_argument("--expert-rollout", action="append", default=[], help="Expert rollout JSONL used to detect recoverable prompts.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--run-label", default="rollout")
    parser.add_argument("--key", default="prompt_id")
    parser.add_argument("--current-success-threshold", type=float, default=1.0)
    parser.add_argument("--expert-positive-threshold", type=float, default=1.0)
    parser.add_argument("--stable-min-success-rate", type=float, default=1.0)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def _build_expert_positive_index(paths: list[str], *, key: str, positive_threshold: float) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            task = str(row.get("task", ""))
            row_key = str(row.get(key, row.get("prompt_id", "")))
            if not row_key or not task:
                continue
            positive_samples = [
                sample
                for sample in row.get("samples", [])
                if _sample_reward_train(sample) >= positive_threshold or bool(sample.get("success"))
            ]
            if not positive_samples:
                continue
            slot = index.setdefault(
                (task, row_key),
                {"task": task, "key": row_key, "positive_count": 0, "source_paths": set(), "max_reward_train": 0.0},
            )
            slot["positive_count"] += len(positive_samples)
            slot["source_paths"].add(str(path))
            slot["max_reward_train"] = max(slot["max_reward_train"], max(_sample_reward_train(sample) for sample in positive_samples))
    for payload in index.values():
        payload["source_paths"] = sorted(payload["source_paths"])
    return index


def _classify_row(
    row: dict[str, Any],
    *,
    key: str,
    expert_index: dict[tuple[str, str], dict[str, Any]],
    success_threshold: float,
    stable_min_success_rate: float,
) -> dict[str, Any]:
    task = str(row.get("task", ""))
    row_key = str(row.get(key, row.get("prompt_id", "")))
    samples = list(row.get("samples", []))
    rewards = [_sample_reward_train(sample) for sample in samples]
    successes = [
        bool(sample.get("success")) or _sample_reward_train(sample) >= success_threshold
        for sample in samples
    ]
    num_samples = len(samples)
    success_count = sum(1 for item in successes if item)
    success_rate = success_count / num_samples if num_samples else 0.0
    reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
    frontier_payload = row.get("frontier") or {}
    has_variance = bool(frontier_payload.get("has_variance")) or reward_std > 0.0
    keep_frontier = bool(row.get("keep_for_policy_loss"))
    expert_positive = expert_index.get((task, row_key))
    if success_rate >= stable_min_success_rate and num_samples:
        state = "stable"
    elif success_count == 0 and expert_positive is not None:
        state = "recoverable"
    elif success_count == 0:
        state = "unsolved"
    elif keep_frontier or has_variance or 0 < success_count < num_samples:
        state = "frontier"
    else:
        state = "other"
    return {
        "key": row_key,
        "prompt_id": row.get("prompt_id"),
        "task": task,
        "state": state,
        "num_samples": num_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "mean_reward_train": mean(rewards) if rewards else 0.0,
        "std_reward_train": reward_std,
        "keep_for_policy_loss": keep_frontier,
        "skip_reason": row.get("skip_reason"),
        "expert_positive_count": 0 if expert_positive is None else int(expert_positive["positive_count"]),
        "expert_positive_max_reward": None if expert_positive is None else float(expert_positive["max_reward_train"]),
    }


def _sample_reward_train(sample: dict[str, Any]) -> float:
    for key in ("reward_train", "task_reward", "reward"):
        value = sample.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _summarize(prompt_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_state = Counter(row["state"] for row in prompt_rows)
    by_task_state: dict[str, Counter] = defaultdict(Counter)
    for row in prompt_rows:
        by_task_state[row["task"]][row["state"]] += 1
    task_rows = defaultdict(list)
    for row in prompt_rows:
        task_rows[row["task"]].append(row)
    task_stats = {}
    for task, rows in sorted(task_rows.items()):
        task_stats[task] = {
            "rows": len(rows),
            "mean_success_rate": mean(row["success_rate"] for row in rows) if rows else 0.0,
            "mean_reward_train": mean(row["mean_reward_train"] for row in rows) if rows else 0.0,
            "mean_reward_std": mean(row["std_reward_train"] for row in rows) if rows else 0.0,
            "state_counts": {state: int(by_task_state[task].get(state, 0)) for state in STATES},
        }
    return {
        "format": "opvec_state_distribution_v1",
        "run_label": args.run_label,
        "notes": args.notes,
        "rollouts": [str(path) for path in args.rollout],
        "expert_rollouts": [str(path) for path in args.expert_rollout],
        "key": args.key,
        "thresholds": {
            "current_success": float(args.current_success_threshold),
            "expert_positive": float(args.expert_positive_threshold),
            "stable_min_success_rate": float(args.stable_min_success_rate),
        },
        "rows": len(prompt_rows),
        "state_counts": {state: int(by_state.get(state, 0)) for state in STATES},
        "task_stats": task_stats,
        "prompt_rows": prompt_rows,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# State Distribution: {summary['run_label']}",
        "",
        f"Notes: {summary.get('notes') or '-'}",
        "",
        "## Overall",
        "",
        "| state | count | ratio |",
        "|---|---:|---:|",
    ]
    rows = int(summary["rows"])
    for state in STATES:
        count = int(summary["state_counts"].get(state, 0))
        ratio = count / rows if rows else 0.0
        lines.append(f"| {state} | {count} | {ratio:.4f} |")
    lines.extend(["", "## By Task", "", "| task | rows | frontier | recoverable | stable | unsolved | other | mean reward | mean success |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for task in sorted(summary["task_stats"]):
        stats = summary["task_stats"][task]
        counts = stats["state_counts"]
        lines.append(
            "| {task} | {rows} | {frontier} | {recoverable} | {stable} | {unsolved} | {other} | {reward:.4f} | {success:.4f} |".format(
                task=task,
                rows=int(stats["rows"]),
                frontier=int(counts.get("frontier", 0)),
                recoverable=int(counts.get("recoverable", 0)),
                stable=int(counts.get("stable", 0)),
                unsolved=int(counts.get("unsolved", 0)),
                other=int(counts.get("other", 0)),
                reward=float(stats["mean_reward_train"]),
                success=float(stats["mean_success_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
