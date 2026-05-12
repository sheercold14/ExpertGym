#!/usr/bin/env python3
"""Select a fixed task-balanced frontier calibration set from rollout files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import stable_hash


def main() -> None:
    args = parse_args()
    rows, summary = select_calibration(args)
    if args.strict and summary["deficits"]:
        raise SystemExit(f"Not enough frontier rows for strict selection: {summary['deficits']}")
    if not args.dry_run:
        written = write_jsonl(args.output, rows)
        summary["written"] = written
        Path(args.output).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def select_calibration(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    quotas = _parse_quotas(args.quota, tasks=tasks, default=args.per_task)
    raw_rows = []
    for path in args.rollouts:
        for row in read_jsonl(path):
            row = dict(row)
            row.setdefault("source_rollout", str(path))
            raw_rows.append(row)

    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen_prompt_ids: set[str] = set()
    for row in raw_rows:
        task = str(row.get("task") or "")
        if task not in quotas:
            skipped["task_not_requested"] += 1
            continue
        prompt_id = str(row.get("prompt_id") or "")
        if not args.allow_duplicate_prompts and prompt_id in seen_prompt_ids:
            skipped["duplicate_prompt_id"] += 1
            continue
        if not _has_gradient_signal(row, min_reward_std=args.min_reward_std, min_frontier_weight=args.min_frontier_weight):
            skipped["no_gradient_signal"] += 1
            continue
        seen_prompt_ids.add(prompt_id)
        row["calibration_selection"] = {
            "selected_by": "select_balanced_frontier_calibration.py",
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "source_rollout": row.get("source_rollout"),
            "frontier_weight": _float(row.get("frontier", {}).get("frontier_weight"), default=0.0),
            "std_reward": _float(row.get("frontier", {}).get("std_reward"), default=0.0),
        }
        candidates_by_task[task].append(row)

    rng = random.Random(args.seed)
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    deficits: dict[str, dict[str, int]] = {}
    for task in tasks:
        pool = candidates_by_task.get(task, [])
        pool.sort(key=lambda row: _stable_row_key(row, seed=args.seed))
        rng.shuffle(pool)
        pool.sort(key=_quality_key, reverse=True)
        want = quotas[task]
        selected_by_task[task] = pool[:want]
        if len(pool) < want:
            deficits[task] = {"wanted": want, "available": len(pool), "missing": want - len(pool)}

    selected = _round_robin_interleave(selected_by_task, tasks)
    summary = {
        "format": "opvec_balanced_frontier_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rollouts": [str(path) for path in args.rollouts],
        "output": str(args.output),
        "seed": args.seed,
        "tasks": tasks,
        "quotas": quotas,
        "raw_rows": len(raw_rows),
        "candidate_counts": {task: len(candidates_by_task.get(task, [])) for task in tasks},
        "selected_counts": dict(sorted(Counter(row.get("task") for row in selected).items())),
        "selected_rows": len(selected),
        "deficits": deficits,
        "skipped": dict(sorted(skipped.items())),
        "filters": {
            "allow_duplicate_prompts": bool(args.allow_duplicate_prompts),
            "min_frontier_weight": args.min_frontier_weight,
            "min_reward_std": args.min_reward_std,
            "require_keep_for_policy_loss": True,
            "require_reward_variance": True,
        },
        "reward_stats": _reward_stats(selected_by_task),
    }
    return selected, summary


def _has_gradient_signal(row: dict[str, Any], *, min_reward_std: float, min_frontier_weight: float) -> bool:
    if not bool(row.get("keep_for_policy_loss")):
        return False
    samples = row.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return False
    rewards = [_float(sample.get("reward"), default=0.0) for sample in samples if isinstance(sample, dict)]
    if len(set(rewards)) < 2:
        return False
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    if not bool(frontier.get("has_variance", True)):
        return False
    if _float(frontier.get("std_reward"), default=0.0) < float(min_reward_std):
        return False
    if _float(frontier.get("frontier_weight"), default=0.0) < float(min_frontier_weight):
        return False
    return True


def _parse_quotas(items: list[str], *, tasks: list[str], default: int) -> dict[str, int]:
    quotas = {task: int(default) for task in tasks}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected task=count quota, got: {item}")
        task, value = item.split("=", 1)
        task = task.strip()
        if task not in quotas:
            raise ValueError(f"Quota task {task!r} is not in --tasks")
        quotas[task] = int(value)
    if any(value < 0 for value in quotas.values()):
        raise ValueError(f"Quotas must be non-negative: {quotas}")
    return quotas


def _stable_row_key(row: dict[str, Any], *, seed: int) -> str:
    return stable_hash({"seed": seed, "prompt_id": row.get("prompt_id"), "source_rollout": row.get("source_rollout")})


def _quality_key(row: dict[str, Any]) -> tuple[float, float, int]:
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    std_reward = _float(frontier.get("std_reward"), default=0.0)
    frontier_weight = _float(frontier.get("frontier_weight"), default=0.0)
    successes = int(frontier.get("num_success") or 0)
    failures = max(int(frontier.get("num_samples") or 0) - successes, 0)
    balance = -abs(successes - failures)
    return (std_reward, frontier_weight, balance)


def _round_robin_interleave(rows_by_task: dict[str, list[dict[str, Any]]], tasks: list[str]) -> list[dict[str, Any]]:
    selected = []
    max_len = max((len(rows_by_task.get(task, [])) for task in tasks), default=0)
    for idx in range(max_len):
        for task in tasks:
            rows = rows_by_task.get(task, [])
            if idx < len(rows):
                selected.append(rows[idx])
    return selected


def _reward_stats(rows_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats = {}
    for task, rows in rows_by_task.items():
        means = []
        stds = []
        for row in rows:
            frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
            means.append(_float(frontier.get("mean_reward"), default=0.0))
            stds.append(_float(frontier.get("std_reward"), default=0.0))
        stats[task] = {
            "rows": len(rows),
            "mean_reward_avg": sum(means) / len(means) if means else 0.0,
            "std_reward_avg": sum(stds) / len(stds) if stds else 0.0,
        }
    return stats


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", action="append", required=True, help="Input rollout JSONL. Repeat for multiple sources.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="tool,memory,code")
    parser.add_argument("--per-task", type=int, default=16)
    parser.add_argument("--quota", action="append", default=[], help="Override per-task quota, e.g. memory=8. Repeatable.")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--min-frontier-weight", type=float, default=0.20)
    parser.add_argument("--min-reward-std", type=float, default=0.05)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any quota is underfilled.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
