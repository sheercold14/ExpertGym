#!/usr/bin/env python3
"""Build fixed calibration rows whose PPO advantages are deltas vs a baseline policy."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import stable_hash


def main() -> None:
    args = parse_args()
    rows, summary = build_calibration(args)
    if args.strict and summary["deficits"]:
        raise SystemExit(f"Not enough self-compare rows for strict selection: {summary['deficits']}")
    if not args.dry_run:
        written = write_jsonl(args.output, rows)
        summary["written"] = written
        Path(args.output).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_calibration(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    quotas = _parse_quotas(args.quota, tasks=tasks, default=args.per_task)
    baseline_rows = _read_rows(args.baseline_rollouts)
    candidate_rows = _read_rows(args.candidate_rollouts)
    baseline_by_key = _baseline_index(baseline_rows, key=args.key, agg=args.baseline_agg)

    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen_keys: set[str] = set()
    for raw_row in candidate_rows:
        task = str(raw_row.get("task") or "")
        if task not in quotas:
            skipped["task_not_requested"] += 1
            continue
        row_key = _key_value(raw_row, args.key)
        if not row_key:
            skipped["missing_key"] += 1
            continue
        if not args.allow_duplicate_prompts and row_key in seen_keys:
            skipped["duplicate_key"] += 1
            continue
        baseline = baseline_by_key.get(row_key)
        if baseline is None:
            skipped["missing_baseline"] += 1
            continue
        row = _annotate_row(raw_row, baseline=baseline, args=args)
        if row is None:
            skipped["invalid_samples"] += 1
            continue
        if not _has_self_compare_signal(row, args=args):
            skipped["no_self_compare_signal"] += 1
            continue
        seen_keys.add(row_key)
        candidates_by_task[task].append(row)

    rng = random.Random(args.seed)
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    deficits: dict[str, dict[str, int]] = {}
    for task in tasks:
        pool = candidates_by_task.get(task, [])
        pool.sort(key=lambda row: stable_hash({"seed": args.seed, "key": _key_value(row, args.key)}))
        rng.shuffle(pool)
        pool.sort(key=_quality_key, reverse=True)
        want = quotas[task]
        selected_by_task[task] = pool[:want]
        if len(pool) < want:
            deficits[task] = {"wanted": want, "available": len(pool), "missing": want - len(pool)}

    selected = _round_robin_interleave(selected_by_task, tasks)
    summary = {
        "format": "opvec_self_compare_advantage_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rollouts": [str(path) for path in args.candidate_rollouts],
        "baseline_rollouts": [str(path) for path in args.baseline_rollouts],
        "output": str(args.output),
        "seed": args.seed,
        "key": args.key,
        "baseline_agg": args.baseline_agg,
        "tasks": tasks,
        "quotas": quotas,
        "raw_candidate_rows": len(candidate_rows),
        "raw_baseline_rows": len(baseline_rows),
        "baseline_index_rows": len(baseline_by_key),
        "candidate_counts": {task: len(candidates_by_task.get(task, [])) for task in tasks},
        "selected_counts": dict(sorted(Counter(row.get("task") for row in selected).items())),
        "selected_rows": len(selected),
        "deficits": deficits,
        "skipped": dict(sorted(skipped.items())),
        "filters": {
            "allow_duplicate_prompts": bool(args.allow_duplicate_prompts),
            "min_abs_delta": args.min_abs_delta,
            "min_delta_std": args.min_delta_std,
            "require_cross_baseline": bool(args.require_cross_baseline),
            "require_source_keep_for_policy_loss": bool(args.require_source_keep_for_policy_loss),
        },
        "delta_stats": _delta_stats(selected_by_task),
    }
    return selected, summary


def _read_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            copied = dict(row)
            copied.setdefault("source_rollout", str(path))
            rows.append(copied)
    return rows


def _baseline_index(rows: list[dict[str, Any]], *, key: str, agg: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_key = _key_value(row, key)
        if row_key:
            grouped[row_key].append(row)
    index = {}
    for row_key, key_rows in grouped.items():
        rewards = []
        for row in key_rows:
            rewards.extend(_sample_rewards(row))
        if not rewards:
            continue
        index[row_key] = {
            "rows": len(key_rows),
            "sample_rewards": rewards,
            "mean": mean(rewards),
            "max": max(rewards),
            "min": min(rewards),
            "std": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "agg": max(rewards) if agg == "max" else mean(rewards),
        }
    return index


def _annotate_row(raw_row: dict[str, Any], *, baseline: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    samples = raw_row.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return None
    row = copy.deepcopy(raw_row)
    row["source_frontier"] = copy.deepcopy(raw_row.get("frontier"))
    row["source_keep_for_policy_loss"] = bool(raw_row.get("keep_for_policy_loss"))
    row["self_compare"] = {
        "created_by": "build_self_compare_advantage_calibration.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_agg": args.baseline_agg,
        "baseline_reward": baseline["agg"],
        "baseline_reward_mean": baseline["mean"],
        "baseline_reward_max": baseline["max"],
        "baseline_reward_min": baseline["min"],
        "baseline_reward_std": baseline["std"],
        "baseline_rows": baseline["rows"],
        "advantage_field": "reward_delta_vs_baseline",
    }
    deltas = []
    annotated_samples = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict) or "reward" not in sample:
            return None
        annotated = copy.deepcopy(sample)
        reward = float(annotated.get("reward", 0.0))
        delta = reward - float(baseline["agg"])
        annotated["absolute_reward"] = reward
        annotated["baseline_reward"] = float(baseline["agg"])
        annotated["baseline_reward_mean"] = float(baseline["mean"])
        annotated["baseline_reward_max"] = float(baseline["max"])
        annotated["reward_delta_vs_baseline"] = delta
        annotated["self_compare_success"] = delta > 0.0
        annotated.setdefault("sample_id", f"{row.get('prompt_id', 'prompt')}__k{idx}")
        deltas.append(delta)
        annotated_samples.append(annotated)
    row["samples"] = annotated_samples
    row["frontier"] = _delta_frontier(deltas)
    row["keep_for_policy_loss"] = True
    row["skip_reason"] = None
    return row


def _has_self_compare_signal(row: dict[str, Any], *, args: argparse.Namespace) -> bool:
    if args.require_source_keep_for_policy_loss and not bool(row.get("source_keep_for_policy_loss")):
        return False
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    if float(frontier.get("std_reward", 0.0)) < float(args.min_delta_std):
        return False
    if float(frontier.get("max_abs_delta", 0.0)) < float(args.min_abs_delta):
        return False
    if args.require_cross_baseline and not bool(frontier.get("crosses_baseline")):
        return False
    return True


def _delta_frontier(deltas: list[float]) -> dict[str, Any]:
    positives = sum(1 for value in deltas if value > 0.0)
    negatives = sum(1 for value in deltas if value < 0.0)
    zeros = len(deltas) - positives - negatives
    success_rate = positives / len(deltas) if deltas else 0.0
    std = pstdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "mean_reward": mean(deltas) if deltas else 0.0,
        "std_reward": std,
        "has_variance": std > 0.0,
        "frontier_weight": 4.0 * success_rate * (1.0 - success_rate),
        "num_success": positives,
        "num_failure": negatives + zeros,
        "num_positive": positives,
        "num_negative": negatives,
        "num_zero": zeros,
        "num_samples": len(deltas),
        "all_success": positives == len(deltas) and bool(deltas),
        "all_failure": positives == 0,
        "crosses_baseline": positives > 0 and negatives > 0,
        "max_abs_delta": max((abs(value) for value in deltas), default=0.0),
        "min_delta": min(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
    }


def _quality_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    positives = int(frontier.get("num_positive") or 0)
    negatives = int(frontier.get("num_negative") or 0)
    balance = -abs(positives - negatives)
    mean_delta = float(frontier.get("mean_reward") or 0.0)
    return (
        float(frontier.get("std_reward") or 0.0),
        float(frontier.get("max_abs_delta") or 0.0),
        balance,
        mean_delta,
    )


def _delta_stats(rows_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats = {}
    for task, rows in rows_by_task.items():
        means = []
        stds = []
        max_abs = []
        pos = 0
        neg = 0
        for row in rows:
            frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
            means.append(float(frontier.get("mean_reward") or 0.0))
            stds.append(float(frontier.get("std_reward") or 0.0))
            max_abs.append(float(frontier.get("max_abs_delta") or 0.0))
            pos += int(frontier.get("num_positive") or 0)
            neg += int(frontier.get("num_negative") or 0)
        stats[task] = {
            "rows": len(rows),
            "mean_delta_avg": sum(means) / len(means) if means else 0.0,
            "std_delta_avg": sum(stds) / len(stds) if stds else 0.0,
            "max_abs_delta_avg": sum(max_abs) / len(max_abs) if max_abs else 0.0,
            "positive_samples": pos,
            "negative_samples": neg,
        }
    return stats


def _sample_rewards(row: dict[str, Any]) -> list[float]:
    samples = row.get("samples")
    if not isinstance(samples, list):
        return []
    rewards = []
    for sample in samples:
        if isinstance(sample, dict) and "reward" in sample:
            rewards.append(float(sample.get("reward", 0.0)))
    return rewards


def _round_robin_interleave(rows_by_task: dict[str, list[dict[str, Any]]], tasks: list[str]) -> list[dict[str, Any]]:
    selected = []
    max_len = max((len(rows_by_task.get(task, [])) for task in tasks), default=0)
    for idx in range(max_len):
        for task in tasks:
            rows = rows_by_task.get(task, [])
            if idx < len(rows):
                selected.append(rows[idx])
    return selected


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
    return quotas


def _key_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rollouts", action="append", required=True)
    parser.add_argument("--baseline-rollouts", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="tool,memory,code")
    parser.add_argument("--per-task", type=int, default=16)
    parser.add_argument("--quota", action="append", default=[], help="Override per-task quota, e.g. memory=8. Repeatable.")
    parser.add_argument("--key", default="prompt_id")
    parser.add_argument("--baseline-agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--min-abs-delta", type=float, default=0.05)
    parser.add_argument("--min-delta-std", type=float, default=0.05)
    parser.add_argument("--require-cross-baseline", dest="require_cross_baseline", action="store_true", default=True)
    parser.add_argument("--no-require-cross-baseline", dest="require_cross_baseline", action="store_false")
    parser.add_argument("--require-source-keep-for-policy-loss", action="store_true")
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any quota is underfilled.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
