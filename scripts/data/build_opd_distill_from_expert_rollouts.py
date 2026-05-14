#!/usr/bin/env python3
"""Build OPD distillation rows from current failures and expert successes."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import stable_hash


def main() -> None:
    args = parse_args()
    summary = build_opd_distill(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_opd_distill(args: argparse.Namespace) -> dict[str, Any]:
    current_rows = _read_many(args.current_rollouts)
    expert_rows = _read_many(args.expert_rollouts)
    tasks = _parse_tasks(args.tasks)
    quotas = {task: int(args.per_task) for task in tasks}
    quotas.update(_parse_quotas(args.quota))
    rng = random.Random(int(args.seed))

    expert_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in expert_rows:
        if str(row.get("task")) not in tasks:
            continue
        key_value = _row_key(row, args.key)
        if key_value:
            expert_by_key[key_value].append(row)

    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    ordered_current = list(current_rows)
    rng.shuffle(ordered_current)
    for current in ordered_current:
        task = str(current.get("task"))
        if task not in tasks:
            skipped["task_filtered"] += 1
            continue
        key_value = _row_key(current, args.key)
        if not key_value:
            skipped["missing_key"] += 1
            continue
        if not _is_current_failure(
            current,
            task=task,
            max_success=int(args.current_max_success),
            positive_threshold=float(args.positive_threshold),
        ):
            skipped["current_not_failure"] += 1
            continue
        positives = _select_expert_positives(
            expert_by_key.get(key_value, []),
            task=task,
            threshold=float(args.positive_threshold),
            limit=int(args.max_positives_per_row),
        )
        if not positives:
            skipped["no_expert_positive"] += 1
            continue
        negatives = _select_current_negatives(
            current,
            task=task,
            threshold=float(args.positive_threshold),
            limit=int(args.max_negatives_per_row),
        )
        if not negatives:
            skipped["no_current_negative"] += 1
            continue
        candidates_by_task[task].append(
            _make_distill_row(
                current,
                positives,
                negatives,
                key_name=str(args.key),
                key_value=key_value,
                positive_threshold=float(args.positive_threshold),
            )
        )

    selected: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = sorted(
            candidates_by_task.get(task, []),
            key=lambda row: stable_hash([row.get("task"), row.get("prompt_id"), row.get("opd")]),
        )
        limit = quotas.get(task, 0)
        if limit < 0:
            selected.extend(task_rows)
        else:
            selected.extend(task_rows[:limit])

    if not args.dry_run:
        write_jsonl(args.output, selected)
        summary_path = Path(args.output).with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(_summary(args, selected, candidates_by_task, skipped), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return _summary(args, selected, candidates_by_task, skipped)


def _read_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in str(value).split(",") if item.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task")
    return tasks


def _parse_quotas(items: list[str]) -> dict[str, int]:
    quotas = {}
    for item in items:
        for part in item.split(","):
            value = part.strip()
            if not value:
                continue
            if "=" not in value:
                raise ValueError(f"Invalid --quota value: {value}")
            task, limit = value.split("=", 1)
            quotas[task.strip()] = int(limit)
    return quotas


def _row_key(row: dict[str, Any], key_name: str) -> str:
    if key_name == "prompt_id":
        return str(row.get("prompt_id") or "")
    if key_name == "group_id":
        direct = row.get("group_id")
        if direct:
            return str(direct)
        reference = row.get("reference") if isinstance(row.get("reference"), dict) else {}
        group_id = reference.get("group_id") or reference.get("id")
        if group_id:
            return str(group_id)
        return str(row.get("prompt_id") or "")
    raise ValueError(f"Unknown key: {key_name}")


def _is_current_failure(
    row: dict[str, Any],
    *,
    task: str,
    max_success: int,
    positive_threshold: float,
) -> bool:
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    if bool(frontier.get("all_failure")):
        return True
    if frontier.get("num_success") is not None:
        return int(frontier.get("num_success") or 0) <= int(max_success)
    successes = sum(
        1
        for sample in row.get("samples", [])
        if _sample_train_reward(sample, task=task) >= float(positive_threshold)
    )
    return successes <= int(max_success)


def _select_expert_positives(
    rows: list[dict[str, Any]],
    *,
    task: str,
    threshold: float,
    limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for row in rows:
        for sample in row.get("samples", []):
            if not sample.get("text"):
                continue
            reward = _sample_train_reward(sample, task=task)
            if reward >= float(threshold):
                candidates.append((row, sample, reward))
    candidates.sort(
        key=lambda item: (
            -float(item[2]),
            stable_hash([item[0].get("prompt_id"), item[0].get("policy_id"), item[1].get("sample_id")]),
        )
    )
    return candidates if limit < 0 else candidates[:limit]


def _select_current_negatives(
    row: dict[str, Any],
    *,
    task: str,
    threshold: float,
    limit: int,
) -> list[tuple[dict[str, Any], float]]:
    candidates: list[tuple[dict[str, Any], float]] = []
    for sample in row.get("samples", []):
        if not sample.get("text"):
            continue
        reward = _sample_train_reward(sample, task=task)
        if reward < float(threshold):
            candidates.append((sample, reward))
    candidates.sort(key=lambda item: (float(item[1]), stable_hash(item[0].get("sample_id"))))
    return candidates if limit < 0 else candidates[:limit]


def _make_distill_row(
    current: dict[str, Any],
    positives: list[tuple[dict[str, Any], dict[str, Any], float]],
    negatives: list[tuple[dict[str, Any], float]],
    *,
    key_name: str,
    key_value: str,
    positive_threshold: float,
) -> dict[str, Any]:
    row = copy.deepcopy(current)
    current_prompt_id = str(current.get("prompt_id") or stable_hash(current)[:16])
    row["run_id"] = "opd_distill"
    row["policy_id"] = "opd_same_prompt_expert_recovery"
    row["source"] = "opd_same_prompt_expert_recovery"
    row["keep_for_policy_loss"] = False
    row["skip_reason"] = None
    samples = []
    expert_policy_ids = []
    for index, (expert_row, sample, reward) in enumerate(positives):
        cloned = copy.deepcopy(sample)
        expert_policy_ids.append(str(expert_row.get("policy_id") or "expert"))
        cloned["sample_id"] = f"opd_pos_{stable_hash([current_prompt_id, 'pos', index, cloned.get('sample_id')])[:16]}"
        cloned["reward_train"] = float(reward)
        cloned["opd_role"] = "positive"
        cloned["opd_source"] = "expert"
        cloned["opd_source_policy_id"] = str(expert_row.get("policy_id") or "expert")
        samples.append(cloned)
    for index, (sample, reward) in enumerate(negatives):
        cloned = copy.deepcopy(sample)
        cloned["sample_id"] = f"opd_neg_{stable_hash([current_prompt_id, 'neg', index, cloned.get('sample_id')])[:16]}"
        cloned["reward_train"] = float(reward)
        cloned["opd_role"] = "negative"
        cloned["opd_source"] = "current"
        cloned["opd_source_policy_id"] = str(current.get("policy_id") or "current")
        samples.append(cloned)
    rewards = [float(sample.get("reward_train", 0.0)) for sample in samples]
    row["samples"] = samples
    row["frontier"] = {
        "all_failure": False,
        "all_success": False,
        "num_success": len(positives),
        "num_failure": len(negatives),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "std_reward": _std(rewards),
        "reward_field": "reward_train",
        "opd_distill": True,
    }
    row["opd"] = {
        "format": "opd_same_prompt_expert_recovery_v1",
        "key": key_name,
        "key_value": key_value,
        "current_prompt_id": current.get("prompt_id"),
        "current_policy_id": current.get("policy_id"),
        "expert_policy_ids": sorted(set(expert_policy_ids)),
        "positive_threshold": float(positive_threshold),
        "num_positives": len(positives),
        "num_negatives": len(negatives),
    }
    return row


def _sample_train_reward(sample: dict[str, Any], *, task: str | None = None) -> float:
    if "reward_train" in sample:
        return float(sample.get("reward_train", 0.0))
    raw = float(sample.get("reward", 0.0))
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    score_range = details.get("toolrl_score_range") if isinstance(details, dict) else None
    if task == "tool" or score_range == [-3.0, 4.0]:
        return max(0.0, min((raw + 3.0) / 7.0, 1.0))
    return max(0.0, min(raw, 1.0))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _summary(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    candidates_by_task: dict[str, list[dict[str, Any]]],
    skipped: Counter,
) -> dict[str, Any]:
    return {
        "format": "opd_distill_builder_summary_v1",
        "current_rollouts": list(args.current_rollouts),
        "expert_rollouts": list(args.expert_rollouts),
        "output": str(args.output),
        "tasks": _parse_tasks(args.tasks),
        "key": str(args.key),
        "positive_threshold": float(args.positive_threshold),
        "current_max_success": int(args.current_max_success),
        "selected_rows": len(selected),
        "selected_task_counts": dict(Counter(str(row.get("task")) for row in selected)),
        "candidate_task_counts": {task: len(rows) for task, rows in candidates_by_task.items()},
        "skipped": dict(skipped),
        "dry_run": bool(args.dry_run),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-rollouts", action="append", required=True)
    parser.add_argument("--expert-rollouts", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="memory,code")
    parser.add_argument("--key", choices=["prompt_id", "group_id"], default="prompt_id")
    parser.add_argument("--current-max-success", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--max-positives-per-row", type=int, default=2)
    parser.add_argument("--max-negatives-per-row", type=int, default=4)
    parser.add_argument("--per-task", type=int, default=64)
    parser.add_argument("--quota", action="append", default=[], help="Per-task output quota, e.g. memory=64. Repeatable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
