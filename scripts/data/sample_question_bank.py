#!/usr/bin/env python3
"""Sample calibration and guard manifests from an OP-VEC question bank."""

from __future__ import annotations

import argparse
import copy
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
from opvec.data.schema import stable_hash, validate_seed_record


DEFAULT_BUCKET_WEIGHTS = {
    "low_but_solvable": 0.40,
    "mid_frontier": 0.35,
    "all_fail_partial": 0.12,
    "high_not_all": 0.10,
    "all_fail_zero": 0.03,
}
DEFAULT_GUARD_BUCKETS = ("all_correct", "high_not_all")


def main() -> None:
    args = parse_args()
    prompts, guard, summary = sample_question_bank(args)
    if args.strict and summary["deficits"]:
        raise SystemExit(f"Not enough question-bank rows for strict selection: {summary['deficits']}")
    if not args.dry_run:
        prompt_path = f"{args.output_prefix}.prompts.jsonl"
        guard_path = f"{args.output_prefix}.guard.jsonl"
        summary_path = f"{args.output_prefix}.summary.json"
        summary["files"] = {
            "prompts": prompt_path,
            "guard": guard_path,
            "summary": summary_path,
        }
        summary["written"] = {
            "prompts": write_jsonl(prompt_path, prompts),
            "guard": write_jsonl(guard_path, guard),
        }
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def sample_question_bank(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    quotas = _parse_task_quotas(args.quota, tasks=tasks, default=args.per_task)
    guard_quotas = _parse_task_quotas(args.guard_quota, tasks=tasks, default=args.guard_per_task)
    weights = _parse_bucket_weights(args.bucket_weight)
    guard_buckets = tuple(item.strip() for item in args.guard_buckets.split(",") if item.strip())
    rows = read_jsonl(args.question_bank)
    pools = _group_rows(rows, tasks=tasks)
    rng = random.Random(args.seed)

    selected_bank_rows: list[dict[str, Any]] = []
    guard_bank_rows: list[dict[str, Any]] = []
    deficits: dict[str, Any] = {}

    used_prompt_ids: set[str] = set()
    selected_by_task_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        task_selected, task_deficits = _sample_task_main(
            pools.get(task, {}),
            quota=quotas[task],
            weights=weights,
            seed=args.seed,
            task=task,
            rng=rng,
        )
        for row in task_selected:
            used_prompt_ids.add(str(row.get("prompt_id")))
            selected_by_task_bucket[task][str(row.get("bucket"))] += 1
        selected_bank_rows.extend(task_selected)
        if task_deficits:
            deficits[f"main:{task}"] = task_deficits

    guard_by_task_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        task_guard, task_deficits = _sample_task_guard(
            pools.get(task, {}),
            quota=guard_quotas[task],
            guard_buckets=guard_buckets,
            used_prompt_ids=used_prompt_ids,
            seed=args.seed,
            task=task,
            rng=rng,
        )
        for row in task_guard:
            guard_by_task_bucket[task][str(row.get("bucket"))] += 1
            used_prompt_ids.add(str(row.get("prompt_id")))
        guard_bank_rows.extend(task_guard)
        if task_deficits:
            deficits[f"guard:{task}"] = task_deficits

    prompts = _round_robin_manifest(selected_bank_rows, tasks=tasks, seed=args.seed, selection_kind="main")
    guard = _round_robin_manifest(guard_bank_rows, tasks=tasks, seed=args.seed, selection_kind="guard")
    summary = {
        "format": "opvec_question_bank_sample_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question_bank": str(args.question_bank),
        "output_prefix": str(args.output_prefix),
        "seed": int(args.seed),
        "tasks": tasks,
        "quotas": quotas,
        "guard_quotas": guard_quotas,
        "bucket_weights": weights,
        "guard_buckets": list(guard_buckets),
        "input_rows": len(rows),
        "selected_rows": len(prompts),
        "guard_rows": len(guard),
        "selected_counts": dict(sorted(Counter(row.get("task") for row in prompts).items())),
        "guard_counts": dict(sorted(Counter(row.get("task") for row in guard).items())),
        "selected_bucket_counts": _nested_counter(selected_by_task_bucket),
        "guard_bucket_counts": _nested_counter(guard_by_task_bucket),
        "deficits": deficits,
    }
    return prompts, guard, summary


def _sample_task_main(
    bucket_pools: dict[str, list[dict[str, Any]]],
    *,
    quota: int,
    weights: dict[str, float],
    seed: int,
    task: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if quota <= 0:
        return [], {}
    wanted_by_bucket = _allocate_counts(quota, weights)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    deficits = {}
    for bucket, wanted in wanted_by_bucket.items():
        pool = _ranked_pool(bucket_pools.get(bucket, []), seed=seed, salt=f"{task}:{bucket}:main", rng=rng)
        take = [row for row in pool if str(row.get("prompt_id")) not in selected_ids][:wanted]
        selected.extend(take)
        selected_ids.update(str(row.get("prompt_id")) for row in take)
        if len(take) < wanted:
            deficits[bucket] = {"wanted": wanted, "got": len(take), "missing": wanted - len(take)}
    if len(selected) < quota:
        fallback = []
        for bucket in weights:
            fallback.extend(bucket_pools.get(bucket, []))
        fallback = _ranked_pool(fallback, seed=seed, salt=f"{task}:fallback:main", rng=rng)
        for row in fallback:
            prompt_id = str(row.get("prompt_id"))
            if prompt_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(prompt_id)
            if len(selected) >= quota:
                break
    if len(selected) < quota:
        deficits["_total"] = {"wanted": quota, "got": len(selected), "missing": quota - len(selected)}
    return selected[:quota], deficits


def _sample_task_guard(
    bucket_pools: dict[str, list[dict[str, Any]]],
    *,
    quota: int,
    guard_buckets: tuple[str, ...],
    used_prompt_ids: set[str],
    seed: int,
    task: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if quota <= 0:
        return [], {}
    pool = []
    for bucket in guard_buckets:
        pool.extend(bucket_pools.get(bucket, []))
    pool = _ranked_pool(pool, seed=seed, salt=f"{task}:guard", rng=rng)
    selected = []
    for row in pool:
        prompt_id = str(row.get("prompt_id"))
        if prompt_id in used_prompt_ids:
            continue
        selected.append(row)
        if len(selected) >= quota:
            break
    deficits = {}
    if len(selected) < quota:
        deficits["_total"] = {"wanted": quota, "got": len(selected), "missing": quota - len(selected)}
    return selected, deficits


def _group_rows(rows: list[dict[str, Any]], *, tasks: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    task_set = set(tasks)
    for row in rows:
        task = str(row.get("task") or "")
        bucket = str(row.get("bucket") or "")
        if task in task_set and bucket:
            pools[task][bucket].append(row)
    return pools


def _ranked_pool(rows: list[dict[str, Any]], *, seed: int, salt: str, rng: random.Random) -> list[dict[str, Any]]:
    output = list(rows)
    output.sort(key=lambda row: stable_hash({"seed": seed, "salt": salt, "prompt_id": row.get("prompt_id")}))
    rng.shuffle(output)
    output.sort(key=_quality_key, reverse=True)
    return output


def _quality_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    success_rate = _float(row.get("success_rate"), default=0.0)
    std_reward = _float(row.get("std_reward"), default=0.0)
    max_reward = _float(row.get("max_reward"), default=0.0)
    frontier_balance = 1.0 - abs(success_rate - 0.5) * 2.0
    return (std_reward, frontier_balance, max_reward, _float(row.get("mean_reward"), default=0.0))


def _round_robin_manifest(
    bank_rows: list[dict[str, Any]],
    *,
    tasks: list[str],
    seed: int,
    selection_kind: str,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank_rows:
        by_task[str(row.get("task"))].append(row)
    for task, rows in by_task.items():
        rows.sort(key=lambda row: stable_hash({"seed": seed, "selection": selection_kind, "prompt_id": row.get("prompt_id")}))
    output = []
    max_len = max((len(by_task.get(task, [])) for task in tasks), default=0)
    for idx in range(max_len):
        for task in tasks:
            rows = by_task.get(task, [])
            if idx < len(rows):
                output.append(_manifest_record(rows[idx], selection_kind=selection_kind))
    return output


def _manifest_record(bank_row: dict[str, Any], *, selection_kind: str) -> dict[str, Any]:
    prompt_record = copy.deepcopy(bank_row.get("prompt_record") or {})
    if not prompt_record:
        prompt_record = {
            "prompt_id": bank_row.get("prompt_id"),
            "task": bank_row.get("task"),
            "source": bank_row.get("source"),
            "source_row": bank_row.get("source_row"),
            "split": bank_row.get("split") or "question_bank_sample",
            "prompt": bank_row.get("prompt", ""),
            "messages": [],
            "reference": bank_row.get("reference", {}),
            "verifier": {"name": bank_row.get("official_reward_adapter") or f"{bank_row.get('task')}_reward_router"},
            "tags": [],
            "prompt_hash": bank_row.get("prompt_hash"),
        }
    prompt_record.setdefault("prompt_id", bank_row.get("prompt_id"))
    prompt_record.setdefault("task", bank_row.get("task"))
    prompt_record.setdefault("prompt_hash", bank_row.get("prompt_hash"))
    prompt_record.setdefault("reference", bank_row.get("reference", {}))
    verifier = prompt_record.get("verifier")
    if not isinstance(verifier, dict) or not verifier.get("name"):
        prompt_record["verifier"] = {"name": bank_row.get("official_reward_adapter") or f"{bank_row.get('task')}_reward_router"}
    prompt_record.setdefault("tags", [])
    tags = set(prompt_record.get("tags") or [])
    tags.update({"question_bank", f"bucket:{bank_row.get('bucket')}", f"selection:{selection_kind}"})
    prompt_record["tags"] = sorted(tags)
    prompt_record["question_bank_selection"] = {
        "selected_by": "sample_question_bank.py",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_kind": selection_kind,
        "question_bank_prompt_id": bank_row.get("prompt_id"),
        "bucket": bank_row.get("bucket"),
        "mean_reward": bank_row.get("mean_reward"),
        "std_reward": bank_row.get("std_reward"),
        "success_rate": bank_row.get("success_rate"),
        "success_count": bank_row.get("success_count"),
        "num_samples": bank_row.get("num_samples"),
        "baseline_rollout_path": bank_row.get("baseline_rollout_path"),
        "baseline_model": bank_row.get("baseline_model"),
    }
    validate_seed_record(prompt_record)
    return prompt_record


def _allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    positive = {bucket: max(float(weight), 0.0) for bucket, weight in weights.items() if float(weight) > 0.0}
    if not positive:
        raise ValueError("At least one bucket weight must be positive")
    weight_sum = sum(positive.values())
    raw = {bucket: total * weight / weight_sum for bucket, weight in positive.items()}
    counts = {bucket: int(value) for bucket, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda bucket: (raw[bucket] - counts[bucket], raw[bucket]), reverse=True)
    for bucket in order[:remainder]:
        counts[bucket] += 1
    return counts


def _parse_task_quotas(items: list[str], *, tasks: list[str], default: int) -> dict[str, int]:
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


def _parse_bucket_weights(items: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_BUCKET_WEIGHTS)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected bucket=weight, got: {item}")
        bucket, value = item.split("=", 1)
        weights[bucket.strip()] = float(value)
    return weights


def _nested_counter(payload: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {task: dict(sorted(counter.items())) for task, counter in sorted(payload.items())}


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-bank", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tasks", default="tool,memory,code")
    parser.add_argument("--per-task", type=int, default=0)
    parser.add_argument("--quota", action="append", default=[], help="Task quota, e.g. tool=34. Repeatable.")
    parser.add_argument("--guard-per-task", type=int, default=3)
    parser.add_argument("--guard-quota", action="append", default=[], help="Guard task quota, e.g. memory=2. Repeatable.")
    parser.add_argument("--bucket-weight", action="append", default=[], help="Bucket sampling weight, e.g. mid_frontier=0.5.")
    parser.add_argument("--guard-buckets", default=",".join(DEFAULT_GUARD_BUCKETS))
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
