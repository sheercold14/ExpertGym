#!/usr/bin/env python3
"""Build a fixed high-information calibration bundle for OP-VEC training.

The bundle separates three roles:

* prompts: fixed prompt set for on-policy rollout + GRPO;
* distill: fixed expert-recovery rollout rows for best-response/pairwise loss;
* guard: fixed prompts reserved for sanity/retention checks.

This script only selects and annotates existing rows. It does not score model
outputs, bake checkpoints, or train gates.
"""

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

TASKS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args)
    if args.strict and bundle["summary"]["deficits"]:
        raise SystemExit(f"High-info calibration deficits: {bundle['summary']['deficits']}")
    if not args.dry_run:
        write_bundle(args.output_prefix, bundle)
    print(json.dumps(bundle["summary"], ensure_ascii=False, indent=2, sort_keys=True))


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    prompt_quotas = _parse_quotas(args.prompt_quota, defaults={"tool": 30, "memory": 35, "code": 25})
    guard_quotas = _parse_quotas(args.guard_quota, defaults={"tool": 4, "memory": 3, "code": 3})
    distill_quotas = _parse_quotas(args.distill_quota, defaults={"tool": 13, "memory": 13, "code": 13})

    source_rows = read_jsonl(args.source_manifest)
    frontier_rows = _read_many(args.frontier_rollout)
    distill_pool_rows = _read_many(args.distill_rollout)

    frontier_prompt_ids = _frontier_prompt_ids(frontier_rows)
    prompts, prompt_stats = _select_prompt_rows(
        source_rows,
        frontier_prompt_ids=frontier_prompt_ids,
        prompt_quotas=prompt_quotas,
        guard_quotas=guard_quotas,
        seed=args.seed,
        created_at=created_at,
    )
    distill_rows, distill_stats = _select_distill_rows(
        distill_pool_rows,
        distill_quotas=distill_quotas,
        seed=args.seed,
        created_at=created_at,
        allow_all_success=args.allow_distill_all_success,
    )

    output_prefix = str(Path(args.output_prefix).expanduser())
    files = {
        "prompts": f"{output_prefix}.prompts.jsonl",
        "distill": f"{output_prefix}.distill.jsonl",
        "guard": f"{output_prefix}.guard.jsonl",
        "bundle": f"{output_prefix}.bundle.json",
        "summary": f"{output_prefix}.summary.json",
    }
    deficits: dict[str, Any] = {}
    deficits.update(_quota_deficits("prompts", prompt_quotas, prompts["main"]))
    deficits.update(_quota_deficits("guard", guard_quotas, prompts["guard"]))
    deficits.update(_quota_deficits("distill", distill_quotas, distill_rows))

    summary = {
        "format": "opvec_high_info_calibration_bundle_v1",
        "created_at": created_at,
        "seed": args.seed,
        "source_manifest": str(args.source_manifest),
        "frontier_rollouts": [str(path) for path in args.frontier_rollout],
        "distill_rollouts": [str(path) for path in args.distill_rollout],
        "files": files,
        "quotas": {
            "prompts": prompt_quotas,
            "guard": guard_quotas,
            "distill": distill_quotas,
        },
        "counts": {
            "prompts": _task_counts(prompts["main"]),
            "guard": _task_counts(prompts["guard"]),
            "distill": _task_counts(distill_rows),
        },
        "rows": {
            "source_manifest": len(source_rows),
            "frontier_pool": len(frontier_rows),
            "distill_pool": len(distill_pool_rows),
            "prompts": len(prompts["main"]),
            "guard": len(prompts["guard"]),
            "distill": len(distill_rows),
        },
        "deficits": deficits,
        "prompt_selection": prompt_stats,
        "distill_selection": distill_stats,
        "training_contract": {
            "stage_a": "Use *.prompts.jsonl as --seed-manifest for on-policy rollout; optimize PPO/GRPO only on current-policy samples.",
            "stage_b": "Use *.distill.jsonl with --ppo-loss-weight 0 and best-response/pairwise loss; do not treat expert samples as PPO on-policy samples.",
            "guard": "Use *.guard.jsonl for fixed sanity/retention checks before accepting a checkpoint.",
        },
    }
    return {
        "prompts": prompts["main"],
        "guard": prompts["guard"],
        "distill": distill_rows,
        "summary": summary,
    }


def write_bundle(output_prefix: str, bundle: dict[str, Any]) -> None:
    prefix = Path(output_prefix).expanduser()
    files = bundle["summary"]["files"]
    write_jsonl(files["prompts"], bundle["prompts"])
    write_jsonl(files["distill"], bundle["distill"])
    write_jsonl(files["guard"], bundle["guard"])
    payload = {
        "format": "opvec_high_info_calibration_bundle_v1",
        "summary": bundle["summary"],
    }
    Path(files["bundle"]).parent.mkdir(parents=True, exist_ok=True)
    Path(files["bundle"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(files["summary"]).write_text(json.dumps(bundle["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_prompt_rows(
    source_rows: list[dict[str, Any]],
    *,
    frontier_prompt_ids: dict[str, set[str]],
    prompt_quotas: dict[str, int],
    guard_quotas: dict[str, int],
    seed: int,
    created_at: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        task = str(row.get("task") or "")
        if task in TASKS and row.get("prompt_id"):
            rows_by_task[task].append(row)

    selected_main: list[dict[str, Any]] = []
    selected_guard: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    stats: dict[str, Any] = {"frontier_seeded": {}, "fill": {}, "available": {}}

    for task in TASKS:
        task_rows = _stable_shuffle(rows_by_task.get(task, []), seed=seed, salt=f"prompt:{task}")
        by_id = {str(row.get("prompt_id")): row for row in task_rows}
        frontier_ids = [prompt_id for prompt_id in sorted(frontier_prompt_ids.get(task, set())) if prompt_id in by_id]
        frontier_ids = sorted(frontier_ids, key=lambda value: stable_hash({"seed": seed, "task": task, "prompt_id": value}))
        want = int(prompt_quotas.get(task, 0))
        chosen: list[dict[str, Any]] = []
        for prompt_id in frontier_ids:
            if len(chosen) >= want:
                break
            row = _annotate_prompt_row(
                by_id[prompt_id],
                role="on_policy_frontier_probe",
                reason="prompt had current-policy reward variance in a prior frontier rollout",
                expected_axis=task,
                created_at=created_at,
            )
            chosen.append(row)
            selected_ids.add(prompt_id)
        fill_candidates = [row for row in task_rows if str(row.get("prompt_id")) not in selected_ids]
        for row in fill_candidates:
            if len(chosen) >= want:
                break
            prompt_id = str(row.get("prompt_id"))
            chosen.append(
                _annotate_prompt_row(
                    row,
                    role="on_policy_official_correct_probe",
                    reason="official/routed correct prompt used to expose on-policy reward or failure modes",
                    expected_axis=task,
                    created_at=created_at,
                )
            )
            selected_ids.add(prompt_id)
        selected_main.extend(chosen)
        stats["frontier_seeded"][task] = sum(
            1 for row in chosen if row.get("high_info_calibration", {}).get("role") == "on_policy_frontier_probe"
        )
        stats["fill"][task] = len(chosen) - stats["frontier_seeded"][task]
        stats["available"][task] = len(task_rows)

    for task in TASKS:
        guard_want = int(guard_quotas.get(task, 0))
        candidates = [
            row
            for row in _stable_shuffle(rows_by_task.get(task, []), seed=seed, salt=f"guard:{task}")
            if str(row.get("prompt_id")) not in selected_ids
        ]
        for row in candidates[:guard_want]:
            selected_guard.append(
                _annotate_prompt_row(
                    row,
                    role="guard_retention_probe",
                    reason="held out from main GRPO prompts for fixed sanity/retention checks",
                    expected_axis=task,
                    created_at=created_at,
                )
            )
            selected_ids.add(str(row.get("prompt_id")))

    return {
        "main": _round_robin_interleave(_group_by_task(selected_main)),
        "guard": _round_robin_interleave(_group_by_task(selected_guard)),
    }, stats


def _select_distill_rows(
    rows: list[dict[str, Any]],
    *,
    distill_quotas: dict[str, int],
    seed: int,
    created_at: str,
    allow_all_success: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for row in rows:
        task = str(row.get("task") or "")
        if task not in TASKS:
            skipped["task_not_requested"] += 1
            continue
        if not row.get("prompt_id"):
            skipped["missing_prompt_id"] += 1
            continue
        rewards = _sample_rewards(row)
        if len(rewards) < 2:
            skipped["too_few_rewarded_samples"] += 1
            continue
        if not any(value > 0.0 for value in rewards):
            skipped["no_positive_sample"] += 1
            continue
        if not allow_all_success and all(value > 0.0 for value in rewards):
            skipped["all_success"] += 1
            continue
        if len(set(rewards)) < 2:
            skipped["no_reward_variance"] += 1
            continue
        pools[task].append(row)

    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for task in TASKS:
        pool = _stable_shuffle(pools.get(task, []), seed=seed, salt=f"distill:{task}")
        pool.sort(key=_distill_quality_key, reverse=True)
        selected: list[dict[str, Any]] = []
        for row in pool:
            prompt_id = str(row.get("prompt_id"))
            if prompt_id in seen_ids:
                continue
            selected.append(
                _annotate_distill_row(
                    row,
                    created_at=created_at,
                    reason="expert/positive sample contrasts against weaker samples and supplies a differentiable recovery direction",
                )
            )
            seen_ids.add(prompt_id)
            if len(selected) >= int(distill_quotas.get(task, 0)):
                break
        selected_by_task[task] = selected

    stats = {
        "candidate_counts": {task: len(pools.get(task, [])) for task in TASKS},
        "skipped": dict(sorted(skipped.items())),
        "reward_stats": _reward_stats(selected_by_task),
    }
    return _round_robin_interleave(selected_by_task), stats


def _frontier_prompt_ids(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_task: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        task = str(row.get("task") or "")
        prompt_id = row.get("prompt_id")
        if task not in TASKS or not prompt_id:
            continue
        if row.get("keep_for_policy_loss") and _has_variance(row):
            by_task[task].add(str(prompt_id))
    return by_task


def _annotate_prompt_row(row: dict[str, Any], *, role: str, reason: str, expected_axis: str, created_at: str) -> dict[str, Any]:
    output = copy.deepcopy(row)
    output["high_info_calibration"] = {
        "role": role,
        "selected_at": created_at,
        "expected_axis": expected_axis,
        "why_selected": reason,
        "training_use": "seed_manifest_for_on_policy_rollout" if role != "guard_retention_probe" else "guard_sanity_only",
    }
    return output


def _annotate_distill_row(row: dict[str, Any], *, created_at: str, reason: str) -> dict[str, Any]:
    output = copy.deepcopy(row)
    output["high_info_calibration"] = {
        "role": "expert_recovery_distill",
        "selected_at": created_at,
        "expected_axis": str(row.get("task")),
        "why_selected": reason,
        "training_use": "offline_best_response_or_pairwise_only",
        "ppo_allowed": False,
    }
    return output


def _distill_quality_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    rewards = _sample_rewards(row)
    std = pstdev(rewards) if len(rewards) > 1 else 0.0
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    frontier_weight = _as_float(frontier.get("frontier_weight"), default=0.0)
    positives = sum(1 for value in rewards if value > 0.0)
    negatives = sum(1 for value in rewards if value <= 0.0)
    balance = -abs(positives - negatives)
    max_reward = max(rewards) if rewards else 0.0
    return (std, frontier_weight, balance, max_reward)


def _sample_rewards(row: dict[str, Any]) -> list[float]:
    rewards = []
    samples = row.get("samples")
    if not isinstance(samples, list):
        return rewards
    for sample in samples:
        if not isinstance(sample, dict) or "reward" not in sample:
            continue
        value = sample.get("reward")
        if isinstance(value, list):
            continue
        try:
            rewards.append(float(value))
        except (TypeError, ValueError):
            continue
    return rewards


def _has_variance(row: dict[str, Any]) -> bool:
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    if frontier.get("has_variance") is False:
        return False
    rewards = _sample_rewards(row)
    return len(set(rewards)) >= 2


def _reward_stats(rows_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats = {}
    for task in TASKS:
        rows = rows_by_task.get(task, [])
        row_means = []
        row_stds = []
        positives = 0
        negatives = 0
        samples = 0
        for row in rows:
            rewards = _sample_rewards(row)
            if rewards:
                row_means.append(mean(rewards))
                row_stds.append(pstdev(rewards) if len(rewards) > 1 else 0.0)
                positives += sum(1 for value in rewards if value > 0.0)
                negatives += sum(1 for value in rewards if value <= 0.0)
                samples += len(rewards)
        stats[task] = {
            "rows": len(rows),
            "samples": samples,
            "positive_samples": positives,
            "nonpositive_samples": negatives,
            "mean_reward_avg": sum(row_means) / len(row_means) if row_means else 0.0,
            "std_reward_avg": sum(row_stds) / len(row_stds) if row_stds else 0.0,
        }
    return stats


def _quota_deficits(prefix: str, quotas: dict[str, int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _task_counts(rows)
    deficits = {}
    for task, want in quotas.items():
        got = int(counts.get(task, 0))
        if got < int(want):
            deficits[f"{prefix}:{task}"] = {"wanted": int(want), "got": got, "missing": int(want) - got}
    return deficits


def _task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("task")) for row in rows).items()))


def _read_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            copied = dict(row)
            copied.setdefault("high_info_source_file", str(path))
            rows.append(copied)
    return rows


def _group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task"))].append(row)
    return grouped


def _round_robin_interleave(rows_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    max_len = max((len(rows_by_task.get(task, [])) for task in TASKS), default=0)
    for idx in range(max_len):
        for task in TASKS:
            rows = rows_by_task.get(task, [])
            if idx < len(rows):
                output.append(rows[idx])
    return output


def _stable_shuffle(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    output = list(rows)
    rng = random.Random(stable_hash({"seed": seed, "salt": salt})[:16])
    output.sort(key=lambda row: stable_hash({"seed": seed, "salt": salt, "prompt_id": row.get("prompt_id")}))
    rng.shuffle(output)
    return output


def _parse_quotas(items: list[str], *, defaults: dict[str, int]) -> dict[str, int]:
    quotas = dict(defaults)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected task=count quota, got: {item}")
        task, value = item.split("=", 1)
        task = task.strip()
        if task not in TASKS:
            raise ValueError(f"Unknown quota task: {task}")
        quotas[task] = int(value)
    return quotas


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, help="Official prompt seed manifest for on-policy rollout.")
    parser.add_argument("--frontier-rollout", action="append", default=[], help="Prior rollout JSONL used only to seed known frontier prompt ids.")
    parser.add_argument("--distill-rollout", action="append", default=[], help="Expert-recovery rollout JSONL for offline distill rows.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix; writes .prompts/.distill/.guard/.summary files.")
    parser.add_argument("--prompt-quota", action="append", default=[], help="Main prompt quota, e.g. tool=30. Repeatable.")
    parser.add_argument("--guard-quota", action="append", default=[], help="Guard prompt quota, e.g. tool=4. Repeatable.")
    parser.add_argument("--distill-quota", action="append", default=[], help="Distill row quota, e.g. tool=16. Repeatable.")
    parser.add_argument("--allow-distill-all-success", action="store_true", help="Allow all-positive distill rows; default keeps contrastive rows only.")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
