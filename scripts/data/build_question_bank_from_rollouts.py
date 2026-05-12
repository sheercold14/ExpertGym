#!/usr/bin/env python3
"""Build an auditable question bank from baseline rollout rewards."""

from __future__ import annotations

import argparse
import json
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
    rows, summary = build_question_bank(args)
    if not args.dry_run:
        written = write_jsonl(args.output, rows)
        summary["written"] = written
        summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_question_bank(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_records = _load_prompt_records(args.seed_manifest)
    rollout_meta = _load_rollout_meta(args.rollouts)
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    seen_prompt_ids: set[str] = set()

    for rollout_path in args.rollouts:
        for row in read_jsonl(rollout_path):
            prompt_id = str(row.get("prompt_id") or "")
            if not prompt_id:
                skipped["missing_prompt_id"] += 1
                continue
            if not args.allow_duplicate_prompts and prompt_id in seen_prompt_ids:
                skipped["duplicate_prompt_id"] += 1
                continue
            bank_row = _bank_row(
                row,
                rollout_path=str(rollout_path),
                rollout_meta=rollout_meta.get(str(rollout_path), {}),
                prompt_record=prompt_records.get(prompt_id),
                args=args,
            )
            if bank_row is None:
                skipped["invalid_samples"] += 1
                continue
            seen_prompt_ids.add(prompt_id)
            rows.append(bank_row)

    summary = {
        "format": "opvec_question_bank_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rollouts": [str(path) for path in args.rollouts],
        "seed_manifests": [str(path) for path in args.seed_manifest],
        "output": str(args.output),
        "rows": len(rows),
        "skipped": dict(sorted(skipped.items())),
        "success_mode": args.success_mode,
        "thresholds": {
            "reward_threshold": float(args.reward_threshold),
            "mid_success_rate": float(args.mid_success_rate),
            "high_success_rate": float(args.high_success_rate),
            "partial_std_threshold": float(args.partial_std_threshold),
        },
        "task_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
        "bucket_counts": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "task_bucket_counts": _task_bucket_counts(rows),
        "reward_stats": _reward_stats(rows),
        "rollout_meta": rollout_meta,
    }
    return rows, summary


def _bank_row(
    row: dict[str, Any],
    *,
    rollout_path: str,
    rollout_meta: dict[str, Any],
    prompt_record: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    reward_list = []
    success_list = []
    sample_summaries = []
    reward_adapters = Counter()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            return None
        reward = _float(sample.get("reward"), default=0.0)
        success = _sample_success(sample, reward=reward, args=args)
        reward_list.append(reward)
        success_list.append(success)
        details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
        reward_source = details.get("reward_source") or details.get("reward_definition")
        if reward_source:
            reward_adapters[str(reward_source)] += 1
        sample_summaries.append(
            {
                "sample_id": sample.get("sample_id", f"{row.get('prompt_id', 'prompt')}__k{index}"),
                "reward": reward,
                "success": success,
                "length": _float(sample.get("length"), default=0.0),
                "details": _compact_details(details),
            }
        )

    stats = _prompt_stats(reward_list, success_list)
    bucket, bucket_reason = _bucket(stats, args=args)
    prompt_payload = prompt_record or _fallback_prompt_record(row)
    source = prompt_payload.get("source") or row.get("source") or ""
    source_row = prompt_payload.get("source_row")
    prompt_hash = prompt_payload.get("prompt_hash") or stable_hash(
        {"prompt_id": row.get("prompt_id"), "prompt": row.get("prompt"), "reference": row.get("reference")}
    )
    task = str(row.get("task") or prompt_payload.get("task") or "unknown")
    official_adapter = _official_reward_adapter(row, prompt_payload, reward_adapters)
    gate_values = row.get("gate_values") if isinstance(row.get("gate_values"), dict) else {}
    return {
        "format": "opvec_question_bank_row_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question_id": str(row.get("prompt_id")),
        "prompt_id": str(row.get("prompt_id")),
        "task": task,
        "source": source,
        "source_path": source,
        "source_row": source_row,
        "split": prompt_payload.get("split"),
        "prompt_hash": prompt_hash,
        "baseline_model": rollout_meta.get("policy_model") or row.get("policy_id"),
        "baseline_rollout_path": rollout_path,
        "baseline_run_id": row.get("run_id"),
        "gate_checkpoint": row.get("gate_checkpoint"),
        "gate_values": gate_values,
        "samples_per_prompt": len(samples),
        "temperature": rollout_meta.get("temperature"),
        "top_p": rollout_meta.get("top_p"),
        "seed": rollout_meta.get("seed"),
        "reward_list": reward_list,
        "success_list": success_list,
        **stats,
        "bucket": bucket,
        "bucket_reason": bucket_reason,
        "recommended_use": _recommended_use(bucket),
        "official_reward_adapter": official_adapter,
        "selected_from_manifest": prompt_record is not None,
        "prompt_record": prompt_payload,
        "frontier": row.get("frontier") if isinstance(row.get("frontier"), dict) else {},
        "keep_for_policy_loss": bool(row.get("keep_for_policy_loss")),
        "skip_reason": row.get("skip_reason"),
        "sample_summaries": sample_summaries,
        "trace": {
            "source_rollout": rollout_path,
            "rollout_step": row.get("step"),
            "run_id": row.get("run_id"),
            "policy_id": row.get("policy_id"),
            "created_at": row.get("created_at"),
        },
    }


def _load_prompt_records(paths: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            prompt_id = str(row.get("prompt_id") or "")
            if prompt_id and prompt_id not in records:
                copied = dict(row)
                copied.setdefault("source_manifest", str(path))
                records[prompt_id] = copied
    return records


def _load_rollout_meta(paths: list[str]) -> dict[str, dict[str, Any]]:
    meta = {}
    for path in paths:
        rollout_path = Path(path)
        summary_path = rollout_path.with_suffix(".summary.json")
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        meta[str(path)] = {
            "policy_model": payload.get("policy_model"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "seed": payload.get("seed"),
            "reward_definition": payload.get("reward_definition"),
            "tensor_parallel_size": payload.get("tensor_parallel_size"),
            "max_model_len": payload.get("max_model_len"),
        }
    return meta


def _fallback_prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": row.get("prompt_id"),
        "task": row.get("task"),
        "source": row.get("source") or row.get("run_id") or row.get("policy_id"),
        "source_row": row.get("step"),
        "split": "question_bank_rollout",
        "prompt": row.get("prompt", ""),
        "messages": row.get("messages", []),
        "reference": row.get("reference", {}),
        "verifier": row.get("verifier", {}),
        "tags": ["question_bank_fallback"],
        "prompt_hash": stable_hash(
            {"prompt_id": row.get("prompt_id"), "prompt": row.get("prompt"), "reference": row.get("reference")}
        ),
    }


def _prompt_stats(rewards: list[float], successes: list[bool]) -> dict[str, Any]:
    success_count = sum(1 for value in successes if value)
    samples = len(rewards)
    return {
        "mean_reward": mean(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "success_count": success_count,
        "success_rate": float(success_count) / float(samples) if samples else 0.0,
        "num_samples": samples,
    }


def _bucket(stats: dict[str, Any], *, args: argparse.Namespace) -> tuple[str, str]:
    samples = int(stats["num_samples"])
    success_count = int(stats["success_count"])
    success_rate = float(stats["success_rate"])
    std_reward = float(stats["std_reward"])
    if samples > 0 and success_count == samples:
        return "all_correct", "success_count == num_samples"
    if success_rate >= float(args.high_success_rate):
        return "high_not_all", f"{args.high_success_rate} <= success_rate < 1.0"
    if success_rate >= float(args.mid_success_rate):
        return "mid_frontier", f"{args.mid_success_rate} <= success_rate < {args.high_success_rate}"
    if success_count > 0:
        return "low_but_solvable", f"0 < success_rate < {args.mid_success_rate}"
    if std_reward >= float(args.partial_std_threshold):
        return "all_fail_partial", f"success_count == 0 and std_reward >= {args.partial_std_threshold}"
    return "all_fail_zero", f"success_count == 0 and std_reward < {args.partial_std_threshold}"


def _recommended_use(bucket: str) -> str:
    if bucket in {"low_but_solvable", "mid_frontier", "all_fail_partial"}:
        return "raw_grpo"
    if bucket in {"high_not_all", "all_correct"}:
        return "guard_or_self_compare"
    return "expert_recovery"


def _sample_success(sample: dict[str, Any], *, reward: float, args: argparse.Namespace) -> bool:
    if args.success_mode in {"field", "field_or_positive"} and "success" in sample:
        return bool(sample.get("success"))
    if args.success_mode == "field":
        return False
    return reward > float(args.reward_threshold)


def _compact_details(details: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "reward_source",
        "reward_definition",
        "parseable",
        "prediction_calls",
        "reference_calls",
        "exact_tool_match",
        "tool_call_parse_error",
        "em",
        "f1",
        "passed",
        "pass_rate",
    }
    return {key: value for key, value in details.items() if key in keep}


def _official_reward_adapter(row: dict[str, Any], prompt_record: dict[str, Any], reward_adapters: Counter[str]) -> str:
    verifier = prompt_record.get("verifier") if isinstance(prompt_record.get("verifier"), dict) else {}
    verifier_name = verifier.get("name")
    if verifier_name:
        return str(verifier_name)
    if reward_adapters:
        return reward_adapters.most_common(1)[0][0]
    task = str(row.get("task") or "unknown")
    return f"{task}_reward_router"


def _task_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row["task"])][str(row["bucket"])] += 1
    return {task: dict(sorted(bucket_counts.items())) for task, bucket_counts in sorted(counts.items())}


def _reward_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    for task, task_rows in sorted(by_task.items()):
        stats[task] = {
            "rows": len(task_rows),
            "mean_reward_avg": mean([float(row["mean_reward"]) for row in task_rows]) if task_rows else 0.0,
            "std_reward_avg": mean([float(row["std_reward"]) for row in task_rows]) if task_rows else 0.0,
            "success_rate_avg": mean([float(row["success_rate"]) for row in task_rows]) if task_rows else 0.0,
        }
    return stats


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", action="append", required=True, help="Input baseline rollout JSONL. Repeatable.")
    parser.add_argument("--seed-manifest", action="append", default=[], help="Original seed manifest used by rollout. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--success-mode", choices=["field_or_positive", "field", "positive"], default="field_or_positive")
    parser.add_argument("--reward-threshold", type=float, default=0.0)
    parser.add_argument("--mid-success-rate", type=float, default=0.25)
    parser.add_argument("--high-success-rate", type=float, default=0.75)
    parser.add_argument("--partial-std-threshold", type=float, default=0.05)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
