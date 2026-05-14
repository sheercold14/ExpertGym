#!/usr/bin/env python3
"""Summarize OP-VEC rollout JSONL files for sanity checks."""

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


def main() -> None:
    args = parse_args()
    summary = summarize_rollouts(args.rollouts)
    failures = guard_failures(summary, args)
    summary["guard_failures"] = failures
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if failures and args.fail_on_guard:
        raise SystemExit(3)


def summarize_rollouts(paths: list[str]) -> dict[str, Any]:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row.setdefault("source_rollout", str(path))
            rows.append(row)

    task_accumulators: dict[str, dict[str, Any]] = defaultdict(_new_accumulator)
    for row in rows:
        task = str(row.get("task") or "unknown")
        acc = task_accumulators[task]
        acc["rows"] += 1
        if row.get("keep_for_policy_loss"):
            acc["kept_frontier_rows"] += 1
        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            acc["samples"] += 1
            reward = _train_reward(sample)
            acc["rewards"].append(reward)
            acc["successes"] += int(bool(sample.get("success", reward > 0.5)))
            length = _as_float(sample.get("length"), default=0.0)
            if length > 0:
                acc["lengths"].append(length)
            if task == "tool":
                details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
                if "parseable" in details:
                    acc["tool_parseable_observed"] += 1
                    acc["tool_parseable"] += int(bool(details.get("parseable")))
                calls = details.get("prediction_calls")
                if calls is not None:
                    acc["tool_calls_observed"] += 1
                    acc["tool_zero_calls"] += int(_as_float(calls, default=0.0) <= 0)
                if details.get("exact_tool_match") is not None:
                    acc["tool_exact_observed"] += 1
                    acc["tool_exact"] += int(bool(details.get("exact_tool_match")))

    return {
        "format": "opvec_rollout_summary_v1",
        "rollouts": list(paths),
        "rows": len(rows),
        "task_stats": {task: _finalize_accumulator(acc) for task, acc in sorted(task_accumulators.items())},
    }


def guard_failures(summary: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    failures = []
    task_stats = summary.get("task_stats", {})
    tool = task_stats.get("tool", {})
    parseable_rate = tool.get("tool_parseable_rate")
    zero_call_rate = tool.get("tool_zero_call_rate")
    if parseable_rate is not None and parseable_rate < float(args.min_tool_parseable_rate):
        failures.append(
            {
                "metric": "tool_parseable_rate",
                "value": parseable_rate,
                "threshold": float(args.min_tool_parseable_rate),
                "direction": "min",
            }
        )
    if zero_call_rate is not None and zero_call_rate > float(args.max_tool_zero_call_rate):
        failures.append(
            {
                "metric": "tool_zero_call_rate",
                "value": zero_call_rate,
                "threshold": float(args.max_tool_zero_call_rate),
                "direction": "max",
            }
        )
    for item in args.min_task_mean_reward or []:
        task, threshold = _parse_metric_threshold(item)
        value = task_stats.get(task, {}).get("mean_reward")
        if value is not None and value < threshold:
            failures.append({"metric": f"{task}.mean_reward", "value": value, "threshold": threshold, "direction": "min"})
    return failures


def _new_accumulator() -> dict[str, Any]:
    return {
        "rows": 0,
        "kept_frontier_rows": 0,
        "samples": 0,
        "rewards": [],
        "successes": 0,
        "lengths": [],
        "tool_parseable_observed": 0,
        "tool_parseable": 0,
        "tool_calls_observed": 0,
        "tool_zero_calls": 0,
        "tool_exact_observed": 0,
        "tool_exact": 0,
    }


def _finalize_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    rewards = acc["rewards"]
    lengths = acc["lengths"]
    samples = int(acc["samples"])
    output = {
        "rows": int(acc["rows"]),
        "samples": samples,
        "kept_frontier_rows": int(acc["kept_frontier_rows"]),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "success_rate": float(acc["successes"]) / samples if samples else 0.0,
        "mean_length": mean(lengths) if lengths else 0.0,
        "max_length": max(lengths) if lengths else 0.0,
    }
    if acc["tool_parseable_observed"]:
        output["tool_parseable_rate"] = float(acc["tool_parseable"]) / float(acc["tool_parseable_observed"])
    if acc["tool_calls_observed"]:
        output["tool_zero_call_rate"] = float(acc["tool_zero_calls"]) / float(acc["tool_calls_observed"])
    if acc["tool_exact_observed"]:
        output["tool_exact_rate"] = float(acc["tool_exact"]) / float(acc["tool_exact_observed"])
    return output


def _parse_metric_threshold(item: str) -> tuple[str, float]:
    if "=" not in item:
        raise ValueError(f"Expected task=value threshold, got: {item}")
    task, value = item.split("=", 1)
    return task.strip(), float(value)


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _train_reward(sample: dict[str, Any]) -> float:
    if "reward_train" in sample:
        return _as_float(sample.get("reward_train"), default=0.0)
    raw = _as_float(sample.get("reward"), default=0.0)
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    if details.get("toolrl_score_range") == [-3.0, 4.0]:
        return max(0.0, min((raw + 3.0) / 7.0, 1.0))
    return max(0.0, min(raw, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", action="append", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-tool-parseable-rate", type=float, default=0.25)
    parser.add_argument("--max-tool-zero-call-rate", type=float, default=0.75)
    parser.add_argument("--min-task-mean-reward", action="append", default=[])
    parser.add_argument("--fail-on-guard", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
