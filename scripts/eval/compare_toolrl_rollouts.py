#!/usr/bin/env python3
"""Compare two ToolRL rollout files at the prompt level.

The script is intentionally model-agnostic: it compares any two rollout JSONL
files that follow the OP-VEC rollout schema and contain ToolRL reward details.
It is used to identify source-distribution capability regressions such as
``init1 correct, filtered merge wrong`` before running heavier residual probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl  # noqa: E402


def main() -> None:
    args = parse_args()
    left_rows = load_rows(args.left)
    right_rows = load_rows(args.right)
    comparison = compare_rows(
        left_rows=left_rows,
        right_rows=right_rows,
        left_name=args.left_name,
        right_name=args.right_name,
        top_k=args.top_k,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2, sort_keys=True))


def load_rows(path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row.get("prompt_id") or row.get("group_id") or row.get("id") or "")
        if not prompt_id:
            raise ValueError(f"Rollout row in {path} has no prompt_id/group_id/id")
        if prompt_id in rows:
            raise ValueError(f"Duplicate prompt_id={prompt_id!r} in {path}")
        rows[prompt_id] = dict(row)
    return rows


def compare_rows(
    *,
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
    top_k: int,
) -> dict[str, Any]:
    shared_ids = sorted(set(left_rows) & set(right_rows))
    missing_left = sorted(set(right_rows) - set(left_rows))
    missing_right = sorted(set(left_rows) - set(right_rows))
    records = []
    success_buckets: Counter[str] = Counter()
    exact_buckets: Counter[str] = Counter()
    parseable_buckets: Counter[str] = Counter()
    reward_deltas = []

    for prompt_id in shared_ids:
        left = summarize_row(left_rows[prompt_id])
        right = summarize_row(right_rows[prompt_id])
        record = {
            "prompt_id": prompt_id,
            "left": left,
            "right": right,
            "delta_reward": right["reward_train"] - left["reward_train"],
            "prompt_excerpt": excerpt(str(left_rows[prompt_id].get("prompt") or left_rows[prompt_id].get("rendered_prompt") or "")),
        }
        records.append(record)
        reward_deltas.append(record["delta_reward"])
        success_buckets[transition(left["success"], right["success"], left_name, right_name)] += 1
        exact_buckets[transition(left["exact_tool_match"], right["exact_tool_match"], left_name, right_name)] += 1
        parseable_buckets[transition(left["parseable"], right["parseable"], left_name, right_name)] += 1

    left_summary = aggregate([record["left"] for record in records])
    right_summary = aggregate([record["right"] for record in records])
    records_by_drop = sorted(records, key=lambda item: (item["delta_reward"], item["prompt_id"]))
    records_by_gain = sorted(records, key=lambda item: (-item["delta_reward"], item["prompt_id"]))
    left_only_success = [
        record for record in records if record["left"]["success"] and not record["right"]["success"]
    ]
    right_only_success = [
        record for record in records if record["right"]["success"] and not record["left"]["success"]
    ]
    left_only_exact = [
        record for record in records if bool(record["left"]["exact_tool_match"]) and not bool(record["right"]["exact_tool_match"])
    ]
    right_only_exact = [
        record for record in records if bool(record["right"]["exact_tool_match"]) and not bool(record["left"]["exact_tool_match"])
    ]

    return {
        "format": "toolrl_rollout_pair_comparison_v1",
        "left_name": left_name,
        "right_name": right_name,
        "summary": {
            "left_name": left_name,
            "right_name": right_name,
            "shared_prompts": len(shared_ids),
            "missing_left": len(missing_left),
            "missing_right": len(missing_right),
            "mean_delta_reward_right_minus_left": mean(reward_deltas) if reward_deltas else 0.0,
            "left": left_summary,
            "right": right_summary,
            "success_transitions": dict(sorted(success_buckets.items())),
            "exact_transitions": dict(sorted(exact_buckets.items())),
            "parseable_transitions": dict(sorted(parseable_buckets.items())),
            "left_only_success_count": len(left_only_success),
            "right_only_success_count": len(right_only_success),
            "left_only_exact_count": len(left_only_exact),
            "right_only_exact_count": len(right_only_exact),
        },
        "missing_left": missing_left,
        "missing_right": missing_right,
        "top_reward_drops": records_by_drop[:top_k],
        "top_reward_gains": records_by_gain[:top_k],
        "left_only_success": left_only_success[:top_k],
        "right_only_success": right_only_success[:top_k],
        "left_only_exact": left_only_exact[:top_k],
        "right_only_exact": right_only_exact[:top_k],
    }


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") if isinstance(row.get("samples"), list) else []
    if not samples:
        raise ValueError(f"Row {row.get('prompt_id')} has no samples")
    sample = samples[0]
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    reward_train = as_float(sample.get("reward_train"), default=as_float(sample.get("reward"), default=0.0))
    prediction_calls = as_float(details.get("prediction_calls"), default=0.0)
    reference_calls = as_float(details.get("reference_calls"), default=0.0)
    return {
        "sample_id": str(sample.get("sample_id") or ""),
        "reward": as_float(sample.get("reward"), default=0.0),
        "reward_train": reward_train,
        "success": bool(sample.get("success", reward_train > 0.5)),
        "parseable": nullable_bool(details.get("parseable")),
        "prediction_calls": prediction_calls,
        "reference_calls": reference_calls,
        "zero_call": prediction_calls <= 0.0,
        "exact_tool_match": nullable_bool(details.get("exact_tool_match")),
        "name_recall": as_float(details.get("name_recall"), default=0.0),
        "prediction_tool_names": details.get("prediction_tool_names") or [],
        "reference_tool_names": details.get("reference_tool_names") or [],
        "tool_call_parse_error": details.get("tool_call_parse_error"),
        "length": as_float(sample.get("length"), default=0.0),
        "text_excerpt": excerpt(str(sample.get("text") or "")),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    if count == 0:
        return {
            "mean_reward_train": 0.0,
            "success_rate": 0.0,
            "exact_tool_rate": 0.0,
            "parseable_rate": 0.0,
            "zero_call_rate": 0.0,
            "mean_name_recall": 0.0,
        }
    exact_items = [item for item in items if item["exact_tool_match"] is not None]
    parse_items = [item for item in items if item["parseable"] is not None]
    return {
        "mean_reward_train": mean(item["reward_train"] for item in items),
        "success_rate": sum(1 for item in items if item["success"]) / float(count),
        "exact_tool_rate": sum(1 for item in exact_items if item["exact_tool_match"]) / float(len(exact_items) or 1),
        "parseable_rate": sum(1 for item in parse_items if item["parseable"]) / float(len(parse_items) or 1),
        "zero_call_rate": sum(1 for item in items if item["zero_call"]) / float(count),
        "mean_name_recall": mean(item["name_recall"] for item in items),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    left_name = payload["left_name"]
    right_name = payload["right_name"]
    lines = [
        f"# ToolRL Rollout Comparison: {left_name} vs {right_name}",
        "",
        "## Summary",
        "",
        f"- shared prompts: `{summary['shared_prompts']}`",
        f"- mean delta reward (`{right_name}` - `{left_name}`): `{summary['mean_delta_reward_right_minus_left']:.6f}`",
        "",
        "| metric | left | right |",
        "| --- | ---: | ---: |",
    ]
    for key in ["mean_reward_train", "success_rate", "exact_tool_rate", "parseable_rate", "zero_call_rate", "mean_name_recall"]:
        lines.append(f"| {key} | {summary['left'][key]:.6f} | {summary['right'][key]:.6f} |")
    lines.extend(["", "## Transitions", ""])
    lines.append("### Success")
    lines.extend(counter_table(summary["success_transitions"]))
    lines.extend(["", "### Exact Tool Match"])
    lines.extend(counter_table(summary["exact_transitions"]))
    lines.extend(["", "### Parseable"])
    lines.extend(counter_table(summary["parseable_transitions"]))
    lines.extend(["", "## Top Reward Drops"])
    lines.extend(record_table(payload["top_reward_drops"], left_name, right_name))
    lines.extend(["", "## Top Reward Gains"])
    lines.extend(record_table(payload["top_reward_gains"], left_name, right_name))
    lines.extend(["", "## Left-Only Exact Cases"])
    lines.extend(record_table(payload["left_only_exact"], left_name, right_name))
    lines.extend(["", "## Right-Only Exact Cases"])
    lines.extend(record_table(payload["right_only_exact"], left_name, right_name))
    return "\n".join(lines) + "\n"


def counter_table(counter: dict[str, int]) -> list[str]:
    lines = ["", "| transition | count |", "| --- | ---: |"]
    for key, value in sorted(counter.items()):
        lines.append(f"| {key} | {value} |")
    return lines


def record_table(records: list[dict[str, Any]], left_name: str, right_name: str) -> list[str]:
    lines = [
        "",
        "| prompt_id | delta_reward | left reward/exact | right reward/exact | ref tools | left pred | right pred | prompt excerpt |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        left = record["left"]
        right = record["right"]
        lines.append(
            "| {prompt_id} | {delta:.4f} | {left_reward:.4f}/{left_exact} | {right_reward:.4f}/{right_exact} | "
            "{ref} | {left_pred} | {right_pred} | {prompt} |".format(
                prompt_id=escape_pipe(record["prompt_id"]),
                delta=record["delta_reward"],
                left_reward=left["reward_train"],
                left_exact=left["exact_tool_match"],
                right_reward=right["reward_train"],
                right_exact=right["exact_tool_match"],
                ref=escape_pipe(", ".join(map(str, left["reference_tool_names"]))),
                left_pred=escape_pipe(", ".join(map(str, left["prediction_tool_names"]))),
                right_pred=escape_pipe(", ".join(map(str, right["prediction_tool_names"]))),
                prompt=escape_pipe(record["prompt_excerpt"]),
            )
        )
    if not records:
        lines.append(f"| _none_ |  |  |  |  | {left_name} | {right_name} |  |")
    return lines


def transition(left_value: Any, right_value: Any, left_name: str, right_name: str) -> str:
    left = bool(left_value)
    right = bool(right_value)
    if left and right:
        return "both_true"
    if left and not right:
        return f"{left_name}_only"
    if right and not left:
        return f"{right_name}_only"
    return "both_false"


def as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def nullable_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def excerpt(text: str, *, limit: int = 140) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def escape_pipe(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left/reference rollout JSONL")
    parser.add_argument("--right", required=True, help="Right/candidate rollout JSONL")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
