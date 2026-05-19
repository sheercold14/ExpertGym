#!/usr/bin/env python3
"""Build a calibration manifest that keeps only expert-recoverable code rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import stable_hash, validate_seed_record


TASK_ORDER = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.input)
    positive_index = _build_positive_index(args.expert_rollout, threshold=args.positive_threshold)
    selected, summary = _select_rows(source_rows, positive_index=positive_index, args=args)
    if not args.dry_run:
        count = write_jsonl(args.output, selected)
        summary["written"] = count
        Path(args.output).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _build_positive_index(paths: list[str], *, threshold: float) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        rows = read_jsonl(path)
        for row in rows:
            prompt_id = str(row.get("prompt_id") or row.get("id") or "")
            if not prompt_id:
                continue
            positives = []
            for sample_idx, sample in enumerate(row.get("samples") or []):
                reward = _reward_value(sample)
                success = bool(sample.get("success")) or reward >= threshold
                if success:
                    positives.append(
                        {
                            "source_rollout": str(path),
                            "sample_idx": sample_idx,
                            "reward": reward,
                            "success": bool(sample.get("success")),
                        }
                    )
            if positives:
                index[prompt_id].extend(positives)
    return index


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    positive_index: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASK_ORDER}
    skipped = Counter()
    for row in rows:
        task = str(row.get("task") or "")
        if task not in by_task:
            skipped["unsupported_task"] += 1
            continue
        validate_seed_record(row)
        by_task[task].append(row)

    selected_by_task = {
        "tool": _take_sorted(by_task["tool"], args.tool_count, args.seed, "tool"),
        "memory": _take_sorted(by_task["memory"], args.memory_count, args.seed, "memory"),
        "code": _select_code_rows(by_task["code"], positive_index=positive_index, args=args),
    }
    selected = []
    for row in _interleave(selected_by_task):
        copied = dict(row)
        tags = set(copied.get("tags") or [])
        tags.add(args.tag)
        tags.add(f"{args.tag}:train")
        copied["tags"] = sorted(tags)
        copied["split"] = args.tag
        if copied.get("task") == "code":
            positives = positive_index.get(str(copied.get("prompt_id")), [])
            copied["recoverable_code_calibration"] = {
                "tag": args.tag,
                "positive_expert_rollouts": positives,
                "positive_expert_count": len(positives),
                "selection_policy": "keep code rows with at least one same-prompt expert positive",
            }
        validate_seed_record(copied)
        selected.append(copied)

    summary = {
        "format": "opvec_recoverable_code_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).expanduser().resolve()),
        "output": str(Path(args.output).expanduser().resolve()),
        "tag": args.tag,
        "seed": int(args.seed),
        "positive_threshold": float(args.positive_threshold),
        "expert_rollouts": [str(Path(path).expanduser().resolve()) for path in args.expert_rollout],
        "source_task_counts": dict(sorted(Counter(str(row.get("task")) for row in rows).items())),
        "selected_rows": len(selected),
        "selected_task_counts": dict(sorted(Counter(str(row.get("task")) for row in selected).items())),
        "selected_role_counts": _role_counts(selected),
        "code_positive_source_counts": _positive_source_counts(selected),
        "skipped": dict(sorted(skipped.items())),
        "note": (
            "This manifest is for training only. Hard non-recoverable code rows should remain in monitor/guard "
            "so ExpertGym is optimized on verified recoverable directions and audited on harder probes."
        ),
    }
    return selected, summary


def _select_code_rows(
    rows: list[dict[str, Any]],
    *,
    positive_index: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    recoverable = [row for row in rows if str(row.get("prompt_id")) in positive_index]
    recoverable.sort(
        key=lambda row: (
            _code_role_rank(row),
            -len(positive_index.get(str(row.get("prompt_id")), [])),
            stable_hash({"seed": args.seed, "task": "code", "prompt_id": row.get("prompt_id")}),
        )
    )
    if args.code_count < 0:
        return recoverable
    return recoverable[: min(args.code_count, len(recoverable))]


def _take_sorted(rows: list[dict[str, Any]], count: int, seed: int, task: str) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: stable_hash({"seed": seed, "task": task, "prompt_id": row.get("prompt_id")}))
    if count < 0:
        return selected
    return selected[: min(count, len(selected))]


def _interleave(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    max_len = max((len(rows) for rows in groups.values()), default=0)
    for idx in range(max_len):
        for task in TASK_ORDER:
            rows = groups.get(task, [])
            if idx < len(rows):
                output.append(rows[idx])
    return output


def _code_role_rank(row: dict[str, Any]) -> int:
    role = _row_role(row)
    if role == "source_code_anchor_from_paper96":
        return 0
    if role == "on_policy_eval_style_code_probe":
        return 1
    if role == "generation":
        return 0
    if role == "frontier":
        return 1
    if role == "partial_edge":
        return 2
    if role == "stable":
        return 3
    return 2


def _reward_value(sample: dict[str, Any]) -> float:
    for key in ("reward_train", "task_reward", "reward"):
        value = sample.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _role_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        task = str(row.get("task") or "")
        role = _row_role(row) or "default"
        counts[task][role] += 1
    return {task: dict(sorted(counter.items())) for task, counter in sorted(counts.items())}


def _row_role(row: dict[str, Any]) -> str:
    eval_payload = row.get("eval_targeted_calibration") if isinstance(row.get("eval_targeted_calibration"), dict) else {}
    if eval_payload.get("role"):
        return str(eval_payload["role"])
    code_payload = row.get("code_p0_calibration") if isinstance(row.get("code_p0_calibration"), dict) else {}
    if code_payload.get("role"):
        return str(code_payload["role"])
    metadata = row.get("reference", {}).get("metadata", {}) if isinstance(row.get("reference"), dict) else {}
    if isinstance(metadata, dict) and metadata.get("code_bank_role"):
        return str(metadata["code_bank_role"])
    return ""


def _positive_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        payload = row.get("recoverable_code_calibration") or {}
        for item in payload.get("positive_expert_rollouts") or []:
            counter[Path(str(item.get("source_rollout") or "")).name] += 1
    return dict(sorted(counter.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-rollout", action="append", required=True)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--tool-count", type=int, default=32)
    parser.add_argument("--memory-count", type=int, default=48)
    parser.add_argument("--code-count", type=int, default=-1, help="-1 keeps every recoverable code row")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--tag", default="sota_calib_v2_recoverable_code_20260518")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
