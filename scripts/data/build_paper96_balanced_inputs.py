#!/usr/bin/env python3
"""Build balanced paper-run calibration and fixed OPD distillation inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


TASKS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt_rows = build_balanced_prompts(args)
    opd_rows = build_fixed_opd(args, tokenizer)
    if not args.dry_run:
        write_jsonl(Path(args.prompt_output), prompt_rows)
        write_jsonl(Path(args.opd_output), opd_rows)
        write_summary(args, prompt_rows, opd_rows)
    print(
        json.dumps(
            {
                "prompt_output": args.prompt_output,
                "prompt_rows": len(prompt_rows),
                "prompt_task_counts": task_counts(prompt_rows),
                "opd_output": args.opd_output,
                "opd_rows": len(opd_rows),
                "opd_task_counts": task_counts(opd_rows),
                "opd_sample_counts": opd_sample_counts(opd_rows),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def build_balanced_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(args.prompt_input))
    per_task = int(args.prompts_per_task)
    selected = []
    counts: Counter[str] = Counter()
    for row in rows:
        task = str(row.get("task"))
        if task not in TASKS or counts[task] >= per_task:
            continue
        cloned = dict(row)
        cloned["paper96_balanced_source"] = str(args.prompt_input)
        cloned["paper96_created_at"] = now()
        selected.append(cloned)
        counts[task] += 1
    missing = {task: per_task - counts[task] for task in TASKS if counts[task] < per_task}
    if missing:
        raise SystemExit(f"Not enough prompt rows for balanced manifest: {missing}")
    return selected


def build_fixed_opd(args: argparse.Namespace, tokenizer) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(args.opd_input))
    selected = []
    counts: Counter[str] = Counter()
    per_task = int(args.opd_rows_per_task)
    for row in rows:
        task = str(row.get("task"))
        if task not in TASKS or counts[task] >= per_task:
            continue
        cloned = json.loads(json.dumps(row, ensure_ascii=False))
        cloned["paper96_fixed_opd_source"] = str(args.opd_input)
        cloned["paper96_created_at"] = now()
        cloned["frontier"] = dict(cloned.get("frontier") or {})
        cloned["frontier"]["reward_field"] = "reward_train"
        for sample in cloned.get("samples", []):
            raw_reward = float(sample.get("reward", 0.0) or 0.0)
            reward_train = 1.0 if raw_reward >= float(args.positive_reward_cutoff) else 0.0
            sample["reward_train"] = reward_train
            sample["opd_role"] = "positive" if reward_train >= 1.0 else "negative"
            sample["opd_source_fix"] = "paper96_reward_train_binary"
            if args.recompute_length or sample.get("length") is None:
                text = str(sample.get("text") or "")
                sample["length"] = len(tokenizer.encode(text, add_special_tokens=False))
        selected.append(cloned)
        counts[task] += 1
    missing = {task: per_task - counts[task] for task in TASKS if counts[task] < per_task}
    if missing:
        raise SystemExit(f"Not enough OPD rows for balanced distill data: {missing}")
    return selected


def write_summary(args: argparse.Namespace, prompt_rows: list[dict[str, Any]], opd_rows: list[dict[str, Any]]) -> None:
    summary = {
        "created_at": now(),
        "prompt_input": str(args.prompt_input),
        "prompt_output": str(args.prompt_output),
        "prompt_rows": len(prompt_rows),
        "prompt_task_counts": task_counts(prompt_rows),
        "opd_input": str(args.opd_input),
        "opd_output": str(args.opd_output),
        "opd_rows": len(opd_rows),
        "opd_task_counts": task_counts(opd_rows),
        "opd_sample_counts": opd_sample_counts(opd_rows),
        "positive_reward_cutoff": float(args.positive_reward_cutoff),
        "tokenizer": str(args.tokenizer),
        "recompute_length": bool(args.recompute_length),
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def opd_sample_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output = {}
    for task in TASKS:
        counter: Counter[str] = Counter()
        for row in rows:
            if row.get("task") != task:
                continue
            for sample in row.get("samples", []):
                counter[str(sample.get("opd_role"))] += 1
        output[task] = dict(counter)
    return output


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("task")) for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-input", required=True)
    parser.add_argument("--prompt-output", required=True)
    parser.add_argument("--opd-input", required=True)
    parser.add_argument("--opd-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--tokenizer", default="/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct")
    parser.add_argument("--prompts-per-task", type=int, default=32)
    parser.add_argument("--opd-rows-per-task", type=int, default=7)
    parser.add_argument("--positive-reward-cutoff", type=float, default=0.5)
    parser.add_argument("--recompute-length", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
