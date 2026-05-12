#!/usr/bin/env python3
"""Build source-reward-aligned OP-VEC calibration manifests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import write_jsonl
from opvec.data.schema import make_prompt_id, stable_hash, validate_seed_record


DEFAULT_TOOLRL_TRAIN = "/tmp/shared-storage/OnPolicy/external_repos/ToolRL/dataset/rlla_4k/train.parquet"
DEFAULT_MEMORY_JSON = "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets/Memory.json"
DEFAULT_CODECONTESTS_TRAIN = "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json"


def main() -> None:
    args = parse_args()
    rows, summary = build_manifest(args)
    if not args.dry_run:
        count = write_jsonl(args.output, rows)
        summary.update({"output": str(args.output), "written": count})
        Path(args.output).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    if args.tool_limit > 0:
        tool_rows, summaries["tool"] = build_tool_rows(
            Path(args.toolrl_train),
            limit=args.tool_limit,
            seed=args.seed,
            require_tool_call=not args.allow_tool_response_only,
            split=args.split,
        )
        all_rows.extend(tool_rows)
    if args.memory_final_limit > 0:
        memory_rows, summaries["memory"] = build_memory_final_rows(
            Path(args.memory_json),
            limit=args.memory_final_limit,
            seed=args.seed,
            split=args.split,
        )
        all_rows.extend(memory_rows)
    if args.code_limit > 0:
        code_rows, summaries["code"] = build_code_rows(
            Path(args.codecontests_train),
            limit=args.code_limit,
            seed=args.seed,
            split=args.split,
        )
        all_rows.extend(code_rows)
    rng.shuffle(all_rows)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    skipped_duplicates = 0
    for row in all_rows:
        key = str(row["prompt_hash"])
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        validate_seed_record(row)
        deduped.append(row)
    summary = {
        "format": "opvec_source_reward_seed_manifest_v1",
        "seed": int(args.seed),
        "split": args.split,
        "rows": len(deduped),
        "task_counts": dict(sorted(Counter(row["task"] for row in deduped).items())),
        "skipped_duplicates": skipped_duplicates,
        "sources": {
            "toolrl_train": str(args.toolrl_train),
            "memory_json": str(args.memory_json),
            "codecontests_train": str(args.codecontests_train),
        },
        "source_summaries": summaries,
    }
    return deduped, summary


def build_tool_rows(
    path: Path,
    *,
    limit: int,
    seed: int,
    require_tool_call: bool,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for source_row, row in df.iterrows():
        reward_model = _jsonable(row.get("reward_model") or {})
        ground_truth = str(reward_model.get("ground_truth") or "")
        if require_tool_call and "<tool_call>" not in ground_truth:
            skipped["response_only"] += 1
            continue
        messages = _messages_from_value(row.get("prompt"))
        if not messages:
            skipped["no_messages"] += 1
            continue
        extra_info = _jsonable(row.get("extra_info") or {})
        prompt = _last_user_content(messages)
        candidates.append(
            _seed_row(
                task="tool",
                source=str(path),
                source_row=int(source_row),
                split=split,
                prompt=prompt,
                messages=messages,
                reference={
                    "answer": None,
                    "response": ground_truth,
                    "metadata": {
                        "source_dataset": "ToolRL/rlla_4k",
                        "source_index": extra_info.get("index", int(source_row)),
                        "data_source": row.get("data_source"),
                        "ability": row.get("ability"),
                        "has_tool_call": "<tool_call>" in ground_truth,
                    },
                },
                verifier={"name": "toolrl_source_reward", "config": {"source": "ToolRL"}},
                tags=["source_reward", "toolrl", "tool_call" if "<tool_call>" in ground_truth else "response_only"],
            )
        )
    selected = _sample(candidates, limit=limit, seed=seed)
    return selected, {"available": len(candidates), "selected": len(selected), "skipped": dict(sorted(skipped.items()))}


def build_memory_final_rows(path: Path, *, limit: int, seed: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for source_row, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            skipped["non_object"] += 1
            continue
        round_type = str(raw.get("round_type") or "").lower()
        if round_type not in {"final", "final_answer"}:
            skipped["not_final"] += 1
            continue
        messages = _messages_from_value(raw.get("messages"))
        response = str(raw.get("response") or "")
        if not messages or not response:
            skipped["missing_messages_or_response"] += 1
            continue
        candidates.append(
            _seed_row(
                task="memory",
                source=str(path),
                source_row=source_row,
                split=split,
                prompt=_last_user_content(messages),
                messages=messages,
                reference={
                    "answer": None,
                    "response": response,
                    "metadata": {
                        "source_dataset": "MemAgent/HotpotQA interaction",
                        "question_id": raw.get("question_id"),
                        "round_type": "final",
                        "round_idx": raw.get("round_idx"),
                        "task_name": raw.get("task_name"),
                    },
                },
                verifier={"name": "memagent_source_reward", "config": {"source": "MemAgent"}},
                tags=["source_reward", "memagent", "hotpotqa", "final_answer"],
            )
        )
    selected = _sample(candidates, limit=limit, seed=seed)
    return selected, {"available": len(candidates), "selected": len(selected), "skipped": dict(sorted(skipped.items()))}


def build_code_rows(path: Path, *, limit: int, seed: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for source_row, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            skipped["non_object"] += 1
            continue
        question = str(raw.get("question") or "")
        task_id = raw.get("task_id")
        if not question or task_id is None:
            skipped["missing_question_or_task_id"] += 1
            continue
        messages = _cure_code_messages(question)
        candidates.append(
            _seed_row(
                task="code",
                source=str(path),
                source_row=source_row,
                split=split,
                prompt=question,
                messages=messages,
                reference={
                    "answer": None,
                    "response": "",
                    "metadata": {
                        "source_dataset": "CURE/CodeContests_train",
                        "question_id": int(task_id),
                        "task_id": int(task_id),
                        "test_time_limit": raw.get("test_time_limit", 1),
                        "exe_method": raw.get("exe_method"),
                    },
                },
                verifier={"name": "cure_code_pass_rate", "config": {"source": "CURE"}},
                tags=["source_reward", "cure", "codecontests_train"],
            )
        )
    selected = _sample(candidates, limit=limit, seed=seed)
    return selected, {"available": len(candidates), "selected": len(selected), "skipped": dict(sorted(skipped.items()))}


def _seed_row(
    *,
    task: str,
    source: str,
    source_row: int,
    split: str,
    prompt: str,
    messages: list[dict[str, str]],
    reference: dict[str, Any],
    verifier: dict[str, Any],
    tags: list[str],
) -> dict[str, Any]:
    payload = messages or prompt
    prompt_hash = stable_hash({"task": task, "payload": payload, "source": source, "source_row": source_row})
    return {
        "prompt_id": make_prompt_id(task, prompt_hash),
        "task": task,
        "source": source,
        "source_row": int(source_row),
        "split": split,
        "prompt": prompt,
        "messages": messages,
        "reference": _jsonable(reference),
        "verifier": verifier,
        "tags": sorted(dict.fromkeys(tags)),
        "difficulty": None,
        "prompt_hash": prompt_hash,
    }


def _messages_from_value(value: Any) -> list[dict[str, str]]:
    value = _jsonable(value)
    if not isinstance(value, list):
        return []
    messages = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "user":
            return str(item.get("content") or "")
    return "\n".join(str(item.get("content") or "") for item in messages)


def _cure_code_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant help user solve problems."},
        {
            "role": "user",
            "content": (
                "You need to think first then write python script. "
                "You should use input() to input and print() to output in your script. "
                "Your code should output the results based on the input read in, "
                "rather than generating the given test example.\n"
                f"This is the problem:\n{question}"
            ),
        },
    ]


def _sample(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    return pool[: min(int(limit), len(pool))]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--toolrl-train", default=DEFAULT_TOOLRL_TRAIN)
    parser.add_argument("--memory-json", default=DEFAULT_MEMORY_JSON)
    parser.add_argument("--codecontests-train", default=DEFAULT_CODECONTESTS_TRAIN)
    parser.add_argument("--tool-limit", type=int, default=80)
    parser.add_argument("--memory-final-limit", type=int, default=80)
    parser.add_argument("--code-limit", type=int, default=80)
    parser.add_argument("--allow-tool-response-only", action="store_true")
    parser.add_argument("--split", default="source_reward_train")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
