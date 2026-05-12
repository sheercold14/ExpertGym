"""Build seed prompt manifests from existing ExpertMerging-style datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import read_json
from .schema import (
    VERIFIER_BY_TASK,
    canonical_task,
    make_prompt_id,
    prompt_payload,
    stable_hash,
    validate_seed_record,
)


TASK_FILE_NAMES = {
    "ToolCall.json": "tool",
    "Tool.json": "tool",
    "Memory.json": "memory",
    "HotpotQA.json": "memory",
    "Code.json": "code",
    "CodeContests.json": "code",
}


def discover_dataset_files(input_roots: Sequence[str | Path], tasks: Sequence[str] | None = None) -> list[Path]:
    """Find candidate JSON files under input roots."""

    task_filter = {canonical_task(task) for task in tasks or []}
    task_filter.discard(None)
    paths: list[Path] = []
    for root_item in input_roots:
        root = Path(root_item).expanduser()
        if root.is_file() and root.name in TASK_FILE_NAMES:
            task = TASK_FILE_NAMES[root.name]
            if not task_filter or task in task_filter:
                paths.append(root)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            task = TASK_FILE_NAMES.get(path.name)
            if task is None:
                continue
            if task_filter and task not in task_filter:
                continue
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def build_seed_records(
    input_roots: Sequence[str | Path],
    *,
    tasks: Sequence[str] | None = None,
    split: str = "train_frontier",
    dedup_by: str = "prompt_hash",
    max_per_task: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize discovered records into the OP-VEC seed manifest schema."""

    files = discover_dataset_files(input_roots, tasks=tasks)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    task_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for path in files:
        task = TASK_FILE_NAMES[path.name]
        payload = read_json(path)
        if not isinstance(payload, list):
            skipped["non_list_json"] += 1
            continue
        for idx, raw in enumerate(payload):
            if not isinstance(raw, Mapping):
                skipped["non_object_record"] += 1
                continue
            if max_per_task is not None and task_counts[task] >= max_per_task:
                continue
            try:
                row = normalize_seed_record(raw, task=task, source_path=path, row_index=idx, split=split)
            except ValueError:
                skipped["invalid_record"] += 1
                continue
            dedup_key = str(row.get(dedup_by) or row["prompt_id"])
            if dedup_key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(dedup_key)
            validate_seed_record(row)
            rows.append(row)
            task_counts[task] += 1

    summary = {
        "num_records": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "input_files": [str(path) for path in files],
        "dedup_by": dedup_by,
    }
    return rows, summary


def normalize_seed_record(
    raw: Mapping[str, Any],
    *,
    task: str,
    source_path: Path,
    row_index: int,
    split: str,
) -> dict[str, Any]:
    payload = prompt_payload(raw)
    if not payload:
        raise ValueError("record has no prompt payload")
    prompt_hash = stable_hash({"task": task, "payload": payload})
    messages = raw.get("messages") if isinstance(raw.get("messages"), list) else None
    prompt = _prompt_text(raw, messages)
    reference = {
        "answer": raw.get("answer"),
        "response": raw.get("response"),
        "metadata": {
            key: value
            for key, value in raw.items()
            if key not in {"messages", "prompt", "question", "input", "answer", "response"}
        },
    }
    tags = ["seed"]
    parts = set(source_path.parts)
    if "correct_samples" in parts:
        tags.append("correct_pool")
    for name in ["routed", "routed_1", "routed_1_merge", "toolrl", "hotpotqa", "codecontests", "deepseek"]:
        if name in parts:
            tags.append(name)
    return {
        "prompt_id": make_prompt_id(task, prompt_hash),
        "task": task,
        "source": str(source_path),
        "source_row": row_index,
        "split": split,
        "prompt": prompt,
        "messages": messages or [],
        "reference": reference,
        "verifier": {"name": VERIFIER_BY_TASK[task], "config": {}},
        "tags": sorted(dict.fromkeys(tags)),
        "difficulty": None,
        "prompt_hash": prompt_hash,
    }


def _prompt_text(raw: Mapping[str, Any], messages: list[Mapping[str, Any]] | None) -> str:
    if raw.get("prompt"):
        return str(raw["prompt"])
    if raw.get("question"):
        return str(raw["question"])
    if raw.get("input"):
        return str(raw["input"])
    if messages:
        user_messages = [str(item.get("content", "")) for item in messages if item.get("role") == "user"]
        if user_messages:
            return user_messages[-1]
        return "\n".join(str(item.get("content", "")) for item in messages)
    return ""
