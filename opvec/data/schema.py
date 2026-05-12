"""Schema validation and stable ids for OP-VEC data rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


TASK_ALIASES = {
    "tool": "tool",
    "toolcall": "tool",
    "toolrl": "tool",
    "memory": "memory",
    "hotpotqa": "memory",
    "code": "code",
    "codecontests": "code",
}

VERIFIER_BY_TASK = {
    "tool": "tool_schema",
    "memory": "hotpotqa_em_f1",
    "code": "code_tests",
}


def canonical_task(value: str) -> str | None:
    normalized = str(value).strip().lower().replace("_", "").replace("-", "")
    return TASK_ALIASES.get(normalized)


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_payload(record: Mapping[str, Any]) -> Any:
    messages = record.get("messages")
    if messages:
        return messages
    return record.get("prompt") or record.get("question") or record.get("input") or ""


def validate_seed_record(record: Mapping[str, Any]) -> None:
    required = ["prompt_id", "task", "source", "split", "verifier", "prompt_hash"]
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"Seed record missing fields: {missing}")
    if record["task"] not in VERIFIER_BY_TASK:
        raise ValueError(f"Unknown task: {record['task']}")
    if not (record.get("prompt") or record.get("messages")):
        raise ValueError(f"Seed record has no prompt/messages: {record['prompt_id']}")
    verifier = record.get("verifier")
    if not isinstance(verifier, Mapping) or not verifier.get("name"):
        raise ValueError(f"Seed record verifier is invalid: {record['prompt_id']}")


def make_prompt_id(task: str, prompt_hash: str) -> str:
    return f"{task}__{prompt_hash[:16]}"
