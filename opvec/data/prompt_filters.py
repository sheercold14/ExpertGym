"""Prompt filtering helpers shared by rollout and replay tooling."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


MEMORY_FINAL_ANSWER = "final_answer"
MEMORY_UPDATE = "memory_update"
MEMORY_UNKNOWN = "memory_unknown"


def memory_prompt_kind(record: Mapping[str, Any]) -> str:
    """Classify MemoryAgent prompts into final-answer or memory-update phases."""

    reference = record.get("reference") or {}
    metadata = reference.get("metadata", {}) if isinstance(reference, Mapping) else {}
    round_type = str(metadata.get("round_type", "")).lower() if isinstance(metadata, Mapping) else ""
    if round_type in {"final", "trajectory", "memagent_trajectory", MEMORY_FINAL_ANSWER}:
        return MEMORY_FINAL_ANSWER
    if round_type in {"chunk", MEMORY_UPDATE}:
        return MEMORY_UPDATE
    prompt = str(record.get("prompt", ""))
    if "put the answer in \\boxed" in prompt or "Your answer:" in prompt:
        return MEMORY_FINAL_ANSWER
    if "update the memory" in prompt.lower():
        return MEMORY_UPDATE
    return MEMORY_UNKNOWN


def parse_memory_kind_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    aliases = {
        "final": MEMORY_FINAL_ANSWER,
        "answer": MEMORY_FINAL_ANSWER,
        "final_answer": MEMORY_FINAL_ANSWER,
        "chunk": MEMORY_UPDATE,
        "update": MEMORY_UPDATE,
        "memory_update": MEMORY_UPDATE,
        "unknown": MEMORY_UNKNOWN,
        "memory_unknown": MEMORY_UNKNOWN,
    }
    kinds = set()
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown memory kind filter: {item}")
        kinds.add(aliases[key])
    return kinds


def filter_memory_records(records: Iterable[Mapping[str, Any]], kinds: set[str] | None) -> list[Mapping[str, Any]]:
    if kinds is None:
        return list(records)
    return [record for record in records if record.get("task") == "memory" and memory_prompt_kind(record) in kinds]
