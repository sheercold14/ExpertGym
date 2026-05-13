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

ROLLOUT_ROW_REQUIRED_FIELDS = (
    "run_id",
    "step",
    "policy_id",
    "prompt_id",
    "task",
    "samples",
)

ROLLOUT_SAMPLE_REQUIRED_FIELDS = (
    "sample_id",
    "text",
    "reward",
    "task_reward",
)

TOKEN_LEVEL_SAMPLE_FIELDS = (
    "response_token_ids",
    "old_logprobs",
    "response_mask",
)


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


def make_gate_id(gate_values: Mapping[str, Any], gate_checkpoint: str | None = None) -> str:
    payload = {
        "gate_checkpoint": gate_checkpoint or "",
        "gate_values": {key: float(value) for key, value in sorted((str(key), value) for key, value in gate_values.items())},
    }
    return stable_hash(payload)[:16]


def validate_rollout_row(record: Mapping[str, Any], *, require_token_fields: bool = False) -> None:
    missing = [name for name in ROLLOUT_ROW_REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"Rollout row missing fields: {missing}")
    if canonical_task(str(record["task"])) is None:
        raise ValueError(f"Rollout row has unknown task: {record['task']}")
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Rollout row has no samples: {record.get('prompt_id')}")
    for sample in samples:
        _validate_rollout_sample(sample, require_token_fields=require_token_fields)


def _validate_rollout_sample(sample: Mapping[str, Any], *, require_token_fields: bool) -> None:
    missing = [name for name in ROLLOUT_SAMPLE_REQUIRED_FIELDS if name not in sample]
    if missing:
        raise ValueError(f"Rollout sample missing fields: {missing}")
    if require_token_fields:
        token_missing = [name for name in TOKEN_LEVEL_SAMPLE_FIELDS if name not in sample]
        if token_missing:
            raise ValueError(f"Rollout sample missing token-level fields: {token_missing}")
        validate_token_level_sample(sample)
    trajectory = sample.get("trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"Rollout sample trajectory is invalid: {sample.get('sample_id')}")
        for turn in trajectory:
            _validate_trajectory_turn(turn, require_token_fields=require_token_fields)


def _validate_trajectory_turn(turn: Mapping[str, Any], *, require_token_fields: bool) -> None:
    for name in ("turn", "kind", "prompt_text", "text"):
        if name not in turn:
            raise ValueError(f"Trajectory turn missing field: {name}")
    if require_token_fields:
        token_missing = [name for name in TOKEN_LEVEL_SAMPLE_FIELDS if name not in turn]
        if token_missing:
            raise ValueError(f"Trajectory turn missing token-level fields: {token_missing}")
        validate_token_level_sample(turn)


def validate_token_level_sample(sample: Mapping[str, Any]) -> None:
    token_ids = sample.get("response_token_ids")
    logprobs = sample.get("old_logprobs")
    mask = sample.get("response_mask")
    if not isinstance(token_ids, list) or not isinstance(logprobs, list) or not isinstance(mask, list):
        raise ValueError("Token-level fields must be lists")
    lengths = {len(token_ids), len(logprobs), len(mask)}
    if len(lengths) != 1:
        raise ValueError(
            "Token-level fields must have the same length: "
            f"response_token_ids={len(token_ids)} old_logprobs={len(logprobs)} response_mask={len(mask)}"
        )
    if not token_ids:
        raise ValueError("Token-level fields are empty")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("response_mask must contain only 0/1 values")
