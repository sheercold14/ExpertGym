"""verl custom reward wrapper for OP-VEC official reward routing."""

from __future__ import annotations

import json
from typing import Any

from .path_utils import ensure_opvec_on_path

ensure_opvec_on_path()

from opvec.rewards.router import RewardRouter  # noqa: E402

_ROUTER = RewardRouter()


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any | None = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    """Return OP-VEC verifier reward using verl's custom reward signature."""

    prompt_record = dict(extra_info or {})
    if "reference" not in prompt_record and prompt_record.get("reference_json"):
        prompt_record["reference"] = _loads_json(prompt_record.get("reference_json"), default={})
    if "reference" not in prompt_record and ground_truth is not None:
        prompt_record["reference"] = _loads_json(ground_truth, default=ground_truth)
    if "task" not in prompt_record:
        prompt_record["task"] = prompt_record.get("ability") or kwargs.get("ability") or data_source or "unknown"
    score_text = solution_str or ""
    if prompt_record.get("task") == "memory" and prompt_record.get("memagent_final_text"):
        score_text = str(prompt_record["memagent_final_text"])
    scored = _ROUTER.score(prompt_record, score_text)
    return float(scored.get("reward", scored.get("task_reward", 0.0)))


def _loads_json(value: Any, *, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
