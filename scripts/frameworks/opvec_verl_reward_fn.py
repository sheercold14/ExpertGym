"""VeRL/slime-style reward function wrapper for OP-VEC tasks.

This module is intentionally tiny: frameworks may import ``compute_score`` or
``reward_func`` and pass the generated response plus dataset metadata.  Native
Gated-GRPO training in this repository uses ``RewardRouter`` directly; this file
exists so the same verifier semantics can be reused in VeRL/slime experiments.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opvec.rewards.router import RewardRouter

_ROUTER = RewardRouter()


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any | None = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    """Return a scalar verifier reward for a generated solution string.

    The signature follows the common VeRL reward-function convention while
    remaining permissive for slime/custom launchers.  ``extra_info`` should
    contain the original OP-VEC prompt row fields; ``ground_truth`` is used as a
    fallback reference.
    """

    prompt_record = dict(extra_info or {})
    if "reference" not in prompt_record and ground_truth is not None:
        prompt_record["reference"] = ground_truth
    if "task" not in prompt_record:
        prompt_record["task"] = prompt_record.get("ability") or kwargs.get("ability") or data_source or "unknown"
    scored = _ROUTER.score(prompt_record, solution_str or "")
    return float(scored.get("reward", scored.get("task_reward", 0.0)))


def reward_func(*args: Any, **kwargs: Any) -> float:
    return compute_score(*args, **kwargs)
