"""Reward adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RewardResult:
    reward: float
    task_reward: float
    reward_train: float | None = None
    contract_reward: float = 0.0
    cost_penalty: float = 0.0
    success: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    critical_spans: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "task_reward": float(self.task_reward),
            "reward_train": float(self.reward if self.reward_train is None else self.reward_train),
            "contract_reward": float(self.contract_reward),
            "cost_penalty": float(self.cost_penalty),
            "success": bool(self.success),
            "details": self.details,
            "critical_spans": self.critical_spans,
        }


class RewardAdapter(Protocol):
    name: str

    def score(self, prompt_record: dict[str, Any], output_text: str) -> RewardResult:
        ...
