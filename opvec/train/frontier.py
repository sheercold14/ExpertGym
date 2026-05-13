"""Frontier prompt filtering and advantage computation."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


def frontier_stats(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute frontier statistics from scored rollout samples."""

    if not samples:
        raise ValueError("frontier_stats requires at least one sample")
    rewards = [_train_reward(sample) for sample in samples]
    raw_rewards = [float(sample.get("reward", reward)) for sample, reward in zip(samples, rewards)]
    successes = [bool(sample.get("success", reward > 0.5)) for sample, reward in zip(samples, rewards)]
    success_rate = sum(1.0 for value in successes if value) / len(successes)
    reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
    raw_reward_std = pstdev(raw_rewards) if len(raw_rewards) > 1 else 0.0
    weight = 4.0 * success_rate * (1.0 - success_rate)
    return {
        "mean_reward": mean(rewards),
        "std_reward": reward_std,
        "mean_raw_reward": mean(raw_rewards),
        "std_raw_reward": raw_reward_std,
        "reward_field": "reward_train",
        "has_variance": reward_std > 0.0,
        "frontier_weight": weight,
        "num_success": int(sum(successes)),
        "num_samples": len(samples),
        "all_success": all(successes),
        "all_failure": not any(successes),
    }


def should_keep_frontier(
    samples: Sequence[Mapping[str, Any]],
    *,
    min_frontier_weight: float = 0.20,
    min_reward_std: float = 0.05,
    drop_all_success: bool = True,
    drop_all_failure_without_contract_signal: bool = True,
) -> tuple[bool, str | None, dict[str, Any]]:
    stats = frontier_stats(samples)
    if drop_all_success and stats["all_success"]:
        return False, "all_success", stats
    if stats["frontier_weight"] >= min_frontier_weight:
        return True, None, stats
    if stats["std_reward"] >= min_reward_std:
        return True, None, stats
    if drop_all_failure_without_contract_signal and stats["all_failure"] and not _has_contract_variance(samples):
        return False, "all_failure_no_contract_signal", stats
    return False, "insufficient_reward_variance", stats


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    frontier_weight: float = 1.0,
    epsilon: float = 1.0e-6,
) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("at least two rewards are required")
    values = [float(value) for value in rewards]
    avg = mean(values)
    std = pstdev(values)
    if std < epsilon or not math.isfinite(std):
        return [0.0 for _ in values]
    return [float(frontier_weight) * ((value - avg) / (std + epsilon)) for value in values]


def policy_frontier_weight(stats: Mapping[str, Any], *, min_reward_std: float = 0.05) -> float:
    """Return the multiplier used for policy advantages.

    Binary frontiers use 4p(1-p). Continuous reward frontiers can have no binary
    successes but still carry useful verifier signal, so they receive weight 1.
    """

    binary_weight = float(stats.get("frontier_weight", 0.0))
    if binary_weight > 0.0:
        return binary_weight
    if float(stats.get("std_reward", 0.0)) >= float(min_reward_std):
        return 1.0
    return 0.0


def _has_contract_variance(samples: Sequence[Mapping[str, Any]]) -> bool:
    values = [float(sample.get("contract_reward", 0.0)) for sample in samples]
    return len(values) > 1 and pstdev(values) > 0.0


def _train_reward(sample: Mapping[str, Any]) -> float:
    if "reward_train" in sample:
        return float(sample.get("reward_train", 0.0))
    raw = float(sample.get("reward", 0.0))
    details = sample.get("details") if isinstance(sample.get("details"), Mapping) else {}
    if details.get("toolrl_score_range") == [-3.0, 4.0]:
        return max(0.0, min((raw + 3.0) / 7.0, 1.0))
    return max(0.0, min(raw, 1.0))
