"""Utilities for OP-VEC Gated-GRPO.

Gated-GRPO freezes the base model and all expert task-vector deltas, then
optimizes only the small gate/coefficient module.  The rollout format is the
same as ``scripts/train/opvec_collect_hf_rollouts.py``: each prompt row stores
multiple samples from the current gated policy, their old summed sequence
log-probabilities, and verifier rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GrpoGroupStats:
    """Audit statistics for one prompt group."""

    num_samples: int
    mean_reward: float
    std_reward: float
    frontier_weight: float
    nonzero_advantages: int


def normalize_rewards_to_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1.0e-6,
    frontier_weight: float = 1.0,
) -> tuple[list[float], GrpoGroupStats]:
    """Return GRPO-style group-relative advantages.

    The baseline is the within-prompt reward mean.  Groups with near-zero
    reward variance intentionally produce zero advantages: they are valid
    rollouts, but they cannot teach coefficient movement.
    """

    if len(rewards) < 2:
        raise ValueError("Gated-GRPO requires at least two samples per prompt group")
    values = [float(item) for item in rewards]
    avg = mean(values)
    std = pstdev(values)
    if std < float(epsilon) or not math.isfinite(std):
        advantages = [0.0 for _ in values]
    else:
        advantages = [float(frontier_weight) * (value - avg) / (std + float(epsilon)) for value in values]
    stats = GrpoGroupStats(
        num_samples=len(values),
        mean_reward=float(avg),
        std_reward=float(std),
        frontier_weight=float(frontier_weight),
        nonzero_advantages=sum(1 for item in advantages if abs(float(item)) > 0.0),
    )
    return advantages, stats


def valid_policy_samples(samples: Sequence[Mapping[str, Any]], *, require_old_logprob: bool = True) -> list[dict[str, Any]]:
    """Filter rollout samples that can be used by a policy-gradient loss."""

    output: list[dict[str, Any]] = []
    for sample in samples:
        text = sample.get("text")
        if not isinstance(text, str) or not text:
            continue
        if require_old_logprob and sample.get("old_logprob") is None:
            continue
        output.append(dict(sample))
    return output


def clipped_grpo_sequence_loss(
    torch: Any,
    *,
    current_logp: Any,
    old_logp: Any,
    advantage: Any,
    clip_epsilon: float,
):
    """Single-sequence PPO/GRPO clipped surrogate loss.

    ``current_logp`` and ``old_logp`` are summed sequence log-probabilities.
    Constants are clamped for numerical stability because sequence logp deltas
    can be large for long Code responses.
    """

    ratio = torch.exp((current_logp.float() - old_logp.float()).clamp(-20.0, 20.0))
    clipped = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    return -torch.minimum(ratio * advantage.float(), clipped * advantage.float())


def clipped_grpo_token_loss(
    torch: Any,
    *,
    current_logprobs: Any,
    old_logprobs: Any,
    response_mask: Any,
    advantage: Any,
    clip_epsilon: float,
):
    """Token-level PPO/GRPO clipped surrogate with a scalar group advantage.

    This matches the VeRL-style contract: the group-level sample advantage is
    broadcast to every response token selected by ``response_mask``.  The return
    value is a masked mean over tokens for one sample.
    """

    current = current_logprobs.float()
    old = old_logprobs.to(current.device).float()
    mask = response_mask.to(current.device).float()
    adv = advantage.to(current.device).float()
    ratio = torch.exp((current - old).clamp(-20.0, 20.0))
    clipped = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    token_loss = -torch.minimum(ratio * adv, clipped * adv)
    return masked_token_mean(torch, token_loss, mask)


def reverse_kl_sequence_penalty(torch: Any, *, current_logp: Any, old_logp: Any):
    """Sequence-level reverse-KL surrogate used by the existing OP-VEC updater."""

    log_ratio = (old_logp.float() - current_logp.float()).clamp(-20.0, 20.0)
    return torch.exp(log_ratio) - log_ratio - 1.0


def reverse_kl_token_penalty(torch: Any, *, current_logprobs: Any, old_logprobs: Any, response_mask: Any):
    """Masked token-level reverse-KL surrogate."""

    current = current_logprobs.float()
    old = old_logprobs.to(current.device).float()
    mask = response_mask.to(current.device).float()
    log_ratio = (old - current).clamp(-20.0, 20.0)
    token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    return masked_token_mean(torch, token_kl, mask)


def masked_token_mean(torch: Any, values: Any, mask: Any):
    mask = mask.to(values.device).float()
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def gate_initialization_prior(torch: Any, gate_manager: Any):
    """Merge-specific trust-region prior around the gate initialization.

    Parameterized managers already expose initial buffers.  For the original
    global manager, this function reconstructs the initialized expert
    coefficients from raw parameters at construction time when available, and
    otherwise falls back to a zero scalar tensor.  Keeping this helper here makes
    it easier to audit whether a Gated-GRPO run is doing local utility
    calibration rather than unconstrained coefficient search.
    """

    if (
        hasattr(gate_manager, "raw_global_coefficients")
        and hasattr(gate_manager, "initial_global_coefficients")
        and hasattr(gate_manager, "raw_residual_coefficients")
        and hasattr(gate_manager, "initial_residual_coefficients")
    ):
        global_loss = (gate_manager.raw_global_coefficients.float() - gate_manager.initial_global_coefficients.float()).pow(2).mean()
        residual_loss = (gate_manager.raw_residual_coefficients.float() - gate_manager.initial_residual_coefficients.float()).pow(2).mean()
        return float(getattr(gate_manager, "global_prior_scale", 0.10)) * global_loss + float(
            getattr(gate_manager, "residual_prior_scale", 1.00)
        ) * residual_loss
    if hasattr(gate_manager, "raw_coefficients") and hasattr(gate_manager, "initial_coefficients"):
        return (gate_manager.raw_coefficients.float() - gate_manager.initial_coefficients.float()).pow(2).mean()
    if hasattr(gate_manager, "raw_common") and hasattr(gate_manager, "initial_raw_common"):
        common_loss = (gate_manager.raw_common.float() - gate_manager.initial_raw_common.float()).pow(2).mean()
        residual_loss = (gate_manager.raw_residual.float() - gate_manager.initial_raw_residual.float()).pow(2).mean()
        return common_loss + residual_loss
    params = list(gate_manager.parameters()) if hasattr(gate_manager, "parameters") else []
    if params:
        return params[0].new_tensor(0.0)
    return torch.tensor(0.0)


def task_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task", "unknown"))
        counts[task] = counts.get(task, 0) + 1
    return counts
