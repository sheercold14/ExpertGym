"""Torch losses for OP-VEC Gate-GRPO."""

from __future__ import annotations

from typing import Any


def grpo_clipped_policy_loss(
    torch: Any,
    *,
    current_logps: Any,
    old_logps: Any,
    advantages: Any,
    clip_epsilon: float,
):
    current = current_logps.float()
    old = old_logps.float().to(device=current.device)
    adv = advantages.float().to(device=current.device)
    if current.numel() != old.numel() or current.numel() != adv.numel():
        raise ValueError("current_logps, old_logps, and advantages must have the same length")
    if current.numel() < 2:
        raise ValueError("GRPO loss requires at least two candidates")
    ratio = torch.exp((current - old).clamp(-20.0, 20.0))
    clipped = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    objective = torch.minimum(ratio * adv, clipped * adv)
    return -objective.mean()


def sequence_kl_penalty(torch: Any, *, current_logps: Any, reference_logps: Any):
    current = current_logps.float()
    reference = reference_logps.float().to(device=current.device)
    log_ratio = (reference - current).clamp(-20.0, 20.0)
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()
