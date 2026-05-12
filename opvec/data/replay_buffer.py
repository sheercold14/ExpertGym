"""Replay-buffer classification for OP-VEC rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from opvec.train.frontier import should_keep_frontier


QUEUE_FRONTIER = "frontier"
QUEUE_RETENTION = "retention"
QUEUE_LOW_INFO_FAILURE = "low_info_failure"
QUEUE_OTHER = "other"


def classify_rollout_row(
    row: Mapping[str, Any],
    *,
    min_frontier_weight: float,
    min_reward_std: float,
) -> tuple[str, dict[str, Any]]:
    """Classify one rollout row into policy, retention, or diagnostic queues."""

    keep, reason, frontier = should_keep_frontier(
        row.get("samples", []),
        min_frontier_weight=min_frontier_weight,
        min_reward_std=min_reward_std,
    )
    updated = dict(row)
    updated["keep_for_policy_loss"] = bool(keep)
    updated["skip_reason"] = reason
    updated["frontier"] = frontier
    if keep:
        return QUEUE_FRONTIER, updated
    if frontier.get("all_success"):
        return QUEUE_RETENTION, updated
    if frontier.get("all_failure"):
        return QUEUE_LOW_INFO_FAILURE, updated
    return QUEUE_OTHER, updated


def build_replay_buffer(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_paths: Sequence[str],
    min_frontier_weight: float,
    min_reward_std: float,
) -> dict[str, Any]:
    """Build an auditable replay buffer payload from rollout rows."""

    queues: dict[str, list[dict[str, Any]]] = {
        QUEUE_FRONTIER: [],
        QUEUE_RETENTION: [],
        QUEUE_LOW_INFO_FAILURE: [],
        QUEUE_OTHER: [],
    }
    counts = Counter()
    task_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        queue, classified = classify_rollout_row(
            row,
            min_frontier_weight=min_frontier_weight,
            min_reward_std=min_reward_std,
        )
        queues[queue].append(classified)
        task = str(classified.get("task", "unknown"))
        counts[queue] += 1
        counts["total"] += 1
        task_counts[task][queue] += 1
        task_counts[task]["total"] += 1
    return {
        "format": "opvec_replay_buffer_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": list(source_paths),
        "frontier_filter": {
            "min_frontier_weight": float(min_frontier_weight),
            "min_reward_std": float(min_reward_std),
        },
        "counts": dict(counts),
        "task_counts": {task: dict(counter) for task, counter in sorted(task_counts.items())},
        "queues": queues,
    }
