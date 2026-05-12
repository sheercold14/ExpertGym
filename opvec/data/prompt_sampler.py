"""Task-balanced prompt sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence


def group_by_task(records: Sequence[Mapping]) -> dict[str, list[Mapping]]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for record in records:
        grouped[str(record["task"])].append(record)
    return dict(grouped)


def task_balanced_sample(
    records: Sequence[Mapping],
    *,
    batch_size: int,
    task_weights: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> list[Mapping]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    grouped = group_by_task(records)
    if not grouped:
        return []
    rng = random.Random(seed)
    weights = dict(task_weights or {task: 1.0 for task in grouped})
    tasks = [task for task in sorted(grouped) if weights.get(task, 0.0) > 0.0]
    if not tasks:
        raise ValueError("No tasks have positive sampling weight")
    total_weight = sum(float(weights[task]) for task in tasks)
    quotas = {task: int(batch_size * float(weights[task]) / total_weight) for task in tasks}
    while sum(quotas.values()) < batch_size:
        for task in tasks:
            quotas[task] += 1
            if sum(quotas.values()) >= batch_size:
                break
    sample = []
    for task in tasks:
        pool = list(grouped[task])
        rng.shuffle(pool)
        quota = quotas[task]
        for idx in range(quota):
            sample.append(pool[idx % len(pool)])
    rng.shuffle(sample)
    return sample[:batch_size]
