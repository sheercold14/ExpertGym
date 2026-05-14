#!/usr/bin/env python3
"""Summarize reward and gate movement for a gate-strategy training run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.summarize_rollouts import summarize_rollouts

EXPERTS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    summary = summarize_run(
        run_dir=Path(args.run_dir),
        init_value=float(args.init_value),
        strategy=args.strategy,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def summarize_run(*, run_dir: Path, init_value: float, strategy: str | None) -> dict[str, Any]:
    iteration_summaries = []
    for iter_dir in sorted(run_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        rollouts = iter_dir / "rollouts.jsonl"
        update_summary_path = iter_dir / "gate_updates.summary.json"
        gates_path = iter_dir / "gate_updates.gates.json"
        item: dict[str, Any] = {
            "iteration": iter_dir.name,
            "rollouts": str(rollouts) if rollouts.exists() else None,
            "update_summary": str(update_summary_path) if update_summary_path.exists() else None,
            "gates": str(gates_path) if gates_path.exists() else None,
        }
        if rollouts.exists():
            rollout_summary = summarize_rollouts([str(rollouts)])
            item["rollout_task_stats"] = rollout_summary.get("task_stats", {})
            item["rollout_rows"] = rollout_summary.get("rows", 0)
        if update_summary_path.exists():
            update = _read_json(update_summary_path)
            item["update"] = {
                "kept_frontier_rows": update.get("kept_frontier_rows"),
                "raw_frontier_task_counts": update.get("raw_frontier_task_counts"),
                "frontier_task_counts": update.get("frontier_task_counts"),
                "updates": update.get("updates"),
                "gate_grad_nonzero": update.get("gate_grad_nonzero"),
                "parameter_coefficients": update.get("parameter_coefficients"),
                "stopped_early_at_step": update.get("stopped_early_at_step"),
                "epoch_summaries": _compact_epoch_summaries(update.get("epoch_summaries", [])),
            }
        if gates_path.exists():
            gates = _load_gates(gates_path)
            item["gate_stats"] = _gate_stats(gates, init_value=init_value)
        iteration_summaries.append(item)

    reward_trends = _reward_trends(iteration_summaries)
    gate_trends = _gate_trends(iteration_summaries)
    alerts = _alerts(iteration_summaries, reward_trends, gate_trends)
    return {
        "format": "opvec_gate_strategy_run_summary_v1",
        "run_dir": str(run_dir),
        "strategy": strategy,
        "init_value": init_value,
        "iterations": iteration_summaries,
        "reward_trends": reward_trends,
        "gate_trends": gate_trends,
        "alerts": alerts,
    }


def _compact_epoch_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "epoch": row.get("epoch"),
                "grad_norm_max": row.get("grad_norm_max"),
                "gate_delta_max": row.get("gate_delta_max"),
                "mean_loss": row.get("mean_loss"),
                "mean_reward": row.get("mean_reward"),
            }
        )
    return compact


def _reward_trends(iteration_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in iteration_summaries:
        stats = item.get("rollout_task_stats") or {}
        for task, task_stats in stats.items():
            per_task[str(task)].append(
                {
                    "iteration": item["iteration"],
                    "mean_reward": _float(task_stats.get("mean_reward")),
                    "success_rate": _float(task_stats.get("success_rate")),
                    "kept_frontier_rows": int(task_stats.get("kept_frontier_rows") or 0),
                }
            )
    trends = {}
    for task, rows in sorted(per_task.items()):
        if len(rows) >= 2:
            reward_delta = rows[-1]["mean_reward"] - rows[0]["mean_reward"]
            success_delta = rows[-1]["success_rate"] - rows[0]["success_rate"]
        else:
            reward_delta = 0.0
            success_delta = 0.0
        trends[task] = {
            "points": rows,
            "mean_reward_delta_first_to_last": reward_delta,
            "success_rate_delta_first_to_last": success_delta,
        }
    return trends


def _gate_trends(iteration_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for item in iteration_summaries:
        gate_stats = item.get("gate_stats") or {}
        output[item["iteration"]] = {
            "num_coefficients": gate_stats.get("num_coefficients"),
            "max_abs_delta_from_init": gate_stats.get("max_abs_delta_from_init"),
            "mean_abs_delta_from_init": gate_stats.get("mean_abs_delta_from_init"),
            "expert_means": gate_stats.get("expert_means"),
            "expert_mean_delta_from_init": gate_stats.get("expert_mean_delta_from_init"),
        }
    return output


def _alerts(
    iteration_summaries: list[dict[str, Any]],
    reward_trends: dict[str, Any],
    gate_trends: dict[str, Any],
) -> list[dict[str, Any]]:
    alerts = []
    if not iteration_summaries:
        alerts.append({"level": "error", "reason": "no_iterations"})
        return alerts
    for item in iteration_summaries:
        update = item.get("update") or {}
        if update and not update.get("gate_grad_nonzero"):
            alerts.append({"level": "error", "iteration": item["iteration"], "reason": "gate_grad_zero"})
        frontier_counts = update.get("frontier_task_counts") or {}
        for task in EXPERTS:
            if frontier_counts and int(frontier_counts.get(task, 0)) == 0:
                alerts.append({"level": "warn", "iteration": item["iteration"], "task": task, "reason": "no_frontier_rows"})
    if len(iteration_summaries) >= 2:
        for task, trend in reward_trends.items():
            delta = float(trend.get("mean_reward_delta_first_to_last") or 0.0)
            if delta < -0.02:
                alerts.append({"level": "warn", "task": task, "reason": "mean_reward_decreased", "delta": delta})
    for iteration, trend in gate_trends.items():
        if trend.get("num_coefficients") and float(trend.get("max_abs_delta_from_init") or 0.0) < 1.0e-5:
            alerts.append({"level": "warn", "iteration": iteration, "reason": "gate_no_movement"})
    return alerts


def _gate_stats(gates: dict[str, float], *, init_value: float) -> dict[str, Any]:
    coeffs = _effective_coefficients(gates)
    values = [value for _, _, value in coeffs]
    by_expert: dict[str, list[float]] = defaultdict(list)
    for _, expert, value in coeffs:
        by_expert[expert].append(value)
    deltas = [value - init_value for value in values]
    abs_deltas = [abs(item) for item in deltas]
    top_changed = sorted(
        (
            {
                "name": name,
                "expert": expert,
                "value": value,
                "delta_from_init": value - init_value,
            }
            for name, expert, value in coeffs
        ),
        key=lambda row: abs(float(row["delta_from_init"])),
        reverse=True,
    )[:20]
    return {
        "num_coefficients": len(values),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean_abs_delta_from_init": mean(abs_deltas) if abs_deltas else 0.0,
        "max_abs_delta_from_init": max(abs_deltas) if abs_deltas else 0.0,
        "expert_means": {expert: mean(vals) for expert, vals in sorted(by_expert.items()) if vals},
        "expert_mean_delta_from_init": {
            expert: mean(vals) - init_value for expert, vals in sorted(by_expert.items()) if vals
        },
        "top_changed": top_changed,
    }


def _effective_coefficients(gates: dict[str, float]) -> list[tuple[str, str, float]]:
    param_keys = [key for key in gates if "::" in key and not key.startswith("__global__::")]
    if param_keys:
        output = []
        for key in sorted(param_keys):
            name, expert = key.rsplit("::", 1)
            if expert in EXPERTS:
                output.append((name, expert, float(gates[key])))
        return output

    if all(expert in gates for expert in EXPERTS):
        return [("global", expert, float(gates[expert])) for expert in EXPERTS]

    band_names = sorted({key.split(".", 1)[0] for key in gates if "." in key and "::" not in key})
    if band_names:
        output = []
        for band in band_names:
            common = float(gates.get(f"{band}.common", gates.get("common", 0.5)))
            residuals = [float(gates.get(f"{band}.{expert}_residual", gates.get(f"{expert}_residual", 0.0))) for expert in EXPERTS]
            residual_mean = sum(residuals) / len(residuals)
            for expert, residual in zip(EXPERTS, residuals):
                output.append((band, expert, common + residual - residual_mean))
        return output

    common = float(gates.get("common", 0.5))
    residuals = [float(gates.get(f"{expert}_residual", 0.0)) for expert in EXPERTS]
    residual_mean = sum(residuals) / len(residuals)
    return [("global", expert, common + residual - residual_mean) for expert, residual in zip(EXPERTS, residuals)]


def _load_gates(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    if isinstance(payload.get("gates"), dict):
        payload = payload["gates"]
    return {str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(output) or math.isinf(output):
        return 0.0
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--init-value", type=float, default=1.0 / 3.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
