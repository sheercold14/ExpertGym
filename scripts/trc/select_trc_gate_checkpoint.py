#!/usr/bin/env python3
"""Select a TRC epoch gate checkpoint from optimized loss."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve() if args.metrics else run_dir / "trc_metrics.jsonl"
    summaries = read_epoch_summaries(metrics_path)
    if not summaries:
        raise ValueError(f"No epoch summaries found in {metrics_path}")
    scored = [score_epoch(row, args) for row in summaries]
    selected = select_by_loss(scored, args)
    epoch = int(selected["epoch"])
    gate_path = run_dir / f"epoch_{epoch:03d}.gates.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Selected gate file not found: {gate_path}")
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "selected.gates.json"
    shutil.copyfile(gate_path, output)
    report = {
        "format": "trc_gate_selection_v1",
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "selected_epoch": epoch,
        "selected_gate": str(output),
        "selected_source_gate": str(gate_path),
        "selection_args": vars(args),
        "selected": selected,
        "ranked": sorted(scored, key=lambda item: item["score"])[: int(args.top_k_report)],
    }
    report_path = run_dir / "selected.gates.selection.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def read_epoch_summaries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_events_by_epoch: dict[int, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") == "epoch":
            rows.append(obj)
        elif obj.get("event") == "row":
            row_events_by_epoch.setdefault(int(obj.get("epoch", 0)), []).append(obj)
    for row in rows:
        epoch = int(row.get("epoch", 0))
        events = row_events_by_epoch.get(epoch) or []
        if not events:
            continue
        row["mean_optimized_total_loss"] = mean(
            float(item.get("total_loss", 0.0)) * float(item.get("loss_scale", 1.0))
            for item in events
        )
        row["mean_optimized_residual_loss"] = mean(
            float(item.get("residual_loss", 0.0)) * float(item.get("loss_scale", 1.0))
            for item in events
        )
    return rows


def score_epoch(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    gates = row.get("gate_means") or {}
    total_loss = finite_float(row.get("mean_optimized_total_loss", row.get("mean_total_loss")), default=1.0e9)
    residual_loss = finite_float(row.get("mean_optimized_residual_loss", row.get("mean_residual_loss")), default=total_loss)
    raw_total_loss = finite_float(row.get("mean_total_loss"), default=total_loss)
    raw_residual_loss = finite_float(row.get("mean_residual_loss"), default=residual_loss)
    task_loss = row.get("task_loss") or {}
    penalties: dict[str, float] = {}
    gate_penalty_weight = float(args.gate_penalty)
    if gate_penalty_weight > 0.0:
        penalties["memory_min"] = squared_shortfall(gates.get("memory"), float(args.min_memory_gate), gate_penalty_weight)
        penalties["tool_min"] = squared_shortfall(gates.get("tool"), float(args.min_tool_gate), gate_penalty_weight)
        penalties["code_min"] = squared_shortfall(gates.get("code"), float(args.min_code_gate), gate_penalty_weight)
        penalties["code_max"] = squared_excess(gates.get("code"), float(args.max_code_gate), gate_penalty_weight)
        penalties["tool_max"] = squared_excess(gates.get("tool"), float(args.max_tool_gate), gate_penalty_weight)
        penalties["memory_max"] = squared_excess(gates.get("memory"), float(args.max_memory_gate), gate_penalty_weight)
    task_penalty = 0.0
    for task, max_loss in parse_task_loss_ceilings(args.max_task_loss).items():
        if task in task_loss:
            task_penalty += squared_excess(task_loss[task].get("total_loss"), max_loss, float(args.task_loss_penalty))
    loss_score = total_loss + float(args.residual_weight) * residual_loss + task_penalty
    score = loss_score + sum(penalties.values())
    return {
        "epoch": int(row.get("epoch", 0)),
        "score": score,
        "loss_score": loss_score,
        "mean_total_loss": raw_total_loss,
        "mean_residual_loss": raw_residual_loss,
        "mean_optimized_total_loss": total_loss,
        "mean_optimized_residual_loss": residual_loss,
        "gate_means": gates,
        "task_loss": {
            task: {
                "total_loss": value.get("total_loss"),
                "residual_loss": value.get("residual_loss"),
                "span_tokens": value.get("span_tokens"),
            }
            for task, value in task_loss.items()
        },
        "penalties": penalties,
        "task_penalty": task_penalty,
    }


def select_by_loss(scored: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ordered = sorted(scored, key=lambda item: int(item["epoch"]))
    mode = str(args.selection_mode)
    if mode == "loss-min":
        return min(scored, key=lambda item: item["loss_score"])
    if mode != "loss-plateau":
        raise ValueError(f"Unsupported selection mode: {mode}")
    patience = max(1, int(args.plateau_patience))
    threshold = float(args.plateau_relative_improvement)
    min_epoch = int(args.plateau_min_epoch)
    for idx, item in enumerate(ordered):
        if int(item["epoch"]) < min_epoch:
            continue
        future_idx = idx + patience
        if future_idx >= len(ordered):
            break
        current = float(item["loss_score"])
        future = min(float(row["loss_score"]) for row in ordered[idx + 1 : future_idx + 1])
        relative_improvement = (current - future) / max(abs(current), 1.0e-12)
        if relative_improvement < threshold:
            item = dict(item)
            item["plateau_selected"] = True
            item["plateau_relative_improvement"] = relative_improvement
            item["plateau_patience"] = patience
            return item
    best = min(scored, key=lambda item: item["loss_score"])
    best = dict(best)
    best["plateau_selected"] = False
    best["plateau_reason"] = "no plateau found; selected minimum loss"
    return best


def parse_task_loss_ceilings(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Task loss ceiling must use task=value format: {raw!r}")
        task, value = raw.split("=", 1)
        out[task.strip()] = float(value.strip())
    return out


def finite_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def squared_shortfall(value: Any, minimum: float, weight: float) -> float:
    number = finite_float(value, default=minimum)
    return max(0.0, minimum - number) ** 2 * weight


def squared_excess(value: Any, maximum: float, weight: float) -> float:
    number = finite_float(value, default=maximum)
    return max(0.0, number - maximum) ** 2 * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--selection-mode", choices=["loss-min", "loss-plateau"], default="loss-plateau")
    parser.add_argument("--plateau-relative-improvement", type=float, default=0.01)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-min-epoch", type=int, default=4)
    parser.add_argument("--min-memory-gate", type=float, default=0.82)
    parser.add_argument("--max-memory-gate", type=float, default=1.20)
    parser.add_argument("--min-tool-gate", type=float, default=1.05)
    parser.add_argument("--max-tool-gate", type=float, default=1.20)
    parser.add_argument("--min-code-gate", type=float, default=1.08)
    parser.add_argument("--max-code-gate", type=float, default=1.18)
    parser.add_argument("--gate-penalty", type=float, default=0.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument("--max-task-loss", action="append", default=[])
    parser.add_argument("--task-loss-penalty", type=float, default=1.0)
    parser.add_argument("--top-k-report", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
