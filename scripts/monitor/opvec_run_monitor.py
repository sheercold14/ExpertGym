#!/usr/bin/env python3
"""Read-only OP-VEC run monitor with a small browser UI."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_EXPERTS = ("tool", "memory", "code")
PAPER96_EVAL_MODELS = {
    "A_gc_opd": "paper96-a-gc-fixedopd-final-iter8",
    "B_gc_noopd": "paper96-b-gc-noopd-final-iter8",
    "C_gp_opd": "paper96-c-gp-fixedopd-final-iter8",
    "D_gc_dynamic_opd": "paper96-d-gc-dynopd-final-iter8",
}
EVAL6_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation")
EVAL6_RUN_ID = "eval6-20260502-125748"


def main() -> None:
    args = parse_args()
    run_specs = [_parse_run_dir_spec(item) for item in args.run_dir]
    iteration_offsets = _parse_iteration_offsets(args.run_iteration_offset)
    missing = [str(run_dir) for _, run_dir in run_specs if not run_dir.exists()]
    if missing:
        raise SystemExit(f"run dir not found: {missing[0]}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                max_prompt_rows = int(query.get("max_prompt_rows", [args.max_prompt_rows])[0])
                state = _build_api_state(
                    run_specs,
                    init_value=args.init_value,
                    max_prompt_rows=max_prompt_rows,
                    iteration_offsets=iteration_offsets,
                )
                self._send_json(state)
                return
            self.send_error(404)

        def log_message(self, fmt: str, *items: Any) -> None:
            if args.quiet:
                return
            super().log_message(fmt, *items)

        def _send_html(self, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[monitor] runs=" + ", ".join(f"{run_id}={run_dir}" for run_id, run_dir in run_specs), flush=True)
    print(f"[monitor] url=http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parse_run_dir_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        run_id, raw_path = value.split("=", 1)
        run_id = run_id.strip()
    else:
        raw_path = value
        run_id = ""
    run_dir = Path(raw_path).expanduser().resolve()
    return run_id or run_dir.name, run_dir


def _parse_iteration_offsets(items: list[str]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for item in items or []:
        if "=" not in str(item):
            raise ValueError(f"--run-iteration-offset expects RUN_ID=OFFSET, got: {item}")
        run_id, raw_offset = str(item).split("=", 1)
        run_id = run_id.strip()
        if not run_id:
            raise ValueError(f"--run-iteration-offset has empty RUN_ID: {item}")
        offsets[run_id] = int(raw_offset)
    return offsets


def _build_api_state(
    run_specs: list[tuple[str, Path]],
    *,
    init_value: float,
    max_prompt_rows: int,
    iteration_offsets: dict[str, int] | None = None,
) -> dict[str, Any]:
    iteration_offsets = dict(iteration_offsets or {})
    if len(run_specs) == 1:
        run_id, run_dir = run_specs[0]
        state = build_state(
            run_dir,
            init_value=init_value,
            max_prompt_rows=max_prompt_rows,
            iteration_offset=iteration_offsets.get(run_id, 0),
        )
        state["run_id"] = run_id
        return state
    runs = []
    for run_id, run_dir in run_specs:
        state = build_state(
            run_dir,
            init_value=init_value,
            max_prompt_rows=max_prompt_rows,
            iteration_offset=iteration_offsets.get(run_id, 0),
        )
        state["run_id"] = run_id
        runs.append(state)
    return {
        "format": "opvec_monitor_multi_state_v1",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
        "comparison_series": _comparison_series(runs),
    }


def build_state(run_dir: Path, *, init_value: float, max_prompt_rows: int, iteration_offset: int = 0) -> dict[str, Any]:
    iterations = []
    loop_manifest = _read_json(run_dir / "gated_grpo_bake_vllm_loop_manifest.json") or {}
    manifest_iterations = {
        f"iter_{int(item.get('iteration')):03d}": item
        for item in loop_manifest.get("iterations", [])
        if isinstance(item, dict) and item.get("iteration") is not None
    }
    initial_gate_values = _manifest_initial_gate_values(loop_manifest)
    for iter_dir in sorted(run_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        iterations.append(
            _iteration_state(
                iter_dir,
                init_value=init_value,
                initial_gate_values=initial_gate_values,
                max_prompt_rows=max_prompt_rows,
                manifest_iteration=manifest_iterations.get(iter_dir.name),
            )
        )
    _apply_iteration_offset(iterations, int(iteration_offset))
    return {
        "format": "opvec_monitor_state_v1",
        "run_dir": str(run_dir),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": iterations,
        "run_summary": _run_summary(iterations, loop_manifest),
        "eval6": _eval6_summary(run_dir),
        "reward_series": _reward_series(iterations),
        "coefficient_series": _coefficient_series(iterations),
        "metric_series": _metric_series(iterations),
        "alerts": _alerts(iterations),
    }


def _apply_iteration_offset(iterations: list[dict[str, Any]], offset: int) -> None:
    if int(offset) == 0:
        return
    for item in iterations:
        physical = str(item.get("iteration") or "")
        item["physical_iteration"] = physical
        item["iteration"] = _offset_iteration_name(physical, offset)


def _offset_iteration_name(name: str, offset: int) -> str:
    if not name.startswith("iter_"):
        return name
    try:
        value = int(name.split("_", 1)[1])
    except ValueError:
        return name
    return f"iter_{value + int(offset):03d}"


def _iteration_state(
    iter_dir: Path,
    *,
    init_value: float,
    initial_gate_values: dict[str, float] | None,
    max_prompt_rows: int,
    manifest_iteration: dict[str, Any] | None,
) -> dict[str, Any]:
    rollout_path = iter_dir / "rollouts.jsonl"
    rollout_summary_path = iter_dir / "rollouts.summary.json"
    update_summary_path = iter_dir / "gate_updates.summary.json"
    gates_path = iter_dir / "gate_updates.gates.json"
    dynamic_opd_summary_path = iter_dir / "opd_distill_from_allfail.summary.json"
    rows = _read_jsonl(rollout_path) if rollout_path.exists() else []
    prompt_rows = [_prompt_row(row) for row in rows[:max_prompt_rows]]
    task_stats = _task_stats(rows)
    rollout_summary = _read_json(rollout_summary_path) if rollout_summary_path.exists() else None
    update_summary = _read_json(update_summary_path) if update_summary_path.exists() else None
    dynamic_opd_summary = _read_json(dynamic_opd_summary_path) if dynamic_opd_summary_path.exists() else None
    gates = _load_gates(gates_path) if gates_path.exists() else {}
    status = _status(rollout_path=rollout_path, update_summary_path=update_summary_path, gates_path=gates_path)
    update = _compact_update(update_summary)
    gate_stats = _gate_stats(gates, init_value=init_value, initial_gates=initial_gate_values) if gates else None
    token_stats = _token_stats(rows)
    metrics = _iteration_metrics(
        task_stats=task_stats,
        rollout_rows=len(rows),
        rollout_summary=rollout_summary,
        update=update,
        gate_stats=gate_stats,
        token_stats=token_stats,
        dynamic_opd_summary=dynamic_opd_summary,
    )
    return {
        "iteration": iter_dir.name,
        "status": status,
        "paths": {
            "rollouts": str(rollout_path) if rollout_path.exists() else None,
            "rollout_summary": str(rollout_summary_path) if rollout_summary_path.exists() else None,
            "update_summary": str(update_summary_path) if update_summary_path.exists() else None,
            "gates": str(gates_path) if gates_path.exists() else None,
            "dynamic_opd_summary": str(dynamic_opd_summary_path) if dynamic_opd_summary_path.exists() else None,
        },
        "mtime": _max_mtime([rollout_path, rollout_summary_path, update_summary_path, gates_path, dynamic_opd_summary_path]),
        "rollout_rows": len(rows),
        "task_stats": task_stats,
        "prompt_rows": prompt_rows,
        "rollout_summary": rollout_summary,
        "update": update,
        "dynamic_opd": _compact_dynamic_opd(dynamic_opd_summary),
        "gate_stats": gate_stats,
        "metrics": metrics,
        "timings": _timings(manifest_iteration, rollout_summary),
        "token_stats": token_stats,
    }


def _status(*, rollout_path: Path, update_summary_path: Path, gates_path: Path) -> str:
    if gates_path.exists():
        return "updated"
    if update_summary_path.exists():
        return "update_summary_only"
    if rollout_path.exists():
        return "rollout_or_update_running"
    return "pending"


def _prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") if isinstance(row.get("samples"), list) else []
    rewards = [_float(sample.get("reward")) for sample in samples if isinstance(sample, dict)]
    successes = [bool(sample.get("success")) or _float(sample.get("reward")) > 0.0 for sample in samples if isinstance(sample, dict)]
    token_lengths = [_token_count(sample) for sample in samples if isinstance(sample, dict)]
    return {
        "prompt_id": row.get("prompt_id"),
        "task": row.get("task"),
        "keep_for_policy_loss": bool(row.get("keep_for_policy_loss")),
        "skip_reason": row.get("skip_reason"),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "success_count": sum(1 for item in successes if item),
        "token_mean": mean(token_lengths) if token_lengths else 0.0,
        "token_max": max(token_lengths) if token_lengths else 0.0,
        "samples": [
            {
                "sample_id": sample.get("sample_id"),
                "reward": _float(sample.get("reward")),
                "success": bool(sample.get("success")) or _float(sample.get("reward")) > 0.0,
                "length": _float(sample.get("length")),
                "tokens": _token_count(sample),
                "text_preview": str(sample.get("text") or "")[:180].replace("\n", " "),
            }
            for sample in samples
            if isinstance(sample, dict)
        ],
    }


def _task_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "samples": 0, "rewards": [], "successes": 0, "kept": 0, "token_lengths": []}
    )
    for row in rows:
        task = str(row.get("task") or "unknown")
        acc[task]["rows"] += 1
        acc[task]["kept"] += int(bool(row.get("keep_for_policy_loss")))
        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            reward = _train_reward(sample)
            acc[task]["samples"] += 1
            acc[task]["rewards"].append(reward)
            acc[task]["successes"] += int(bool(sample.get("success", reward > 0.5)))
            acc[task]["token_lengths"].append(_token_count(sample))
    output = {}
    for task, item in sorted(acc.items()):
        rewards = item["rewards"]
        token_lengths = item["token_lengths"]
        samples = int(item["samples"])
        output[task] = {
            "rows": int(item["rows"]),
            "samples": samples,
            "kept_frontier_rows": int(item["kept"]),
            "frontier_ratio": float(item["kept"]) / float(item["rows"]) if item["rows"] else 0.0,
            "mean_reward": mean(rewards) if rewards else 0.0,
            "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "success_rate": float(item["successes"]) / samples if samples else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "token_mean": mean(token_lengths) if token_lengths else 0.0,
            "token_max": max(token_lengths) if token_lengths else 0.0,
            "reward_buckets": _reward_buckets(rewards),
        }
    return output


def _compact_update(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "kept_frontier_rows": payload.get("kept_frontier_rows"),
        "frontier_task_counts": payload.get("frontier_task_counts"),
        "raw_frontier_task_counts": payload.get("raw_frontier_task_counts"),
        "updates": payload.get("updates"),
        "optimizer_steps": payload.get("optimizer_steps"),
        "skipped_optimizer_steps": payload.get("skipped_optimizer_steps"),
        "filled_missing_old_logprobs": payload.get("filled_missing_old_logprobs"),
        "gate_grad_nonzero": payload.get("gate_grad_nonzero"),
        "gate_parameterization": payload.get("gate_parameterization"),
        "parameter_coefficients": payload.get("parameter_coefficients"),
        "stopped_early_at_step": payload.get("stopped_early_at_step"),
        "opd_distill_rows": payload.get("opd_distill_rows"),
        "opd_distill_task_counts": payload.get("opd_distill_task_counts"),
        "opd_all_success_rows": payload.get("opd_all_success_rows"),
        "opd_all_success_task_counts": payload.get("opd_all_success_task_counts"),
        "retention_rows": payload.get("retention_rows"),
        "optimizer": payload.get("optimizer"),
        "loss_weights": payload.get("loss_weights"),
        "epoch_summaries": [
            {
                "step": item.get("step"),
                "updates": item.get("updates"),
                "grad_norm_max": item.get("grad_norm_max"),
                "gate_delta_max": item.get("gate_delta_max"),
                "approx_kl_mean": item.get("approx_kl_mean"),
                "clip_frac_mean": item.get("clip_frac_mean"),
            }
            for item in payload.get("epoch_summaries", [])
            if isinstance(item, dict)
        ],
    }


def _compact_dynamic_opd(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    skipped = payload.get("skipped") if isinstance(payload.get("skipped"), dict) else {}
    selected = int(payload.get("selected_rows") or 0)
    all_fail_rows = selected + sum(int(value or 0) for key, value in skipped.items() if key != "current_not_failure")
    return {
        "selected_rows": selected,
        "selected_task_counts": payload.get("selected_task_counts") or {},
        "candidate_task_counts": payload.get("candidate_task_counts") or {},
        "all_fail_rows": all_fail_rows,
        "current_not_failure": int(skipped.get("current_not_failure") or 0),
        "skipped": skipped,
        "current_max_success": payload.get("current_max_success"),
    }


def _iteration_metrics(
    *,
    task_stats: dict[str, Any],
    rollout_rows: int,
    rollout_summary: dict[str, Any] | None,
    update: dict[str, Any] | None,
    gate_stats: dict[str, Any] | None,
    token_stats: dict[str, Any],
    dynamic_opd_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    samples = sum(int(stats.get("samples") or 0) for stats in task_stats.values())
    reward_numerator = sum(_float(stats.get("mean_reward")) * int(stats.get("samples") or 0) for stats in task_stats.values())
    kept_from_tasks = sum(int(stats.get("kept_frontier_rows") or 0) for stats in task_stats.values())
    frontier_counts = (update or {}).get("frontier_task_counts") or {}
    kept_from_update = (update or {}).get("kept_frontier_rows")
    if kept_from_update is None and frontier_counts:
        kept_from_update = sum(int(value or 0) for value in frontier_counts.values())
    kept_from_summary = (rollout_summary or {}).get("kept_frontiers")
    epochs = (update or {}).get("epoch_summaries") or []
    last_epoch = epochs[-1] if epochs else {}
    optimizer_steps = (update or {}).get("optimizer_steps")
    dynamic_opd = _compact_dynamic_opd(dynamic_opd_summary)
    opd_rows = int((dynamic_opd or {}).get("selected_rows") or (update or {}).get("opd_distill_rows") or 0)
    opd_task_counts = (dynamic_opd or {}).get("selected_task_counts") or (update or {}).get("opd_distill_task_counts") or {}
    return {
        "mean_reward": reward_numerator / samples if samples else 0.0,
        "samples": samples,
        "rollout_rows": rollout_rows,
        "kept_frontier_rows": int(kept_from_update if kept_from_update is not None else kept_from_summary or kept_from_tasks),
        "optimizer_steps": int(optimizer_steps or 0),
        "updates": int((update or {}).get("updates") or last_epoch.get("updates") or 0),
        "retention_rows": int((update or {}).get("retention_rows") or 0),
        "opd_distill_rows": opd_rows,
        "opd_all_success_rows": int((update or {}).get("opd_all_success_rows") or 0),
        "all_fail_rows": int((dynamic_opd or {}).get("all_fail_rows") or 0),
        "grad_norm_max": _float(last_epoch.get("grad_norm_max")),
        "gate_delta_max": _float(last_epoch.get("gate_delta_max")),
        "clip_frac_mean": _float(last_epoch.get("clip_frac_mean")),
        "approx_kl_mean": _float(last_epoch.get("approx_kl_mean")),
        "delta_from_init_mean_abs": _float((gate_stats or {}).get("mean_abs_delta_from_init")),
        "delta_from_init_max_abs": _float((gate_stats or {}).get("max_abs_delta_from_init")),
        "token_p95": _float(token_stats.get("p95")),
        "old_logprob_coverage": (
            _float(token_stats.get("samples_with_old_logprobs")) / _float(token_stats.get("samples"))
            if token_stats.get("samples")
            else 0.0
        ),
        "task_rewards": {task: _float(stats.get("mean_reward")) for task, stats in sorted(task_stats.items())},
        "task_frontier_rows": {task: int(stats.get("kept_frontier_rows") or 0) for task, stats in sorted(task_stats.items())},
        "opd_task_rows": {task: int(opd_task_counts.get(task) or 0) for task in DEFAULT_EXPERTS},
        "expert_coefficients": dict((gate_stats or {}).get("expert_means") or {}),
        "expert_delta_from_init": dict((gate_stats or {}).get("expert_delta") or {}),
    }


def _run_summary(iterations: list[dict[str, Any]], loop_manifest: dict[str, Any]) -> dict[str, Any]:
    latest = iterations[-1] if iterations else {}
    total_rows = sum(int(item.get("rollout_rows") or 0) for item in iterations)
    total_frontiers = 0
    for item in iterations:
        for stats in (item.get("task_stats") or {}).values():
            total_frontiers += int(stats.get("kept_frontier_rows") or 0)
    timings = [item.get("timings") or {} for item in iterations]
    return {
        "status": latest.get("status", "pending"),
        "iterations": len(iterations),
        "total_rows": total_rows,
        "total_frontier_rows": total_frontiers,
        "elapsed_seconds": loop_manifest.get("elapsed_seconds"),
        "final_gate_checkpoint": loop_manifest.get("final_gate_checkpoint"),
        "timing_totals": {
            "bake_seconds": sum(_float(item.get("bake_seconds")) for item in timings),
            "collect_seconds": sum(_float(item.get("collect_seconds", item.get("collect_internal_seconds"))) for item in timings),
            "update_seconds": sum(_float(item.get("update_seconds")) for item in timings),
            "iteration_seconds": sum(_float(item.get("iteration_seconds")) for item in timings),
        },
    }


def _timings(manifest_iteration: dict[str, Any] | None, rollout_summary: dict[str, Any] | None) -> dict[str, Any]:
    timings = dict((manifest_iteration or {}).get("timings") or {})
    if rollout_summary and "elapsed_seconds" in rollout_summary:
        timings.setdefault("collect_internal_seconds", rollout_summary.get("elapsed_seconds"))
        timings.setdefault("collect_seconds", rollout_summary.get("elapsed_seconds"))
    return timings


def _token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = []
    samples_with_old = 0
    samples = 0
    for row in rows:
        for sample in row.get("samples", []) if isinstance(row.get("samples"), list) else []:
            if not isinstance(sample, dict):
                continue
            samples += 1
            length = _token_count(sample)
            if length:
                lengths.append(length)
            if isinstance(sample.get("old_logprobs"), list) and sample.get("old_logprobs"):
                samples_with_old += 1
    return {
        "samples": samples,
        "samples_with_old_logprobs": samples_with_old,
        "mean": mean(lengths) if lengths else 0.0,
        "max": max(lengths) if lengths else 0.0,
        "p95": _percentile(lengths, 0.95),
    }


def _token_count(sample: dict[str, Any]) -> int:
    old_logprobs = sample.get("old_logprobs")
    if isinstance(old_logprobs, list):
        return len(old_logprobs)
    token_ids = sample.get("response_token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)
    return int(_float(sample.get("length")))


def _reward_buckets(rewards: list[float]) -> dict[str, int]:
    buckets = {"zero": 0, "low": 0, "mid": 0, "high": 0, "max": 0}
    if not rewards:
        return buckets
    max_reward = max(rewards) or 1.0
    for reward in rewards:
        norm = reward / max_reward if max_reward else 0.0
        if reward <= 0:
            buckets["zero"] += 1
        elif norm < 0.33:
            buckets["low"] += 1
        elif norm < 0.66:
            buckets["mid"] += 1
        elif norm < 0.999:
            buckets["high"] += 1
        else:
            buckets["max"] += 1
    return buckets


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def _manifest_initial_gate_values(loop_manifest: dict[str, Any]) -> dict[str, float] | None:
    for item in loop_manifest.get("iterations", []):
        if not isinstance(item, dict):
            continue
        path = item.get("input_gate_checkpoint")
        if not path:
            continue
        gate_path = Path(str(path))
        if gate_path.exists():
            return _load_gates(gate_path)
    return None


def _gate_stats(gates: dict[str, float], *, init_value: float, initial_gates: dict[str, float] | None = None) -> dict[str, Any]:
    coeffs = _effective_coefficients(gates)
    initial_coeffs = {
        (name, expert): value
        for name, expert, value in _effective_coefficients(initial_gates or {})
    }
    values = [value for _, _, value in coeffs]
    by_expert: dict[str, list[float]] = defaultdict(list)
    by_expert_delta: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    by_source_expert: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    deltas: list[float] = []
    for name, expert, value in coeffs:
        anchor = initial_coeffs.get((name, expert), init_value)
        delta = value - anchor
        by_expert[expert].append(value)
        by_expert_delta[expert].append(delta)
        source = _coefficient_source(name)
        by_source[source].append(value)
        by_source_expert[source][expert].append(value)
        deltas.append(delta)
    abs_deltas = [abs(delta) for delta in deltas]
    top_changed = sorted(
        [
            {
                "name": name,
                "expert": expert,
                "value": value,
                "delta": value - initial_coeffs.get((name, expert), init_value),
            }
            for name, expert, value in coeffs
        ],
        key=lambda item: abs(float(item["delta"])),
        reverse=True,
    )[:24]
    return {
        "num_coefficients": len(values),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean_abs_delta_from_init": mean(abs_deltas) if abs_deltas else 0.0,
        "max_abs_delta_from_init": max(abs_deltas) if abs_deltas else 0.0,
        "expert_means": {expert: mean(items) for expert, items in sorted(by_expert.items()) if items},
        "expert_delta": {expert: mean(items) for expert, items in sorted(by_expert_delta.items()) if items},
        "source_stats": {
            source: {
                "n": len(items),
                "mean": mean(items),
                "min": min(items),
                "max": max(items),
                "task_mean": {
                    expert: mean(expert_items)
                    for expert, expert_items in sorted(by_source_expert[source].items())
                    if expert_items
                },
            }
            for source, items in sorted(by_source.items())
            if items
        },
        "top_changed": top_changed,
    }


def _coefficient_source(name: str) -> str:
    if name == "global":
        return "global"
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


def _effective_coefficients(gates: dict[str, float]) -> list[tuple[str, str, float]]:
    experts = _infer_experts(gates)
    param_keys = [key for key in gates if "::" in key and not key.startswith("__global__::")]
    if param_keys:
        output = []
        for key in sorted(param_keys):
            name, expert = key.rsplit("::", 1)
            if expert in experts:
                output.append((name, expert, float(gates[key])))
        return output
    if all(expert in gates for expert in experts):
        return [("global", expert, float(gates[expert])) for expert in experts]
    band_names = sorted({key.split(".", 1)[0] for key in gates if "." in key and "::" not in key})
    if band_names:
        output = []
        for band in band_names:
            if all(f"{band}.{expert}" in gates for expert in experts):
                for expert in experts:
                    output.append((band, expert, float(gates[f"{band}.{expert}"])))
                continue
            common = float(gates.get(f"{band}.common", gates.get("common", 0.5)))
            residuals = [float(gates.get(f"{band}.{expert}_residual", gates.get(f"{expert}_residual", 0.0))) for expert in experts]
            residual_mean = sum(residuals) / len(residuals)
            for expert, residual in zip(experts, residuals):
                output.append((band, expert, common + residual - residual_mean))
        return output
    common = float(gates.get("common", 0.5))
    residuals = [float(gates.get(f"{expert}_residual", 0.0)) for expert in experts]
    residual_mean = sum(residuals) / len(residuals)
    return [("global", expert, common + residual - residual_mean) for expert, residual in zip(experts, residuals)]


def _infer_experts(gates: dict[str, float]) -> tuple[str, ...]:
    experts = []
    for key in sorted(gates):
        if "::" in key:
            expert = key.rsplit("::", 1)[1]
        elif key.endswith("_residual"):
            tail = key.split(".", 1)[1] if "." in key else key
            expert = tail[: -len("_residual")]
        elif "." in key:
            expert = key.split(".", 1)[1]
            if expert == "common":
                continue
        elif "." not in key and key not in {"common"}:
            expert = key
        else:
            continue
        if expert and expert not in experts:
            experts.append(expert)
    return tuple(experts) if experts else DEFAULT_EXPERTS


def _reward_series(iterations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in iterations:
        for task, stats in (item.get("task_stats") or {}).items():
            series[task].append(
                {
                    "iteration": item["iteration"],
                    "mean_reward": stats.get("mean_reward", 0.0),
                    "success_rate": stats.get("success_rate", 0.0),
                    "kept_frontier_rows": stats.get("kept_frontier_rows", 0),
                }
            )
    return dict(sorted(series.items()))


def _coefficient_series(iterations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    experts = sorted(
        {
            str(expert)
            for item in iterations
            for expert in ((item.get("gate_stats") or {}).get("expert_means") or {})
        }
    )
    if not experts:
        experts = list(DEFAULT_EXPERTS)
    series: dict[str, list[dict[str, Any]]] = {expert: [] for expert in experts}
    for item in iterations:
        gate_stats = item.get("gate_stats") or {}
        means = gate_stats.get("expert_means") or {}
        deltas = gate_stats.get("expert_delta") or {}
        for expert in experts:
            if expert in means:
                series[expert].append(
                    {
                        "iteration": item["iteration"],
                        "value": means[expert],
                        "delta": deltas.get(expert, 0.0),
                    }
                )
    return series


def _metric_series(iterations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scalar_fields = (
        "mean_reward",
        "kept_frontier_rows",
        "retention_rows",
        "opd_distill_rows",
        "opd_all_success_rows",
        "all_fail_rows",
        "optimizer_steps",
        "updates",
        "grad_norm_max",
        "gate_delta_max",
        "clip_frac_mean",
        "approx_kl_mean",
        "delta_from_init_mean_abs",
        "delta_from_init_max_abs",
        "token_p95",
        "old_logprob_coverage",
    )
    series: dict[str, list[dict[str, Any]]] = {field: [] for field in scalar_fields}
    for item in iterations:
        metrics = item.get("metrics") or {}
        for field in scalar_fields:
            series[field].append({"iteration": item["iteration"], "value": _float(metrics.get(field))})
        for task, value in (metrics.get("task_rewards") or {}).items():
            series.setdefault(f"task_reward/{task}", []).append({"iteration": item["iteration"], "value": _float(value)})
        for task, value in (metrics.get("task_frontier_rows") or {}).items():
            series.setdefault(f"task_frontier_rows/{task}", []).append({"iteration": item["iteration"], "value": _float(value)})
        for task, value in (metrics.get("opd_task_rows") or {}).items():
            series.setdefault(f"opd_task_rows/{task}", []).append({"iteration": item["iteration"], "value": _float(value)})
        for expert, value in (metrics.get("expert_coefficients") or {}).items():
            series.setdefault(f"coefficient/{expert}", []).append({"iteration": item["iteration"], "value": _float(value)})
        for expert, value in (metrics.get("expert_delta_from_init") or {}).items():
            series.setdefault(f"delta_from_init/{expert}", []).append({"iteration": item["iteration"], "value": _float(value)})
        for field in ("bake_seconds", "collect_seconds", "update_seconds", "dynamic_opd_seconds", "iteration_seconds"):
            value = (item.get("timings") or {}).get(field)
            if value is not None:
                series.setdefault(f"timing/{field}", []).append({"iteration": item["iteration"], "value": _float(value)})
    return {key: values for key, values in sorted(series.items()) if values}


def _comparison_series(runs: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for run in runs:
        run_id = str(run.get("run_id") or Path(str(run.get("run_dir") or "")).name)
        for metric, values in (run.get("metric_series") or {}).items():
            output[metric][run_id] = values
    return {metric: dict(items) for metric, items in sorted(output.items())}


def _eval6_summary(run_dir: Path) -> dict[str, Any]:
    model_name = _paper96_eval_model_name(run_dir)
    output: dict[str, Any] = {
        "model_name": model_name,
        "run_id": EVAL6_RUN_ID,
        "tool": {"status": "missing"},
        "memory": {"status": "missing"},
        "code": {"status": "missing"},
    }
    if not model_name:
        return output
    tool_path = EVAL6_ROOT / "eval-batch" / EVAL6_RUN_ID / "runs" / model_name / "tool" / "summary.json"
    memory_path = EVAL6_ROOT / "eval6-memory-hotpotqa" / model_name / EVAL6_RUN_ID / "summary.json"
    code_path = EVAL6_ROOT / "eval6-code-cure-full" / model_name / EVAL6_RUN_ID / "summary.json"
    output["tool"] = _compact_tool_eval(tool_path)
    output["memory"] = _compact_memory_eval(memory_path)
    output["code"] = _compact_code_eval(code_path)
    return output


def _paper96_eval_model_name(run_dir: Path) -> str | None:
    name = run_dir.name
    for key, model_name in PAPER96_EVAL_MODELS.items():
        if key in name:
            return model_name
    return None


def _compact_tool_eval(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        return {"status": "missing", "path": str(path)}
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    values = [_float(item.get("accuracy")) for item in scores.values() if isinstance(item, dict)]
    live_values = [
        _float(scores[key].get("accuracy"))
        for key in ("live_parallel", "live_parallel_multiple")
        if isinstance(scores.get(key), dict)
    ]
    return {
        "status": "ready",
        "path": str(path),
        "mean": mean(values) if values else 0.0,
        "live_mean": mean(live_values) if live_values else 0.0,
        "scores": {key: _float(value.get("accuracy")) for key, value in sorted(scores.items()) if isinstance(value, dict)},
    }


def _compact_memory_eval(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        return {"status": "missing", "path": str(path)}
    datasets = [item for item in payload.get("datasets", []) if isinstance(item, dict)]
    f1_values = [_float(item.get("avg_f1")) for item in datasets]
    em_values = [_float(item.get("exact_match_rate")) for item in datasets]
    return {
        "status": "ready",
        "path": str(path),
        "f1_mean": mean(f1_values) if f1_values else 0.0,
        "em_mean": mean(em_values) if em_values else 0.0,
        "datasets": {
            str(item.get("dataset")): {
                "f1": _float(item.get("avg_f1")),
                "em": _float(item.get("exact_match_rate")),
            }
            for item in datasets
        },
    }


def _compact_code_eval(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        return {"status": "missing", "path": str(path)}
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    scores: dict[str, dict[str, float]] = {}
    for item in datasets:
        if not isinstance(item, dict):
            continue
        dataset = str(item.get("dataset") or item.get("name") or "unknown")
        scores[dataset] = {
            "acc": _float(item.get("code_acc", item.get("acc", item.get("accuracy")))),
            "tp": _float(item.get("test_point_acc", item.get("test_point_accuracy"))),
            "bon": _float(item.get("bon_acc", item.get("best_of_n_acc"))),
        }
    acc_values = [value["acc"] for value in scores.values()]
    return {"status": "ready", "path": str(path), "acc_mean": mean(acc_values) if acc_values else 0.0, "datasets": scores}


def _alerts(iterations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    for item in iterations:
        update = item.get("update") or {}
        if update and update.get("gate_grad_nonzero") is False:
            alerts.append({"level": "error", "iteration": item["iteration"], "reason": "gate_grad_zero"})
        counts = update.get("frontier_task_counts") or {}
        for task in DEFAULT_EXPERTS:
            if counts and int(counts.get(task, 0)) == 0:
                alerts.append({"level": "warn", "iteration": item["iteration"], "task": task, "reason": "no_frontier_rows"})
    return alerts


def _load_gates(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    if isinstance(payload.get("gates"), dict):
        payload = payload["gates"]
    return {str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def _max_mtime(paths: list[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else None


def _float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(output) or math.isinf(output):
        return 0.0
    return output


def _train_reward(sample: dict[str, Any]) -> float:
    if "reward_train" in sample:
        return _float(sample.get("reward_train"))
    raw = _float(sample.get("reward"))
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    if details.get("toolrl_score_range") == [-3.0, 4.0]:
        return max(0.0, min((raw + 3.0) / 7.0, 1.0))
    return max(0.0, min(raw, 1.0))


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OP-VEC Monitor</title>
  <style>
    :root { color-scheme: light; --bg:#f7f8fb; --panel:#ffffff; --panel2:#f2f5f9; --line:#d8dee9; --text:#19202d; --muted:#687386; --tool:#0a8f5a; --memory:#2f64c5; --code:#c46a16; --reasoning:#7a3fc0; --bad:#c93333; --ok:#18864b; --warn:#9b6a00; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
    header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; align-items:flex-start; justify-content:space-between; gap:18px; position:sticky; top:0; background:rgba(255,255,255,.96); z-index:5; }
    h1 { margin:0 0 4px; font-size:20px; font-weight:650; }
    .path { color:var(--muted); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
    .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:12px; padding:12px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px; min-width:0; box-shadow:0 1px 2px rgba(16,24,40,.06); }
    .span3 { grid-column:span 3; } .span4 { grid-column:span 4; } .span6 { grid-column:span 6; } .span8 { grid-column:span 8; } .span12 { grid-column:span 12; }
    h2 { margin:0 0 8px; font-size:14px; font-weight:700; color:var(--text); }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-weight:600; font-size:12px; }
    .pill { display:inline-block; padding:2px 7px; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px; }
    .num { font-variant-numeric:tabular-nums; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
    .task-tool { color:var(--tool); } .task-memory { color:var(--memory); } .task-code { color:var(--code); } .task-reasoning { color:var(--reasoning); }
    canvas { width:100%; height:270px; display:block; background:#fff; border:1px solid var(--line); border-radius:4px; }
    .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
    select, input { background:#fff; color:var(--text); border:1px solid var(--line); border-radius:4px; padding:5px 7px; }
    .samples { display:flex; gap:4px; align-items:center; min-width:140px; }
    .bar { height:14px; min-width:8px; border-radius:2px; background:#d3d9e5; }
    .bar.pos { background:#43c878; } .bar.neg { background:#d85d5d; }
    .alert { color:var(--bad); }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .card { background:var(--panel2); border:1px solid var(--line); border-radius:4px; padding:9px; min-height:74px; }
    .label { color:var(--muted); font-size:12px; margin-bottom:4px; }
    .value { font:650 22px/1.1 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .sub { color:var(--muted); font-size:12px; margin-top:4px; overflow-wrap:anywhere; }
    .spark { height:8px; border-radius:99px; background:#222936; overflow:hidden; margin-top:8px; display:flex; }
    .seg-tool { background:var(--tool); } .seg-memory { background:var(--memory); } .seg-code { background:var(--code); } .seg-reasoning { background:var(--reasoning); }
    .small { font-size:12px; color:var(--muted); }
    .header-right { display:flex; flex-direction:column; gap:6px; align-items:flex-end; min-width:260px; }
    .run-select { display:none; align-items:center; gap:8px; }
    #runSelect { min-width:250px; }
    .chart-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px; }
    .legend { display:flex; flex-wrap:wrap; gap:8px 14px; color:var(--muted); font-size:12px; margin-top:8px; }
    .legend span { display:inline-flex; align-items:center; gap:5px; }
    .swatch { width:18px; height:3px; border-radius:99px; display:inline-block; }
    @media (max-width: 980px) { .span3, .span4, .span6, .span8 { grid-column:span 12; } .cards { grid-template-columns:1fr 1fr; } header { position:static; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>OP-VEC Run Monitor</h1>
      <div id="runDir" class="path"></div>
    </div>
    <div class="header-right">
      <label id="runSelectWrap" class="small run-select">run <select id="runSelect"></select></label>
      <div class="num" id="updatedAt">loading</div>
    </div>
  </header>
  <main class="grid">
    <section class="panel span12"><h2>四实验总览</h2><div id="allRuns"></div></section>
    <section class="panel span12"><h2>Run Overview</h2><div id="overview" class="cards"></div></section>
    <section class="panel span12"><h2>Eval6 状态入口</h2><div id="eval6"></div></section>
    <section class="panel span12"><h2>Alerts</h2><div id="alerts" class="path">loading</div></section>
    <section class="panel span4"><h2>Tool 训练奖励</h2><canvas id="taskToolChart" width="720" height="260"></canvas><div id="taskToolLegend" class="legend"></div></section>
    <section class="panel span4"><h2>Memory 训练奖励</h2><canvas id="taskMemoryChart" width="720" height="260"></canvas><div id="taskMemoryLegend" class="legend"></div></section>
    <section class="panel span4"><h2>Code 训练奖励</h2><canvas id="taskCodeChart" width="720" height="260"></canvas><div id="taskCodeLegend" class="legend"></div></section>
    <section class="panel span4"><h2>Frontier Rows</h2><canvas id="frontierChart" width="720" height="260"></canvas><div id="frontierLegend" class="legend"></div></section>
    <section class="panel span4"><h2>OPD / All-fail Rows</h2><canvas id="opdChart" width="720" height="260"></canvas><div id="opdLegend" class="legend"></div></section>
    <section class="panel span4"><h2>Update 耗时</h2><canvas id="updateTimeChart" width="720" height="260"></canvas><div id="updateTimeLegend" class="legend"></div></section>
    <section class="panel span6">
      <div class="chart-head"><h2 id="rewardChartTitle">Strategy Metrics</h2><select id="metricSelect"></select></div>
      <canvas id="rewardChart" width="900" height="330"></canvas><div id="rewardLegend" class="legend"></div>
    </section>
    <section class="panel span6">
      <div class="chart-head"><h2>Effective Gate / Task-Vector Coefficients</h2><select id="coefSelect"></select></div>
      <canvas id="coefChart" width="900" height="330"></canvas><div id="coefLegend" class="legend"></div>
    </section>
    <section class="panel span6"><h2>Stage Timing</h2><div id="timings"></div></section>
    <section class="panel span6"><h2>Gate / Update</h2><div id="gateUpdate"></div></section>
    <section class="panel span12"><h2>Iterations</h2><div id="iterations"></div></section>
    <section class="panel span12">
      <h2>Prompt Rewards</h2>
      <div class="controls">
        <label>iteration <select id="iterFilter"></select></label>
        <label>task <select id="taskFilter"><option value="">all</option><option>tool</option><option>memory</option><option>code</option></select></label>
        <label>max rows <input id="maxRows" type="number" min="20" max="1000" value="300"></label>
      </div>
      <div id="promptRows"></div>
    </section>
  </main>
  <script>
    const taskNames = ['tool','memory','code'];
    const preferredExperts = ['tool','memory','code','reasoning'];
    const colors = {tool:'#0a8f5a', memory:'#2f64c5', code:'#c46a16', reasoning:'#7a3fc0'};
    const strategyColors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#17becf'];
    const metricLabels = {
      mean_reward:'mean reward',
      'task_reward/tool':'tool reward',
      'task_reward/memory':'memory reward',
      'task_reward/code':'code reward',
      kept_frontier_rows:'kept frontier rows',
      retention_rows:'retention rows',
      opd_distill_rows:'OPD rows',
      opd_all_success_rows:'all-success OPD rows',
      all_fail_rows:'all-fail rows',
      'opd_task_rows/tool':'OPD tool rows',
      'opd_task_rows/memory':'OPD memory rows',
      'opd_task_rows/code':'OPD code rows',
      'timing/bake_seconds':'bake seconds',
      'timing/collect_seconds':'rollout seconds',
      'timing/update_seconds':'update seconds',
      'timing/dynamic_opd_seconds':'dynamic OPD seconds',
      'timing/iteration_seconds':'iteration seconds',
      optimizer_steps:'optimizer steps',
      updates:'policy updates',
      grad_norm_max:'grad norm max',
      gate_delta_max:'gate delta max',
      clip_frac_mean:'clip frac',
      approx_kl_mean:'approx KL',
      delta_from_init_mean_abs:'mean |delta from init|',
      delta_from_init_max_abs:'max |delta from init|',
      old_logprob_coverage:'old logprob coverage',
      token_p95:'p95 response tokens',
    };
    let rootState = null;
    let state = null;
    let selectedRunId = localStorage.getItem('opvecMonitorRunId') || '';
    async function loadState() {
      const maxRows = document.getElementById('maxRows').value || 300;
      const res = await fetch('/api/state?max_prompt_rows=' + encodeURIComponent(maxRows), {cache:'no-store'});
      rootState = await res.json();
      state = pickState(rootState);
      render();
    }
    function pickState(payload) {
      if (!payload || !Array.isArray(payload.runs)) return payload;
      const run = payload.runs.find(r => r.run_id === selectedRunId) || payload.runs[0];
      selectedRunId = run ? run.run_id : '';
      return run;
    }
    function render() {
      renderRunSelector();
      document.getElementById('runDir').textContent = state.run_id ? `[${state.run_id}] ${state.run_dir}` : state.run_dir;
      document.getElementById('updatedAt').textContent = (rootState && rootState.updated_at) || state.updated_at;
      renderAllRuns();
      renderOverview();
      renderEval6();
      renderAlerts();
      renderTimings();
      renderGateUpdate();
      renderIterations();
      renderFilters();
      renderChartControls();
      drawMetricChart();
      drawCoef();
      drawPaperCharts();
      renderPromptRows();
    }
    function renderRunSelector() {
      const wrap = document.getElementById('runSelectWrap');
      const select = document.getElementById('runSelect');
      if (!rootState || !Array.isArray(rootState.runs)) {
        wrap.style.display = 'none';
        return;
      }
      wrap.style.display = 'inline-flex';
      const old = select.value || selectedRunId;
      select.innerHTML = rootState.runs.map(run => {
        const summary = run.run_summary || {};
        const label = `${run.run_id} · ${summary.status || 'pending'} · i${summary.iterations || 0}`;
        return `<option value="${esc(run.run_id)}">${esc(label)}</option>`;
      }).join('');
      if ([...select.options].some(option => option.value === old)) {
        select.value = old;
      } else if (select.options.length) {
        select.value = select.options[0].value;
      }
      selectedRunId = select.value;
      localStorage.setItem('opvecMonitorRunId', selectedRunId);
    }
    function renderAlerts() {
      const box = document.getElementById('alerts');
      if (!state.alerts.length) { box.textContent = 'no alerts'; return; }
      box.innerHTML = state.alerts.map(a => '<span class="alert">' + esc(JSON.stringify(a)) + '</span>').join('<br>');
    }
    function orderedExperts(experts) {
      const set = new Set((experts || []).filter(Boolean).map(String));
      const ordered = preferredExperts.filter(name => set.has(name));
      const rest = [...set].filter(name => !preferredExperts.includes(name)).sort();
      return [...ordered, ...rest];
    }
    function gateExpertsForRun(run) {
      const names = new Set();
      for (const it of (run.iterations || [])) {
        const coeffs = ((it.metrics || {}).expert_coefficients) || {};
        Object.keys(coeffs).forEach(name => names.add(name));
      }
      return orderedExperts(names.size ? [...names] : taskNames);
    }
    function expertsFromMetricKeys(keys) {
      const names = new Set();
      for (const key of keys || []) {
        if (key.startsWith('coefficient/')) names.add(key.slice('coefficient/'.length));
        if (key.startsWith('delta_from_init/')) names.add(key.slice('delta_from_init/'.length));
      }
      return orderedExperts(names.size ? [...names] : taskNames);
    }
    function taskClass(name) {
      return 'task-' + String(name || '').replace(/[^A-Za-z0-9_-]/g, '_');
    }
    function renderAllRuns() {
      const box = document.getElementById('allRuns');
      const runs = isMulti() ? rootState.runs : [state];
      let html = '<table><thead><tr><th>run</th><th>状态 / 轮次</th><th>当前阶段</th><th>耗时</th><th>reward tool/memory/code</th><th>gate coefficients</th><th>frontier</th><th>OPD / all-fail</th><th>Eval6</th></tr></thead><tbody>';
      for (const run of runs) {
        const summary = run.run_summary || {};
        const latest = run.iterations[run.iterations.length - 1] || {};
        const metrics = latest.metrics || {};
        const gateIteration = latestWithGate(run) || latest;
        const gateMetrics = gateIteration.metrics || {};
        const rewards = taskNames.map(t => `<span class="${taskClass(t)}">${t}</span> <span class="num">${fmt((metrics.task_rewards || {})[t])}</span>`).join('<br>');
        const gateNote = gateIteration.iteration && gateIteration.iteration !== latest.iteration ? `<div class="path">from ${esc(gateIteration.iteration)}</div>` : '';
        const gates = gateExpertsForRun(run).map(t => `<span class="${taskClass(t)}">${t}</span> <span class="num">${fmt((gateMetrics.expert_coefficients || {})[t])}</span>`).join('<br>') + gateNote;
        const frontier = JSON.stringify((latest.update || {}).frontier_task_counts || metrics.task_frontier_rows || {});
        const opdCounts = (latest.dynamic_opd || {}).selected_task_counts || (latest.update || {}).opd_distill_task_counts || {};
        const opd = `OPD ${metrics.opd_distill_rows || 0} ${esc(JSON.stringify(opdCounts))}<br>all-fail <span class="num">${metrics.all_fail_rows || 0}</span> retention <span class="num">${metrics.retention_rows || 0}</span>`;
        const eval6 = run.eval6 || {};
        html += `<tr><td><button type="button" onclick="selectRun('${esc(run.run_id)}')">${esc(run.run_id || '')}</button><div class="path">${esc(shortRunName(run.run_dir || ''))}</div></td><td><span class="pill">${esc(summary.status || 'pending')}</span><br><span class="num">${summary.iterations || 0} iters</span></td><td>${esc(latest.status || '')}</td><td class="num">${sec(summary.elapsed_seconds || (summary.timing_totals || {}).iteration_seconds)}</td><td>${rewards}</td><td>${gates}</td><td class="num">${esc(frontier)}</td><td class="num">${opd}</td><td>${evalCell(eval6)}</td></tr>`;
      }
      html += '</tbody></table>';
      box.innerHTML = html;
    }
    function latestWithGate(run) {
      const iterations = run.iterations || [];
      for (let i = iterations.length - 1; i >= 0; i--) {
        const coeffs = ((iterations[i].metrics || {}).expert_coefficients) || {};
        if (Object.keys(coeffs).length) return iterations[i];
      }
      return null;
    }
    function renderEval6() {
      const box = document.getElementById('eval6');
      const runs = isMulti() ? rootState.runs : [state];
      let html = '<table><thead><tr><th>run</th><th>model</th><th>Tool</th><th>Memory</th><th>Code</th></tr></thead><tbody>';
      for (const run of runs) {
        const e = run.eval6 || {};
        html += `<tr><td>${esc(run.run_id || '')}</td><td class="path">${esc(e.model_name || '')}</td><td>${toolEval(e.tool)}</td><td>${memoryEval(e.memory)}</td><td>${codeEval(e.code)}</td></tr>`;
      }
      html += '</tbody></table>';
      box.innerHTML = html;
    }
    function renderOverview() {
      const s = state.run_summary || {};
      const latest = state.iterations[state.iterations.length - 1] || {};
      const tasks = latest.task_stats || {};
      const totalTaskRows = Object.values(tasks).reduce((a, x) => a + Number(x.rows || 0), 0) || 1;
      const spark = taskNames.map(t => `<span class="seg-${t}" style="width:${100 * Number((tasks[t] || {}).rows || 0) / totalTaskRows}%"></span>`).join('');
      const timing = s.timing_totals || {};
      document.getElementById('overview').innerHTML = [
        card('status', esc(s.status || 'pending'), `iters ${s.iterations || 0}`),
        card('rollout rows', s.total_rows || 0, `frontier ${s.total_frontier_rows || 0}`),
        card('stage time', sec(timing.iteration_seconds), `bake ${sec(timing.bake_seconds)} / rollout ${sec(timing.collect_seconds)} / update ${sec(timing.update_seconds)}`),
        `<div class="card"><div class="label">latest task mix</div><div class="value">${totalTaskRows}</div><div class="spark">${spark}</div><div class="sub">tool/memory/code rows</div></div>`,
      ].join('');
    }
    function renderTimings() {
      let html = '<table><thead><tr><th>iter</th><th>bake</th><th>rollout</th><th>update</th><th>total</th><th>internal rollout</th></tr></thead><tbody>';
      for (const it of state.iterations) {
        const t = it.timings || {};
        html += `<tr><td class="num">${it.iteration}</td><td class="num">${sec(t.bake_seconds)}</td><td class="num">${sec(t.collect_seconds)}</td><td class="num">${sec(t.update_seconds)}</td><td class="num">${sec(t.iteration_seconds)}</td><td class="num">${sec(t.collect_internal_seconds)}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('timings').innerHTML = html;
    }
    function renderGateUpdate() {
      const latest = [...state.iterations].reverse().find(it => it.gate_stats || it.update) || state.iterations[state.iterations.length - 1] || {};
      const update = latest.update || {};
      const gate = latest.gate_stats || {};
      const epochs = update.epoch_summaries || [];
      const lastEpoch = epochs[epochs.length - 1] || {};
      const means = gate.expert_means || {};
      const deltas = gate.expert_delta || {};
      const experts = gateExpertsForRun(state);
      let cards = [
        card('source iteration', latest.iteration || '', latest.status || ''),
        card('optimizer steps', update.optimizer_steps ?? '', `updates ${update.updates ?? ''}`),
        card('grad norm max', fmt(lastEpoch.grad_norm_max), `KL ${fmt(lastEpoch.approx_kl_mean)} / clip ${fmt(lastEpoch.clip_frac_mean)}`),
        card('gate max delta', fmt(gate.max_abs_delta_from_init), `mean abs ${fmt(gate.mean_abs_delta_from_init)}`),
      ].join('');
      let table = '<table><thead><tr><th>expert</th><th>coefficient</th><th>delta</th></tr></thead><tbody>';
      for (const task of experts) table += `<tr><td class="${taskClass(task)}">${task}</td><td class="num">${fmt(means[task])}</td><td class="num">${fmt(deltas[task])}</td></tr>`;
      table += '</tbody></table>';
      const sources = Object.entries(gate.source_stats || {}).slice(0, 12);
      let sourceTable = '';
      if (sources.length > 1 || (sources[0] && sources[0][0] !== 'global')) {
        sourceTable = '<h2>Global-parameter source 聚合</h2><table><thead><tr><th>source</th><th>n</th><th>mean/min/max</th><th>task-wise mean</th></tr></thead><tbody>';
        for (const [source, s] of sources) {
          const tm = s.task_mean || {};
          sourceTable += `<tr><td class="path">${esc(source)}</td><td class="num">${s.n}</td><td class="num">${fmt(s.mean)} / ${fmt(s.min)} / ${fmt(s.max)}</td><td>${experts.map(t => `<span class="${taskClass(t)}">${t}</span> <span class="num">${fmt(tm[t])}</span>`).join('<br>')}</td></tr>`;
        }
        sourceTable += '</tbody></table>';
      }
      document.getElementById('gateUpdate').innerHTML = `<div class="cards">${cards}</div>${table}${sourceTable}`;
    }
    function renderIterations() {
      let html = '<table><thead><tr><th>iter</th><th>status</th><th>rows</th><th>task rewards</th><th>tokens</th><th>frontier</th><th>OPD / all-fail</th><th>gate mean / max delta</th></tr></thead><tbody>';
      for (const it of state.iterations) {
        const rewards = Object.entries(it.task_stats || {}).map(([task,s]) => `<span class="${taskClass(task)}">${task}</span> <span class="num">${fmt(s.mean_reward)}</span> sr <span class="num">${fmt(s.success_rate)}</span> fr <span class="num">${fmt(s.frontier_ratio)}</span>`).join('<br>');
        const frontier = it.update ? JSON.stringify(it.update.frontier_task_counts || {}) : '';
        const dyn = it.dynamic_opd || {};
        const opdCounts = dyn.selected_task_counts || (it.update || {}).opd_distill_task_counts || {};
        const opd = `OPD ${fmt((it.metrics || {}).opd_distill_rows)} ${esc(JSON.stringify(opdCounts))}<br>all-fail ${fmt((it.metrics || {}).all_fail_rows)} retention ${fmt((it.metrics || {}).retention_rows)}`;
        const gs = it.gate_stats;
        const gate = gs ? `n=${gs.num_coefficients} mean=${fmt(gs.mean)} max_d=${fmt(gs.max_abs_delta_from_init)}` : '';
        const toks = it.token_stats ? `old ${it.token_stats.samples_with_old_logprobs}/${it.token_stats.samples} mean ${fmt(it.token_stats.mean)} p95 ${fmt(it.token_stats.p95)}` : '';
        html += `<tr><td class="num">${it.iteration}</td><td><span class="pill">${it.status}</span></td><td class="num">${it.rollout_rows}</td><td>${rewards}</td><td class="num">${toks}</td><td class="num">${esc(frontier)}</td><td class="num">${opd}</td><td class="num">${gate}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('iterations').innerHTML = html;
    }
    function renderFilters() {
      const select = document.getElementById('iterFilter');
      const old = select.value;
      select.innerHTML = '<option value="">latest</option>' + state.iterations.map(it => `<option>${it.iteration}</option>`).join('');
      if ([...select.options].some(o => o.value === old)) select.value = old;
    }
    function renderChartControls() {
      const metricSelect = document.getElementById('metricSelect');
      const oldMetric = metricSelect.value || 'mean_reward';
      const metricKeys = isMulti() ? Object.keys(rootState.comparison_series || {}) : Object.keys(state.metric_series || {});
      const preferred = ['mean_reward','task_reward/tool','task_reward/memory','task_reward/code','kept_frontier_rows','task_frontier_rows/tool','task_frontier_rows/memory','task_frontier_rows/code','opd_distill_rows','all_fail_rows','opd_task_rows/tool','opd_task_rows/memory','opd_task_rows/code','retention_rows','timing/bake_seconds','timing/collect_seconds','timing/update_seconds','optimizer_steps','updates','grad_norm_max','clip_frac_mean','approx_kl_mean','delta_from_init_mean_abs','delta_from_init_max_abs','old_logprob_coverage'];
      const orderedMetrics = [...preferred.filter(k => metricKeys.includes(k)), ...metricKeys.filter(k => !preferred.includes(k)).sort()];
      metricSelect.innerHTML = orderedMetrics.map(key => `<option value="${esc(key)}">${esc(metricLabels[key] || key)}</option>`).join('');
      metricSelect.value = orderedMetrics.includes(oldMetric) ? oldMetric : (orderedMetrics[0] || '');
      const coefSelect = document.getElementById('coefSelect');
      const oldCoef = coefSelect.value || 'coefficient/tool';
      const coefKeys = (isMulti() ? Object.keys(rootState.comparison_series || {}) : Object.keys(state.metric_series || {}))
        .filter(key => key.startsWith('coefficient/') || key.startsWith('delta_from_init/'));
      const coefExperts = expertsFromMetricKeys(coefKeys);
      const preferredCoef = [
        ...coefExperts.map(expert => `coefficient/${expert}`),
        ...coefExperts.map(expert => `delta_from_init/${expert}`),
      ];
      const orderedCoef = [
        ...preferredCoef.filter(key => coefKeys.includes(key)),
        ...coefKeys.filter(key => !preferredCoef.includes(key)).sort(),
      ];
      coefSelect.innerHTML = orderedCoef.map(key => `<option value="${esc(key)}">${esc(key.replace('coefficient/', 'coef ').replace('delta_from_init/', 'delta '))}</option>`).join('');
      coefSelect.value = orderedCoef.includes(oldCoef) ? oldCoef : (orderedCoef[0] || '');
    }
    function renderPromptRows() {
      const iterValue = document.getElementById('iterFilter').value;
      const taskValue = document.getElementById('taskFilter').value;
      const it = iterValue ? state.iterations.find(x => x.iteration === iterValue) : state.iterations[state.iterations.length - 1];
      if (!it) { document.getElementById('promptRows').textContent = 'no rows'; return; }
      let rows = it.prompt_rows || [];
      if (taskValue) rows = rows.filter(r => r.task === taskValue);
      let html = '<table><thead><tr><th>prompt</th><th>task</th><th>mean/std</th><th>success</th><th>tokens</th><th>samples</th><th>status</th></tr></thead><tbody>';
      for (const row of rows) {
        const bars = row.samples.map(s => `<span title="${esc(s.sample_id)} reward=${fmt(s.reward)}" class="bar ${s.reward > 0 ? 'pos' : (s.reward < 0 ? 'neg' : '')}" style="width:${Math.max(8, Math.min(70, 12 + Math.abs(s.reward) * 12))}px"></span>`).join('');
        html += `<tr><td class="path">${esc(row.prompt_id)}</td><td class="${taskClass(row.task)}">${row.task}</td><td class="num">${fmt(row.mean_reward)} / ${fmt(row.std_reward)}</td><td class="num">${row.success_count}/${row.samples.length}</td><td class="num">${fmt(row.token_mean)} max ${fmt(row.token_max)}</td><td><div class="samples">${bars}</div></td><td>${row.keep_for_policy_loss ? '<span class="pill">frontier</span>' : esc(row.skip_reason || '')}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('promptRows').innerHTML = html;
    }
    function drawMetricChart() {
      const metric = document.getElementById('metricSelect').value || 'mean_reward';
      document.getElementById('rewardChartTitle').textContent = isMulti() ? 'Strategy Metrics' : 'Run Metrics';
      if (isMulti()) drawComparisonSeries('rewardChart', rootState.comparison_series?.[metric] || {}, 'rewardLegend');
      else drawSingleMetricSeries('rewardChart', metric, 'rewardLegend');
    }
    function drawCoef() {
      const metric = document.getElementById('coefSelect').value || 'coefficient/tool';
      if (isMulti()) drawComparisonSeries('coefChart', rootState.comparison_series?.[metric] || {}, 'coefLegend', {autoZoom: true});
      else drawSingleMetricSeries('coefChart', metric, 'coefLegend', {autoZoom: true});
    }
    function drawPaperCharts() {
      if (!isMulti()) return;
      drawComparisonSeries('taskToolChart', rootState.comparison_series?.['task_reward/tool'] || {}, 'taskToolLegend');
      drawComparisonSeries('taskMemoryChart', rootState.comparison_series?.['task_reward/memory'] || {}, 'taskMemoryLegend');
      drawComparisonSeries('taskCodeChart', rootState.comparison_series?.['task_reward/code'] || {}, 'taskCodeLegend');
      drawComparisonSeries('frontierChart', rootState.comparison_series?.kept_frontier_rows || {}, 'frontierLegend');
      drawComparisonSeries('opdChart', rootState.comparison_series?.opd_distill_rows || rootState.comparison_series?.all_fail_rows || {}, 'opdLegend');
      drawComparisonSeries('updateTimeChart', rootState.comparison_series?.['timing/update_seconds'] || {}, 'updateTimeLegend');
    }
    function drawSingleMetricSeries(canvasId, metric, legendId, options = {}) {
      const raw = (state.metric_series || {})[metric] || [];
      const series = {};
      series[metricLabels[metric] || metric] = raw.map(p => ({iteration:p.iteration, value:p.value}));
      drawSeries(canvasId, series, 'value', legendId, options);
    }
    function drawComparisonSeries(canvasId, rawSeries, legendId, options = {}) {
      const series = {};
      for (const [runId, points] of Object.entries(rawSeries || {})) {
        series[runId] = points.map(p => ({iteration:p.iteration, value:p.value}));
      }
      drawSeries(canvasId, series, 'value', legendId, options);
    }
    function drawSeries(id, series, field, legendId, options = {}) {
      const canvas = document.getElementById(id), ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      const names = Object.keys(series);
      const labels = [...new Set(names.flatMap(t => series[t].map(p => p.iteration)))];
      let vals = names.flatMap(t => series[t].map(p => Number(p[field] || 0)));
      if (!vals.length) vals = [0,1];
      let min = Math.min(...vals), max = Math.max(...vals);
      if (options.autoZoom && max !== min) {
        const span = max - min;
        min -= span * 0.18;
        max += span * 0.18;
      } else if (min > 0 && min < 1) {
        min = 0;
      }
      if (max === min) { max += 1; min -= 1; }
      const padL = 60, padR = 18, padT = 18, padB = 42;
      const w = canvas.width - padL - padR, h = canvas.height - padT - padB;
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.strokeStyle = '#d8dee9'; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = padT + h * i / 4;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + w, y); ctx.stroke();
      }
      ctx.strokeStyle = '#7f8b9d'; ctx.strokeRect(padL, padT, w, h);
      ctx.fillStyle = '#687386'; ctx.font = '18px ui-monospace';
      for (let i = 0; i <= 4; i++) {
        const value = max - (max - min) * i / 4;
        const y = padT + h * i / 4 + 6;
        ctx.fillText(shortFmt(value), 6, y);
      }
      names.forEach((name, seriesIndex) => {
        const pts = series[name];
        const color = colors[name] || strategyColors[seriesIndex % strategyColors.length];
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3; ctx.beginPath();
        pts.forEach((p, i) => {
          const x = padL + (labels.indexOf(p.iteration) / Math.max(1, labels.length-1)) * w;
          const y = padT + h - ((Number(p[field] || 0) - min) / Math.max(1e-9, max - min)) * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        pts.forEach(p => {
          const x = padL + (labels.indexOf(p.iteration) / Math.max(1, labels.length-1)) * w;
          const y = padT + h - ((Number(p[field] || 0) - min) / Math.max(1e-9, max - min)) * h;
          ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill();
        });
      });
      labels.forEach((label, i) => {
        if (labels.length > 12 && i % Math.ceil(labels.length / 10) !== 0 && i !== labels.length - 1) return;
        const x = padL + (i / Math.max(1, labels.length-1)) * w;
        ctx.fillStyle = '#687386'; ctx.fillText(label.replace('iter_','i'), x-18, canvas.height-14);
      });
      if (legendId) {
        document.getElementById(legendId).innerHTML = names.map((name, i) => {
          const color = colors[name] || strategyColors[i % strategyColors.length];
          return `<span><i class="swatch" style="background:${color}"></i>${esc(shortRunName(name))}</span>`;
        }).join('');
      }
    }
    function fmt(x) { return Number(x || 0).toFixed(4); }
    function shortFmt(x) {
      const v = Number(x || 0);
      if (Math.abs(v) >= 1000) return v.toExponential(1);
      if (Math.abs(v) >= 10) return v.toFixed(1);
      return v.toFixed(3);
    }
    function shortRunName(name) {
      return String(name).split('/').pop().replace('paper96_', '').replace('_20260514_paper96_dynopd_i8', '').replace('_20260514_paper96_i8', '').replace('A_gc_opd_i8', 'A gc+fixedOPD').replace('B_gc_noopd_i8', 'B gc noOPD').replace('C_gp_opd_i8', 'C gp+fixedOPD').replace('D_gc_dynamic_opd_i8', 'D gc+dynamicOPD');
    }
    function evalCell(e) {
      return `T ${statusText(e.tool)}<br>M ${statusText(e.memory)}<br>C ${statusText(e.code)}`;
    }
    function statusText(item) { return item && item.status === 'ready' ? '<span class="pill">ready</span>' : '<span class="pill">pending</span>'; }
    function toolEval(item) {
      if (!item || item.status !== 'ready') return `<span class="pill">missing</span><div class="path">${esc((item || {}).path || '')}</div>`;
      return `<span class="num">mean ${fmt(item.mean)} live ${fmt(item.live_mean)}</span><br>${Object.entries(item.scores || {}).map(([k,v]) => `${esc(k)} <span class="num">${fmt(v)}</span>`).join('<br>')}<div class="path">${esc(item.path || '')}</div>`;
    }
    function memoryEval(item) {
      if (!item || item.status !== 'ready') return `<span class="pill">missing</span><div class="path">${esc((item || {}).path || '')}</div>`;
      return `<span class="num">F1 ${fmt(item.f1_mean)} EM ${fmt(item.em_mean)}</span><br>${Object.entries(item.datasets || {}).map(([k,v]) => `${esc(k)} F1 <span class="num">${fmt(v.f1)}</span> EM <span class="num">${fmt(v.em)}</span>`).join('<br>')}<div class="path">${esc(item.path || '')}</div>`;
    }
    function codeEval(item) {
      if (!item || item.status !== 'ready') return `<span class="pill">missing</span><div class="path">${esc((item || {}).path || '')}</div>`;
      return `<span class="num">ACC ${fmt(item.acc_mean)}</span><br>${Object.entries(item.datasets || {}).map(([k,v]) => `${esc(k)} acc <span class="num">${fmt(v.acc)}</span> tp <span class="num">${fmt(v.tp)}</span> bon <span class="num">${fmt(v.bon)}</span>`).join('<br>')}<div class="path">${esc(item.path || '')}</div>`;
    }
    function selectRun(runId) {
      selectedRunId = runId;
      localStorage.setItem('opvecMonitorRunId', selectedRunId);
      state = pickState(rootState);
      render();
      window.scrollTo({top: 0, behavior: 'smooth'});
    }
    function isMulti() { return !!(rootState && Array.isArray(rootState.runs)); }
    function sec(x) { if (x === null || x === undefined || x === '') return ''; const v = Number(x || 0); return v < 60 ? v.toFixed(1) + 's' : (v/60).toFixed(1) + 'm'; }
    function card(label, value, sub) { return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub || ''}</div></div>`; }
    function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    document.getElementById('iterFilter').addEventListener('change', renderPromptRows);
    document.getElementById('taskFilter').addEventListener('change', renderPromptRows);
    document.getElementById('maxRows').addEventListener('change', loadState);
    document.getElementById('metricSelect').addEventListener('change', drawMetricChart);
    document.getElementById('coefSelect').addEventListener('change', drawCoef);
    document.getElementById('runSelect').addEventListener('change', () => {
      selectedRunId = document.getElementById('runSelect').value;
      localStorage.setItem('opvecMonitorRunId', selectedRunId);
      state = pickState(rootState);
      render();
    });
    loadState();
    setInterval(loadState, 10000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Run dir or run_id=run_dir. Repeat for multi-run UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--init-value", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-prompt-rows", type=int, default=300)
    parser.add_argument(
        "--run-iteration-offset",
        action="append",
        default=[],
        help="Display-only iteration offset for a run_id, e.g. --run-iteration-offset r1_continue=10.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
