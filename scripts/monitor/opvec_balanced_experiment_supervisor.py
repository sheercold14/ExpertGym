#!/usr/bin/env python3
"""Supervise OP-VEC balanced matrix runs and launch eval6 for proxy-best models."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.summarize_rollouts import summarize_rollouts


TASKS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    runs = [_parse_run(item) for item in args.run]
    sessions = dict(_parse_session(item) for item in args.session)
    status_path = Path(args.status_json)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_md) if args.report_md else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        payload = _status_payload(args=args, runs=runs, sessions=sessions)
        _write_json(status_path, payload)
        if report_path:
            report_path.write_text(_render_report(payload), encoding="utf-8")

        if args.stop_on_collapse:
            _stop_collapsed_runs(payload, sessions=sessions)

        terminal = all(run["state"] in {"completed", "stopped", "failed"} for run in payload["runs"])
        if terminal:
            if args.eval_when_terminal:
                _launch_eval_once(payload, args=args)
            break
        time.sleep(float(args.poll_seconds))


def _status_payload(*, args: argparse.Namespace, runs: list[dict[str, Any]], sessions: dict[str, str]) -> dict[str, Any]:
    run_items = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        metrics = _iteration_metrics(run_dir)
        state = _run_state(run_dir, expected_iters=int(args.expected_iters), session=sessions.get(run["label"]))
        collapse = _collapse_signal(metrics, args=args)
        best = _best_validated_iteration(metrics)
        run_items.append(
            {
                "label": run["label"],
                "run_dir": str(run_dir),
                "session": sessions.get(run["label"]),
                "state": state,
                "latest_iteration": metrics[-1]["iteration"] if metrics else None,
                "metrics": metrics,
                "collapse": collapse,
                "best_validated": best,
            }
        )
    return {
        "format": "opvec_balanced_experiment_supervisor_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_iters": int(args.expected_iters),
        "runs": run_items,
        "eval": {
            "enabled": bool(args.eval_when_terminal),
            "eval_gpus": args.eval_gpus,
            "eval_runner": args.eval_runner,
            "eval_session": args.eval_session,
            "work_root": args.eval_work_root,
        },
    }


def _iteration_metrics(run_dir: Path) -> list[dict[str, Any]]:
    output = []
    for iter_dir in sorted(run_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("_", 1)[1])
        except Exception:
            continue
        rollouts = iter_dir / "rollouts.jsonl"
        if not rollouts.exists():
            continue
        try:
            rollout_summary = summarize_rollouts([str(rollouts)])
        except Exception as error:
            output.append({"iteration": iteration, "error": f"summarize_rollouts: {error}"})
            continue
        task_stats = rollout_summary.get("task_stats") or {}
        rewards = {
            task: _float((task_stats.get(task) or {}).get("mean_reward"))
            for task in TASKS
            if task in task_stats
        }
        successes = {
            task: _float((task_stats.get(task) or {}).get("success_rate"))
            for task in TASKS
            if task in task_stats
        }
        overall = mean(rewards.values()) if rewards else 0.0
        update_summary_path = iter_dir / "gate_updates.summary.json"
        gate_path = iter_dir / "gate_updates.gates.json"
        update_summary = _read_json(update_summary_path) if update_summary_path.exists() else {}
        output.append(
            {
                "iteration": iteration,
                "baked_policy": str(iter_dir / "baked_policy"),
                "rollouts": str(rollouts),
                "gate_checkpoint": str(gate_path) if gate_path.exists() else None,
                "has_update": gate_path.exists(),
                "overall_reward": overall,
                "task_reward": rewards,
                "task_success": successes,
                "kept_frontiers": update_summary.get("kept_frontier_rows"),
                "frontier_task_counts": update_summary.get("frontier_task_counts"),
                "retention_rows": update_summary.get("retention_rows"),
                "opd_distill_rows": update_summary.get("opd_distill_rows"),
                "grad_norm_max": _epoch_max(update_summary, "grad_norm_max"),
                "gate_delta_max": _epoch_max(update_summary, "gate_delta_max"),
            }
        )
    return output


def _run_state(run_dir: Path, *, expected_iters: int, session: str | None) -> str:
    final_gate = run_dir / f"iter_{expected_iters:03d}" / "gate_updates.gates.json"
    if final_gate.exists():
        return "completed"
    if session and not _tmux_session_exists(session):
        if any(run_dir.glob("iter_*/run.failed")):
            return "failed"
        return "stopped"
    if (run_dir / "run.log").exists():
        return "running"
    return "pending"


def _collapse_signal(metrics: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    usable = [m for m in metrics if "overall_reward" in m]
    if not usable:
        return {"triggered": False}
    latest = usable[-1]
    iteration = int(latest["iteration"])
    if iteration < int(args.collapse_start_iter) or iteration > int(args.collapse_end_iter):
        return {"triggered": False}
    previous = [m for m in usable[:-1] if int(m["iteration"]) >= 2]
    if not previous:
        return {"triggered": False}
    peak = max(float(m["overall_reward"]) for m in previous)
    latest_overall = float(latest["overall_reward"])
    tool_rewards = [float(m.get("task_reward", {}).get("tool", 0.0)) for m in usable if "tool" in m.get("task_reward", {})]
    tool_peak = max(tool_rewards[:-1]) if len(tool_rewards) > 1 else tool_rewards[0] if tool_rewards else 0.0
    latest_tool = float(latest.get("task_reward", {}).get("tool", 0.0))
    reasons = []
    if peak - latest_overall >= float(args.collapse_overall_drop):
        reasons.append({"metric": "overall_reward", "latest": latest_overall, "peak": peak})
    if tool_peak - latest_tool >= float(args.collapse_tool_drop) and latest_tool < float(args.min_tool_reward):
        reasons.append({"metric": "tool_reward", "latest": latest_tool, "peak": tool_peak})
    return {"triggered": bool(reasons), "iteration": iteration, "reasons": reasons}


def _best_validated_iteration(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [m for m in metrics if Path(str(m.get("baked_policy", ""))).exists() and "overall_reward" in m]
    if not candidates:
        return None
    best = max(candidates, key=lambda m: (_balanced_score(m), float(m["overall_reward"])))
    return {
        "iteration": int(best["iteration"]),
        "baked_policy": best["baked_policy"],
        "overall_reward": float(best["overall_reward"]),
        "task_reward": best.get("task_reward", {}),
        "balanced_score": _balanced_score(best),
    }


def _balanced_score(metric: dict[str, Any]) -> float:
    rewards = [float(metric.get("task_reward", {}).get(task, 0.0)) for task in TASKS]
    if not rewards:
        return 0.0
    return mean(rewards) - 0.25 * (max(rewards) - min(rewards))


def _stop_collapsed_runs(payload: dict[str, Any], *, sessions: dict[str, str]) -> None:
    for run in payload["runs"]:
        if run["state"] != "running":
            continue
        collapse = run.get("collapse") or {}
        if not collapse.get("triggered"):
            continue
        session = sessions.get(run["label"])
        if session and _tmux_session_exists(session):
            subprocess.run(["tmux", "kill-session", "-t", session], check=False)
            marker = Path(run["run_dir"]) / "stopped_by_supervisor.json"
            _write_json(
                marker,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "collapse_signal",
                    "collapse": collapse,
                },
            )


def _launch_eval_once(payload: dict[str, Any], *, args: argparse.Namespace) -> None:
    eval_runner = Path(args.eval_runner)
    launched_marker = eval_runner.with_suffix(".launched.json")
    if launched_marker.exists():
        return
    selected = []
    gpus = [gpu.strip() for gpu in args.eval_gpus.split(",") if gpu.strip()]
    for idx, run in enumerate(payload["runs"]):
        best = run.get("best_validated")
        if not best:
            continue
        if idx >= len(gpus):
            break
        selected.append(
            {
                "name": f"opvec-bal15-{_sanitize(run['label'])}-bestiter{int(best['iteration']):02d}",
                "path": best["baked_policy"],
                "gpu": gpus[idx],
                "port": int(args.eval_port_base) + idx,
            }
        )
    if not selected:
        return
    _write_eval_runner(eval_runner, selected=selected, work_root=args.eval_work_root)
    eval_runner.chmod(0o755)
    cmd = (
        f"cd {shlex.quote(args.eval_repo_root)} && "
        f"{shlex.quote(args.python)} {shlex.quote(str(eval_runner))} "
        f"2>&1 | tee {shlex.quote(str(eval_runner.with_suffix('.log')))}"
    )
    if not _tmux_session_exists(args.eval_session):
        subprocess.run(["tmux", "new-session", "-d", "-s", args.eval_session, cmd], check=True)
    _write_json(
        launched_marker,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected": selected,
            "command": cmd,
            "session": args.eval_session,
        },
    )


def _write_eval_runner(path: Path, *, selected: list[dict[str, Any]], work_root: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = [item["name"] for item in selected]
    text = f'''#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).with_name("run_eval6.py")
SPEC = importlib.util.spec_from_file_location("eval6_runner", BASE)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

MODELS = {json.dumps(selected, ensure_ascii=False, indent=4)}

runner.MODELS = MODELS
runner.WORK_ROOT = Path({work_root!r})


def main() -> None:
    runner.BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    runner.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    runner.preflight()
    runner.prepare_bfcl_workdirs()
    runner.run_parallel("bfcl", runner.run_bfcl, MODELS)
    runner.run_parallel("memory-infer", runner.run_memory_inference_only, MODELS)
    runner.score_memory_all()
    runner.run_parallel("cure", runner.run_cure, MODELS)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("append_eval6_models.py")), *{json.dumps(model_names, ensure_ascii=False)}],
        check=False,
    )


if __name__ == "__main__":
    main()
'''
    path.write_text(text, encoding="utf-8")


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# OP-VEC Balanced W3 Retention 0.5 Supervisor",
        "",
        f"- updated_at: `{payload['created_at']}`",
        f"- expected_iters: `{payload['expected_iters']}`",
        "",
        "| run | state | latest | overall | tool | memory | code | best | collapse |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        latest = (run.get("metrics") or [{}])[-1] if run.get("metrics") else {}
        task = latest.get("task_reward") or {}
        best = run.get("best_validated") or {}
        collapse = run.get("collapse") or {}
        lines.append(
            "| {label} | {state} | {latest} | {overall:.4f} | {tool:.4f} | {memory:.4f} | {code:.4f} | {best_iter} {best_score:.4f} | {collapse} |".format(
                label=run["label"],
                state=run["state"],
                latest=latest.get("iteration", ""),
                overall=float(latest.get("overall_reward", 0.0) or 0.0),
                tool=float(task.get("tool", 0.0) or 0.0),
                memory=float(task.get("memory", 0.0) or 0.0),
                code=float(task.get("code", 0.0) or 0.0),
                best_iter=best.get("iteration", ""),
                best_score=float(best.get("balanced_score", 0.0) or 0.0),
                collapse="yes" if collapse.get("triggered") else "",
            )
        )
    return "\n".join(lines) + "\n"


def _parse_run(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise SystemExit(f"--run must be label=/path, got {raw!r}")
    label, path = raw.split("=", 1)
    return {"label": label, "run_dir": path}


def _parse_session(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"--session must be label=tmux_session, got {raw!r}")
    label, session = raw.split("=", 1)
    return label, session


def _tmux_session_exists(session: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def _epoch_max(summary: dict[str, Any], key: str) -> float | None:
    values = []
    for row in summary.get("epoch_summaries") or []:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return max(values) if values else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if math.isfinite(output) else 0.0


def _sanitize(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=/path/to/run_dir")
    parser.add_argument("--session", action="append", default=[], help="label=tmux_session")
    parser.add_argument("--expected-iters", type=int, default=15)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--collapse-start-iter", type=int, default=5)
    parser.add_argument("--collapse-end-iter", type=int, default=8)
    parser.add_argument("--collapse-overall-drop", type=float, default=0.10)
    parser.add_argument("--collapse-tool-drop", type=float, default=0.30)
    parser.add_argument("--min-tool-reward", type=float, default=0.55)
    parser.add_argument("--stop-on-collapse", action="store_true")
    parser.add_argument("--eval-when-terminal", action="store_true")
    parser.add_argument("--eval-gpus", default="2,3,4,5")
    parser.add_argument("--eval-port-base", type=int, default=8110)
    parser.add_argument("--eval-session", default="eval_bal15_balanced_w3_ret05_i15_v1")
    parser.add_argument(
        "--eval-runner",
        default="/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_bal15_balanced_w3_ret05_i15_v1_addons.py",
    )
    parser.add_argument(
        "--eval-work-root",
        default="/tmp/shared-storage/AgentMerging_plan/evaluation_workdirs/eval6-20260502-125748-bal15-balanced-w3-ret05-i15-v1",
    )
    parser.add_argument("--eval-repo-root", default="/mnt/cache/wuruixiao/users/lsc/AgentMerging")
    parser.add_argument("--python", default="/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python")
    parser.add_argument(
        "--status-json",
        default="/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_bal15_balanced_w3_ret05_i15_v1_supervisor.json",
    )
    parser.add_argument(
        "--report-md",
        default="/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_bal15_balanced_w3_ret05_i15_v1_supervisor.md",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
