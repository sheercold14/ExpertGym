#!/usr/bin/env python3
"""Run Gated-GRPO with per-iteration baking and vLLM rollout generation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    gate_checkpoint = args.init_gate_checkpoint
    manifest = {
        "format": "opvec_gated_grpo_bake_vllm_loop_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "mode_manifest": args.mode_manifest,
        "run_dir": str(run_dir),
        "iterations": [],
        "notes": [
            "Each iteration bakes the current OP-VEC gates into a normal HF checkpoint.",
            "vLLM generates rollouts from the baked checkpoint; update recomputes missing old_logprob once under the initial gate policy before optimizing gates.",
        ],
    }
    loop_start = time.time()
    if args.start_iteration < 1:
        raise SystemExit("--start-iteration must be >= 1")
    end_iteration = args.start_iteration + args.num_iters - 1
    for iteration in range(args.start_iteration, end_iteration + 1):
        iter_start = time.time()
        iter_dir = run_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        baked_policy = iter_dir / "baked_policy"
        rollouts = iter_dir / "rollouts.jsonl"
        updates = iter_dir / "gate_updates.jsonl"

        bake_cmd = _bake_command(args, baked_policy=baked_policy, gate_checkpoint=gate_checkpoint)
        collect_cmd = _collect_command(
            args,
            baked_policy=baked_policy,
            rollouts=rollouts,
            gate_checkpoint=gate_checkpoint,
            iteration=iteration,
        )
        update_cmd = _update_command(args, rollouts=rollouts, updates=updates, gate_checkpoint=gate_checkpoint)

        timings = {}
        if args.dry_run:
            print("[dry-run]", _fmt_cmd(bake_cmd))
            print("[dry-run]", _fmt_cmd(collect_cmd))
            print("[dry-run]", _fmt_cmd(update_cmd))
        else:
            timings["bake_seconds"] = _run_timed("bake", bake_cmd)
            timings["collect_seconds"] = _run_timed("collect", collect_cmd)
            timings["update_seconds"] = _run_timed("update", update_cmd)

        next_gate = updates.with_suffix(".gates.json")
        summary_path = updates.with_suffix(".summary.json")
        iter_summary = {
            "iteration": iteration,
            "baked_policy": str(baked_policy),
            "rollouts": str(rollouts),
            "updates": str(updates),
            "input_gate_checkpoint": gate_checkpoint,
            "output_gate_checkpoint": str(next_gate),
            "bake_command": bake_cmd,
            "collect_command": collect_cmd,
            "update_command": update_cmd,
            "timings": {**timings, "iteration_seconds": time.time() - iter_start},
        }
        if summary_path.exists():
            try:
                iter_summary["update_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as error:
                iter_summary["update_summary_error"] = str(error)
        manifest["iterations"].append(iter_summary)
        gate_checkpoint = str(next_gate)
        _write_json(run_dir / "gated_grpo_bake_vllm_loop_manifest.json", manifest)

    manifest["elapsed_seconds"] = time.time() - loop_start
    _write_json(run_dir / "gated_grpo_bake_vllm_loop_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "final_gate_checkpoint": gate_checkpoint, "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False, indent=2))


def _bake_command(args: argparse.Namespace, *, baked_policy: Path, gate_checkpoint: str | None) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/opvec_bake_checkpoint.py"),
        "--config",
        args.config,
        "--mode-manifest",
        args.mode_manifest,
        "--output",
        str(baked_policy),
    ]
    if gate_checkpoint:
        cmd += ["--gate-checkpoint", gate_checkpoint]
    return cmd


def _collect_command(
    args: argparse.Namespace,
    *,
    baked_policy: Path,
    rollouts: Path,
    gate_checkpoint: str | None,
    iteration: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/train/opvec_collect_vllm_rollouts.py"),
        "--config",
        args.config,
        "--mode-manifest",
        args.mode_manifest,
        "--policy-model",
        str(baked_policy),
        "--output",
        str(rollouts),
        "--run-id",
        f"{args.run_id}-iter{iteration:03d}",
        "--num-prompts",
        str(args.num_prompts),
        "--samples-per-prompt",
        str(args.samples_per_prompt),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--vllm-batch-size",
        str(args.vllm_batch_size),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.torch_dtype,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed + iteration - 1),
        "--stream-output",
        "--progress-every",
        str(args.progress_every),
    ]
    if args.seed_manifest:
        cmd += ["--seed-manifest", args.seed_manifest]
    if args.tasks:
        cmd += ["--tasks", args.tasks]
    if args.memory_kind:
        cmd += ["--memory-kind", args.memory_kind]
    if args.prompt_id:
        cmd += ["--prompt-id", args.prompt_id]
    if args.use_manifest_order:
        cmd.append("--use-manifest-order")
    if args.greedy:
        cmd.append("--greedy")
    if gate_checkpoint:
        cmd += ["--gate-checkpoint", gate_checkpoint]
    if args.behavior_span_reward_weight is not None:
        cmd += ["--behavior-span-reward-weight", str(args.behavior_span_reward_weight)]
    return cmd


def _update_command(args: argparse.Namespace, *, rollouts: Path, updates: Path, gate_checkpoint: str | None) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/train/opvec_update_gates_from_rollouts.py"),
        "--config",
        args.config,
        "--mode-manifest",
        args.mode_manifest,
        "--rollouts",
        str(rollouts),
        "--output",
        str(updates),
        "--max-steps",
        str(args.update_epochs),
        "--max-logprob-tokens",
        str(args.max_logprob_tokens),
        "--fill-missing-old-logprob",
        "--lr",
        str(args.lr),
        "--prior-loss-weight",
        str(args.prior_loss_weight),
        "--ppo-loss-weight",
        str(args.ppo_loss_weight),
        "--best-response-loss-weight",
        str(args.best_response_loss_weight),
        "--pairwise-loss-weight",
        str(args.pairwise_loss_weight),
        "--pairwise-margin",
        str(args.pairwise_margin),
        "--max-pairwise-pairs-per-row",
        str(args.max_pairwise_pairs_per_row),
        "--min-grad-norm-for-step",
        str(args.min_grad_norm_for_step),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--gate-parameterization",
        args.gate_parameterization,
    ]
    if args.device_map:
        cmd += ["--device-map", args.device_map]
    for item in args.max_memory or []:
        cmd += ["--max-memory", item]
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.max_gated_modules is not None:
        cmd += ["--max-gated-modules", str(args.max_gated_modules)]
    if gate_checkpoint:
        cmd += ["--init-gate-checkpoint", gate_checkpoint]
    for item in args.task_weight or []:
        cmd += ["--task-weight", item]
    for item in args.frontier_task_quota or []:
        cmd += ["--frontier-task-quota", item]
    if args.max_frontier_rows_per_task is not None:
        cmd += ["--max-frontier-rows-per-task", str(args.max_frontier_rows_per_task)]
    if args.recompute_frontier:
        cmd.append("--recompute-frontier")
    if args.length_normalize_logprob:
        cmd.append("--length-normalize-logprob")
    if args.length_normalize_policy_logprob:
        cmd.append("--length-normalize-policy-logprob")
    if args.task_normalize_advantages:
        cmd.append("--task-normalize-advantages")
    if args.advantage_field:
        cmd += ["--advantage-field", args.advantage_field]
        if not args.advantage_field_apply_frontier_weight:
            cmd.append("--no-advantage-field-frontier-weight")
    for item in args.train_coefficient or []:
        cmd += ["--train-coefficient", item]
    if args.tool_min_margin_over_memory:
        cmd += ["--tool-min-margin-over-memory", str(args.tool_min_margin_over_memory)]
    if args.tool_min_margin_over_code:
        cmd += ["--tool-min-margin-over-code", str(args.tool_min_margin_over_code)]
    if args.positive_reward_threshold is not None:
        cmd += ["--positive-reward-threshold", str(args.positive_reward_threshold)]
    if args.max_coefficient_delta_from_init is not None:
        cmd += ["--max-coefficient-delta-from-init", str(args.max_coefficient_delta_from_init)]
    return cmd


def _run_timed(label: str, cmd: list[str]) -> float:
    print(f"[{label}]", _fmt_cmd(cmd), flush=True)
    start = time.time()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    elapsed = time.time() - start
    print(f"[{label}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def _fmt_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--seed-manifest", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", default="gated-grpo-bake-vllm")
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--start-iteration", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=48)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--memory-kind", default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--use-manifest-order", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-logprob-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--prior-loss-weight", type=float, default=0.02)
    parser.add_argument("--ppo-loss-weight", type=float, default=1.0)
    parser.add_argument("--best-response-loss-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--max-pairwise-pairs-per-row", type=int, default=0)
    parser.add_argument("--min-grad-norm-for-step", type=float, default=0.0)
    parser.add_argument("--max-coefficient-delta-from-init", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--gate-parameterization", choices=["global", "layer-band", "parameter", "global-parameter"], default="global")
    parser.add_argument("--init-gate-checkpoint", default=None)
    parser.add_argument("--max-gated-modules", type=int, default=None)
    parser.add_argument("--task-weight", action="append", default=[])
    parser.add_argument("--frontier-task-quota", action="append", default=[])
    parser.add_argument("--max-frontier-rows-per-task", type=int, default=None)
    parser.add_argument("--recompute-frontier", action="store_true")
    parser.add_argument("--length-normalize-logprob", action="store_true")
    parser.add_argument("--length-normalize-policy-logprob", action="store_true")
    parser.add_argument("--task-normalize-advantages", action="store_true")
    parser.add_argument("--advantage-field", default=None)
    parser.add_argument("--no-advantage-field-frontier-weight", dest="advantage_field_apply_frontier_weight", action="store_false")
    parser.set_defaults(advantage_field_apply_frontier_weight=True)
    parser.add_argument("--train-coefficient", action="append", default=[])
    parser.add_argument("--tool-min-margin-over-memory", type=float, default=0.0)
    parser.add_argument("--tool-min-margin-over-code", type=float, default=0.0)
    parser.add_argument("--positive-reward-threshold", type=float, default=None)
    parser.add_argument("--behavior-span-reward-weight", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
