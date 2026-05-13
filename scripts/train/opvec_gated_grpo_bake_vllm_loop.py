#!/usr/bin/env python3
"""Run Gated-GRPO with per-iteration baking and vLLM rollout generation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = parse_args()
    args.rollout_shards = _resolve_rollout_shards(args)
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
        shard_specs = (
            _collect_shard_specs(
                args,
                baked_policy=baked_policy,
                rollouts=rollouts,
                gate_checkpoint=gate_checkpoint,
                iteration=iteration,
            )
            if args.rollout_shards > 1
            else []
        )
        update_cmd = _update_command(
            args,
            rollouts=rollouts,
            updates=updates,
            gate_checkpoint=gate_checkpoint,
            iteration=iteration,
        )

        timings = {}
        if args.dry_run:
            print("[dry-run]", _fmt_cmd(bake_cmd))
            if shard_specs:
                for spec in shard_specs:
                    print(
                        f"[dry-run][collect-shard {spec['shard_index']:02d}] "
                        f"CUDA_VISIBLE_DEVICES={spec['cuda_visible_devices']} {_fmt_cmd(spec['cmd'])}"
                    )
                print("[dry-run][merge]", " ".join(str(spec["output"]) for spec in shard_specs), "->", rollouts)
            else:
                print("[dry-run]", _fmt_cmd(collect_cmd))
            print("[dry-run]", _fmt_cmd(update_cmd))
        else:
            timings["bake_seconds"] = _run_timed("bake", bake_cmd)
            if args.post_bake_sleep_seconds > 0:
                print(f"[bake] sleeping {args.post_bake_sleep_seconds:.1f}s before vLLM collect", flush=True)
                time.sleep(float(args.post_bake_sleep_seconds))
            if shard_specs:
                timings["collect_seconds"] = _run_sharded_collect(args, shard_specs, merged_rollouts=rollouts)
            else:
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
            "collect_commands": [spec["cmd"] for spec in shard_specs] if shard_specs else [collect_cmd],
            "rollout_shards": _compact_shard_specs(shard_specs),
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
    num_prompts: int | None = None,
    prompt_offset: int = 0,
    run_id_suffix: str = "",
    seed: int | None = None,
) -> list[str]:
    collect_seed = int(seed if seed is not None else args.seed + iteration - 1)
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
        f"{args.run_id}-iter{iteration:03d}{run_id_suffix}",
        "--num-prompts",
        str(num_prompts if num_prompts is not None else args.num_prompts),
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
        str(collect_seed),
        "--stream-output",
        "--progress-every",
        str(args.progress_every),
    ]
    if args.tool_max_new_tokens is not None:
        cmd += ["--tool-max-new-tokens", str(args.tool_max_new_tokens)]
    if args.code_max_new_tokens is not None:
        cmd += ["--code-max-new-tokens", str(args.code_max_new_tokens)]
    if args.memory_update_max_new_tokens is not None:
        cmd += ["--memory-update-max-new-tokens", str(args.memory_update_max_new_tokens)]
    if args.memory_final_max_new_tokens is not None:
        cmd += ["--memory-final-max-new-tokens", str(args.memory_final_max_new_tokens)]
    if prompt_offset:
        cmd += ["--prompt-offset", str(prompt_offset)]
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
    if args.store_token_logprobs:
        cmd.append("--store-token-logprobs")
    if gate_checkpoint:
        cmd += ["--gate-checkpoint", gate_checkpoint]
    if args.behavior_span_reward_weight is not None:
        cmd += ["--behavior-span-reward-weight", str(args.behavior_span_reward_weight)]
    return cmd


def _collect_shard_specs(
    args: argparse.Namespace,
    *,
    baked_policy: Path,
    rollouts: Path,
    gate_checkpoint: str | None,
    iteration: int,
) -> list[dict[str, Any]]:
    if not args.use_manifest_order:
        raise SystemExit("--rollout-shards > 1 requires --use-manifest-order to avoid overlapping sampled prompts")
    if args.prompt_id:
        raise SystemExit("--rollout-shards > 1 is not supported together with --prompt-id")
    if int(args.tensor_parallel_size) != 1:
        raise SystemExit("--rollout-shards > 1 expects --tensor-parallel-size 1; use one vLLM process per GPU")
    ranges = _shard_ranges(int(args.num_prompts), int(args.rollout_shards))
    gpus = _rollout_gpus(args)
    if len(gpus) < len(ranges):
        raise SystemExit(f"--rollout-shards={len(ranges)} needs at least {len(ranges)} rollout GPUs, got {gpus}")
    specs = []
    for shard_index, (offset, count) in enumerate(ranges):
        shard_output = rollouts.with_name(f"{rollouts.stem}.shard_{shard_index:02d}{rollouts.suffix}")
        specs.append(
            {
                "shard_index": shard_index,
                "prompt_offset": offset,
                "num_prompts": count,
                "cuda_visible_devices": gpus[shard_index],
                "output": shard_output,
                "cmd": _collect_command(
                    args,
                    baked_policy=baked_policy,
                    rollouts=shard_output,
                    gate_checkpoint=gate_checkpoint,
                    iteration=iteration,
                    num_prompts=count,
                    prompt_offset=offset,
                    run_id_suffix=f"-shard{shard_index:02d}",
                    seed=int(args.seed + iteration - 1 + shard_index * 1009),
                ),
            }
        )
    return specs


def _resolve_rollout_shards(args: argparse.Namespace) -> int:
    raw = str(args.rollout_shards).strip().lower()
    if raw in {"auto", "gpu", "gpus"}:
        gpus = _rollout_gpus(args)
        total_samples = int(args.num_prompts) * int(args.samples_per_prompt)
        # The current collector keeps all samples for one prompt in the same
        # process, so prompt count is the hard upper bound for safe sharding.
        return max(1, min(len(gpus), int(args.num_prompts), max(1, total_samples)))
    try:
        value = int(raw)
    except ValueError as error:
        raise SystemExit("--rollout-shards must be a positive integer or 'auto'") from error
    if value < 1:
        raise SystemExit("--rollout-shards must be >= 1")
    return max(1, min(value, int(args.num_prompts)))


def _shard_ranges(total: int, shards: int) -> list[tuple[int, int]]:
    if total < 1:
        raise SystemExit("--num-prompts must be >= 1")
    shard_count = max(1, min(int(shards), int(total)))
    base = total // shard_count
    extra = total % shard_count
    ranges = []
    offset = 0
    for shard_index in range(shard_count):
        count = base + (1 if shard_index < extra else 0)
        ranges.append((offset, count))
        offset += count
    return ranges


def _rollout_gpus(args: argparse.Namespace) -> list[str]:
    raw = args.rollout_gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("No rollout GPUs found; set --rollout-gpus or CUDA_VISIBLE_DEVICES")
    return gpus


def _update_command(
    args: argparse.Namespace,
    *,
    rollouts: Path,
    updates: Path,
    gate_checkpoint: str | None,
    iteration: int,
) -> list[str]:
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
        "--update-batch-size",
        str(args.update_batch_size),
        "--batch-loss-reduction",
        args.batch_loss_reduction,
        "--loss-granularity",
        args.loss_granularity,
        "--frontier-order",
        args.frontier_order,
        "--frontier-shuffle-seed",
        str(args.frontier_shuffle_seed if args.frontier_shuffle_seed is not None else int(args.seed + iteration - 1)),
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
    if args.use_retention:
        cmd.append("--use-retention")
    if args.max_retention_rows is not None:
        cmd += ["--max-retention-rows", str(args.max_retention_rows)]
    if args.retention_loss_weight is not None:
        cmd += ["--retention-loss-weight", str(args.retention_loss_weight)]
    if args.recompute_frontier:
        cmd.append("--recompute-frontier")
    if args.length_normalize_logprob:
        cmd.append("--length-normalize-logprob")
    if args.length_normalize_policy_logprob:
        cmd.append("--length-normalize-policy-logprob")
    if args.task_normalize_advantages:
        cmd.append("--task-normalize-advantages")
    cmd += ["--advantage-normalization", args.advantage_normalization]
    if args.use_frontier_weight:
        cmd.append("--use-frontier-weight")
    if args.advantage_field:
        cmd += ["--advantage-field", args.advantage_field]
        if args.advantage_field_apply_frontier_weight:
            cmd.append("--advantage-field-frontier-weight")
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


def _run_sharded_collect(args: argparse.Namespace, shard_specs: list[dict[str, Any]], *, merged_rollouts: Path) -> float:
    print(f"[collect] launching {len(shard_specs)} vLLM rollout shards -> {merged_rollouts}", flush=True)
    start = time.time()
    procs: list[tuple[dict[str, Any], subprocess.Popen]] = []
    try:
        for spec in shard_specs:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(spec["cuda_visible_devices"])
            env.setdefault("PYTHONUNBUFFERED", "1")
            print(
                f"[collect-shard {spec['shard_index']:02d}] "
                f"gpu={spec['cuda_visible_devices']} offset={spec['prompt_offset']} "
                f"num_prompts={spec['num_prompts']} output={spec['output']}",
                flush=True,
            )
            print(f"[collect-shard {spec['shard_index']:02d}] {_fmt_cmd(spec['cmd'])}", flush=True)
            procs.append((spec, subprocess.Popen(spec["cmd"], cwd=str(REPO_ROOT), env=env)))
            if args.rollout_shard_stagger_seconds > 0:
                time.sleep(float(args.rollout_shard_stagger_seconds))

        failures = []
        for spec, proc in procs:
            returncode = proc.wait()
            if returncode != 0:
                failures.append((spec, returncode))
        if failures:
            for spec, returncode in failures:
                print(f"[collect-shard {spec['shard_index']:02d}] failed with returncode={returncode}", flush=True)
            raise subprocess.CalledProcessError(failures[0][1], failures[0][0]["cmd"])
    except Exception:
        for _, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        raise

    elapsed = time.time() - start
    _merge_shard_rollouts(shard_specs, merged_rollouts=merged_rollouts, elapsed_seconds=elapsed)
    print(f"[collect] sharded completed in {elapsed:.1f}s", flush=True)
    return elapsed


def _merge_shard_rollouts(
    shard_specs: list[dict[str, Any]],
    *,
    merged_rollouts: Path,
    elapsed_seconds: float,
) -> None:
    merged_rollouts.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = merged_rollouts.with_name(f"{merged_rollouts.name}.tmp")
    prompt_ids = set()
    task_counts: dict[str, int] = {}
    rows = 0
    kept = 0
    token_logprob_samples = 0
    shard_summaries = []
    with tmp_path.open("w", encoding="utf-8") as output:
        for spec in shard_specs:
            shard_path = Path(spec["output"])
            if not shard_path.exists():
                raise FileNotFoundError(f"missing shard rollout: {shard_path}")
            shard_summary = _read_json(shard_path.with_suffix(".summary.json")) or {}
            shard_summaries.append(shard_summary)
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    prompt_id = str(row.get("prompt_id"))
                    if prompt_id in prompt_ids:
                        raise ValueError(f"duplicate prompt_id across rollout shards: {prompt_id}")
                    prompt_ids.add(prompt_id)
                    task = str(row.get("task") or "unknown")
                    task_counts[task] = task_counts.get(task, 0) + 1
                    kept += int(bool(row.get("keep_for_policy_loss")))
                    rows += 1
                    for sample in row.get("samples", []):
                        if isinstance(sample, dict) and all(
                            key in sample for key in ("response_token_ids", "old_logprobs", "response_mask")
                        ):
                            token_logprob_samples += 1
                    output.write(line + "\n")
    tmp_path.replace(merged_rollouts)

    base_summary = next((item for item in shard_summaries if item), {})
    summary = {
        **base_summary,
        "format": "opvec_vllm_gated_grpo_rollout_sharded_v1",
        "output": str(merged_rollouts),
        "rows": rows,
        "kept_frontiers": kept,
        "tasks": dict(sorted(task_counts.items())),
        "token_logprob_samples": token_logprob_samples,
        "elapsed_seconds": float(elapsed_seconds),
        "rollout_shards": _compact_shard_specs(shard_specs),
        "shard_summaries": shard_summaries,
        "merge": {
            "strategy": "contiguous_prompt_offset",
            "update_reads": str(merged_rollouts),
            "duplicate_prompt_ids": False,
        },
    }
    _write_json(merged_rollouts.with_suffix(".summary.json"), summary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _compact_shard_specs(shard_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "shard_index": int(spec["shard_index"]),
            "cuda_visible_devices": str(spec["cuda_visible_devices"]),
            "prompt_offset": int(spec["prompt_offset"]),
            "num_prompts": int(spec["num_prompts"]),
            "output": str(spec["output"]),
        }
        for spec in shard_specs
    ]


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
    parser.add_argument("--tool-max-new-tokens", type=int, default=None)
    parser.add_argument("--code-max-new-tokens", type=int, default=None)
    parser.add_argument("--memory-update-max-new-tokens", type=int, default=None)
    parser.add_argument("--memory-final-max-new-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-logprob-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--rollout-shards",
        default="1",
        help="Number of independent single-GPU vLLM collector processes, or 'auto' to use available rollout GPUs.",
    )
    parser.add_argument(
        "--rollout-gpus",
        default=None,
        help="Comma-separated GPU ids assigned to rollout shards. Defaults to CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--rollout-shard-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--store-token-logprobs", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--update-batch-size", type=int, default=1)
    parser.add_argument("--batch-loss-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--loss-granularity", choices=["sequence", "token"], default="sequence")
    parser.add_argument("--frontier-order", choices=["as-is", "shuffle", "task-interleaved"], default="as-is")
    parser.add_argument("--frontier-shuffle-seed", type=int, default=None)
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
    parser.add_argument("--use-retention", action="store_true")
    parser.add_argument("--max-retention-rows", type=int, default=None)
    parser.add_argument("--retention-loss-weight", type=float, default=None)
    parser.add_argument("--recompute-frontier", action="store_true")
    parser.add_argument("--length-normalize-logprob", action="store_true")
    parser.add_argument("--length-normalize-policy-logprob", action="store_true")
    parser.add_argument("--task-normalize-advantages", action="store_true")
    parser.add_argument("--advantage-normalization", choices=["centered", "zscore"], default="centered")
    parser.add_argument("--use-frontier-weight", action="store_true")
    parser.add_argument("--advantage-field", default=None)
    parser.add_argument("--advantage-field-frontier-weight", dest="advantage_field_apply_frontier_weight", action="store_true")
    parser.add_argument("--no-advantage-field-frontier-weight", dest="advantage_field_apply_frontier_weight", action="store_false")
    parser.set_defaults(advantage_field_apply_frontier_weight=False)
    parser.add_argument("--train-coefficient", action="append", default=[])
    parser.add_argument("--tool-min-margin-over-memory", type=float, default=0.0)
    parser.add_argument("--tool-min-margin-over-code", type=float, default=0.0)
    parser.add_argument("--positive-reward-threshold", type=float, default=None)
    parser.add_argument("--behavior-span-reward-weight", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--post-bake-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
