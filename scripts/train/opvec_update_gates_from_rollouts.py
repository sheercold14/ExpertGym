#!/usr/bin/env python3
"""Update OP-VEC gates from collected rollout JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config, write_json
from opvec.data.io import read_jsonl
from opvec.data.replay_buffer import QUEUE_FRONTIER, QUEUE_RETENTION, classify_rollout_row
from opvec.modeling.apply_gates import install_gated_linears_from_manifest
from opvec.modeling.bake import load_gate_values
from opvec.modeling.devices import model_input_device, model_load_device_kwargs
from opvec.modeling.gate_parameters import make_torch_gate_manager
from opvec.modeling.manifest import manifest_param_names
from opvec.modeling.logprob import response_logprob_tensor_from_text
from opvec.train.frontier import group_relative_advantages, policy_frontier_weight, should_keep_frontier
from opvec.train.gated_grpo import gate_initialization_prior


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = update_gates(config, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def update_gates(config: dict, args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.init_gate_checkpoint:
        config = {**config, "initial_gates": load_gate_values(args.init_gate_checkpoint)}
    rows = []
    retention_rows = []
    for replay_path in args.replay_buffer or []:
        payload = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        queues = payload.get("queues", {})
        rows.extend(queues.get(QUEUE_FRONTIER, []))
        if args.use_retention:
            retention_rows.extend(queues.get(QUEUE_RETENTION, []))
    for replay_path in args.retention_only_replay_buffer or []:
        payload = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        queues = payload.get("queues", {})
        retention_rows.extend(queues.get(QUEUE_FRONTIER, []))
        retention_rows.extend(queues.get(QUEUE_RETENTION, []))
    raw_rows = []
    for rollout_path in args.rollouts or []:
        raw_rows.extend(read_jsonl(rollout_path))
    for row in raw_rows:
        if args.recompute_frontier:
            queue, row = classify_rollout_row(
                row,
                min_frontier_weight=float(config["frontier"]["min_frontier_weight"]),
                min_reward_std=float(config["frontier"]["min_reward_std"]),
            )
            if args.use_retention and queue == QUEUE_RETENTION:
                retention_rows.append(row)
        elif args.use_retention and _is_retention_candidate(row):
            retention_rows.append(row)
        if row.get("keep_for_policy_loss"):
            rows.append(row)
    raw_frontier_task_counts = _task_counts(rows)
    rows = _limit_frontier_rows(
        rows,
        task_quota=_merged_task_quota(config, args.frontier_task_quota),
        max_per_task=(
            args.max_frontier_rows_per_task
            if args.max_frontier_rows_per_task is not None
            else config.get("calibration", {}).get("max_frontier_rows_per_task")
        ),
    )
    frontier_task_counts = _task_counts(rows)
    if args.max_retention_rows is not None:
        retention_rows = retention_rows[: args.max_retention_rows]
    if not rows:
        raise SystemExit("No kept frontier rows found in rollout/replay-buffer inputs")
    device = args.device
    dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["base"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["base"],
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **model_load_device_kwargs(device_map=args.device_map, max_memory=args.max_memory),
    )
    if not args.device_map:
        model.to(device)
    device = model_input_device(model, torch, device)
    model.train()
    if args.gradient_checkpointing:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    for param in model.parameters():
        param.requires_grad_(False)
    gate_parameterization = _normalize_gate_parameterization(args.gate_parameterization)
    param_names = manifest_param_names(args.mode_manifest) if _uses_parameter_names(gate_parameterization) else None
    gate_manager = make_torch_gate_manager(
        torch,
        config,
        parameterization=gate_parameterization,
        param_names=param_names,
    ).to(device)
    installed = install_gated_linears_from_manifest(
        torch,
        model,
        mode_manifest_path=args.mode_manifest,
        gate_manager=gate_manager,
        max_modules=None if args.max_gated_modules in (None, 0) else args.max_gated_modules,
        device=None if args.device_map else device,
    )
    filled_old_logprobs = 0
    if args.fill_missing_old_logprob:
        filled_old_logprobs += _fill_missing_old_logprobs(
            torch,
            model,
            tokenizer,
            rows,
            device=device,
            max_logprob_tokens=args.max_logprob_tokens,
        )
        if args.use_retention and retention_rows:
            filled_old_logprobs += _fill_missing_old_logprobs(
                torch,
                model,
                tokenizer,
                retention_rows,
                device=device,
                max_logprob_tokens=args.max_logprob_tokens,
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    weight_decay = float(
        args.weight_decay if args.weight_decay is not None else config["optimizer"].get("weight_decay", 0.0)
    )
    optimizer = torch.optim.AdamW(gate_manager.parameters(), lr=float(args.lr or config["optimizer"]["lr"]), weight_decay=weight_decay)
    retention_weight = float(args.retention_loss_weight if args.retention_loss_weight is not None else config["loss"].get("lambda_retention", 0.0))
    prior_weight = float(args.prior_loss_weight if args.prior_loss_weight is not None else config["loss"].get("lambda_prior", 0.0))
    task_weights = _merged_float_mapping(config.get("calibration", {}).get("task_loss_weight", {}), args.task_weight)
    category_weights = _parse_task_weights(args.category_weight)
    source_weights = _parse_task_weights(args.source_weight)
    train_coefficients = _parse_train_coefficients(args.train_coefficient)
    if train_coefficients and gate_parameterization not in {"global", "layer-band"}:
        raise SystemExit("--train-coefficient only applies to --gate-parameterization global or layer-band; parameterized modes train every mergeable task-vector coefficient")
    advantage_task_scales = _advantage_task_scales(
        rows,
        config,
        args,
        enabled=bool(args.task_normalize_advantages),
    )
    coefficient_anchor_gates = dict(config.get("initial_gates", {}))
    log_rows = []
    epoch_summaries = []
    early_stop_hits = 0
    stopped_early_at_step = None
    for step in range(1, args.max_steps + 1):
        epoch_log_start = len(log_rows)
        epoch_start_gates = gate_manager.gate_values()
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
            best_response_only = args.ppo_loss_weight == 0.0 and (
                float(args.best_response_loss_weight) != 0.0 or float(args.pairwise_loss_weight) != 0.0
            )
            valid_samples = _objective_samples(row["samples"], require_old_logprob=not best_response_only)
            if len(valid_samples) < 2:
                continue
            _validate_logprob_lengths(
                valid_samples,
                args.max_logprob_tokens,
                require_match=not best_response_only,
            )
            rewards = [float(sample["reward"]) for sample in valid_samples]
            category = _row_category(row)
            source = str(row.get("source") or "")
            source_weight = source_weights.get(source, 1.0)
            task_weight = task_weights.get(str(row.get("task")), 1.0) * category_weights.get(category, 1.0) * source_weight
            frontier_weight = policy_frontier_weight(
                row.get("frontier", {}),
                min_reward_std=float(config["frontier"]["min_reward_std"]),
            )
            if args.ppo_loss_weight == 0.0 and (
                float(args.best_response_loss_weight) != 0.0 or float(args.pairwise_loss_weight) != 0.0
            ):
                objective_stats = _backward_incremental_best_response_losses(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    valid_samples=valid_samples,
                    device=device,
                    max_logprob_tokens=args.max_logprob_tokens,
                    task_weight=task_weight,
                    best_response_loss_weight=float(args.best_response_loss_weight),
                    pairwise_loss_weight=float(args.pairwise_loss_weight),
                    pairwise_margin=float(args.pairwise_margin),
                    length_normalize=bool(args.length_normalize_logprob),
                    positive_reward_threshold=args.positive_reward_threshold,
                    max_pairwise_pairs_per_row=args.max_pairwise_pairs_per_row,
                )
                if objective_stats["processed"] < 1:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
                prior_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(gate_manager.parameters(), float(config["optimizer"]["grad_clip_norm"]))
                grad_norm_value = float(grad_norm.detach().cpu().item())
                skipped_step = grad_norm_value <= float(args.min_grad_norm_for_step)
                if not skipped_step:
                    optimizer.step()
                    _project_after_optimizer_step_(torch, gate_manager, train_coefficients, coefficient_anchor_gates, args)
                log_rows.append(
                    {
                        "step": step,
                        "prompt_id": row["prompt_id"],
                        "task": row["task"],
                        "category": category,
                        "source": source,
                        "source_weight": source_weight,
                        "task_weight": task_weight,
                        "queue": "frontier",
                        "loss": objective_stats["loss"] + float(prior_loss.detach().cpu().item()),
                        "policy_loss": 0.0,
                        "kl_loss": 0.0,
                        "best_response_loss": objective_stats["best_response_loss"],
                        "pairwise_loss": objective_stats["pairwise_loss"],
                        "retention_loss": 0.0,
                        "grad_norm": grad_norm_value,
                        "skipped_step": skipped_step,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": frontier_weight,
                        "gates": gate_manager.gate_values(),
                    }
                )
                continue
            advantage_values, advantage_source = _row_advantage_values(
                valid_samples,
                rewards,
                frontier_weight=frontier_weight,
                config=config,
                args=args,
            )
            advantages = torch.tensor(advantage_values, dtype=torch.float32, device=device)
            advantage_task_scale = float(advantage_task_scales.get(str(row.get("task")), 1.0))
            if advantage_task_scale != 1.0:
                advantages = advantages * advantage_task_scale
            if float(args.best_response_loss_weight) == 0.0 and float(args.pairwise_loss_weight) == 0.0:
                objective_stats = _backward_incremental_grpo_losses(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    valid_samples=valid_samples,
                    advantages=advantages,
                    device=device,
                    max_logprob_tokens=args.max_logprob_tokens,
                    task_weight=task_weight,
                    ppo_loss_weight=float(args.ppo_loss_weight),
                    beta_kl=float(config["loss"]["beta_kl"]),
                    eps_clip=float(config["loss"]["eps_clip"]),
                    length_normalize_policy_logprob=bool(args.length_normalize_policy_logprob),
                )
                if objective_stats["processed"] < 2:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
                prior_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(gate_manager.parameters(), float(config["optimizer"]["grad_clip_norm"]))
                grad_norm_value = float(grad_norm.detach().cpu().item())
                skipped_step = grad_norm_value <= float(args.min_grad_norm_for_step)
                if not skipped_step:
                    optimizer.step()
                    _project_after_optimizer_step_(torch, gate_manager, train_coefficients, coefficient_anchor_gates, args)
                loss_total = objective_stats["loss"] + float(prior_loss.detach().cpu().item())
                log_rows.append(
                    {
                        "step": step,
                        "prompt_id": row["prompt_id"],
                        "task": row["task"],
                        "category": category,
                        "source": source,
                        "source_weight": source_weight,
                        "task_weight": task_weight,
                        "queue": "frontier",
                        "loss": loss_total,
                        "policy_loss": objective_stats["policy_loss"],
                        "kl_loss": objective_stats["kl_loss"],
                        "best_response_loss": 0.0,
                        "pairwise_loss": 0.0,
                        "retention_loss": 0.0,
                        "grad_norm": grad_norm_value,
                        "skipped_step": skipped_step,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": frontier_weight,
                        "advantage_source": advantage_source,
                        "advantage_task_scale": advantage_task_scale,
                        "mean_abs_advantage": _mean_abs([float(value) for value in advantages.detach().cpu().tolist()]),
                        "gates": gate_manager.gate_values(),
                    }
                )
                continue
            policy_loss_total = 0.0
            kl_loss_total = 0.0
            logp_entries = []
            for sample_idx, sample in enumerate(valid_samples):
                logp = _sample_response_logprob_tensor(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    sample=sample,
                    device=device,
                    max_length=args.max_logprob_tokens,
                )
                if logp is None:
                    continue
                old_logp = torch.tensor(float(sample["old_logprob"]), dtype=torch.float32, device=device)
                current = logp.float()
                logp_entries.append(
                    {
                        "sample_idx": sample_idx,
                        "sample": sample,
                        "current": current,
                        "old": old_logp,
                        "reward": float(sample.get("reward", 0.0)),
                        "length": int(sample.get("length", 0) or 0),
                    }
                )
            if len(logp_entries) < 2:
                optimizer.zero_grad(set_to_none=True)
                continue
            denominator = float(len(logp_entries))
            loss_tensor = logp_entries[0]["current"].new_tensor(0.0)
            for entry in logp_entries:
                sample_idx = int(entry["sample_idx"])
                advantage = advantages[sample_idx]
                current = _entry_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                old_logp = _entry_old_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                if args.ppo_loss_weight != 0.0:
                    ratio = torch.exp((current - old_logp).clamp(-20.0, 20.0))
                    clipped = torch.clamp(ratio, 1.0 - float(config["loss"]["eps_clip"]), 1.0 + float(config["loss"]["eps_clip"]))
                    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage) / denominator
                    loss_tensor = loss_tensor + task_weight * float(args.ppo_loss_weight) * policy_loss
                    policy_loss_total += task_weight * float(args.ppo_loss_weight) * float(policy_loss.detach().cpu().item())
                log_ratio = (old_logp - current).clamp(-20.0, 20.0)
                kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
                loss_tensor = loss_tensor + task_weight * float(config["loss"]["beta_kl"]) * kl_loss
                kl_loss_total += task_weight * float(kl_loss.detach().cpu().item())
            best_response_loss = _best_response_loss(
                torch,
                logp_entries,
                length_normalize=bool(args.length_normalize_logprob),
                positive_reward_threshold=args.positive_reward_threshold,
            )
            pairwise_loss = _pairwise_best_response_loss(
                torch,
                logp_entries,
                margin=float(args.pairwise_margin),
                length_normalize=bool(args.length_normalize_logprob),
                positive_reward_threshold=args.positive_reward_threshold,
            )
            loss_tensor = loss_tensor + task_weight * float(args.best_response_loss_weight) * best_response_loss
            loss_tensor = loss_tensor + task_weight * float(args.pairwise_loss_weight) * pairwise_loss
            prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
            loss_tensor = loss_tensor + prior_loss
            loss_tensor.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(gate_manager.parameters(), float(config["optimizer"]["grad_clip_norm"]))
            grad_norm_value = float(grad_norm.detach().cpu().item())
            skipped_step = grad_norm_value <= float(args.min_grad_norm_for_step)
            if not skipped_step:
                optimizer.step()
                _project_after_optimizer_step_(torch, gate_manager, train_coefficients, coefficient_anchor_gates, args)
            loss_total = float(loss_tensor.detach().cpu().item())
            log_rows.append(
                {
                    "step": step,
                    "prompt_id": row["prompt_id"],
                    "task": row["task"],
                    "category": category,
                    "source": source,
                    "source_weight": source_weight,
                    "task_weight": task_weight,
                    "queue": "frontier",
                    "loss": loss_total,
                    "policy_loss": policy_loss_total,
                    "kl_loss": kl_loss_total,
                    "best_response_loss": task_weight * float(args.best_response_loss_weight) * float(best_response_loss.detach().cpu().item()),
                    "pairwise_loss": task_weight * float(args.pairwise_loss_weight) * float(pairwise_loss.detach().cpu().item()),
                    "retention_loss": 0.0,
                    "grad_norm": grad_norm_value,
                    "skipped_step": skipped_step,
                    "mean_reward": sum(rewards) / len(rewards),
                    "frontier_weight": frontier_weight,
                    "advantage_source": advantage_source,
                    "advantage_task_scale": advantage_task_scale,
                    "mean_abs_advantage": _mean_abs([float(value) for value in advantages.detach().cpu().tolist()]),
                    "gates": gate_manager.gate_values(),
                }
            )
        if args.use_retention and retention_weight > 0.0:
            for row in retention_rows:
                optimizer.zero_grad(set_to_none=True)
                prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
                valid_samples = _objective_samples(row["samples"], require_old_logprob=True)
                if not valid_samples:
                    continue
                _validate_logprob_lengths(valid_samples, args.max_logprob_tokens)
                retention_loss_total = 0.0
                processed = 0
                denominator = float(len(valid_samples))
                rewards = [float(sample.get("reward", 0.0)) for sample in valid_samples]
                category = _row_category(row)
                source = str(row.get("source") or "")
                source_weight = source_weights.get(source, 1.0)
                task_weight = task_weights.get(str(row.get("task")), 1.0) * category_weights.get(category, 1.0) * source_weight
                for sample in valid_samples:
                    logp = _sample_response_logprob_tensor(
                        torch,
                        model,
                        tokenizer,
                        prompt_text=prompt_text,
                        sample=sample,
                        device=device,
                        max_length=args.max_logprob_tokens,
                    )
                    if logp is None:
                        continue
                    old_logp = torch.tensor(float(sample["old_logprob"]), dtype=torch.float32, device=device)
                    current = logp.float()
                    log_ratio = (old_logp - current).clamp(-20.0, 20.0)
                    kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
                    sample_loss = task_weight * retention_weight * kl_loss
                    sample_loss.backward()
                    retention_loss_total += float(sample_loss.detach().cpu().item())
                    processed += 1
                if processed < 1:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(gate_manager.parameters(), float(config["optimizer"]["grad_clip_norm"]))
                grad_norm_value = float(grad_norm.detach().cpu().item())
                skipped_step = grad_norm_value <= float(args.min_grad_norm_for_step)
                if not skipped_step:
                    optimizer.step()
                    _project_after_optimizer_step_(torch, gate_manager, train_coefficients, coefficient_anchor_gates, args)
                log_rows.append(
                    {
                        "step": step,
                        "prompt_id": row["prompt_id"],
                        "task": row["task"],
                        "category": category,
                        "source": source,
                        "source_weight": source_weight,
                        "task_weight": task_weight,
                        "queue": "retention",
                        "loss": retention_loss_total,
                        "policy_loss": 0.0,
                        "kl_loss": retention_loss_total / retention_weight if retention_weight else 0.0,
                        "retention_loss": retention_loss_total,
                        "grad_norm": grad_norm_value,
                        "skipped_step": skipped_step,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": 0.0,
                        "gates": gate_manager.gate_values(),
                    }
                )
        epoch_rows = log_rows[epoch_log_start:]
        epoch_end_gates = gate_manager.gate_values()
        epoch_grad_max = max((float(item.get("grad_norm", 0.0)) for item in epoch_rows), default=0.0)
        epoch_gate_delta_max = max(
            (abs(float(epoch_end_gates.get(key, 0.0)) - float(epoch_start_gates.get(key, 0.0))) for key in set(epoch_start_gates) | set(epoch_end_gates)),
            default=0.0,
        )
        epoch_summary = {
            "step": step,
            "updates": len(epoch_rows),
            "grad_norm_max": epoch_grad_max,
            "gate_delta_max": epoch_gate_delta_max,
            "gates": epoch_end_gates,
        }
        epoch_summaries.append(epoch_summary)
        if _early_stop_reached(args, epoch_grad_max=epoch_grad_max, epoch_gate_delta_max=epoch_gate_delta_max):
            early_stop_hits += 1
        else:
            early_stop_hits = 0
        if early_stop_hits and early_stop_hits >= int(args.early_stop_patience):
            stopped_early_at_step = step
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in log_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "format": "opvec_gate_update_from_rollouts_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rollouts": list(args.rollouts),
        "replay_buffers": list(args.replay_buffer or []),
        "retention_only_replay_buffers": list(args.retention_only_replay_buffer or []),
        "output": str(output),
        "kept_frontier_rows": len(rows),
        "raw_frontier_task_counts": raw_frontier_task_counts,
        "frontier_task_counts": frontier_task_counts,
        "retention_rows": len(retention_rows),
        "updates": len(log_rows),
        "installed_modules": installed,
        "gate_parameterization": gate_parameterization,
        "device_map": args.device_map,
        "parameter_coefficients": 0 if param_names is None else len(param_names) * 3,
        "init_gate_checkpoint": args.init_gate_checkpoint,
        "filled_missing_old_logprobs": filled_old_logprobs,
        "final_gates": gate_manager.gate_values(),
        "gate_grad_nonzero": any(item["grad_norm"] > 0.0 for item in log_rows),
        "epoch_summaries": epoch_summaries,
        "stopped_early_at_step": stopped_early_at_step,
        "optimizer": {
            "lr": float(args.lr or config["optimizer"]["lr"]),
            "weight_decay": weight_decay,
            "prior_loss_weight": prior_weight,
            "min_grad_norm_for_step": float(args.min_grad_norm_for_step),
            "max_coefficient_delta_from_init": args.max_coefficient_delta_from_init,
            "early_stop_grad_norm": args.early_stop_grad_norm,
            "early_stop_gate_delta": args.early_stop_gate_delta,
            "early_stop_patience": args.early_stop_patience,
        },
        "task_weights": task_weights,
        "category_weights": category_weights,
        "source_weights": source_weights,
        "advantage_normalization": {
            "task_normalize_advantages": bool(args.task_normalize_advantages),
            "advantage_field": args.advantage_field,
            "advantage_field_apply_frontier_weight": bool(args.advantage_field_apply_frontier_weight),
            "task_scales": advantage_task_scales,
        },
        "train_coefficients": sorted(train_coefficients) if not _uses_parameter_names(gate_parameterization) else [f"{gate_parameterization}:all"],
        "expert_margin_constraints": {
            "tool_min_margin_over_memory": float(args.tool_min_margin_over_memory),
            "tool_min_margin_over_code": float(args.tool_min_margin_over_code),
        },
        "loss_weights": {
            "ppo": float(args.ppo_loss_weight),
            "best_response": float(args.best_response_loss_weight),
            "pairwise": float(args.pairwise_loss_weight),
            "pairwise_margin": float(args.pairwise_margin),
            "max_pairwise_pairs_per_row": int(args.max_pairwise_pairs_per_row),
            "length_normalize_logprob": bool(args.length_normalize_logprob),
            "length_normalize_policy_logprob": bool(args.length_normalize_policy_logprob),
            "positive_reward_threshold": args.positive_reward_threshold,
        },
    }
    write_json(output.with_suffix(".summary.json"), summary)
    write_json(output.with_suffix(".gates.json"), {"gates": summary["final_gates"]})
    return summary


def _early_stop_reached(args: argparse.Namespace, *, epoch_grad_max: float, epoch_gate_delta_max: float) -> bool:
    checks = []
    if args.early_stop_grad_norm is not None:
        checks.append(float(epoch_grad_max) <= float(args.early_stop_grad_norm))
    if args.early_stop_gate_delta is not None:
        checks.append(float(epoch_gate_delta_max) <= float(args.early_stop_gate_delta))
    return bool(checks) and all(checks)


def _validate_logprob_lengths(samples: list[dict], max_logprob_tokens: int, *, require_match: bool = True) -> None:
    if not require_match:
        return
    for sample in samples:
        recorded = sample.get("old_logprob_max_length")
        if recorded is None:
            continue
        if int(recorded) != int(max_logprob_tokens):
            raise ValueError(
                "old_logprob_max_length mismatch: "
                f"sample={sample.get('sample_id')} old={recorded} current={max_logprob_tokens}"
            )


def _objective_samples(samples: list[dict], *, require_old_logprob: bool) -> list[dict]:
    output = []
    for sample in samples:
        if not sample.get("text"):
            continue
        if require_old_logprob and sample.get("old_logprob") is None:
            continue
        output.append(sample)
    return output


def _fill_missing_old_logprobs(
    torch,
    model,
    tokenizer,
    rows: list[dict],
    *,
    device,
    max_logprob_tokens: int,
) -> int:
    filled = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
            for sample in row.get("samples", []):
                if sample.get("old_logprob") is not None:
                    continue
                if not sample.get("text"):
                    continue
                logp = _sample_response_logprob_tensor(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    sample=sample,
                    device=device,
                    max_length=max_logprob_tokens,
                )
                if logp is None:
                    continue
                sample["old_logprob"] = float(logp.detach().float().cpu().item())
                sample["old_logprob_max_length"] = int(max_logprob_tokens)
                filled += 1
    model.train()
    return filled


def _sample_response_logprob_tensor(torch, model, tokenizer, *, prompt_text: str, sample: dict, device, max_length: int):
    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return response_logprob_tensor_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            response_text=str(sample["text"]),
            device=device,
            max_length=max_length,
        )
    total = None
    for turn in trajectory:
        if not isinstance(turn, dict):
            continue
        turn_prompt = str(turn.get("prompt_text") or "")
        turn_text = str(turn.get("text") or "")
        if not turn_prompt or not turn_text:
            continue
        logp = response_logprob_tensor_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=turn_prompt,
            response_text=turn_text,
            device=device,
            max_length=max_length,
        )
        if logp is None:
            continue
        total = logp.float() if total is None else total + logp.float()
    return total


def _normalize_gate_parameterization(value: str) -> str:
    aliases = {
        "layer_band": "layer-band",
        "param": "parameter",
        "param-coefficients": "parameter",
        "parameter-coefficients": "parameter",
        "global_parameter": "global-parameter",
        "global-param": "global-parameter",
        "global_param": "global-parameter",
        "global-residual": "global-parameter",
        "global_residual": "global-parameter",
    }
    return aliases.get(str(value), str(value))


def _uses_parameter_names(gate_parameterization: str) -> bool:
    return gate_parameterization in {"parameter", "global-parameter"}


def _gate_prior_loss(torch, gate_manager):
    return gate_initialization_prior(torch, gate_manager)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--rollouts", action="append", default=[])
    parser.add_argument("--replay-buffer", action="append", default=[])
    parser.add_argument(
        "--retention-only-replay-buffer",
        action="append",
        default=[],
        help="Replay buffer whose frontier and retention queues are used only for KL retention.",
    )
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-gated-modules", type=int, default=None, help="Maximum gated Linear modules to install. Default None means all mergeable modules; use 1 only for smoke tests.")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-logprob-tokens", type=int, default=3072)
    parser.add_argument(
        "--fill-missing-old-logprob",
        action="store_true",
        help="Compute missing old_logprob values once under the initial gate policy before optimizing gates.",
    )
    parser.add_argument("--recompute-frontier", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--prior-loss-weight", type=float, default=None)
    parser.add_argument("--min-grad-norm-for-step", type=float, default=0.0)
    parser.add_argument(
        "--max-coefficient-delta-from-init",
        type=float,
        default=None,
        help="Hard per-parameter bound on how far trainable coefficients may move from their initialization.",
    )
    parser.add_argument("--use-retention", action="store_true")
    parser.add_argument("--max-retention-rows", type=int, default=None)
    parser.add_argument("--retention-loss-weight", type=float, default=None)
    parser.add_argument("--early-stop-grad-norm", type=float, default=None)
    parser.add_argument("--early-stop-gate-delta", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--task-weight", action="append", default=[], help="Per-task loss weight, e.g. memory=2.0. Repeatable.")
    parser.add_argument("--frontier-task-quota", action="append", default=[], help="Max kept frontier rows per task, e.g. memory=64. Repeatable.")
    parser.add_argument("--max-frontier-rows-per-task", type=int, default=None, help="Shared cap for kept frontier rows per task.")
    parser.add_argument("--category-weight", action="append", default=[], help="Per-category loss weight, e.g. tool:live_parallel=2.0. Repeatable.")
    parser.add_argument("--source-weight", action="append", default=[], help="Per-row-source loss weight, e.g. bfcl_success_anchor_frontier=0.2. Repeatable.")
    parser.add_argument(
        "--train-coefficient",
        action="append",
        default=[],
        help=(
            "Trainable expert coefficients for global/layer-band gates, e.g. "
            "global.memory,global.code or mid.tool. Repeatable; supports comma-separated values."
        ),
    )
    parser.add_argument(
        "--tool-min-margin-over-memory",
        type=float,
        default=0.0,
        help="Hard post-step projection requiring effective tool coefficient >= memory coefficient + margin for global/layer-band gates.",
    )
    parser.add_argument(
        "--tool-min-margin-over-code",
        type=float,
        default=0.0,
        help="Hard post-step projection requiring effective tool coefficient >= code coefficient + margin for global/layer-band gates.",
    )
    parser.add_argument("--ppo-loss-weight", type=float, default=1.0)
    parser.add_argument("--best-response-loss-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument(
        "--max-pairwise-pairs-per-row",
        type=int,
        default=0,
        help="Cap best-response pairwise comparisons per frontier row; 0 keeps all positive-negative pairs.",
    )
    parser.add_argument("--length-normalize-logprob", action="store_true")
    parser.add_argument(
        "--length-normalize-policy-logprob",
        action="store_true",
        help="Use per-token average logprob for PPO ratio/KL terms instead of full response logprob.",
    )
    parser.add_argument(
        "--task-normalize-advantages",
        action="store_true",
        help="Rescale row-normalized GRPO advantages so each task has the same mean absolute advantage over kept frontier rows.",
    )
    parser.add_argument(
        "--advantage-field",
        default=None,
        help=(
            "Use this per-sample numeric field directly as the PPO advantage instead of "
            "row-normalized reward advantages. Intended for self-compare fields such as "
            "reward_delta_vs_baseline; the values are not row-mean centered."
        ),
    )
    parser.add_argument(
        "--no-advantage-field-frontier-weight",
        dest="advantage_field_apply_frontier_weight",
        action="store_false",
        help="Do not multiply --advantage-field values by row frontier_weight.",
    )
    parser.set_defaults(advantage_field_apply_frontier_weight=True)
    parser.add_argument("--positive-reward-threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto, for multi-GPU sharding.")
    parser.add_argument("--max-memory", action="append", default=[], help="HF max_memory entry, e.g. 0=70GiB. Repeatable.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable HF gradient checkpointing during gate updates.")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--gate-parameterization", choices=["global", "layer-band", "parameter", "global-parameter"], default="global")
    parser.add_argument("--init-gate-checkpoint", default=None)
    args = parser.parse_args()
    if not args.rollouts and not args.replay_buffer and not args.retention_only_replay_buffer:
        parser.error("at least one --rollouts, --replay-buffer, or --retention-only-replay-buffer input is required")
    return args


def _parse_task_weights(items: list[str]) -> dict[str, float]:
    weights = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --task-weight value: {item}")
        task, value = item.split("=", 1)
        weights[task.strip()] = float(value)
    return weights


def _parse_task_quota(items: list[str]) -> dict[str, int]:
    quotas = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid task quota value: {item}")
        task, value = item.split("=", 1)
        quotas[task.strip()] = int(value)
    return quotas


def _merged_float_mapping(config_values: dict | None, cli_items: list[str]) -> dict[str, float]:
    values = {str(key): float(value) for key, value in dict(config_values or {}).items()}
    values.update(_parse_task_weights(cli_items))
    return values


def _merged_task_quota(config: dict, cli_items: list[str]) -> dict[str, int]:
    values = {
        str(key): int(value)
        for key, value in dict(config.get("calibration", {}).get("frontier_task_quota", {}) or {}).items()
    }
    values.update(_parse_task_quota(cli_items))
    return values


def _task_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(str(row.get("task")) for row in rows))


def _is_retention_candidate(row: dict) -> bool:
    frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else {}
    return bool(frontier.get("all_success")) or row.get("skip_reason") == "all_success"


def _limit_frontier_rows(
    rows: list[dict],
    *,
    task_quota: dict[str, int],
    max_per_task: int | None,
) -> list[dict]:
    if not task_quota and max_per_task is None:
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task"))].append(row)
    selected_by_task: dict[str, set[int]] = {}
    for task, task_rows in grouped.items():
        limit = task_quota.get(task)
        if max_per_task is not None:
            limit = min(int(max_per_task), int(limit)) if limit is not None else int(max_per_task)
        if limit is None or limit < 0:
            selected_by_task[task] = {id(row) for row in task_rows}
        else:
            selected_by_task[task] = {id(row) for row in task_rows[: int(limit)]}
    return [row for row in rows if id(row) in selected_by_task.get(str(row.get("task")), set())]


def _row_advantage_values(
    samples: list[dict],
    rewards: list[float],
    *,
    frontier_weight: float,
    config: dict,
    args: argparse.Namespace,
) -> tuple[list[float], str]:
    if args.advantage_field:
        values = []
        missing = []
        for sample in samples:
            if args.advantage_field not in sample:
                missing.append(str(sample.get("sample_id") or "<unknown>"))
                continue
            values.append(float(sample[args.advantage_field]))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing --advantage-field {args.advantage_field!r} on samples: {preview}")
        if bool(args.advantage_field_apply_frontier_weight):
            values = [float(frontier_weight) * value for value in values]
            source = f"field:{args.advantage_field}:frontier_weighted"
        else:
            source = f"field:{args.advantage_field}"
        return values, source
    return (
        group_relative_advantages(
            rewards,
            frontier_weight=frontier_weight,
            epsilon=float(config["loss"]["advantage_eps"]),
        ),
        "group_relative_reward",
    )


def _advantage_task_scales(rows: list[dict], config: dict, args: argparse.Namespace, *, enabled: bool) -> dict[str, float]:
    if not enabled:
        return {}
    task_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
        if len(samples) < 2:
            continue
        rewards = [float(sample.get("reward", 0.0)) for sample in samples]
        frontier_weight = policy_frontier_weight(
            row.get("frontier", {}),
            min_reward_std=float(config["frontier"]["min_reward_std"]),
        )
        advantages, _ = _row_advantage_values(
            samples,
            rewards,
            frontier_weight=frontier_weight,
            config=config,
            args=args,
        )
        mean_abs = _mean_abs(advantages)
        if mean_abs > 0.0:
            task_values[str(row.get("task"))].append(mean_abs)
    task_means = {
        task: sum(values) / len(values)
        for task, values in task_values.items()
        if values and sum(values) > 0.0
    }
    if not task_means:
        return {}
    target = sum(task_means.values()) / len(task_means)
    if target <= 0.0:
        return {}
    return {
        task: float(target / value)
        for task, value in sorted(task_means.items())
        if value > 0.0
    }


def _mean_abs(values) -> float:
    items = [abs(float(value)) for value in values]
    return float(sum(items) / len(items)) if items else 0.0


def _parse_train_coefficients(items: list[str]) -> set[str]:
    coefficients: set[str] = set()
    valid_experts = {"tool", "memory", "code", "*"}
    for item in items:
        for part in item.split(","):
            value = part.strip()
            if not value:
                continue
            if "." not in value:
                raise ValueError(f"Invalid --train-coefficient value: {value}")
            band, expert = value.split(".", 1)
            if expert not in valid_experts:
                raise ValueError(f"Invalid expert in --train-coefficient value: {value}")
            coefficients.add(f"{band}.{expert}")
    return coefficients


def _project_after_optimizer_step_(torch, gate_manager, train_coefficients: set[str], anchor_gate_values: dict[str, float], args: argparse.Namespace) -> None:
    gate_manager.project_()
    _project_trainable_coefficients_(torch, gate_manager, train_coefficients, anchor_gate_values)
    _project_max_delta_from_initial_(torch, gate_manager, args.max_coefficient_delta_from_init)
    _project_tool_margin_constraints_(
        torch,
        gate_manager,
        train_coefficients,
        anchor_gate_values,
        max_delta=args.max_coefficient_delta_from_init,
        tool_min_margin_over_memory=float(args.tool_min_margin_over_memory),
        tool_min_margin_over_code=float(args.tool_min_margin_over_code),
    )


def _backward_incremental_best_response_losses(
    torch,
    model,
    tokenizer,
    *,
    prompt_text: str,
    valid_samples: list[dict],
    device: str,
    max_logprob_tokens: int,
    task_weight: float,
    best_response_loss_weight: float,
    pairwise_loss_weight: float,
    pairwise_margin: float,
    length_normalize: bool,
    positive_reward_threshold: float | None,
    max_pairwise_pairs_per_row: int = 0,
) -> dict[str, float]:
    positives, negatives = _positive_and_negative_samples(valid_samples, positive_reward_threshold)
    if not positives:
        return {"processed": 0, "loss": 0.0, "best_response_loss": 0.0, "pairwise_loss": 0.0}
    processed = 0
    total_loss = 0.0
    best_response_total = 0.0
    pairwise_total = 0.0
    if best_response_loss_weight != 0.0:
        denominator = float(len(positives))
        for sample in positives:
            logp = _sample_response_logprob_tensor(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                sample=sample,
                device=device,
                max_length=max_logprob_tokens,
            )
            if logp is None:
                continue
            loss = task_weight * best_response_loss_weight * (-_sample_score(logp.float(), sample, length_normalize=length_normalize)) / denominator
            loss.backward()
            value = float(loss.detach().cpu().item())
            total_loss += value
            best_response_total += value
            processed += 1
    if pairwise_loss_weight != 0.0 and negatives:
        pairs = _bounded_pairwise_samples(
            positives,
            negatives,
            max_pairs=int(max_pairwise_pairs_per_row),
        )
        denominator = float(len(pairs))
        for positive, negative in pairs:
            positive_logp = _sample_response_logprob_tensor(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                sample=positive,
                device=device,
                max_length=max_logprob_tokens,
            )
            negative_logp = _sample_response_logprob_tensor(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                sample=negative,
                device=device,
                max_length=max_logprob_tokens,
            )
            if positive_logp is None or negative_logp is None:
                continue
            positive_score = _sample_score(positive_logp.float(), positive, length_normalize=length_normalize)
            negative_score = _sample_score(negative_logp.float(), negative, length_normalize=length_normalize)
            pair_loss = torch.nn.functional.softplus(negative_score - positive_score + float(pairwise_margin))
            loss = task_weight * pairwise_loss_weight * pair_loss / denominator
            loss.backward()
            value = float(loss.detach().cpu().item())
            total_loss += value
            pairwise_total += value
            processed += 1
    return {
        "processed": float(processed),
        "loss": total_loss,
        "best_response_loss": best_response_total,
        "pairwise_loss": pairwise_total,
    }


def _backward_incremental_grpo_losses(
    torch,
    model,
    tokenizer,
    *,
    prompt_text: str,
    valid_samples: list[dict],
    advantages,
    device,
    max_logprob_tokens: int,
    task_weight: float,
    ppo_loss_weight: float,
    beta_kl: float,
    eps_clip: float,
    length_normalize_policy_logprob: bool,
) -> dict[str, float]:
    denominator = float(len(valid_samples))
    processed = 0
    loss_total = 0.0
    policy_loss_total = 0.0
    kl_loss_total = 0.0
    for sample_idx, sample in enumerate(valid_samples):
        logp = _sample_response_logprob_tensor(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            sample=sample,
            device=device,
            max_length=max_logprob_tokens,
        )
        if logp is None:
            continue
        current_raw = logp.float()
        old_logp_raw = torch.tensor(float(sample["old_logprob"]), dtype=torch.float32, device=current_raw.device)
        current = _sample_score(current_raw, sample, length_normalize=length_normalize_policy_logprob)
        old_logp = _sample_score(old_logp_raw, sample, length_normalize=length_normalize_policy_logprob)
        sample_loss = current_raw.new_tensor(0.0)
        if ppo_loss_weight != 0.0:
            advantage = advantages[sample_idx].to(current.device)
            ratio = torch.exp((current - old_logp).clamp(-20.0, 20.0))
            clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage) / denominator
            weighted_policy = task_weight * ppo_loss_weight * policy_loss
            sample_loss = sample_loss + weighted_policy
            policy_loss_total += float(weighted_policy.detach().cpu().item())
        log_ratio = (old_logp - current).clamp(-20.0, 20.0)
        kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
        weighted_kl = task_weight * beta_kl * kl_loss
        sample_loss = sample_loss + weighted_kl
        kl_loss_total += float(weighted_kl.detach().cpu().item())
        sample_loss.backward()
        loss_total += float(sample_loss.detach().cpu().item())
        processed += 1
    return {
        "processed": float(processed),
        "loss": loss_total,
        "policy_loss": policy_loss_total,
        "kl_loss": kl_loss_total,
    }


def _positive_and_negative_samples(samples: list[dict], positive_reward_threshold: float | None) -> tuple[list[dict], list[dict]]:
    best_reward = max(float(sample.get("reward", 0.0)) for sample in samples)
    threshold = best_reward - 1e-8 if positive_reward_threshold is None else float(positive_reward_threshold)
    positives = [sample for sample in samples if float(sample.get("reward", 0.0)) >= threshold]
    negative_cutoff = best_reward if positive_reward_threshold is None else threshold
    negatives = [sample for sample in samples if float(sample.get("reward", 0.0)) < negative_cutoff]
    return positives, negatives


def _bounded_pairwise_samples(
    positives: list[dict],
    negatives: list[dict],
    *,
    max_pairs: int,
) -> list[tuple[dict, dict]]:
    pairs = [(positive, negative) for positive in positives for negative in negatives]
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    pairs.sort(
        key=lambda item: (
            -float(item[0].get("reward", 0.0)),
            float(item[1].get("reward", 0.0)),
            str(item[0].get("sample_id", "")),
            str(item[1].get("sample_id", "")),
        )
    )
    return pairs[:max_pairs]


def _sample_score(logp, sample: dict, *, length_normalize: bool):
    if not length_normalize:
        return logp
    return logp / max(int(sample.get("length", 0) or 0), 1)


def _best_response_loss(
    torch,
    entries: list[dict],
    *,
    length_normalize: bool,
    positive_reward_threshold: float | None,
):
    positives = _positive_entries(entries, positive_reward_threshold)
    if not positives:
        return entries[0]["current"].new_tensor(0.0)
    scores = torch.stack([_entry_score(entry, length_normalize=length_normalize) for entry in positives])
    return -scores.mean()


def _pairwise_best_response_loss(
    torch,
    entries: list[dict],
    *,
    margin: float,
    length_normalize: bool,
    positive_reward_threshold: float | None,
):
    positives = _positive_entries(entries, positive_reward_threshold)
    if not positives:
        return entries[0]["current"].new_tensor(0.0)
    best_reward = max(float(entry["reward"]) for entry in positives)
    negatives = [entry for entry in entries if float(entry["reward"]) < best_reward]
    if not negatives:
        return entries[0]["current"].new_tensor(0.0)
    pos_scores = torch.stack([_entry_score(entry, length_normalize=length_normalize) for entry in positives])
    pos_score = pos_scores.mean()
    losses = [
        torch.nn.functional.softplus(_entry_score(entry, length_normalize=length_normalize) - pos_score + float(margin))
        for entry in negatives
    ]
    return torch.stack(losses).mean()


def _positive_entries(entries: list[dict], positive_reward_threshold: float | None) -> list[dict]:
    best_reward = max(float(entry["reward"]) for entry in entries)
    threshold = best_reward if positive_reward_threshold is None else float(positive_reward_threshold)
    cutoff = max(best_reward - 1e-8, threshold)
    return [entry for entry in entries if float(entry["reward"]) >= cutoff]


def _entry_score(entry: dict, *, length_normalize: bool):
    current = entry["current"]
    if not length_normalize:
        return current
    return current / max(int(entry.get("length", 0) or 0), 1)


def _entry_old_score(entry: dict, *, length_normalize: bool):
    old = entry["old"]
    if not length_normalize:
        return old
    return old / max(int(entry.get("length", 0) or 0), 1)


def _project_trainable_coefficients_(torch, gate_manager, train_coefficients: set[str], anchor_gate_values: dict[str, float]) -> None:
    if not train_coefficients:
        return
    band_names = tuple(getattr(gate_manager, "band_names", ()))
    if not band_names:
        if not (
            hasattr(gate_manager, "raw_common")
            and hasattr(gate_manager, "raw_residual")
            and callable(getattr(gate_manager, "expert_coefficients", None))
        ):
            raise ValueError("coefficient projection requires a global or layer-band gate manager")
        experts = ("tool", "memory", "code")
        with torch.no_grad():
            current = gate_manager.expert_coefficients()
            anchor = _anchor_expert_coefficients(anchor_gate_values, "global")
            coeffs = []
            for expert in experts:
                if _coefficient_is_trainable(train_coefficients, "global", expert):
                    coeffs.append(current[expert].detach().to(device=gate_manager.raw_common.device, dtype=gate_manager.raw_common.dtype))
                else:
                    coeffs.append(
                        torch.tensor(
                            anchor[expert],
                            device=gate_manager.raw_common.device,
                            dtype=gate_manager.raw_common.dtype,
                        )
                    )
            coeff_tensor = torch.stack(coeffs)
            common = coeff_tensor.mean()
            gate_manager.raw_common.copy_(common)
            gate_manager.raw_residual.copy_(coeff_tensor - common)
        gate_manager.project_()
        return
    experts = ("tool", "memory", "code")
    with torch.no_grad():
        for band_idx, band in enumerate(band_names):
            current = gate_manager.expert_coefficients(band=band)
            anchor = _anchor_expert_coefficients(anchor_gate_values, band)
            coeffs = []
            for expert in experts:
                if _coefficient_is_trainable(train_coefficients, band, expert):
                    coeffs.append(current[expert].detach().to(device=gate_manager.raw_common.device, dtype=gate_manager.raw_common.dtype))
                else:
                    coeffs.append(
                        torch.tensor(
                            anchor[expert],
                            device=gate_manager.raw_common.device,
                            dtype=gate_manager.raw_common.dtype,
                        )
                    )
            coeff_tensor = torch.stack(coeffs)
            common = coeff_tensor.mean()
            gate_manager.raw_common[band_idx].copy_(common)
            gate_manager.raw_residual[band_idx].copy_(coeff_tensor - common)
    gate_manager.project_()


def _project_max_delta_from_initial_(torch, gate_manager, max_delta: float | None) -> None:
    """Hard trust-region projection for all OP-VEC gate parameterizations.

    The soft prior keeps updates local on average; this projection is the safety
    rail that prevents a single high-variance frontier batch from moving expert
    strengths too far from the chosen TA/cg anchor.  It must apply to the
    default ``global`` manager as well as parameterized managers.
    """

    if max_delta is None:
        return
    delta = float(max_delta)
    if delta < 0:
        raise ValueError("--max-coefficient-delta-from-init must be non-negative")
    with torch.no_grad():
        if (
            hasattr(gate_manager, "raw_global_coefficients")
            and hasattr(gate_manager, "initial_global_coefficients")
            and hasattr(gate_manager, "raw_residual_coefficients")
            and hasattr(gate_manager, "initial_residual_coefficients")
        ):
            gate_manager.raw_global_coefficients.copy_(
                torch.clamp(
                    gate_manager.raw_global_coefficients,
                    min=gate_manager.initial_global_coefficients - delta,
                    max=gate_manager.initial_global_coefficients + delta,
                )
            )
            gate_manager.raw_residual_coefficients.copy_(
                torch.clamp(
                    gate_manager.raw_residual_coefficients,
                    min=gate_manager.initial_residual_coefficients - delta,
                    max=gate_manager.initial_residual_coefficients + delta,
                )
            )
            gate_manager.project_()
            return
        if hasattr(gate_manager, "raw_coefficients") and hasattr(gate_manager, "initial_coefficients"):
            gate_manager.raw_coefficients.copy_(
                torch.clamp(
                    gate_manager.raw_coefficients,
                    min=gate_manager.initial_coefficients - delta,
                    max=gate_manager.initial_coefficients + delta,
                )
            )
            gate_manager.project_()
            return
        if (
            hasattr(gate_manager, "raw_common")
            and hasattr(gate_manager, "initial_raw_common")
            and hasattr(gate_manager, "raw_residual")
            and hasattr(gate_manager, "initial_raw_residual")
        ):
            # For global/layer-band gates the behaviorally meaningful values are
            # expert coefficients common + zero-mean residual.  Clamp those
            # coefficients to the initialization, then reparameterize back into
            # common/residual form.  Clamping raw residuals directly can violate
            # the bound after the zero-mean projection.
            current_common = gate_manager.raw_common
            current_residual = gate_manager.raw_residual
            initial_common = gate_manager.initial_raw_common
            initial_residual = gate_manager.initial_raw_residual
            if current_residual.dim() == 1:
                current_common_value = current_common.reshape(-1)[0]
                initial_common_value = initial_common.reshape(-1)[0]
                centered = current_residual - current_residual.mean()
                initial_centered = initial_residual - initial_residual.mean()
                coeffs = current_common_value + centered
                anchors = initial_common_value + initial_centered
                clamped = torch.clamp(coeffs, min=anchors - delta, max=anchors + delta)
                new_common = clamped.mean()
                gate_manager.raw_common.copy_(new_common.reshape_as(gate_manager.raw_common))
                gate_manager.raw_residual.copy_(clamped - new_common)
            else:
                centered = current_residual - current_residual.mean(dim=1, keepdim=True)
                initial_centered = initial_residual - initial_residual.mean(dim=1, keepdim=True)
                coeffs = current_common.unsqueeze(1) + centered
                anchors = initial_common.unsqueeze(1) + initial_centered
                clamped = torch.clamp(coeffs, min=anchors - delta, max=anchors + delta)
                new_common = clamped.mean(dim=1)
                gate_manager.raw_common.copy_(new_common)
                gate_manager.raw_residual.copy_(clamped - new_common.unsqueeze(1))
            gate_manager.project_()
            return


def _project_tool_margin_constraints_(
    torch,
    gate_manager,
    train_coefficients: set[str],
    anchor_gate_values: dict[str, float],
    *,
    max_delta: float | None,
    tool_min_margin_over_memory: float,
    tool_min_margin_over_code: float,
) -> None:
    margins = {
        "memory": float(tool_min_margin_over_memory),
        "code": float(tool_min_margin_over_code),
    }
    margins = {expert: margin for expert, margin in margins.items() if margin > 0.0}
    if not margins:
        return
    if not (hasattr(gate_manager, "raw_common") and hasattr(gate_manager, "raw_residual")):
        raise ValueError("Tool margin constraints currently require global or layer-band gates")
    with torch.no_grad():
        band_names = tuple(getattr(gate_manager, "band_names", ()))
        if not band_names:
            current = gate_manager.expert_coefficients()
            projected = _project_tool_margin_coefficients(
                {expert: float(value.detach().cpu().item()) for expert, value in current.items()},
                train_coefficients=train_coefficients,
                band="global",
                anchor=_anchor_expert_coefficients(anchor_gate_values, "global"),
                max_delta=max_delta,
                margins=margins,
            )
            _write_global_coefficients_(torch, gate_manager, projected)
            gate_manager.project_()
            return
        for band_idx, band in enumerate(band_names):
            current = gate_manager.expert_coefficients(band=band)
            projected = _project_tool_margin_coefficients(
                {expert: float(value.detach().cpu().item()) for expert, value in current.items()},
                train_coefficients=train_coefficients,
                band=band,
                anchor=_anchor_expert_coefficients(anchor_gate_values, band),
                max_delta=max_delta,
                margins=margins,
            )
            _write_band_coefficients_(torch, gate_manager, band_idx, projected)
        gate_manager.project_()


def _project_tool_margin_coefficients(
    values: dict[str, float],
    *,
    train_coefficients: set[str],
    band: str,
    anchor: dict[str, float],
    max_delta: float | None,
    margins: dict[str, float],
) -> dict[str, float]:
    projected = {expert: float(values[expert]) for expert in ("tool", "memory", "code")}
    bounds = _coefficient_bounds(anchor, max_delta)
    for other, margin in margins.items():
        projected = _enforce_pair_margin(
            projected,
            leader="tool",
            follower=other,
            margin=margin,
            train_coefficients=train_coefficients,
            band=band,
            bounds=bounds,
        )
    return projected


def _enforce_pair_margin(
    values: dict[str, float],
    *,
    leader: str,
    follower: str,
    margin: float,
    train_coefficients: set[str],
    band: str,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    if values[leader] >= values[follower] + margin:
        return values
    can_move_leader = _expert_coefficient_trainable(train_coefficients, band, leader)
    can_move_follower = _expert_coefficient_trainable(train_coefficients, band, follower)
    if not can_move_leader and not can_move_follower:
        raise ValueError(f"Infeasible margin constraint: {leader}>={follower}+{margin}, but neither coefficient is trainable")

    projected = dict(values)
    for _ in range(4):
        if projected[leader] >= projected[follower] + margin - 1.0e-8:
            break
        if can_move_leader and can_move_follower:
            pair_sum = projected[leader] + projected[follower]
            projected[leader] = (pair_sum + margin) / 2.0
            projected[follower] = (pair_sum - margin) / 2.0
        elif can_move_leader:
            projected[leader] = projected[follower] + margin
        elif can_move_follower:
            projected[follower] = projected[leader] - margin
        projected[leader] = _clamp(projected[leader], bounds[leader])
        projected[follower] = _clamp(projected[follower], bounds[follower])
        if projected[leader] < projected[follower] + margin - 1.0e-8 and can_move_follower:
            projected[follower] = _clamp(projected[leader] - margin, bounds[follower])
        if projected[leader] < projected[follower] + margin - 1.0e-8 and can_move_leader:
            projected[leader] = _clamp(projected[follower] + margin, bounds[leader])
    if projected[leader] < projected[follower] + margin - 1.0e-6:
        raise ValueError(
            f"Infeasible margin constraint after projection: "
            f"{leader}={projected[leader]:.6f}, {follower}={projected[follower]:.6f}, margin={margin:.6f}"
        )
    return projected


def _coefficient_bounds(anchor: dict[str, float], max_delta: float | None) -> dict[str, tuple[float, float]]:
    if max_delta is None:
        return {expert: (-float("inf"), float("inf")) for expert in ("tool", "memory", "code")}
    delta = float(max_delta)
    return {
        expert: (float(anchor[expert]) - delta, float(anchor[expert]) + delta)
        for expert in ("tool", "memory", "code")
    }


def _expert_coefficient_trainable(train_coefficients: set[str], band: str, expert: str) -> bool:
    return not train_coefficients or _coefficient_is_trainable(train_coefficients, band, expert)


def _write_global_coefficients_(torch, gate_manager, coeffs: dict[str, float]) -> None:
    tensor = torch.tensor(
        [float(coeffs["tool"]), float(coeffs["memory"]), float(coeffs["code"])],
        device=gate_manager.raw_common.device,
        dtype=gate_manager.raw_common.dtype,
    )
    common = tensor.mean()
    gate_manager.raw_common.copy_(common)
    gate_manager.raw_residual.copy_(tensor - common)


def _write_band_coefficients_(torch, gate_manager, band_idx: int, coeffs: dict[str, float]) -> None:
    tensor = torch.tensor(
        [float(coeffs["tool"]), float(coeffs["memory"]), float(coeffs["code"])],
        device=gate_manager.raw_common.device,
        dtype=gate_manager.raw_common.dtype,
    )
    common = tensor.mean()
    gate_manager.raw_common[band_idx].copy_(common)
    gate_manager.raw_residual[band_idx].copy_(tensor - common)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(float(value), float(bounds[0])), float(bounds[1]))


def _coefficient_is_trainable(train_coefficients: set[str], band: str, expert: str) -> bool:
    return (
        f"{band}.{expert}" in train_coefficients
        or f"*.{expert}" in train_coefficients
        or f"{band}.*" in train_coefficients
        or "*.*" in train_coefficients
    )


def _anchor_expert_coefficients(anchor_gate_values: dict[str, float], band: str) -> dict[str, float]:
    common = float(anchor_gate_values.get(f"{band}.common", anchor_gate_values.get("common", 0.5)))
    residuals = {
        "tool": float(anchor_gate_values.get(f"{band}.tool_residual", anchor_gate_values.get("tool_residual", 0.0))),
        "memory": float(anchor_gate_values.get(f"{band}.memory_residual", anchor_gate_values.get("memory_residual", 0.0))),
        "code": float(anchor_gate_values.get(f"{band}.code_residual", anchor_gate_values.get("code_residual", 0.0))),
    }
    mean_residual = sum(residuals.values()) / 3.0
    return {expert: common + residual - mean_residual for expert, residual in residuals.items()}


def _row_category(row: dict) -> str:
    task = str(row.get("task"))
    if task == "tool":
        bfcl_category = row.get("reference", {}).get("bfcl", {}).get("category")
        if bfcl_category:
            return f"tool:{bfcl_category}"
    return task


if __name__ == "__main__":
    main()
