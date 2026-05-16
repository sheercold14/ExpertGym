#!/usr/bin/env python3
"""Update OP-VEC gates from collected rollout JSONL."""

from __future__ import annotations

import argparse
import json
import random
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
from opvec.modeling.logprob import (
    response_logprob_tensor_details_from_text,
    response_logprob_tensor_details_from_token_ids,
    response_logprob_tensor_from_text,
)
from opvec.train.frontier import group_relative_advantages, policy_frontier_weight, should_keep_frontier
from opvec.train.gated_grpo import clipped_grpo_token_loss, gate_initialization_prior, reverse_kl_token_penalty


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
    opd_rows = []
    opd_all_success_rows = []
    for replay_path in args.replay_buffer or []:
        payload = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        queues = payload.get("queues", {})
        rows.extend(queues.get(QUEUE_FRONTIER, []))
        if args.use_retention:
            retention_rows.extend(queues.get(QUEUE_RETENTION, []))
        if args.use_opd_all_success:
            opd_all_success_rows.extend(queues.get(QUEUE_RETENTION, []))
    for replay_path in args.retention_only_replay_buffer or []:
        payload = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        queues = payload.get("queues", {})
        retention_rows.extend(queues.get(QUEUE_FRONTIER, []))
        retention_rows.extend(queues.get(QUEUE_RETENTION, []))
    raw_rows = []
    for rollout_path in args.rollouts or []:
        raw_rows.extend(read_jsonl(rollout_path))
    for rollout_path in args.opd_distill_rollout or []:
        opd_rows.extend(read_jsonl(rollout_path))
    for row in raw_rows:
        if args.recompute_frontier:
            queue, row = classify_rollout_row(
                row,
                min_frontier_weight=float(config["frontier"]["min_frontier_weight"]),
                min_reward_std=float(config["frontier"]["min_reward_std"]),
            )
            if args.use_retention and queue == QUEUE_RETENTION:
                retention_rows.append(row)
            if args.use_opd_all_success and queue == QUEUE_RETENTION:
                opd_all_success_rows.append(row)
        elif _is_retention_candidate(row):
            if args.use_retention:
                retention_rows.append(row)
            if args.use_opd_all_success:
                opd_all_success_rows.append(row)
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
    rows = _order_frontier_rows(
        rows,
        order=str(args.frontier_order),
        seed=_frontier_shuffle_seed(args),
    )
    frontier_task_counts = _task_counts(rows)
    if args.max_retention_rows_per_task is not None:
        retention_rows = _limit_rows_per_task(retention_rows, int(args.max_retention_rows_per_task))
    if args.max_retention_rows is not None:
        retention_rows = retention_rows[: args.max_retention_rows]
    if args.max_opd_distill_rows is not None:
        opd_rows = opd_rows[: args.max_opd_distill_rows]
    if args.max_opd_all_success_rows is not None:
        opd_all_success_rows = opd_all_success_rows[: args.max_opd_all_success_rows]
    if not rows and not opd_rows and not opd_all_success_rows and not (args.use_retention and retention_rows):
        raise SystemExit("No kept frontier, retention, or OPD distill rows found in rollout/replay-buffer inputs")
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
            fill_token_logprobs=args.loss_granularity == "token",
        )
        if args.use_retention and retention_rows and args.retention_objective == "kl":
            filled_old_logprobs += _fill_missing_old_logprobs(
                torch,
                model,
                tokenizer,
                retention_rows,
                device=device,
                max_logprob_tokens=args.max_logprob_tokens,
                fill_token_logprobs=args.loss_granularity == "token",
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    weight_decay = float(
        args.weight_decay if args.weight_decay is not None else config["optimizer"].get("weight_decay", 0.0)
    )
    optimizer = _make_optimizer(
        torch,
        gate_manager,
        optimizer_name=str(args.optimizer),
        lr=float(args.lr or config["optimizer"]["lr"]),
        weight_decay=weight_decay,
        sgd_momentum=float(args.sgd_momentum),
        sgd_nesterov=bool(args.sgd_nesterov),
    )
    optimizer_state_loaded = False
    if args.optimizer_state_in:
        state_path = Path(args.optimizer_state_in)
        if state_path.exists():
            optimizer.load_state_dict(torch.load(state_path, map_location=device))
            optimizer_state_loaded = True
        else:
            raise SystemExit(f"--optimizer-state-in not found: {state_path}")
    retention_weight = float(args.retention_loss_weight if args.retention_loss_weight is not None else config["loss"].get("lambda_retention", 0.0))
    prior_weight = float(args.prior_loss_weight if args.prior_loss_weight is not None else config["loss"].get("lambda_prior", 0.0))
    if args.pcgrad_gate_gradients and args.optimizer_step_scope != "epoch":
        raise ValueError("--pcgrad-gate-gradients currently requires --optimizer-step-scope epoch")
    task_weights = _merged_float_mapping(config.get("calibration", {}).get("task_loss_weight", {}), args.task_weight)
    category_weights = _parse_task_weights(args.category_weight)
    source_weights = _parse_task_weights(args.source_weight)
    opd_length_normalize = _resolve_component_length_normalize(
        args.opd_length_normalize_logprob,
        fallback=bool(args.length_normalize_logprob),
    )
    retention_length_normalize = _resolve_component_length_normalize(
        args.retention_length_normalize_logprob,
        fallback=bool(args.length_normalize_logprob),
    )
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
    update_batcher = _UpdateBatcher(
        torch=torch,
        optimizer=optimizer,
        gate_manager=gate_manager,
        grad_clip_norm=float(config["optimizer"]["grad_clip_norm"]),
        min_grad_norm_for_step=float(args.min_grad_norm_for_step),
        update_batch_size=int(args.update_batch_size),
        batch_loss_reduction=str(args.batch_loss_reduction),
        optimizer_step_scope=str(args.optimizer_step_scope),
        loss_normalizer=_planned_optimizer_loss_normalizer(
            args=args,
            frontier_rows=rows,
            retention_rows=retention_rows,
            opd_rows=opd_rows,
            opd_all_success_rows=opd_all_success_rows,
            retention_weight=retention_weight,
        ),
        train_coefficients=train_coefficients,
        coefficient_anchor_gates=coefficient_anchor_gates,
        args=args,
    )
    opd_scale_plan = _build_opd_scale_plan(
        torch,
        model,
        tokenizer,
        opd_rows=opd_rows,
        raw_rows=raw_rows,
        args=args,
        device=device,
        max_logprob_tokens=int(args.max_logprob_tokens),
        task_weights=task_weights,
        category_weights=category_weights,
        source_weights=source_weights,
        loss_normalizer=update_batcher.loss_normalizer,
        length_normalize=opd_length_normalize,
    )
    retention_scale_plan = _build_retention_scale_plan(
        torch,
        model,
        tokenizer,
        retention_rows=retention_rows,
        raw_rows=raw_rows,
        args=args,
        device=device,
        max_logprob_tokens=int(args.max_logprob_tokens),
        task_weights=task_weights,
        category_weights=category_weights,
        source_weights=source_weights,
        retention_weight=retention_weight,
        loss_normalizer=update_batcher.loss_normalizer,
        length_normalize=retention_length_normalize,
    )
    if args.pcgrad_gate_gradients:
        update_batcher.pcgrad_recompute_fn = lambda: _replace_gate_grads_with_pcgrad(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            gate_manager=gate_manager,
            optimizer=optimizer,
            rows=rows,
            opd_rows=opd_rows,
            opd_all_success_rows=opd_all_success_rows,
            retention_rows=retention_rows,
            raw_rows=raw_rows,
            config=config,
            args=args,
            device=device,
            max_logprob_tokens=int(args.max_logprob_tokens),
            update_batcher=update_batcher,
            task_weights=task_weights,
            category_weights=category_weights,
            source_weights=source_weights,
            advantage_task_scales=advantage_task_scales,
            opd_scale_plan=opd_scale_plan,
            retention_scale_plan=retention_scale_plan,
            retention_weight=retention_weight,
            prior_weight=prior_weight,
            opd_length_normalize=opd_length_normalize,
            retention_length_normalize=retention_length_normalize,
        )
    log_rows = []
    epoch_summaries = []
    early_stop_hits = 0
    stopped_early_at_step = None
    for step in range(1, args.max_steps + 1):
        epoch_log_start = len(log_rows)
        epoch_start_gates = gate_manager.gate_values()
        for row in rows:
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
            task_name = str(row.get("task", ""))
            rewards = [_sample_train_reward(sample, task=task_name) for sample in valid_samples]
            category = _row_category(row)
            source = str(row.get("source") or "")
            source_weight = source_weights.get(source, 1.0)
            task_weight = task_weights.get(str(row.get("task")), 1.0) * category_weights.get(category, 1.0) * source_weight
            frontier_weight = _policy_frontier_multiplier(row, config, args)
            if args.ppo_loss_weight == 0.0 and (
                float(args.best_response_loss_weight) != 0.0 or float(args.pairwise_loss_weight) != 0.0
            ):
                objective_stats = _backward_incremental_best_response_losses(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    valid_samples=valid_samples,
                    task=task_name,
                    device=device,
                    max_logprob_tokens=args.max_logprob_tokens,
                    task_weight=task_weight,
                    best_response_loss_weight=float(args.best_response_loss_weight),
                    pairwise_loss_weight=float(args.pairwise_loss_weight),
                    pairwise_margin=float(args.pairwise_margin),
                    length_normalize=bool(args.length_normalize_logprob),
                    positive_reward_threshold=args.positive_reward_threshold,
                    max_pairwise_pairs_per_row=args.max_pairwise_pairs_per_row,
                    loss_scale=update_batcher.loss_scale,
                )
                if objective_stats["processed"] < 1:
                    continue
                prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
                (prior_loss * update_batcher.loss_scale).backward()
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
                        "clip_frac": 0.0,
                        "approx_kl": 0.0,
                        "best_response_loss": objective_stats["best_response_loss"],
                        "pairwise_loss": objective_stats["pairwise_loss"],
                        "retention_loss": 0.0,
                        "grad_norm": 0.0,
                        "skipped_step": False,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": frontier_weight,
                        "reward_field": "reward_train",
                        "loss_granularity": str(args.loss_granularity),
                        "gates": {},
                    }
                )
                update_batcher.add(log_rows, len(log_rows) - 1)
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
                    loss_granularity=str(args.loss_granularity),
                    loss_scale=update_batcher.loss_scale,
                )
                if objective_stats["processed"] < 2:
                    continue
                prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
                (prior_loss * update_batcher.loss_scale).backward()
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
                        "clip_frac": objective_stats["clip_frac"],
                        "approx_kl": objective_stats["approx_kl"],
                        "best_response_loss": 0.0,
                        "pairwise_loss": 0.0,
                        "retention_loss": 0.0,
                        "grad_norm": 0.0,
                        "skipped_step": False,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": frontier_weight,
                        "reward_field": "reward_train",
                        "advantage_source": advantage_source,
                        "advantage_task_scale": advantage_task_scale,
                        "mean_abs_advantage": _mean_abs([float(value) for value in advantages.detach().cpu().tolist()]),
                        "loss_granularity": str(args.loss_granularity),
                        "gates": {},
                    }
                )
                update_batcher.add(log_rows, len(log_rows) - 1)
                continue
            policy_loss_total = 0.0
            kl_loss_total = 0.0
            clip_frac_total = 0.0
            approx_kl_total = 0.0
            logp_entries = []
            for sample_idx, sample in enumerate(valid_samples):
                entry = _sample_response_logprob_entry(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    sample=sample,
                    device=device,
                    max_length=args.max_logprob_tokens,
                    loss_granularity=str(args.loss_granularity),
                )
                if entry is None:
                    continue
                entry.update(
                    {
                        "sample_idx": sample_idx,
                        "sample": sample,
                        "reward": _sample_train_reward(sample, task=task_name),
                        "raw_reward": float(sample.get("reward", 0.0)),
                        "length": int(sample.get("length", 0) or 0),
                    }
                )
                logp_entries.append(entry)
            if len(logp_entries) < 2:
                continue
            denominator = float(len(logp_entries))
            loss_tensor = logp_entries[0]["current"].new_tensor(0.0)
            for entry in logp_entries:
                sample_idx = int(entry["sample_idx"])
                advantage = advantages[sample_idx]
                if args.ppo_loss_weight != 0.0:
                    if args.loss_granularity == "token":
                        policy_loss = clipped_grpo_token_loss(
                            torch,
                            current_logprobs=entry["current_logprobs"],
                            old_logprobs=entry["old_logprobs"],
                            response_mask=entry["response_mask"],
                            advantage=advantage,
                            clip_epsilon=float(config["loss"]["eps_clip"]),
                        ) / denominator
                    else:
                        current = _entry_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                        old_logp = _entry_old_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                        ratio = torch.exp((current - old_logp).clamp(-20.0, 20.0))
                        clipped = torch.clamp(ratio, 1.0 - float(config["loss"]["eps_clip"]), 1.0 + float(config["loss"]["eps_clip"]))
                        policy_loss = -torch.minimum(ratio * advantage, clipped * advantage) / denominator
                    loss_tensor = loss_tensor + task_weight * float(args.ppo_loss_weight) * policy_loss
                    policy_loss_total += task_weight * float(args.ppo_loss_weight) * float(policy_loss.detach().cpu().item())
                if args.loss_granularity == "token":
                    kl_loss = reverse_kl_token_penalty(
                        torch,
                        current_logprobs=entry["current_logprobs"],
                        old_logprobs=entry["old_logprobs"],
                        response_mask=entry["response_mask"],
                    ) / denominator
                else:
                    current = _entry_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                    old_logp = _entry_old_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                    log_ratio = (old_logp - current).clamp(-20.0, 20.0)
                    kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
                loss_tensor = loss_tensor + task_weight * float(config["loss"]["beta_kl"]) * kl_loss
                kl_loss_total += task_weight * float(kl_loss.detach().cpu().item())
                metrics = _entry_policy_metrics(
                    torch,
                    entry,
                    entry["sample"],
                    length_normalize=bool(args.length_normalize_policy_logprob),
                    loss_granularity=str(args.loss_granularity),
                    eps_clip=float(config["loss"]["eps_clip"]),
                )
                clip_frac_total += metrics["clip_frac"] / denominator
                approx_kl_total += metrics["approx_kl"] / denominator
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
            loss_total = float(loss_tensor.detach().cpu().item())
            (loss_tensor * update_batcher.loss_scale).backward()
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
                    "clip_frac": clip_frac_total,
                    "approx_kl": approx_kl_total,
                    "best_response_loss": task_weight * float(args.best_response_loss_weight) * float(best_response_loss.detach().cpu().item()),
                    "pairwise_loss": task_weight * float(args.pairwise_loss_weight) * float(pairwise_loss.detach().cpu().item()),
                    "retention_loss": 0.0,
                    "grad_norm": 0.0,
                    "skipped_step": False,
                    "mean_reward": sum(rewards) / len(rewards),
                    "frontier_weight": frontier_weight,
                    "reward_field": "reward_train",
                    "advantage_source": advantage_source,
                    "advantage_task_scale": advantage_task_scale,
                    "mean_abs_advantage": _mean_abs([float(value) for value in advantages.detach().cpu().tolist()]),
                    "loss_granularity": str(args.loss_granularity),
                    "gates": {},
                }
            )
            update_batcher.add(log_rows, len(log_rows) - 1)
        update_batcher.flush(log_rows)
        if opd_rows and (float(args.opd_loss_weight) != 0.0 or float(args.opd_pairwise_loss_weight) != 0.0):
            for row in opd_rows:
                prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
                valid_samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
                if len(valid_samples) < 2:
                    continue
                task_name = str(row.get("task", ""))
                rewards = [_sample_train_reward(sample, task=task_name) for sample in valid_samples]
                category = _row_category(row)
                source = str(row.get("source") or "opd_distill")
                source_weight = source_weights.get(source, 1.0)
                task_weight = task_weights.get(str(row.get("task")), 1.0) * category_weights.get(category, 1.0) * source_weight
                opd_row_scale = _opd_row_loss_scale(
                    row,
                    opd_scale_plan=opd_scale_plan,
                    default_loss_scale=update_batcher.loss_scale,
                )
                opd_component_scale = _opd_component_scale(row, opd_scale_plan=opd_scale_plan)
                objective_stats = _backward_incremental_best_response_losses(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    valid_samples=valid_samples,
                    task=task_name,
                    device=device,
                    max_logprob_tokens=args.max_logprob_tokens,
                    task_weight=task_weight,
                    best_response_loss_weight=float(args.opd_loss_weight),
                    pairwise_loss_weight=float(args.opd_pairwise_loss_weight),
                    pairwise_margin=float(args.opd_pairwise_margin),
                    length_normalize=opd_length_normalize,
                    positive_reward_threshold=args.opd_positive_reward_threshold,
                    max_pairwise_pairs_per_row=args.max_opd_pairwise_pairs_per_row,
                    loss_scale=opd_row_scale,
                    component_scale=opd_component_scale,
                )
                if objective_stats["processed"] < 1:
                    continue
                prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
                (prior_loss * opd_row_scale).backward()
                log_rows.append(
                    {
                        "step": step,
                        "prompt_id": row.get("prompt_id", ""),
                        "task": row.get("task", ""),
                        "category": category,
                        "source": source,
                        "source_weight": source_weight,
                        "task_weight": task_weight,
                        "queue": "opd_distill",
                        "loss": objective_stats["loss"] + float(prior_loss.detach().cpu().item()),
                        "policy_loss": 0.0,
                        "kl_loss": 0.0,
                        "clip_frac": 0.0,
                        "approx_kl": 0.0,
                        "best_response_loss": objective_stats["best_response_loss"],
                        "pairwise_loss": objective_stats["pairwise_loss"],
                        "retention_loss": 0.0,
                        "grad_norm": 0.0,
                        "skipped_step": False,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": 0.0,
                        "reward_field": "reward_train",
                        "opd_positive_reward_threshold": args.opd_positive_reward_threshold,
                        "opd_length_normalize_logprob": opd_length_normalize,
                        "opd_dynamic_scale": opd_component_scale,
                        "opd_row_loss_scale": opd_row_scale,
                        "opd_scale_target_ratio": _opd_task_plan_value(row, opd_scale_plan, "target_ratio"),
                        "opd_recoverable_all_fail_rate": _opd_task_plan_value(row, opd_scale_plan, "recoverable_all_fail_rate"),
                        "loss_granularity": "sequence",
                        "gates": {},
                    }
                )
                update_batcher.add(log_rows, len(log_rows) - 1)
            update_batcher.flush(log_rows)
        if args.use_opd_all_success and opd_all_success_rows and float(args.opd_all_success_loss_weight) != 0.0:
            for row in opd_all_success_rows:
                prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
                valid_samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
                if not valid_samples:
                    continue
                task_name = str(row.get("task", ""))
                rewards = [_sample_train_reward(sample, task=task_name) for sample in valid_samples]
                category = _row_category(row)
                source = str(row.get("source") or "opd_all_success")
                source_weight = source_weights.get(source, 1.0)
                task_weight = task_weights.get(str(row.get("task")), 1.0) * category_weights.get(category, 1.0) * source_weight
                objective_stats = _backward_incremental_best_response_losses(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    valid_samples=valid_samples,
                    task=task_name,
                    device=device,
                    max_logprob_tokens=args.max_logprob_tokens,
                    task_weight=task_weight,
                    best_response_loss_weight=float(args.opd_all_success_loss_weight),
                    pairwise_loss_weight=0.0,
                    pairwise_margin=0.0,
                    length_normalize=retention_length_normalize,
                    positive_reward_threshold=args.opd_all_success_positive_reward_threshold,
                    max_pairwise_pairs_per_row=0,
                    loss_scale=update_batcher.loss_scale,
                )
                if objective_stats["processed"] < 1:
                    continue
                prior_loss = _gate_prior_loss(torch, gate_manager) * prior_weight
                (prior_loss * update_batcher.loss_scale).backward()
                log_rows.append(
                    {
                        "step": step,
                        "prompt_id": row.get("prompt_id", ""),
                        "task": row.get("task", ""),
                        "category": category,
                        "source": source,
                        "source_weight": source_weight,
                        "task_weight": task_weight,
                        "queue": "opd_all_success",
                        "loss": objective_stats["loss"] + float(prior_loss.detach().cpu().item()),
                        "policy_loss": 0.0,
                        "kl_loss": 0.0,
                        "clip_frac": 0.0,
                        "approx_kl": 0.0,
                        "best_response_loss": objective_stats["best_response_loss"],
                        "pairwise_loss": 0.0,
                        "retention_loss": 0.0,
                        "grad_norm": 0.0,
                        "skipped_step": False,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": 0.0,
                        "reward_field": "reward_train",
                        "opd_positive_reward_threshold": args.opd_all_success_positive_reward_threshold,
                        "opd_length_normalize_logprob": retention_length_normalize,
                        "loss_granularity": "sequence",
                        "gates": {},
                    }
                )
                update_batcher.add(log_rows, len(log_rows) - 1)
            update_batcher.flush(log_rows)
        if args.use_retention and retention_weight > 0.0:
            for row in retention_rows:
                prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
                valid_samples = _objective_samples(row["samples"], require_old_logprob=args.retention_objective == "kl")
                if not valid_samples:
                    continue
                if args.retention_objective == "kl":
                    _validate_logprob_lengths(valid_samples, args.max_logprob_tokens)
                task_name = str(row.get("task", ""))
                rewards = [_sample_train_reward(sample, task=task_name) for sample in valid_samples]
                category = _row_category(row)
                source = str(row.get("source") or "")
                source_weight = source_weights.get(source, 1.0)
                task_weight = task_weights.get(str(row.get("task")), 1.0) * category_weights.get(category, 1.0) * source_weight
                if args.retention_objective == "nll":
                    retention_row_scale = _retention_row_loss_scale(
                        row,
                        retention_scale_plan=retention_scale_plan,
                        default_loss_scale=update_batcher.loss_scale,
                    )
                    retention_component_scale = _retention_component_scale(
                        row,
                        retention_scale_plan=retention_scale_plan,
                    )
                    objective_stats = _backward_incremental_best_response_losses(
                        torch,
                        model,
                        tokenizer,
                        prompt_text=prompt_text,
                        valid_samples=valid_samples,
                        task=task_name,
                        device=device,
                        max_logprob_tokens=args.max_logprob_tokens,
                        task_weight=task_weight,
                        best_response_loss_weight=retention_weight,
                        pairwise_loss_weight=0.0,
                        pairwise_margin=0.0,
                        length_normalize=retention_length_normalize,
                        positive_reward_threshold=args.retention_positive_reward_threshold,
                        max_pairwise_pairs_per_row=0,
                        loss_scale=retention_row_scale,
                        component_scale=retention_component_scale,
                    )
                    if objective_stats["processed"] < 1:
                        continue
                    retention_loss_total = objective_stats["loss"]
                    kl_loss_total = 0.0
                    best_response_loss_total = objective_stats["best_response_loss"]
                else:
                    retention_row_scale = update_batcher.loss_scale
                    retention_component_scale = 1.0
                    retention_loss_total = 0.0
                    kl_loss_total = 0.0
                    processed = 0
                    denominator = float(len(valid_samples))
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
                        sample_loss = task_weight * retention_weight * retention_component_scale * kl_loss
                        (sample_loss * retention_row_scale).backward()
                        value = float(sample_loss.detach().cpu().item())
                        retention_loss_total += value
                        kl_loss_total += value / retention_weight if retention_weight else 0.0
                        processed += 1
                    if processed < 1:
                        continue
                    best_response_loss_total = 0.0
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
                        "kl_loss": kl_loss_total,
                        "clip_frac": 0.0,
                        "approx_kl": 0.0,
                        "best_response_loss": best_response_loss_total,
                        "pairwise_loss": 0.0,
                        "retention_loss": retention_loss_total,
                        "retention_objective": args.retention_objective,
                        "retention_positive_reward_threshold": args.retention_positive_reward_threshold,
                        "retention_length_normalize_logprob": retention_length_normalize,
                        "retention_dynamic_scale": retention_component_scale,
                        "retention_row_loss_scale": retention_row_scale,
                        "retention_scale_target_ratio": _retention_task_plan_value(row, retention_scale_plan, "target_ratio"),
                        "grad_norm": 0.0,
                        "skipped_step": False,
                        "mean_reward": sum(rewards) / len(rewards),
                        "frontier_weight": 0.0,
                        "gates": {},
                    }
                )
                update_batcher.add(log_rows, len(log_rows) - 1)
            update_batcher.flush(log_rows)
        update_batcher.flush(log_rows, force=True)
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
            "clip_frac_mean": _mean([float(item.get("clip_frac", 0.0)) for item in epoch_rows]),
            "approx_kl_mean": _mean([float(item.get("approx_kl", 0.0)) for item in epoch_rows]),
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
        "opd_distill_rollouts": list(args.opd_distill_rollout or []),
        "output": str(output),
        "kept_frontier_rows": len(rows),
        "raw_frontier_task_counts": raw_frontier_task_counts,
        "frontier_task_counts": frontier_task_counts,
        "frontier_order": {
            "order": str(args.frontier_order),
            "seed": _frontier_shuffle_seed(args),
        },
        "retention_rows": len(retention_rows),
        "opd_distill_rows": len(opd_rows),
        "opd_distill_task_counts": _task_counts(opd_rows),
        "opd_scale_plan": opd_scale_plan,
        "retention_scale_plan": retention_scale_plan,
        "opd_all_success_rows": len(opd_all_success_rows),
        "opd_all_success_task_counts": _task_counts(opd_all_success_rows),
        "updates": len(log_rows),
        "optimizer_steps": update_batcher.optimizer_steps,
        "skipped_optimizer_steps": update_batcher.skipped_optimizer_steps,
        "installed_modules": installed,
        "gate_parameterization": gate_parameterization,
        "device_map": args.device_map,
        "parameter_coefficients": 0
        if param_names is None
        else len(param_names) * len(tuple(getattr(gate_manager, "expert_names", ("tool", "memory", "code")))),
        "init_gate_checkpoint": args.init_gate_checkpoint,
        "filled_missing_old_logprobs": filled_old_logprobs,
        "final_gates": gate_manager.gate_values(),
        "gate_grad_nonzero": any(item["grad_norm"] > 0.0 for item in log_rows),
        "epoch_summaries": epoch_summaries,
        "stopped_early_at_step": stopped_early_at_step,
        "optimizer": {
            "name": str(args.optimizer),
            "lr": float(args.lr or config["optimizer"]["lr"]),
            "weight_decay": weight_decay,
            "sgd_momentum": float(args.sgd_momentum),
            "sgd_nesterov": bool(args.sgd_nesterov),
            "optimizer_state_in": args.optimizer_state_in,
            "optimizer_state_loaded": optimizer_state_loaded,
            "optimizer_state_out": args.optimizer_state_out,
            "prior_loss_weight": prior_weight,
            "min_grad_norm_for_step": float(args.min_grad_norm_for_step),
            "update_batch_size": int(args.update_batch_size),
            "batch_loss_reduction": str(args.batch_loss_reduction),
            "optimizer_step_scope": str(args.optimizer_step_scope),
            "loss_normalizer": update_batcher.loss_normalizer,
            "loss_granularity": str(args.loss_granularity),
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
            "advantage_normalization": str(args.advantage_normalization),
            "reward_field": "reward_train",
            "use_frontier_weight": bool(args.use_frontier_weight),
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
            "opd": float(args.opd_loss_weight),
            "opd_pairwise": float(args.opd_pairwise_loss_weight),
            "opd_pairwise_margin": float(args.opd_pairwise_margin),
            "max_opd_pairwise_pairs_per_row": int(args.max_opd_pairwise_pairs_per_row),
            "opd_positive_reward_threshold": args.opd_positive_reward_threshold,
            "retention_objective": str(args.retention_objective),
            "retention_positive_reward_threshold": args.retention_positive_reward_threshold,
            "opd_all_success": float(args.opd_all_success_loss_weight),
            "opd_all_success_positive_reward_threshold": args.opd_all_success_positive_reward_threshold,
            "length_normalize_logprob": bool(args.length_normalize_logprob),
            "length_normalize_policy_logprob": bool(args.length_normalize_policy_logprob),
            "opd_length_normalize_logprob": opd_length_normalize,
            "retention_length_normalize_logprob": retention_length_normalize,
            "opd_dynamic_scale": bool(args.opd_dynamic_scale),
            "opd_task_balanced_loss_scale": bool(args.opd_task_balanced_loss_scale),
            "opd_scale_mode": str(args.opd_scale_mode),
            "retention_dynamic_scale": bool(args.retention_dynamic_scale),
            "retention_task_balanced_loss_scale": bool(args.retention_task_balanced_loss_scale),
            "retention_scale_target": float(args.retention_scale_target),
            "positive_reward_threshold": args.positive_reward_threshold,
        },
    }
    if args.pcgrad_gate_gradients:
        pcgrad_rows = [item for item in log_rows if item.get("pcgrad_enabled")]
        summary["pcgrad"] = {
            "enabled": True,
            "eps": float(args.pcgrad_eps),
            "tasks": list(args.pcgrad_task or []),
            "conflict_count_max": max((int(item.get("pcgrad_conflict_count", 0)) for item in pcgrad_rows), default=0),
        }
    write_json(output.with_suffix(".summary.json"), summary)
    write_json(output.with_suffix(".gates.json"), {"gates": summary["final_gates"]})
    if args.optimizer_state_out:
        state_out = Path(args.optimizer_state_out)
        state_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(optimizer.state_dict(), state_out)
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
    fill_token_logprobs: bool = False,
) -> int:
    filled = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
            for sample in row.get("samples", []):
                needs_sequence_logprob = sample.get("old_logprob") is None
                needs_token_logprobs = bool(fill_token_logprobs) and not _sample_has_token_logprobs(sample)
                if not needs_sequence_logprob and not needs_token_logprobs:
                    continue
                if not sample.get("text"):
                    continue
                if fill_token_logprobs:
                    payload = _sample_response_token_logprob_payload(
                        torch,
                        model,
                        tokenizer,
                        prompt_text=prompt_text,
                        sample=sample,
                        device=device,
                        max_length=max_logprob_tokens,
                    )
                    if payload is None:
                        continue
                    sample.update(payload)
                    sample["old_logprob"] = float(payload["old_logprob"])
                    sample["old_logprob_max_length"] = int(max_logprob_tokens)
                    filled += 1
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


def _sample_has_token_logprobs(sample: dict) -> bool:
    return (
        isinstance(sample.get("response_token_ids"), list)
        and isinstance(sample.get("old_logprobs"), list)
        and isinstance(sample.get("response_mask"), list)
    )


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


def _sample_response_logprob_entry(
    torch,
    model,
    tokenizer,
    *,
    prompt_text: str,
    sample: dict,
    device,
    max_length: int,
    loss_granularity: str,
) -> dict | None:
    if loss_granularity != "token":
        logp = _sample_response_logprob_tensor(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            sample=sample,
            device=device,
            max_length=max_length,
        )
        if logp is None:
            return None
        old_logp = torch.tensor(float(sample["old_logprob"]), dtype=torch.float32, device=device)
        return {"current": logp.float(), "old": old_logp}
    token_details = _sample_response_token_logprob_details(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        sample=sample,
        device=device,
        max_length=max_length,
    )
    if token_details is None:
        return None
    old_logprobs = sample.get("old_logprobs")
    response_mask = sample.get("response_mask")
    if not isinstance(old_logprobs, list) or not isinstance(response_mask, list):
        return None
    current_logprobs = token_details["logprobs"].float()
    if len(old_logprobs) != int(current_logprobs.numel()) or len(response_mask) != int(current_logprobs.numel()):
        raise ValueError(
            "token-level logprob length mismatch: "
            f"sample={sample.get('sample_id')} current={int(current_logprobs.numel())} "
            f"old={len(old_logprobs)} mask={len(response_mask)}"
        )
    old_tensor = torch.tensor([float(value) for value in old_logprobs], dtype=torch.float32, device=current_logprobs.device)
    mask_tensor = torch.tensor([float(value) for value in response_mask], dtype=torch.float32, device=current_logprobs.device)
    current_sum = (current_logprobs * mask_tensor).sum()
    old_sum = (old_tensor * mask_tensor).sum()
    return {
        "current": current_sum,
        "old": old_sum,
        "current_logprobs": current_logprobs,
        "old_logprobs": old_tensor,
        "response_mask": mask_tensor,
    }


def _sample_response_token_logprob_details(torch, model, tokenizer, *, prompt_text: str, sample: dict, device, max_length: int):
    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        response_token_ids = sample.get("response_token_ids")
        if isinstance(response_token_ids, list) and response_token_ids:
            return response_logprob_tensor_details_from_token_ids(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                response_token_ids=[int(value) for value in response_token_ids],
                device=device,
                max_length=max_length,
            )
        return response_logprob_tensor_details_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            response_text=str(sample["text"]),
            device=device,
            max_length=max_length,
        )
    logprob_tensors = []
    token_ids = []
    response_mask = []
    for turn in trajectory:
        if not isinstance(turn, dict):
            continue
        turn_prompt = str(turn.get("prompt_text") or "")
        turn_text = str(turn.get("text") or "")
        if not turn_prompt or not turn_text:
            continue
        turn_token_ids = turn.get("response_token_ids")
        if isinstance(turn_token_ids, list) and turn_token_ids:
            details = response_logprob_tensor_details_from_token_ids(
                torch,
                model,
                tokenizer,
                prompt_text=turn_prompt,
                response_token_ids=[int(value) for value in turn_token_ids],
                device=device,
                max_length=max_length,
            )
        else:
            details = response_logprob_tensor_details_from_text(
                torch,
                model,
                tokenizer,
                prompt_text=turn_prompt,
                response_text=turn_text,
                device=device,
                max_length=max_length,
            )
        if details is None:
            continue
        logprob_tensors.append(details["logprobs"].float())
        token_ids.extend(details["response_token_ids"])
        response_mask.extend(details["response_mask"])
    if not logprob_tensors:
        return None
    return {
        "response_token_ids": token_ids,
        "logprobs": torch.cat(logprob_tensors),
        "response_mask": response_mask,
    }


def _sample_response_token_logprob_payload(torch, model, tokenizer, *, prompt_text: str, sample: dict, device, max_length: int) -> dict | None:
    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        details = response_logprob_tensor_details_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            response_text=str(sample["text"]),
            device=device,
            max_length=max_length,
        )
        return _token_logprob_payload_from_details(details)
    token_ids = []
    old_logprobs = []
    response_mask = []
    total_logprob = 0.0
    for turn in trajectory:
        if not isinstance(turn, dict):
            continue
        turn_prompt = str(turn.get("prompt_text") or "")
        turn_text = str(turn.get("text") or "")
        if not turn_prompt or not turn_text:
            continue
        details = response_logprob_tensor_details_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=turn_prompt,
            response_text=turn_text,
            device=device,
            max_length=max_length,
        )
        turn_payload = _token_logprob_payload_from_details(details)
        if turn_payload is None:
            return None
        turn.update(turn_payload)
        turn["old_logprob"] = float(turn_payload["old_logprob"])
        turn["old_logprob_max_length"] = int(max_length)
        token_ids.extend(turn_payload["response_token_ids"])
        old_logprobs.extend(turn_payload["old_logprobs"])
        response_mask.extend(turn_payload["response_mask"])
        total_logprob += float(turn_payload["old_logprob"])
    if not token_ids:
        return None
    return {
        "response_token_ids": token_ids,
        "old_logprobs": old_logprobs,
        "response_mask": response_mask,
        "old_logprob": total_logprob,
    }


def _token_logprob_payload_from_details(details: dict | None) -> dict | None:
    if details is None:
        return None
    logprobs = [float(value) for value in details["logprobs"].detach().float().cpu().tolist()]
    return {
        "response_token_ids": [int(value) for value in details["response_token_ids"]],
        "old_logprobs": logprobs,
        "response_mask": [int(value) for value in details["response_mask"]],
        "old_logprob": float(sum(logprobs)),
    }


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
        "global_coefficient": "global-coefficient",
        "global-coefficients": "global-coefficient",
        "global_coefficients": "global-coefficient",
        "global-direct": "global-coefficient",
        "global_direct": "global-coefficient",
        "expert-coefficient": "global-coefficient",
        "expert_coefficient": "global-coefficient",
    }
    return aliases.get(str(value), str(value))


def _uses_parameter_names(gate_parameterization: str) -> bool:
    return gate_parameterization in {"parameter", "global-parameter"}


def _gate_prior_loss(torch, gate_manager):
    return gate_initialization_prior(torch, gate_manager)


def _make_optimizer(
    torch,
    gate_manager,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    sgd_momentum: float,
    sgd_nesterov: bool,
):
    params = gate_manager.parameters()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(lr),
            momentum=float(sgd_momentum),
            weight_decay=float(weight_decay),
            nesterov=bool(sgd_nesterov),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _planned_optimizer_loss_normalizer(
    *,
    args: argparse.Namespace,
    frontier_rows: list[dict],
    retention_rows: list[dict],
    opd_rows: list[dict],
    opd_all_success_rows: list[dict],
    retention_weight: float,
) -> int:
    """Planned row count for epoch-scope mean loss scaling."""
    count = len(frontier_rows)
    if args.use_retention and retention_weight > 0.0:
        count += len(retention_rows)
    if float(args.opd_loss_weight) != 0.0 or float(args.opd_pairwise_loss_weight) != 0.0:
        count += len(opd_rows)
    if args.use_opd_all_success and float(args.opd_all_success_loss_weight) != 0.0:
        count += len(opd_all_success_rows)
    return max(1, int(count))


class _UpdateBatcher:
    """Accumulate row-level losses and apply optimizer steps at batch or epoch scope."""

    def __init__(
        self,
        *,
        torch,
        optimizer,
        gate_manager,
        grad_clip_norm: float,
        min_grad_norm_for_step: float,
        update_batch_size: int,
        batch_loss_reduction: str,
        optimizer_step_scope: str,
        loss_normalizer: int,
        train_coefficients: set[str],
        coefficient_anchor_gates: dict[str, float],
        args: argparse.Namespace,
    ) -> None:
        self.torch = torch
        self.optimizer = optimizer
        self.gate_manager = gate_manager
        self.grad_clip_norm = float(grad_clip_norm)
        self.min_grad_norm_for_step = float(min_grad_norm_for_step)
        self.update_batch_size = max(1, int(update_batch_size))
        self.batch_loss_reduction = str(batch_loss_reduction)
        self.optimizer_step_scope = str(optimizer_step_scope)
        self.loss_normalizer = max(1, int(loss_normalizer))
        self.train_coefficients = train_coefficients
        self.coefficient_anchor_gates = coefficient_anchor_gates
        self.args = args
        self.pending_log_indices: list[int] = []
        self.optimizer_steps = 0
        self.skipped_optimizer_steps = 0
        self.pcgrad_recompute_fn = None
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def loss_scale(self) -> float:
        if self.batch_loss_reduction == "sum":
            return 1.0
        if self.optimizer_step_scope == "epoch":
            return 1.0 / float(self.loss_normalizer)
        return 1.0 / float(self.update_batch_size)

    @property
    def pending(self) -> int:
        return len(self.pending_log_indices)

    def add(self, log_rows: list[dict], log_index: int) -> None:
        self.pending_log_indices.append(int(log_index))
        if self.optimizer_step_scope == "batch" and self.pending >= self.update_batch_size:
            self.flush(log_rows)

    def flush(self, log_rows: list[dict], *, force: bool = False) -> None:
        if not self.pending_log_indices:
            return
        if self.optimizer_step_scope == "epoch" and not force:
            return
        pcgrad_stats = None
        if bool(getattr(self.args, "pcgrad_gate_gradients", False)):
            if self.optimizer_step_scope != "epoch":
                raise ValueError("--pcgrad-gate-gradients currently requires --optimizer-step-scope epoch")
            if self.pcgrad_recompute_fn is None:
                raise RuntimeError("PCGrad requested but no gate-gradient recompute function was installed")
            pcgrad_stats = self.pcgrad_recompute_fn()
        grad_norm = self.torch.nn.utils.clip_grad_norm_(self.gate_manager.parameters(), self.grad_clip_norm)
        grad_norm_value = float(grad_norm.detach().cpu().item())
        skipped_step = grad_norm_value <= self.min_grad_norm_for_step
        if not skipped_step:
            self.optimizer.step()
            _project_after_optimizer_step_(
                self.torch,
                self.gate_manager,
                self.train_coefficients,
                self.coefficient_anchor_gates,
                self.args,
            )
            self.optimizer_steps += 1
        else:
            self.skipped_optimizer_steps += 1
        gates = self.gate_manager.gate_values()
        for index in self.pending_log_indices:
            log_rows[index]["grad_norm"] = grad_norm_value
            log_rows[index]["skipped_step"] = skipped_step
            log_rows[index]["gates"] = gates
            log_rows[index]["optimizer_step_index"] = self.optimizer_steps
            log_rows[index]["update_batch_size"] = self.update_batch_size
            log_rows[index]["batch_loss_reduction"] = self.batch_loss_reduction
            log_rows[index]["batch_loss_scale"] = self.loss_scale
            log_rows[index]["optimizer_step_scope"] = self.optimizer_step_scope
            log_rows[index]["loss_normalizer"] = self.loss_normalizer
            if pcgrad_stats is not None:
                log_rows[index]["pcgrad_enabled"] = True
                log_rows[index]["pcgrad_conflict_count"] = pcgrad_stats["conflict_count"]
                log_rows[index]["pcgrad_task_grad_norms"] = pcgrad_stats["task_grad_norms"]
                log_rows[index]["pcgrad_regularizer_grad_norm"] = pcgrad_stats["regularizer_grad_norm"]
                log_rows[index]["pcgrad_pre_cosines"] = pcgrad_stats["pre_cosines"]
                log_rows[index]["pcgrad_post_cosines"] = pcgrad_stats["post_cosines"]
        self.pending_log_indices.clear()
        self.optimizer.zero_grad(set_to_none=True)


def _gate_params(gate_manager):
    return [param for param in gate_manager.parameters() if param.requires_grad]


def _flatten_grads(grads, params):
    if not params:
        raise ValueError("No trainable gate parameters found")
    import torch

    flat = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat.append(param.detach().new_zeros(param.shape).reshape(-1))
        else:
            flat.append(grad.detach().to(device=param.device, dtype=param.dtype).reshape(-1))
    return torch.cat(flat)


def _flatten_current_gate_grads(params):
    return _flatten_grads([param.grad for param in params], params)


def _unflatten_flat_grad(flat_grad, params):
    grads = []
    offset = 0
    for param in params:
        numel = int(param.numel())
        grads.append(flat_grad[offset : offset + numel].view_as(param).to(device=param.device, dtype=param.dtype))
        offset += numel
    if offset != int(flat_grad.numel()):
        raise ValueError("Flat gradient size does not match gate parameters")
    return grads


def _write_flat_grad_to_gate_params_(flat_grad, params) -> None:
    for param, grad in zip(params, _unflatten_flat_grad(flat_grad, params)):
        param.grad = grad.detach().clone()


def _cosine_value(torch, left, right, eps: float) -> float:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denom = left_norm * right_norm
    if float(denom.detach().cpu().item()) <= float(eps):
        return 0.0
    return float((torch.dot(left, right) / (denom + float(eps))).detach().cpu().item())


def _pcgrad_cosines(torch, task_grads: dict[str, object], eps: float) -> dict[str, float]:
    tasks = sorted(task_grads)
    values = {}
    for idx, left_task in enumerate(tasks):
        for right_task in tasks[idx + 1 :]:
            values[f"{left_task}|{right_task}"] = _cosine_value(
                torch,
                task_grads[left_task].float(),
                task_grads[right_task].float(),
                eps,
            )
    return values


def _pcgrad_project(task_grads: dict[str, object], eps: float):
    """Project conflicting task gradients and return projected gradients plus diagnostics."""
    if not task_grads:
        return {}, {
            "conflict_count": 0,
            "task_grad_norms": {},
            "pre_cosines": {},
            "post_cosines": {},
        }
    import torch

    tasks = sorted(task_grads)
    projected = {task: task_grads[task].detach().clone() for task in tasks}
    task_grad_norms = {
        task: float(torch.linalg.vector_norm(task_grads[task].float()).detach().cpu().item())
        for task in tasks
    }
    pre_cosines = _pcgrad_cosines(torch, task_grads, eps)
    conflict_count = 0
    for task_a in tasks:
        grad_a = projected[task_a]
        if task_grad_norms[task_a] <= float(eps):
            continue
        for task_b in tasks:
            if task_a == task_b:
                continue
            grad_b = task_grads[task_b]
            norm_b_sq = torch.dot(grad_b.float(), grad_b.float())
            if float(norm_b_sq.detach().cpu().item()) <= float(eps):
                continue
            dot = torch.dot(grad_a.float(), grad_b.float())
            if float(dot.detach().cpu().item()) < 0.0:
                grad_a = grad_a - (dot / (norm_b_sq + float(eps))).to(dtype=grad_a.dtype) * grad_b
                conflict_count += 1
        projected[task_a] = grad_a
    post_cosines = _pcgrad_cosines(torch, projected, eps)
    return projected, {
        "conflict_count": int(conflict_count),
        "task_grad_norms": task_grad_norms,
        "pre_cosines": pre_cosines,
        "post_cosines": post_cosines,
    }


def _combine_pcgrad_task_and_regularizer_grads(
    task_grads: dict[str, object],
    regularizer_grad,
    *,
    eps: float,
    project_tasks: set[str] | None = None,
):
    import torch

    if project_tasks is None:
        project_tasks = set(task_grads)
    projected_input = {task: grad for task, grad in task_grads.items() if task in project_tasks}
    projected, stats = _pcgrad_project(projected_input, eps)
    final_grad = regularizer_grad.detach().clone()
    for task in sorted(task_grads):
        final_grad = final_grad + projected.get(task, task_grads[task])
    stats["task_grad_norms"] = {
        task: float(torch.linalg.vector_norm(task_grads[task].float()).detach().cpu().item())
        for task in sorted(task_grads)
    }
    stats["regularizer_grad_norm"] = float(torch.linalg.vector_norm(regularizer_grad.float()).detach().cpu().item())
    stats["projected_task_grad_norms"] = {
        task: float(torch.linalg.vector_norm(projected.get(task, task_grads[task]).float()).detach().cpu().item())
        for task in sorted(task_grads)
    }
    stats["project_tasks"] = sorted(project_tasks)
    return final_grad, projected, stats


def _replace_gate_grads_with_pcgrad(
    *,
    torch,
    model,
    tokenizer,
    gate_manager,
    optimizer,
    rows: list[dict],
    opd_rows: list[dict],
    opd_all_success_rows: list[dict],
    retention_rows: list[dict],
    raw_rows: list[dict],
    config: dict,
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    update_batcher: _UpdateBatcher,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    advantage_task_scales: dict[str, float],
    opd_scale_plan: dict,
    retention_scale_plan: dict,
    retention_weight: float,
    prior_weight: float,
    opd_length_normalize: bool,
    retention_length_normalize: bool,
) -> dict:
    """Recompute task-specific gate gradients, apply PCGrad, and replace p.grad."""
    params = _gate_params(gate_manager)
    observed_tasks = _observed_pcgrad_tasks(rows, opd_rows, opd_all_success_rows, retention_rows, raw_rows)
    if args.pcgrad_task:
        requested = {str(task) for task in args.pcgrad_task}
        unknown = sorted(requested - observed_tasks)
        if unknown:
            raise ValueError(f"--pcgrad-task contains tasks not observed in this update: {unknown}")
        project_tasks = requested
    else:
        project_tasks = set(observed_tasks)
    task_grads = {}
    prior_scale_sum = 0.0
    for task in sorted(observed_tasks):
        optimizer.zero_grad(set_to_none=True)
        task_stats = _backward_pcgrad_task_losses(
            torch,
            model,
            tokenizer,
            rows=rows,
            opd_rows=opd_rows,
            opd_all_success_rows=opd_all_success_rows,
            retention_rows=retention_rows,
            config=config,
            args=args,
            task=task,
            device=device,
            max_logprob_tokens=max_logprob_tokens,
            update_batcher=update_batcher,
            task_weights=task_weights,
            category_weights=category_weights,
            source_weights=source_weights,
            advantage_task_scales=advantage_task_scales,
            opd_scale_plan=opd_scale_plan,
            retention_scale_plan=retention_scale_plan,
            retention_weight=retention_weight,
            opd_length_normalize=opd_length_normalize,
            retention_length_normalize=retention_length_normalize,
        )
        task_grads[task] = _flatten_current_gate_grads(params)
        prior_scale_sum += float(task_stats["prior_scale_sum"])
    optimizer.zero_grad(set_to_none=True)
    if float(prior_weight) != 0.0 and prior_scale_sum != 0.0:
        (_gate_prior_loss(torch, gate_manager) * float(prior_weight) * prior_scale_sum).backward()
        regularizer_grad = _flatten_current_gate_grads(params)
    else:
        regularizer_grad = _flatten_grads([None for _ in params], params)
    optimizer.zero_grad(set_to_none=True)
    final_grad, _projected, stats = _combine_pcgrad_task_and_regularizer_grads(
        task_grads,
        regularizer_grad,
        eps=float(args.pcgrad_eps),
        project_tasks=project_tasks,
    )
    _write_flat_grad_to_gate_params_(final_grad, params)
    return stats


def _observed_pcgrad_tasks(*row_sets: list[dict]) -> set[str]:
    tasks = set()
    for rows in row_sets:
        for row in rows:
            task = str(row.get("task", "") or "")
            if task:
                tasks.add(task)
    return tasks


def _backward_pcgrad_task_losses(
    torch,
    model,
    tokenizer,
    *,
    rows: list[dict],
    opd_rows: list[dict],
    opd_all_success_rows: list[dict],
    retention_rows: list[dict],
    config: dict,
    args: argparse.Namespace,
    task: str,
    device,
    max_logprob_tokens: int,
    update_batcher: _UpdateBatcher,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    advantage_task_scales: dict[str, float],
    opd_scale_plan: dict,
    retention_scale_plan: dict,
    retention_weight: float,
    opd_length_normalize: bool,
    retention_length_normalize: bool,
) -> dict[str, float]:
    prior_scale_sum = 0.0
    processed = 0
    for row in rows:
        if str(row.get("task", "")) != task:
            continue
        stats = _backward_pcgrad_frontier_row(
            torch,
            model,
            tokenizer,
            row=row,
            config=config,
            args=args,
            device=device,
            max_logprob_tokens=max_logprob_tokens,
            update_batcher=update_batcher,
            task_weights=task_weights,
            category_weights=category_weights,
            source_weights=source_weights,
            advantage_task_scales=advantage_task_scales,
        )
        if stats["processed"] > 0:
            prior_scale_sum += float(stats["prior_scale"])
            processed += int(stats["processed"])
    if opd_rows and (float(args.opd_loss_weight) != 0.0 or float(args.opd_pairwise_loss_weight) != 0.0):
        for row in opd_rows:
            if str(row.get("task", "")) != task:
                continue
            stats = _backward_pcgrad_opd_row(
                torch,
                model,
                tokenizer,
                row=row,
                args=args,
                device=device,
                max_logprob_tokens=max_logprob_tokens,
                update_batcher=update_batcher,
                task_weights=task_weights,
                category_weights=category_weights,
                source_weights=source_weights,
                opd_scale_plan=opd_scale_plan,
                opd_length_normalize=opd_length_normalize,
            )
            if stats["processed"] > 0:
                prior_scale_sum += float(stats["prior_scale"])
                processed += int(stats["processed"])
    if args.use_opd_all_success and opd_all_success_rows and float(args.opd_all_success_loss_weight) != 0.0:
        for row in opd_all_success_rows:
            if str(row.get("task", "")) != task:
                continue
            stats = _backward_pcgrad_opd_all_success_row(
                torch,
                model,
                tokenizer,
                row=row,
                args=args,
                device=device,
                max_logprob_tokens=max_logprob_tokens,
                update_batcher=update_batcher,
                task_weights=task_weights,
                category_weights=category_weights,
                source_weights=source_weights,
                retention_length_normalize=retention_length_normalize,
            )
            if stats["processed"] > 0:
                prior_scale_sum += float(stats["prior_scale"])
                processed += int(stats["processed"])
    if args.use_retention and retention_weight > 0.0:
        for row in retention_rows:
            if str(row.get("task", "")) != task:
                continue
            stats = _backward_pcgrad_retention_row(
                torch,
                model,
                tokenizer,
                row=row,
                args=args,
                device=device,
                max_logprob_tokens=max_logprob_tokens,
                update_batcher=update_batcher,
                task_weights=task_weights,
                category_weights=category_weights,
                source_weights=source_weights,
                retention_scale_plan=retention_scale_plan,
                retention_weight=retention_weight,
                retention_length_normalize=retention_length_normalize,
            )
            processed += int(stats["processed"])
    return {"processed": float(processed), "prior_scale_sum": prior_scale_sum}


def _backward_pcgrad_frontier_row(
    torch,
    model,
    tokenizer,
    *,
    row: dict,
    config: dict,
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    update_batcher: _UpdateBatcher,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    advantage_task_scales: dict[str, float],
) -> dict[str, float]:
    prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
    task_name = str(row.get("task", ""))
    best_response_only = args.ppo_loss_weight == 0.0 and (
        float(args.best_response_loss_weight) != 0.0 or float(args.pairwise_loss_weight) != 0.0
    )
    valid_samples = _objective_samples(row["samples"], require_old_logprob=not best_response_only)
    if len(valid_samples) < 2:
        return {"processed": 0.0, "prior_scale": 0.0}
    rewards = [_sample_train_reward(sample, task=task_name) for sample in valid_samples]
    category = _row_category(row)
    source = str(row.get("source") or "")
    task_weight = (
        task_weights.get(str(row.get("task")), 1.0)
        * category_weights.get(category, 1.0)
        * source_weights.get(source, 1.0)
    )
    if best_response_only:
        objective_stats = _backward_incremental_best_response_losses(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            valid_samples=valid_samples,
            task=task_name,
            device=device,
            max_logprob_tokens=max_logprob_tokens,
            task_weight=task_weight,
            best_response_loss_weight=float(args.best_response_loss_weight),
            pairwise_loss_weight=float(args.pairwise_loss_weight),
            pairwise_margin=float(args.pairwise_margin),
            length_normalize=bool(args.length_normalize_logprob),
            positive_reward_threshold=args.positive_reward_threshold,
            max_pairwise_pairs_per_row=args.max_pairwise_pairs_per_row,
            loss_scale=update_batcher.loss_scale,
        )
        return {"processed": objective_stats["processed"], "prior_scale": update_batcher.loss_scale if objective_stats["processed"] >= 1 else 0.0}
    advantage_values, _advantage_source = _row_advantage_values(
        valid_samples,
        rewards,
        frontier_weight=_policy_frontier_multiplier(row, config, args),
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
            max_logprob_tokens=max_logprob_tokens,
            task_weight=task_weight,
            ppo_loss_weight=float(args.ppo_loss_weight),
            beta_kl=float(config["loss"]["beta_kl"]),
            eps_clip=float(config["loss"]["eps_clip"]),
            length_normalize_policy_logprob=bool(args.length_normalize_policy_logprob),
            loss_granularity=str(args.loss_granularity),
            loss_scale=update_batcher.loss_scale,
        )
        return {"processed": objective_stats["processed"], "prior_scale": update_batcher.loss_scale if objective_stats["processed"] >= 2 else 0.0}
    processed = _backward_pcgrad_combined_frontier_loss(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        valid_samples=valid_samples,
        advantages=advantages,
        task_name=task_name,
        task_weight=task_weight,
        config=config,
        args=args,
        device=device,
        max_logprob_tokens=max_logprob_tokens,
        loss_scale=update_batcher.loss_scale,
    )
    return {"processed": processed, "prior_scale": update_batcher.loss_scale if processed >= 1 else 0.0}


def _backward_pcgrad_combined_frontier_loss(
    torch,
    model,
    tokenizer,
    *,
    prompt_text: str,
    valid_samples: list[dict],
    advantages,
    task_name: str,
    task_weight: float,
    config: dict,
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    loss_scale: float,
) -> float:
    logp_entries = []
    for sample_idx, sample in enumerate(valid_samples):
        entry = _sample_response_logprob_entry(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            sample=sample,
            device=device,
            max_length=max_logprob_tokens,
            loss_granularity=str(args.loss_granularity),
        )
        if entry is None:
            continue
        entry.update(
            {
                "sample_idx": sample_idx,
                "sample": sample,
                "reward": _sample_train_reward(sample, task=task_name),
                "raw_reward": float(sample.get("reward", 0.0)),
                "length": int(sample.get("length", 0) or 0),
            }
        )
        logp_entries.append(entry)
    if len(logp_entries) < 2:
        return 0.0
    denominator = float(len(logp_entries))
    loss_tensor = logp_entries[0]["current"].new_tensor(0.0)
    for entry in logp_entries:
        sample_idx = int(entry["sample_idx"])
        advantage = advantages[sample_idx]
        if args.ppo_loss_weight != 0.0:
            if args.loss_granularity == "token":
                policy_loss = clipped_grpo_token_loss(
                    torch,
                    current_logprobs=entry["current_logprobs"],
                    old_logprobs=entry["old_logprobs"],
                    response_mask=entry["response_mask"],
                    advantage=advantage,
                    clip_epsilon=float(config["loss"]["eps_clip"]),
                ) / denominator
            else:
                current = _entry_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                old_logp = _entry_old_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
                ratio = torch.exp((current - old_logp).clamp(-20.0, 20.0))
                clipped = torch.clamp(ratio, 1.0 - float(config["loss"]["eps_clip"]), 1.0 + float(config["loss"]["eps_clip"]))
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage) / denominator
            loss_tensor = loss_tensor + task_weight * float(args.ppo_loss_weight) * policy_loss
        if args.loss_granularity == "token":
            kl_loss = reverse_kl_token_penalty(
                torch,
                current_logprobs=entry["current_logprobs"],
                old_logprobs=entry["old_logprobs"],
                response_mask=entry["response_mask"],
            ) / denominator
        else:
            current = _entry_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
            old_logp = _entry_old_score(entry, length_normalize=bool(args.length_normalize_policy_logprob))
            log_ratio = (old_logp - current).clamp(-20.0, 20.0)
            kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
        loss_tensor = loss_tensor + task_weight * float(config["loss"]["beta_kl"]) * kl_loss
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
    (loss_tensor * float(loss_scale)).backward()
    return float(len(logp_entries))


def _backward_pcgrad_opd_row(
    torch,
    model,
    tokenizer,
    *,
    row: dict,
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    update_batcher: _UpdateBatcher,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    opd_scale_plan: dict,
    opd_length_normalize: bool,
) -> dict[str, float]:
    prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
    valid_samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
    if len(valid_samples) < 2:
        return {"processed": 0.0, "prior_scale": 0.0}
    task_name = str(row.get("task", ""))
    category = _row_category(row)
    source = str(row.get("source") or "opd_distill")
    task_weight = (
        task_weights.get(str(row.get("task")), 1.0)
        * category_weights.get(category, 1.0)
        * source_weights.get(source, 1.0)
    )
    opd_row_scale = _opd_row_loss_scale(row, opd_scale_plan=opd_scale_plan, default_loss_scale=update_batcher.loss_scale)
    objective_stats = _backward_incremental_best_response_losses(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        valid_samples=valid_samples,
        task=task_name,
        device=device,
        max_logprob_tokens=max_logprob_tokens,
        task_weight=task_weight,
        best_response_loss_weight=float(args.opd_loss_weight),
        pairwise_loss_weight=float(args.opd_pairwise_loss_weight),
        pairwise_margin=float(args.opd_pairwise_margin),
        length_normalize=opd_length_normalize,
        positive_reward_threshold=args.opd_positive_reward_threshold,
        max_pairwise_pairs_per_row=args.max_opd_pairwise_pairs_per_row,
        loss_scale=opd_row_scale,
        component_scale=_opd_component_scale(row, opd_scale_plan=opd_scale_plan),
    )
    return {"processed": objective_stats["processed"], "prior_scale": opd_row_scale if objective_stats["processed"] >= 1 else 0.0}


def _backward_pcgrad_opd_all_success_row(
    torch,
    model,
    tokenizer,
    *,
    row: dict,
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    update_batcher: _UpdateBatcher,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    retention_length_normalize: bool,
) -> dict[str, float]:
    prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
    valid_samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
    if not valid_samples:
        return {"processed": 0.0, "prior_scale": 0.0}
    task_name = str(row.get("task", ""))
    category = _row_category(row)
    source = str(row.get("source") or "opd_all_success")
    task_weight = (
        task_weights.get(str(row.get("task")), 1.0)
        * category_weights.get(category, 1.0)
        * source_weights.get(source, 1.0)
    )
    objective_stats = _backward_incremental_best_response_losses(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        valid_samples=valid_samples,
        task=task_name,
        device=device,
        max_logprob_tokens=max_logprob_tokens,
        task_weight=task_weight,
        best_response_loss_weight=float(args.opd_all_success_loss_weight),
        pairwise_loss_weight=0.0,
        pairwise_margin=0.0,
        length_normalize=retention_length_normalize,
        positive_reward_threshold=args.opd_all_success_positive_reward_threshold,
        max_pairwise_pairs_per_row=0,
        loss_scale=update_batcher.loss_scale,
    )
    return {"processed": objective_stats["processed"], "prior_scale": update_batcher.loss_scale if objective_stats["processed"] >= 1 else 0.0}


def _backward_pcgrad_retention_row(
    torch,
    model,
    tokenizer,
    *,
    row: dict,
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    update_batcher: _UpdateBatcher,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    retention_scale_plan: dict,
    retention_weight: float,
    retention_length_normalize: bool,
) -> dict[str, float]:
    prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
    valid_samples = _objective_samples(row["samples"], require_old_logprob=args.retention_objective == "kl")
    if not valid_samples:
        return {"processed": 0.0}
    task_name = str(row.get("task", ""))
    category = _row_category(row)
    source = str(row.get("source") or "")
    task_weight = (
        task_weights.get(str(row.get("task")), 1.0)
        * category_weights.get(category, 1.0)
        * source_weights.get(source, 1.0)
    )
    if args.retention_objective == "nll":
        objective_stats = _backward_incremental_best_response_losses(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            valid_samples=valid_samples,
            task=task_name,
            device=device,
            max_logprob_tokens=max_logprob_tokens,
            task_weight=task_weight,
            best_response_loss_weight=retention_weight,
            pairwise_loss_weight=0.0,
            pairwise_margin=0.0,
            length_normalize=retention_length_normalize,
            positive_reward_threshold=args.retention_positive_reward_threshold,
            max_pairwise_pairs_per_row=0,
            loss_scale=_retention_row_loss_scale(row, retention_scale_plan=retention_scale_plan, default_loss_scale=update_batcher.loss_scale),
            component_scale=_retention_component_scale(row, retention_scale_plan=retention_scale_plan),
        )
        return {"processed": objective_stats["processed"]}
    processed = 0
    denominator = float(len(valid_samples))
    for sample in valid_samples:
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
        old_logp = torch.tensor(float(sample["old_logprob"]), dtype=torch.float32, device=device)
        current = logp.float()
        log_ratio = (old_logp - current).clamp(-20.0, 20.0)
        kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
        sample_loss = task_weight * retention_weight * kl_loss
        (sample_loss * update_batcher.loss_scale).backward()
        processed += 1
    return {"processed": float(processed)}


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
    parser.add_argument(
        "--opd-distill-rollout",
        action="append",
        default=[],
        help=(
            "JSONL rows containing same-prompt expert positives and current-policy negatives for "
            "on-policy distillation. Default unused."
        ),
    )
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-gated-modules", type=int, default=None, help="Maximum gated Linear modules to install. Default None means all mergeable modules; use 1 only for smoke tests.")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--update-batch-size",
        type=int,
        default=1,
        help=(
            "Number of kept frontier/retention rows to accumulate before one optimizer.step(). "
            "Default 1 preserves the legacy row-by-row native updater."
        ),
    )
    parser.add_argument(
        "--batch-loss-reduction",
        choices=["mean", "sum"],
        default="mean",
        help=(
            "Scale accumulated row losses for mean reduction, or leave them unscaled for sum. "
            "With optimizer-step-scope=batch, mean uses 1/update_batch_size; with epoch, mean uses "
            "the planned number of rows in that epoch."
        ),
    )
    parser.add_argument(
        "--optimizer-step-scope",
        choices=["batch", "epoch"],
        default="batch",
        help=(
            "batch applies optimizer.step() every update_batch_size rows. "
            "epoch keeps mini-batches as gradient-accumulation chunks and applies one optimizer.step() "
            "after all rows in an update epoch."
        ),
    )
    parser.add_argument(
        "--pcgrad-gate-gradients",
        action="store_true",
        default=False,
        help="Enable optional PCGrad projection across task-specific gate gradients before optimizer.step().",
    )
    parser.add_argument(
        "--pcgrad-eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon used by PCGrad projection.",
    )
    parser.add_argument(
        "--pcgrad-task",
        action="append",
        default=[],
        help="Optional task allowlist for PCGrad, e.g. --pcgrad-task tool --pcgrad-task memory --pcgrad-task code. If empty, use all observed tasks.",
    )
    parser.add_argument(
        "--loss-granularity",
        choices=["sequence", "token"],
        default="sequence",
        help=(
            "Use legacy sequence-sum PPO/GRPO or token-level PPO/GRPO. "
            "token requires samples to contain old_logprobs and response_mask."
        ),
    )
    parser.add_argument("--max-logprob-tokens", type=int, default=3072)
    parser.add_argument(
        "--fill-missing-old-logprob",
        action="store_true",
        help="Compute missing old_logprob values once under the initial gate policy before optimizing gates.",
    )
    parser.add_argument("--recompute-frontier", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--sgd-nesterov", action="store_true")
    parser.add_argument(
        "--optimizer-state-in",
        default=None,
        help="Optional torch optimizer state_dict to restore before gate updates.",
    )
    parser.add_argument(
        "--optimizer-state-out",
        default=None,
        help="Optional path where the updated torch optimizer state_dict is saved.",
    )
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
    parser.add_argument("--max-retention-rows-per-task", type=int, default=None)
    parser.add_argument("--retention-loss-weight", type=float, default=None)
    parser.add_argument(
        "--retention-objective",
        choices=["kl", "nll"],
        default="kl",
        help=(
            "Objective for --use-retention rows. kl preserves old-policy logprob; "
            "nll applies all-success best-response NLL so preservation has non-zero "
            "gradient even before the optimizer step."
        ),
    )
    parser.add_argument(
        "--retention-positive-reward-threshold",
        type=float,
        default=1.0,
        help="Positive reward_train threshold for --retention-objective nll. Default 1.0 preserves fully correct rows.",
    )
    parser.add_argument("--max-opd-distill-rows", type=int, default=None)
    parser.add_argument(
        "--use-opd-all-success",
        action="store_true",
        help="Add auxiliary OPD best-response loss on all-success rows that GRPO would otherwise skip.",
    )
    parser.add_argument("--opd-all-success-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-opd-all-success-rows", type=int, default=None)
    parser.add_argument("--early-stop-grad-norm", type=float, default=None)
    parser.add_argument("--early-stop-gate-delta", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--task-weight", action="append", default=[], help="Per-task loss weight, e.g. memory=2.0. Repeatable.")
    parser.add_argument("--frontier-task-quota", action="append", default=[], help="Max kept frontier rows per task, e.g. memory=64. Repeatable.")
    parser.add_argument("--max-frontier-rows-per-task", type=int, default=None, help="Shared cap for kept frontier rows per task.")
    parser.add_argument(
        "--frontier-order",
        choices=["as-is", "shuffle", "task-interleaved"],
        default="as-is",
        help=(
            "Order kept frontier rows before update. as-is preserves rollout order; "
            "shuffle randomly shuffles all rows; task-interleaved shuffles within each task "
            "then round-robins tasks to reduce same-task optimizer batches."
        ),
    )
    parser.add_argument(
        "--frontier-shuffle-seed",
        type=int,
        default=None,
        help="Seed for --frontier-order shuffle modes. Default 0 for standalone updater calls.",
    )
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
    parser.add_argument(
        "--opd-loss-weight",
        type=float,
        default=0.0,
        help="Sequence best-response loss weight for --opd-distill-rollout rows. Default 0 disables OPD.",
    )
    parser.add_argument(
        "--opd-pairwise-loss-weight",
        type=float,
        default=0.0,
        help="Pairwise expert-positive versus current-negative loss weight for OPD rows. Default 0 disables it.",
    )
    parser.add_argument("--opd-pairwise-margin", type=float, default=0.0)
    parser.add_argument(
        "--max-opd-pairwise-pairs-per-row",
        type=int,
        default=0,
        help="Cap OPD pairwise comparisons per row; 0 keeps all positive-negative pairs.",
    )
    parser.add_argument("--length-normalize-logprob", action="store_true")
    parser.add_argument(
        "--opd-length-normalize-logprob",
        dest="opd_length_normalize_logprob",
        action="store_true",
        default=None,
        help="Use per-token average logprob for OPD best-response/pairwise losses. Defaults to --length-normalize-logprob.",
    )
    parser.add_argument(
        "--no-opd-length-normalize-logprob",
        dest="opd_length_normalize_logprob",
        action="store_false",
        help="Use sequence-sum logprob for OPD even when --length-normalize-logprob is set.",
    )
    parser.add_argument(
        "--retention-length-normalize-logprob",
        dest="retention_length_normalize_logprob",
        action="store_true",
        default=None,
        help="Use per-token average logprob for retention NLL / all-success preservation. Defaults to --length-normalize-logprob.",
    )
    parser.add_argument(
        "--no-retention-length-normalize-logprob",
        dest="retention_length_normalize_logprob",
        action="store_false",
        help="Use sequence-sum logprob for retention NLL / all-success preservation.",
    )
    parser.add_argument(
        "--retention-dynamic-scale",
        action="store_true",
        help="Estimate retention NLL magnitudes before backward and scale retention to a target ratio.",
    )
    parser.add_argument(
        "--retention-task-balanced-loss-scale",
        action="store_true",
        help="Scale retention row reductions as one third of each task mean, so all-success rows stay task-balanced.",
    )
    parser.add_argument("--retention-scale-target", type=float, default=0.5)
    parser.add_argument("--retention-scale-min", type=float, default=0.05)
    parser.add_argument("--retention-scale-max", type=float, default=100.0)
    parser.add_argument("--retention-scale-eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--opd-dynamic-scale",
        action="store_true",
        help="Estimate per-task OPD loss magnitudes before backward and scale OPD to a recoverable-all-fail target ratio.",
    )
    parser.add_argument(
        "--opd-task-balanced-loss-scale",
        action="store_true",
        help="Scale OPD row reductions as one third of each task mean, so tasks with more OPD rows do not dominate.",
    )
    parser.add_argument(
        "--opd-scale-mode",
        choices=["loss"],
        default="loss",
        help="Statistic used by --opd-dynamic-scale. loss uses no-grad OPD loss magnitudes.",
    )
    parser.add_argument("--opd-scale-min", type=float, default=0.05)
    parser.add_argument("--opd-scale-max", type=float, default=100.0)
    parser.add_argument("--opd-scale-eps", type=float, default=1.0e-6)
    parser.add_argument("--opd-scale-rate-high", type=float, default=0.20)
    parser.add_argument("--opd-scale-rate-mid", type=float, default=0.10)
    parser.add_argument("--opd-scale-rate-low", type=float, default=0.03)
    parser.add_argument("--opd-scale-target-high", type=float, default=5.0)
    parser.add_argument("--opd-scale-target-mid", type=float, default=3.0)
    parser.add_argument("--opd-scale-target-low", type=float, default=1.0)
    parser.add_argument("--opd-scale-target-tail", type=float, default=0.33)
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
        "--advantage-normalization",
        choices=["centered", "zscore"],
        default="centered",
        help="How to convert reward_train values to row advantages. centered subtracts row mean only; zscore also divides by row std.",
    )
    parser.add_argument(
        "--use-frontier-weight",
        action="store_true",
        help="Multiply policy advantages by row frontier_weight. Default off; frontier filtering still uses frontier stats.",
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
        "--advantage-field-frontier-weight",
        dest="advantage_field_apply_frontier_weight",
        action="store_true",
        help="Multiply --advantage-field values by row frontier_weight.",
    )
    parser.add_argument(
        "--no-advantage-field-frontier-weight",
        dest="advantage_field_apply_frontier_weight",
        action="store_false",
        help="Do not multiply --advantage-field values by row frontier_weight.",
    )
    parser.set_defaults(advantage_field_apply_frontier_weight=False)
    parser.add_argument("--positive-reward-threshold", type=float, default=None)
    parser.add_argument(
        "--opd-positive-reward-threshold",
        type=float,
        default=None,
        help="Positive threshold for OPD rows. Default uses the best reward_train in each OPD row.",
    )
    parser.add_argument(
        "--opd-all-success-positive-reward-threshold",
        type=float,
        default=1.0,
        help="Positive threshold for all-success OPD rows. Default 1.0 uses fully correct samples only.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto, for multi-GPU sharding.")
    parser.add_argument("--max-memory", action="append", default=[], help="HF max_memory entry, e.g. 0=70GiB. Repeatable.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable HF gradient checkpointing during gate updates.")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--gate-parameterization",
        choices=["global", "layer-band", "parameter", "global-parameter", "global-coefficient"],
        default="global",
    )
    parser.add_argument("--init-gate-checkpoint", default=None)
    args = parser.parse_args()
    if args.advantage_field and args.advantage_field_apply_frontier_weight:
        args.use_frontier_weight = True
    if (
        not args.rollouts
        and not args.replay_buffer
        and not args.retention_only_replay_buffer
        and not args.opd_distill_rollout
    ):
        parser.error(
            "at least one --rollouts, --replay-buffer, --retention-only-replay-buffer, "
            "or --opd-distill-rollout input is required"
        )
    if int(args.update_batch_size) < 1:
        parser.error("--update-batch-size must be >= 1")
    if args.pcgrad_gate_gradients and args.optimizer_step_scope != "epoch":
        parser.error("--pcgrad-gate-gradients currently requires --optimizer-step-scope epoch")
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


def _resolve_component_length_normalize(value: bool | None, *, fallback: bool) -> bool:
    return bool(fallback) if value is None else bool(value)


def _build_opd_scale_plan(
    torch,
    model,
    tokenizer,
    *,
    opd_rows: list[dict],
    raw_rows: list[dict],
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    loss_normalizer: int,
    length_normalize: bool,
) -> dict:
    raw_task_counts = _task_counts(raw_rows)
    opd_task_counts = _task_counts(opd_rows)
    tasks = sorted(set(raw_task_counts) | set(opd_task_counts))
    task_stats: dict[str, dict] = {}
    for task in tasks:
        opd_count = int(opd_task_counts.get(task, 0))
        raw_count = int(raw_task_counts.get(task, 0))
        rate = float(opd_count) / float(raw_count) if raw_count > 0 else 0.0
        target_ratio = _opd_target_ratio(rate, args=args) if opd_count > 0 else 0.0
        task_stats[task] = {
            "raw_rows": raw_count,
            "opd_rows": opd_count,
            "recoverable_all_fail_rate": rate,
            "target_ratio": target_ratio,
            "mean_abs_loss": None,
            "sum_abs_loss": None,
            "component_scale": 1.0 if opd_count > 0 else 0.0,
            "row_loss_scale": None,
        }
    if args.opd_task_balanced_loss_scale:
        for task, stats in task_stats.items():
            count = int(stats.get("opd_rows") or 0)
            stats["row_loss_scale"] = 1.0 / (3.0 * float(count)) if count > 0 else 0.0
    else:
        for stats in task_stats.values():
            stats["row_loss_scale"] = 1.0 / float(max(1, loss_normalizer))

    if not args.opd_dynamic_scale or not opd_rows:
        return {
            "enabled": bool(args.opd_dynamic_scale),
            "mode": str(args.opd_scale_mode),
            "task_balanced_loss_scale": bool(args.opd_task_balanced_loss_scale),
            "length_normalize": bool(length_normalize),
            "loss_normalizer": int(loss_normalizer),
            "task_stats": task_stats,
        }

    losses_by_task: dict[str, list[float]] = defaultdict(list)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        for row in opd_rows:
            task = str(row.get("task", ""))
            prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
            valid_samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
            if len(valid_samples) < 2:
                continue
            category = _row_category(row)
            source = str(row.get("source") or "opd_distill")
            source_weight = source_weights.get(source, 1.0)
            task_weight = task_weights.get(task, 1.0) * category_weights.get(category, 1.0) * source_weight
            stats = _best_response_loss_value_no_grad(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                valid_samples=valid_samples,
                task=task,
                device=device,
                max_logprob_tokens=max_logprob_tokens,
                task_weight=task_weight,
                best_response_loss_weight=float(args.opd_loss_weight),
                pairwise_loss_weight=float(args.opd_pairwise_loss_weight),
                pairwise_margin=float(args.opd_pairwise_margin),
                length_normalize=bool(length_normalize),
                positive_reward_threshold=args.opd_positive_reward_threshold,
                max_pairwise_pairs_per_row=int(args.max_opd_pairwise_pairs_per_row),
            )
            if stats["processed"] > 0:
                losses_by_task[task].append(abs(float(stats["loss"])))
    if was_training:
        model.train()

    for task, stats in task_stats.items():
        losses = losses_by_task.get(task, [])
        mean_abs = _mean(losses)
        sum_abs = sum(float(value) for value in losses)
        stats["mean_abs_loss"] = mean_abs
        stats["sum_abs_loss"] = sum_abs
        if int(stats.get("opd_rows") or 0) <= 0 or mean_abs <= 0.0:
            stats["component_scale"] = 0.0
            continue
        target_loss = float(args.ppo_loss_weight) * float(stats.get("target_ratio") or 0.0)
        raw_scale = target_loss / (mean_abs + float(args.opd_scale_eps))
        stats["component_scale"] = max(float(args.opd_scale_min), min(float(args.opd_scale_max), raw_scale))
        stats["target_loss"] = target_loss
        stats["raw_component_scale"] = raw_scale
    return {
        "enabled": True,
        "mode": str(args.opd_scale_mode),
        "task_balanced_loss_scale": bool(args.opd_task_balanced_loss_scale),
        "length_normalize": bool(length_normalize),
        "loss_normalizer": int(loss_normalizer),
        "task_stats": task_stats,
    }


def _opd_target_ratio(rate: float, *, args: argparse.Namespace) -> float:
    if float(rate) >= float(args.opd_scale_rate_high):
        return float(args.opd_scale_target_high)
    if float(rate) >= float(args.opd_scale_rate_mid):
        return float(args.opd_scale_target_mid)
    if float(rate) >= float(args.opd_scale_rate_low):
        return float(args.opd_scale_target_low)
    if float(rate) > 0.0:
        return float(args.opd_scale_target_tail)
    return 0.0


def _opd_row_loss_scale(row: dict, *, opd_scale_plan: dict, default_loss_scale: float) -> float:
    task = str(row.get("task", ""))
    stats = dict(opd_scale_plan.get("task_stats", {}).get(task, {}) or {})
    if stats.get("row_loss_scale") is None:
        return float(default_loss_scale)
    return float(stats.get("row_loss_scale") or 0.0)


def _opd_component_scale(row: dict, *, opd_scale_plan: dict) -> float:
    task = str(row.get("task", ""))
    stats = dict(opd_scale_plan.get("task_stats", {}).get(task, {}) or {})
    return float(stats.get("component_scale", 1.0))


def _opd_task_plan_value(row: dict, opd_scale_plan: dict, key: str):
    task = str(row.get("task", ""))
    stats = dict(opd_scale_plan.get("task_stats", {}).get(task, {}) or {})
    return stats.get(key)


def _build_retention_scale_plan(
    torch,
    model,
    tokenizer,
    *,
    retention_rows: list[dict],
    raw_rows: list[dict],
    args: argparse.Namespace,
    device,
    max_logprob_tokens: int,
    task_weights: dict[str, float],
    category_weights: dict[str, float],
    source_weights: dict[str, float],
    retention_weight: float,
    loss_normalizer: int,
    length_normalize: bool,
) -> dict:
    raw_task_counts = _task_counts(raw_rows)
    retention_task_counts = _task_counts(retention_rows)
    tasks = sorted(set(raw_task_counts) | set(retention_task_counts))
    task_stats: dict[str, dict] = {}
    for task in tasks:
        row_count = int(retention_task_counts.get(task, 0))
        raw_count = int(raw_task_counts.get(task, 0))
        rate = float(row_count) / float(raw_count) if raw_count > 0 else 0.0
        task_stats[task] = {
            "raw_rows": raw_count,
            "retention_rows": row_count,
            "all_success_rate": rate,
            "target_ratio": float(args.retention_scale_target) if row_count > 0 else 0.0,
            "mean_abs_loss": None,
            "sum_abs_loss": None,
            "component_scale": 1.0 if row_count > 0 else 0.0,
            "row_loss_scale": None,
        }
    if args.retention_task_balanced_loss_scale:
        for stats in task_stats.values():
            count = int(stats.get("retention_rows") or 0)
            stats["row_loss_scale"] = 1.0 / (3.0 * float(count)) if count > 0 else 0.0
    else:
        for stats in task_stats.values():
            stats["row_loss_scale"] = 1.0 / float(max(1, loss_normalizer))

    enabled = bool(args.retention_dynamic_scale) and bool(retention_rows) and str(args.retention_objective) == "nll"
    if not enabled:
        return {
            "enabled": bool(args.retention_dynamic_scale),
            "objective": str(args.retention_objective),
            "task_balanced_loss_scale": bool(args.retention_task_balanced_loss_scale),
            "length_normalize": bool(length_normalize),
            "loss_normalizer": int(loss_normalizer),
            "task_stats": task_stats,
        }

    losses_by_task: dict[str, list[float]] = defaultdict(list)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        for row in retention_rows:
            task = str(row.get("task", ""))
            prompt_text = row.get("rendered_prompt") or row.get("prompt") or ""
            valid_samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
            if not valid_samples:
                continue
            category = _row_category(row)
            source = str(row.get("source") or "")
            source_weight = source_weights.get(source, 1.0)
            task_weight = task_weights.get(task, 1.0) * category_weights.get(category, 1.0) * source_weight
            stats = _best_response_loss_value_no_grad(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                valid_samples=valid_samples,
                task=task,
                device=device,
                max_logprob_tokens=max_logprob_tokens,
                task_weight=task_weight,
                best_response_loss_weight=retention_weight,
                pairwise_loss_weight=0.0,
                pairwise_margin=0.0,
                length_normalize=bool(length_normalize),
                positive_reward_threshold=args.retention_positive_reward_threshold,
                max_pairwise_pairs_per_row=0,
            )
            if stats["processed"] > 0:
                losses_by_task[task].append(abs(float(stats["loss"])))
    if was_training:
        model.train()

    for task, stats in task_stats.items():
        losses = losses_by_task.get(task, [])
        mean_abs = _mean(losses)
        sum_abs = sum(float(value) for value in losses)
        stats["mean_abs_loss"] = mean_abs
        stats["sum_abs_loss"] = sum_abs
        if int(stats.get("retention_rows") or 0) <= 0 or mean_abs <= 0.0:
            stats["component_scale"] = 0.0
            continue
        target_loss = float(args.ppo_loss_weight) * float(stats.get("target_ratio") or 0.0)
        raw_scale = target_loss / (mean_abs + float(args.retention_scale_eps))
        stats["component_scale"] = max(float(args.retention_scale_min), min(float(args.retention_scale_max), raw_scale))
        stats["target_loss"] = target_loss
        stats["raw_component_scale"] = raw_scale
    return {
        "enabled": True,
        "objective": str(args.retention_objective),
        "task_balanced_loss_scale": bool(args.retention_task_balanced_loss_scale),
        "length_normalize": bool(length_normalize),
        "loss_normalizer": int(loss_normalizer),
        "task_stats": task_stats,
    }


def _retention_row_loss_scale(row: dict, *, retention_scale_plan: dict, default_loss_scale: float) -> float:
    task = str(row.get("task", ""))
    stats = dict(retention_scale_plan.get("task_stats", {}).get(task, {}) or {})
    if stats.get("row_loss_scale") is None:
        return float(default_loss_scale)
    return float(stats.get("row_loss_scale") or 0.0)


def _retention_component_scale(row: dict, *, retention_scale_plan: dict) -> float:
    task = str(row.get("task", ""))
    stats = dict(retention_scale_plan.get("task_stats", {}).get(task, {}) or {})
    return float(stats.get("component_scale", 1.0))


def _retention_task_plan_value(row: dict, retention_scale_plan: dict, key: str):
    task = str(row.get("task", ""))
    stats = dict(retention_scale_plan.get("task_stats", {}).get(task, {}) or {})
    return stats.get(key)


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


def _limit_rows_per_task(rows: list[dict], max_per_task: int) -> list[dict]:
    if max_per_task < 0:
        return rows
    counts: Counter[str] = Counter()
    selected = []
    for row in rows:
        task = str(row.get("task"))
        if counts[task] >= max_per_task:
            continue
        selected.append(row)
        counts[task] += 1
    return selected


def _frontier_shuffle_seed(args: argparse.Namespace) -> int:
    value = getattr(args, "frontier_shuffle_seed", None)
    return 0 if value is None else int(value)


def _order_frontier_rows(rows: list[dict], *, order: str, seed: int) -> list[dict]:
    if order == "as-is":
        return rows
    ordered = list(rows)
    rng = random.Random(int(seed))
    if order == "shuffle":
        rng.shuffle(ordered)
        return ordered
    if order != "task-interleaved":
        raise ValueError(f"Unknown frontier order: {order}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ordered:
        grouped[str(row.get("task"))].append(row)
    for task_rows in grouped.values():
        rng.shuffle(task_rows)
    preferred = ["tool", "memory", "code"]
    task_order = [task for task in preferred if task in grouped]
    task_order.extend(sorted(task for task in grouped if task not in set(task_order)))
    interleaved = []
    while any(grouped[task] for task in task_order):
        for task in task_order:
            if grouped[task]:
                interleaved.append(grouped[task].pop(0))
    return interleaved


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
        _reward_advantages(
            rewards,
            frontier_weight=frontier_weight,
            normalization=str(args.advantage_normalization),
            epsilon=float(config["loss"]["advantage_eps"]),
        ),
        f"group_relative_reward:{args.advantage_normalization}",
    )


def _reward_advantages(
    rewards: list[float],
    *,
    frontier_weight: float,
    normalization: str,
    epsilon: float,
) -> list[float]:
    if normalization == "zscore":
        return group_relative_advantages(rewards, frontier_weight=frontier_weight, epsilon=epsilon)
    if normalization != "centered":
        raise ValueError(f"Unknown advantage normalization: {normalization}")
    avg = _mean(rewards)
    return [float(frontier_weight) * (float(value) - avg) for value in rewards]


def _policy_frontier_multiplier(row: dict, config: dict, args: argparse.Namespace) -> float:
    if not bool(args.use_frontier_weight):
        return 1.0
    return policy_frontier_weight(
        row.get("frontier", {}),
        min_reward_std=float(config["frontier"]["min_reward_std"]),
    )


def _advantage_task_scales(rows: list[dict], config: dict, args: argparse.Namespace, *, enabled: bool) -> dict[str, float]:
    if not enabled:
        return {}
    task_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        samples = _objective_samples(row.get("samples", []), require_old_logprob=False)
        if len(samples) < 2:
            continue
        task_name = str(row.get("task", ""))
        rewards = [_sample_train_reward(sample, task=task_name) for sample in samples]
        frontier_weight = _policy_frontier_multiplier(row, config, args)
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


def _mean(values) -> float:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else 0.0


def _sample_train_reward(sample: dict, *, task: str | None = None) -> float:
    if "reward_train" in sample:
        return float(sample.get("reward_train", 0.0))
    raw = float(sample.get("reward", 0.0))
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    score_range = details.get("toolrl_score_range") if isinstance(details, dict) else None
    if task == "tool" or score_range == [-3.0, 4.0]:
        return max(0.0, min((raw + 3.0) / 7.0, 1.0))
    return max(0.0, min(raw, 1.0))


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
    task: str | None,
    device: str,
    max_logprob_tokens: int,
    task_weight: float,
    best_response_loss_weight: float,
    pairwise_loss_weight: float,
    pairwise_margin: float,
    length_normalize: bool,
    positive_reward_threshold: float | None,
    max_pairwise_pairs_per_row: int = 0,
    loss_scale: float = 1.0,
    component_scale: float = 1.0,
) -> dict[str, float]:
    positives, negatives = _positive_and_negative_samples(valid_samples, positive_reward_threshold, task=task)
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
            loss = (
                float(component_scale)
                * task_weight
                * best_response_loss_weight
                * (-_sample_score(logp.float(), sample, length_normalize=length_normalize))
                / denominator
            )
            (loss * float(loss_scale)).backward()
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
            loss = float(component_scale) * task_weight * pairwise_loss_weight * pair_loss / denominator
            (loss * float(loss_scale)).backward()
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


def _best_response_loss_value_no_grad(
    torch,
    model,
    tokenizer,
    *,
    prompt_text: str,
    valid_samples: list[dict],
    task: str | None,
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
    positives, negatives = _positive_and_negative_samples(valid_samples, positive_reward_threshold, task=task)
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
            value = float(
                (
                    task_weight
                    * best_response_loss_weight
                    * (-_sample_score(logp.float(), sample, length_normalize=length_normalize))
                    / denominator
                )
                .detach()
                .cpu()
                .item()
            )
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
            value = float((task_weight * pairwise_loss_weight * pair_loss / denominator).detach().cpu().item())
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
    loss_granularity: str = "sequence",
    loss_scale: float = 1.0,
) -> dict[str, float]:
    denominator = float(len(valid_samples))
    entries = []
    for sample_idx, sample in enumerate(valid_samples):
        entry = _sample_response_logprob_entry(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            sample=sample,
            device=device,
            max_length=max_logprob_tokens,
            loss_granularity=loss_granularity,
        )
        if entry is None:
            continue
        entries.append((sample_idx, sample, entry))
    if len(entries) < 2:
        return {
            "processed": float(len(entries)),
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "clip_frac": 0.0,
            "approx_kl": 0.0,
        }

    loss_total = 0.0
    policy_loss_total = 0.0
    kl_loss_total = 0.0
    clip_frac_total = 0.0
    approx_kl_total = 0.0
    for sample_idx, sample, entry in entries:
        sample_loss = entry["current"].new_tensor(0.0)
        if ppo_loss_weight != 0.0:
            advantage = advantages[sample_idx].to(entry["current"].device)
            if loss_granularity == "token":
                policy_loss = clipped_grpo_token_loss(
                    torch,
                    current_logprobs=entry["current_logprobs"],
                    old_logprobs=entry["old_logprobs"],
                    response_mask=entry["response_mask"],
                    advantage=advantage,
                    clip_epsilon=eps_clip,
                ) / denominator
            else:
                current = _sample_score(entry["current"], sample, length_normalize=length_normalize_policy_logprob)
                old_logp = _sample_score(entry["old"], sample, length_normalize=length_normalize_policy_logprob)
                ratio = torch.exp((current - old_logp).clamp(-20.0, 20.0))
                clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage) / denominator
            weighted_policy = task_weight * ppo_loss_weight * policy_loss
            sample_loss = sample_loss + weighted_policy
            policy_loss_total += float(weighted_policy.detach().cpu().item())
        if loss_granularity == "token":
            kl_loss = reverse_kl_token_penalty(
                torch,
                current_logprobs=entry["current_logprobs"],
                old_logprobs=entry["old_logprobs"],
                response_mask=entry["response_mask"],
            ) / denominator
        else:
            current = _sample_score(entry["current"], sample, length_normalize=length_normalize_policy_logprob)
            old_logp = _sample_score(entry["old"], sample, length_normalize=length_normalize_policy_logprob)
            log_ratio = (old_logp - current).clamp(-20.0, 20.0)
            kl_loss = (torch.exp(log_ratio) - log_ratio - 1.0) / denominator
        weighted_kl = task_weight * beta_kl * kl_loss
        sample_loss = sample_loss + weighted_kl
        kl_loss_total += float(weighted_kl.detach().cpu().item())
        metrics = _entry_policy_metrics(
            torch,
            entry,
            sample,
            length_normalize=length_normalize_policy_logprob,
            loss_granularity=loss_granularity,
            eps_clip=eps_clip,
        )
        clip_frac_total += metrics["clip_frac"] / denominator
        approx_kl_total += metrics["approx_kl"] / denominator
        (sample_loss * float(loss_scale)).backward()
        loss_total += float(sample_loss.detach().cpu().item())
    return {
        "processed": float(len(entries)),
        "loss": loss_total,
        "policy_loss": policy_loss_total,
        "kl_loss": kl_loss_total,
        "clip_frac": clip_frac_total,
        "approx_kl": approx_kl_total,
    }


def _positive_and_negative_samples(
    samples: list[dict],
    positive_reward_threshold: float | None,
    *,
    task: str | None = None,
) -> tuple[list[dict], list[dict]]:
    best_reward = max(_sample_train_reward(sample, task=task) for sample in samples)
    threshold = best_reward - 1e-8 if positive_reward_threshold is None else float(positive_reward_threshold)
    positives = [sample for sample in samples if _sample_train_reward(sample, task=task) >= threshold]
    negative_cutoff = best_reward if positive_reward_threshold is None else threshold
    negatives = [sample for sample in samples if _sample_train_reward(sample, task=task) < negative_cutoff]
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


def _entry_policy_metrics(
    torch,
    entry: dict,
    sample: dict,
    *,
    length_normalize: bool,
    loss_granularity: str,
    eps_clip: float,
) -> dict[str, float]:
    if loss_granularity == "token":
        current = entry["current_logprobs"].detach().float()
        old = entry["old_logprobs"].detach().to(current.device).float()
        mask = entry["response_mask"].detach().to(current.device).float()
        denominator = mask.sum().clamp_min(1.0)
        ratio = torch.exp((current - old).clamp(-20.0, 20.0))
        clipped = ((ratio < 1.0 - float(eps_clip)) | (ratio > 1.0 + float(eps_clip))).float()
        clip_frac = (clipped * mask).sum() / denominator
        approx_kl = ((old - current) * mask).sum() / denominator
        return {
            "clip_frac": float(clip_frac.cpu().item()),
            "approx_kl": float(approx_kl.cpu().item()),
        }
    current = _entry_score(entry, length_normalize=length_normalize).detach().float()
    old = _entry_old_score(entry, length_normalize=length_normalize).detach().float()
    ratio = torch.exp((current - old).clamp(-20.0, 20.0))
    clipped = bool((ratio < 1.0 - float(eps_clip)).cpu().item() or (ratio > 1.0 + float(eps_clip)).cpu().item())
    return {
        "clip_frac": 1.0 if clipped else 0.0,
        "approx_kl": float((old - current).cpu().item()),
    }


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
